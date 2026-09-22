#!/usr/bin/env python3
"""Standalone ridge regression variability evaluation in raw and pre-normalised space.

Fits a multioutput Ridge regressor from probe-gene expression to
full-transcriptome expression in both raw count space and pre-normalised
(``adata.X``) space.  The trivial baseline in each space is the per-gene
training mean (constant predictor).  The caller provides both matrices;
no on-the-fly normalisation is applied.

The per-gene training-mean baseline is reported as an absolute reference
(``mse_mean_baseline_*``); compare ``mse_test_probe_*`` against it directly
(there are no ``mse_ratio_*`` columns).

Usage::

    python run_ridge_evaluation.py \\
        --input_file preprocessed/full_transcriptome.h5ad \\
        --preprocessed_dir preprocessed/ \\
        --output_dir Evaluation-Results/ \\
        --celltype_col cluster \\
        --alpha 1.0 \\
        --n_splits 5

    # Use cross-validated alpha selection (the default; shown here explicitly)
    python run_ridge_evaluation.py \\
        --input_file preprocessed/full_transcriptome.h5ad \\
        --preprocessed_dir preprocessed/ \\
        --output_dir Evaluation-Results/ \\
        --use_ridgecv

    # Disable RidgeCV and fit a single Ridge at a fixed --alpha instead
    python run_ridge_evaluation.py \\
        --input_file preprocessed/full_transcriptome.h5ad \\
        --preprocessed_dir preprocessed/ \\
        --output_dir Evaluation-Results/ \\
        --no-use_ridgecv --alpha 1.0

    # Dry-run: print resolved config without executing
    python run_ridge_evaluation.py --dry_run \\
        --input_file preprocessed/full_transcriptome.h5ad \\
        --preprocessed_dir preprocessed/ \\
        --output_dir Evaluation-Results/
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Internal module imports
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(_MODULE_DIR))

from _cli_common import (
    setup_logging as _setup_logging,
    log_memory as _log_memory,
    to_dense as _to_dense_f64,
    save_results_per_dataset as _save_results_per_dataset,
    aggregate_fold_results as _aggregate_fold_results,
    load_preprocessed_datasets as _load_preprocessed_datasets,
)
from _splits import generate_evaluation_splits
from metrics import calculate_explained_variance, calculate_mse
from ridge import ridge_reconstruction, ridge_reconstruction_by_celltype

# ---------------------------------------------------------------------------
# Utility module
# ---------------------------------------------------------------------------

_UTILITY_DIR = _MODULE_DIR.parent / "Utility-module"
sys.path.insert(0, str(_UTILITY_DIR))
from _validation import is_anndata_raw, is_anndata_raw_layer  # type: ignore[import]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class RidgeEvaluationConfig:
    """Runtime configuration for the ridge regression variability evaluation.

    Attributes:
        input_file: Path to the preprocessed full-transcriptome h5ad.
        preprocessed_dir: Directory containing panel h5ad files.
        output_dir: Root output directory. Results go into
            ``Variability-Evaluation/results/ridge-regression/``.
        celltype_col: obs column holding cell-type labels.
        alpha: Ridge regularisation strength (ignored when *use_ridgecv*
            is ``True``).
        use_ridgecv: If ``True``, use GCV to select the best alpha per fold.
        alpha_values: Candidate alphas for ``RidgeCV``.
        n_splits: Number of folds for stratified k-fold CV.
        random_state: Random seed for reproducibility.
        external_panels: Paths to additional panel CSV files.
        external_names: Display names for external panels.
        strategies: Panel strategy filter (empty = no filter).
        probeset_sizes: Panel size filter.
        expvar_mode: Explained-variance aggregation mode (see ``metrics.EXPVAR_MODES``).
        expvar_modes: Optional additional explained-variance modes to report.
        gene_subsets: Which gene subset(s) to score the probe reconstruction against.
        dry_run: If True, print config and exit without running.
    """

    input_file: Path
    preprocessed_dir: Path
    output_dir: Path
    celltype_col: str = "cluster"
    alpha: float = 1.0
    use_ridgecv: bool = True
    alpha_values: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])
    n_splits: int = 5
    random_state: int = 42
    external_panels: list[str] = field(default_factory=list)
    external_names: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    probeset_sizes: list[int] = field(default_factory=list)
    expvar_mode: str = "global_mean"
    expvar_modes: list[str] = field(default_factory=list)
    gene_subsets: list[str] = field(
        default_factory=lambda: ["all_genes", "panel_genes_only", "non_panel_genes_only"]
    )
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_mean_baseline_raw_and_lognorm(
    A_train_raw: np.ndarray,
    A_test_raw: np.ndarray,
    A_train_ln: np.ndarray,
    A_test_ln: np.ndarray,
    fold_num: int,
) -> None:
    """Log per-gene mean baseline in raw and pre-normalised space.

    Args:
        A_train_raw: Raw training counts (cells × genes), float64.
        A_test_raw: Raw test counts (cells × genes), float64.
        A_train_ln: Pre-normalised training data (cells × genes), float64.
        A_test_ln: Pre-normalised test data (cells × genes), float64.
        fold_num: 1-based fold index for log messages.
    """
    mean_raw = A_train_raw.mean(axis=0)
    mse_raw = calculate_mse(A_test_raw, np.broadcast_to(mean_raw, A_test_raw.shape))
    expvar_raw = calculate_explained_variance(A_test_raw, np.broadcast_to(mean_raw, A_test_raw.shape))

    mean_ln = A_train_ln.mean(axis=0)
    mse_ln = calculate_mse(A_test_ln, np.broadcast_to(mean_ln, A_test_ln.shape))
    expvar_ln = calculate_explained_variance(A_test_ln, np.broadcast_to(mean_ln, A_test_ln.shape))

    logger.info(
        "Fold %d mean baseline | raw: MSE=%.4f ExpVar=%.4f | lognorm: MSE=%.4f ExpVar=%.4f",
        fold_num, mse_raw, expvar_raw, mse_ln, expvar_ln,
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_ridge_variability_evaluation(
    adata_full: sc.AnnData,
    preprocessed_datasets: dict[str, sc.AnnData],
    output_dir: str | Path,
    splits: list[tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    alpha: float = 1.0,
    use_ridgecv: bool = True,
    alpha_values: list[float] | None = None,
    celltype_col: str = "cluster",
    random_state: int = 42,
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
) -> dict[str, Any]:
    """Run ridge regression variability evaluation in raw and pre-normalised space.

    Fits a multioutput Ridge regressor from probe genes → all genes on the
    training set in both raw count space and pre-normalised (``adata.X``)
    space.  Performance on the held-out test set is compared to the per-gene
    training mean (trivial baseline) in each space.

    Args:
        adata_full: Full-transcriptome AnnData reference. Must have
            ``layers['counts']`` (raw counts) and pre-normalised ``X``.
        preprocessed_datasets: ``{name: AnnData}`` dict. Each non-reference
            dataset provides ``var_names`` as the probe gene list.
        output_dir: Root output directory. Results are written to
            ``Variability-Evaluation/results/ridge-regression/``.
        splits: List of ``(train_idx, test_idx, per_celltype_splits)`` tuples.
        alpha: Ridge regularisation strength.
        use_ridgecv: If ``True``, use cross-validated alpha selection.
        alpha_values: Candidate alphas for ``RidgeCV``.
        celltype_col: obs column for cell-type labels.
        random_state: Random seed.
        expvar_mode: Explained-variance aggregation mode (see ``metrics.EXPVAR_MODES``).
        expvar_modes: Optional list of explained-variance modes to additionally report
            (see ``ridge.py::ridge_reconstruction``). Defaults to ``None``, i.e. ``[expvar_mode]``.
        gene_subsets: Optional list of gene subsets to score against. Defaults to
            ``None``, i.e. ``["all_genes"]``.

    Returns:
        Dict with ``"ridge"`` (aggregated) and ``"ridge_per_fold"`` DataFrames.

    Raises:
        ValueError: If ``layers['counts']`` or pre-normalised ``X`` are absent.
    """
    logger.info("=" * 80)
    logger.info("RIDGE REGRESSION VARIABILITY EVALUATION (%d fold(s))", len(splits))
    logger.info("=" * 80)

    if "counts" not in adata_full.layers:
        raise ValueError(
            "run_ridge_variability_evaluation: adata_full.layers['counts'] not found. "
            "Raw counts are required for the raw-space ridge branch."
        )

    if not is_anndata_raw_layer(adata_full, "counts"):
        raise ValueError(
            "adata_full.layers['counts'] does not appear to contain raw integer counts. "
            "Check preprocessing."
        )

    if is_anndata_raw(adata_full):
        raise ValueError(
            "adata_full.X appears to contain raw counts. "
            "Pre-normalised data is required in adata.X for the lognorm branch."
        )

    output_dir = Path(output_dir)
    results_dir = output_dir / "Variability-Evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if celltype_col not in adata_full.obs.columns:
        logger.error("Cell-type column '%s' not found in adata_full.obs.", celltype_col)
        return {}

    X_raw = adata_full.layers["counts"]
    X_ln = adata_full.X
    panel_names = [n for n in preprocessed_datasets if n != "full_transcriptome"]
    logger.info(
        "Evaluating Ridge regression for %d panels across %d fold(s)...",
        len(panel_names), len(splits),
    )

    all_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for fold, (train_idx, test_idx, ct_splits_idx) in enumerate(splits):
        logger.info("-" * 60)
        logger.info(
            "FOLD %d/%d: %d training cells, %d test cells",
            fold + 1, len(splits), len(train_idx), len(test_idx),
        )
        logger.info("-" * 60)

        A_train_raw = _to_dense_f64(X_raw, train_idx)
        A_test_raw = _to_dense_f64(X_raw, test_idx)
        A_train_ln = _to_dense_f64(X_ln, train_idx)
        A_test_ln = _to_dense_f64(X_ln, test_idx)

        _log_mean_baseline_raw_and_lognorm(
            A_train_raw, A_test_raw, A_train_ln, A_test_ln, fold + 1,
        )

        for panel_name in tqdm(panel_names, desc=f"Fold {fold + 1} Ridge eval", leave=False):
            adata_panel = preprocessed_datasets[panel_name]
            probeset_genes = adata_panel.var_names.tolist()

            try:
                _evaluate_single_panel(
                    panel_name=panel_name,
                    adata_full=adata_full,
                    probeset_genes=probeset_genes,
                    A_train_raw=A_train_raw,
                    A_test_raw=A_test_raw,
                    A_train_ln=A_train_ln,
                    A_test_ln=A_test_ln,
                    ct_splits_idx=ct_splits_idx,
                    alpha=alpha,
                    use_ridgecv=use_ridgecv,
                    alpha_values=alpha_values,
                    celltype_col=celltype_col,
                    random_state=random_state,
                    rows=all_rows,
                    fold=fold,
                    expvar_mode=expvar_mode,
                    expvar_modes=expvar_modes,
                    gene_subsets=gene_subsets,
                )
            except Exception as exc:
                logger.warning(
                    "Fold %d: Ridge evaluation failed for '%s': %s",
                    fold, panel_name, exc, exc_info=True,
                )
                if panel_name not in skipped:
                    skipped.append(panel_name)

        gc.collect()
        _log_memory(f"after fold {fold + 1}")

    results: dict[str, Any] = {}
    if all_rows:
        df_all = pd.DataFrame(all_rows)

        per_fold_dir = results_dir / "ridge-regression" / "per_fold"
        per_fold_dir.mkdir(parents=True, exist_ok=True)
        _save_results_per_dataset(df_all, per_fold_dir, "Ridge (per-fold)")

        df_agg = _aggregate_fold_results(df_all)
        agg_dir = results_dir / "ridge-regression"
        _save_results_per_dataset(df_agg, agg_dir, "Ridge (aggregated)")

        results["ridge"] = df_agg
        results["ridge_per_fold"] = df_all

    logger.info("Ridge variability evaluation complete. Skipped: %s", skipped or "none")
    return results


def _evaluate_single_panel(
    panel_name: str,
    adata_full: sc.AnnData,
    probeset_genes: list[str],
    A_train_raw: np.ndarray,
    A_test_raw: np.ndarray,
    A_train_ln: np.ndarray,
    A_test_ln: np.ndarray,
    ct_splits_idx: dict[str, tuple[np.ndarray, np.ndarray]],
    alpha: float,
    use_ridgecv: bool,
    alpha_values: list[float] | None,
    celltype_col: str,
    random_state: int,
    rows: list[dict[str, Any]],
    fold: int | None = None,
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
) -> None:
    """Run ridge evaluation for one panel and append result rows in place.

    Args:
        panel_name: Dataset identifier.
        adata_full: Full-transcriptome AnnData.
        probeset_genes: Genes in this panel.
        A_train_raw: Global train raw counts (cells × genes), float64.
        A_test_raw: Global test raw counts (cells × genes), float64.
        A_train_ln: Global train pre-normalised data (cells × genes), float64.
        A_test_ln: Global test pre-normalised data (cells × genes), float64.
        ct_splits_idx: Per-cell-type global index splits.
        alpha: Ridge regularisation strength.
        use_ridgecv: Use cross-validated alpha selection.
        alpha_values: Candidate alphas for RidgeCV.
        celltype_col: obs column for cell-type labels.
        random_state: Random seed.
        rows: Output list (mutated in place).
        fold: Fold number.
        expvar_mode: Explained-variance aggregation mode.
        expvar_modes: Optional additional explained-variance modes to report.
        gene_subsets: Optional gene subsets to score the reconstruction against.
    """
    # ── Global ridge evaluation ───────────────────────────────────────────────
    logger.info("  [%s] Global Ridge...", panel_name)

    global_result = ridge_reconstruction(
        adata=adata_full,
        probeset_genes=probeset_genes,
        A_train_raw=A_train_raw,
        A_test_raw=A_test_raw,
        A_train_ln=A_train_ln,
        A_test_ln=A_test_ln,
        alpha=alpha,
        use_ridgecv=use_ridgecv,
        alpha_values=alpha_values,
        random_state=random_state,
        expvar_mode=expvar_mode,
        expvar_modes=expvar_modes,
        gene_subsets=gene_subsets,
    )

    if global_result:
        row = {**global_result, "dataset": panel_name, "gene_list": panel_name,
               "analysis_type": "global"}
        if fold is not None:
            row["fold"] = fold
        rows.append(row)

    # ── Per-cell-type ridge evaluation ────────────────────────────────────────
    if celltype_col in adata_full.obs.columns and ct_splits_idx:
        logger.info("  [%s] Per-cell-type Ridge...", panel_name)

        ct_result = ridge_reconstruction_by_celltype(
            adata=adata_full,
            probeset_genes=probeset_genes,
            celltype_column=celltype_col,
            alpha=alpha,
            use_ridgecv=use_ridgecv,
            alpha_values=alpha_values,
            random_state=random_state,
            per_celltype_splits=ct_splits_idx,
            expvar_mode=expvar_mode,
            expvar_modes=expvar_modes,
            gene_subsets=gene_subsets,
        )

        if ct_result:
            summary = ct_result.get("summary", {})
            if summary:
                row = {
                    **summary,
                    "dataset": panel_name,
                    "gene_list": panel_name,
                    "celltype": "summary",
                    "analysis_type": "per_celltype",
                }
                if fold is not None:
                    row["fold"] = fold
                rows.append(row)

            for ct, res in ct_result.get("celltype_results", {}).items():
                row = {
                    **res,
                    "dataset": panel_name,
                    "gene_list": panel_name,
                    "celltype": ct,
                    "analysis_type": "per_celltype",
                }
                if fold is not None:
                    row["fold"] = fold
                rows.append(row)


# ---------------------------------------------------------------------------
# Argument parsing and configuration
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the ridge evaluation script."""
    p = argparse.ArgumentParser(
        description=(
            "Ridge regression variability evaluation: fits multioutput Ridge "
            "from probe-gene log-normalised expression to full transcriptome."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--input_file", required=True,
        help="Path to the preprocessed full-transcriptome h5ad.",
    )
    p.add_argument(
        "--preprocessed_dir", required=True,
        help="Directory containing preprocessed panel h5ad files.",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Root output directory. Results go into Variability-Evaluation/results/ridge-regression/.",
    )

    p.add_argument("--celltype_col", default="cluster",
                   help="obs column holding cell-type labels (default: cluster).")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Ridge regularisation strength (default: 1.0). Ignored if --use_ridgecv.")
    p.add_argument(
        "--use_ridgecv", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Use generalised cross-validation to select alpha (RidgeCV). "
            "Default: enabled. Pass --no-use_ridgecv to fit a single Ridge "
            "at --alpha instead."
        ),
    )
    p.add_argument(
        "--alpha_values", nargs="+", type=float,
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
        help="Candidate alpha values for RidgeCV (default: 0.01 0.1 1.0 10.0 100.0).",
    )
    p.add_argument("--n_splits", type=int, default=5,
                   help="Number of k-fold cross-validation splits (default: 5).")
    p.add_argument("--random_state", type=int, default=42,
                   help="Random seed (default: 42).")

    p.add_argument("--external_panels", nargs="*", default=[],
                   help="Paths to additional panel CSV files.")
    p.add_argument("--external_names", nargs="*", default=[],
                   help="Display names for external panels.")

    p.add_argument("--strategies", nargs="*", default=[],
                   help="Filter to specific selection strategies.")
    p.add_argument("--probeset_sizes", nargs="*", type=int, default=[],
                   help="Filter to specific probeset sizes.")

    p.add_argument(
        "--expvar_mode",
        type=str,
        default="global_mean",
        choices=["global_mean", "variance_weighted_sum", "mean", "median", "expression_weighted_sum"],
        help=(
            "Gene-aggregation mode for explained variance (see metrics.EXPVAR_MODES). "
            "'global_mean' (default): pools all cell x gene entries, R2 around one global "
            "scalar mean. 'variance_weighted_sum': per-gene R2 weighted by per-gene variance. "
            "'mean'/'median': unweighted average/median of per-gene R2. "
            "'expression_weighted_sum': per-gene R2 weighted by per-gene mean expression."
        ),
    )
    p.add_argument(
        "--expvar_modes",
        type=str,
        nargs="+",
        default=None,
        choices=["global_mean", "variance_weighted_sum", "mean", "median", "expression_weighted_sum"],
        help=(
            "Optional list of explained-variance aggregation modes to additionally report, "
            "on top of --expvar_mode (unaffected), in both raw and lognorm space. E.g. "
            "'--expvar_modes global_mean variance_weighted_sum mean' adds "
            "expvar_test_probe_{gene_subset}_{mode}_{raw,lognorm} columns for every mode listed. "
            "Defaults to None, i.e. just [--expvar_mode] (today's single-mode behavior)."
        ),
    )
    p.add_argument(
        "--gene_subsets",
        type=str,
        nargs="+",
        default=["all_genes", "panel_genes_only", "non_panel_genes_only"],
        choices=["all_genes", "panel_genes_only", "non_panel_genes_only"],
        help=(
            "Which gene subset(s) to score the probe reconstruction against. This does NOT "
            "change what is reconstructed (always the full transcriptome) -- it is a "
            "scoring-time column mask on the already-computed reconstruction. Defaults to "
            "all three subsets (all_genes, panel_genes_only, non_panel_genes_only)."
        ),
    )

    p.add_argument("--dry_run", action="store_true", default=False,
                   help="Print resolved config and exit without running.")

    return p


def _args_to_config(args: argparse.Namespace) -> RidgeEvaluationConfig:
    return RidgeEvaluationConfig(
        input_file=Path(args.input_file),
        preprocessed_dir=Path(args.preprocessed_dir),
        output_dir=Path(args.output_dir),
        celltype_col=args.celltype_col,
        alpha=args.alpha,
        use_ridgecv=args.use_ridgecv,
        alpha_values=args.alpha_values or [0.01, 0.1, 1.0, 10.0, 100.0],
        n_splits=args.n_splits,
        random_state=args.random_state,
        external_panels=args.external_panels or [],
        external_names=args.external_names or [],
        strategies=args.strategies or [],
        probeset_sizes=args.probeset_sizes or [],
        expvar_mode=args.expvar_mode,
        expvar_modes=args.expvar_modes or [],
        gene_subsets=args.gene_subsets or ["all_genes"],
        dry_run=args.dry_run,
    )


def _save_parameters(config: RidgeEvaluationConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    params = asdict(config)
    params["timestamp"] = datetime.now().isoformat()
    params["script"] = "run_ridge_evaluation.py"
    param_file = config.output_dir / "ridge_evaluation_parameters.json"
    with open(param_file, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True, default=str)
    logger.info("Parameters saved to: %s", param_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse CLI arguments and run the ridge evaluation."""
    parser = _build_parser()
    args = parser.parse_args()
    config = _args_to_config(args)

    log_path = (
        config.output_dir
        / "logs"
        / f"ridge_evaluation_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    _setup_logging(log_path)

    if config.dry_run:
        logger.info("[DRY RUN] Resolved configuration:")
        for field_name, value in vars(config).items():
            logger.info("  %-35s = %s", field_name, value)
        logger.info("[DRY RUN] Exiting without execution.")
        return

    _save_parameters(config)

    logger.info("=" * 80)
    logger.info("RIDGE REGRESSION VARIABILITY EVALUATION PIPELINE")
    logger.info("=" * 80)
    _log_memory("start")

    # ── Load reference ───────────────────────────────────────────────────────
    ref_path = config.preprocessed_dir / "full_transcriptome.h5ad"
    if not ref_path.exists():
        if config.input_file.exists():
            ref_path = config.input_file
        else:
            logger.error(
                "Reference file not found at %s. Run run_evaluation.py --mode preprocess first.",
                ref_path,
            )
            sys.exit(1)

    logger.info("Loading reference from %s", ref_path)
    adata_full = sc.read_h5ad(ref_path)
    adata_full.obs_names_make_unique()
    _log_memory("after loading reference")

    if "counts" not in adata_full.layers:
        logger.error(
            "adata_full.layers['counts'] not found in %s. "
            "The ridge evaluation requires raw counts in layers['counts'].",
            ref_path,
        )
        sys.exit(1)

    if is_anndata_raw(adata_full):
        logger.error(
            "adata_full.X appears to contain raw counts in %s. "
            "Pre-normalised data is required in adata.X for the lognorm branch.",
            ref_path,
        )
        sys.exit(1)

    # ── Load preprocessed panel datasets ────────────────────────────────────
    filter_args: dict[str, Any] | None = None
    fa: dict[str, Any] = {}
    if config.strategies:
        fa["strategies"] = config.strategies
    if config.probeset_sizes:
        fa["probeset_sizes"] = config.probeset_sizes
    if fa:
        filter_args = fa

    preprocessed_datasets = _load_preprocessed_datasets(
        config.preprocessed_dir,
        reference_adata=adata_full,
        filter_args=filter_args,
    )
    _log_memory("after loading panels")

    # ── Generate train/test splits ───────────────────────────────────────────
    splits = generate_evaluation_splits(
        adata_full,
        celltype_col=config.celltype_col,
        n_splits=config.n_splits,
        random_state=config.random_state,
    )
    logger.info(
        "Generated %d stratified fold(s) (random_state=%d)",
        len(splits), config.random_state,
    )

    # ── Run ridge variability evaluation ─────────────────────────────────────
    run_ridge_variability_evaluation(
        adata_full=adata_full,
        preprocessed_datasets=preprocessed_datasets,
        output_dir=config.output_dir,
        splits=splits,
        alpha=config.alpha,
        use_ridgecv=config.use_ridgecv,
        alpha_values=config.alpha_values,
        celltype_col=config.celltype_col,
        random_state=config.random_state,
        expvar_mode=config.expvar_mode,
        expvar_modes=config.expvar_modes or [config.expvar_mode],
        gene_subsets=config.gene_subsets,
    )
    _log_memory("after ridge variability evaluation")

    logger.info("=" * 80)
    logger.info("RIDGE PIPELINE COMPLETE – outputs in: %s", config.output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
