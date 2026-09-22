#!/usr/bin/env python3
"""Standalone scVI variability evaluation using nico2_lib's ScviPredictor.

Fits scVI on **raw counts** (``adata.layers['counts']``) — unlike ``run_pca_evaluation.py``/
``run_ica_evaluation.py``, which require pre-normalised ``adata.X`` — then compares
reconstruction quality using only probe-gene expression. Global reconstruction only; there
is no per-celltype variant (see ``scvi_eval.py``'s module docstring for why: nico2_lib's
``ScviPredictor`` retrains a full VAE from scratch on every ``.predict()`` call with no
caching, so a per-celltype loop would multiply an already-expensive cost by the number of
cell types).

**Cost warning**: per fold, this runs 2 SCVI trainings for the baseline plus 2 more per
panel (train-probe + test-probe). For N panels across K folds that's ``K * (2 + 2*N)``
independent VAE trainings at ``--max_epochs`` each — by far the most expensive of the five
reconstruction methods in this pipeline. Size ``--max_epochs``/``--n_splits`` accordingly.

Usage::

    python run_scvi_evaluation.py \\
        --input_file preprocessed/full_transcriptome.h5ad \\
        --preprocessed_dir preprocessed/ \\
        --output_dir Evaluation-Results/ \\
        --celltype_col cluster \\
        --n_components 10 \\
        --max_epochs 200 \\
        --n_splits 5

    # Dry-run: print resolved config without executing
    python run_scvi_evaluation.py --dry_run \\
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
    to_dense as _to_dense,
    save_results_per_dataset as _save_results_per_dataset,
    aggregate_fold_results as _aggregate_fold_results,
    load_preprocessed_datasets as _load_preprocessed_datasets,
)
from _splits import generate_evaluation_splits
from scvi_eval import scvi_reconstruction, _compute_full_scvi_baseline

# ---------------------------------------------------------------------------
# Utility module
# ---------------------------------------------------------------------------

_UTILITY_DIR = _MODULE_DIR.parent / "Utility-module"
sys.path.insert(0, str(_UTILITY_DIR))
from _validation import is_anndata_raw_layer  # type: ignore[import]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScviEvaluationConfig:
    """Runtime configuration for the scVI variability evaluation pipeline.

    Attributes:
        input_file: Path to the preprocessed full-transcriptome h5ad.
        preprocessed_dir: Directory containing panel h5ad files.
        output_dir: Root output directory. Results go into
            ``Variability-Evaluation/results/SCVI/``.
        celltype_col: obs column holding cell-type labels. Used only to
            cell-type-stratify the k-fold splits (``generate_evaluation_splits``) — there
            is no per-celltype scVI evaluation.
        n_components: scVI latent dimension. Fixed as an explicit int (not ``None``) —
            leaving it unset triggers ``ScviPredictor``'s own hidden KMeans/kneedle
            search, an extra cost this evaluation deliberately avoids defaulting into.
        max_epochs: Training epochs per VAE fit. Each ``.predict()`` call retrains a
            fresh VAE from scratch — see ``scvi_eval.py``'s module docstring for the cost
            model this implies.
        n_splits: Number of folds for stratified k-fold CV.
        random_state: Random seed for the fold/split assignment only — it has no effect
            on scVI training itself, since ``ScviPredictor`` exposes no seed control.
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
    n_components: int = 10
    max_epochs: int = 200
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
# Main evaluation
# ---------------------------------------------------------------------------


def run_scvi_variability_evaluation(
    adata_full: sc.AnnData,
    preprocessed_datasets: dict[str, sc.AnnData],
    output_dir: str | Path,
    splits: list[tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    n_components: int = 10,
    max_epochs: int = 200,
    celltype_col: str = "cluster",
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
) -> dict[str, Any]:
    """Run scVI variability evaluation using raw counts (``adata.layers['counts']``).

    Global reconstruction only — see ``scvi_eval.py``'s module docstring for why there is no
    per-celltype variant, and for the cost model (``K * (2 + 2*N)`` independent VAE
    trainings across ``K`` folds and ``N`` panels).

    Args:
        adata_full: Full-transcriptome AnnData reference. ``adata.layers['counts']``
            must contain raw integer counts.
        preprocessed_datasets: ``{name: AnnData}`` dict. Each non-reference dataset
            provides ``var_names`` as the probe gene list.
        output_dir: Root output directory. Results are written to
            ``Variability-Evaluation/results/SCVI/``.
        splits: List of ``(train_idx, test_idx, per_celltype_splits)`` tuples from
            :func:`_splits.generate_evaluation_splits`. Only ``train_idx``/``test_idx``
            are used here (no per-celltype loop).
        n_components: scVI latent dimension.
        max_epochs: Training epochs per VAE fit.
        celltype_col: obs column for cell-type labels — used only for logging context;
            the splits are already stratified by this column before being passed in.
        expvar_mode: Explained-variance aggregation mode (see ``metrics.EXPVAR_MODES``).
        expvar_modes: Optional list of explained-variance modes to additionally report
            (see ``scvi_eval.py::scvi_reconstruction``). Defaults to ``None``, i.e.
            ``[expvar_mode]``.
        gene_subsets: Optional list of gene subsets to score against. Defaults to
            ``None``, i.e. ``["all_genes"]``.

    Returns:
        Dict with ``"scvi"`` (aggregated) and ``"scvi_per_fold"`` DataFrames.

    Raises:
        ValueError: If ``adata_full.layers['counts']`` is missing or not raw counts.
    """
    logger.info("=" * 80)
    logger.info("scVI VARIABILITY EVALUATION (%d fold(s))", len(splits))
    logger.info("=" * 80)

    if "counts" not in adata_full.layers:
        raise ValueError(
            "run_scvi_variability_evaluation: adata_full.layers['counts'] not found. "
            "scVI evaluation requires raw counts in layers['counts']."
        )
    if not is_anndata_raw_layer(adata_full, "counts"):
        raise ValueError(
            "run_scvi_variability_evaluation: adata_full.layers['counts'] does not "
            "contain raw integer counts."
        )

    output_dir = Path(output_dir)
    results_dir = output_dir / "Variability-Evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X_full = adata_full.layers["counts"]
    panel_names = [n for n in preprocessed_datasets if n != "full_transcriptome"]
    logger.info(
        "Evaluating scVI variability for %d panels across %d fold(s) "
        "(n_components=%d, max_epochs=%d) — expect ~%d SCVI training runs...",
        len(panel_names), len(splits), n_components, max_epochs,
        len(splits) * (2 + 2 * len(panel_names)),
    )

    all_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for fold, (train_idx, test_idx, _ct_splits_idx) in enumerate(splits):
        logger.info("-" * 60)
        logger.info(
            "FOLD %d/%d: %d training cells, %d test cells",
            fold + 1, len(splits), len(train_idx), len(test_idx),
        )
        logger.info("-" * 60)

        A_train = _to_dense(X_full, train_idx, dtype=np.float32)
        A_test = _to_dense(X_full, test_idx, dtype=np.float32)

        # Pre-compute scVI baseline for the default n_components, reusing the same
        # helper scvi_reconstruction() falls back to on a cache miss (single
        # implementation — see scvi_eval.py::_compute_full_scvi_baseline).
        cached_full_scvi: dict[int, Any] = {
            n_components: _compute_full_scvi_baseline(
                A_train, A_test, n_components, max_epochs,
            )
        }

        for panel_name in tqdm(panel_names, desc=f"Fold {fold + 1} scVI eval", leave=False):
            adata_panel = preprocessed_datasets[panel_name]
            probeset_genes = adata_panel.var_names.tolist()
            panel_n = int(adata_panel.uns.get("n_components", n_components))

            try:
                _evaluate_single_panel(
                    panel_name=panel_name,
                    adata_full=adata_full,
                    probeset_genes=probeset_genes,
                    A_train=A_train,
                    A_test=A_test,
                    cached_full_scvi=cached_full_scvi,
                    n_components=panel_n,
                    max_epochs=max_epochs,
                    rows=all_rows,
                    fold=fold,
                    expvar_mode=expvar_mode,
                    expvar_modes=expvar_modes,
                    gene_subsets=gene_subsets,
                )
            except Exception as exc:
                logger.warning(
                    "Fold %d: scVI evaluation failed for '%s': %s",
                    fold, panel_name, exc, exc_info=True,
                )
                if panel_name not in skipped:
                    skipped.append(panel_name)

        gc.collect()
        _log_memory(f"after fold {fold + 1}")

    results: dict[str, Any] = {}
    if all_rows:
        df_all = pd.DataFrame(all_rows)

        per_fold_dir = results_dir / "SCVI" / "per_fold"
        per_fold_dir.mkdir(parents=True, exist_ok=True)
        _save_results_per_dataset(df_all, per_fold_dir, "scVI (per-fold)")

        df_agg = _aggregate_fold_results(df_all)
        agg_dir = results_dir / "SCVI"
        _save_results_per_dataset(df_agg, agg_dir, "scVI (aggregated)")

        results["scvi"] = df_agg
        results["scvi_per_fold"] = df_all

    logger.info("scVI variability evaluation complete. Skipped: %s", skipped or "none")
    return results


def _evaluate_single_panel(
    panel_name: str,
    adata_full: sc.AnnData,
    probeset_genes: list[str],
    A_train: np.ndarray,
    A_test: np.ndarray,
    cached_full_scvi: dict[int, Any],
    n_components: int,
    max_epochs: int,
    rows: list[dict[str, Any]],
    fold: int | None = None,
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
) -> None:
    """Run scVI evaluation for one panel and append a result row in place.

    Global evaluation only (no per-celltype row).

    Args:
        panel_name: Dataset identifier.
        adata_full: Full-transcriptome AnnData.
        probeset_genes: Genes in this panel.
        A_train: Raw-count training data (cells × genes).
        A_test: Raw-count test data (cells × genes).
        cached_full_scvi: Global scVI cache (mutated in place), keyed by ``n_components``.
        n_components: scVI latent dimension for this panel.
        max_epochs: Training epochs per VAE fit.
        rows: Output list (mutated in place).
        fold: Fold number.
        expvar_mode: Explained-variance aggregation mode.
        expvar_modes: Optional additional explained-variance modes to report.
        gene_subsets: Optional gene subsets to score the reconstruction against.
    """
    logger.info("  [%s] Global scVI (n=%d, max_epochs=%d)...", panel_name, n_components, max_epochs)

    global_result = scvi_reconstruction(
        adata=adata_full,
        probeset_genes=probeset_genes,
        A_train=A_train,
        A_test=A_test,
        n_components=n_components,
        max_epochs=max_epochs,
        cached_full_scvi=cached_full_scvi.get(n_components),
        expvar_mode=expvar_mode,
        expvar_modes=expvar_modes,
        gene_subsets=gene_subsets,
    )

    if global_result:
        if "computed_full_scvi" in global_result:
            cached_full_scvi[n_components] = global_result.pop("computed_full_scvi")
        else:
            global_result.pop("computed_full_scvi", None)

        row = {**global_result, "dataset": panel_name, "gene_list": panel_name,
               "analysis_type": "global"}
        if fold is not None:
            row["fold"] = fold
        rows.append(row)


# ---------------------------------------------------------------------------
# Argument parsing and configuration
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the scVI evaluation script."""
    p = argparse.ArgumentParser(
        description=(
            "scVI variability evaluation: fits a scVI VAE on raw counts and compares "
            "full-transcriptome reconstruction quality from probe genes. Global "
            "reconstruction only (no per-celltype variant). WARNING: nico2_lib's "
            "ScviPredictor retrains a full VAE from scratch on every .predict() call "
            "with no caching -- expect K * (2 + 2*N) independent VAE trainings for "
            "N panels across K folds. Size --max_epochs/--n_splits accordingly."
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
        help="Root output directory. Results go into Variability-Evaluation/results/SCVI/.",
    )

    p.add_argument("--celltype_col", default="cluster",
                   help="obs column holding cell-type labels, used only to stratify "
                        "the k-fold splits (default: cluster). No per-celltype evaluation.")
    p.add_argument("--n_components", type=int, default=10,
                   help="scVI latent dimension (default: 10). Left as an explicit int "
                        "rather than unset, to avoid ScviPredictor's hidden KMeans/kneedle "
                        "component search.")
    p.add_argument("--max_epochs", type=int, default=200,
                   help="Training epochs per VAE fit (default: 200). Every "
                        ".predict() call retrains a fresh VAE from scratch -- see "
                        "module docstring for the resulting cost model.")
    p.add_argument("--n_splits", type=int, default=5,
                   help="Number of k-fold cross-validation splits (default: 5).")
    p.add_argument("--random_state", type=int, default=42,
                   help="Random seed for the fold/split assignment only (default: 42). "
                        "Has no effect on scVI training itself -- ScviPredictor exposes "
                        "no seed control, so results are not reproducible run-to-run.")

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
            "on top of --expvar_mode (unaffected). Defaults to None, i.e. just "
            "[--expvar_mode] (today's single-mode behavior)."
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


def _args_to_config(args: argparse.Namespace) -> ScviEvaluationConfig:
    return ScviEvaluationConfig(
        input_file=Path(args.input_file),
        preprocessed_dir=Path(args.preprocessed_dir),
        output_dir=Path(args.output_dir),
        celltype_col=args.celltype_col,
        n_components=args.n_components,
        max_epochs=args.max_epochs,
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


def _save_parameters(config: ScviEvaluationConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    params = asdict(config)
    params["timestamp"] = datetime.now().isoformat()
    params["script"] = "run_scvi_evaluation.py"
    param_file = config.output_dir / "scvi_evaluation_parameters.json"
    with open(param_file, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True, default=str)
    logger.info("Parameters saved to: %s", param_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse CLI arguments and run the scVI evaluation."""
    parser = _build_parser()
    args = parser.parse_args()
    config = _args_to_config(args)

    log_path = (
        config.output_dir
        / "logs"
        / f"scvi_evaluation_{datetime.now():%Y%m%d_%H%M%S}.log"
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
    logger.info("scVI VARIABILITY EVALUATION PIPELINE")
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

    if "counts" not in adata_full.layers or not is_anndata_raw_layer(adata_full, "counts"):
        logger.error(
            "adata_full.layers['counts'] missing or does not contain raw integer counts "
            "in %s. scVI evaluation requires raw counts in layers['counts'].",
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
        "Generated %d stratified fold(s) (random_state=%d; controls fold assignment "
        "only, not scVI training)",
        len(splits), config.random_state,
    )

    # ── Run scVI variability evaluation ──────────────────────────────────────
    run_scvi_variability_evaluation(
        adata_full=adata_full,
        preprocessed_datasets=preprocessed_datasets,
        output_dir=config.output_dir,
        splits=splits,
        n_components=config.n_components,
        max_epochs=config.max_epochs,
        celltype_col=config.celltype_col,
        expvar_mode=config.expvar_mode,
        expvar_modes=config.expvar_modes or [config.expvar_mode],
        gene_subsets=config.gene_subsets,
    )
    _log_memory("after scVI variability evaluation")

    logger.info("=" * 80)
    logger.info("scVI PIPELINE COMPLETE – outputs in: %s", config.output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
