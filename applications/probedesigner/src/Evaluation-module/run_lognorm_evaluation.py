#!/usr/bin/env python3
"""Standalone NMF variability evaluation with log-normalized reconstruction comparison.

Fits NMF on raw counts (same as the main variability evaluation in
``run_evaluation.py``), but compares reconstruction quality in
**log-normalized space**: both the raw test counts and the raw reconstructed
counts are log-normalized with an identical ``target_sum`` before computing
MSE and explained variance.

The ``target_sum`` is derived from the reference (not recomputed on the
reconstruction), so the comparison is on a consistent scale.  Per-cell-type
evaluation uses a cell-type-specific ``target_sum`` for the same reason.

Usage::

    python run_lognorm_evaluation.py \\
        --input_file preprocessed/full_transcriptome.h5ad \\
        --preprocessed_dir preprocessed/ \\
        --output_dir Evaluation-Results/ \\
        --celltype_col cluster \\
        --n_components 5 \\
        --n_splits 5

    # Dry-run: print resolved config without executing
    python run_lognorm_evaluation.py --dry_run \\
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

try:
    import psutil
except ImportError:
    psutil = None

import scanpy as sc
import scipy.sparse
from tqdm import tqdm

pd.options.mode.string_storage = "python"

# ---------------------------------------------------------------------------
# Internal module imports
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(_MODULE_DIR))

from _filters import filter_datasets_by_args
from _splits import generate_evaluation_splits
from nmf_lognorm import nmf_reconstruction_lognorm, nmf_reconstruction_by_celltype_lognorm

# ---------------------------------------------------------------------------
# Utility module
# ---------------------------------------------------------------------------

_UTILITY_DIR = _MODULE_DIR.parent / "Utility-module"
sys.path.insert(0, str(_UTILITY_DIR))
from _validation import is_anndata_raw_layer  # type: ignore[import]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _setup_logging(log_file: str | Path) -> None:
    """Configure root logger to write to both a file and stdout."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _log_memory(stage: str = "") -> None:
    if psutil is not None:
        mem_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        logger.info("Memory %s: %.2f GB", stage, mem_gb)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class LognormEvaluationConfig:
    """Runtime configuration for the lognorm NMF evaluation pipeline.

    Attributes:
        input_file: Path to the preprocessed full-transcriptome h5ad.
        preprocessed_dir: Directory containing panel h5ad files.
        output_dir: Root output directory. Results go into
            ``Variability-Evaluation-Lognorm/results/nmf/``.
        celltype_col: obs column holding cell-type labels.
        n_components: Number of NMF components.
        n_splits: Number of folds for stratified k-fold CV.
        random_state: Random seed for reproducibility.
        external_panels: Paths to additional panel CSV files.
        external_names: Display names for external panels.
        strategies: Panel strategy filter (empty = no filter).
        probeset_sizes: Panel size filter.
        dry_run: If True, print config and exit without running.
    """

    input_file: Path
    preprocessed_dir: Path
    output_dir: Path
    celltype_col: str = "cluster"
    n_components: int = 5
    n_splits: int = 5
    random_state: int = 42
    external_panels: list[str] = field(default_factory=list)
    external_names: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    probeset_sizes: list[int] = field(default_factory=list)
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dense_f64(X: Any, idx: np.ndarray) -> np.ndarray:
    s = X[idx]
    if scipy.sparse.issparse(s):
        s = s.toarray()
    return np.asarray(s, dtype=np.float64)


def _compute_global_target_sum(adata: Any) -> float:
    """Compute the global median per-cell total from raw counts.

    This is the value that ``sc.pp.normalize_total(adata)`` (no args) uses
    internally. By computing it once from the reference and reusing it for
    both the reference and the reconstruction, both matrices are normalized
    to the same scale.

    Args:
        adata: AnnData with ``layers['counts']``.

    Returns:
        Median per-cell total count as a float.
    """
    raw = adata.layers["counts"]
    cell_totals = np.array(raw.sum(axis=1)).ravel()
    return float(np.median(cell_totals))



def _save_results_per_dataset(
    results_df: pd.DataFrame,
    output_subdir: str | Path,
    metric_name: str,
) -> None:
    """Save evaluation results as per-dataset CSV files.

    Args:
        results_df: DataFrame with a ``"dataset"`` column.
        output_subdir: Directory where per-dataset CSV files are saved.
        metric_name: Human-readable label used in log messages.
    """
    if results_df is None or results_df.empty or "dataset" not in results_df.columns:
        logger.warning("No results to save for %s (empty or missing 'dataset' column)", metric_name)
        return

    os.makedirs(output_subdir, exist_ok=True)
    datasets = results_df["dataset"].unique()
    logger.info("Saving %s results for %d datasets to: %s", metric_name, len(datasets), output_subdir)
    for dataset_name in datasets:
        subset = results_df[results_df["dataset"] == dataset_name]
        safe_name = str(dataset_name).replace("/", "_").replace(" ", "_")
        subset.to_csv(Path(output_subdir) / f"{safe_name}.csv", index=False)
        logger.info("  Saved %s: %d rows", dataset_name, len(subset))
    logger.info("%s results saved.", metric_name)


def _aggregate_fold_results(df_per_fold: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold results by computing mean ± std across folds.

    Args:
        df_per_fold: DataFrame with per-fold results (must have ``"fold"`` column).

    Returns:
        Aggregated DataFrame with ``_mean`` / ``_std`` columns.
    """
    if "fold" not in df_per_fold.columns:
        logger.warning("No 'fold' column found in results. Returning original DataFrame.")
        return df_per_fold

    group_cols = ["dataset", "gene_list", "analysis_type"]
    if "celltype" in df_per_fold.columns:
        group_cols.append("celltype")

    # Metric columns = every numeric column except the grouping columns and "fold".
    # Dynamic numeric-dtype detection (matches run_evaluation.py) so any new metric
    # column is fold-aggregated automatically instead of being silently dropped.
    exclude_cols = set(group_cols) | {"fold"}
    available_metrics = [
        c for c in df_per_fold.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]
    if not available_metrics:
        logger.warning("No metric columns found for aggregation.")
        return df_per_fold

    agg_dict = {col: ["mean", "std"] for col in available_metrics}
    df_agg = df_per_fold.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    df_agg.columns = [
        "_".join(col).rstrip("_") if col[1] else col[0]
        for col in df_agg.columns.values
    ]
    return df_agg


def _load_preprocessed_datasets(
    preprocessed_dir: str | Path,
    reference_adata: sc.AnnData | None = None,
    filter_args: dict[str, Any] | None = None,
) -> dict[str, sc.AnnData]:
    """Load preprocessed panel h5ad files from *preprocessed_dir*.

    Args:
        preprocessed_dir: Directory containing panel h5ad files.
        reference_adata: Optional full-transcriptome AnnData to inject as
            ``"full_transcriptome"`` key.
        filter_args: Optional filter criteria (see
            :func:`_filters.filter_datasets_by_args`).

    Returns:
        Mapping of ``{dataset_name: AnnData}``.

    Raises:
        FileNotFoundError: If *preprocessed_dir* does not exist.
        ValueError: If no h5ad files survive loading / filtering.
    """
    preprocessed_dir = Path(preprocessed_dir)
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")

    h5ad_files = glob.glob(str(preprocessed_dir / "*.h5ad"))
    if not h5ad_files:
        raise ValueError(f"No h5ad files found in {preprocessed_dir}")

    logger.info("Found %d preprocessed h5ad files in %s", len(h5ad_files), preprocessed_dir)

    if filter_args:
        h5ad_files = filter_datasets_by_args(h5ad_files, filter_args)

    datasets: dict[str, sc.AnnData] = {}

    if reference_adata is not None:
        ref = reference_adata.copy()
        ref.obs_names_make_unique()
        datasets["full_transcriptome"] = ref
        logger.info("Injected full_transcriptome reference: %d genes", reference_adata.n_vars)

    for h5ad_file in sorted(h5ad_files):
        dataset_name = Path(h5ad_file).stem
        try:
            adata = sc.read_h5ad(h5ad_file)
            adata.obs_names_make_unique()
            if "dataset_name" not in adata.uns:
                adata.uns["dataset_name"] = dataset_name
            datasets[dataset_name] = adata
            logger.info(
                "Loaded %-50s  %d genes", dataset_name, adata.n_vars,
            )
        except Exception as exc:
            logger.warning("Failed to load %s: %s – skipping.", h5ad_file, exc, exc_info=True)

    if not datasets or (reference_adata is not None and len(datasets) == 1):
        raise ValueError("No valid preprocessed datasets loaded!")

    logger.info("Loaded %d datasets total.", len(datasets))
    return datasets


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_lognorm_variability_evaluation(
    adata_full: sc.AnnData,
    preprocessed_datasets: dict[str, sc.AnnData],
    output_dir: str | Path,
    splits: list[tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    n_components: int = 5,
    celltype_col: str = "cluster",
    random_state: int = 42,
) -> dict[str, Any]:
    """Run NMF variability evaluation in log-normalized reconstruction space.

    Fits NMF on raw counts, reconstructs the full transcriptome from probe
    genes, then evaluates reconstruction quality after log-normalizing both
    the raw test counts and the raw reconstruction with the same
    ``target_sum``.

    The ``target_sum`` is computed once from the reference (global median
    per-cell total), not from the reconstruction, ensuring a consistent scale.

    Args:
        adata_full: Full-transcriptome AnnData reference. Must have
            ``layers['counts']`` with raw integer counts.
        preprocessed_datasets: ``{name: AnnData}`` dict. Each non-reference
            dataset provides ``var_names`` as the probe gene list.
        output_dir: Root output directory. Results are written to
            ``Variability-Evaluation-Lognorm/results/nmf/``.
        splits: List of ``(train_idx, test_idx, per_celltype_splits)`` tuples
            from :func:`_splits.generate_evaluation_splits`.
        n_components: Number of NMF components.
        celltype_col: obs column for cell-type labels.
        random_state: Random seed.

    Returns:
        Dict with ``"nmf"`` (aggregated) and ``"nmf_per_fold"`` DataFrames.

    Raises:
        ValueError: If ``layers['counts']`` is not present in ``adata_full``.
    """
    logger.info("=" * 80)
    logger.info("LOGNORM VARIABILITY EVALUATION: NMF (%d fold(s))", len(splits))
    logger.info("=" * 80)

    if "counts" not in adata_full.layers:
        raise ValueError(
            "run_lognorm_variability_evaluation: adata_full.layers['counts'] not found. "
            "Raw counts are required."
        )

    if not is_anndata_raw_layer(adata_full, "counts"):
        raise ValueError(
            "adata_full.layers['counts'] does not appear to contain raw integer counts. "
            "Check preprocessing."
        )

    output_dir = Path(output_dir)
    results_dir = output_dir / "Variability-Evaluation-Lognorm" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if celltype_col not in adata_full.obs.columns:
        logger.error("Cell-type column '%s' not found in adata_full.obs.", celltype_col)
        return {}

    # ── Precompute target sums (fold-independent) ────────────────────────────
    logger.info("Computing global target_sum from raw counts...")
    target_sum_global = _compute_global_target_sum(adata_full)
    logger.info("Global target_sum = %.2f", target_sum_global)

    # Use the global target_sum for per-CT evaluation so that NMF lognorm
    # metrics are comparable with PCA and ridge (which operate on adata.X,
    # which was normalised with this same global target_sum during preprocessing).
    celltypes = adata_full.obs[celltype_col].unique()
    ct_target_sums = {ct: target_sum_global for ct in celltypes}
    logger.info("Using global target_sum=%.2f for all %d cell types.", target_sum_global, len(ct_target_sums))

    X_full = adata_full.layers["counts"]
    panel_names = [n for n in preprocessed_datasets if n != "full_transcriptome"]
    logger.info(
        "Evaluating lognorm variability for %d panels across %d fold(s)...",
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

        A_train = _to_dense_f64(X_full, train_idx)
        A_test = _to_dense_f64(X_full, test_idx)

        # NMF caches: keyed by n_components so baseline is computed once per fold
        cached_full_nmf: dict[int, Any] = {}
        cached_ct_nmf: dict[str, dict] = {}

        for panel_name in tqdm(panel_names, desc=f"Fold {fold + 1} lognorm eval", leave=False):
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
                    target_sum_global=target_sum_global,
                    ct_splits_idx=ct_splits_idx,
                    ct_target_sums=ct_target_sums,
                    cached_full_nmf=cached_full_nmf,
                    cached_ct_nmf=cached_ct_nmf,
                    n_components=panel_n,
                    celltype_col=celltype_col,
                    random_state=random_state,
                    rows=all_rows,
                    fold=fold,
                )
            except Exception as exc:
                logger.warning(
                    "Fold %d: lognorm evaluation failed for '%s': %s",
                    fold, panel_name, exc, exc_info=True,
                )
                if panel_name not in skipped:
                    skipped.append(panel_name)

        gc.collect()
        _log_memory(f"after fold {fold + 1}")

    results: dict[str, Any] = {}
    if all_rows:
        df_all = pd.DataFrame(all_rows)

        per_fold_dir = results_dir / "nmf" / "per_fold"
        per_fold_dir.mkdir(parents=True, exist_ok=True)
        _save_results_per_dataset(df_all, per_fold_dir, "Lognorm NMF (per-fold)")

        df_agg = _aggregate_fold_results(df_all)
        agg_dir = results_dir / "nmf"
        _save_results_per_dataset(df_agg, agg_dir, "Lognorm NMF (aggregated)")

        results["nmf"] = df_agg
        results["nmf_per_fold"] = df_all

    logger.info("Lognorm variability evaluation complete. Skipped: %s", skipped or "none")
    return results


def _evaluate_single_panel(
    panel_name: str,
    adata_full: sc.AnnData,
    probeset_genes: list[str],
    A_train: np.ndarray,
    A_test: np.ndarray,
    target_sum_global: float,
    ct_splits_idx: dict[str, tuple[np.ndarray, np.ndarray]],
    ct_target_sums: dict[str, float],
    cached_full_nmf: dict[int, Any],
    cached_ct_nmf: dict[str, dict],
    n_components: int,
    celltype_col: str,
    random_state: int,
    rows: list[dict[str, Any]],
    fold: int | None = None,
) -> None:
    """Run lognorm NMF evaluation for one panel.

    Appends result rows in-place to *rows*.

    Args:
        panel_name: Dataset identifier.
        adata_full: Full-transcriptome AnnData.
        probeset_genes: Genes in this panel.
        A_train: Global train raw counts (cells × genes), float64.
        A_test: Global test raw counts (cells × genes), float64.
        target_sum_global: Normalization target derived from all cells.
        ct_splits_idx: Per-cell-type global index splits.
        ct_target_sums: Per-cell-type normalization targets.
        cached_full_nmf: Global NMF cache (mutated in place), keyed by
            ``n_components``.
        cached_ct_nmf: Per-cell-type NMF cache (mutated in place).
        n_components: NMF components for this panel.
        celltype_col: obs column for cell-type labels.
        random_state: Random seed.
        rows: Output list (mutated in place).
        fold: Fold number.
    """
    # ── Global lognorm evaluation ─────────────────────────────────────────────
    logger.info("  [%s] Global lognorm NMF (n=%d)...", panel_name, n_components)

    global_result = nmf_reconstruction_lognorm(
        adata=adata_full,
        probeset_genes=probeset_genes,
        A_train=A_train,
        A_test=A_test,
        target_sum_global=target_sum_global,
        n_components=n_components,
        random_state=random_state,
        cached_full_nmf=cached_full_nmf.get(n_components),
    )

    if global_result:
        # Update cache so subsequent panels can reuse the NMF W/H
        if "computed_full_nmf" in global_result:
            cached_full_nmf[n_components] = global_result.pop("computed_full_nmf")
        elif n_components in cached_full_nmf:
            # result came from cache — no new nmf to store
            global_result.pop("computed_full_nmf", None)

        row = {**global_result, "dataset": panel_name, "gene_list": panel_name,
               "analysis_type": "global"}
        if fold is not None:
            row["fold"] = fold
        rows.append(row)

    # ── Per-cell-type lognorm evaluation ──────────────────────────────────────
    if celltype_col in adata_full.obs.columns and ct_splits_idx:
        logger.info("  [%s] Per-cell-type lognorm NMF...", panel_name)

        ct_result = nmf_reconstruction_by_celltype_lognorm(
            adata=adata_full,
            probeset_genes=probeset_genes,
            celltype_column=celltype_col,
            per_celltype_splits=ct_splits_idx,
            ct_target_sums=ct_target_sums,
            n_components=n_components,
            random_state=random_state,
            cached_full_nmf_by_celltype=cached_ct_nmf,
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
    """Build CLI argument parser for the lognorm evaluation script.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    p = argparse.ArgumentParser(
        description=(
            "Lognorm NMF variability evaluation: "
            "fits NMF on raw counts and compares reconstruction quality "
            "in log-normalized space."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
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
        help="Root output directory. Results go into Variability-Evaluation-Lognorm/.",
    )

    # Evaluation parameters
    p.add_argument("--celltype_col", default="cluster",
                   help="obs column holding cell-type labels (default: cluster).")
    p.add_argument("--n_components", type=int, default=5,
                   help="Number of NMF components (default: 5).")
    p.add_argument("--n_splits", type=int, default=5,
                   help="Number of k-fold cross-validation splits (default: 5).")
    p.add_argument("--random_state", type=int, default=42,
                   help="Random seed (default: 42).")

    # External panels
    p.add_argument("--external_panels", nargs="*", default=[],
                   help="Paths to additional panel CSV files.")
    p.add_argument("--external_names", nargs="*", default=[],
                   help="Display names for external panels.")

    # Dataset filters
    p.add_argument("--strategies", nargs="*", default=[],
                   help="Filter to specific selection strategies.")
    p.add_argument("--probeset_sizes", nargs="*", type=int, default=[],
                   help="Filter to specific probeset sizes.")

    # Dry-run
    p.add_argument("--dry_run", action="store_true", default=False,
                   help="Print resolved config and exit without running.")

    return p


def _args_to_config(args: argparse.Namespace) -> LognormEvaluationConfig:
    """Convert parsed CLI arguments to a :class:`LognormEvaluationConfig`.

    Args:
        args: Parsed namespace from :func:`_build_parser`.

    Returns:
        Populated :class:`LognormEvaluationConfig` instance.
    """
    return LognormEvaluationConfig(
        input_file=Path(args.input_file),
        preprocessed_dir=Path(args.preprocessed_dir),
        output_dir=Path(args.output_dir),
        celltype_col=args.celltype_col,
        n_components=args.n_components,
        n_splits=args.n_splits,
        random_state=args.random_state,
        external_panels=args.external_panels or [],
        external_names=args.external_names or [],
        strategies=args.strategies or [],
        probeset_sizes=args.probeset_sizes or [],
        dry_run=args.dry_run,
    )


def _save_parameters(config: LognormEvaluationConfig) -> None:
    """Save configuration parameters to JSON in the output directory.

    Args:
        config: Pipeline configuration.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    params = asdict(config)
    params["timestamp"] = datetime.now().isoformat()
    params["script"] = "run_lognorm_evaluation.py"
    param_file = config.output_dir / "lognorm_evaluation_parameters.json"
    with open(param_file, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True, default=str)
    logger.info("Parameters saved to: %s", param_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse CLI arguments and run the lognorm evaluation."""
    parser = _build_parser()
    args = parser.parse_args()
    config = _args_to_config(args)

    log_path = (
        config.output_dir
        / "logs"
        / f"lognorm_evaluation_{datetime.now():%Y%m%d_%H%M%S}.log"
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
    logger.info("LOGNORM VARIABILITY EVALUATION PIPELINE")
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
            "The lognorm evaluation requires raw counts.",
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

    # ── Run lognorm variability evaluation ───────────────────────────────────
    run_lognorm_variability_evaluation(
        adata_full=adata_full,
        preprocessed_datasets=preprocessed_datasets,
        output_dir=config.output_dir,
        splits=splits,
        n_components=config.n_components,
        celltype_col=config.celltype_col,
        random_state=config.random_state,
    )
    _log_memory("after lognorm variability evaluation")

    logger.info("=" * 80)
    logger.info("LOGNORM PIPELINE COMPLETE – outputs in: %s", config.output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
