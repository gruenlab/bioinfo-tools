"""Shared CLI helpers for the standalone Evaluation-module scripts
(``run_pca_evaluation.py``, ``run_ica_evaluation.py``, ``run_ridge_evaluation.py``,
``run_scvi_evaluation.py``).

These six functions were previously duplicated byte-for-byte (or, for ``to_dense``,
identical except for the target dtype) across all four scripts. Consolidated here
after an audit of why ``pca.py``/``ica.py``/``ridge.py``/``scvi_eval.py`` exist as
separate files found this layer of duplication was pure, zero-risk boilerplate.

Not used by ``run_evaluation.py`` (the main orchestrator), which has its own,
independently-evolved versions of similar helpers -- out of scope for this
consolidation, to avoid that file's much larger blast radius.
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse

try:
    import psutil
except ImportError:
    psutil = None

from _filters import filter_datasets_by_args

pd.options.mode.string_storage = "python"

logger = logging.getLogger(__name__)


def setup_logging(log_file: str | Path) -> None:
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


def log_memory(stage: str = "") -> None:
    if psutil is not None:
        mem_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        logger.info("Memory %s: %.2f GB", stage, mem_gb)


def to_dense(X: Any, idx: np.ndarray, dtype: type = np.float64) -> np.ndarray:
    """Densify (if needed) and cast a row-indexed slice of *X* to *dtype*."""
    s = X[idx]
    if scipy.sparse.issparse(s):
        s = s.toarray()
    return np.asarray(s, dtype=dtype)


def save_results_per_dataset(
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


def aggregate_fold_results(df_per_fold: pd.DataFrame) -> pd.DataFrame:
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


def load_preprocessed_datasets(
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
            logger.info("Loaded %-50s  %d genes", dataset_name, adata.n_vars)
        except Exception as exc:
            logger.warning("Failed to load %s: %s – skipping.", h5ad_file, exc, exc_info=True)

    if not datasets or (reference_adata is not None and len(datasets) == 1):
        raise ValueError("No valid preprocessed datasets loaded!")

    logger.info("Loaded %d datasets total.", len(datasets))
    return datasets
