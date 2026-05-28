#!/usr/bin/env python3
"""Unified evaluation pipeline entry point.

This script provides a single CLI to run any combination of:

- **preprocess**: Subset panels to selected genes and compute PCA/NMF,
  clustering, and KNN graphs.
- **evaluate**: Compute baseline (clustering/KNN/celltype) and/or
  variability (NMF) metrics on preprocessed panels.
- **both**: Run preprocessing immediately followed by evaluation in one
  command (equivalent to running the two modes sequentially).

Usage::

    # Preprocessing only
    python run_evaluation.py --mode preprocess \\
        --input_file data.h5ad \\
        --gene_lists_dir Selected-panels/ \\
        --preprocessed_dir preprocessed/ \\
        --output_dir Evaluation-Results/

    # Evaluation only (on existing preprocessed data)
    python run_evaluation.py --mode evaluate \\
        --input_file preprocessed/full_transcriptome.h5ad \\
        --preprocessed_dir preprocessed/ \\
        --output_dir Evaluation-Results/ \\
        --evaluation_type both

    # Everything in one go
    python run_evaluation.py --mode both \\
        --input_file data.h5ad \\
        --gene_lists_dir Selected-panels/ \\
        --preprocessed_dir preprocessed/ \\
        --output_dir Evaluation-Results/ \\
        --evaluation_type both \\
        --include_tangram

    # Dry-run: print resolved config without executing
    python run_evaluation.py --mode both --dry_run ...
"""

from __future__ import annotations

import argparse
import gc
import glob
import importlib.util
import json
import logging
import os
import sys
import traceback
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
from nico2_lib.predictors._nmf._nmf_pred import NmfPredictor
from tqdm import tqdm

# Configure pandas to use object dtype for strings instead of ArrowStringArray
# This ensures compatibility with anndata's HDF5 writer (pandas 2.x+ uses PyArrow strings by default)
pd.options.mode.string_storage = "python"

# Presentation-friendly defaults for Scanpy-generated figures
sc.set_figure_params(fontsize=16)

# ---------------------------------------------------------------------------
# Internal module imports
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(_MODULE_DIR))

from _clustering import (
    evaluate_celltype_identification,
    evaluate_clustering_quality,
    evaluate_neighborhood_preservation,
    compute_rare_celltype_marker_coverage,
)
from _filters import (
    filter_datasets_by_args,
    filter_datasets_by_keywords,
    group_datasets_by_attributes,
    parse_dataset_attributes,
)
from _preprocessing import (
    load_all_gene_lists,
    load_gene_list_from_csv,
    preprocess_reference_dataset,
    process_data_for_panel_evaluation,
    _convert_arrow_strings_to_object,
)
from metrics import (
    calculate_explained_variance,
    calculate_mse,
)
from nmf import nmf_reconstruction, nmf_reconstruction_by_celltype
from _splits import generate_evaluation_splits

# Optional Tangram reconstruction check
try:
    from _tangram import run_tangram_reconstruction_check
    _RECONSTRUCTION_AVAILABLE = True
except ImportError:
    _RECONSTRUCTION_AVAILABLE = False

# Constants
# Load local _constants.py by exact path to avoid collisions with sibling
# modules that also expose a flat "_constants" module name.
_EVAL_CONSTANTS_PATH = _MODULE_DIR / "_constants.py"
_eval_constants_spec = importlib.util.spec_from_file_location(
    "_evaluation_module_constants", _EVAL_CONSTANTS_PATH
)
if _eval_constants_spec is None or _eval_constants_spec.loader is None:
    raise ImportError(f"Could not load evaluation constants from {_EVAL_CONSTANTS_PATH}")
_eval_constants = importlib.util.module_from_spec(_eval_constants_spec)
_eval_constants_spec.loader.exec_module(_eval_constants)

DEFAULT_CELLTYPE_COLUMN = _eval_constants.DEFAULT_CELLTYPE_COLUMN
DEFAULT_DIMENSIONALITY_REDUCTION = _eval_constants.DEFAULT_DIMENSIONALITY_REDUCTION
DEFAULT_DIM_REDUCTION_PREPROCESS = _eval_constants.DEFAULT_DIM_REDUCTION_PREPROCESS
DEFAULT_EVALUATION_TYPE = _eval_constants.DEFAULT_EVALUATION_TYPE
DEFAULT_N_COMPONENTS = _eval_constants.DEFAULT_N_COMPONENTS
DEFAULT_N_NEIGHBORS = _eval_constants.DEFAULT_N_NEIGHBORS
DEFAULT_RANDOM_STATE = _eval_constants.DEFAULT_RANDOM_STATE
DEFAULT_TANGRAM_N_EPOCHS = _eval_constants.DEFAULT_TANGRAM_N_EPOCHS
DEFAULT_TEST_SIZE = _eval_constants.DEFAULT_TEST_SIZE

# ---------------------------------------------------------------------------
# Utility module (external dependency)
# ---------------------------------------------------------------------------

_UTILITY_DIR = _MODULE_DIR.parent / "Utility-module"
if _UTILITY_DIR.exists():
    sys.path.insert(0, str(_UTILITY_DIR))
try:
    from _validation import is_anndata_raw, is_anndata_raw_layer  # type: ignore[import]
except ImportError:
    def is_anndata_raw(adata) -> bool:  # type: ignore[misc]
        return True
    def is_anndata_raw_layer(adata, layer_name: str) -> bool:  # type: ignore[misc]
        return True
try:
    from _utils import convert_ensembl_to_gene_symbols  # type: ignore[import]
except ImportError:
    def convert_ensembl_to_gene_symbols(adata: sc.AnnData, inplace: bool = True) -> None:  # type: ignore[misc]
        """Stub – utility module not found."""
        pass

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _setup_logging(log_file: str | Path) -> None:
    """Configure root logger to write to both a file and stdout.

    Args:
        log_file: Path to the log file. Parent directories are created
            automatically.
    """
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
    """Log current RSS memory usage.

    Args:
        stage: Human-readable label for the current pipeline stage.
    """
    if psutil is not None:
        mem_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        logger.info("Memory %s: %.2f GB", stage, mem_gb)
    else:
        logger.debug("Memory logging skipped (psutil not installed)")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvaluationConfig:
    """Runtime configuration for the evaluation pipeline.

    Attributes:
        mode: Execution mode – ``"preprocess"``, ``"evaluate"``, or ``"both"``.
        input_file: Path to the raw or reference h5ad file.
        preprocessed_dir: Directory containing (or to receive) preprocessed
            panel h5ad files.
        output_dir: Root directory for all evaluation outputs.
        gene_lists_dir: Directory containing selection pipeline gene lists
            (required for ``preprocess`` / ``both`` modes).
        gene_list_files_txt: Path to a newline-separated text file of gene
            list CSV paths (alternative to *gene_lists_dir*).
        evaluation_type: Evaluation types to run:
            ``"baseline"``, ``"variability"``, or ``"both"``.
        celltype_col: obs column holding cell-type labels.
        dimensionality_reduction: Embedding for baseline evaluation:
            ``"pca"``, ``"nmf"``, or ``"both"``.
        n_components: Number of NMF components (variability evaluation).
        test_size: Fraction of cells held out for testing.
        random_state: Global random seed.
        n_splits: Number of folds for stratified k-fold cross-validation.
        split_mode: ``"kfold"`` (default, stratified k-fold) or ``"simple"``
            (single 80/20 random split). Both NMF and Tangram use the identical
            splits generated by this setting.
        n_neighbors: Nearest neighbours for KNN graph construction.
        dim_reduction_preprocess: Dimensionality reduction to compute during
            preprocessing (``"pca"``, ``"nmf"``, or ``"both"``).
        include_tangram: Run the optional Tangram reconstruction check.
        tangram_n_epochs: Number of Tangram mapping epochs.
        external_panels: Paths to external gene-panel CSV files.
        external_names: Display names for the external panels.
        external_probeset_sizes: Per-panel gene count limits (0 = keep all).
        add_10x_panels: 10x panel integration mode for preprocessing.
        data_dir: Base directory for 10x panel files.
        dry_run: Print resolved config and exit without executing.
        strategies: Strategy filter for evaluation dataset selection.
        probeset_sizes: Panel-size filter.
        filter_methods: Preprocessing filter method filter.
        hvg_subset_options: HVG option filter.
        reduction_types: Reduction type filter.
        analysis_types: Analysis type filter.
        dimred_methods: Dimred method filter.
        dt_percentages: DT percentage filter.
        dimred_percentages: Dimred percentage filter.
        run_celltype_specific_filling: Gap-filling filter.
        run_global_gene_filling: Gap-filling filter.
        run_deg_based_filling: Gap-filling filter.
        preferred_strategy: Preferred strategy filter.
    """

    # Required
    mode: str
    input_file: Path
    preprocessed_dir: Path
    output_dir: Path

    # Preprocessing inputs
    gene_lists_dir: Path | None = None
    gene_list_files_txt: Path | None = None

    # Evaluation settings
    evaluation_type: str = DEFAULT_EVALUATION_TYPE
    celltype_col: str = DEFAULT_CELLTYPE_COLUMN
    dimensionality_reduction: str = DEFAULT_DIMENSIONALITY_REDUCTION
    n_components: int = DEFAULT_N_COMPONENTS
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    n_splits: int = 5
    split_mode: str = "kfold"

    # Preprocessing settings
    n_neighbors: int = DEFAULT_N_NEIGHBORS
    dim_reduction_preprocess: str = DEFAULT_DIM_REDUCTION_PREPROCESS

    # Optional Tangram
    include_tangram: bool = False
    tangram_n_epochs: int = DEFAULT_TANGRAM_N_EPOCHS

    # External panels
    external_panels: list[str] = field(default_factory=list)
    external_names: list[str] = field(default_factory=list)
    external_probeset_sizes: list[int] = field(default_factory=list)

    # 10x integration
    add_10x_panels: str = "no-10x-panel"
    data_dir: str = ""

    # Dry-run
    dry_run: bool = False

    # Dataset filters
    strategies: list[str] = field(default_factory=list)
    probeset_sizes: list[int] = field(default_factory=list)
    filter_methods: list[str] = field(default_factory=list)
    hvg_subset_options: list[str] = field(default_factory=list)
    reduction_types: list[str] = field(default_factory=list)
    analysis_types: list[str] = field(default_factory=list)
    dimred_methods: list[str] = field(default_factory=list)
    dt_percentages: list[float] = field(default_factory=list)
    dimred_percentages: list[float] = field(default_factory=list)
    run_celltype_specific_filling: str = ""
    run_global_gene_filling: str = ""
    run_deg_based_filling: str = ""
    preferred_strategy: str = ""

    # NMF / Tangram count input
    nmf_counts_input: str = "raw"


def _save_evaluation_parameters(config: EvaluationConfig) -> None:
    """Save all evaluation parameter settings to a JSON file in the output directory."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    params = asdict(config)
    params["timestamp"] = datetime.now().isoformat()
    params["script"] = "run_evaluation.py"

    param_file = config.output_dir / "evaluation_parameters.json"
    with open(param_file, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True, default=str)

    logger.info("Parameter settings saved to: %s", param_file)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    p = argparse.ArgumentParser(
        description="Evaluation pipeline: preprocess and/or evaluate probe panels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    p.add_argument(
        "--mode",
        choices=["preprocess", "evaluate", "both"],
        default="evaluate",
        help="Pipeline mode: preprocess, evaluate, or both (default: evaluate).",
    )

    # Paths
    p.add_argument("--input_file", required=True, help="Path to raw or reference h5ad file.")
    p.add_argument("--preprocessed_dir", required=True, help="Preprocessed panel directory.")
    p.add_argument("--output_dir", required=True, help="Root output directory.")
    p.add_argument("--gene_lists_dir", default=None, help="Selection pipeline gene lists directory.")
    p.add_argument("--gene_list_files_txt", default=None, help="Text file with gene-list CSV paths (one per line).")

    # Evaluation
    p.add_argument("--evaluation_type", choices=["baseline", "variability", "both"], default="both")
    p.add_argument("--celltype_col", default="cluster")
    p.add_argument("--dimensionality_reduction", choices=["pca", "nmf", "both"], default="pca")
    p.add_argument("--n_components", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.3)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--n_splits", type=int, default=5, help="Number of folds for stratified k-fold cross-validation")
    p.add_argument(
        "--split_mode",
        choices=["simple", "kfold"],
        default="kfold",
        help=(
            "kfold (default): stratified k-fold using --n_splits folds (default 5). "
            "simple: one 80/20 random split. "
            "Both NMF and Tangram use the identical splits."
        ),
    )

    # Preprocessing
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--dim_reduction_preprocess", choices=["pca", "nmf", "both"], default="both")

    # Tangram
    p.add_argument("--include_tangram", action="store_true", default=False,
                   help="Run optional Tangram reconstruction check.")
    p.add_argument("--tangram_n_epochs", type=int, default=1000)

    # NMF / Tangram count input
    p.add_argument(
        "--nmf_counts_input",
        type=str,
        default="raw",
        choices=["raw", "lognorm"],
        help=(
            "Count matrix used as NMF and Tangram input. "
            "'raw' (default): raw integer counts from adata.raw or adata.layers['counts']. "
            "'lognorm': log-normalised counts from adata.X."
        ),
    )

    # External panels
    p.add_argument("--external_panels", nargs="*", default=[])
    p.add_argument("--external_names", nargs="*", default=[])
    p.add_argument("--external_probeset_sizes", nargs="*", type=int, default=[])

    # 10x integration
    p.add_argument("--add_10x_panels", default="no-10x-panel")
    p.add_argument("--data_dir", default="")

    # Dry-run
    p.add_argument("--dry_run", action="store_true", default=False,
                   help="Print resolved config and exit without executing.")

    # Dataset filters
    p.add_argument("--strategies", nargs="*", default=[])
    p.add_argument("--probeset_sizes", nargs="*", type=int, default=[])
    p.add_argument("--filter_methods", nargs="*", default=[])
    p.add_argument("--hvg_subset_options", nargs="*", default=[])
    p.add_argument("--reduction_types", nargs="*", default=[])
    p.add_argument("--analysis_types", nargs="*", default=[])
    p.add_argument("--dimred_methods", nargs="*", default=[])
    p.add_argument("--dt_percentages", nargs="*", type=float, default=[])
    p.add_argument("--dimred_percentages", nargs="*", type=float, default=[])
    p.add_argument("--run_celltype_specific_filling", default="")
    p.add_argument("--run_global_gene_filling", default="")
    p.add_argument("--run_deg_based_filling", default="")
    p.add_argument("--preferred_strategy", default="")
    p.add_argument("--external_names_filter", nargs="*", dest="external_names_filter", default=[],
                   help="External panel names to include in evaluation (display-name filter).")

    return p


def _args_to_config(args: argparse.Namespace) -> EvaluationConfig:
    """Convert parsed CLI arguments to an :class:`EvaluationConfig`.

    Args:
        args: Parsed namespace from :func:`_build_parser`.

    Returns:
        Populated :class:`EvaluationConfig` instance.
    """
    return EvaluationConfig(
        mode=args.mode,
        input_file=Path(args.input_file),
        preprocessed_dir=Path(args.preprocessed_dir),
        output_dir=Path(args.output_dir),
        gene_lists_dir=Path(args.gene_lists_dir) if args.gene_lists_dir else None,
        gene_list_files_txt=Path(args.gene_list_files_txt) if args.gene_list_files_txt else None,
        evaluation_type=args.evaluation_type,
        celltype_col=args.celltype_col,
        dimensionality_reduction=args.dimensionality_reduction,
        n_components=args.n_components,
        test_size=args.test_size,
        random_state=args.random_state,
        n_splits=args.n_splits,
        split_mode=args.split_mode,
        n_neighbors=args.n_neighbors,
        dim_reduction_preprocess=args.dim_reduction_preprocess,
        include_tangram=args.include_tangram,
        tangram_n_epochs=args.tangram_n_epochs,
        external_panels=args.external_panels or [],
        external_names=args.external_names or [],
        external_probeset_sizes=args.external_probeset_sizes or [],
        add_10x_panels=args.add_10x_panels,
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        strategies=args.strategies or [],
        probeset_sizes=args.probeset_sizes or [],
        filter_methods=args.filter_methods or [],
        hvg_subset_options=args.hvg_subset_options or [],
        reduction_types=args.reduction_types or [],
        analysis_types=args.analysis_types or [],
        dimred_methods=args.dimred_methods or [],
        dt_percentages=args.dt_percentages or [],
        dimred_percentages=args.dimred_percentages or [],
        run_celltype_specific_filling=args.run_celltype_specific_filling or "",
        run_global_gene_filling=args.run_global_gene_filling or "",
        run_deg_based_filling=args.run_deg_based_filling or "",
        preferred_strategy=args.preferred_strategy or "",
        nmf_counts_input=args.nmf_counts_input,
    )


# ---------------------------------------------------------------------------
# CSV saving helpers
# ---------------------------------------------------------------------------


def _save_results_per_dataset(
    results_df: pd.DataFrame,
    output_subdir: str | Path,
    metric_name: str,
) -> None:
    """Save evaluation results as per-dataset CSV files.

    For cell-type identification results (which have both dataset-level and
    per-cell-type rows), both row types are written to a single combined CSV.

    Args:
        results_df: DataFrame with a ``"dataset"`` column.
        output_subdir: Directory where per-dataset CSV files are saved.
        metric_name: Human-readable label used in log messages.
    """
    if results_df is None or results_df.empty or "dataset" not in results_df.columns:
        logger.warning("No results to save for %s (empty or missing 'dataset' column)", metric_name)
        return

    os.makedirs(output_subdir, exist_ok=True)
    is_celltype = "celltype" in results_df.columns

    if is_celltype:
        base_datasets = results_df[results_df["celltype"].isna()]["dataset"].unique()
        logger.info("Saving %s results for %d datasets to: %s", metric_name, len(base_datasets), output_subdir)

        for base_name in base_datasets:
            dataset_level = results_df[results_df["dataset"] == base_name].copy()
            per_ct_rows = [
                row for _, row in results_df[results_df["celltype"].notna()].iterrows()
                if row["dataset"] == f"{base_name}_{row['celltype']}"
            ]
            per_celltype = pd.DataFrame(per_ct_rows) if per_ct_rows else pd.DataFrame()
            combined = pd.concat([dataset_level, per_celltype], ignore_index=True)
            safe_name = base_name.replace("/", "_").replace(" ", "_")
            combined.to_csv(Path(output_subdir) / f"{safe_name}.csv", index=False)
            logger.info("  Saved %s: %d rows", base_name, len(combined))
    else:
        datasets = results_df["dataset"].unique()
        logger.info("Saving %s results for %d datasets to: %s", metric_name, len(datasets), output_subdir)
        for dataset_name in datasets:
            subset = results_df[results_df["dataset"] == dataset_name]
            safe_name = dataset_name.replace("/", "_").replace(" ", "_")
            subset.to_csv(Path(output_subdir) / f"{safe_name}.csv", index=False)
            logger.info("  Saved %s: %d rows", dataset_name, len(subset))

    logger.info("%s results saved.", metric_name)


# ---------------------------------------------------------------------------
# Preprocessed data loading
# ---------------------------------------------------------------------------


def _build_filter_args(config: EvaluationConfig) -> dict[str, Any] | None:
    """Build the filter_args dict from config fields.

    Returns ``None`` if no filters are active (avoids unnecessary work).

    Args:
        config: Pipeline configuration.

    Returns:
        Filter-args dictionary or ``None``.
    """
    fa: dict[str, Any] = {}
    if config.strategies:
        fa["strategies"] = config.strategies
    if config.probeset_sizes:
        fa["probeset_sizes"] = config.probeset_sizes
    if config.filter_methods:
        fa["filter_methods"] = config.filter_methods
    if config.hvg_subset_options:
        fa["hvg_subset_options"] = config.hvg_subset_options
    if config.reduction_types:
        fa["reduction_types"] = config.reduction_types
    if config.analysis_types:
        fa["analysis_types"] = config.analysis_types
    if config.dimred_methods:
        fa["dimred_methods"] = config.dimred_methods
    if config.dt_percentages:
        fa["dt_percentages"] = config.dt_percentages
    if config.dimred_percentages:
        fa["dimred_percentages"] = config.dimred_percentages
    if config.run_celltype_specific_filling:
        fa["run_celltype_specific_filling"] = config.run_celltype_specific_filling
    if config.run_global_gene_filling:
        fa["run_global_gene_filling"] = config.run_global_gene_filling
    if config.run_deg_based_filling:
        fa["run_deg_based_filling"] = config.run_deg_based_filling
    if config.preferred_strategy:
        fa["preferred_strategy"] = config.preferred_strategy
    return fa if fa else None


def _load_preprocessed_datasets(
    preprocessed_dir: str | Path,
    reference_adata: sc.AnnData | None = None,
    filter_args: dict[str, Any] | None = None,
) -> dict[str, sc.AnnData]:
    """Load preprocessed panel h5ad files.

    Args:
        preprocessed_dir: Directory containing panel h5ad files.
        reference_adata: Optional full-transcriptome AnnData to inject as
            ``"full_transcriptome"`` key.
        filter_args: Optional filter criteria (see :func:`filter_datasets_by_args`).

    Returns:
        Mapping of ``{dataset_name: AnnData}``.

    Raises:
        FileNotFoundError: If *preprocessed_dir* does not exist.
        ValueError: If no h5ad files are found or none survive filtering.
    """
    preprocessed_dir = Path(preprocessed_dir)
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")

    h5ad_files = glob.glob(str(preprocessed_dir / "*.h5ad"))
    if not h5ad_files:
        raise ValueError(f"No h5ad files found in {preprocessed_dir}")

    logger.info("Found %d preprocessed h5ad files in %s", len(h5ad_files), preprocessed_dir)

    if filter_args:
        logger.info("Applying dataset filters...")
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
            has_pca = "X_pca" in adata.obsm
            has_nmf = "X_nmf" in adata.obsm
            n_leiden = sum(1 for c in adata.obs.columns if c.startswith("leiden_"))
            n_knn = sum(1 for k in adata.uns if k.startswith("neighbors_"))
            logger.info(
                "Loaded %-50s  %d genes, PCA:%s  NMF:%s  leiden:%d  knn:%d",
                dataset_name, adata.n_vars,
                "yes" if has_pca else "no", "yes" if has_nmf else "no",
                n_leiden, n_knn,
            )
        except Exception as exc:
            logger.warning("Failed to load %s: %s – skipping.", h5ad_file, exc, exc_info=True)

    if not datasets or (reference_adata is not None and len(datasets) == 1):
        raise ValueError("No valid preprocessed datasets loaded!")

    logger.info("Loaded %d datasets total.", len(datasets))
    return datasets


# ---------------------------------------------------------------------------
# NMF baseline computation helpers
# ---------------------------------------------------------------------------


def _compute_full_nmf_baseline(
    A_train: np.ndarray,
    A_test: np.ndarray,
    n_components: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute the full-transcriptome NMF baseline.

    Uses the standard NMF convention: ``A ≈ W @ H``.

    Args:
        A_train: Training data (cells × genes).
        A_test: Testing data (cells × genes).
        n_components: Number of NMF components.
        random_state: Random seed.

    Returns:
        Dictionary with ``"training"`` and ``"testing"`` sub-dicts each
        containing ``W``, ``H``, reconstructed matrix, MSE, and explained
        variance.
    """
    def _fit(X: np.ndarray) -> dict[str, Any]:
        pred = NmfPredictor(
            embedding_size=n_components, seed=random_state, max_iter=1000,
            beta_loss="frobenius", init=None, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0,
        ).fit(X)
        W, H = pred.ref_embedding, pred.h_reference
        X_recon = W @ H
        return {
            "W": W, "H": H, "X_recon": X_recon,
            "mse": calculate_mse(X, X_recon),
            "expvar": calculate_explained_variance(X, X_recon),
        }

    return {"training": _fit(A_train), "testing": _fit(A_test)}


def _compute_full_nmf_baseline_swapped(
    A_train: np.ndarray,
    A_test: np.ndarray,
    n_components: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute the full-transcriptome NMF baseline (mapping convention).

    Uses the transposed convention: ``A.T ≈ H.T @ W.T`` so that H spans
    the gene dimension.

    Args:
        A_train: Training data (cells × genes).
        A_test: Testing data (cells × genes).
        n_components: Number of NMF components.
        random_state: Random seed.

    Returns:
        Dictionary with ``"training"`` and ``"testing"`` sub-dicts.
    """
    def _fit_swapped(X: np.ndarray) -> dict[str, Any]:
        # Fit on transposed matrix so ref_embedding spans gene space.
        # pred.ref_embedding  ≡ model.fit_transform(X.T)  — shape (n_genes × n_components)
        # pred.h_reference    ≡ model.components_          — shape (n_components × n_cells)
        pred = NmfPredictor(
            embedding_size=n_components, seed=random_state, max_iter=1000,
            beta_loss="frobenius", init=None, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0,
        ).fit(X.T)
        H = pred.ref_embedding.T    # model.fit_transform(X.T).T  → (n_components × n_genes)
        W = pred.h_reference.T      # model.components_.T          → (n_cells × n_components)
        X_recon = W @ H
        return {
            "W": W, "H": H, "X_recon": X_recon,
            "mse": calculate_mse(X, X_recon),
            "expvar": calculate_explained_variance(X, X_recon),
        }

    return {"training": _fit_swapped(A_train), "testing": _fit_swapped(A_test)}


# ---------------------------------------------------------------------------
# Preprocessing stage
# ---------------------------------------------------------------------------


def run_preprocessing_stage(config: EvaluationConfig) -> None:
    """Run the preprocessing stage: subset panels and compute embeddings.

    Loads the raw AnnData, discovers gene lists from the selection pipeline
    directory (and/or external panel files), and calls
    :func:`process_data_for_panel_evaluation` for each panel. Preprocessed
    h5ad files are written to ``config.preprocessed_dir``.

    Also preprocesses the full-transcriptome reference if
    ``full_transcriptome.h5ad`` does not already exist.

    Args:
        config: Pipeline configuration.
    """
    logger.info("=" * 80)
    logger.info("PREPROCESSING STAGE")
    logger.info("=" * 80)

    if not config.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {config.input_file}")

    config.preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    logger.info("Loading input data from %s", config.input_file)
    adata = sc.read_h5ad(config.input_file)
    logger.info("Loaded: %d cells × %d genes", adata.n_obs, adata.n_vars)

    # ENSEMBL ID conversion
    if adata.var_names[0].startswith(("ENSMUSG", "ENSG")):
        logger.info("Detected ENSEMBL IDs – converting to gene symbols...")
        convert_ensembl_to_gene_symbols(adata, inplace=True)

    adata.obs_names_make_unique()

    # ── Reference preprocessing ──────────────────────────────────────
    ref_out = config.preprocessed_dir / "full_transcriptome.h5ad"
    if not ref_out.exists():
        logger.info("Preprocessing full-transcriptome reference...")
        preprocess_reference_dataset(
            adata=adata,
            output_file=ref_out,
            dimensionality_reduction=config.dim_reduction_preprocess,
            n_neighbors=config.n_neighbors,
        )
    else:
        logger.info("Reference already preprocessed: %s", ref_out)

    # ── Collect gene lists ────────────────────────────────────────────
    gene_lists: dict[str, list[str]] = {}

    if config.gene_list_files_txt is not None and config.gene_list_files_txt.exists():
        csv_paths = config.gene_list_files_txt.read_text().strip().splitlines()
        logger.info("Loading %d gene lists from text file...", len(csv_paths))
        for path in csv_paths:
            path = path.strip()
            if not path:
                continue
            genes = load_gene_list_from_csv(path)
            if genes:
                from _preprocessing import extract_genelist_name_from_path
                name = extract_genelist_name_from_path(path)
                gene_lists[name] = genes
    elif config.gene_lists_dir is not None and config.gene_lists_dir.exists():
        gene_lists = load_all_gene_lists(str(config.gene_lists_dir), adata)

    # External panels
    for panel_path, panel_name, panel_size in zip(
        config.external_panels,
        config.external_names,
        config.external_probeset_sizes if config.external_probeset_sizes
        else [0] * len(config.external_panels),
    ):
        genes = load_gene_list_from_csv(panel_path)
        if not genes:
            logger.warning("No genes loaded from external panel: %s", panel_path)
            continue
        if panel_size and panel_size > 0:
            genes = genes[:panel_size]
        gene_lists[panel_name] = genes

    if not gene_lists:
        logger.warning("No gene lists found – only reference preprocessing was done.")
        return

    logger.info("Preprocessing %d gene panels...", len(gene_lists))

    # ── Process each panel ───────────────────────────────────────────
    skipped: list[str] = []
    for panel_name, genes in tqdm(gene_lists.items(), desc="Preprocessing panels"):
        out_file = config.preprocessed_dir / f"{panel_name}.h5ad"
        if out_file.exists():
            logger.info("Already exists, skipping: %s", panel_name)
            continue
        try:
            adata_panel = process_data_for_panel_evaluation(
                adata=adata,
                probeset=genes,
                n_neighbors=config.n_neighbors,
                layer="counts",
                dataset_name=panel_name,
                dimensionality_reduction=config.dim_reduction_preprocess,
                filter_genes=False,
                nmf_counts_input=config.nmf_counts_input,
            )
            adata_panel = _convert_arrow_strings_to_object(adata_panel)

            # Set anndata flag as fallback for any remaining string arrays (requires anndata >= 0.11)
            try:
                import anndata
                if hasattr(anndata.settings, 'allow_write_nullable_strings'):
                    anndata.settings.allow_write_nullable_strings = True
            except Exception:
                pass

            adata_panel.write_h5ad(out_file, compression="gzip")
            logger.info("Saved: %s (%d genes)", panel_name, adata_panel.n_vars)
            del adata_panel
            gc.collect()
        except Exception as exc:
            logger.warning("Preprocessing failed for '%s': %s. Skipping.", panel_name, exc, exc_info=True)
            skipped.append(panel_name)

    _log_memory("after preprocessing")
    logger.info("Preprocessing complete. Skipped %d panels: %s", len(skipped), skipped or "none")


# ---------------------------------------------------------------------------
# Baseline evaluation stage
# ---------------------------------------------------------------------------


def run_baseline_evaluation(
    preprocessed_datasets: dict[str, sc.AnnData],
    output_dir: str | Path,
    celltype_col: str = "cluster",
    dimensionality_reduction: str = "pca",
    external_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run baseline evaluation on preprocessed datasets.

    Computes clustering quality (ARI/NMI), neighbourhood preservation (kNN),
    and cell-type identification accuracy.

    Args:
        preprocessed_datasets: ``{dataset_name: AnnData}`` dict. Must include
            a ``"full_transcriptome"`` key as the reference.
        output_dir: Root output directory. Results go into
            ``Baseline-Evaluation/results/`` and ``Baseline-Evaluation/plots/``.
        celltype_col: obs column holding cell-type labels.
        dimensionality_reduction: Embedding to use (``"pca"``, ``"nmf"``,
            or ``"both"``).
        external_names: Optional list of external panel names for plot colouring.

    Returns:
        Dictionary with keys ``"neighborhood"``, ``"clustering"``,
        ``"celltype"`` holding result DataFrames.
    """
    logger.info("=" * 80)
    logger.info("BASELINE EVALUATION: Clustering / Neighbourhood / Cell-type")
    logger.info("=" * 80)

    output_dir = Path(output_dir)
    results_dir = output_dir / "Baseline-Evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    # ── 1. Neighbourhood preservation ────────────────────────────────
    logger.info("-" * 60)
    logger.info("1. NEIGHBOURHOOD PRESERVATION")
    logger.info("-" * 60)
    try:
        nb_results = evaluate_neighborhood_preservation(
            preprocessed_datasets,
            reference_key="full_transcriptome",
            dimensionality_reduction=dimensionality_reduction,
            celltype_col=celltype_col,
        )
        results["neighborhood"] = nb_results
        _save_results_per_dataset(nb_results, results_dir / "neighborhood", "Neighbourhood")
    except Exception as exc:
        logger.warning("Neighbourhood evaluation failed: %s", exc, exc_info=True)
    gc.collect()
    _log_memory("after neighbourhood evaluation")

    # ── 2. Clustering quality ─────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("2. CLUSTERING QUALITY")
    logger.info("-" * 60)
    try:
        cl_results = evaluate_clustering_quality(
            preprocessed_datasets,
            reference_key="full_transcriptome",
            dimensionality_reduction=dimensionality_reduction,
            celltype_col=celltype_col,
        )
        results["clustering"] = cl_results
        _save_results_per_dataset(cl_results, results_dir / "clustering", "Clustering")
    except Exception as exc:
        logger.warning("Clustering evaluation failed: %s", exc, exc_info=True)
    gc.collect()
    _log_memory("after clustering evaluation")

    # ── 3. Cell-type identification ───────────────────────────────────
    logger.info("-" * 60)
    logger.info("3. CELL-TYPE IDENTIFICATION")
    logger.info("-" * 60)
    try:
        ct_results = evaluate_celltype_identification(
            preprocessed_datasets,
            reference_key="full_transcriptome",
            celltype_col=celltype_col,
        )
        results["celltype"] = ct_results
        _save_results_per_dataset(ct_results, results_dir / "celltype", "Cell-type")
    except Exception as exc:
        logger.warning("Cell-type identification failed: %s", exc, exc_info=True)
    gc.collect()
    _log_memory("after cell-type evaluation")

    # ── 4. Rare cell type marker coverage ────────────────────────────
    logger.info("-" * 60)
    logger.info("4. RARE CELL TYPE MARKER COVERAGE")
    logger.info("-" * 60)
    try:
        adata_full = preprocessed_datasets["full_transcriptome"]
        rare_rows = []
        for dataset_name, dataset in preprocessed_datasets.items():
            if dataset_name == "full_transcriptome":
                continue
            panel_genes = list(dataset.var_names)
            coverage = compute_rare_celltype_marker_coverage(
                adata_full,
                panel_genes=panel_genes,
                celltype_col=celltype_col,
            )
            # Summary row
            rare_rows.append(
                {
                    "dataset": dataset_name,
                    "celltype": None,
                    "n_rare_celltypes": coverage["n_rare_celltypes"],
                    "panel_n_rare_ct_covered": coverage["panel_n_rare_ct_covered"],
                    "panel_fraction_rare_ct_covered": coverage["panel_fraction_rare_ct_covered"],
                }
            )
            # Per-CT rows
            for ct, ct_info in coverage["per_rare_ct"].items():
                rare_rows.append(
                    {
                        "dataset": dataset_name,
                        "celltype": ct,
                        "n_cells": ct_info["n_cells"],
                        "fraction_cells": ct_info["fraction_cells"],
                        "n_exclusive_markers_total": ct_info["n_exclusive_markers_total"],
                        "n_exclusive_markers_in_panel": ct_info["n_exclusive_markers_in_panel"],
                    }
                )
        rare_df = pd.DataFrame(rare_rows)
        results["rare_coverage"] = rare_df
        _save_results_per_dataset(rare_df, results_dir / "rare_coverage", "Rare CT Coverage")
    except Exception as exc:
        logger.warning("Rare cell type coverage evaluation failed: %s", exc, exc_info=True)
    gc.collect()
    _log_memory("after rare coverage evaluation")

    logger.info("Baseline evaluation complete.")
    return results


# ---------------------------------------------------------------------------
# Variability evaluation helpers
# ---------------------------------------------------------------------------


def _aggregate_fold_results(df_per_fold: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold results by computing mean ± std across folds.

    Args:
        df_per_fold: DataFrame with per-fold results (must have "fold" column).

    Returns:
        Aggregated DataFrame with mean and std columns.
    """
    if "fold" not in df_per_fold.columns:
        logger.warning("No 'fold' column found in results. Returning original DataFrame.")
        return df_per_fold

    # Identify grouping columns (non-metric columns)
    metric_cols = ["mse_train_baseline", "mse_test_baseline", "mse_test_probe",
                   "expvar_train_baseline", "expvar_test_baseline", "expvar_test_probe",
                   "mse_ratio", "expvar_ratio", "probeset_size", "probeset_genes_found",
                   "weighted_mse_test_probe", "weighted_mse_test_baseline",
                   "weighted_expvar_test_probe", "weighted_expvar_test_baseline",
                   "macro_mse_test_probe", "macro_mse_test_baseline",
                   "macro_expvar_test_probe", "macro_expvar_test_baseline",
                   "total_cells", "n_celltypes_processed", "n_celltypes_skipped",
                   "mse_train_probe", "expvar_train_probe"]

    group_cols = ["dataset", "gene_list", "analysis_type"]
    if "celltype" in df_per_fold.columns:
        group_cols.append("celltype")

    # Filter to only existing metric columns
    available_metrics = [col for col in metric_cols if col in df_per_fold.columns]

    if not available_metrics:
        logger.warning("No metric columns found for aggregation.")
        return df_per_fold

    # Group and aggregate
    agg_dict = {col: ["mean", "std"] for col in available_metrics}
    df_agg = df_per_fold.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

    # Flatten column names: ('mse_test_probe', 'mean') → 'mse_test_probe_mean'
    df_agg.columns = ['_'.join(col).rstrip('_') if col[1] else col[0]
                      for col in df_agg.columns.values]

    return df_agg


# ---------------------------------------------------------------------------
# Variability evaluation stage
# ---------------------------------------------------------------------------


def run_variability_evaluation(
    adata_full: sc.AnnData,
    preprocessed_datasets: dict[str, sc.AnnData],
    output_dir: str | Path,
    splits: list[tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
    n_components: int = 5,
    celltype_col: str = "cluster",
    random_state: int = 42,
    external_names: list[str] | None = None,
    nmf_counts_input: str = "raw",
) -> dict[str, Any]:
    """Run NMF-based variability evaluation using pre-generated train/test splits.

    Evaluates NMF representation (how well probe-derived biological
    factors explain the full transcriptome) across the provided folds.

    The ``splits`` argument must be generated once by :func:`generate_evaluation_splits`
    and passed identically to both this function and :func:`run_tangram_stage` so
    that NMF and Tangram operate on the same train/test partitions.

    Args:
        adata_full: Full-transcriptome AnnData reference.
        preprocessed_datasets: ``{name: AnnData}`` dict from
            :func:`_load_preprocessed_datasets`.
        output_dir: Root output directory.
        splits: List of ``(train_idx, test_idx, per_celltype_splits)`` tuples
            produced by :func:`generate_evaluation_splits`.
        n_components: Maximum number of NMF components.
        celltype_col: obs column for cell-type labels.
        random_state: Random seed.
        external_names: External panel names for plot colouring.

    Returns:
        Dictionary with ``"nmf"`` and ``"nmf_per_fold"`` DataFrames.
    """
    logger.info("=" * 80)
    logger.info("VARIABILITY EVALUATION: NMF (%d fold(s))", len(splits))
    logger.info("=" * 80)

    output_dir = Path(output_dir)
    results_dir = output_dir / "Variability-Evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Get expression matrix ────────────────────────────────────────
    if nmf_counts_input == "raw":
        if "counts" in adata_full.layers:
            if not is_anndata_raw_layer(adata_full, "counts"):
                raise ValueError(
                    "nmf_counts_input='raw': adata_full.layers['counts'] does not contain "
                    "raw integer counts. Check preprocessing."
                )
            X_full = adata_full.layers["counts"]
            logger.info("run_variability_evaluation: using layers['counts'] (verified raw counts)")
        elif hasattr(adata_full, "raw") and adata_full.raw is not None:
            if not is_anndata_raw(adata_full.raw):
                raise ValueError(
                    "nmf_counts_input='raw': adata_full.raw.X does not contain raw integer counts."
                )
            X_full = adata_full.raw.X
            logger.info("run_variability_evaluation: using adata.raw.X (verified raw counts)")
        else:
            raise ValueError(
                "nmf_counts_input='raw': no raw counts found in adata.layers['counts'] or adata.raw"
            )
    elif nmf_counts_input == "lognorm":
        if is_anndata_raw(adata_full):
            raise ValueError(
                "nmf_counts_input='lognorm': adata_full.X appears to contain raw integer counts, "
                "not log-normalized data. Normalize before evaluation."
            )
        X_full = adata_full.X
        logger.info("run_variability_evaluation: using adata.X (log-normalized, verified)")
    else:
        raise ValueError(
            f"Unknown nmf_counts_input='{nmf_counts_input}'. Choose 'raw' or 'lognorm'."
        )
    if scipy.sparse.issparse(X_full):
        X_full = X_full.toarray()
    X_full = np.asarray(X_full, dtype=np.float32)

    if celltype_col not in adata_full.obs.columns:
        logger.error("Cell-type column '%s' not found.", celltype_col)
        return {}

    panel_names = [n for n in preprocessed_datasets if n != "full_transcriptome"]
    logger.info("Evaluating variability for %d panels across %d fold(s)...", len(panel_names), len(splits))

    all_nmf_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    # ── Iterate over folds ───────────────────────────────────────────
    for fold, (train_idx, test_idx, ct_splits_idx) in enumerate(splits):
        logger.info("-" * 60)
        logger.info("FOLD %d/%d: %d training cells, %d test cells", fold + 1, len(splits), len(train_idx), len(test_idx))
        logger.info("-" * 60)

        A_train = X_full[train_idx]
        A_test = X_full[test_idx]

        # Convert index-based per-celltype splits to array-based splits
        ct_splits: dict[str, tuple[np.ndarray, np.ndarray]] = {
            ct: (X_full[tr], X_full[te])
            for ct, (tr, te) in ct_splits_idx.items()
        }

        # Initialize NMF caches per fold
        cached_full_nmf: dict[int, Any] = {}
        cached_full_nmf_swapped: dict[int, Any] = {}
        cached_ct_nmf: dict[str, dict] = {}
        cached_ct_nmf_swapped: dict[str, dict] = {}

        for panel_name in tqdm(panel_names, desc=f"Fold {fold + 1} evaluation", leave=False):
            adata_panel = preprocessed_datasets[panel_name]
            probeset_genes = adata_panel.var_names.tolist()
            panel_n = adata_panel.uns.get("n_components", n_components)

            try:
                _evaluate_single_panel_variability(
                    panel_name=panel_name,
                    adata_full=adata_full,
                    probeset_genes=probeset_genes,
                    A_train=A_train,
                    A_test=A_test,
                    ct_splits=ct_splits,
                    cached_full_nmf=cached_full_nmf,
                    cached_full_nmf_swapped=cached_full_nmf_swapped,
                    cached_ct_nmf=cached_ct_nmf,
                    cached_ct_nmf_swapped=cached_ct_nmf_swapped,
                    n_components=panel_n,
                    celltype_col=celltype_col,
                    random_state=random_state,
                    nmf_rows=all_nmf_rows,
                    per_celltype_splits=ct_splits_idx,
                    fold=fold,
                    nmf_counts_input=nmf_counts_input,
                )
            except Exception as exc:
                logger.warning("Fold %d: Evaluation failed for '%s': %s. Skipping.", fold, panel_name, exc, exc_info=True)
                if panel_name not in skipped:
                    skipped.append(panel_name)

        gc.collect()
        _log_memory(f"after fold {fold + 1}")

    # ── Save results ─────────────────────────────────────────────────
    results: dict[str, Any] = {}
    if all_nmf_rows:
        df_all = pd.DataFrame(all_nmf_rows)

        per_fold_dir = results_dir / "nmf" / "per_fold"
        per_fold_dir.mkdir(parents=True, exist_ok=True)
        _save_results_per_dataset(df_all, per_fold_dir, "NMF (per-fold)")

        df_agg = _aggregate_fold_results(df_all)

        agg_dir = results_dir / "nmf"
        _save_results_per_dataset(df_agg, agg_dir, "NMF (aggregated)")

        results["nmf"] = df_agg
        results["nmf_per_fold"] = df_all

    logger.info("Variability evaluation complete. Skipped: %s", skipped or "none")
    return results


def _evaluate_single_panel_variability(
    panel_name: str,
    adata_full: sc.AnnData,
    probeset_genes: list[str],
    A_train: np.ndarray,
    A_test: np.ndarray,
    ct_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    cached_full_nmf: dict[int, Any],
    cached_full_nmf_swapped: dict[int, Any],
    cached_ct_nmf: dict[str, dict],
    cached_ct_nmf_swapped: dict[str, dict],
    n_components: int,
    celltype_col: str,
    random_state: int,
    nmf_rows: list[dict[str, Any]],
    per_celltype_splits: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    fold: int | None = None,
    nmf_counts_input: str = "raw",
) -> None:
    """Run NMF evaluation for one panel.

    Results are appended in-place to *nmf_rows*.

    Args:
        panel_name: Dataset identifier.
        adata_full: Full-transcriptome AnnData.
        probeset_genes: Genes in this panel.
        A_train: Global train data (cells × full genes).
        A_test: Global test data (cells × full genes).
        ct_splits: Per-cell-type ``{celltype: (A_train, A_test)}``.
        cached_full_nmf: Cache for global NMF baselines (mutated in place).
        cached_full_nmf_swapped: Cache for swapped global NMF baselines.
        cached_ct_nmf: Cache for per-cell-type NMF baselines.
        cached_ct_nmf_swapped: Cache for per-cell-type swapped NMF baselines.
        n_components: NMF components for this panel.
        celltype_col: obs column for cell-type labels.
        random_state: Random seed.
        nmf_rows: NMF result rows output list (mutated in place).
        per_celltype_splits: Per-cell-type index splits.
        fold: Fold number (optional, for k-fold evaluation).
        nmf_counts_input: Which count matrix to use (``"raw"`` or ``"lognorm"``).
    """
    # Lazily compute global NMF baseline
    if n_components not in cached_full_nmf:
        logger.info("Computing global NMF baseline (n_components=%d)...", n_components)
        cached_full_nmf[n_components] = _compute_full_nmf_baseline(
            A_train, A_test, n_components, random_state
        )
        cached_full_nmf_swapped[n_components] = _compute_full_nmf_baseline_swapped(
            A_train, A_test, n_components, random_state
        )

    baseline = cached_full_nmf[n_components]
    baseline_sw = cached_full_nmf_swapped[n_components]

    # ── Translate evaluation-baseline format → variability-cache format ───────
    # _compute_full_nmf_baseline stores keys ("W", "H", "X_recon", "mse", "expvar")
    # but nmf_reconstruction expects ("A_train", "W_full_train",
    # "H_full_train", "A_train_baseline", "mse_train_baseline", …).
    # Build the translated dict so the variability function can reuse the NMF.
    _b_train = baseline["training"]
    _b_test  = baseline["testing"]
    variability_cache = {
        "training": {
            "A_train":              A_train,
            "W_full_train":         _b_train["W"],
            "H_full_train":         _b_train["H"],
            "A_train_baseline":     _b_train["X_recon"],
            "mse_train_baseline":   _b_train["mse"],
            "expvar_train_baseline":_b_train["expvar"],
        },
        "testing": {
            "A_test":               A_test,
            "W_full_test":          _b_test["W"],
            "H_full_test":          _b_test["H"],
            "A_test_baseline":      _b_test["X_recon"],
            "mse_test_baseline":    _b_test["mse"],
            "expvar_test_baseline": _b_test["expvar"],
        },
    }

    # ── Global NMF ───────────────────────────────────────────────────
    logger.info("  [%s] Global NMF (n=%d)...", panel_name, n_components)
    nmf_global = nmf_reconstruction(
        adata=adata_full,
        probeset_genes=probeset_genes,
        A_train=A_train,
        A_test=A_test,
        n_components=n_components,
        cached_full_nmf=variability_cache,
    )
    if nmf_global:
        nmf_global["dataset"] = panel_name
        nmf_global["gene_list"] = panel_name
        nmf_global["analysis_type"] = "global"
        if fold is not None:
            nmf_global["fold"] = fold
        nmf_rows.append(nmf_global)

    # ── Per-cell-type NMF ─────────────────────────────────────────────
    if celltype_col in adata_full.obs.columns:
        logger.info("  [%s] Per-cell-type NMF...", panel_name)
        mech_ct = nmf_reconstruction_by_celltype(
            adata=adata_full,
            probeset_genes=probeset_genes,
            celltype_column=celltype_col,
            n_components=n_components,
            cached_full_nmf_by_celltype=cached_ct_nmf,
            per_celltype_splits=per_celltype_splits,
            nmf_counts_input=nmf_counts_input,
        )
        for ct, res in (mech_ct or {}).items():
            row = {**res, "dataset": panel_name, "gene_list": panel_name, "celltype": ct, "analysis_type": "per_celltype"}
            if fold is not None:
                row["fold"] = fold
            nmf_rows.append(row)

# ---------------------------------------------------------------------------
# Feature plots & dot plot generation
# ---------------------------------------------------------------------------


def generate_panel_plots(
    adata_full: sc.AnnData,
    preprocessed_datasets: dict[str, sc.AnnData],
    celltype_col: str,
    output_dir: Path,
) -> None:
    """Generate and save UMAP featureplots and a dot plot for each panel.

    Plots are saved under *output_dir* / "Feature-Plots" / *panel_name* /.

    Args:
        adata_full: Full-transcriptome AnnData (used for UMAP embedding).
        preprocessed_datasets: Dict of panel name → panel AnnData.
        celltype_col: obs column for cell-type labels.
        output_dir: Root evaluation output directory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("=" * 80)
    logger.info("GENERATING FEATURE PLOTS AND DOT PLOTS")
    logger.info("=" * 80)

    # --- Compute UMAP on the full transcriptome once ---
    adata = adata_full.copy()
    try:
        if "X_pca" not in adata.obsm:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
            sc.pp.pca(adata, n_comps=30)
            sc.pp.neighbors(adata)
        if "X_umap" not in adata.obsm:
            sc.tl.umap(adata)
    except Exception as exc:
        logger.warning("Could not compute UMAP embedding: %s", exc, exc_info=True)
        return

    panel_names = [n for n in preprocessed_datasets if n != "full_transcriptome"]

    for panel_name in panel_names:
        panel_adata = preprocessed_datasets[panel_name]
        panel_genes = panel_adata.var_names.tolist()
        panel_in_adata = [g for g in panel_genes if g in adata.var_names]

        plots_dir = output_dir / "Feature-Plots" / panel_name
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving feature plots for panel '%s' (%d genes)...", panel_name, len(panel_in_adata))

        # --- UMAP coloured by cell type (compact, high-res) ---
        if celltype_col in adata.obs.columns:
            try:
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sc.pl.umap(adata, color=celltype_col, ax=ax, show=False, frameon=False, size=6)
                fig.tight_layout()
                fig.savefig(plots_dir / "umap_celltypes.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                logger.warning("Could not save celltype UMAP for %s: %s", panel_name, exc)

            # --- Cell-type count bar chart ---
            try:
                ct_counts = adata.obs[celltype_col].value_counts().sort_values(ascending=True)
                n_ct = len(ct_counts)
                # Scale height tightly per cell type; cap width so it fits next to UMAP
                fig_h = max(1.8, n_ct * 0.22)
                fig2, ax2 = plt.subplots(figsize=(2.8, fig_h))
                colors = plt.cm.tab20.colors  # up to 20 distinct colours
                bar_colors = [colors[i % len(colors)] for i in range(n_ct)]
                ax2.barh(ct_counts.index.tolist(), ct_counts.values.tolist(),
                         color=bar_colors, edgecolor="white", linewidth=0.5)
                # Add count labels on each bar
                x_max = ct_counts.values.max()
                for i, v in enumerate(ct_counts.values):
                    ax2.text(v + x_max * 0.02, i, f"{v:,}", va="center", fontsize=6)
                ax2.set_xlim(0, x_max * 1.30)
                ax2.set_xlabel("Cell count", fontsize=8)
                ax2.set_title("Cells per cell type", fontsize=9)
                ax2.tick_params(axis="y", labelsize=7)
                ax2.tick_params(axis="x", labelsize=7)
                for spine in ["top", "right"]:
                    ax2.spines[spine].set_visible(False)
                fig2.tight_layout()
                # 120 DPI keeps the image compact when rendered in the browser column
                fig2.savefig(plots_dir / "celltype_counts.png", dpi=120, bbox_inches="tight")
                plt.close(fig2)
            except Exception as exc:
                logger.warning("Could not save celltype count bar chart for %s: %s", panel_name, exc)

        # --- Gene featureplots: individual per-gene PNGs + batch PNGs of 4 ---
        gene_index: dict[str, str] = {}  # gene → filename mapping saved as JSON
        batch_size = 4
        for i in range(0, len(panel_in_adata), batch_size):
            batch = panel_in_adata[i : i + batch_size]
            batch_fname = f"featureplot_{i // batch_size + 1:03d}.png"
            try:
                fig, axes = plt.subplots(1, len(batch), figsize=(4 * len(batch), 3.5))
                if len(batch) == 1:
                    axes = [axes]
                for ax, gene in zip(axes, batch):
                    sc.pl.umap(adata, color=gene, ax=ax, show=False, frameon=False, title=gene, size=6)
                fig.tight_layout()
                fig.savefig(plots_dir / batch_fname, dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                logger.warning("Could not save featureplot batch %d for %s: %s", i // batch_size + 1, panel_name, exc)

            # --- Individual per-gene featureplots ---
            for gene in batch:
                gene_fname = f"gene_{gene}.png"
                try:
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    sc.pl.umap(adata, color=gene, ax=ax, show=False, frameon=False, title=gene, size=6)
                    fig.tight_layout()
                    fig.savefig(plots_dir / gene_fname, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    gene_index[gene] = gene_fname
                except Exception as exc:
                    logger.warning("Could not save individual featureplot for gene %s in %s: %s", gene, panel_name, exc)

        # Save gene index JSON so the app can build the gene selector
        try:
            import json as _json
            with open(plots_dir / "gene_index.json", "w") as fh:
                _json.dump(gene_index, fh)
        except Exception as exc:
            logger.warning("Could not save gene_index.json for %s: %s", panel_name, exc)

        # --- Dot plot ---
        # For combination strategies we use the custom coloured dotplot so
        # genes are colour-coded by their selection origin (rf_deg = red,
        # dimred = blue).  Gene-source info is read preferentially from
        # ranked_gene_list.csv (most reliable source of truth); if that file
        # is absent we fall back to panel_adata.var["gene_source"].
        if panel_in_adata and celltype_col in adata.obs.columns:
            genes_to_show = panel_in_adata[:60]
            n_groups = adata.obs[celltype_col].nunique()

            # ── 1. Collect gene_source information ──────────────────────
            gene_sources: dict[str, str] | None = None

            # ranked_gene_list.csv lives one level above the evaluation dir
            # (i.e., at the run root) — NOT inside gene_lists/.
            ranked_csv = output_dir.parent / "ranked_gene_list.csv"
            if ranked_csv.exists():
                try:
                    gl_df = pd.read_csv(ranked_csv, index_col=0)
                    if "gene_source" in gl_df.columns:
                        _src_map = {
                            str(g): str(s)
                            for g, s in zip(gl_df.index, gl_df["gene_source"].fillna("other"))
                        }
                        # Only activate combination mode when rf_deg or dimred
                        # sources are actually present in the panel
                        _known = {"rf_deg", "dimred"}
                        if any(v in _known for v in _src_map.values()):
                            # Restrict to genes that will actually be plotted
                            gene_sources = {
                                g: _src_map.get(g, "other") for g in genes_to_show
                            }
                            logger.debug(
                                "  Read gene_source from %s for panel '%s'",
                                ranked_csv, panel_name,
                            )
                except Exception as exc:
                    logger.debug(
                        "Could not read gene_source from %s: %s", ranked_csv, exc
                    )

            # Fallback: panel_adata.var (may exist if preprocessing preserved it)
            if gene_sources is None:
                if (
                    "gene_source" in panel_adata.var.columns
                    and panel_adata.var["gene_source"].notna().any()
                ):
                    _src_series = panel_adata.var["gene_source"].fillna("other").astype(str)
                    _src_map = dict(zip(panel_adata.var_names, _src_series))
                    _known = {"rf_deg", "dimred"}
                    if any(v in _known for v in _src_map.values()):
                        gene_sources = {g: _src_map.get(g, "other") for g in genes_to_show}

            has_gene_source = gene_sources is not None

            # ── 2. Detect dimred type for legend label (PCA vs NMF) ─────
            _pname_lower = panel_name.lower()
            if "_nmf_" in _pname_lower or _pname_lower.endswith("_nmf"):
                _dimred_label = "NMF"
            else:
                _dimred_label = "PCA"

            if has_gene_source:
                # Combination strategy → coloured dotplot
                try:
                    _plot_dir = Path(__file__).parent.parent / "Plotting-module"
                    if str(_plot_dir) not in sys.path:
                        sys.path.insert(0, str(_plot_dir))

                    # Guard against _constants collision (same technique as
                    # generate_baseline_evaluation_plots)
                    _stash = {k: sys.modules.pop(k) for k in ["_constants"]
                              if k in sys.modules}
                    try:
                        from _combination_dotplot import plot_combination_dotplot
                    finally:
                        for k, v in _stash.items():
                            sys.modules[k] = v
                        sys.modules.pop("_constants", None)
                        for k, v in _stash.items():
                            sys.modules[k] = v

                    plot_combination_dotplot(
                        adata,
                        var_names=genes_to_show,
                        groupby=celltype_col,
                        gene_sources=gene_sources,
                        output_path=plots_dir / "dotplot.png",
                        title=f"{panel_name} — {len(panel_in_adata)} genes × cell types",
                        source_label_overrides={
                            "rf_deg": "DEG",
                            "dimred": _dimred_label,
                        },
                        dpi=300,
                    )
                    plt.close("all")
                    logger.info("  Saved combination dotplot for '%s'", panel_name)
                except Exception as exc:
                    logger.warning("Could not save combination dotplot for %s: %s", panel_name, exc)
                    has_gene_source = False  # fall through to standard dotplot

            if not has_gene_source:
                # Standard strategy → scanpy DotPlot
                try:
                    width = max(14, len(genes_to_show) * 0.4)
                    dp = sc.pl.DotPlot(
                        adata,
                        var_names=genes_to_show,
                        groupby=celltype_col,
                        figsize=(width, max(5, n_groups * 0.4)),
                        title=f"Panel genes ({len(panel_in_adata)}) × cell types",
                    )
                    dp.savefig(str(plots_dir / "dotplot.png"), dpi=300, bbox_inches="tight")
                    plt.close("all")
                except Exception as exc:
                    logger.warning("Could not save dotplot for %s: %s", panel_name, exc)

        logger.info("  Saved plots for '%s' to %s", panel_name, plots_dir)


# ---------------------------------------------------------------------------
# Plotting module integration — generate metric plots from saved CSVs
# ---------------------------------------------------------------------------


def generate_baseline_evaluation_plots(output_dir: Path) -> None:
    """Call the Plotting-module functions to create metric PNGs from saved CSVs.

    Reads the CSVs produced by :func:`run_baseline_evaluation` and calls the
    ``_clustering_plots`` functions to produce publication-quality plots.  The
    resulting PNGs are saved next to the CSVs inside each result sub-directory
    so the app can find and display them.

    Key detail: the Plotting-module has its own ``_constants.py`` with column
    name constants (``COL_DATASET``, ``COL_ARI``, …) that differ from the
    Evaluation-module's ``_constants.py``.  When Python has already loaded the
    Evaluation-module's ``_constants`` it will be cached in ``sys.modules`` and
    shadow the Plotting-module's version.  We therefore temporarily pop all
    Evaluation-module flat-module entries from ``sys.modules``, load the
    Plotting-module functions, then restore the originals.
    """
    import pandas as pd
    import traceback

    # Resolve the Plotting-module directory (sibling of Evaluation-module)
    _eval_dir_abs = Path(__file__).parent.absolute()
    _plot_dir = _eval_dir_abs.parent / "Plotting-module"
    if not _plot_dir.exists():
        logger.warning("Plotting-module not found at %s; skipping metric plots", _plot_dir)
        return

    logger.info("Generating baseline evaluation plots from saved CSVs...")
    logger.info("  Plotting-module path: %s", _plot_dir)

    import importlib.util as _ilu
    import matplotlib
    matplotlib.use("Agg")

    # ------------------------------------------------------------------ #
    # Load _clustering_plots via importlib.util.spec_from_file_location   #
    # (exact file path) to completely bypass sys.path ambiguity.          #
    #                                                                      #
    # The Evaluation-module, Utility-module, and Plotting-module each     #
    # have their own _constants.py.  Because Utility-module is added to   #
    # sys.path[0] at startup, a plain `import _constants` would find the  #
    # wrong version.  Loading by exact path sidesteps that entirely.      #
    # ------------------------------------------------------------------ #
    _constants_file = _plot_dir / "_constants.py"
    _cplots_file    = _plot_dir / "_clustering_plots.py"

    if not _constants_file.exists() or not _cplots_file.exists():
        logger.error(
            "Plotting-module files not found: %s or %s — skipping metric plots",
            _constants_file, _cplots_file,
        )
        return

    try:
        # 1. Load Plotting-module's _constants as an isolated module object.
        _pc_spec = _ilu.spec_from_file_location("_plot_module_constants", _constants_file)
        _pc_mod  = _ilu.module_from_spec(_pc_spec)
        _pc_spec.loader.exec_module(_pc_mod)

        # 2. Temporarily register it under the bare name "_constants" so that
        #    _clustering_plots.py's top-level `from _constants import …` finds
        #    the correct version via sys.modules (no sys.path search needed).
        _old_constants = sys.modules.pop("_constants", None)
        sys.modules["_constants"] = _pc_mod
        try:
            # 3. Load _clustering_plots from its exact file path.
            _cp_spec = _ilu.spec_from_file_location("_plot_clustering_plots", _cplots_file)
            _cp_mod  = _ilu.module_from_spec(_cp_spec)
            # Register before exec so any internal self-reference resolves:
            sys.modules["_clustering_plots"] = _cp_mod
            _cp_spec.loader.exec_module(_cp_mod)
        finally:
            # 4. Restore sys.modules to original state regardless of success.
            sys.modules.pop("_constants", None)
            sys.modules.pop("_clustering_plots", None)
            if _old_constants is not None:
                sys.modules["_constants"] = _old_constants

        # 5. Extract functions from the loaded module (they reference the
        #    Plotting-module's _constants via _cp_mod.__globals__ — still valid
        #    even after the module is removed from sys.modules).
        plot_clustering_quality_ari            = _cp_mod.plot_clustering_quality_ari
        plot_clustering_quality_nmi            = _cp_mod.plot_clustering_quality_nmi
        plot_neighborhood_preservation_by_k    = _cp_mod.plot_neighborhood_preservation_by_k
        plot_optimal_neighborhood_preservation = _cp_mod.plot_optimal_neighborhood_preservation
        plot_celltype_accuracy_barchart        = _cp_mod.plot_celltype_accuracy_barchart
        plot_celltype_f1_heatmap               = _cp_mod.plot_celltype_f1_heatmap
        logger.info("  Successfully loaded Plotting-module functions via importlib")

    except Exception as exc:
        logger.error("Could not load clustering plot functions: %s\n%s", exc, traceback.format_exc())
        return

    baseline_dir = output_dir / "Baseline-Evaluation" / "results"
    if not baseline_dir.exists():
        logger.info("No Baseline-Evaluation results found; skipping metric plots")
        return

    # --- Clustering ---
    clustering_csv_dir = baseline_dir / "clustering"
    if clustering_csv_dir.exists():
        dfs = []
        for f in clustering_csv_dir.glob("*.csv"):
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                pass
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            logger.info("  Clustering df shape: %s, columns: %s", df.shape, list(df.columns))
            # Restrict to PCA representation only
            if "representation" in df.columns:
                df = df[df["representation"].str.lower() == "pca"].copy()
                logger.info("  Clustering df after PCA filter: %s rows", len(df))
            try:
                plot_clustering_quality_ari(df, str(clustering_csv_dir))
                plot_clustering_quality_nmi(df, str(clustering_csv_dir))
                logger.info("  Clustering plots saved to %s", clustering_csv_dir)
            except Exception as exc:
                logger.warning("Could not create clustering plots: %s\n%s", exc, traceback.format_exc())

    # --- Neighborhood ---
    neighborhood_csv_dir = baseline_dir / "neighborhood"
    if neighborhood_csv_dir.exists():
        dfs = []
        for f in neighborhood_csv_dir.glob("*.csv"):
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                pass
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            logger.info("  Neighborhood df shape: %s, columns: %s", df.shape, list(df.columns))
            try:
                plot_neighborhood_preservation_by_k(df, str(neighborhood_csv_dir))
                # plot_optimal_neighborhood_preservation skipped (not needed)
                logger.info("  Neighborhood plots saved to %s", neighborhood_csv_dir)
            except Exception as exc:
                logger.warning("Could not create neighborhood plots: %s\n%s", exc, traceback.format_exc())

    # --- Celltype classification ---
    celltype_csv_dir = baseline_dir / "celltype"
    if celltype_csv_dir.exists():
        dfs = []
        for f in celltype_csv_dir.glob("*.csv"):
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                pass
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            logger.info("  Celltype df shape: %s, columns: %s", df.shape, list(df.columns))
            try:
                plot_celltype_accuracy_barchart(df, str(celltype_csv_dir))
                plot_celltype_f1_heatmap(df, str(celltype_csv_dir))
                logger.info("  Celltype plots saved to %s", celltype_csv_dir)
            except Exception as exc:
                logger.warning("Could not create celltype plots: %s\n%s", exc, traceback.format_exc())

    logger.info("Baseline evaluation plots complete.")


# ---------------------------------------------------------------------------
# Plotting module integration — variability (NMF) evaluation plots
# ---------------------------------------------------------------------------


def generate_variability_evaluation_plots(output_dir: Path) -> None:
    """Generate variability metric plots using the Plotting-module's _variability_plots.py.

    Reads NMF CSVs from
    ``output_dir/Variability-Evaluation/results/nmf/``, reconstructs
    the dict structure expected by the Plotting-module, then calls
    ``plot_aggregated_celltype_metrics`` and ``plot_celltype_evaluation_results``
    to produce publication-quality PNGs.

    The same ``importlib.util.spec_from_file_location`` technique used in
    :func:`generate_baseline_evaluation_plots` is applied here to avoid the
    three-way ``_constants.py`` / ``_clustering_plots.py`` collision.
    """
    import ast
    import re as _re_var
    import traceback
    import importlib.util as _ilu
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    # ------------------------------------------------------------------ #
    # Helper: parse dict strings that contain np.float32(…) calls        #
    # ------------------------------------------------------------------ #
    def _parse_ct_dict(raw: str) -> dict:
        cleaned = _re_var.sub(r"np\.\w+\(([^)]+)\)", r"\1", str(raw))
        return ast.literal_eval(cleaned)

    def _pick_metric_value(row: pd.Series, base_col: str) -> float:
        """Read metric value from either legacy unsuffixed or aggregated *_mean schema."""
        val = row.get(base_col, np.nan)
        if pd.notna(val):
            return float(val)
        return float(row.get(f"{base_col}_mean", np.nan))

    def _pick_metric_std(row: pd.Series, base_col: str) -> float:
        """Read metric std from aggregated *_std schema if present."""
        std = row.get(f"{base_col}_std", np.nan)
        if pd.isna(std):
            return float("nan")
        return float(std)

    # ------------------------------------------------------------------ #
    # Locate NMF CSVs                                                      #
    # ------------------------------------------------------------------ #
    variability_dir = output_dir / "Variability-Evaluation" / "results" / "nmf"
    if not variability_dir.exists():
        logger.info("No Variability-Evaluation NMF results found; skipping plots")
        return

    csv_files = sorted(variability_dir.glob("*.csv"))
    if not csv_files:
        logger.info("No NMF CSVs found in %s; skipping plots", variability_dir)
        return

    logger.info("Generating variability evaluation plots from %d CSV(s)...", len(csv_files))

    # ------------------------------------------------------------------ #
    # Load Plotting-module functions via importlib (bypass _constants     #
    # collision — identical technique to generate_baseline_evaluation_plots)
    # ------------------------------------------------------------------ #
    _eval_dir_abs = Path(__file__).parent.absolute()
    _plot_dir     = _eval_dir_abs.parent / "Plotting-module"

    if not _plot_dir.exists():
        logger.warning("Plotting-module not found at %s; skipping variability plots", _plot_dir)
        return

    _constants_file    = _plot_dir / "_constants.py"
    _cplots_file       = _plot_dir / "_clustering_plots.py"
    _vplots_file       = _plot_dir / "_variability_plots.py"

    for _f in (_constants_file, _cplots_file, _vplots_file):
        if not _f.exists():
            logger.error("Plotting-module file missing: %s — skipping variability plots", _f)
            return

    try:
        # 1. Load _constants from its exact path
        _pc_spec = _ilu.spec_from_file_location("_plot_module_constants", _constants_file)
        _pc_mod  = _ilu.module_from_spec(_pc_spec)
        _pc_spec.loader.exec_module(_pc_mod)

        # 2. Temporarily register so downstream imports resolve correctly
        _old_constants = sys.modules.pop("_constants", None)
        sys.modules["_constants"] = _pc_mod
        try:
            # 3. Load _clustering_plots (needed by _variability_plots)
            _cp_spec = _ilu.spec_from_file_location("_plot_clustering_plots", _cplots_file)
            _cp_mod  = _ilu.module_from_spec(_cp_spec)
            sys.modules["_clustering_plots"] = _cp_mod
            _cp_spec.loader.exec_module(_cp_mod)

            # 4. Load _variability_plots
            _vp_spec = _ilu.spec_from_file_location("_plot_variability_plots", _vplots_file)
            _vp_mod  = _ilu.module_from_spec(_vp_spec)
            sys.modules["_variability_plots"] = _vp_mod
            _vp_spec.loader.exec_module(_vp_mod)

        finally:
            sys.modules.pop("_constants",         None)
            sys.modules.pop("_clustering_plots",   None)
            sys.modules.pop("_variability_plots",  None)
            if _old_constants is not None:
                sys.modules["_constants"] = _old_constants

        plot_aggregated_celltype_metrics  = _vp_mod.plot_aggregated_celltype_metrics
        plot_celltype_evaluation_results  = _vp_mod.plot_celltype_evaluation_results
        logger.info("  Loaded _variability_plots functions via importlib")

    except Exception as exc:
        logger.error("Could not load _variability_plots: %s\n%s", exc, traceback.format_exc())
        return

    # ------------------------------------------------------------------ #
    # Columns that are NOT cell-type names in the NMF CSV                 #
    # ------------------------------------------------------------------ #
    _META_COLS = {
        "mse_train_baseline", "mse_test_baseline", "mse_test_probe",
        "expvar_train_baseline", "expvar_test_baseline", "expvar_test_probe",
        "mse_ratio", "expvar_ratio", "probeset_size", "probeset_genes_found",
        "dataset", "gene_list", "analysis_type", "celltype",
        "weighted_mse_test_probe", "weighted_mse_test_baseline",
        "weighted_expvar_test_probe", "weighted_expvar_test_baseline",
        "macro_mse_test_probe", "macro_mse_test_baseline",
        "macro_expvar_test_probe", "macro_expvar_test_baseline",
        "total_cells", "n_celltypes_processed", "n_celltypes_skipped",
        "mse_train_probe", "expvar_train_probe",
    }

    # ------------------------------------------------------------------ #
    # Process each CSV                                                     #
    # ------------------------------------------------------------------ #
    for csv_path in csv_files:
        probeset_name = csv_path.stem   # e.g. "Xenium-Filter_All-Genes_rf_nmf_100"
        try:
            df = pd.read_csv(csv_path)

            global_row  = df[df["analysis_type"] == "global"]
            summary_row = df[(df["analysis_type"] == "per_celltype") & (df["celltype"] == "summary")]
            ct_row      = df[(df["analysis_type"] == "per_celltype") & (df["celltype"] == "celltype_results")]

            # ---------------------------------------------------------- #
            # Build the evaluation_results dict expected by _variability_ #
            # plots.py                                                     #
            #                                                              #
            # {probeset_name: {                                            #
            #     "nmf_celltype_summary": {...},                           #
            #     "nmf_celltype_celltype_results": {ct: {...},...}         #
            # }}                                                           #
            # ---------------------------------------------------------- #
            probeset_result: dict = {}

            # --- Summary block ---
            if not summary_row.empty:
                sr = summary_row.iloc[0]
                probeset_result["nmf_celltype_summary"] = {
                    "weighted_mse_test_probe": _pick_metric_value(sr, "weighted_mse_test_probe"),
                    "macro_mse_test_probe": _pick_metric_value(sr, "macro_mse_test_probe"),
                    "weighted_expvar_test_probe": _pick_metric_value(sr, "weighted_expvar_test_probe"),
                    "macro_expvar_test_probe": _pick_metric_value(sr, "macro_expvar_test_probe"),
                    "weighted_mse_test_baseline": _pick_metric_value(sr, "weighted_mse_test_baseline"),
                    "macro_mse_test_baseline": _pick_metric_value(sr, "macro_mse_test_baseline"),
                    "weighted_expvar_test_baseline": _pick_metric_value(sr, "weighted_expvar_test_baseline"),
                    "macro_expvar_test_baseline": _pick_metric_value(sr, "macro_expvar_test_baseline"),
                    # Propagate fold-level variability when available in aggregated CSVs.
                    "weighted_mse_test_probe_std": _pick_metric_std(sr, "weighted_mse_test_probe"),
                    "macro_mse_test_probe_std": _pick_metric_std(sr, "macro_mse_test_probe"),
                    "weighted_expvar_test_probe_std": _pick_metric_std(sr, "weighted_expvar_test_probe"),
                    "macro_expvar_test_probe_std": _pick_metric_std(sr, "macro_expvar_test_probe"),
                }

            # --- Per-celltype block ---
            if not ct_row.empty:
                ct_data_row = ct_row.iloc[0]
                ct_cols = [c for c in df.columns if c not in _META_COLS]
                ct_results: dict = {}
                for col in ct_cols:
                    raw = ct_data_row.get(col, None)
                    if pd.isna(raw) or raw is None:
                        continue
                    try:
                        d = _parse_ct_dict(raw)
                        if isinstance(d, dict):
                            # Normalise key names to match constants used in _variability_plots.py
                            ct_results[col] = {
                                "mse_test_probe":    float(d.get("mse_test_probe",    0)),
                                "expvar_test_probe": float(d.get("expvar_test_probe", 0)),
                                "mse_test_baseline": float(d.get("mse_test_baseline", 0)),
                                "expvar_test_baseline": float(d.get("expvar_test_baseline", 0)),
                                "n_cells":           int(d.get("n_cells", 0)),
                                "skipped":           bool(d.get("skipped", False)),
                            }
                    except Exception as parse_exc:
                        logger.debug("  Could not parse celltype '%s': %s", col, parse_exc)
                if ct_results:
                    probeset_result["nmf_celltype_celltype_results"] = ct_results

            if not probeset_result:
                logger.warning("  No usable data in %s — skipping", csv_path.name)
                continue

            evaluation_results = {probeset_name: probeset_result}
            plots_dir = str(variability_dir)

            # ---------------------------------------------------------- #
            # Call Plotting-module functions                               #
            # ---------------------------------------------------------- #
            logger.info("  Calling plot_aggregated_celltype_metrics for '%s'", probeset_name)
            try:
                plot_aggregated_celltype_metrics(
                    evaluation_results,
                    plots_dir,
                    title_suffix=f" — {probeset_name}",
                )
            except Exception as exc:
                logger.warning("  plot_aggregated_celltype_metrics failed: %s\n%s",
                               exc, traceback.format_exc())

            logger.info("  Calling plot_celltype_evaluation_results for '%s'", probeset_name)
            try:
                plot_celltype_evaluation_results(
                    evaluation_results,
                    plots_dir,
                    title_suffix=f" — {probeset_name}",
                )
            except Exception as exc:
                logger.warning("  plot_celltype_evaluation_results failed: %s\n%s",
                               exc, traceback.format_exc())

        except Exception as exc:
            logger.warning("Could not generate variability plots for %s: %s\n%s",
                           csv_path.name, exc, traceback.format_exc())

    logger.info("Variability evaluation plots complete.")


def generate_tangram_vs_nmf_fold_aggregate_plots(output_dir: Path) -> None:
    """Create Tangram vs NMF aggregate comparison plots per panel.

    NMF values are aggregated from per-fold summary rows (mean ± std).
    Tangram currently runs once with shared splits and therefore contributes
    a single value (no fold std) for each metric.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    tangram_ct_dir = output_dir / "Tangram-Evaluation" / "per_celltype"
    nmf_per_fold_dir = output_dir / "Variability-Evaluation" / "results" / "nmf" / "per_fold"
    out_dir = output_dir / "Tangram-Evaluation" / "plots"

    if not tangram_ct_dir.exists() or not nmf_per_fold_dir.exists():
        logger.info("Tangram/NMF comparison inputs missing; skipping Tangram-vs-NMF aggregate plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("macro_mse", "Macro MSE", True),
        ("weighted_mse", "Weighted MSE", True),
        ("macro_expvar", "Macro ExpVar", False),
        ("weighted_expvar", "Weighted ExpVar", False),
    ]

    created = 0
    for tangram_csv in sorted(tangram_ct_dir.glob("*.csv")):
        panel = tangram_csv.stem
        nmf_csv = nmf_per_fold_dir / f"{panel}.csv"
        if not nmf_csv.exists():
            logger.debug("No matching NMF per-fold CSV for Tangram panel %s", panel)
            continue

        try:
            tg_df = pd.read_csv(tangram_csv)
            nmf_df = pd.read_csv(nmf_csv)
        except Exception as exc:
            logger.warning("Could not read Tangram/NMF CSV for %s: %s", panel, exc)
            continue

        tg_summary = tg_df[tg_df.get("celltype") == "__summary__"]
        nmf_summary_rows = nmf_df[
            (nmf_df.get("analysis_type") == "per_celltype")
            & (nmf_df.get("celltype") == "summary")
        ]

        if tg_summary.empty or nmf_summary_rows.empty:
            logger.debug("Missing summary rows for %s (tangram=%s, nmf=%s)", panel, not tg_summary.empty, not nmf_summary_rows.empty)
            continue

        tg_row = tg_summary.iloc[0]
        nmf_mean = {
            "macro_mse": float(nmf_summary_rows["macro_mse_test_probe"].mean()),
            "weighted_mse": float(nmf_summary_rows["weighted_mse_test_probe"].mean()),
            "macro_expvar": float(nmf_summary_rows["macro_expvar_test_probe"].mean()),
            "weighted_expvar": float(nmf_summary_rows["weighted_expvar_test_probe"].mean()),
        }
        nmf_std = {
            "macro_mse": float(nmf_summary_rows["macro_mse_test_probe"].std(ddof=1)),
            "weighted_mse": float(nmf_summary_rows["weighted_mse_test_probe"].std(ddof=1)),
            "macro_expvar": float(nmf_summary_rows["macro_expvar_test_probe"].std(ddof=1)),
            "weighted_expvar": float(nmf_summary_rows["weighted_expvar_test_probe"].std(ddof=1)),
        }
        tg_vals = {
            "macro_mse": float(tg_row.get("macro_mse", np.nan)),
            "weighted_mse": float(tg_row.get("weighted_mse", np.nan)),
            "macro_expvar": float(tg_row.get("macro_expvar", np.nan)),
            "weighted_expvar": float(tg_row.get("weighted_expvar", np.nan)),
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, mse_panel in zip(axes, [True, False]):
            specs = [s for s in metric_specs if s[2] == mse_panel]
            labels = [s[1] for s in specs]
            keys = [s[0] for s in specs]
            x = np.arange(len(labels))
            width = 0.38

            nmf_y = [nmf_mean[k] for k in keys]
            nmf_err = [0.0 if np.isnan(nmf_std[k]) else nmf_std[k] for k in keys]
            tg_y = [tg_vals[k] for k in keys]

            ax.bar(
                x - width / 2,
                nmf_y,
                width,
                yerr=nmf_err,
                capsize=4,
                color="#1f77b4",
                alpha=0.88,
                label="NMF (fold mean ± std)",
            )
            ax.bar(
                x + width / 2,
                tg_y,
                width,
                color="#ff9800",
                alpha=0.88,
                label="Tangram (single run)",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.grid(axis="y", alpha=0.25, linestyle="--")
            ax.set_ylabel("Score")
            ax.set_title("MSE" if mse_panel else "Explained Variance")
            ax.legend(fontsize=14, loc="best")

        fig.suptitle(f"Tangram vs NMF Aggregate Metrics ({panel})")
        fig.tight_layout()
        out = out_dir / f"{panel}_tangram_vs_nmf_fold_aggregate.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        created += 1

    logger.info("Created %d Tangram-vs-NMF aggregate comparison plot(s)", created)


# ---------------------------------------------------------------------------
# Tangram stage
# ---------------------------------------------------------------------------


def _aggregate_tangram_fold_results(
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Average scalar Tangram metrics across folds; keep arrays from fold 0.

    Args:
        fold_results: List of per-fold result dicts from
            :func:`run_tangram_reconstruction_check`.

    Returns:
        Aggregated result dict with ``_mean`` and ``_std`` suffixes for scalars
        and ``n_folds`` added.
    """
    if not fold_results:
        return {}
    if len(fold_results) == 1:
        return {**fold_results[0], "n_folds": 1}

    valid = [r for r in fold_results if not r.get("skipped")]
    if not valid:
        return {**fold_results[0], "n_folds": 0}

    aggregated: dict[str, Any] = {}
    # Determine scalar keys from the first valid result
    scalar_keys = [k for k, v in valid[0].items() if isinstance(v, (int, float))]
    for key in scalar_keys:
        values = [r[key] for r in valid if key in r]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
    # Keep non-scalar entries (arrays, DataFrames) from fold 0
    for key, val in valid[0].items():
        if not isinstance(val, (int, float)):
            aggregated[key] = val
    aggregated["n_folds"] = len(valid)
    return aggregated


def run_tangram_stage(
    adata_full: sc.AnnData,
    preprocessed_datasets: dict[str, sc.AnnData],
    output_dir: str | Path,
    config: EvaluationConfig,
    splits: list[tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]],
) -> dict[str, Any]:
    """Run Tangram reconstruction check for all panels across all folds.

    Uses the same ``splits`` object as :func:`run_variability_evaluation` so
    that NMF and Tangram operate on identical train/test partitions.

    Note: With k-fold, Tangram runs ``n_splits × n_panels`` times. Tangram is
    compute-intensive (~1000 epochs per fit), so k-fold is significantly slower
    than a simple split. Use ``--split_mode simple`` if runtime is a concern.

    Args:
        adata_full: Full-transcriptome AnnData reference.
        preprocessed_datasets: Preprocessed panel datasets.
        output_dir: Root output directory.
        config: Pipeline configuration.
        splits: List of ``(train_idx, test_idx, per_celltype_splits)`` tuples
            produced by :func:`generate_evaluation_splits`.

    Returns:
        Dictionary mapping panel name → aggregated Tangram result dict.
    """
    logger.info("=" * 80)
    logger.info("TANGRAM RECONSTRUCTION CHECK (%d fold(s))", len(splits))
    logger.info("=" * 80)

    if not _RECONSTRUCTION_AVAILABLE:
        logger.warning("_tangram module not available – skipping Tangram stage.")
        return {}

    tangram_dir = Path(output_dir) / "Tangram-Evaluation"
    panel_names = [n for n in preprocessed_datasets if n != "full_transcriptome"]

    # Accumulate per-fold results per panel
    fold_results_per_panel: dict[str, list[dict[str, Any]]] = {n: [] for n in panel_names}

    for fold, (train_idx, test_idx, per_celltype_splits) in enumerate(splits):
        logger.info("-" * 60)
        logger.info("TANGRAM FOLD %d/%d", fold + 1, len(splits))
        logger.info("-" * 60)

        for panel_name in tqdm(panel_names, desc=f"Tangram fold {fold + 1}", leave=False):
            adata_panel = preprocessed_datasets[panel_name]
            try:
                result = run_tangram_reconstruction_check(
                    adata_full=adata_full,
                    adata_subset=adata_panel,
                    output_dir=tangram_dir,
                    dataset_name=panel_name,
                    celltype_col=config.celltype_col,
                    num_epochs=config.tangram_n_epochs,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    per_celltype_splits=per_celltype_splits,
                    nmf_counts_input=config.nmf_counts_input,
                )
            except Exception as exc:
                logger.warning(
                    "Tangram fold %d failed for '%s': %s. Skipping.", fold, panel_name, exc, exc_info=True
                )
                result = {"skipped": True, "skip_reason": str(exc)}
            fold_results_per_panel[panel_name].append(result)
            gc.collect()

    # Aggregate across folds
    results = {
        name: _aggregate_tangram_fold_results(fold_list)
        for name, fold_list in fold_results_per_panel.items()
    }
    return results


# ---------------------------------------------------------------------------
# Main pipeline orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse CLI arguments and run the requested pipeline mode."""
    parser = _build_parser()
    args = parser.parse_args()
    config = _args_to_config(args)

    # Logging
    log_path = config.output_dir / "logs" / f"evaluation_{datetime.now():%Y%m%d_%H%M%S}.log"
    _setup_logging(log_path)

    # ── Dry run ───────────────────────────────────────────────────────
    if config.dry_run:
        logger.info("[DRY RUN] Resolved configuration:")
        for field_name, value in vars(config).items():
            logger.info("  %-35s = %s", field_name, value)
        logger.info("[DRY RUN] Exiting without execution.")
        return

    _save_evaluation_parameters(config)

    logger.info("=" * 80)
    logger.info("EVALUATION PIPELINE  |  mode=%s", config.mode)
    logger.info("=" * 80)
    _log_memory("start")

    # ── Preprocessing ─────────────────────────────────────────────────
    if config.mode in ("preprocess", "both"):
        run_preprocessing_stage(config)
        _log_memory("after preprocessing")

    if config.mode == "preprocess":
        logger.info("Preprocessing complete. Use --mode evaluate to run evaluation.")
        return

    # ── Load reference ────────────────────────────────────────────────
    ref_path = config.preprocessed_dir / "full_transcriptome.h5ad"
    if not ref_path.exists():
        # Fall back to input_file if it points to a preprocessed reference
        if config.input_file.exists():
            ref_path = config.input_file
        else:
            logger.error(
                "Reference file not found at %s. Run --mode preprocess first.", ref_path
            )
            sys.exit(1)

    logger.info("Loading reference from %s", ref_path)
    adata_full = sc.read_h5ad(ref_path)
    adata_full.obs_names_make_unique()
    _log_memory("after loading reference")

    # ── Load preprocessed datasets ────────────────────────────────────
    filter_args = _build_filter_args(config)
    preprocessed_datasets = _load_preprocessed_datasets(
        config.preprocessed_dir,
        reference_adata=adata_full,
        filter_args=filter_args,
    )
    _log_memory("after loading panels")

    # ── Evaluation ────────────────────────────────────────────────────
    external_names = config.external_names if config.external_names else None

    # Generate train/test splits once — NMF and Tangram use the exact same partitions.
    splits = generate_evaluation_splits(
        adata_full,
        celltype_col=config.celltype_col,
        split_mode=config.split_mode,
        test_size=config.test_size,
        n_splits=config.n_splits,
        random_state=config.random_state,
    )
    logger.info(
        "Generated %d split(s) (mode=%s, test_size=%.2f, random_state=%d)",
        len(splits), config.split_mode, config.test_size, config.random_state,
    )

    if config.evaluation_type in ("baseline", "both"):
        run_baseline_evaluation(
            preprocessed_datasets=preprocessed_datasets,
            output_dir=config.output_dir,
            celltype_col=config.celltype_col,
            dimensionality_reduction=config.dimensionality_reduction,
            external_names=external_names,
        )
        _log_memory("after baseline evaluation")
        generate_baseline_evaluation_plots(output_dir=config.output_dir)
        _log_memory("after baseline evaluation plots")

    if config.evaluation_type in ("variability", "both"):
        run_variability_evaluation(
            adata_full=adata_full,
            preprocessed_datasets=preprocessed_datasets,
            output_dir=config.output_dir,
            splits=splits,
            n_components=config.n_components,
            celltype_col=config.celltype_col,
            random_state=config.random_state,
            external_names=external_names,
            nmf_counts_input=config.nmf_counts_input,
        )
        _log_memory("after variability evaluation")
        generate_variability_evaluation_plots(output_dir=config.output_dir)
        _log_memory("after variability evaluation plots")

    # ── Tangram (optional) ────────────────────────────────────────────
    if config.include_tangram:
        run_tangram_stage(
            adata_full=adata_full,
            preprocessed_datasets=preprocessed_datasets,
            output_dir=config.output_dir,
            config=config,
            splits=splits,
        )
        _log_memory("after tangram")
        generate_tangram_vs_nmf_fold_aggregate_plots(output_dir=config.output_dir)
        _log_memory("after tangram-vs-nmf aggregate plots")

    # ── Feature plots & dot plots ──────────────────────────────────────
    generate_panel_plots(
        adata_full=adata_full,
        preprocessed_datasets=preprocessed_datasets,
        celltype_col=config.celltype_col,
        output_dir=config.output_dir,
    )
    _log_memory("after feature plots")

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE – outputs in: %s", config.output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()