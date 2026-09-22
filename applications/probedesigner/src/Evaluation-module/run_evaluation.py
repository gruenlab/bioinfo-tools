#!/usr/bin/env python3
"""Unified evaluation pipeline entry point.

This script provides a single CLI to run any combination of:

- **preprocess**: Subset panels to selected genes and compute PCA/NMF,
  clustering, and KNN graphs.
- **evaluate**: Compute baseline (clustering/KNN/celltype), variability
  (NMF), and/or biology (pathway enrichment) metrics on preprocessed panels.
- **both** (pipeline mode): Run preprocessing immediately followed by
  evaluation in one command (equivalent to running the two modes
  sequentially). Not to be confused with ``--evaluation_type``, below.

``--evaluation_type`` selects which metric(s) to compute: ``baseline``,
``variability``, ``biology``, ``all`` (baseline + variability + biology), or
``tangram_only``.

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
        --evaluation_type all

    # Everything in one go
    python run_evaluation.py --mode both \\
        --input_file data.h5ad \\
        --gene_lists_dir Selected-panels/ \\
        --preprocessed_dir preprocessed/ \\
        --output_dir Evaluation-Results/ \\
        --evaluation_type all \\
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
import time
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

# ---------------------------------------------------------------------------
# Internal module imports
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(_MODULE_DIR))

from _clustering import (
    evaluate_celltype_identification,
    evaluate_clustering_quality,
    evaluate_neighborhood_preservation,
)
from _filters import filter_datasets_by_args
from _preprocessing import (
    load_all_gene_lists,
    load_gene_list_from_csv,
    extract_genelist_name_from_path,
    preprocess_reference_dataset,
    process_data_for_panel_evaluation,
    filter_reference_blacklist,
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
    from _tangram import (
        run_tangram_reconstruction_check,
        _save_tangram_results,
        _aggregate_per_celltype_metrics,
    )
    _RECONSTRUCTION_AVAILABLE = True
except ImportError:
    _RECONSTRUCTION_AVAILABLE = False

# Optional pathway enrichment (biology evaluation)
try:
    from _pathway import (
        run_pathway_enrichment_for_panel,
        run_pathway_enrichment_by_celltype,
        load_panel_informative_map,
        resolve_pathway_libraries,
    )
    _PATHWAY_AVAILABLE = True
except ImportError:
    _PATHWAY_AVAILABLE = False

    def resolve_pathway_libraries(organism, override=None):  # noqa: D401 - fallback stub
        if override:
            return list(override)
        return (
            ["GO_Biological_Process_2023", "KEGG_2019_Mouse", "Reactome_2022"]
            if organism == "mouse"
            else ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"]
        )

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
DEFAULT_CELLTYPE_CLF_MAX_DEPTH = _eval_constants.DEFAULT_CELLTYPE_CLF_MAX_DEPTH
DEFAULT_DIM_REDUCTION_PREPROCESS = _eval_constants.DEFAULT_DIM_REDUCTION_PREPROCESS
DEFAULT_EVALUATION_TYPE = _eval_constants.DEFAULT_EVALUATION_TYPE
DEFAULT_N_COMPONENTS = _eval_constants.DEFAULT_N_COMPONENTS
DEFAULT_N_NEIGHBORS = _eval_constants.DEFAULT_N_NEIGHBORS
DEFAULT_RANDOM_STATE = _eval_constants.DEFAULT_RANDOM_STATE
DEFAULT_TANGRAM_N_EPOCHS = _eval_constants.DEFAULT_TANGRAM_N_EPOCHS
NMF_PREDICTOR_FIXED_KWARGS = _eval_constants.NMF_PREDICTOR_FIXED_KWARGS

# ---------------------------------------------------------------------------
# Utility module (external dependency)
# ---------------------------------------------------------------------------

_UTILITY_DIR = _MODULE_DIR.parent / "Utility-module"
sys.path.insert(0, str(_UTILITY_DIR))
from _validation import is_anndata_raw, is_anndata_raw_layer  # type: ignore[import]
from _nmf_objective import resolve_nmf_objective
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
        reference_blacklist_patterns: Extra gene-name prefix patterns to strip
            from the reference dataset before preprocessing (e.g. ``["rps", "rpl"]``).
        reference_use_default_blacklist: Also strip genes matching
            Selection-module's default blacklist (``mt-``, ``hsp``, ``rps``, ``rpl``)
            from the reference. Defaults to ``True``; set
            ``--reference_disable_default_blacklist`` on the CLI to turn it off.
        evaluation_type: Evaluation types to run:
            ``"baseline"``, ``"variability"``, ``"biology"`` (Enrichr pathway
            enrichment), ``"all"`` (baseline + variability + biology), or
            ``"tangram_only"`` (skip NMF and baseline; run only the Tangram
            reconstruction stage).
        celltype_col: obs column holding cell-type labels.
        celltype_clf_max_depth: Fixed ``max_depth`` for the cell-type-classification
            decision tree (baseline evaluation), applied identically to every panel.
            Defaults to ``DEFAULT_CELLTYPE_CLF_MAX_DEPTH``; changing it changes the
            measuring instrument, so hold it constant across any panels being compared.
        n_components: Number of NMF components (variability evaluation).
        random_state: Global random seed.
        n_splits: Number of folds for the stratified k-fold cross-validation used
            by both NMF and Tangram (they share the identical splits).
        n_neighbors: Nearest neighbours for KNN graph construction.
        dim_reduction_preprocess: Dimensionality reduction to compute during
            preprocessing (``"pca"``, ``"nmf"``, or ``"both"``).
        include_tangram: Run the optional Tangram reconstruction check.
        tangram_n_epochs: Number of Tangram mapping epochs.
        pathway_libraries: Enrichr gene-set libraries to query for the
            ``"biology"``/``"all"`` evaluation types.
        pathway_organism: Enrichr organism (``"human"`` or ``"mouse"``).
        pathway_sleep: Seconds to sleep between per-panel Enrichr API calls.
        pathway_top_n: Number of top pathways to plot per library.
        external_panels: Paths to external gene-panel CSV files.
        external_names: Display names for the external panels.
        external_probeset_sizes: Per-panel gene count limits (0 = keep all).
        dry_run: Print resolved config and exit without executing.
        strategies: Strategy filter for evaluation dataset selection.
        probeset_sizes: Panel-size filter.
        filter_methods: Preprocessing filter method filter.
        hvg_subset_options: HVG option filter.
        reduction_types: Reduction type filter.
        analysis_types: Analysis type filter.
        dt_percentages: DT percentage filter.
        dimred_percentages: Dimred percentage filter.
        run_celltype_specific_filling: Gap-filling filter.
        run_global_gene_filling: Gap-filling filter.
        run_deg_based_filling: Gap-filling filter.
        preferred_strategy: Preferred strategy filter.
        expvar_mode: Explained-variance gene-aggregation mode passed to
            ``metrics.calculate_explained_variance`` (see ``metrics.EXPVAR_MODES``
            for the 5 available modes). Defaults to ``"global_mean"``.
    """

    # Required
    mode: str
    input_file: Path
    preprocessed_dir: Path
    output_dir: Path

    # Preprocessing inputs
    gene_lists_dir: Path | None = None
    gene_list_files_txt: Path | None = None

    # Reference blacklist filtering (mirrors Selection-module's blacklist filter)
    reference_blacklist_patterns: list[str] = field(default_factory=list)
    reference_use_default_blacklist: bool = True

    # Evaluation settings
    evaluation_type: str = DEFAULT_EVALUATION_TYPE
    celltype_col: str = DEFAULT_CELLTYPE_COLUMN
    celltype_clf_max_depth: int = DEFAULT_CELLTYPE_CLF_MAX_DEPTH
    n_components: int = DEFAULT_N_COMPONENTS
    random_state: int = DEFAULT_RANDOM_STATE
    n_splits: int = 5

    # Preprocessing settings
    n_neighbors: int = DEFAULT_N_NEIGHBORS
    dim_reduction_preprocess: str = DEFAULT_DIM_REDUCTION_PREPROCESS

    # Optional Tangram
    include_tangram: bool = False
    tangram_n_epochs: int = DEFAULT_TANGRAM_N_EPOCHS

    # Pathway enrichment (biology evaluation).
    # None => resolve the library set from `pathway_organism` (see _pathway.resolve_pathway_libraries).
    pathway_libraries: list[str] | None = None
    pathway_organism: str = "human"
    pathway_sleep: float = 1.0
    pathway_top_n: int = 12
    # Per-cell-type pathway enrichment (in addition to the whole-panel run). On by
    # default; disable with --no-pathway_per_celltype.
    pathway_per_celltype: bool = True
    pathway_celltype_gene_source: str = "informative"  # "informative" | "all_panel"
    pathway_celltype_min_genes: int = 5

    # External panels
    external_panels: list[str] = field(default_factory=list)
    external_names: list[str] = field(default_factory=list)
    external_probeset_sizes: list[int] = field(default_factory=list)

    # Dry-run
    dry_run: bool = False

    # Dataset filters
    strategies: list[str] = field(default_factory=list)
    probeset_sizes: list[int] = field(default_factory=list)
    filter_methods: list[str] = field(default_factory=list)
    hvg_subset_options: list[str] = field(default_factory=list)
    reduction_types: list[str] = field(default_factory=list)
    analysis_types: list[str] = field(default_factory=list)
    dt_percentages: list[float] = field(default_factory=list)
    dimred_percentages: list[float] = field(default_factory=list)
    run_celltype_specific_filling: str = ""
    run_global_gene_filling: str = ""
    run_deg_based_filling: str = ""
    preferred_strategy: str = ""

    # NMF / Tangram count input
    nmf_counts_input: str = "raw"

    # NMF factorization objective: "auto" derives from nmf_counts_input (raw -> mu/KL,
    # lognorm -> cd/Frobenius); "frobenius"/"kl" force one regardless of input.
    nmf_objective: str = "auto"

    # Explained-variance aggregation mode
    expvar_mode: str = "global_mean"
    # Optional multi-mode / multi-gene-subset explained-variance breakdown (additive on
    # top of expvar_mode above -- see nmf.py::nmf_reconstruction for the full explanation).
    expvar_modes: list[str] = field(default_factory=list)
    gene_subsets: list[str] = field(
        default_factory=lambda: ["all_genes", "panel_genes_only", "non_panel_genes_only"]
    )


def _save_evaluation_parameters(config: EvaluationConfig) -> None:
    """Save all evaluation parameter settings to a JSON file in the output directory."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    params = asdict(config)

    # Record the NMF solver/beta_loss actually resolved from
    # nmf_counts_input + nmf_objective (e.g. 'auto' alone doesn't say whether this run
    # factorized with Frobenius or Kullback-Leibler).
    _nmf_kwargs = resolve_nmf_objective(config.nmf_counts_input, config.nmf_objective)
    params["nmf_solver"] = _nmf_kwargs["solver"]
    params["nmf_beta_loss"] = _nmf_kwargs["beta_loss"]

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

    # Reference blacklist filtering (mirrors Selection-module's --blacklist_patterns)
    p.add_argument(
        "--reference_blacklist_patterns",
        nargs="+",
        default=[],
        help=(
            "Extra gene-name prefix patterns to remove from the reference dataset before "
            "preprocessing (case-insensitive, e.g. 'rps' 'rpl'). Use to match genes that "
            "were blacklisted out of the panel during selection, so the reference doesn't "
            "unfairly penalize the panel for genes it was barred from selecting. "
            "Default: no extra patterns (off)."
        ),
    )
    p.add_argument(
        "--reference_disable_default_blacklist",
        action="store_true",
        default=False,
        help=(
            "Disable removal of genes matching Selection-module's DEFAULT_BLACKLIST_PATTERNS "
            "('mt-', 'hsp', 'rps', 'rpl') from the reference dataset. Default: enabled, "
            "so the reference isn't unfairly penalized for genes barred from selection."
        ),
    )

    # Evaluation
    p.add_argument(
        "--evaluation_type",
        choices=["baseline", "variability", "biology", "all", "tangram_only"],
        default="all",
        help=(
            "baseline: clustering/kNN/celltype metrics. variability: NMF representation "
            "metrics. biology: Enrichr pathway enrichment per panel. all: baseline + "
            "variability + biology. tangram_only: skip NMF/baseline/biology, run only the "
            "Tangram reconstruction stage (requires --include_tangram)."
        ),
    )
    p.add_argument("--celltype_col", default=DEFAULT_CELLTYPE_COLUMN)
    p.add_argument(
        "--celltype_clf_max_depth", type=int, default=DEFAULT_CELLTYPE_CLF_MAX_DEPTH,
        help=(
            "max_depth of the cell-type-classification DecisionTreeClassifier (baseline "
            "evaluation), fixed identically for every panel in the run. This is a "
            "measuring instrument, not a panel property -- only override it deliberately, "
            "and hold it constant across any set of panels/benchmarks being compared "
            f"(default: {DEFAULT_CELLTYPE_CLF_MAX_DEPTH})."
        ),
    )
    p.add_argument("--n_components", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of folds for the stratified k-fold cross-validation (NMF + Tangram share the splits)",
    )

    # Preprocessing
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--dim_reduction_preprocess", choices=["pca", "nmf", "both"], default="both")

    # Tangram
    p.add_argument("--include_tangram", action="store_true", default=False,
                   help="Run optional Tangram reconstruction check.")
    p.add_argument("--tangram_n_epochs", type=int, default=DEFAULT_TANGRAM_N_EPOCHS)

    # Pathway enrichment (biology evaluation)
    p.add_argument("--pathway_libraries", nargs="*", default=None,
                   help="Enrichr gene-set libraries to query for the 'biology'/'all' evaluation types. "
                        "Default: resolved from --pathway_organism (human: GO BP 2023 / KEGG 2021 Human / "
                        "Reactome 2022; mouse: same but KEGG 2019 Mouse).")
    p.add_argument("--pathway_organism", choices=["human", "mouse"], default="human",
                   help="Enrichr organism. Selects the default library set unless --pathway_libraries "
                        "is given.")
    p.add_argument("--pathway_sleep", type=float, default=1.0,
                   help="Seconds to sleep between per-panel Enrichr API calls (politeness/rate-limit).")
    p.add_argument("--pathway_top_n", type=int, default=12,
                   help="Number of top (lowest adjusted p-value) pathways to plot per library.")
    p.add_argument("--pathway_per_celltype", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Also run Enrichr per cell type (in addition to the whole-panel run) "
                        "for the 'biology'/'all' evaluation types. On by default; pass "
                        "--no-pathway_per_celltype to disable.")
    p.add_argument("--pathway_celltype_gene_source", choices=["informative", "all_panel"],
                   default="informative",
                   help="Per-cell-type gene set. 'informative': panel genes whose "
                        "ranked_gene_list.csv 'informative_celltypes' column lists that cell "
                        "type (requires --gene_lists_dir or --gene_list_files_txt). "
                        "'all_panel': the full panel gene list for every cell type in "
                        "adata.obs[--celltype_col].")
    p.add_argument("--pathway_celltype_min_genes", type=int, default=5,
                   help="Skip per-cell-type enrichment for cell types with fewer than this "
                        "many genes.")

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
    p.add_argument(
        "--nmf_objective",
        type=str,
        default="auto",
        choices=["auto", "frobenius", "kl"],
        help=(
            "NMF factorization objective. "
            "'auto' (default): derive from --nmf_counts_input — raw -> mu solver + "
            "Kullback-Leibler beta_loss + nndsvda init + max_iter 2000; lognorm -> cd + "
            "Frobenius + nndsvd + 1000 (identical to the historical pinned config). "
            "'frobenius' / 'kl': force that objective regardless of input."
        ),
    )
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
            "Optional list of explained-variance aggregation modes to additionally report in "
            "the variability stage, on top of --expvar_mode (unaffected). E.g. "
            "'--expvar_modes global_mean variance_weighted_sum mean' adds "
            "expvar_test_probe_{gene_subset}_{mode} columns for every mode listed. "
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
            "Which gene subset(s) to score the variability stage's probe reconstruction "
            "against. This does NOT change what is reconstructed (always the full "
            "transcriptome) -- it is a scoring-time column mask on the already-computed "
            "reconstruction, matching run_expvar_aggregation_test.py's convention. "
            "Defaults to all three subsets (all_genes, panel_genes_only, "
            "non_panel_genes_only) so the gene-subset explained-variance breakdown plot "
            "is always populated; the extra cost is 2 masked expvar calls per fold "
            "(no NMF refit)."
        ),
    )

    # External panels
    p.add_argument("--external_panels", nargs="*", default=[])
    p.add_argument("--external_names", nargs="*", default=[])
    p.add_argument("--external_probeset_sizes", nargs="*", type=int, default=[])

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
    p.add_argument("--dt_percentages", nargs="*", type=float, default=[])
    p.add_argument("--dimred_percentages", nargs="*", type=float, default=[])
    p.add_argument("--run_celltype_specific_filling", default="")
    p.add_argument("--run_global_gene_filling", default="")
    p.add_argument("--run_deg_based_filling", default="")
    p.add_argument("--preferred_strategy", default="")

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
        reference_blacklist_patterns=args.reference_blacklist_patterns or [],
        reference_use_default_blacklist=not args.reference_disable_default_blacklist,
        evaluation_type=args.evaluation_type,
        celltype_col=args.celltype_col,
        celltype_clf_max_depth=args.celltype_clf_max_depth,
        n_components=args.n_components,
        random_state=args.random_state,
        n_splits=args.n_splits,
        n_neighbors=args.n_neighbors,
        dim_reduction_preprocess=args.dim_reduction_preprocess,
        include_tangram=args.include_tangram,
        tangram_n_epochs=args.tangram_n_epochs,
        pathway_libraries=args.pathway_libraries,
        pathway_organism=args.pathway_organism,
        pathway_sleep=args.pathway_sleep,
        pathway_top_n=args.pathway_top_n,
        pathway_per_celltype=args.pathway_per_celltype,
        pathway_celltype_gene_source=args.pathway_celltype_gene_source,
        pathway_celltype_min_genes=args.pathway_celltype_min_genes,
        external_panels=args.external_panels or [],
        external_names=args.external_names or [],
        external_probeset_sizes=args.external_probeset_sizes or [],
        dry_run=args.dry_run,
        strategies=args.strategies or [],
        probeset_sizes=args.probeset_sizes or [],
        filter_methods=args.filter_methods or [],
        hvg_subset_options=args.hvg_subset_options or [],
        reduction_types=args.reduction_types or [],
        analysis_types=args.analysis_types or [],
        dt_percentages=args.dt_percentages or [],
        dimred_percentages=args.dimred_percentages or [],
        run_celltype_specific_filling=args.run_celltype_specific_filling or "",
        run_global_gene_filling=args.run_global_gene_filling or "",
        run_deg_based_filling=args.run_deg_based_filling or "",
        preferred_strategy=args.preferred_strategy or "",
        nmf_counts_input=args.nmf_counts_input,
        nmf_objective=args.nmf_objective,
        expvar_mode=args.expvar_mode,
        expvar_modes=args.expvar_modes or [],
        gene_subsets=args.gene_subsets or ["all_genes"],
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
        # Cell-type results carry both a dataset-level row (celltype NaN) and per-cell-type
        # rows under the same "dataset" value; write them together, one CSV per dataset.
        base_datasets = results_df[results_df["celltype"].isna()]["dataset"].unique()
        logger.info("Saving %s results for %d datasets to: %s", metric_name, len(base_datasets), output_subdir)

        for base_name in base_datasets:
            combined = results_df[results_df["dataset"] == base_name].copy()
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
    expvar_mode: str = "global_mean",
    nmf_counts_input: str = "raw",
    nmf_objective: str = "auto",
) -> dict[str, Any]:
    """Compute the full-transcriptome NMF baseline.

    Uses the standard NMF convention: ``A ≈ W @ H``.

    Args:
        A_train: Training data (cells × genes).
        A_test: Testing data (cells × genes).
        n_components: Number of NMF components.
        random_state: Random seed.
        expvar_mode: Explained-variance aggregation mode (see ``metrics.EXPVAR_MODES``).
        nmf_counts_input: Which count matrix ``A_train``/``A_test`` were derived from
            (``"raw"`` or ``"lognorm"``) — used only to resolve ``nmf_objective``.
        nmf_objective: NMF factorization objective — ``"auto"`` (default) derives the
            solver/beta_loss from ``nmf_counts_input``; ``"frobenius"``/``"kl"`` force
            that objective regardless of input. See ``_nmf_objective.py``.

    Returns:
        Dictionary with ``"training"`` and ``"testing"`` sub-dicts each
        containing ``W``, ``H``, reconstructed matrix, MSE, and explained
        variance.
    """
    # solver/beta_loss/init/max_iter derived from the count-input choice (raw -> mu/KL,
    # lognorm -> cd/Frobenius) unless nmf_objective forces one; shared with Selection-module.
    _obj_kwargs = resolve_nmf_objective(nmf_counts_input, nmf_objective)

    def _fit(X: np.ndarray) -> dict[str, Any]:
        pred = NmfPredictor(
            embedding_size=n_components, random_state=random_state,
            **_obj_kwargs,
        ).fit(X)
        W, H = pred.ref_embedding, pred.h_reference
        X_recon = W @ H
        return {
            "W": W, "H": H, "X_recon": X_recon,
            "mse": calculate_mse(X, X_recon),
            "expvar": calculate_explained_variance(X, X_recon, mode=expvar_mode),
        }

    return {"training": _fit(A_train), "testing": _fit(A_test)}


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

    # ── Reference blacklist filtering ────────────────────────────────
    # Applied before the reference cache check so both the cached
    # full_transcriptome.h5ad and the per-panel subsetting loop below see
    # the same filtered gene universe.
    adata = filter_reference_blacklist(
        adata,
        blacklist_patterns=config.reference_blacklist_patterns or None,
        use_default_blacklist=config.reference_use_default_blacklist,
    )

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
    external_names: list[str] | None = None,
    celltype_clf_max_depth: int = DEFAULT_CELLTYPE_CLF_MAX_DEPTH,
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
        external_names: Optional list of external panel names for plot colouring.
        celltype_clf_max_depth: Fixed ``max_depth`` for the cell-type-classification
            decision tree, applied identically to every panel here. See
            ``evaluate_celltype_identification`` / ``EvaluationConfig.celltype_clf_max_depth``.

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
            celltype_clf_max_depth=celltype_clf_max_depth,
        )
        results["celltype"] = ct_results
        _save_results_per_dataset(ct_results, results_dir / "celltype", "Cell-type")
    except Exception as exc:
        logger.warning("Cell-type identification failed: %s", exc, exc_info=True)
    gc.collect()
    _log_memory("after cell-type evaluation")

    logger.info("Baseline evaluation complete.")
    return results


# ---------------------------------------------------------------------------
# Biology evaluation stage (pathway enrichment)
# ---------------------------------------------------------------------------


def run_biology_evaluation(
    preprocessed_datasets: dict[str, sc.AnnData],
    output_dir: str | Path,
    libraries: list[str] | None = None,
    organism: str = "human",
    sleep: float = 1.0,
    top_n: int = 12,
    per_celltype: bool = False,
    per_celltype_gene_source: str = "informative",
    per_celltype_min_genes: int = 5,
    gene_list_paths: dict[str, str] | None = None,
    celltype_col: str = "cluster",
    reference_celltypes: list[str] | None = None,
) -> dict[str, Any]:
    """Run Enrichr pathway enrichment on each preprocessed panel.

    For every panel (excluding ``"full_transcriptome"``), runs Enrichr over
    the panel's genes and saves significant pathways + top-hit bar charts.
    Requires ``gseapy``; if unavailable, logs a warning and returns ``{}``.

    When *per_celltype* is set, additionally runs Enrichr once per cell type
    (results under ``<panel>/per_celltype/<safe celltype>/``):

    - ``per_celltype_gene_source="informative"``: for each cell type, the panel
      genes whose ``ranked_gene_list.csv`` ``informative_celltypes`` column lists
      it. Needs *gene_list_paths* (``{panel_name: ranked_gene_list.csv path}``);
      panels with no entry are logged and skipped for the per-cell-type pass.
    - ``per_celltype_gene_source="all_panel"``: the full panel gene list for each
      cell type in *reference_celltypes*.

    Args:
        preprocessed_datasets: ``{dataset_name: AnnData}`` dict. Must include
            a ``"full_transcriptome"`` key (skipped, not itself enriched).
        output_dir: Root output directory. Results go into
            ``Biology-Evaluation/results/``.
        libraries: Enrichr gene-set libraries to query.
        organism: Enrichr organism (``"human"`` or ``"mouse"``).
        sleep: Seconds to sleep between per-panel Enrichr API calls.
        top_n: Number of top pathways to plot per library.
        per_celltype: Also run the per-cell-type enrichment pass.
        per_celltype_gene_source: ``"informative"`` or ``"all_panel"`` (see above).
        per_celltype_min_genes: Skip cell types with fewer than this many genes.
        gene_list_paths: ``{panel_name: ranked_gene_list.csv path}`` for the
            ``"informative"`` source.
        celltype_col: Reference cell-type column name (recorded in the summary).
        reference_celltypes: Cell-type labels for the ``"all_panel"`` source.

    Returns:
        Dictionary with ``"pathway_summary"`` (DataFrame),
        ``"significant_pathways"`` (``{panel_name: DataFrame}``) and, when
        *per_celltype* is set, ``"pathway_per_celltype_summary"`` (DataFrame).
    """
    logger.info("=" * 80)
    logger.info("BIOLOGY EVALUATION: Pathway Enrichment (Enrichr)")
    logger.info("=" * 80)

    if not _PATHWAY_AVAILABLE:
        logger.warning("gseapy not installed – skipping biology evaluation.")
        return {}

    libraries = resolve_pathway_libraries(organism, libraries)
    logger.info("Enrichr organism=%s, libraries=%s", organism, libraries)
    output_dir = Path(output_dir)
    results_dir = output_dir / "Biology-Evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    panel_names = [n for n in preprocessed_datasets if n != "full_transcriptome"]
    logger.info("Evaluating pathway enrichment for %d panels...", len(panel_names))

    if per_celltype:
        logger.info(
            "Per-cell-type pathway enrichment enabled (gene_source=%s, min_genes=%d)",
            per_celltype_gene_source, per_celltype_min_genes,
        )
        gene_list_paths = gene_list_paths or {}

    summary_rows: list[dict[str, Any]] = []
    significant_pathways: dict[str, pd.DataFrame] = {}
    per_celltype_rows: list[dict[str, Any]] = []

    for panel_name in tqdm(panel_names, desc="Biology evaluation"):
        adata_panel = preprocessed_datasets[panel_name]
        genes = adata_panel.var_names.tolist()
        sig = run_pathway_enrichment_for_panel(
            label=panel_name,
            genes=genes,
            out_dir=results_dir / panel_name,
            libraries=libraries,
            organism=organism,
            top_n=top_n,
        )
        n_significant = 0 if sig is None else len(sig)
        summary_rows.append({
            "dataset": panel_name,
            "n_genes": len(genes),
            "n_significant": n_significant,
        })
        if sig is not None:
            significant_pathways[panel_name] = sig
        time.sleep(sleep)

        if not per_celltype:
            continue

        if per_celltype_gene_source == "all_panel":
            if not reference_celltypes:
                logger.warning(
                    "per_celltype_gene_source='all_panel' but no reference cell types "
                    "provided – skipping per-cell-type pass for '%s'", panel_name,
                )
                continue
            genes_by_celltype = {ct: list(genes) for ct in reference_celltypes}
        else:  # "informative"
            csv_path = gene_list_paths.get(panel_name)
            if not csv_path:
                # An empty map means no --gene_lists_dir/--gene_list_files_txt was given
                # (already warned once upfront); a populated map missing just this panel
                # is unexpected and worth a warning.
                log_fn = logger.warning if gene_list_paths else logger.info
                log_fn(
                    "No ranked_gene_list.csv mapped for panel '%s' – skipping its "
                    "per-cell-type pass (pass --gene_lists_dir / --gene_list_files_txt "
                    "or --pathway_celltype_gene_source all_panel)",
                    panel_name,
                )
                continue
            genes_by_celltype = load_panel_informative_map(csv_path, genes)
            if not genes_by_celltype:
                logger.warning(
                    "No per-cell-type gene attribution found for panel '%s'", panel_name,
                )
                continue

        ct_summary = run_pathway_enrichment_by_celltype(
            label=panel_name,
            genes_by_celltype=genes_by_celltype,
            out_dir=results_dir / panel_name / "per_celltype",
            libraries=libraries,
            organism=organism,
            top_n=top_n,
            min_genes=per_celltype_min_genes,
            sleep=sleep,
        )
        for row in ct_summary.to_dict("records"):
            per_celltype_rows.append({
                "dataset": panel_name,
                "celltype_col": celltype_col,
                "gene_source": per_celltype_gene_source,
                **row,
            })

    results: dict[str, Any] = {}
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(results_dir / "pathway_summary.csv", index=False)
        results["pathway_summary"] = df_summary
        results["significant_pathways"] = significant_pathways

    if per_celltype:
        df_ct_summary = pd.DataFrame(
            per_celltype_rows,
            columns=["dataset", "celltype_col", "gene_source", "celltype",
                     "n_genes", "n_significant", "skipped_reason"],
        )
        df_ct_summary.to_csv(
            results_dir / "pathway_per_celltype_summary.csv", index=False
        )
        results["pathway_per_celltype_summary"] = df_ct_summary

    gc.collect()
    _log_memory("after biology evaluation")
    logger.info("Biology evaluation complete.")
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

    group_cols = ["dataset", "gene_list", "analysis_type"]
    if "celltype" in df_per_fold.columns:
        group_cols.append("celltype")

    # Metric columns = every numeric column except the grouping columns and "fold"
    # itself. Generalized from a hardcoded name list to dynamic numeric-dtype
    # detection so the gene-subset x expvar-mode columns (e.g.
    # expvar_test_probe_panel_genes_only_variance_weighted_sum) get fold-aggregated
    # automatically without enumerating every (subset, mode) combination by hand.
    exclude_cols = set(group_cols) | {"fold"}
    available_metrics = [
        c for c in df_per_fold.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

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
    nmf_objective: str = "auto",
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
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
        nmf_counts_input: Which count matrix to use (``"raw"`` or ``"lognorm"``).
        nmf_objective: NMF factorization objective — ``"auto"`` (default) derives the
            solver/beta_loss from ``nmf_counts_input``; ``"frobenius"``/``"kl"`` force
            that objective regardless of input. See ``_nmf_objective.py``.
        expvar_mode: Explained-variance aggregation mode (see ``metrics.EXPVAR_MODES``).
        expvar_modes: Optional list of explained-variance modes to additionally report
            (see ``nmf.py::nmf_reconstruction``). Defaults to ``None``, i.e. ``[expvar_mode]``.
        gene_subsets: Optional list of gene subsets to score against (``"all_genes"``,
            ``"panel_genes_only"``, ``"non_panel_genes_only"``). Defaults to ``None``,
            i.e. ``["all_genes"]``.

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
    # Do NOT densify X_full upfront.  For the full-transcriptome reference
    # (167K × 28K genes) this alone costs ~18 GB.  Instead we materialise only
    # the per-fold train/test slices inside the loop below, so at most one
    # fold's worth of dense data coexists with the sparse source matrix.

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

        # Materialise only the current fold's slices as dense float32.
        # Keeps peak usage to ~one fold's worth of data instead of the full matrix.
        def _to_dense_f32(X: np.ndarray | scipy.sparse.spmatrix, idx: np.ndarray) -> np.ndarray:
            s = X[idx]
            if scipy.sparse.issparse(s):
                s = s.toarray()
            return np.asarray(s, dtype=np.float32)

        A_train = _to_dense_f32(X_full, train_idx)
        A_test = _to_dense_f32(X_full, test_idx)

        # Convert index-based per-celltype splits to array-based splits
        ct_splits: dict[str, tuple[np.ndarray, np.ndarray]] = {
            ct: (_to_dense_f32(X_full, tr), _to_dense_f32(X_full, te))
            for ct, (tr, te) in ct_splits_idx.items()
        }

        # Initialize NMF caches per fold
        cached_full_nmf: dict[int, Any] = {}
        cached_ct_nmf: dict[str, dict] = {}

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
                    cached_ct_nmf=cached_ct_nmf,
                    n_components=panel_n,
                    celltype_col=celltype_col,
                    random_state=random_state,
                    nmf_rows=all_nmf_rows,
                    per_celltype_splits=ct_splits_idx,
                    fold=fold,
                    nmf_counts_input=nmf_counts_input,
                    nmf_objective=nmf_objective,
                    expvar_mode=expvar_mode,
                    expvar_modes=expvar_modes,
                    gene_subsets=gene_subsets,
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
    cached_ct_nmf: dict[str, dict],
    n_components: int,
    celltype_col: str,
    random_state: int,
    nmf_rows: list[dict[str, Any]],
    per_celltype_splits: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    fold: int | None = None,
    nmf_counts_input: str = "raw",
    nmf_objective: str = "auto",
    expvar_mode: str = "global_mean",
    expvar_modes: list[str] | None = None,
    gene_subsets: list[str] | None = None,
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
        cached_ct_nmf: Cache for per-cell-type NMF baselines.
        n_components: NMF components for this panel.
        celltype_col: obs column for cell-type labels.
        random_state: Random seed.
        nmf_rows: NMF result rows output list (mutated in place).
        per_celltype_splits: Per-cell-type index splits.
        fold: Fold number (optional, for k-fold evaluation).
        nmf_counts_input: Which count matrix to use (``"raw"`` or ``"lognorm"``).
        nmf_objective: NMF factorization objective — ``"auto"`` (default) derives the
            solver/beta_loss from ``nmf_counts_input``; ``"frobenius"``/``"kl"`` force
            that objective regardless of input. See ``_nmf_objective.py``.
        expvar_mode: Explained-variance aggregation mode (see ``metrics.EXPVAR_MODES``).
        expvar_modes: Optional list of explained-variance modes to additionally report
            (see ``nmf.py::nmf_reconstruction``). Defaults to ``None``, i.e. ``[expvar_mode]``.
        gene_subsets: Optional list of gene subsets to score against. Defaults to
            ``None``, i.e. ``["all_genes"]``.
    """
    # Lazily compute global NMF baseline
    if n_components not in cached_full_nmf:
        logger.info("Computing global NMF baseline (n_components=%d)...", n_components)
        cached_full_nmf[n_components] = _compute_full_nmf_baseline(
            A_train, A_test, n_components, random_state, expvar_mode=expvar_mode,
            nmf_counts_input=nmf_counts_input, nmf_objective=nmf_objective,
        )

    baseline = cached_full_nmf[n_components]

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
        random_state=random_state,
        cached_full_nmf=variability_cache,
        nmf_counts_input=nmf_counts_input,
        nmf_objective=nmf_objective,
        expvar_mode=expvar_mode,
        expvar_modes=expvar_modes,
        gene_subsets=gene_subsets,
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
            random_state=random_state,
            cached_full_nmf_by_celltype=cached_ct_nmf,
            per_celltype_splits=per_celltype_splits,
            nmf_counts_input=nmf_counts_input,
            nmf_objective=nmf_objective,
            expvar_mode=expvar_mode,
            expvar_modes=expvar_modes,
            gene_subsets=gene_subsets,
        )
        if mech_ct:
            # Summary row: aggregated macro/weighted metrics
            summary = mech_ct.get("summary", {})
            if summary:
                row = {**summary, "dataset": panel_name, "gene_list": panel_name, "celltype": "summary", "analysis_type": "per_celltype"}
                if fold is not None:
                    row["fold"] = fold
                nmf_rows.append(row)
            # Per-cell-type rows: one row per cell type with individual metrics
            for ct, res in mech_ct.get("celltype_results", {}).items():
                row = {**res, "dataset": panel_name, "gene_list": panel_name, "celltype": ct, "analysis_type": "per_celltype"}
                if fold is not None:
                    row["fold"] = fold
                nmf_rows.append(row)


# ---------------------------------------------------------------------------
# Tangram stage
# ---------------------------------------------------------------------------


def _aggregate_scalar_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean/std across a list of flat dicts with numeric values.

    Non-numeric values (arrays, strings, etc.) are kept from the first dict.
    Keys starting with ``_`` (internal arrays such as ``_mean_ref``) are
    excluded from mean/std computation but preserved from the first dict.

    Args:
        dicts: List of flat dicts with numeric values to aggregate.

    Returns:
        Dict with ``{key}_mean`` and ``{key}_std`` for each numeric key, plus
        non-numeric entries copied from ``dicts[0]``.
    """
    if not dicts:
        return {}
    scalar_keys = [
        k for k, v in dicts[0].items()
        if isinstance(v, (int, float)) and not k.startswith("_")
    ]
    agg: dict[str, Any] = {}
    for k in scalar_keys:
        vals = [d[k] for d in dicts if k in d and not np.isnan(d[k])]
        agg[f"{k}_mean"] = float(np.mean(vals)) if vals else np.nan
        # ddof=1 (sample sd) to match pandas' default in _aggregate_fold_results and
        # the seed-robustness aggregation — one convention module-wide (#39).
        agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    for k, v in dicts[0].items():
        if not isinstance(v, (int, float)):
            agg[k] = v
    return agg


def _aggregate_tangram_fold_results(
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate Tangram results across folds using a two-stage approach.

    Stage 1 — per-celltype across folds: for each cell type, compute mean/std
    of ``mse``, ``expvar``, etc. over all valid folds.

    Stage 2 — macro/weighted across cell types: recompute summary metrics from
    the per-celltype fold-averaged values using :func:`_aggregate_per_celltype_metrics`.

    Args:
        fold_results: List of per-fold result dicts from
            :func:`run_tangram_reconstruction_check`.

    Returns:
        Aggregated result dict with keys ``"global"``, ``"per_celltype"``,
        ``"per_celltype_summary"``, and ``"n_folds"``.
    """
    if not fold_results:
        return {}
    if len(fold_results) == 1:
        return {**fold_results[0], "n_folds": 1}

    valid = [r for r in fold_results if not r.get("skipped")]
    if not valid:
        return {**fold_results[0], "n_folds": 0}

    # Aggregate global metrics (flat dict of scalars)
    global_dicts = [
        r["global"] for r in valid
        if "global" in r and not r["global"].get("skipped")
    ]
    global_agg = _aggregate_scalar_dicts(global_dicts)

    # Stage 1: per-celltype across folds — mean/std per ct
    all_cts: set[str] = set()
    for r in valid:
        all_cts.update((r.get("per_celltype") or {}).keys())

    per_ct_agg: dict[str, dict[str, Any]] = {}
    for ct in all_cts:
        ct_dicts = [
            r["per_celltype"][ct]
            for r in valid
            if ct in (r.get("per_celltype") or {})
            and not r["per_celltype"][ct].get("skipped")
        ]
        if ct_dicts:
            per_ct_agg[ct] = _aggregate_scalar_dicts(ct_dicts)

    # Stage 2: macro/weighted from per-ct means → summary row.
    # Build a synthetic per-ct dict using fold-averaged mse/expvar values so
    # _aggregate_per_celltype_metrics() can compute macro/weighted scores with
    # the same logic used within individual folds.
    synthetic_per_ct: dict[str, dict[str, Any]] = {}
    # n_cells comes from the first valid fold (cell counts don't change across folds)
    first_per_ct = valid[0].get("per_celltype") or {}
    for ct, agg_metrics in per_ct_agg.items():
        n_cells = first_per_ct.get(ct, {}).get("n_cells", 0)
        synthetic_per_ct[ct] = {
            "mse": agg_metrics.get("mse_mean", np.nan),
            "expvar": agg_metrics.get("expvar_mean", np.nan),
            "n_cells": n_cells,
            "skipped": False,
        }

    summary_agg: dict[str, Any] = {}
    if _RECONSTRUCTION_AVAILABLE and synthetic_per_ct:
        summary_agg = _aggregate_per_celltype_metrics(synthetic_per_ct)

    return {
        "global": global_agg,
        "per_celltype": per_ct_agg,
        "per_celltype_summary": summary_agg,
        "n_folds": len(valid),
    }


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

    Note: Tangram runs ``n_splits × n_panels`` times and is compute-intensive
    (~1000 epochs per fit); lower ``--n_splits`` if runtime is a concern.

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
                    fold=fold,
                )
            except Exception as exc:
                logger.warning(
                    "Tangram fold %d failed for '%s': %s. Skipping.", fold, panel_name, exc, exc_info=True
                )
                result = {"skipped": True, "skip_reason": str(exc)}
            fold_results_per_panel[panel_name].append(result)
            gc.collect()

    # Aggregate across folds and save aggregated results
    results: dict[str, Any] = {}
    for name, fold_list in fold_results_per_panel.items():
        agg = _aggregate_tangram_fold_results(fold_list)
        results[name] = agg
        # Save aggregated (mean/std) results to global/ and per_celltype/
        _save_tangram_results(
            tangram_dir,
            name,
            global_metrics=agg.get("global") or None,
            per_celltype_results=agg.get("per_celltype") or None,
            per_celltype_summary=agg.get("per_celltype_summary") or None,
            fold=None,
        )
        logger.info("Saved aggregated Tangram results for '%s' (%d folds)", name, agg.get("n_folds", 0))
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
        gc.collect()  # ensure the raw AnnData loaded inside run_preprocessing_stage is freed
                      # before we load the preprocessed reference + all panel h5ads

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
        n_splits=config.n_splits,
        random_state=config.random_state,
    )
    logger.info(
        "Generated %d stratified fold(s) (random_state=%d)",
        len(splits), config.random_state,
    )

    if config.evaluation_type in ("baseline", "all"):
        run_baseline_evaluation(
            preprocessed_datasets=preprocessed_datasets,
            output_dir=config.output_dir,
            celltype_col=config.celltype_col,
            external_names=external_names,
            celltype_clf_max_depth=config.celltype_clf_max_depth,
        )
        _log_memory("after baseline evaluation")

    if config.evaluation_type in ("biology", "all"):
        gene_list_paths: dict[str, str] = {}
        reference_celltypes: list[str] | None = None
        if config.pathway_per_celltype:
            if config.celltype_col in adata_full.obs:
                reference_celltypes = (
                    adata_full.obs[config.celltype_col].astype(str).unique().tolist()
                )
            if config.pathway_celltype_gene_source == "informative":
                if config.gene_list_files_txt and config.gene_list_files_txt.exists():
                    gene_list_paths = {
                        extract_genelist_name_from_path(p.strip()): p.strip()
                        for p in config.gene_list_files_txt.read_text().splitlines()
                        if p.strip()
                    }
                elif config.gene_lists_dir and config.gene_lists_dir.exists():
                    _, gene_list_paths = load_all_gene_lists(
                        str(config.gene_lists_dir), adata_full, return_paths=True
                    )
                # --external_panels/--external_names map directly by their given name --
                # no path-derived naming risk, so these are always safe to add (merged,
                # not exclusive with gene_list_files_txt/gene_lists_dir above).
                if config.external_panels and config.external_names:
                    gene_list_paths.update(
                        dict(zip(config.external_names, config.external_panels))
                    )
                if not gene_list_paths:
                    logger.warning(
                        "--pathway_per_celltype --pathway_celltype_gene_source informative "
                        "needs --gene_lists_dir, --gene_list_files_txt, or "
                        "--external_panels/--external_names; per-cell-type pass will be "
                        "skipped for every panel."
                    )
        run_biology_evaluation(
            preprocessed_datasets=preprocessed_datasets,
            output_dir=config.output_dir,
            libraries=config.pathway_libraries,
            organism=config.pathway_organism,
            sleep=config.pathway_sleep,
            top_n=config.pathway_top_n,
            per_celltype=config.pathway_per_celltype,
            per_celltype_gene_source=config.pathway_celltype_gene_source,
            per_celltype_min_genes=config.pathway_celltype_min_genes,
            gene_list_paths=gene_list_paths,
            celltype_col=config.celltype_col,
            reference_celltypes=reference_celltypes,
        )
        _log_memory("after biology evaluation")

    if config.evaluation_type in ("variability", "all"):
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
            nmf_objective=config.nmf_objective,
            expvar_mode=config.expvar_mode,
            expvar_modes=config.expvar_modes or [config.expvar_mode],
            gene_subsets=config.gene_subsets,
        )
        _log_memory("after variability evaluation")

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

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE – outputs in: %s", config.output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()