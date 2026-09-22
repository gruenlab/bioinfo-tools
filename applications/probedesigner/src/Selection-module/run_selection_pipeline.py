"""
Main orchestrator for gene selection pipeline.

This script routes gene selection strategies to the appropriate modules:
- Single strategies (deg_only, rf_simple, rf_deg, hvg, random, dimred_only) → run_single_selection
- Combination strategies (RecoVar, RecoVar_PCA) → run_combination_selection

Supports caching, filtering (blacklist, Xenium), and all strategy parameters.

Usage:
    python run_selection_pipeline.py \\
        --strategy RecoVar \\
        --input_file data.h5ad \\
        --output_dir results/ \\
        --probeset_size 100 \\
        --reduction_type nmf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import scanpy as sc
from anndata import AnnData

# Support script execution by adding module directory to sys.path
MODULE_DIR = Path(__file__).parent.absolute()
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

UTILITY_DIR = MODULE_DIR.parent / "Utility-module"
if str(UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_DIR))
from _nmf_objective import resolve_nmf_objective

# Use absolute imports (for script execution)
from _filtering import apply_blacklist_filter, compute_global_mean_expression
# --- load THIS directory's _constants.py by path (sibling dirs share the name) ---
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _cpath
_cspec = _ilu.spec_from_file_location("_constants", _cpath(__file__).resolve().parent / "_constants.py")
_sys.modules["_constants"] = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_sys.modules["_constants"])


from _constants import (
    DEFAULT_ALL_STRATEGIES,
    DEFAULT_COMBINATION_STRATEGIES,
    DEFAULT_DIMRED_PERCENTAGE,
    DEFAULT_MIN_CELLS_PER_CELLTYPE,
    DEFAULT_PROBESET_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RF_PERCENTAGE,
    DEFAULT_SINGLE_STRATEGIES,
)
from _gene_list_builder import GeneListBuilder, panel_information_filename
from run_combination_selection import run_combination_selection
from run_single_selection import run_single_selection


# Strategy definitions (loaded from _constants.py, overridden by shell script)
SINGLE_STRATEGIES = DEFAULT_SINGLE_STRATEGIES
COMBINATION_STRATEGIES = DEFAULT_COMBINATION_STRATEGIES
ALL_STRATEGIES = DEFAULT_ALL_STRATEGIES


def setup_logging(log_file: Optional[str] = None, level: str = 'INFO') -> None:
    """
    Configure logging for the pipeline.
    
    Parameters
    ----------
    log_file : str, optional
        Path to log file. If None, logs to console only.
    level : str, default='INFO'
        Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def save_parameters_to_file(args: argparse.Namespace, output_dir: str) -> str:
    """
    Save all parameter settings to a JSON file in the output directory.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing all parameters
    output_dir : str
        Directory to save the parameter file
        
    Returns
    -------
    str
        Path to the saved parameter file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert args to dictionary
    params = vars(args).copy()

    # Record the NMF solver/beta_loss actually resolved from
    # --dimred_counts_input + --nmf_objective (e.g. 'auto' alone doesn't say whether
    # this run factorized with Frobenius or Kullback-Leibler).
    _nmf_kwargs = resolve_nmf_objective(args.dimred_counts_input, args.nmf_objective)
    params['nmf_solver'] = _nmf_kwargs['solver']
    params['nmf_beta_loss'] = _nmf_kwargs['beta_loss']

    # Add metadata
    params['timestamp'] = datetime.now().isoformat()
    params['script'] = 'run_selection_pipeline.py'
    
    # Save to JSON file
    param_file = os.path.join(output_dir, 'selection_parameters.json')
    with open(param_file, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)
    
    logging.info(f"Parameter settings saved to: {param_file}")
    return param_file


def update_selection_parameters_with_panel_metadata(
    output_dir: str,
    metadata: dict,
) -> None:
    """Merge JSON-safe panel metadata into selection_parameters.json."""
    if not metadata:
        return

    param_file = os.path.join(output_dir, 'selection_parameters.json')
    if not os.path.exists(param_file):
        return

    safe_metadata = {
        str(key): value
        for key, value in metadata.items()
        if not str(key).startswith("_")
    }
    if not safe_metadata:
        return

    with open(param_file, 'r') as f:
        params = json.load(f)
    params.update(safe_metadata)
    with open(param_file, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)
    logging.info(f"Updated selection_parameters.json with panel metadata: {param_file}")


def resolve_pool_size_per_celltype(probeset_size: int) -> int:
    """Target-aware default for per-celltype dimred phase-1 pools."""
    pool_size = (probeset_size + 2) // 3
    return min(5000, max(200, pool_size))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description='Gene Selection Pipeline - Main Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single strategy (DEG)
  python run_selection_pipeline.py \\
    --strategy deg_only \\
    --input_file data.h5ad \\
    --output_dir results/deg/ \\
    --probeset_size 100

  # Single strategy (dimred only)
  python run_selection_pipeline.py \\
    --strategy dimred_only \\
    --input_file data.h5ad \\
    --output_dir results/nmf/ \\
    --probeset_size 100 \\
    --reduction_type nmf

  # Combination strategy (RecoVar) with caching
  python run_selection_pipeline.py \\
    --strategy RecoVar \\
    --input_file data.h5ad \\
    --output_dir results/RecoVar/ \\
    --probeset_size 100 \\
    --rf_deg_cache_dir results/rf_deg/ \\
    --dimred_cache_dir results/nmf/
        """
    )
    
    # =========================================================================
    # Required arguments
    # =========================================================================
    required = parser.add_argument_group('Required arguments')
    
    required.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=ALL_STRATEGIES,
        help=f'Gene selection strategy. Single strategies: {SINGLE_STRATEGIES}. '
             f'Combination strategies: {COMBINATION_STRATEGIES}'
    )
    
    required.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to input AnnData file (.h5ad)'
    )
    
    required.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save results'
    )
    
    # =========================================================================
    # General parameters
    # =========================================================================
    general = parser.add_argument_group('General parameters')
    
    general.add_argument(
        '--probeset_size',
        type=int,
        default=DEFAULT_PROBESET_SIZE,
        help=f'Target number of genes to select (default: {DEFAULT_PROBESET_SIZE})'
    )
    
    general.add_argument(
        '--celltype_column',
        type=str,
        default='celltype',
        help='Column name in adata.obs for cell type labels (default: celltype)'
    )
    
    general.add_argument(
        '--min_cells_per_celltype',
        type=int,
        default=DEFAULT_MIN_CELLS_PER_CELLTYPE,
        help=f'Minimum cells per cell type (default: {DEFAULT_MIN_CELLS_PER_CELLTYPE})'
    )
    
    general.add_argument(
        '--random_state',
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f'Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE})'
    )
    
    general.add_argument(
        '--experiment_name',
        type=str,
        default='gene_selection',
        help='Experiment name for output files (default: gene_selection)'
    )

    general.add_argument(
        '--celltype_class_sizes_dir',
        type=str,
        default=None,
        help=(
            'Directory to write the cells-per-celltype bar chart into (a property of '
            'the input data, independent of strategy/panel). Opt-in only: the plot is '
            'skipped entirely when this is not given -- it is never written into '
            '--output_dir.'
        )
    )

    general.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    # =========================================================================
    # Dimensionality reduction parameters (for dimred_only, RecoVar, RecoVar_PCA)
    # =========================================================================
    dimred = parser.add_argument_group('Dimensionality reduction parameters')
    
    dimred.add_argument(
        '--reduction_type',
        type=str,
        choices=['nmf', 'pca'],
        help='Dimensionality reduction type (required for dimred_only, RecoVar, RecoVar_PCA)'
    )
    
    dimred.add_argument(
        '--n_components',
        type=int,
        default=5,
        help='Number of components for dimensionality reduction (default: 5)'
    )
    
    dimred.add_argument(
        '--pool_size_per_celltype',
        type=int,
        default=None,
        help=(
            'Pool size per celltype for Phase 1 pool creation. '
            'Default is target-aware: min(5000, max(200, ceil(probeset_size/3))).'
        )
    )

    dimred.add_argument(
        '--dimred_n_jobs',
        type=int,
        default=1,
        help=(
            'Workers for per-cell-type NMF/PCA fitting (dimred_only / RecoVar / RecoVar_PCA). '
            '1 (default) = sequential; -1 = all cores; capped at the number of cell types. '
            'Each cell type is an independent fit — results are identical to sequential.'
        )
    )

    dimred.add_argument(
        '--nmf_model_cache_dir',
        type=str,
        default=None,
        help=(
            'Shared directory for NMF/PCA model pkl files. '
            'When set, all strategies (dimred_only, RecoVar, RecoVar_PCA) load/save models here '
            'instead of inside the per-run output directory. '
            'Use this to share a single NMF fit across different probeset sizes or filter settings '
            '(e.g. --nmf_model_cache_dir /path/to/experiment/nmf_cache). '
            'The cache is keyed on n_components; change --n_components to force a fresh fit.'
        )
    )

    dimred.add_argument(
        '--require_nmf_model_cache',
        action='store_true',
        default=False,
        help=(
            'For NMF/PCA strategies that use --nmf_model_cache_dir, require the '
            'requested cached NMF model file to exist and match --n_components. '
            'When enabled, missing or incompatible NMF caches fail instead of recomputing.'
        )
    )

    dimred.add_argument(
        '--dimred_counts_input',
        type=str,
        default='raw',
        choices=['raw', 'lognorm'],
        help=(
            "Count matrix used by dimensionality-reduction selection methods only. "
            "'raw' (default): use raw integer counts from adata.layers['counts'] or adata.raw. "
            "'lognorm': use preprocessed adata.X. "
            "Non-dimred methods and Xenium filtering always use lognorm adata.X."
        )
    )

    dimred.add_argument(
        '--nmf_objective',
        type=str,
        default='auto',
        choices=['auto', 'frobenius', 'kl'],
        help=(
            "NMF factorization objective. 'auto' (default): derive from --dimred_counts_input"
            " -- raw -> mu solver + Kullback-Leibler beta_loss + nndsvda init + max_iter 2000;"
            " lognorm -> cd + Frobenius + nndsvd + 1000 (identical to the historical pinned"
            " config). 'frobenius' / 'kl': force that objective regardless of input. NMF only."
        )
    )

    # =========================================================================
    # Combination strategy parameters (for RecoVar, RecoVar_PCA)
    # =========================================================================
    combo = parser.add_argument_group('Combination strategy parameters')
    
    combo.add_argument(
        '--rf_percentage',
        type=float,
        default=DEFAULT_RF_PERCENTAGE,
        help=f'RF fraction for combination strategies (default: {DEFAULT_RF_PERCENTAGE})'
    )
    
    combo.add_argument(
        '--dimred_percentage',
        type=float,
        default=DEFAULT_DIMRED_PERCENTAGE,
        help=f'Dimred fraction for combination strategies (default: {DEFAULT_DIMRED_PERCENTAGE})'
    )
    
    combo.add_argument(
        '--rf_deg_cache_dir',
        type=str,
        help='Directory with cached rf_deg results (for combination strategies)'
    )

    combo.add_argument(
        '--rf_score_cache_file',
        type=str,
        default=None,
        help=(
            'Explicit path to rf_gene_scores.pkl for RF single strategies. '
            'When set, rf_simple/rf_deg reuse this ranking instead of retraining.'
        )
    )
    
    combo.add_argument(
        '--dimred_cache_dir',
        type=str,
        help='Directory with cached dimred_only results (for combination strategies)'
    )
    
    combo.add_argument(
        '--force_recompute',
        action='store_true',
        help='Ignore cached results and recompute components'
    )

    combo.add_argument(
        '--dotplot_dir',
        type=str,
        default=None,
        help=(
            'Directory to write the final-gene expression dotplot into (combination '
            'strategies only). Opt-in only: the plot is skipped entirely when this is '
            'not given -- it is never written into --output_dir.'
        )
    )

    combo.add_argument(
        '--expression_distribution_dir',
        type=str,
        default=None,
        help=(
            'Directory to write the panel-gene raw-vs-lognorm expression distribution '
            'histogram into (combination strategies only). Opt-in only: the plot is '
            'skipped entirely when this is not given -- it is never written into '
            '--output_dir.'
        )
    )
    
    combo.add_argument(
        '--disable_celltype_filling',
        action='store_true',
        help='Disable cell-type-specific gap-filling strategy'
    )

    combo.add_argument(
        '--disable_deg_filling',
        action='store_true',
        help='Disable DEG-based gap-filling strategy'
    )
    
    # =========================================================================
    # Filtering parameters
    # =========================================================================
    filtering = parser.add_argument_group('Filtering parameters')
    
    filtering.add_argument(
        '--blacklist_patterns',
        type=str,
        nargs='+',
        help='Custom gene name prefixes to blacklist (e.g., mt- rps rpl). Combined with default patterns unless --disable_default_blacklist is set.'
    )
    
    filtering.add_argument(
        '--disable_default_blacklist',
        action='store_true',
        help='Disable default blacklist patterns (mt-, hsp, rps, rpl). Only custom patterns from --blacklist_patterns will be used.'
    )
    
    filtering.add_argument(
        '--force_include_genes',
        type=str,
        nargs='+',
        help='Genes to force-include (highest priority, overrides blacklist)'
    )
    
    filtering.add_argument(
        '--disable_xenium_filter',
        action='store_true',
        help='Disable Xenium expression filtering'
    )

    filtering.add_argument(
        '--disable_xenium_celltype_aware',
        action='store_true',
        help='Use global Xenium filter instead of celltype-aware mode'
    )

    filtering.add_argument(
        '--xenium_min_expr',
        type=float,
        default=0.1,
        help='Minimum mean expression for Xenium filter (default: 0.1)'
    )
    
    filtering.add_argument(
        '--xenium_max_expr',
        type=float,
        default=100.0,
        help='Maximum mean expression for Xenium filter (default: 100.0)'
    )

    general.add_argument(
        '--strict_panel_size',
        action='store_true',
        default=False,
        help=(
            'Fail after selection if the panel-information CSV does not contain exactly '
            '--probeset_size final panel genes.'
        )
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate argument combinations.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
        
    Raises
    ------
    ValueError
        If invalid argument combination detected
    """
    
    # Check reduction_type for dimred strategies
    if args.strategy in ['dimred_only', 'RecoVar', 'RecoVar_PCA']:
        if not args.reduction_type:
            raise ValueError(
                f"--reduction_type required for strategy '{args.strategy}'. "
                "Specify 'nmf' or 'pca'."
            )
        
        # Validate reduction_type matches strategy.
        # 'RecoVar' is the NMF hybrid (renamed from 'rf_nmf'); 'RecoVar_PCA' the
        # PCA one. 'dimred_only' accepts either. (Substring checks like
        # "'nmf' in args.strategy" broke with the rename.)
        if args.strategy in ['RecoVar', 'RecoVar_PCA']:
            expected_reduction = 'pca' if args.strategy == 'RecoVar_PCA' else 'nmf'
            if args.reduction_type != expected_reduction:
                raise ValueError(
                    f"Strategy '{args.strategy}' requires --reduction_type {expected_reduction}, "
                    f"got '{args.reduction_type}'"
                )
    
    # Check percentages for combination strategies
    if args.strategy in COMBINATION_STRATEGIES:
        if abs((args.rf_percentage + args.dimred_percentage) - 1.0) > 0.01:
            raise ValueError(
                f"RF and dimred percentages must sum to 1.0, got "
                f"rf_percentage={args.rf_percentage} + dimred_percentage={args.dimred_percentage} "
                f"= {args.rf_percentage + args.dimred_percentage}"
            )

    if args.pool_size_per_celltype < 1:
        raise ValueError(
            f"--pool_size_per_celltype must be >= 1, got {args.pool_size_per_celltype}"
        )

    # Check input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Check cache directories exist if provided
    if args.rf_deg_cache_dir and not os.path.exists(args.rf_deg_cache_dir):
        raise FileNotFoundError(f"RF cache directory not found: {args.rf_deg_cache_dir}")
    
    if args.dimred_cache_dir and not os.path.exists(args.dimred_cache_dir):
        raise FileNotFoundError(f"Dimred cache directory not found: {args.dimred_cache_dir}")

    if args.rf_score_cache_file and not os.path.exists(args.rf_score_cache_file):
        raise FileNotFoundError(f"RF score cache file not found: {args.rf_score_cache_file}")


def _resolve_ranked_csv_path(
    output_dir: str, strategy: str, reduction_type: Optional[str] = None
) -> str:
    """Resolve the ranked-gene-list CSV path for a given strategy (single or
    combination). See docs/doc-pipeline/audit_3.md, "Selection-module CSV output
    reorganization" -- single strategies each get their own
    `{strategy}_panel_information.csv` (via `panel_information_filename()`);
    combination strategies (RecoVar/RecoVar_PCA) write `RecoVar_panel_information.csv`.
    """
    if strategy in COMBINATION_STRATEGIES:
        return os.path.join(output_dir, "RecoVar_panel_information.csv")
    return os.path.join(output_dir, panel_information_filename(strategy, reduction_type))


def validate_ranked_panel_size(
    output_dir: str, expected_size: int, strategy: str, reduction_type: Optional[str] = None
) -> None:
    """Validate that the strategy's panel-information CSV has the requested final panel size."""
    ranked_file = _resolve_ranked_csv_path(output_dir, strategy, reduction_type)
    if not os.path.exists(ranked_file):
        raise FileNotFoundError(
            f"Strict panel-size validation requested but {os.path.basename(ranked_file)} "
            f"was not found: {ranked_file}"
        )

    import pandas as pd

    ranked_df = pd.read_csv(ranked_file)
    # Every schema this pipeline writes (full single-strategy, trimmed rf/dimred, and
    # combination RecoVar_panel_information.csv) carries an 'in_panel' column -- for
    # combination output this file now also carries non-panel candidates, so filtering
    # on in_panel (rather than counting every row) is required there too.
    final_count = int(ranked_df["in_panel"].fillna(False).astype(bool).sum())

    if final_count != expected_size:
        summary_file = os.path.join(output_dir, "filtering_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                summary = json.load(f)
            if (
                summary.get("strategy") == "rf_deg"
                and summary.get("panel_size_status") == "short"
                and summary.get("short_panel_reason")
                == "rf_deg_candidate_pool_exhausted_after_retries"
                and int(summary.get("final_panel_size", -1)) == final_count
            ):
                logging.warning(
                    "RF-DEG short panel accepted after bounded recompute: "
                    "final_selection=%d, expected=%d, reason=%s",
                    final_count,
                    expected_size,
                    summary.get("short_panel_reason"),
                )
                return
        raise ValueError(
            f"Panel-size validation failed for {ranked_file}: "
            f"final_selection={final_count}, expected={expected_size}"
        )

    logging.info(
        f"✓ Strict panel-size validation passed: final_selection={final_count}, "
        f"expected={expected_size}"
    )


def count_ranked_panel_genes(
    output_dir: str, strategy: str, reduction_type: Optional[str] = None
) -> Optional[int]:
    """Return the number of final panel genes in the strategy's panel-information CSV,
    if present."""
    ranked_file = _resolve_ranked_csv_path(output_dir, strategy, reduction_type)
    if not os.path.exists(ranked_file):
        return None

    import pandas as pd

    ranked_df = pd.read_csv(ranked_file)
    return int(ranked_df["in_panel"].fillna(False).astype(bool).sum())


def load_data(input_file: str) -> AnnData:
    """
    Load AnnData from file.
    
    Parameters
    ----------
    input_file : str
        Path to .h5ad file
        
    Returns
    -------
    AnnData
        Loaded data matrix
    """
    logging.info(f"Loading data from: {input_file}")
    adata = sc.read_h5ad(input_file)
    logging.info(f"Loaded data: {adata.shape[0]} cells × {adata.shape[1]} genes")
    return adata


def _generate_celltype_class_sizes_plot(
    adata: AnnData,
    celltype_column: str,
    output_dir: str,
) -> None:
    """Plot cells-per-celltype directly from the selection pipeline's own input data.

    A property of the input dataset, not of any strategy or panel -- needs nothing but
    ``adata.obs[celltype_column]``, so this calls ``Plotting-module``'s
    ``plot_celltype_class_sizes()`` straight from here rather than via the
    Evaluation-module-preprocessing-dependent ``Analysis-scripts/pipeline/
    plot_recovar_celltype_diagnostics.py`` driver (same rationale as the final-gene
    dotplot in ``run_combination_selection.py``). Non-fatal: a plotting failure is
    logged and swallowed rather than failing the selection run.
    """
    if celltype_column not in adata.obs.columns:
        logging.warning(
            f"Skipping celltype class-sizes plot: column '{celltype_column}' "
            f"not found in adata.obs"
        )
        return
    try:
        import sys as _sys3
        from pathlib import Path as _cpath3
        _plotting_dir = str(_cpath3(__file__).parent.parent / "Plotting-module")
        if _plotting_dir not in _sys3.path:
            _sys3.path.insert(0, _plotting_dir)
        from _clustering_plots import plot_celltype_class_sizes

        os.makedirs(output_dir, exist_ok=True)
        plot_celltype_class_sizes(
            adata.obs[celltype_column].value_counts(),
            output_dir,
            dataset_name=celltype_column,
        )
        logging.info(f"✓ Saved {celltype_column}_celltype_class_sizes plot to: {output_dir}")
    except Exception as exc:  # never let a diagnostic plot fail the selection run
        logging.warning(f"celltype class-sizes plot generation failed (non-fatal): {exc}")


def run_pipeline(args: argparse.Namespace) -> GeneListBuilder:
    """
    Main pipeline orchestrator.
    
    Routes strategies to appropriate modules and runs selection.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
        
    Returns
    -------
    GeneListBuilder
        Final gene selection results
    """
    
    # Setup logging
    log_file = os.path.join(args.output_dir, f'{args.experiment_name}.log')
    setup_logging(log_file, args.log_level)

    if args.pool_size_per_celltype is None:
        args.pool_size_per_celltype_auto = True
        args.pool_size_per_celltype = resolve_pool_size_per_celltype(args.probeset_size)
    else:
        args.pool_size_per_celltype_auto = False
    
    # Save parameter settings to JSON file
    save_parameters_to_file(args, args.output_dir)
    
    logging.info("=" * 80)
    logging.info("GENE SELECTION PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Strategy: {args.strategy}")
    logging.info(f"Target panel size: {args.probeset_size} genes")
    logging.info(
        "Pool size per celltype: %d%s",
        args.pool_size_per_celltype,
        " (auto)" if args.pool_size_per_celltype_auto else "",
    )
    logging.info(f"Output directory: {args.output_dir}")
    
    # Validate arguments
    validate_arguments(args)

    # Default NMF model cache dir to output_dir/nmf_models so models are always saved
    if args.nmf_model_cache_dir is None:
        args.nmf_model_cache_dir = os.path.join(args.output_dir, 'nmf_models')
        logging.info(f"NMF model cache dir defaulting to: {args.nmf_model_cache_dir}")

    # Load data. Keep adata.X unchanged as the preprocessed/lognorm matrix for
    # non-dimred selection methods and Xenium filtering. Dimred methods resolve
    # raw/lognorm input internally via args.dimred_counts_input.
    adata = load_data(args.input_file)
    logging.info("Selection matrix: using preprocessed/lognorm adata.X for non-dimred methods and Xenium filtering")
    logging.info(f"Dimred matrix input: {args.dimred_counts_input}")

    if args.celltype_class_sizes_dir:
        _generate_celltype_class_sizes_plot(
            adata=adata,
            celltype_column=args.celltype_column,
            output_dir=args.celltype_class_sizes_dir,
        )
    else:
        logging.info(
            "Skipping celltype class-sizes plot: no --celltype_class_sizes_dir given "
            "(this plot is opt-in only -- it is never written into --output_dir)"
        )

    # Apply blacklist filter ONCE here, before any strategy runs.
    # All strategies receive the already-filtered adata; blacklist params passed
    # downstream are set to None/False so the filter is not applied a second time.
    logging.info("")
    logging.info("=" * 80)
    logging.info("STEP 1: PRE-SELECTION BLACKLIST FILTER")
    logging.info("=" * 80)
    _needs_blacklist = args.blacklist_patterns or (not args.disable_default_blacklist)
    if _needs_blacklist:
        _n_genes_before = adata.n_vars
        _force_include_for_bl = list(args.force_include_genes) if args.force_include_genes else None
        _filtered_genes, _removed_genes, _ = apply_blacklist_filter(
            gene_list=adata.var_names.tolist(),
            blacklist_patterns=args.blacklist_patterns,
            use_default_blacklist=not args.disable_default_blacklist,
            force_include_genes=_force_include_for_bl,
        )
        if _removed_genes:
            adata = adata[:, _filtered_genes].copy()
            logging.info(
                f"Blacklist applied once (pipeline): removed {len(_removed_genes)} genes "
                f"({_n_genes_before} → {adata.n_vars} remaining). "
                f"All strategies will use this filtered gene set."
            )
        else:
            logging.info("Blacklist applied (pipeline): no genes matched patterns")
    else:
        logging.info("Blacklist filtering disabled — skipping")

    # Compute mean expression for Xenium filter
    # - mean_expr_per_ct: per-celltype means, used for dimred Phase 1 pool filtering
    # - global_mean_expr: global mean across all cells, used for post-selection
    #   filter when xenium_celltype_aware=False
    mean_expr_per_ct = None
    global_mean_expr = None
    if not args.disable_xenium_filter:
        import numpy as np
        import scipy.sparse
        celltype_col = args.celltype_column
        if celltype_col in adata.obs.columns:
            logging.info("Computing mean expression per celltype for Xenium filter...")
            mean_expr_per_ct = {}
            for ct in adata.obs[celltype_col].unique():
                mask = (adata.obs[celltype_col] == ct).values
                ct_data = adata.X[mask, :]
                if scipy.sparse.issparse(ct_data):
                    ct_means = np.array(ct_data.mean(axis=0)).flatten()
                else:
                    ct_means = np.array(ct_data).mean(axis=0)
                mean_expr_per_ct[ct] = dict(zip(adata.var_names, ct_means.tolist()))
            logging.info(f"✓ Computed mean expression for {len(mean_expr_per_ct)} celltypes")
        else:
            logging.warning(
                f"Celltype column '{celltype_col}' not found in adata.obs; "
                f"Xenium filter for dimred Phase 1 will be skipped"
            )
        if args.disable_xenium_celltype_aware:
            logging.info("Computing global mean expression for post-selection Xenium filter...")
            global_mean_expr = compute_global_mean_expression(adata)
            logging.info(f"✓ Computed global mean expression for {len(global_mean_expr)} genes")
    common_params = {
        'adata': adata,
        'probeset_size': args.probeset_size,
        'celltype_column': args.celltype_column,
        'min_cells_per_celltype': args.min_cells_per_celltype,
        'random_state': args.random_state,
        # Blacklist already applied above — pass disabled flags so strategies skip it
        'blacklist_patterns': None,
        'use_default_blacklist': False,
        'force_include_genes': args.force_include_genes,
        'apply_xenium_filter': not args.disable_xenium_filter,
        'xenium_celltype_aware': not args.disable_xenium_celltype_aware,
        'xenium_min_expr': args.xenium_min_expr,
        'xenium_max_expr': args.xenium_max_expr,
        'mean_expr_per_ct': mean_expr_per_ct,
        'global_mean_expr': global_mean_expr,
        'results_dir': args.output_dir,
    }
    
    # Route to appropriate module
    if args.strategy in SINGLE_STRATEGIES:
        logging.info("Routing to SINGLE strategy module")
        
        # Add dimred-specific parameters if needed
        if args.strategy == 'dimred_only':
            common_params.update({
                'reduction_type': args.reduction_type,
                'n_components': args.n_components,
                'pool_size_per_celltype': args.pool_size_per_celltype,
                'dimred_n_jobs': args.dimred_n_jobs,
                'nmf_model_cache_dir': args.nmf_model_cache_dir,
                'dimred_counts_input': args.dimred_counts_input,
                'require_nmf_model_cache': args.require_nmf_model_cache,
                'nmf_objective': args.nmf_objective,
            })

        result = run_single_selection(
            strategy=args.strategy,
            rf_score_cache_file=args.rf_score_cache_file,
            **common_params
        )

    elif args.strategy in COMBINATION_STRATEGIES:
        logging.info("Routing to COMBINATION strategy module")

        result = run_combination_selection(
            strategy=args.strategy,
            reduction_type=args.reduction_type,
            n_components=args.n_components,
            pool_size_per_celltype=args.pool_size_per_celltype,
            dimred_n_jobs=args.dimred_n_jobs,
            rf_percentage=args.rf_percentage,
            dimred_percentage=args.dimred_percentage,
            rf_deg_cache_dir=args.rf_deg_cache_dir,
            dimred_cache_dir=args.dimred_cache_dir,
            nmf_model_cache_dir=args.nmf_model_cache_dir,
            dimred_counts_input=args.dimred_counts_input,
            nmf_objective=args.nmf_objective,
            force_recompute=args.force_recompute,
            run_celltype_filling=not args.disable_celltype_filling,
            run_deg_filling=not args.disable_deg_filling,
            experiment_name=args.experiment_name,
            dotplot_dir=args.dotplot_dir,
            expression_distribution_dir=args.expression_distribution_dir,
            **common_params
        )
    
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    update_selection_parameters_with_panel_metadata(args.output_dir, result.metadata)

    if args.strict_panel_size:
        validate_ranked_panel_size(
            args.output_dir, args.probeset_size, args.strategy, args.reduction_type
        )
    
    logging.info("=" * 80)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY")
    logging.info("=" * 80)
    
    return result


def main() -> int:
    """Main entry point."""
    try:
        args = parse_arguments()
        result = run_pipeline(args)
        
        # Print summary
        final_count = count_ranked_panel_genes(args.output_dir, args.strategy, args.reduction_type)
        if final_count is None:
            final_count = len(result.get_selected_genes('final'))
        print(f"\n{'='*80}")
        print(f"SUCCESS: Selected {final_count} genes")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*80}\n")
        
        return 0
    
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
