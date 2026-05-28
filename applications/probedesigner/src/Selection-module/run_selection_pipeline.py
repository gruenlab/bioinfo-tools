"""
Main orchestrator for gene selection pipeline.

This script routes gene selection strategies to the appropriate modules:
- Single strategies (deg_only, rf_simple, rf_deg, hvg, random, dimred_only) → run_single_selection
- Combination strategies (rf_nmf, rf_pca) → run_combination_selection

Supports caching, filtering (blacklist, Xenium), and all strategy parameters.

Usage:
    python run_selection_pipeline.py \\
        --strategy rf_nmf \\
        --input_file data.h5ad \\
        --output_dir results/ \\
        --probeset_size 100 \\
        --reduction_type nmf \\
        --analysis_type per_celltype

Written by: Helene Hemmer
Date: 2026-02-08
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
import sys
from pathlib import Path
MODULE_DIR = Path(__file__).parent.absolute()
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

# Use absolute imports (for script execution)
from _filtering import apply_blacklist_filter, compute_global_mean_expression
from _constants import (
    DEFAULT_ALL_STRATEGIES,
    DEFAULT_BLACKLIST_PATTERNS,
    DEFAULT_COMBINATION_STRATEGIES,
    DEFAULT_DIMRED_PERCENTAGE,
    DEFAULT_MIN_CELLS_PER_CELLTYPE,
    DEFAULT_PROBESET_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RF_PERCENTAGE,
    DEFAULT_SINGLE_STRATEGIES,
)
from _gene_list_builder import GeneListBuilder
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
    
    # Add metadata
    params['timestamp'] = datetime.now().isoformat()
    params['script'] = 'run_selection_pipeline.py'
    
    # Save to JSON file
    param_file = os.path.join(output_dir, 'selection_parameters.json')
    with open(param_file, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)
    
    logging.info(f"Parameter settings saved to: {param_file}")
    return param_file


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
    --reduction_type nmf \\
    --analysis_type per_celltype

  # Combination strategy (rf_nmf) with caching
  python run_selection_pipeline.py \\
    --strategy rf_nmf \\
    --input_file data.h5ad \\
    --output_dir results/rf_nmf/ \\
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
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    # =========================================================================
    # Dimensionality reduction parameters (for dimred_only, rf_nmf, rf_pca)
    # =========================================================================
    dimred = parser.add_argument_group('Dimensionality reduction parameters')
    
    dimred.add_argument(
        '--reduction_type',
        type=str,
        choices=['nmf', 'pca'],
        help='Dimensionality reduction type (required for dimred_only, rf_nmf, rf_pca)'
    )
    
    dimred.add_argument(
        '--analysis_type',
        type=str,
        default='per_celltype',
        choices=['global', 'per_celltype'],
        help='Analysis level: global or per_celltype (default: per_celltype)'
    )
    
    dimred.add_argument(
        '--dimred_method',
        type=str,
        default='method_a',
        choices=['method_a'],
        help='Dimred gene selection method (only method_a — top genes by absolute factor loading)'
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
        default=200,
        help='Pool size per celltype for Phase 1 pool creation in per_celltype mode (default: 200)'
    )
    
    dimred.add_argument(
        '--pool_size_per_factor',
        type=int,
        default=200,
        help='Pool size per factor/component for Phase 1 pool creation in global mode (default: 200)'
    )

    dimred.add_argument(
        '--nmf_model_cache_dir',
        type=str,
        default=None,
        help=(
            'Shared directory for NMF/PCA model pkl files. '
            'When set, all strategies (dimred_only, rf_nmf, rf_pca) load/save models here '
            'instead of inside the per-run output directory. '
            'Use this to share a single NMF fit across different probeset sizes or filter settings '
            '(e.g. --nmf_model_cache_dir /path/to/experiment/nmf_cache). '
            'The cache is keyed on n_components; change --n_components to force a fresh fit.'
        )
    )

    dimred.add_argument(
        '--use_consensus_nmf',
        action='store_true',
        default=False,
        help=(
            'Run Consensus NMF (cNMF) to automatically determine the optimal number of '
            'factors before the main NMF-based selection. When enabled, --n_components is '
            'used as a fallback only if cNMF fails. Requires strategies that use NMF '
            '(dimred_only with reduction_type=nmf, rf_nmf).'
        )
    )
    dimred.add_argument(
        '--k_min',
        type=int,
        default=3,
        help='Minimum number of NMF factors to test in cNMF (default: 3). Requires --use_consensus_nmf.'
    )
    dimred.add_argument(
        '--k_max',
        type=int,
        default=15,
        help='Maximum number of NMF factors to test in cNMF (default: 15). Requires --use_consensus_nmf.'
    )
    dimred.add_argument(
        '--k_step',
        type=int,
        default=2,
        help='Step size between K values tested in cNMF (default: 2). Requires --use_consensus_nmf.'
    )
    dimred.add_argument(
        '--cnmf_n_iter',
        type=int,
        default=20,
        help='Number of NMF iterations per K value in cNMF (default: 20). Requires --use_consensus_nmf.'
    )
    dimred.add_argument(
        '--k_selection_method',
        type=str,
        default='silhouette',
        choices=['silhouette', 'elbow'],
        help='Method to select optimal K from cNMF stability metrics (default: silhouette).'
    )
    dimred.add_argument(
        '--use_consensus_H',
        action='store_true',
        default=False,
        help=(
            'Use the consensus H matrix from cNMF directly for gene selection instead of '
            're-running standard NMF with the optimal K. Only relevant when --use_consensus_nmf '
            'is set. Default: re-run standard NMF with optimal K.'
        )
    )
    dimred.add_argument(
        '--nmf_counts_input',
        type=str,
        default='raw',
        choices=['raw', 'lognorm'],
        help=(
            "Count matrix used as NMF input. "
            "'raw' (default): raw integer counts from adata.raw or adata.layers['counts'] "
            "(validated with is_anndata_raw_layer). "
            "'lognorm': log-normalised counts from adata.X."
        )
    )

    # =========================================================================
    # Combination strategy parameters (for rf_nmf, rf_pca)
    # =========================================================================
    combo = parser.add_argument_group('Combination strategy parameters')
    
    combo.add_argument(
        '--rf_percentage',
        type=float,
        default=DEFAULT_RF_PERCENTAGE,
        help=f'RF percentage for combination strategies (default: {DEFAULT_RF_PERCENTAGE:.0%})'
    )
    
    combo.add_argument(
        '--dimred_percentage',
        type=float,
        default=DEFAULT_DIMRED_PERCENTAGE,
        help=f'Dimred percentage for combination strategies (default: {DEFAULT_DIMRED_PERCENTAGE:.0%})'
    )
    
    combo.add_argument(
        '--rf_deg_cache_dir',
        type=str,
        help='Directory with cached rf_deg results (for combination strategies)'
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
        '--disable_celltype_filling',
        action='store_true',
        help='Disable cell-type-specific gap-filling strategy'
    )
    
    combo.add_argument(
        '--disable_global_filling',
        action='store_true',
        help='Disable global gene gap-filling strategy'
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
        help='Disable default blacklist patterns (mt-, hsp). Only custom patterns from --blacklist_patterns will be used.'
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
    if args.strategy in ['dimred_only', 'rf_nmf', 'rf_pca']:
        if not args.reduction_type:
            raise ValueError(
                f"--reduction_type required for strategy '{args.strategy}'. "
                "Specify 'nmf' or 'pca'."
            )
        
        # Validate reduction_type matches strategy
        if 'nmf' in args.strategy and args.reduction_type != 'nmf':
            raise ValueError(
                f"Strategy '{args.strategy}' requires --reduction_type nmf, "
                f"got '{args.reduction_type}'"
            )
        if 'pca' in args.strategy and args.reduction_type != 'pca':
            raise ValueError(
                f"Strategy '{args.strategy}' requires --reduction_type pca, "
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
    
    # Check input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Check cache directories exist if provided
    if args.rf_deg_cache_dir and not os.path.exists(args.rf_deg_cache_dir):
        raise FileNotFoundError(f"RF cache directory not found: {args.rf_deg_cache_dir}")
    
    if args.dimred_cache_dir and not os.path.exists(args.dimred_cache_dir):
        raise FileNotFoundError(f"Dimred cache directory not found: {args.dimred_cache_dir}")


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
    
    # Save parameter settings to JSON file
    save_parameters_to_file(args, args.output_dir)
    
    logging.info("=" * 80)
    logging.info("GENE SELECTION PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Strategy: {args.strategy}")
    logging.info(f"Target panel size: {args.probeset_size} genes")
    logging.info(f"Output directory: {args.output_dir}")
    
    # Validate arguments
    validate_arguments(args)

    # Default NMF model cache dir to output_dir/nmf_models so models are always saved
    if args.nmf_model_cache_dir is None:
        args.nmf_model_cache_dir = os.path.join(args.output_dir, 'nmf_models')
        logging.info(f"NMF model cache dir defaulting to: {args.nmf_model_cache_dir}")

    # Load data
    adata = load_data(args.input_file)

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
                'analysis_type': args.analysis_type,
                'dimred_method': args.dimred_method,
                'n_components': args.n_components,
                'pool_size_per_celltype': args.pool_size_per_celltype,
                'pool_size_per_factor': args.pool_size_per_factor,
                'nmf_model_cache_dir': args.nmf_model_cache_dir,
                'nmf_counts_input': args.nmf_counts_input,
                'use_consensus_nmf': args.use_consensus_nmf,
                'k_min': args.k_min,
                'k_max': args.k_max,
                'k_step': args.k_step,
                'cnmf_n_iter': args.cnmf_n_iter,
                'k_selection_method': args.k_selection_method,
                'use_consensus_H': args.use_consensus_H,
            })

        result = run_single_selection(
            strategy=args.strategy,
            **common_params
        )

    elif args.strategy in COMBINATION_STRATEGIES:
        logging.info("Routing to COMBINATION strategy module")

        result = run_combination_selection(
            strategy=args.strategy,
            reduction_type=args.reduction_type,
            analysis_type=args.analysis_type,
            dimred_method=args.dimred_method,
            n_components=args.n_components,
            pool_size_per_celltype=args.pool_size_per_celltype,
            pool_size_per_factor=args.pool_size_per_factor,
            rf_percentage=args.rf_percentage,
            dimred_percentage=args.dimred_percentage,
            rf_deg_cache_dir=args.rf_deg_cache_dir,
            dimred_cache_dir=args.dimred_cache_dir,
            nmf_model_cache_dir=args.nmf_model_cache_dir,
            nmf_counts_input=args.nmf_counts_input,
            force_recompute=args.force_recompute,
            run_celltype_filling=not args.disable_celltype_filling,
            run_global_filling=not args.disable_global_filling,
            run_deg_filling=not args.disable_deg_filling,
            experiment_name=args.experiment_name,
            use_consensus_nmf=args.use_consensus_nmf,
            k_min=args.k_min,
            k_max=args.k_max,
            k_step=args.k_step,
            cnmf_n_iter=args.cnmf_n_iter,
            k_selection_method=args.k_selection_method,
            use_consensus_H=args.use_consensus_H,
            **common_params
        )
    
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
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
        final_genes = result.get_all_genes()
        print(f"\n{'='*80}")
        print(f"SUCCESS: Selected {len(final_genes)} genes")
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
