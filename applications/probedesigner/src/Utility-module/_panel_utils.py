"""
Panel selection and gene provenance tracking utilities.

This module provides utilities for managing gene panel caching, loading preprocessed
data, handling logging/metadata for the gene selection pipeline, and tracking gene
origin and filtering history through the multi-stage selection and filtering pipeline.
"""

from __future__ import annotations

import os
import sys
import logging
import psutil
import pandas as pd
import scanpy as sc
from datetime import datetime as dt
from typing import Dict, List, Optional
import json

__all__ = [
    'save_experiment_parameters',
    'setup_logging',
    'log_memory_usage',
    'get_preprocessing_name',
    'get_panel_cache_prefix',
    'load_preprocessed_data',
    'get_component_panel_path',
    'load_cached_panel',
    'save_panel_to_cache',
    'save_gene_details',
    'GeneProvenanceTracker',
]

logger = logging.getLogger(__name__)


##############################################################################
# SAVE EXPERIMENT SETTINGS FOR REPRODUCIBILITY
##############################################################################


def save_experiment_parameters(args, output_dir: str, adata_shape: tuple) -> str:
    """
    Save all experiment parameters to a JSON file for reproducibility.

    Args:
        args: Command line arguments.
        output_dir: Base output directory where the parameters file will be saved.
        adata_shape: Shape of the data (n_cells, n_genes).

    Returns:
        Path to the saved parameters file.
    """
    import datetime

    params = {
        'experiment_info': {
            'experiment_name': args.experiment_name,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_shape': {
                'n_cells': int(adata_shape[0]),
                'n_genes': int(adata_shape[1])
            }
        },
        'strategy': {
            'name': args.strategy,
            'n_top_genes': args.n_top_genes,
            'random_state': args.random_state
        },
        'preprocessing': {
            'filter_method': args.filter_method,
            'subset_to_hvg': args.subset_to_hvg,
            'use_preprocessed': args.use_preprocessed
        },
        'cell_type_settings': {
            'celltype_column': args.celltype,
            'min_cells_per_celltype': args.min_cells_per_celltype
        },
        'post_selection_filtering': {
            'order': 'Blacklist → Xenium (automatic) → ODT (if enabled)',
            'blacklist': {
                'disable_default': args.disable_default_blacklist,
                'custom_patterns': args.gene_blacklist if args.gene_blacklist else []
            },
            'force_include': {
                'genes': args.force_include_genes if args.force_include_genes else []
            },
            'xenium_filtering': {
                'applied': 'automatic',
                'description': 'Per-cell-type expression filtering (0.1-100 mean expr)'
            },
            'odt_filtering': {
                'enabled': args.run_odt_filtering if hasattr(args, 'run_odt_filtering') else False
            }
        }
    }

    # Add ODT-specific parameters if enabled
    if hasattr(args, 'run_odt_filtering') and args.run_odt_filtering:
        params['post_selection_filtering']['odt_filtering'].update({
            'method': args.odt_method if hasattr(args, 'odt_method') else 'SCRINSHOT',
            'min_probes_threshold': args.odt_min_probes_threshold if hasattr(args, 'odt_min_probes_threshold') else 3,
            'genome_file': args.genome_file if hasattr(args, 'genome_file') and args.genome_file else 'auto-download',
            'gtf_file': args.gtf_file if hasattr(args, 'gtf_file') and args.gtf_file else 'auto-download'
        })

    # Add strategy-specific parameters
    if args.strategy in ['dimred_only', 'dt_pca', 'dt_nmf']:
        params['strategy']['dimensionality_reduction'] = {
            'reduction_type': args.reduction_type,
            'analysis_type': args.analysis_type,
            'dimred_method': args.dimred_method,
            'n_components': args.n_components
        }

    if args.strategy in ['dt_pca', 'dt_nmf']:
        params['strategy']['hybrid_composition'] = {
            'dt_percentage': float(args.dt_percentage),
            'dimred_percentage': float(args.dimred_percentage)
        }
        params['strategy']['gap_filling'] = {
            'celltype_specific_filling': args.run_celltype_specific_filling if hasattr(args, 'run_celltype_specific_filling') else False,
            'global_gene_filling': args.run_global_gene_filling if hasattr(args, 'run_global_gene_filling') else False,
            'deg_based_filling': args.run_deg_based_filling if hasattr(args, 'run_deg_based_filling') else False
        }

    if args.strategy == 'random':
        params['strategy']['random_sampling'] = {
            'n_bootstrap': args.random_n_bootstrap
        }

    # Add batch analysis info if applicable
    if hasattr(args, 'batch_column') and args.batch_column:
        params['batch_analysis'] = {
            'enabled': True,
            'batch_column': args.batch_column
        }
    else:
        params['batch_analysis'] = {'enabled': False}

    # Save to JSON file at the base output directory level
    params_file = os.path.join(output_dir, 'experiment_parameters.json')
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)

    logger.info(f"Saved experiment parameters to: {params_file}")

    return params_file


##############################################################################
# LOGGING UTILITIES
##############################################################################


def setup_logging(log_file: str | None = None, log_level: int = logging.INFO) -> None:
    """
    Configure logging with both file and console outputs.

    Args:
        log_file: Path to the log file.
        log_level: Logging level (default: logging.INFO).
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Reset root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.root.setLevel(log_level)
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")

    logging.info(f"Starting gene selection at {dt.now().strftime(date_format)}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Working directory: {os.getcwd()}")


def log_memory_usage(label: str) -> None:
    """
    Log memory usage at a specific point.

    Args:
        label: Label for the log entry.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_mb = memory_info.rss / (1024 * 1024)
    logger.info(f"Memory usage ({label}): {rss_mb:.2f} MB")


##############################################################################
# PREPROCESSING PATH UTILITIES
##############################################################################


def get_preprocessing_name(filter_method: str, subset_to_hvg: bool) -> str:
    """
    Generate standardized preprocessing name for PREPROCESSED DATA directories.

    This is used for the centralized preprocessed data location:
    /Data/.../preprocessed/Scanpy-Filter_All-Genes/

    Args:
        filter_method: One of: 'scanpy', '10x', 'no_filter'.
        subset_to_hvg: Whether to subset to HVGs.

    Returns:
        Preprocessing name with underscore (e.g., 'Scanpy-Filter_All-Genes').
    """
    filter_names = {
        "scanpy": "Scanpy-Filter",
        "10x": "Xenium-Filter",
        "no_filter": "No-Filter"
    }

    filter_part = filter_names.get(filter_method, "Unknown-Filter")
    hvg_part = "HVG" if subset_to_hvg else "All-Genes"

    return f"{filter_part}_{hvg_part}"


def get_panel_cache_prefix(filter_method: str, subset_to_hvg: bool) -> str:
    """
    Generate preprocessing path prefix for PANEL CACHE directories.

    This is used for the panel cache location in experiments directory:
    /Experiments/.../Selected-panels/Scanpy-Filter/All-Genes/strategy/

    Args:
        filter_method: One of: 'scanpy', '10x', 'no_filter'.
        subset_to_hvg: Whether to subset to HVGs.

    Returns:
        Path prefix with slash separator (e.g., 'Scanpy-Filter/All-Genes').
    """
    filter_names = {
        "scanpy": "Scanpy-Filter",
        "10x": "Xenium-Filter",
        "no_filter": "No-Filter"
    }

    filter_part = filter_names.get(filter_method, "Unknown-Filter")
    hvg_part = "HVG-Subset" if subset_to_hvg else "All-Genes"

    return f"{filter_part}/{hvg_part}"


##############################################################################
# DATA LOADING UTILITIES
##############################################################################


def load_preprocessed_data(
        preprocessed_dir: str,
        filter_method: str,
        subset_to_hvg: bool,
        load_global_loadings: bool = True) -> tuple:
    """
    Load preprocessed data from central directory.

    Args:
        preprocessed_dir: Base directory containing preprocessed datasets.
        filter_method: Filter method: 'scanpy', '10x', 'no_filter'.
        subset_to_hvg: Whether to use HVG subset.
        load_global_loadings: Whether to load global PCA/NMF loadings
            (default: True). Set to False for per_celltype strategies.

    Returns:
        Tuple of (adata, pca_df, nmf_df). pca_df and nmf_df will be None
        if load_global_loadings=False.

    Raises:
        FileNotFoundError: If preprocessed data directory or required files not found.
    """
    # Get preprocessing name
    preproc_name = get_preprocessing_name(filter_method, subset_to_hvg)
    preproc_path = os.path.join(preprocessed_dir, preproc_name)

    logger.info(f"Loading preprocessed data: {preproc_name}")
    logger.info(f"Path: {preproc_path}")

    # Check if directory exists
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(
            f"Preprocessed data not found: {preproc_path}\n"
            f"Please run the preprocessing script first:\n"
            f"  sbatch 20251010_Preprocess-Data.sh"
        )

    # Load AnnData
    adata_path = os.path.join(preproc_path, "preprocessed.h5ad")
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")

    logger.info(f"Loading AnnData from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    logger.info(f"Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # Conditionally load global PCA and NMF loadings
    pca_df = None
    nmf_df = None

    if load_global_loadings:
        # Load PCA loadings
        pca_path = os.path.join(preproc_path, "pca_loadings.csv")
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA loadings not found: {pca_path}")

        logger.info(f"Loading PCA loadings from: {pca_path}")
        pca_df = pd.read_csv(pca_path, index_col=0)
        logger.info(f"PCA loadings: {pca_df.shape[0]:,} genes × {pca_df.shape[1]} components")

        # Load NMF loadings
        nmf_path = os.path.join(preproc_path, "nmf_loadings.csv")
        if not os.path.exists(nmf_path):
            raise FileNotFoundError(f"NMF loadings not found: {nmf_path}")

        logger.info(f"Loading NMF loadings from: {nmf_path}")
        nmf_df = pd.read_csv(nmf_path, index_col=0)
        logger.info(f"NMF loadings: {nmf_df.shape[0]:,} genes × {nmf_df.shape[1]} components")
    else:
        logger.info("Skipping global PCA/NMF loadings (not needed for per_celltype strategy)")

    logger.info("Preprocessed data loaded successfully")

    return adata, pca_df, nmf_df


##############################################################################
# PANEL CACHING UTILITIES
##############################################################################


def get_component_panel_path(
        base_output_dir: str,
        preprocessing_name: str,
        strategy_name: str,
        probeset_size: int,
        component_params: Dict | None = None) -> str:
    """
    Generate standardized path for a component gene panel.

    This creates a predictable location where individual strategy panels are saved
    so they can be reused by combination strategies.

    Args:
        base_output_dir: Base experiments directory (e.g., .../Selected-panels/).
            Should be the root Selected-panels directory, NOT the strategy-specific
            subdirectory.
        preprocessing_name: Preprocessing identifier (e.g., 'Scanpy-Filter/All-Genes').
        strategy_name: Strategy name (e.g., 'dt_deg', 'dimred_only').
        probeset_size: Target number of genes.
        component_params: Additional parameters for the strategy:
            - 'reduction_type': 'pca' or 'nmf'
            - 'analysis_type': 'global' or 'per_celltype'
            - 'dimred_method': 'method_a' or 'method_b'.

    Returns:
        Path to the gene list CSV file.

    Raises:
        ValueError: If unsupported strategy provided.
    """
    # Build strategy-specific directory path
    if strategy_name in ['dt_deg', 'dt_simple']:
        # Decision tree strategies
        strategy_dir = os.path.join(base_output_dir, preprocessing_name, strategy_name,
                                    f"{probeset_size}-genes")

    elif strategy_name == 'deg_only':
        # DEG-only strategy
        strategy_dir = os.path.join(base_output_dir, preprocessing_name, strategy_name,
                                    f"{probeset_size}-genes")

    elif strategy_name == 'dimred_only' and component_params:
        # Dimensionality reduction only
        reduction_type = component_params.get('reduction_type', 'nmf')
        analysis_type = component_params.get('analysis_type', 'global')
        dimred_method = component_params.get('dimred_method', 'method_a')

        subdir = f"{reduction_type}_{analysis_type}_{dimred_method}"
        strategy_dir = os.path.join(base_output_dir, preprocessing_name, subdir,
                                    f"{probeset_size}-genes")
    else:
        raise ValueError(f"Unsupported strategy for panel caching: {strategy_name}")

    # Standardized filename
    gene_list_path = os.path.join(strategy_dir, "results", "selected_genes.csv")

    return gene_list_path


def load_cached_panel(
        panel_path: str,
        expected_size: int | None = None,
        filter_method: str | None = None,
        subset_to_hvg: bool | None = None) -> list | None:
    """
    Load a previously generated gene panel from cache with validation.

    Args:
        panel_path: Path to the cached gene list CSV file.
        expected_size: Expected number of genes for validation. If None,
            size is not validated.
        filter_method: Expected filter method ('scanpy', '10x', 'no_filter')
            for validation.
        subset_to_hvg: Expected HVG subsetting setting (True/False) for validation.

    Returns:
        List of gene names, or None if panel doesn't exist or validation fails.
    """
    if not os.path.exists(panel_path):
        logger.debug(f"Cached panel not found: {panel_path}")
        return None

    try:
        # Extract preprocessing settings from path for validation
        path_parts = panel_path.split(os.sep)

        # Validate filter method if provided
        if filter_method is not None:
            filter_name_map = {
                'scanpy': 'Scanpy-Filter',
                '10x': 'Xenium-Filter',
                'no_filter': 'No-Filter'
            }
            expected_filter = filter_name_map.get(filter_method)

            if expected_filter and expected_filter not in path_parts:
                logger.warning(f"Cached panel filter method mismatch:")
                logger.warning(f"  Expected: {expected_filter}, Panel path: {panel_path}")
                logger.warning(f"  Will regenerate panel with correct filter method")
                return None

        # Validate HVG subsetting if provided
        if subset_to_hvg is not None:
            expected_hvg = 'HVG-Subset' if subset_to_hvg else 'All-Genes'

            if expected_hvg not in path_parts:
                logger.warning(f"Cached panel HVG setting mismatch:")
                logger.warning(f"  Expected: {expected_hvg}, Panel path: {panel_path}")
                logger.warning(f"  Will regenerate panel with correct geneset")
                return None

        # Try to load the CSV
        df = pd.read_csv(panel_path)

        # Handle different possible column names
        if 'gene' in df.columns:
            genes = df['gene'].tolist()
        elif 'genes' in df.columns:
            genes = df['genes'].tolist()
        elif len(df.columns) == 1:
            genes = df.iloc[:, 0].tolist()
        else:
            logger.warning(f"Unexpected CSV format in {panel_path}, columns: {df.columns.tolist()}")
            return None

        # Validate size if expected
        if expected_size is not None and len(genes) != expected_size:
            logger.warning(f"Cached panel size mismatch:")
            logger.warning(f"  Expected: {expected_size} genes, Got: {len(genes)} genes")
            logger.warning(f"  Using cached panel anyway (size validation disabled)")

        logger.info(f"Loaded cached panel: {len(genes)} genes from {os.path.basename(os.path.dirname(os.path.dirname(panel_path)))}")
        return genes

    except Exception as e:
        logger.warning(f"Failed to load cached panel from {panel_path}: {e}")
        return None


def save_panel_to_cache(genes: list, panel_path: str) -> None:
    """
    Save a gene panel to cache for reuse by combination strategies.

    Args:
        genes: List of gene names.
        panel_path: Path where to save the gene list.
    """
    try:
        os.makedirs(os.path.dirname(panel_path), exist_ok=True)

        df = pd.DataFrame({'gene': genes})
        df.to_csv(panel_path, index=False)

        logger.info(f"Saved panel to cache: {len(genes)} genes -> {os.path.basename(panel_path)}")

    except Exception as e:
        logger.warning(f"Failed to save panel to cache at {panel_path}: {e}")


##############################################################################
# GENE METADATA UTILITIES
##############################################################################


def save_gene_details(
        final_genes: list,
        results: dict,
        args,
        results_dir: str) -> str | None:
    """
    Save detailed metadata about selected genes based on the strategy used.

    Args:
        final_genes: List of final selected gene names (after filtering).
        results: Results dictionary from the selection strategy.
        args: Command line arguments.
        results_dir: Directory to save the gene details file.

    Returns:
        Path to saved gene details file, or None if not applicable.
    """
    try:
        gene_details = []

        # Extract gene details based on strategy
        if args.strategy == 'dimred_only':
            # For dimred strategies, check if gene_details already exist
            if 'gene_details' in results and results['gene_details']:
                # Use existing detailed metadata from select_genes_from_components
                for detail in results['gene_details']:
                    if detail['gene'] in final_genes:
                        gene_details.append(detail)
            else:
                # Fallback: create minimal details
                gene_weights = results.get('gene_weights', {})
                for gene in final_genes:
                    if gene in gene_weights:
                        weights = gene_weights[gene]
                        best_comp = max(weights.keys(), key=lambda c: abs(weights[c]))
                        gene_details.append({
                            'gene': gene,
                            'best_component': best_comp,
                            'best_weight': weights[best_comp]
                        })
                    else:
                        gene_details.append({'gene': gene})

        elif args.strategy in ['dt_deg', 'dt_simple']:
            # For decision tree strategies, use consensus gene scores
            consensus_genes_df = None
            if 'selection_details' in results and 'model_dir' in results.get('selection_details', {}):
                model_dir = results['selection_details']['model_dir']
                consensus_path = os.path.join(model_dir, 'consensus_genes.csv')
                if os.path.exists(consensus_path):
                    try:
                        consensus_genes_df = pd.read_csv(consensus_path)
                        consensus_genes_df = consensus_genes_df.set_index('gene')
                    except Exception as e:
                        logger.warning(f"Could not load consensus genes: {e}")

            if consensus_genes_df is not None:
                for gene in final_genes:
                    if gene in consensus_genes_df.index:
                        row = consensus_genes_df.loc[gene]
                        gene_details.append({
                            'gene': gene,
                            'final_score': row.get('final_score', 0),
                            'avg_importance': row.get('avg_importance', 0),
                            'max_importance': row.get('max_importance', 0),
                            'avg_f1_score': row.get('avg_f1_score', 0)
                        })
                    else:
                        gene_details.append({'gene': gene})
            else:
                # Fallback: try to use gene_scores from results
                gene_scores = results.get('gene_scores', {})
                for gene in final_genes:
                    if gene in gene_scores:
                        gene_details.append({
                            'gene': gene,
                            'score': gene_scores[gene]
                        })
                    else:
                        gene_details.append({'gene': gene})

        elif args.strategy == 'deg_only':
            # For DEG strategy, try to load DEG results from file
            deg_df = None

            # First check if it's in results
            if 'deg_results' in results:
                deg_df = results['deg_results']
            else:
                # Try to load from results directory
                deg_file = os.path.join(results_dir, 'deg_results_all.csv')
                if os.path.exists(deg_file):
                    try:
                        deg_df = pd.read_csv(deg_file)
                    except Exception as e:
                        logger.warning(f"Could not load DEG results: {e}")

            if deg_df is not None:
                for gene in final_genes:
                    gene_rows = deg_df[deg_df['gene'] == gene]
                    if not gene_rows.empty:
                        # Get best (most significant) result for this gene
                        best_row = gene_rows.loc[gene_rows['pvals_adj'].idxmin()]
                        gene_details.append({
                            'gene': gene,
                            'best_group': best_row['group'],
                            'logfoldchange': best_row['logfoldchanges'],
                            'pval_adj': best_row['pvals_adj'],
                            'deg_score': best_row['scores']
                        })
                    else:
                        gene_details.append({'gene': gene})
            else:
                # Fallback: use gene_scores from results
                gene_scores = results.get('gene_scores', {})
                for gene in final_genes:
                    if gene in gene_scores:
                        gene_details.append({
                            'gene': gene,
                            'deg_score': gene_scores[gene]
                        })
                    else:
                        gene_details.append({'gene': gene})

        elif args.strategy in ['dt_pca', 'dt_nmf']:
            # For combination strategies, indicate source
            dt_genes_set = set(results.get('dt_genes', []))
            dimred_genes_set = set(results.get('dimred_genes', []))
            overlapping_genes_set = set(results.get('overlapping_genes', []))

            # Track gap-filling genes
            celltype_filling = set(results.get('genes_added_by_celltype_filling', []))
            global_filling = set(results.get('genes_added_by_global_filling', []))
            deg_filling = set(results.get('genes_added_by_deg_filling', []))
            gap_filling_genes = celltype_filling | global_filling | deg_filling

            for gene in final_genes:
                source = []
                is_gap_filling = False

                # Check if gene is from gap-filling
                if gene in gap_filling_genes:
                    source.append('gap_filling')
                    is_gap_filling = True

                # Check original sources (for base combination genes)
                if gene in overlapping_genes_set:
                    # Gene was in both DT and dimred
                    source.append('decision_tree')
                    source.append('dimensionality_reduction')
                elif gene in dt_genes_set:
                    source.append('decision_tree')
                elif gene in dimred_genes_set:
                    source.append('dimensionality_reduction')

                # If no source identified, it's likely a replacement gene
                if not source:
                    source.append('blacklist_replacement')

                gene_details.append({
                    'gene': gene,
                    'is_shared': 'decision_tree' in source and 'dimensionality_reduction' in source and not is_gap_filling,
                    'is_gap_filling': is_gap_filling
                })

        # Save gene details if available
        if gene_details:
            gene_details_df = pd.DataFrame(gene_details)
            details_path = os.path.join(results_dir, 'selected_genes_with_details.csv')
            gene_details_df.to_csv(details_path, index=False)
            logger.info(f"Saved gene details: {details_path}")
            return details_path
        else:
            logger.warning("No gene details available to save")
            return None

    except Exception as e:
        logger.error(f"Error saving gene details: {e}")
        import traceback
        traceback.print_exc()
        return None


##############################################################################
# GENE PROVENANCE TRACKING
##############################################################################


class GeneProvenanceTracker:
    """
    Track gene provenance through the entire selection and filtering pipeline.

    This class maintains a detailed history of each gene's journey:
    - Where it came from (strategy-specific source)
    - Which filters it passed/failed
    - What it replaced (if it's a replacement gene)
    - What replaced it (if it was removed).
    """

    def __init__(self, initial_genes: List[str], strategy_name: str):
        """
        Initialize tracker with initial gene selection.

        Args:
            initial_genes: Initial selected genes before any filtering.
            strategy_name: Name of the selection strategy used.
        """
        self.strategy_name = strategy_name

        # Initialize tracking dataframe
        self.provenance_df = pd.DataFrame({
            'gene': initial_genes,
            'selection_rank': range(1, len(initial_genes) + 1),
            'initial_source': 'initial_selection',
            'source_details': strategy_name,
            'blacklist_status': 'passed',
            'blacklist_replacement_for': None,
            'xenium_status': 'not_tested',
            'xenium_replacement_for': None,
            'odt_status': 'not_tested',
            'odt_replacement_for': None,
            'final_status': 'selected',
            'removal_reason': None,
            'replacement_iteration': None
        })

        self.current_genes = set(initial_genes)

        logger.info(f"Initialized provenance tracker for {len(initial_genes)} genes")

    def set_source_details(self, gene_source_mapping: Dict[str, str]) -> None:
        """
        Set detailed source information for genes.

        Args:
            gene_source_mapping: Maps gene name to detailed source string
                (e.g., 'DT', 'dimred', 'NMF_factor_3').
        """
        for gene, source in gene_source_mapping.items():
            mask = self.provenance_df['gene'] == gene
            if mask.any():
                self.provenance_df.loc[mask, 'source_details'] = source

        logger.info(f"Updated source details for {len(gene_source_mapping)} genes")

    def set_celltype_mapping(self, gene_celltype_mapping: Dict[str, str]) -> None:
        """
        Add cell type information for per-celltype strategies.

        Args:
            gene_celltype_mapping: Maps gene name to cell type.
        """
        # Add celltype column if not exists
        if 'celltype' not in self.provenance_df.columns:
            self.provenance_df['celltype'] = None

        for gene, celltype in gene_celltype_mapping.items():
            mask = self.provenance_df['gene'] == gene
            if mask.any():
                self.provenance_df.loc[mask, 'celltype'] = celltype

        logger.info(f"Updated celltype mapping for {len(gene_celltype_mapping)} genes")

    def record_blacklist_filtering(
            self,
            removed_genes: List[str],
            replacement_mapping: Dict[str, str],
            replacement_source_mapping: Dict[str, str] | None = None) -> None:
        """
        Record results of blacklist filtering stage.

        Args:
            removed_genes: Genes removed by blacklist filter.
            replacement_mapping: Maps replacement_gene -> removed_gene.
            replacement_source_mapping: Maps replacement_gene -> source
                (e.g., 'DT', 'dimred').
        """
        # Mark removed genes
        for gene in removed_genes:
            mask = self.provenance_df['gene'] == gene
            if mask.any():
                self.provenance_df.loc[mask, 'blacklist_status'] = 'failed'
                self.provenance_df.loc[mask, 'final_status'] = 'removed'
                self.provenance_df.loc[mask, 'removal_reason'] = 'blacklist_filter'
                self.current_genes.discard(gene)

        # Add replacement genes
        for replacement_gene, removed_gene in replacement_mapping.items():
            # Create new row for replacement
            new_row = {
                'gene': replacement_gene,
                'selection_rank': None,
                'initial_source': 'blacklist_replacement',
                'source_details': replacement_source_mapping.get(replacement_gene, 'unknown') if replacement_source_mapping else 'unknown',
                'blacklist_status': 'passed',
                'blacklist_replacement_for': removed_gene,
                'xenium_status': 'not_tested',
                'xenium_replacement_for': None,
                'odt_status': 'not_tested',
                'odt_replacement_for': None,
                'final_status': 'selected',
                'removal_reason': None,
                'replacement_iteration': None
            }

            # Copy celltype from removed gene if available
            if 'celltype' in self.provenance_df.columns:
                removed_mask = self.provenance_df['gene'] == removed_gene
                if removed_mask.any():
                    removed_celltype = self.provenance_df.loc[removed_mask, 'celltype'].iloc[0]
                    new_row['celltype'] = removed_celltype

            # Add to dataframe
            self.provenance_df = pd.concat([
                self.provenance_df,
                pd.DataFrame([new_row])
            ], ignore_index=True)

            self.current_genes.add(replacement_gene)

        logger.info(f"Blacklist filtering: {len(removed_genes)} removed, {len(replacement_mapping)} replaced")

    def record_force_include(
            self,
            force_included_genes: List[str],
            replaced_genes: List[str]) -> None:
        """
        Record force-included genes and what they replaced.

        Args:
            force_included_genes: Genes that were force-included.
            replaced_genes: Genes that were removed to make room.
        """
        # Mark replaced genes
        for gene in replaced_genes:
            mask = self.provenance_df['gene'] == gene
            if mask.any():
                self.provenance_df.loc[mask, 'final_status'] = 'removed'
                self.provenance_df.loc[mask, 'removal_reason'] = 'replaced_by_force_include'
                self.current_genes.discard(gene)

        # Add force-included genes (or mark if already present)
        for gene in force_included_genes:
            mask = self.provenance_df['gene'] == gene
            if mask.any():
                # Gene already exists, mark as force-included
                self.provenance_df.loc[mask, 'initial_source'] = 'force_include'
            else:
                # Add new row
                new_row = {
                    'gene': gene,
                    'selection_rank': None,
                    'initial_source': 'force_include',
                    'source_details': 'user_specified',
                    'blacklist_status': 'exempt',
                    'blacklist_replacement_for': None,
                    'xenium_status': 'not_tested',
                    'xenium_replacement_for': None,
                    'odt_status': 'not_tested',
                    'odt_replacement_for': None,
                    'final_status': 'selected',
                    'removal_reason': None,
                    'replacement_iteration': None
                }

                self.provenance_df = pd.concat([
                    self.provenance_df,
                    pd.DataFrame([new_row])
                ], ignore_index=True)

                self.current_genes.add(gene)

        logger.info(f"Force-include: {len(force_included_genes)} added, {len(replaced_genes)} removed")

    def record_xenium_filtering(
            self,
            removed_genes: List[str],
            replacement_mapping: Dict[str, str],
            iteration: int = 1) -> None:
        """
        Record results of Xenium expression filtering stage.

        Args:
            removed_genes: Genes removed by Xenium filter.
            replacement_mapping: Maps replacement_gene -> removed_gene.
            iteration: Iteration number (for iterative filtering).
        """
        # Mark removed genes as passed Xenium initially
        for gene in self.current_genes:
            mask = self.provenance_df['gene'] == gene
            if mask.any() and self.provenance_df.loc[mask, 'xenium_status'].iloc[0] == 'not_tested':
                self.provenance_df.loc[mask, 'xenium_status'] = 'passed'

        # Mark removed genes
        for gene in removed_genes:
            mask = self.provenance_df['gene'] == gene
            if mask.any():
                self.provenance_df.loc[mask, 'xenium_status'] = 'failed'
                self.provenance_df.loc[mask, 'final_status'] = 'removed'
                self.provenance_df.loc[mask, 'removal_reason'] = 'xenium_expression_filter'
                self.current_genes.discard(gene)

        # Add replacement genes
        for replacement_gene, removed_gene in replacement_mapping.items():
            new_row = {
                'gene': replacement_gene,
                'selection_rank': None,
                'initial_source': 'xenium_replacement',
                'source_details': 'expression_filtered',
                'blacklist_status': 'not_tested',
                'blacklist_replacement_for': None,
                'xenium_status': 'passed',
                'xenium_replacement_for': removed_gene,
                'odt_status': 'not_tested',
                'odt_replacement_for': None,
                'final_status': 'selected',
                'removal_reason': None,
                'replacement_iteration': iteration
            }

            # Copy celltype from removed gene if available
            if 'celltype' in self.provenance_df.columns:
                removed_mask = self.provenance_df['gene'] == removed_gene
                if removed_mask.any():
                    removed_celltype = self.provenance_df.loc[removed_mask, 'celltype'].iloc[0]
                    new_row['celltype'] = removed_celltype

            self.provenance_df = pd.concat([
                self.provenance_df,
                pd.DataFrame([new_row])
            ], ignore_index=True)

            self.current_genes.add(replacement_gene)

        logger.info(f"Xenium filtering (iter {iteration}): {len(removed_genes)} removed, {len(replacement_mapping)} replaced")

    def record_odt_filtering(
            self,
            removed_genes: List[str],
            replacement_mapping: Dict[str, str],
            iteration: int = 1) -> None:
        """
        Record results of ODT designability filtering stage.

        Args:
            removed_genes: Genes removed by ODT filter.
            replacement_mapping: Maps replacement_gene -> removed_gene.
            iteration: Iteration number (for iterative filtering).
        """
        # Mark tested genes as passed ODT initially
        for gene in self.current_genes:
            mask = self.provenance_df['gene'] == gene
            if mask.any() and self.provenance_df.loc[mask, 'odt_status'].iloc[0] == 'not_tested':
                self.provenance_df.loc[mask, 'odt_status'] = 'passed'

        # Mark removed genes
        for gene in removed_genes:
            mask = self.provenance_df['gene'] == gene
            if mask.any():
                self.provenance_df.loc[mask, 'odt_status'] = 'failed'
                self.provenance_df.loc[mask, 'final_status'] = 'removed'
                self.provenance_df.loc[mask, 'removal_reason'] = 'odt_designability_filter'
                self.current_genes.discard(gene)

        # Add replacement genes
        for replacement_gene, removed_gene in replacement_mapping.items():
            new_row = {
                'gene': replacement_gene,
                'selection_rank': None,
                'initial_source': 'odt_replacement',
                'source_details': 'designability_filtered',
                'blacklist_status': 'not_tested',
                'blacklist_replacement_for': None,
                'xenium_status': 'not_tested',
                'xenium_replacement_for': None,
                'odt_status': 'passed',
                'odt_replacement_for': removed_gene,
                'final_status': 'selected',
                'removal_reason': None,
                'replacement_iteration': iteration
            }

            # Copy celltype from removed gene if available
            if 'celltype' in self.provenance_df.columns:
                removed_mask = self.provenance_df['gene'] == removed_gene
                if removed_mask.any():
                    removed_celltype = self.provenance_df.loc[removed_mask, 'celltype'].iloc[0]
                    new_row['celltype'] = removed_celltype

            self.provenance_df = pd.concat([
                self.provenance_df,
                pd.DataFrame([new_row])
            ], ignore_index=True)

            self.current_genes.add(replacement_gene)

        logger.info(f"ODT filtering (iter {iteration}): {len(removed_genes)} removed, {len(replacement_mapping)} replaced")

    def get_final_panel_with_provenance(self) -> pd.DataFrame:
        """
        Get final gene panel with complete provenance information.

        Returns:
            Final genes with full tracking history.
        """
        final_df = self.provenance_df[
            self.provenance_df['final_status'] == 'selected'
        ].copy()

        return final_df

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of filtering pipeline.

        Returns:
            Summary statistics.
        """
        stats = {
            'initial_genes': len(self.provenance_df[self.provenance_df['initial_source'] == 'initial_selection']),
            'final_genes': len(self.provenance_df[self.provenance_df['final_status'] == 'selected']),
            'blacklist_removed': len(self.provenance_df[self.provenance_df['blacklist_status'] == 'failed']),
            'blacklist_replacements': len(self.provenance_df[self.provenance_df['initial_source'] == 'blacklist_replacement']),
            'xenium_removed': len(self.provenance_df[self.provenance_df['xenium_status'] == 'failed']),
            'xenium_replacements': len(self.provenance_df[self.provenance_df['initial_source'] == 'xenium_replacement']),
            'odt_removed': len(self.provenance_df[self.provenance_df['odt_status'] == 'failed']),
            'odt_replacements': len(self.provenance_df[self.provenance_df['initial_source'] == 'odt_replacement']),
            'force_included': len(self.provenance_df[self.provenance_df['initial_source'] == 'force_include']),
            'removed_by_force_include': len(self.provenance_df[self.provenance_df['removal_reason'] == 'replaced_by_force_include'])
        }

        return stats

    def save_provenance_report(
            self,
            results_dir: str,
            filename: str = 'gene_provenance_report.csv') -> None:
        """
        Save complete provenance report to file.

        Args:
            results_dir: Directory to save report.
            filename: Output filename.
        """
        os.makedirs(results_dir, exist_ok=True)

        # Save full provenance table
        output_path = os.path.join(results_dir, filename)
        self.provenance_df.to_csv(output_path, index=False)
        logger.info(f"Saved complete provenance report: {output_path}")

        # Save final panel with provenance
        final_path = os.path.join(results_dir, 'final_panel_with_provenance.csv')
        final_df = self.get_final_panel_with_provenance()
        final_df.to_csv(final_path, index=False)
        logger.info(f"Saved final panel with provenance: {final_path}")

        # Save summary statistics
        stats = self.get_summary_statistics()
        stats_path = os.path.join(results_dir, 'provenance_summary_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved summary statistics: {stats_path}")

        # Print summary
        logger.info("="*80)
        logger.info("GENE PROVENANCE SUMMARY")
        logger.info("="*80)
        logger.info(f"Initial selection: {stats['initial_genes']} genes")
        logger.info(f"Blacklist filtering: -{stats['blacklist_removed']}, +{stats['blacklist_replacements']}")
        logger.info(f"Xenium filtering: -{stats['xenium_removed']}, +{stats['xenium_replacements']}")
        logger.info(f"ODT filtering: -{stats['odt_removed']}, +{stats['odt_replacements']}")
        logger.info(f"Force-included: +{stats['force_included']} (replaced {stats['removed_by_force_include']})")
        logger.info(f"Final panel: {stats['final_genes']} genes")
        logger.info("="*80)
