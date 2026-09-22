"""Flexible plotting script for evaluation results.

This script creates plots from pre-computed evaluation results without
re-running the evaluation pipeline. It provides maximum flexibility to
compare any combination of probe panels in visualizations.

Supports both baseline and variability evaluation result plotting.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd
import numpy as np

from _clustering_plots import (
    generate_method_specific_colors_and_markers,
    plot_clustering_quality_ari,
    plot_clustering_quality_nmi,
    plot_celltype_f1_heatmap,
    plot_celltype_classification_diagnostics_global,
    plot_neighborhood_preservation_by_k,
    plot_neighborhood_preservation_celltype_heatmap,
)
from _variability_plots import (
    get_category_colors,
    plot_celltype_evaluation_results,
    create_combined_plot_with_info,
    plot_aggregated_celltype_metrics,
    plot_expvar_gene_subset_breakdown,
    plot_expvar_gene_subset_breakdown_by_celltype,
    plot_ridge_aggregated_celltype_metrics,
)
from _reconstruction_plots import (
    plot_reconstruction_per_celltype,
    plot_reconstruction_aggregated_metrics,
)
# --- load THIS directory's _constants.py by path (sibling dirs share the name) ---
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _cpath
_cspec = _ilu.spec_from_file_location("_constants", _cpath(__file__).resolve().parent / "_constants.py")
_sys.modules["_constants"] = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_sys.modules["_constants"])


from _constants import DEFAULT_PNG_DPI, METHOD_DISPLAY_NAMES

logger = logging.getLogger(__name__)

__all__ = [
    "load_baseline_results",
    "load_variability_results",
    "main",
]

def get_color_map_for_datasets(dataset_names, user_specified_map=None):
    """
    Get color mapping for datasets. Uses user-specified map or returns None for automatic coloring.
    
    Parameters:
    -----------
    dataset_names : list
        List of dataset names
    user_specified_map : dict, optional
        User-specified color map
        
    Returns:
    --------
    dict or None
        Color mapping for datasets, or None to use automatic coloring
    """
    if user_specified_map:
        return user_specified_map
    
    # Return None to let plotting functions use their automatic color schemes
    return None


# ===================================================================
# HELPER FUNCTIONS FOR VARIABILITY PLOTTING
# ===================================================================

def prepare_global_combined_plot_data(df, group_type='nmf'):
    """
    Prepare data structure for create_combined_plot_with_info function.
    
    Extracts global evaluation metrics and organizes them by metric type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: gene_list, analysis_type, mse_test_probe, expvar_test_probe, etc.
    group_type : str
        Either 'nmf' or 'mapping'

    Returns:
    --------
    dict
        Dictionary structure: {
            f'{group_type}_mse': {method_name: value, ...},
            f'{group_type}_expvar': {method_name: value, ...}
        }
    """
    # Filter to global analysis rows only
    global_df = df[df.get('analysis_type', '') == 'global']
    
    if global_df.empty:
        logger.warning(f"No global analysis rows found for {group_type} combined plots")
        return {}
    
    # Initialize results dictionary with metric types
    results = {
        f'{group_type}_mse': {},
        f'{group_type}_expvar': {}
    }
    
    # Extract metrics for each gene list
    for _, row in global_df.iterrows():
        gene_list_name = row['gene_list']
        mse_value = row.get('mse_test_probe')
        expvar_value = row.get('expvar_test_probe')
        
        if pd.notna(mse_value):
            results[f'{group_type}_mse'][gene_list_name] = mse_value
        if pd.notna(expvar_value):
            results[f'{group_type}_expvar'][gene_list_name] = expvar_value
    
    logger.info(f"Prepared global combined plot data for {group_type}:")
    logger.info(f"  {len(results[f'{group_type}_mse'])} methods for MSE")
    logger.info(f"  {len(results[f'{group_type}_expvar'])} methods for ExpVar")
    
    return results


def convert_variability_df_to_dict(df, group_type='nmf'):
    """Convert variability results DataFrame to the nested dict format expected by plotting functions.

    Reads pre-computed macro/weighted summary statistics from the dedicated
    ``celltype="summary"`` row in the CSV (computed by the evaluation pipeline).
    Does NOT recompute aggregates from per-cell-type rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from a variability evaluation CSV.  Must have columns:
        ``gene_list``, ``analysis_type``, optionally ``celltype``, and metric
        columns (``mse_test_probe``, ``expvar_test_probe``, …).  Aggregated
        CSVs produced by k-fold evaluation append ``_mean``/``_std`` suffixes;
        this function reads both forms.
    group_type : str
        Prefix for result dict keys — either ``'nmf'`` or ``'mapping'``.

    Returns
    -------
    dict
        ``{gene_list_name: {f'{group_type}_celltype_summary': {...},
                            f'{group_type}_celltype_celltype_results': {...}}}``
    """

    def _pick(row, col):
        """Read a metric column, falling back to ``<col>_mean`` for aggregated CSVs."""
        val = row.get(col, np.nan)
        if pd.notna(val):
            return float(val)
        mean_val = row.get(f"{col}_mean", np.nan)
        return float(mean_val) if pd.notna(mean_val) else np.nan

    evaluation_results = {}

    for gene_list_name in df['gene_list'].unique():
        gene_list_df = df[df['gene_list'] == gene_list_name]
        results_dict = {}

        logger.info("Converting results for gene_list: %s (%d rows)", gene_list_name, len(gene_list_df))

        has_per_celltype = (
            'analysis_type' in gene_list_df.columns
            and (gene_list_df['analysis_type'] == 'per_celltype').any()
        )

        if has_per_celltype:
            celltype_results: dict = {}
            summary_results: dict = {}

            for _, row in gene_list_df.iterrows():
                if row.get('analysis_type') != 'per_celltype':
                    continue
                celltype = row.get('celltype')
                if pd.isna(celltype) or celltype == '':
                    continue

                if celltype == 'summary':
                    # Read pre-computed weighted/macro aggregates — do NOT recompute
                    summary_results = {
                        'weighted_mse_test_baseline':   _pick(row, 'weighted_mse_test_baseline'),
                        'weighted_mse_test_probe':      _pick(row, 'weighted_mse_test_probe'),
                        'weighted_expvar_test_baseline': _pick(row, 'weighted_expvar_test_baseline'),
                        'weighted_expvar_test_probe':   _pick(row, 'weighted_expvar_test_probe'),
                        'macro_mse_test_baseline':      _pick(row, 'macro_mse_test_baseline'),
                        'macro_mse_test_probe':         _pick(row, 'macro_mse_test_probe'),
                        'macro_expvar_test_baseline':   _pick(row, 'macro_expvar_test_baseline'),
                        'macro_expvar_test_probe':      _pick(row, 'macro_expvar_test_probe'),
                    }
                else:
                    # Individual cell-type row
                    celltype_results[celltype] = {
                        'mse_train_baseline':   _pick(row, 'mse_train_baseline'),
                        'mse_test_baseline':    _pick(row, 'mse_test_baseline'),
                        'mse_test_probe':       _pick(row, 'mse_test_probe'),
                        'expvar_train_baseline': _pick(row, 'expvar_train_baseline'),
                        'expvar_test_baseline': _pick(row, 'expvar_test_baseline'),
                        'expvar_test_probe':    _pick(row, 'expvar_test_probe'),
                        'n_cells':              int(row.get('n_cells') or 1),
                        'skipped':              bool(row.get('skipped', False)),
                    }

            if celltype_results:
                results_dict[f'{group_type}_celltype_celltype_results'] = celltype_results
                logger.info("  Stored %d cell types", len(celltype_results))
            else:
                logger.warning("  No per-cell-type rows found for %s", gene_list_name)

            if summary_results:
                results_dict[f'{group_type}_celltype_summary'] = summary_results
                logger.info("  Read pre-computed summary from 'summary' row")
            else:
                logger.warning("  No 'summary' row found; aggregated plots will be skipped for %s", gene_list_name)

        else:
            # Global evaluation — no per-cell-type breakdown
            global_rows = gene_list_df[gene_list_df.get('analysis_type', '') == 'global']
            row = global_rows.iloc[0] if not global_rows.empty else gene_list_df.iloc[0]

            results_dict[f'{group_type}_global'] = {
                'mse_train_baseline':   _pick(row, 'mse_train_baseline'),
                'mse_test_baseline':    _pick(row, 'mse_test_baseline'),
                'mse_test_probe':       _pick(row, 'mse_test_probe'),
                'expvar_train_baseline': _pick(row, 'expvar_train_baseline'),
                'expvar_test_baseline': _pick(row, 'expvar_test_baseline'),
                'expvar_test_probe':    _pick(row, 'expvar_test_probe'),
            }
            # Mirror as summary so unified downstream functions work the same way
            results_dict[f'{group_type}_global_summary'] = {
                'weighted_mse_test_baseline':   _pick(row, 'mse_test_baseline'),
                'weighted_mse_test_probe':      _pick(row, 'mse_test_probe'),
                'weighted_expvar_test_baseline': _pick(row, 'expvar_test_baseline'),
                'weighted_expvar_test_probe':   _pick(row, 'expvar_test_probe'),
                'macro_mse_test_baseline':      _pick(row, 'mse_test_baseline'),
                'macro_mse_test_probe':         _pick(row, 'mse_test_probe'),
                'macro_expvar_test_baseline':   _pick(row, 'expvar_test_baseline'),
                'macro_expvar_test_probe':      _pick(row, 'expvar_test_probe'),
            }

        evaluation_results[gene_list_name] = results_dict

    return evaluation_results


# ===================================================================
# RESULT LOADING FUNCTIONS
# ===================================================================

def load_baseline_results(results_dir, panel_names):
    """
    Load baseline evaluation results for specified panels.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing result CSV files (either subdirectories or consolidated files)
    panel_names : list
        List of panel names to load results for
        
    Returns:
    --------
    dict
        Dictionary with keys: 'clustering', 'neighborhood', 'celltype'
        Each containing a filtered DataFrame with only the requested panels
    """
    logger.info("Loading baseline evaluation results...")
    
    results = {}
    
    # Check if results are in subdirectories or consolidated files
    subdirs = ['clustering', 'neighborhood', 'celltype']
    # Use subdirectory loading if ANY subdirectory exists
    use_subdirs = any(os.path.isdir(os.path.join(results_dir, sd)) for sd in subdirs)
    
    if use_subdirs:
        logger.info("  Loading from subdirectories (individual CSV files per panel)")
        
        for subdir in subdirs:
            subdir_path = os.path.join(results_dir, subdir)
            
            # Check if subdirectory exists
            if not os.path.exists(subdir_path):
                logger.debug(f"  Subdirectory not found: {subdir_path}, skipping {subdir}")
                results[subdir] = pd.DataFrame()
                continue
                
            dfs = []
            
            for panel_name in panel_names:
                # Try to find file for this panel
                panel_file = os.path.join(subdir_path, f"{panel_name}.csv")
                
                if os.path.exists(panel_file):
                    df = pd.read_csv(panel_file)
                    # Add dataset_name column if not present
                    if 'dataset_name' not in df.columns:
                        df['dataset_name'] = panel_name
                    dfs.append(df)
                else:
                    logger.debug(f"    File not found: {panel_file}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                results[subdir] = combined_df
                logger.info(f"  Loaded {subdir} results: {len(combined_df)} rows from {len(dfs)} panels")
            else:
                logger.warning(f"  No {subdir} results found for requested panels")
                results[subdir] = pd.DataFrame()
    else:
        # Original logic for consolidated files
        logger.info("  Loading from consolidated CSV files")
        
        # Load clustering results
        clustering_file = os.path.join(results_dir, 'clustering_results.csv')
        if os.path.exists(clustering_file):
            df = pd.read_csv(clustering_file)
            # Filter to requested panels
            df_filtered = df[df['dataset_name'].isin(panel_names)]
            results['clustering'] = df_filtered
            logger.info(f"  Loaded clustering results: {len(df_filtered)} rows")
        else:
            logger.warning(f"  Clustering results not found: {clustering_file}")
            results['clustering'] = pd.DataFrame()
        
        # Load neighborhood results
        neighborhood_file = os.path.join(results_dir, 'neighborhood_results.csv')
        if os.path.exists(neighborhood_file):
            df = pd.read_csv(neighborhood_file)
            df_filtered = df[df['dataset_name'].isin(panel_names)]
            results['neighborhood'] = df_filtered
            logger.info(f"  Loaded neighborhood results: {len(df_filtered)} rows")
        else:
            logger.warning(f"  Neighborhood results not found: {neighborhood_file}")
            results['neighborhood'] = pd.DataFrame()
        
        # Load celltype results
        celltype_file = os.path.join(results_dir, 'celltype_results.csv')
        if os.path.exists(celltype_file):
            df = pd.read_csv(celltype_file)
            df_filtered = df[df['dataset_name'].isin(panel_names)]
            results['celltype'] = df_filtered
            logger.info(f"  Loaded celltype results: {len(df_filtered)} rows")
        else:
            logger.warning(f"  Celltype results not found: {celltype_file}")
            results['celltype'] = pd.DataFrame()
    
    return results


def load_variability_results(results_dir, panel_names, subdirs=('nmf', 'mapping')):
    """
    Load variability evaluation results for specified panels.

    Parameters:
    -----------
    results_dir : str
        Directory containing result CSV files (either subdirectories or consolidated files)
    panel_names : list
        List of panel names to load results for
    subdirs : tuple of str, optional (default=('nmf', 'mapping'))
        Which subdirectory names under results_dir to look for (e.g. add 'pca'/'ica'/
        'ridge' to load those methods' results too). The subdirectory name is used
        verbatim as the result dict key. The legacy no-subdir consolidated-file
        fallback (nmf_representation.csv/mapping_performance.csv) only applies when
        this is left at its default.

    Returns:
    --------
    dict
        Dictionary with one key per entry in ``subdirs`` plus ``'celltype_specific'``
        (subdirectory-loading path only), each a filtered DataFrame with only the
        requested panels.
    """
    logger.info("Loading variability evaluation results...")

    results = {}

    # Use subdirectory loading if ANY requested subdirectory exists
    use_subdirs = any(os.path.isdir(os.path.join(results_dir, sd)) for sd in subdirs)

    if use_subdirs:
        logger.info("  Loading from subdirectories (individual CSV files per panel)")

        for subdir in subdirs:
            result_key = subdir
            subdir_path = os.path.join(results_dir, subdir)
            if not os.path.exists(subdir_path):
                logger.warning(f"  Subdirectory not found: {subdir_path}")
                results[result_key] = pd.DataFrame()
                continue
                
            dfs = []
            found_count = 0
            missing_count = 0
            
            for panel_name in panel_names:
                # Try to find file for this panel
                panel_file = os.path.join(subdir_path, f"{panel_name}.csv")
                
                if os.path.exists(panel_file):
                    df = pd.read_csv(panel_file)
                    # Add probeset_name column if not present (variability uses probeset_name)
                    if 'probeset_name' not in df.columns and 'dataset_name' not in df.columns:
                        df['probeset_name'] = panel_name
                    elif 'dataset_name' in df.columns and 'probeset_name' not in df.columns:
                        df['probeset_name'] = df['dataset_name']
                    dfs.append(df)
                    found_count += 1
                else:
                    missing_count += 1
                    if missing_count <= 3:  # Log first 3 missing files at INFO level
                        logger.info(f"    {result_key}: File not found: {panel_file}")
                    logger.debug(f"    File not found: {panel_file}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                results[result_key] = combined_df
                logger.info(f"  Loaded {result_key} results: {len(combined_df)} rows from {found_count}/{len(panel_names)} panels")
            else:
                logger.warning(f"  No {result_key} results found for requested panels ({missing_count} files missing)")
                results[result_key] = pd.DataFrame()
        
        # Note: celltype_specific might not be in subdirectories, leave as empty for now
        results['celltype_specific'] = pd.DataFrame()
        
    elif tuple(subdirs) != ('nmf', 'mapping'):
        # Non-default subdirs (e.g. pca/ica/ridge) have no legacy consolidated-file
        # format to fall back to — just report empty results per requested subdir.
        logger.warning(
            f"  No {list(subdirs)} subdirectories found under {results_dir}; "
            "non-default subdirs have no consolidated-file fallback."
        )
        for subdir in subdirs:
            results[subdir] = pd.DataFrame()
    else:
        # Original logic for consolidated files (nmf/mapping default only)
        logger.info("  Loading from consolidated CSV files")

        # Load NMF representation results
        nmf_file = os.path.join(results_dir, 'nmf_representation.csv')
        if os.path.exists(nmf_file):
            df = pd.read_csv(nmf_file)
            df_filtered = df[df['probeset_name'].isin(panel_names)]
            results['nmf'] = df_filtered
            logger.info(f"  Loaded NMF results: {len(df_filtered)} rows")
        else:
            logger.warning(f"  NMF results not found: {nmf_file}")
            results['nmf'] = pd.DataFrame()
        
        # Load mapping performance results
        mapping_file = os.path.join(results_dir, 'mapping_performance.csv')
        if os.path.exists(mapping_file):
            df = pd.read_csv(mapping_file)
            df_filtered = df[df['probeset_name'].isin(panel_names)]
            results['mapping'] = df_filtered
            logger.info(f"  Loaded mapping results: {len(df_filtered)} rows")
        else:
            logger.warning(f"  Mapping results not found: {mapping_file}")
            results['mapping'] = pd.DataFrame()
        
        # Load celltype-specific results
        celltype_file = os.path.join(results_dir, 'celltype_specific_results.csv')
        if os.path.exists(celltype_file):
            df = pd.read_csv(celltype_file)
            df_filtered = df[df['probeset_name'].isin(panel_names)]
            results['celltype_specific'] = df_filtered
            logger.info(f"  Loaded celltype-specific results: {len(df_filtered)} rows")
        else:
            logger.warning(f"  Celltype-specific results not found: {celltype_file}")
            results['celltype_specific'] = pd.DataFrame()
    
    return results

# ===================================================================
# PLOTTING FUNCTIONS
# ===================================================================

def create_baseline_plots(results, output_dir, group_name, color_map=None, 
                         png_dpi=DEFAULT_PNG_DPI, plot_clustering=True, plot_neighborhood=True, 
                         plot_celltype=True, use_hardcoded_colors=False, external_names=None):
    """
    Create baseline evaluation plots.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing baseline results DataFrames
    output_dir : str
        Output directory for plots
    group_name : str
        Name of the plot group (used in titles)
    color_map : dict, optional
        Color mapping for datasets. If None, uses plotting module's comprehensive color scheme.
    png_dpi : int
        DPI for saved plots
    plot_clustering : bool
        Whether to create clustering plots
    plot_neighborhood : bool
        Whether to create neighborhood plots
    plot_celltype : bool
        Whether to create celltype plots
    use_hardcoded_colors : bool
        If True, uses get_color_map_for_datasets (hardcoded evaluation groups only).
        If False, uses plotting module's comprehensive color scheme (all methods).
    external_names : list, optional
        List of external panel names for proper color assignment (e.g., ['5k', 'mMulti_v1', 'Spapros'])
    """
    logger.info(f"Creating baseline plots in: {output_dir}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate color and marker maps for all datasets
    if 'clustering' in results and not results['clustering'].empty:
        dataset_names = results['clustering']['dataset_name'].unique().tolist()
    elif 'neighborhood' in results and not results['neighborhood'].empty:
        dataset_names = results['neighborhood']['dataset'].unique().tolist()
    elif 'celltype' in results and not results['celltype'].empty:
        dataset_names = results['celltype']['dataset'].unique().tolist()
    else:
        logger.warning("No results data found for plotting")
        return
    
    logger.info(f"Generating plots for {len(dataset_names)} datasets")
    
    # Detect factor-range mode: check if dataset names contain factor numbers
    # Factor-range datasets have names like "..._2factors_100" or "..._15factors_200"
    import re
    factor_range_mode = False
    if dataset_names:
        # Check if any dataset name contains a factor number pattern
        factor_counts = sum(1 for name in dataset_names if re.search(r'_\d+factors_', name))
        # If more than half the datasets have factor numbers, we're in factor-range mode
        factor_range_mode = factor_counts > len(dataset_names) / 2
        logger.info(f"Factor-range mode detected: {factor_range_mode} ({factor_counts}/{len(dataset_names)} datasets with factors)")
    
    # Get color map based on mode
    if color_map is not None:
        # User provided explicit color map - use it
        pass
    elif use_hardcoded_colors:
        # Use hardcoded evaluation group colors (limited to specific methods)
        color_map = get_color_map_for_datasets(dataset_names)
    else:
        # Use plotting module's comprehensive color scheme (all methods)
        # This calls the baseline plotting module's function which handles ALL probe panel types
        color_map, marker_map = generate_method_specific_colors_and_markers(
            dataset_names, external_names=external_names, group_name=group_name
        )
        
    # Generate marker map if not already set
    if 'marker_map' not in locals():
        marker_map = {}
        for i, name in enumerate(dataset_names):
            marker_map[name] = ['o', 's', '^', 'D', 'v', '<', '>', 'p'][i % 8]
    
    # Clustering quality plots
    if plot_clustering and 'clustering' in results and not results['clustering'].empty:
        logger.info("Creating clustering quality plots...")
        clustering_dir = os.path.join(output_dir, 'clustering')
        os.makedirs(clustering_dir, exist_ok=True)
        
        # ARI plots
        plot_clustering_quality_ari(
            results['clustering'], 
            clustering_dir, 
            PNG_DPI=png_dpi,
            color_map=color_map,
            marker_map=marker_map,
            dimensionality_reduction='pca',
            group_name=group_name
        )
        
        # NMI plots
        plot_clustering_quality_nmi(
            results['clustering'], 
            clustering_dir, 
            PNG_DPI=png_dpi,
            color_map=color_map,
            marker_map=marker_map,
            dimensionality_reduction='pca',
            group_name=group_name
        )
                
        logger.info(f"✓ Clustering plots saved to: {clustering_dir}")
    
    # Neighborhood preservation plots
    if plot_neighborhood and 'neighborhood' in results and not results['neighborhood'].empty:
        logger.info("Creating neighborhood preservation plots...")
        neighborhood_dir = os.path.join(output_dir, 'neighborhood')
        os.makedirs(neighborhood_dir, exist_ok=True)
        
        plot_neighborhood_preservation_by_k(
            results['neighborhood'], 
            neighborhood_dir, 
            PNG_DPI=png_dpi,
            color_map=color_map,
            marker_map=marker_map,
            group_name=group_name
        )
        plot_neighborhood_preservation_celltype_heatmap(
            results['neighborhood'],
            neighborhood_dir,
            PNG_DPI=png_dpi
        )

        logger.info(f"✓ Neighborhood plots saved to: {neighborhood_dir}")
    
    # Celltype identification plots
    if plot_celltype and 'celltype' in results and not results['celltype'].empty:
        logger.info("Creating celltype identification plots...")
        celltype_dir = os.path.join(output_dir, 'celltype')
        os.makedirs(celltype_dir, exist_ok=True)
        
        plot_celltype_f1_heatmap(
            results['celltype'], 
            celltype_dir, 
            PNG_DPI=png_dpi
        )
        plot_celltype_classification_diagnostics_global(
            results['celltype'],
            celltype_dir,
            PNG_DPI=png_dpi
        )

        logger.info(f"✓ Celltype plots saved to: {celltype_dir}")


def create_variability_plots(results, output_dir, group_name, color_map=None,
                            png_dpi=DEFAULT_PNG_DPI, plot_nmf=True, plot_mapping=True,
                            plot_celltype_specific=True, use_hardcoded_colors=False, external_names=None,
                            plot_pca=False, plot_ica=False):
    """
    Create variability evaluation plots.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing variability results DataFrames
    output_dir : str
        Output directory for plots
    group_name : str
        Name of the plot group (used in titles)
    color_map : dict, optional
        Color mapping for datasets. If None, uses plotting module's comprehensive color scheme.
    png_dpi : int
        DPI for saved plots
    plot_nmf : bool
        Whether to create NMF representation plots
    plot_mapping : bool
        Whether to create mapping performance plots
    plot_celltype_specific : bool
        Whether to create per-celltype plots
    use_hardcoded_colors : bool
        If True, uses get_color_map_for_datasets (hardcoded evaluation groups only).
        If False, uses variability plotting module's comprehensive color scheme (all methods).
    external_names : list, optional
        List of external panel names for proper color assignment (e.g., ['5k', 'mMulti_v1', 'Spapros'])
    plot_pca : bool, optional (default=False)
        Whether to create PCA reconstruction plots (results['pca'], same pipeline as NMF).
    plot_ica : bool, optional (default=False)
        Whether to create ICA reconstruction plots (results['ica'], same pipeline as NMF).
    """
    logger.info(f"Creating variability plots in: {output_dir}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Get dataset names for color mapping
    dataset_names = []
    for _key in ('nmf', 'mapping', 'pca', 'ica'):
        if _key in results and not results[_key].empty:
            dataset_names = results[_key]['gene_list'].unique().tolist()
            break
    
    # Get color map based on mode
    if color_map is not None:
        # User provided explicit color map - use it
        pass
    elif use_hardcoded_colors and len(dataset_names) > 0:
        # Use hardcoded evaluation group colors (limited to specific methods)
        color_map = get_color_map_for_datasets(dataset_names)
    elif len(dataset_names) > 0:
        # Use variability plotting module's comprehensive color scheme (all methods)
        # Get comprehensive color scheme from variability plotting module
        # Pass external_names if provided, otherwise use empty list
        external_names_for_colors = external_names if external_names is not None else []
        all_colors = get_category_colors(external_names_for_colors)
        
        # Map dataset names to colors
        color_map = {}
        for name in dataset_names:
            # Try exact match
            if name in all_colors:
                color_map[name] = all_colors[name]
            else:
                # Try partial match for categories
                matched = False
                name_lower = name.lower()
                for category, color in all_colors.items():
                    if category.lower() in name_lower:
                        color_map[name] = color
                        matched = True
                        break
                if not matched:
                    color_map[name] = '#808080'  # Gray fallback
    
    # NMF representation plots
    if plot_nmf and 'nmf' in results and not results['nmf'].empty:
        logger.info("Creating NMF representation plots...")
        nmf_dir = os.path.join(output_dir, 'nmf')
        os.makedirs(nmf_dir, exist_ok=True)

        # Convert DataFrame to dictionary format expected by plotting functions
        nmf_dict = convert_variability_df_to_dict(results['nmf'], group_type='nmf')

        # 1. Plot aggregated celltype metrics (weighted vs macro)
        logger.info("  Creating aggregated celltype metrics plots...")
        plot_aggregated_celltype_metrics(
            nmf_dict,
            nmf_dir,
            title_suffix=" - NMF",
            PNG_DPI=png_dpi,
            external_names=external_names,
            group_name=group_name
        )

        # 2. Plot individual celltype evaluation results
        logger.info("  Creating individual celltype evaluation plots...")
        plot_celltype_evaluation_results(
            nmf_dict,
            nmf_dir,
            title_suffix=" - NMF",
            PNG_DPI=png_dpi,
            external_names=external_names,
            group_name=group_name,
            plot_celltype=False  # Skip celltype-specific plots for variability
        )

        # 3. Create combined plots with comprehensive information
        logger.info("  Creating global combined plots...")
        try:
            global_plot_data = prepare_global_combined_plot_data(results['nmf'], group_type='nmf')
            if global_plot_data:
                for metric in ['nmf_mse', 'nmf_expvar']:
                    if metric in global_plot_data and global_plot_data[metric]:
                        create_combined_plot_with_info(
                            classifier_type="Evaluation",
                            gene_count=group_name,
                            metric=metric,
                            results=global_plot_data,
                            output_dir=nmf_dir,
                            filter_type=group_name,
                            gene_sel='combined',
                            PNG_DPI=png_dpi,
                            external_names=external_names,
                            group_name=group_name
                        )
        except Exception as e:
            logger.warning(f"  Could not generate global combined plots: {e}")

        # 4. Gene-subset explained-variance breakdown (all / panel-only / non-panel).
        #    Always attempted; skips gracefully if the eval run predates the
        #    --gene_subsets default (no panel/non-panel columns).
        logger.info("  Creating gene-subset explained-variance breakdown...")
        try:
            plot_expvar_gene_subset_breakdown(
                results['nmf'],
                nmf_dir,
                title_suffix=" - NMF",
                PNG_DPI=png_dpi,
                external_names=external_names,
                group_name=group_name,
            )
        except Exception as e:
            logger.warning(f"  Could not generate gene-subset expvar breakdown: {e}")

        # 5. Per-cell-type gene-subset explained-variance breakdown (same guards).
        logger.info("  Creating per-celltype gene-subset explained-variance breakdown...")
        try:
            plot_expvar_gene_subset_breakdown_by_celltype(
                results['nmf'],
                nmf_dir,
                title_suffix=" - NMF",
                PNG_DPI=png_dpi,
                external_names=external_names,
                group_name=group_name,
            )
        except Exception as e:
            logger.warning(f"  Could not generate per-celltype gene-subset expvar breakdown: {e}")

        logger.info(f"✓ NMF plots saved to: {nmf_dir}")

    # PCA / ICA reconstruction plots — same pipeline as NMF, just a different
    # results key / output subfolder / title suffix / methods= value.
    for _method, _do_plot in (('PCA', plot_pca), ('ICA', plot_ica)):
        if not (_do_plot and _method in results and not results[_method].empty):
            continue
        _display = METHOD_DISPLAY_NAMES.get(_method.lower(), _method.title())
        logger.info(f"Creating {_display} reconstruction plots...")
        _method_dir = os.path.join(output_dir, _method)
        os.makedirs(_method_dir, exist_ok=True)

        _method_dict = convert_variability_df_to_dict(results[_method], group_type=_method)

        logger.info("  Creating aggregated celltype metrics plots...")
        plot_aggregated_celltype_metrics(
            _method_dict, _method_dir, title_suffix=f" - {_display}", PNG_DPI=png_dpi,
            external_names=external_names, group_name=group_name, methods=(_method,)
        )

        logger.info("  Creating individual celltype evaluation plots...")
        plot_celltype_evaluation_results(
            _method_dict, _method_dir, title_suffix=f" - {_display}", PNG_DPI=png_dpi,
            external_names=external_names, group_name=group_name, plot_celltype=False,
            methods=(_method,)
        )

        logger.info("  Creating global combined plots...")
        try:
            _global_plot_data = prepare_global_combined_plot_data(results[_method], group_type=_method)
            if _global_plot_data:
                for _metric in [f'{_method}_mse', f'{_method}_expvar']:
                    if _metric in _global_plot_data and _global_plot_data[_metric]:
                        create_combined_plot_with_info(
                            classifier_type="Evaluation", gene_count=group_name, metric=_metric,
                            results=_global_plot_data, output_dir=_method_dir, filter_type=group_name,
                            gene_sel='combined', PNG_DPI=png_dpi, external_names=external_names,
                            group_name=group_name
                        )
        except Exception as e:
            logger.warning(f"  Could not generate global combined plots: {e}")

        logger.info("  Creating gene-subset explained-variance breakdown...")
        try:
            plot_expvar_gene_subset_breakdown(
                results[_method], _method_dir, title_suffix=f" - {_display}", PNG_DPI=png_dpi,
                external_names=external_names, group_name=group_name,
            )
        except Exception as e:
            logger.warning(f"  Could not generate gene-subset expvar breakdown: {e}")

        logger.info("  Creating per-celltype gene-subset explained-variance breakdown...")
        try:
            plot_expvar_gene_subset_breakdown_by_celltype(
                results[_method], _method_dir, title_suffix=f" - {_display}", PNG_DPI=png_dpi,
                external_names=external_names, group_name=group_name,
            )
        except Exception as e:
            logger.warning(f"  Could not generate per-celltype gene-subset expvar breakdown: {e}")

        logger.info(f"✓ {_display} plots saved to: {_method_dir}")

    # Mapping performance plots
    if plot_mapping and 'mapping' in results and not results['mapping'].empty:
        logger.info("Creating mapping performance plots...")
        mapping_dir = os.path.join(output_dir, 'mapping')
        os.makedirs(mapping_dir, exist_ok=True)
        
        # Convert DataFrame to dictionary format expected by plotting functions
        mapping_dict = convert_variability_df_to_dict(results['mapping'], group_type='mapping')
        
        # 1. Plot aggregated celltype metrics (weighted vs macro)
        logger.info("  Creating aggregated celltype metrics plots...")
        plot_aggregated_celltype_metrics(
            mapping_dict,
            mapping_dir,
            title_suffix=" - Mapping",
            PNG_DPI=png_dpi,
            external_names=external_names,
            group_name=group_name
        )
        
        # 2. Plot individual celltype evaluation results
        logger.info("  Creating individual celltype evaluation results...")
        plot_celltype_evaluation_results(
            mapping_dict,
            mapping_dir,
            title_suffix=" - Mapping",
            PNG_DPI=png_dpi,
            external_names=external_names,
            group_name=group_name,
            plot_celltype=False  # Skip celltype-specific plots for variability
        )
        
        # 3. Create combined plots with comprehensive information
        logger.info("  Creating global combined plots...")
        try:
            global_plot_data = prepare_global_combined_plot_data(results['mapping'], group_type='mapping')
            if global_plot_data:
                for metric in ['mapping_mse', 'mapping_expvar']:
                    if metric in global_plot_data and global_plot_data[metric]:
                        create_combined_plot_with_info(
                            classifier_type="Evaluation",
                            gene_count=group_name,
                            metric=metric,
                            results=global_plot_data,
                            output_dir=mapping_dir,
                            filter_type=group_name,
                            gene_sel='combined',
                            PNG_DPI=png_dpi,
                            external_names=external_names,
                            group_name=group_name
                        )
        except Exception as e:
            logger.warning(f"  Could not generate global combined plots: {e}")
        
        logger.info(f"✓ Mapping plots saved to: {mapping_dir}")
    
    # Celltype-specific plots
    if plot_celltype_specific and 'celltype_specific' in results and not results['celltype_specific'].empty:
        logger.info("Creating celltype-specific plots...")
        celltype_dir = os.path.join(output_dir, 'celltype_specific')
        os.makedirs(celltype_dir, exist_ok=True)
        
        # Use celltype-specific plotting if available
        # Note: This depends on your actual variability plotting module implementation
        logger.info("Celltype-specific variability plots not yet implemented")
        
        logger.info(f"✓ Celltype-specific directory created: {celltype_dir}")

# ===================================================================
# ARGUMENT PARSING
# ===================================================================
# RECONSTRUCTION COMPARISON PLOTS (Tangram vs NMF)
# ===================================================================


def create_reconstruction_plots(
    tangram_results_dir: str,
    panel_names: list[str],
    output_dir: str,
    nmf_results: dict | None = None,
    png_dpi: int = DEFAULT_PNG_DPI,
) -> None:
    """Create Tangram vs NMF reconstruction comparison bar charts.

    For each panel, reads the per-celltype Tangram CSV and (optionally) the
    NMF CV results, then produces two plots:
      - Per-celltype MSE and ExpVar side-by-side bars with error bars.
      - Aggregated (macro/weighted) MSE and ExpVar grouped bars.

    Args:
        tangram_results_dir: Root directory produced by the Tangram evaluation
            (should contain ``per_celltype/<panel>.csv`` files).
        panel_names: List of panel dataset names to plot.
        output_dir: Directory where plots are saved.
        nmf_results: Optional dict loaded from ``load_variability_results``;
            used to extract NMF per-celltype CV results for comparison.
        png_dpi: Output resolution.
    """
    from pathlib import Path

    tangram_root = Path(tangram_results_dir)
    out_root = Path(output_dir) / "tangram"
    out_root.mkdir(parents=True, exist_ok=True)

    for panel_name in panel_names:
        tangram_csv = tangram_root / panel_name / "per_celltype" / f"{panel_name}.csv"
        if not tangram_csv.exists():
            # Try flat structure (no panel subdirectory)
            tangram_csv = tangram_root / "per_celltype" / f"{panel_name}.csv"
        if not tangram_csv.exists():
            logger.warning("Tangram per-celltype CSV not found for panel '%s' — skipping.", panel_name)
            continue

        # Extract NMF per-celltype results if available
        nmf_ct_results: dict | None = None
        nmf_summary: dict | None = None
        nmf_global: dict | None = None
        if nmf_results is not None and "nmf" in nmf_results:
            mech_df = nmf_results["nmf"]
            if not mech_df.empty and "gene_list" in mech_df.columns and "analysis_type" in mech_df.columns:
                panel_df = mech_df[mech_df["gene_list"] == panel_name]
                per_ct = panel_df[panel_df["analysis_type"] == "per_celltype"]
                if "celltype" in per_ct.columns:
                    # The macro/weighted aggregates live in the per_celltype row
                    # whose celltype is "summary" (the global row has none).
                    nmf_ct_results = {
                        row["celltype"]: row.to_dict()
                        for _, row in per_ct.iterrows()
                        if pd.notna(row.get("celltype")) and row["celltype"] != "summary"
                    }
                    summary_row = per_ct[per_ct["celltype"] == "summary"]
                    if not summary_row.empty:
                        nmf_summary = summary_row.iloc[0].to_dict()
                global_row = panel_df[panel_df["analysis_type"] == "global"]
                if not global_row.empty:
                    nmf_global = global_row.iloc[0].to_dict()
            if nmf_ct_results is None:
                logger.warning("No NMF results found for panel '%s' — plotting Tangram only.", panel_name)

        tangram_global_csv = tangram_csv.parent.parent / "global" / f"{panel_name}.csv"

        panel_out = out_root / panel_name
        panel_out.mkdir(parents=True, exist_ok=True)

        plot_reconstruction_per_celltype(
            tangram_csv=tangram_csv,
            nmf_celltype_results=nmf_ct_results,
            output_path=panel_out / f"{panel_name}_reconstruction_per_celltype.png",
            dataset_name=panel_name,
            png_dpi=png_dpi,
        )

        plot_reconstruction_aggregated_metrics(
            tangram_csv=tangram_csv,
            nmf_summary=nmf_summary,
            output_path=panel_out / f"{panel_name}_reconstruction_aggregated.png",
            dataset_name=panel_name,
            png_dpi=png_dpi,
            tangram_global_csv=tangram_global_csv,
            nmf_global=nmf_global,
        )

    logger.info("Reconstruction comparison plots saved to: %s", out_root)


def create_ridge_plots(
    ridge_results_dir: str,
    panel_names: list[str],
    output_dir: str,
    png_dpi: int = DEFAULT_PNG_DPI,
    external_names: list[str] | None = None,
    group_name: str | None = None,
) -> None:
    """Create Ridge (raw + lognorm) aggregated reconstruction comparison plots.

    Loads Ridge's per-panel results (via ``load_variability_results`` with
    ``subdirs=('ridge-regression',)``) and calls
    ``plot_ridge_aggregated_celltype_metrics`` once per space, writing into
    ``<output_dir>/ridge-regression/``. v1 scope: only the aggregated weighted/macro
    MSE + ExpVar comparison (see ``_variability_plots.plot_ridge_aggregated_celltype_metrics``).

    Args:
        ridge_results_dir: Directory containing Ridge's ``ridge-regression/`` results
            subfolder (i.e. the shared ``Variability-Evaluation/results`` dir, same
            parent NMF/PCA/ICA use).
        panel_names: List of panel dataset names to plot.
        output_dir: Root output directory (a ``ridge-regression`` subfolder is created
            under it).
        png_dpi: Output resolution.
        external_names: Passed through to the plotting function.
        group_name: Passed through to the plotting function.
    """
    logger.info(f"Loading Ridge results from: {ridge_results_dir}")
    ridge_results = load_variability_results(ridge_results_dir, panel_names, subdirs=('ridge-regression',))
    ridge_df = ridge_results.get('ridge-regression')
    if ridge_df is None or ridge_df.empty:
        logger.warning("No Ridge results found — skipping Ridge plots.")
        return

    ridge_dir = os.path.join(output_dir, 'ridge-regression')
    os.makedirs(ridge_dir, exist_ok=True)
    for space in ('raw', 'lognorm'):
        plot_ridge_aggregated_celltype_metrics(
            ridge_df, ridge_dir, space=space,
            title_suffix=f" ({group_name})" if group_name else "",
            PNG_DPI=png_dpi, external_names=external_names, group_name=group_name,
        )
    logger.info(f"✓ Ridge plots saved to: {ridge_dir}")


# ===================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Create plots from evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--evaluation_type', required=True,
                       choices=['baseline', 'variability', 'all'],
                       help='Type of evaluation to plot')
    
    parser.add_argument('--panels', required=True,
                       help='Comma-separated list of panel names to include in plots')
    
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for plots')
    
    # Result directory arguments
    parser.add_argument('--results_dir',
                       help='Results directory (for single evaluation type)')
    
    parser.add_argument('--baseline_results_dir',
                       help='Baseline results directory (for both evaluation types)')
    
    parser.add_argument('--variability_results_dir',
                       help='Variability results directory (for both evaluation types)')
    
    # Optional arguments
    parser.add_argument('--group_name', default='custom_comparison',
                       help='Name for this plot group (used in titles)')
    
    parser.add_argument("--png_dpi", type=int, default=DEFAULT_PNG_DPI,
                       help=f'DPI for saved plots (default: {DEFAULT_PNG_DPI})')
    
    parser.add_argument('--allow_mixed_sizes', action='store_true',
                       help='Allow panels with different gene counts in same plot')
    
    # Plot type flags for baseline
    parser.add_argument('--plot_clustering', action='store_true',
                       help='Create clustering quality plots')
    
    parser.add_argument('--plot_neighborhood', action='store_true',
                       help='Create neighborhood preservation plots')
    
    parser.add_argument('--plot_celltype', action='store_true',
                       help='Create celltype identification plots')
    
    # Plot type flags for variability
    parser.add_argument('--plot_nmf', action='store_true',
                       help='Create NMF representation plots')
    
    parser.add_argument('--plot_mapping', action='store_true',
                       help='Create mapping performance plots')
    
    parser.add_argument('--plot_celltype_specific', action='store_true',
                       help='Create per-celltype variability plots')

    parser.add_argument('--plot_pca', action='store_true',
                       help="Create PCA reconstruction plots (Variability-Evaluation/results/PCA/ -- "
                            "shares the same Variability-Evaluation/ root as NMF/ICA/Ridge, just a "
                            "different leaf subdir). Loaded from --variability_results_dir / "
                            "--results_dir, same as --plot_nmf.")

    parser.add_argument('--plot_ica', action='store_true',
                       help="Create ICA reconstruction plots (Variability-Evaluation/results/ICA/ -- "
                            "shares the same Variability-Evaluation/ root as NMF/PCA/Ridge, just a "
                            "different leaf subdir). Loaded from --variability_results_dir / "
                            "--results_dir, same as --plot_nmf.")

    parser.add_argument('--plot_ridge', action='store_true',
                       help='Create Ridge (raw+lognorm) reconstruction plots. Requires --ridge_results_dir.')

    parser.add_argument('--ridge_results_dir',
                       help="Ridge results directory, e.g. '.../Variability-Evaluation/results' -- Ridge "
                            "writes into a 'ridge-regression' leaf subdir under the same shared "
                            "Variability-Evaluation/ root NMF/PCA/ICA use (in production this is "
                            "typically pointed at the same directory as --variability_results_dir / "
                            "--results_dir). Required when --plot_ridge is set.")

    # External panel configuration
    parser.add_argument('--external_names',
                       help='Comma-separated list of external panel names (e.g., "5k,mMulti_v1,Spapros")')

    # Tangram reconstruction comparison
    parser.add_argument('--tangram_results_dir',
                       help='Root Tangram evaluation directory (contains per_celltype/ sub-dirs). '
                            'When provided, reconstruction comparison plots (Tangram vs NMF) are created.')

    return parser.parse_args()

# ===================================================================
# MAIN EXECUTION
# ===================================================================

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Setup logging to console only (no log file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("="*80)
    logger.info("FLEXIBLE PLOTTING PIPELINE")
    logger.info("="*80)
    logger.info(f"Evaluation type: {args.evaluation_type}")
    logger.info(f"Group name: {args.group_name}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Parse panel names
    panel_names = [p.strip() for p in args.panels.split(',')]
    logger.info(f"Panels to plot: {panel_names}")
    
    # Parse external names
    external_names = None
    if args.external_names:
        external_names = [n.strip() for n in args.external_names.split(',')]
        logger.info(f"External panels: {external_names}")
    
    # Check mixed sizes warning
    if args.allow_mixed_sizes:
        logger.info("Mixed panel sizes allowed - plots will include panels of different sizes")
    
    # Validate arguments
    if args.evaluation_type == 'all':
        if not args.baseline_results_dir or not args.variability_results_dir:
            logger.error("Both --baseline_results_dir and --variability_results_dir required when evaluation_type='all'")
            return 1
    else:
        if not args.results_dir:
            logger.error("--results_dir required when evaluation_type is 'baseline' or 'variability'")
            return 1

        # Process baseline evaluation
    if args.evaluation_type in ['baseline', 'all']:
        results_dir = args.baseline_results_dir if args.evaluation_type == 'all' else args.results_dir

        logger.info("")
        logger.info("="*80)
        logger.info("BASELINE EVALUATION PLOTS")
        logger.info("="*80)

        baseline_results = load_baseline_results(results_dir, panel_names)

        baseline_output = os.path.join(args.output_dir, 'Baseline') if args.evaluation_type == 'all' else args.output_dir
        
        # Use comprehensive color scheme (not hardcoded) to allow all methods
        create_baseline_plots(
            baseline_results,
            baseline_output,
            args.group_name,
            color_map=None,  # Let function use comprehensive color scheme
            png_dpi=args.png_dpi,
            plot_clustering=args.plot_clustering,
            plot_neighborhood=args.plot_neighborhood,
            plot_celltype=args.plot_celltype,
            use_hardcoded_colors=False,  # Use comprehensive colors, not hardcoded
            external_names=external_names  # Pass for proper color assignment
        )
    
    # Process variability evaluation
    if args.evaluation_type in ['variability', 'all']:
        results_dir = args.variability_results_dir if args.evaluation_type == 'all' else args.results_dir
        
        logger.info("")
        logger.info("="*80)
        logger.info("VARIABILITY EVALUATION PLOTS")
        logger.info("="*80)
        
        _variability_subdirs = ['nmf', 'mapping']
        if args.plot_pca:
            _variability_subdirs.append('PCA')
        if args.plot_ica:
            _variability_subdirs.append('ICA')
        variability_results = load_variability_results(results_dir, panel_names, subdirs=tuple(_variability_subdirs))
        
        variability_output = os.path.join(args.output_dir, 'Variability') if args.evaluation_type == 'all' else args.output_dir
        
        # Use comprehensive color scheme (not hardcoded) to allow all methods
        create_variability_plots(
            variability_results,
            variability_output,
            args.group_name,
            color_map=None,  # Let function use comprehensive color scheme
            png_dpi=args.png_dpi,
            plot_nmf=args.plot_nmf,
            plot_mapping=args.plot_mapping,
            plot_celltype_specific=args.plot_celltype_specific,
            use_hardcoded_colors=False,  # Use comprehensive colors, not hardcoded
            external_names=external_names,  # Pass for proper color assignment
            plot_pca=args.plot_pca,
            plot_ica=args.plot_ica,
        )

    # Ridge reconstruction plots — independent of --evaluation_type (own results tree),
    # same pattern as the Tangram block below.
    if getattr(args, 'plot_ridge', False):
        if not args.ridge_results_dir:
            logger.error("--ridge_results_dir is required when --plot_ridge is set")
            return 1
        logger.info("")
        logger.info("="*80)
        logger.info("RIDGE VARIABILITY PLOTS")
        logger.info("="*80)
        ridge_output = os.path.join(args.output_dir, 'Variability') \
            if args.evaluation_type == 'all' else args.output_dir
        create_ridge_plots(
            ridge_results_dir=args.ridge_results_dir,
            panel_names=panel_names,
            output_dir=ridge_output,
            png_dpi=args.png_dpi,
            external_names=external_names,
            group_name=args.group_name,
        )

    # Reconstruction comparison plots (Tangram vs NMF)
    if getattr(args, 'tangram_results_dir', None):
        logger.info("")
        logger.info("="*80)
        logger.info("RECONSTRUCTION COMPARISON PLOTS (Tangram vs NMF)")
        logger.info("="*80)
        # Load NMF results for comparison if variability results are available
        nmf_for_comparison = None
        if args.evaluation_type in ['variability', 'all']:
            results_dir_var = args.variability_results_dir if args.evaluation_type == 'all' else args.results_dir
            nmf_for_comparison = load_variability_results(results_dir_var, panel_names)
        reconstruction_output = os.path.join(args.output_dir, 'Reconstruction') \
            if args.evaluation_type == 'all' else args.output_dir
        create_reconstruction_plots(
            tangram_results_dir=args.tangram_results_dir,
            panel_names=panel_names,
            output_dir=reconstruction_output,
            nmf_results=nmf_for_comparison,
            png_dpi=args.png_dpi,
        )

    logger.info("")
    logger.info("="*80)
    logger.info("PLOTTING COMPLETE")
    logger.info("="*80)
    logger.info(f"All plots saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
