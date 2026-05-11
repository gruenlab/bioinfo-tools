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
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from _clustering_plots import (
    generate_method_specific_colors_and_markers,
    generate_method_specific_colors,
    extract_display_name_from_dataset,
    plot_clustering_quality_ari,
    plot_clustering_quality_nmi,
    plot_celltype_f1_heatmap,
    plot_celltype_accuracy_barchart,
    plot_neighborhood_preservation_by_k,
    plot_optimal_neighborhood_preservation,
    plot_neighborhood_preservation_heatmap,
)
from _variability_plots import (
    get_category_colors,
    plot_celltype_evaluation_results,
    create_combined_plot_with_info,
    plot_aggregated_celltype_metrics,
)
from _reconstruction_plots import (
    load_tangram_per_celltype_csv,
    plot_reconstruction_per_celltype,
    plot_reconstruction_aggregated_metrics,
)
from _constants import DEFAULT_PNG_DPI

logger = logging.getLogger(__name__)

__all__ = [
    "load_evaluation_results",
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
    """
    Convert variability results DataFrame back to dictionary format expected by plotting functions.
    
    The plotting functions expect:
    evaluation_results[gene_list_name] = {
        'nmf_celltype_summary': {...} or 'mapping_celltype_summary': {...},
        'nmf_celltype_AT2': {...}, ...
    }

    Or for global evaluation:
    evaluation_results[gene_list_name] = {
        'nmf_global_summary': {...} or 'mapping_global_summary': {...}
    }

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: gene_list, celltype (optional), analysis_type, mse_*, expvar_*, etc.
    group_type : str
        Either 'nmf' or 'mapping'
        
    Returns:
    --------
    dict
        Nested dictionary structure expected by plotting functions
    """
    evaluation_results = {}
    
    for gene_list_name in df['gene_list'].unique():
        gene_list_df = df[df['gene_list'] == gene_list_name]
        results_dict = {}
        
        logger.info(f"Converting results for gene_list: {gene_list_name}")
        logger.info(f"  DataFrame shape: {gene_list_df.shape}")
        logger.info(f"  Columns: {list(gene_list_df.columns)}")
        
        # Check if this is per-celltype or global evaluation based on analysis_type column
        has_per_celltype = 'analysis_type' in gene_list_df.columns and \
                          (gene_list_df['analysis_type'] == 'per_celltype').any()
        logger.info(f"  has_per_celltype: {has_per_celltype}")
        
        if has_per_celltype:
            # Per-celltype evaluation
            # Process each celltype row
            celltype_results = {}
            logger.info(f"  Processing {len(gene_list_df)} rows for celltype extraction")
            for idx, row in gene_list_df.iterrows():
                analysis_type = row.get('analysis_type')
                celltype_val = row.get('celltype')
                logger.debug(f"    Row {idx}: analysis_type={analysis_type}, celltype={celltype_val}")
                
                if row.get('analysis_type') == 'per_celltype' and pd.notna(row.get('celltype')) and row.get('celltype') != '':
                    celltype = row['celltype']
                    
                    # Store full celltype results for later aggregation
                    celltype_results[celltype] = {
                        'mse_train_baseline': row.get('mse_train_baseline'),
                        'mse_test_baseline': row.get('mse_test_baseline'),
                        'mse_test_probe': row.get('mse_test_probe'),
                        'expvar_train_baseline': row.get('expvar_train_baseline'),
                        'expvar_test_baseline': row.get('expvar_test_baseline'),
                        'expvar_test_probe': row.get('expvar_test_probe'),
                        'mse_ratio': row.get('mse_ratio'),
                        'expvar_ratio': row.get('expvar_ratio'),
                        'n_cells': row.get('n_cells', 1),  # For weighted averaging
                        'skipped': row.get('skipped', False)
                    }
            
            # Store celltype results in the expected nested structure for plotting functions
            if celltype_results:
                results_dict[f'{group_type}_celltype_celltype_results'] = celltype_results
                logger.info(f"  Stored {len(celltype_results)} celltypes in celltype_results dict")
            else:
                logger.warning(f"  No celltype_results found!")
            
            # Compute weighted and macro aggregated summary statistics from celltype results
            if celltype_results:
                # IMPORTANT: Filter out skipped celltypes first (matching old code behavior)
                valid_celltype_results = {
                    ct: res for ct, res in celltype_results.items() 
                    if not res.get('skipped', False)
                }
                
                logger.info(f"  Computing aggregated statistics: {len(celltype_results)} total, {len(valid_celltype_results)} valid (non-skipped)")
                
                # DEBUG: Show sample of valid celltype_results values
                for ct_name, ct_vals in list(valid_celltype_results.items())[:2]:
                    logger.info(f"    Sample {ct_name}: mse_test_probe={ct_vals.get('mse_test_probe')}, n_cells={ct_vals.get('n_cells')}")
                
                if not valid_celltype_results:
                    logger.warning(f"  All celltypes were skipped! Cannot compute aggregated statistics.")
                    # Don't create summary if no valid results
                else:
                    # Weighted average (weighted by cell count) - filter NaN values
                    total_cells = sum(res['n_cells'] for res in valid_celltype_results.values())
                    
                    # MSE baseline - weighted
                    mse_test_baseline_values = [(res['mse_test_baseline'], res['n_cells']) 
                                                for res in valid_celltype_results.values() 
                                                if pd.notna(res['mse_test_baseline'])]
                    weighted_mse_test_baseline = (sum(val * n for val, n in mse_test_baseline_values) / 
                                              sum(n for _, n in mse_test_baseline_values)) if mse_test_baseline_values else np.nan
                    
                    # MSE probe - weighted
                    mse_test_probe_values = [(res['mse_test_probe'], res['n_cells']) 
                                            for res in valid_celltype_results.values() 
                                            if pd.notna(res['mse_test_probe'])]
                    weighted_mse_test_probe = (sum(val * n for val, n in mse_test_probe_values) / 
                                           sum(n for _, n in mse_test_probe_values)) if mse_test_probe_values else np.nan
                    
                    # ExpVar baseline - weighted
                    expvar_test_baseline_values = [(res['expvar_test_baseline'], res['n_cells']) 
                                                   for res in valid_celltype_results.values() 
                                                   if pd.notna(res['expvar_test_baseline'])]
                    weighted_expvar_test_baseline = (sum(val * n for val, n in expvar_test_baseline_values) / 
                                                 sum(n for _, n in expvar_test_baseline_values)) if expvar_test_baseline_values else np.nan
                    
                    # ExpVar probe - weighted
                    expvar_test_probe_values = [(res['expvar_test_probe'], res['n_cells']) 
                                               for res in valid_celltype_results.values() 
                                               if pd.notna(res['expvar_test_probe'])]
                    weighted_expvar_test_probe = (sum(val * n for val, n in expvar_test_probe_values) / 
                                              sum(n for _, n in expvar_test_probe_values)) if expvar_test_probe_values else np.nan
                    
                    # Macro average (unweighted - equal weight per celltype) - filter NaN values
                    macro_mse_test_baseline = np.nanmean([res['mse_test_baseline'] for res in valid_celltype_results.values()])
                    macro_mse_test_probe = np.nanmean([res['mse_test_probe'] for res in valid_celltype_results.values()])
                    macro_expvar_test_baseline = np.nanmean([res['expvar_test_baseline'] for res in valid_celltype_results.values()])
                    macro_expvar_test_probe = np.nanmean([res['expvar_test_probe'] for res in valid_celltype_results.values()])
                    
                    logger.info(f"  Computed weighted/macro MSE: {weighted_mse_test_probe:.2f}/{macro_mse_test_probe:.2f}")
                    logger.info(f"  Computed weighted/macro ExpVar: {weighted_expvar_test_probe:.4f}/{macro_expvar_test_probe:.4f}")
                    
                    results_dict[f'{group_type}_celltype_summary'] = {
                        'weighted_mse_test_baseline': weighted_mse_test_baseline,
                        'weighted_mse_test_probe': weighted_mse_test_probe,
                        'weighted_expvar_test_baseline': weighted_expvar_test_baseline,
                        'weighted_expvar_test_probe': weighted_expvar_test_probe,
                        'macro_mse_test_baseline': macro_mse_test_baseline,
                        'macro_mse_test_probe': macro_mse_test_probe,
                        'macro_expvar_test_baseline': macro_expvar_test_baseline,
                        'macro_expvar_test_probe': macro_expvar_test_probe
                    }
        else:
            # Global evaluation (no celltype breakdown)
            # Find the row with analysis_type == 'global'
            global_rows = gene_list_df[gene_list_df.get('analysis_type', '') == 'global']
            if len(global_rows) > 0:
                row = global_rows.iloc[0]
            else:
                # Fallback to first row if no 'global' type found
                row = gene_list_df.iloc[0]
            
            results_dict[f'{group_type}_global'] = {
                'mse_train_baseline': row.get('mse_train_baseline'),
                'mse_test_baseline': row.get('mse_test_baseline'),
                'mse_test_probe': row.get('mse_test_probe'),
                'expvar_train_baseline': row.get('expvar_train_baseline'),
                'expvar_test_baseline': row.get('expvar_test_baseline'),
                'expvar_test_probe': row.get('expvar_test_probe'),
                'mse_ratio': row.get('mse_ratio'),
                'expvar_ratio': row.get('expvar_ratio')
            }
            
            # For global results, also create a summary structure
            # that mimics the per-celltype summary format for unified plotting
            results_dict[f'{group_type}_global_summary'] = {
                'weighted_mse_test_baseline': row.get('mse_test_baseline'),
                'weighted_mse_test_probe': row.get('mse_test_probe'),
                'weighted_expvar_test_baseline': row.get('expvar_test_baseline'),
                'weighted_expvar_test_probe': row.get('expvar_test_probe'),
                'macro_mse_test_baseline': row.get('mse_test_baseline'),  # Same as weighted for global
                'macro_mse_test_probe': row.get('mse_test_probe'),
                'macro_expvar_test_baseline': row.get('expvar_test_baseline'),
                'macro_expvar_test_probe': row.get('expvar_test_probe')
            }
        
        evaluation_results[gene_list_name] = results_dict
    
    return evaluation_results


# ===================================================================
# LOGGING SETUP
# ===================================================================

def setup_logging(log_file=None):
    """Configure logging to both file and console"""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

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


def load_variability_results(results_dir, panel_names):
    """
    Load variability evaluation results for specified panels.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing result CSV files (either subdirectories or consolidated files)
    panel_names : list
        List of panel names to load results for
        
    Returns:
    --------
    dict
        Dictionary with keys: 'nmf', 'mapping', 'celltype_specific'
        Each containing a filtered DataFrame with only the requested panels
    """
    logger.info("Loading variability evaluation results...")
    
    results = {}
    
    # Check if results are in subdirectories or consolidated files
    subdirs = ['nmf', 'mapping']
    # Use subdirectory loading if ANY subdirectory exists
    use_subdirs = any(os.path.isdir(os.path.join(results_dir, sd)) for sd in subdirs)
    
    if use_subdirs:
        logger.info("  Loading from subdirectories (individual CSV files per panel)")
        
        # Map subdirectory names to result keys
        subdir_mapping = {
            'nmf': 'nmf',
            'mapping': 'mapping'
        }
        
        for subdir, result_key in subdir_mapping.items():
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
        
    else:
        # Original logic for consolidated files
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
                         png_dpi=300, plot_clustering=True, plot_neighborhood=True, 
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
        plot_optimal_neighborhood_preservation(
            results['neighborhood'], 
            neighborhood_dir, 
            PNG_DPI=png_dpi
        )
        
        # Skip heatmap for Category 9 - not meaningful with multiple datasets having same k values
        if not (group_name and 'category-9' in group_name.lower()):
            plot_neighborhood_preservation_heatmap(
                results['neighborhood'], 
                neighborhood_dir, 
                PNG_DPI=png_dpi
            )
        else:
            logger.info("Skipping heatmap for Category 9 (not applicable for this comparison)")

        
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
        plot_celltype_accuracy_barchart(
            results['celltype'], 
            celltype_dir, 
            PNG_DPI=png_dpi
        )
        
        logger.info(f"✓ Celltype plots saved to: {celltype_dir}")


def create_variability_plots(results, output_dir, group_name, color_map=None,
                            png_dpi=300, plot_nmf=True, plot_mapping=True,
                            plot_celltype_specific=True, use_hardcoded_colors=False, external_names=None):
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
    """
    logger.info(f"Creating variability plots in: {output_dir}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset names for color mapping
    dataset_names = []
    if 'nmf' in results and not results['nmf'].empty:
        dataset_names = results['nmf']['gene_list'].unique().tolist()
    elif 'mapping' in results and not results['mapping'].empty:
        dataset_names = results['mapping']['gene_list'].unique().tolist()
    
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

        logger.info(f"✓ NMF plots saved to: {nmf_dir}")
    
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
    import os
    from pathlib import Path

    tangram_root = Path(tangram_results_dir)
    out_root = Path(output_dir) / "reconstruction"
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
        if nmf_results is not None and "nmf" in nmf_results:
            mech_df = nmf_results["nmf"]
            if not mech_df.empty and "gene_list" in mech_df.columns:
                panel_df = mech_df[
                    (mech_df["gene_list"] == panel_name) &
                    (mech_df.get("analysis_type", "per_celltype") == "per_celltype")
                ]
                if not panel_df.empty and "celltype" in panel_df.columns:
                    nmf_ct_results = {
                        row["celltype"]: row.to_dict()
                        for _, row in panel_df.iterrows()
                        if pd.notna(row.get("celltype"))
                    }
                # Summary (macro/weighted) from global row
                global_row = mech_df[
                    (mech_df["gene_list"] == panel_name) &
                    (mech_df.get("analysis_type", "global") == "global")
                ]
                if not global_row.empty:
                    nmf_summary = global_row.iloc[0].to_dict()

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
        )

    logger.info("Reconstruction comparison plots saved to: %s", out_root)


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
                       choices=['baseline', 'variability', 'both'],
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
    
    parser.add_argument('--png_dpi', type=int, default=300,
                       help='DPI for saved plots')
    
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
    if args.evaluation_type == 'both':
        if not args.baseline_results_dir or not args.variability_results_dir:
            logger.error("Both --baseline_results_dir and --variability_results_dir required when evaluation_type='both'")
            return 1
    else:
        if not args.results_dir:
            logger.error("--results_dir required when evaluation_type is 'baseline' or 'variability'")
            return 1
    
        # Process baseline evaluation
    if args.evaluation_type in ['baseline', 'both']:
        results_dir = args.baseline_results_dir if args.evaluation_type == 'both' else args.results_dir
        
        logger.info("")
        logger.info("="*80)
        logger.info("BASELINE EVALUATION PLOTS")
        logger.info("="*80)
        
        baseline_results = load_baseline_results(results_dir, panel_names)
        
        baseline_output = os.path.join(args.output_dir, 'Baseline') if args.evaluation_type == 'both' else args.output_dir
        
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
    if args.evaluation_type in ['variability', 'both']:
        results_dir = args.variability_results_dir if args.evaluation_type == 'both' else args.results_dir
        
        logger.info("")
        logger.info("="*80)
        logger.info("VARIABILITY EVALUATION PLOTS")
        logger.info("="*80)
        
        variability_results = load_variability_results(results_dir, panel_names)
        
        variability_output = os.path.join(args.output_dir, 'Variability') if args.evaluation_type == 'both' else args.output_dir
        
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
            external_names=external_names  # Pass for proper color assignment
        )
    
    # Reconstruction comparison plots (Tangram vs NMF)
    if getattr(args, 'tangram_results_dir', None):
        logger.info("")
        logger.info("="*80)
        logger.info("RECONSTRUCTION COMPARISON PLOTS (Tangram vs NMF)")
        logger.info("="*80)
        # Load NMF results for comparison if variability results are available
        nmf_for_comparison = None
        if args.evaluation_type in ['variability', 'both']:
            results_dir_var = args.variability_results_dir if args.evaluation_type == 'both' else args.results_dir
            nmf_for_comparison = load_variability_results(results_dir_var, panel_names)
        reconstruction_output = os.path.join(args.output_dir, 'Reconstruction') \
            if args.evaluation_type == 'both' else args.output_dir
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
