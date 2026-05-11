"""NMF-independent evaluation plotting script.

Creates comparison plots for NMF-independent evaluation results,
with separate plots for each method showing all datasets together.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _constants import (
    COL_EXPVAR_TEST_PROBE,
    COL_MACRO_EXPVAR,
    COL_MACRO_MSE,
    COL_MSE_TEST_PROBE,
    COL_N_CELLS,
    COL_RMSE_TEST_PROBE,
    COL_SKIPPED,
    COL_WEIGHTED_EXPVAR,
    COL_WEIGHTED_MSE,
    DEFAULT_PNG_DPI,
)

logger = logging.getLogger(__name__)

__all__ = [
    "calculate_macro_metrics",
    "calculate_weighted_metrics",
    "main",
]

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

##############################################################################
# METRIC CALCULATION FUNCTIONS
##############################################################################

def calculate_macro_metrics(df):
    """
    Calculate macro-averaged metrics from per-cell-type results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with per-cell-type metrics
        
    Returns:
    --------
    metrics : dict
        Dictionary with macro metrics
    """
    # Filter out skipped cell types
    valid_df = df[~df[COL_SKIPPED]].copy()
    
    if len(valid_df) == 0:
        return {
            'macro_mse': np.nan,
            'macro_expvar': np.nan,
            'macro_rmse': np.nan
        }
    
    return {
        'macro_mse': valid_df[COL_MSE_TEST_PROBE].mean(),
        'macro_expvar': valid_df[COL_EXPVAR_TEST_PROBE].mean(),
        'macro_rmse': valid_df[COL_RMSE_TEST_PROBE].mean()
    }


def calculate_weighted_metrics(df):
    """
    Calculate weighted-averaged metrics from per-cell-type results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with per-cell-type metrics
        
    Returns:
    --------
    metrics : dict
        Dictionary with weighted metrics
    """
    # Filter out skipped cell types
    valid_df = df[~df[COL_SKIPPED]].copy()
    
    if len(valid_df) == 0:
        return {
            'weighted_mse': np.nan,
            'weighted_expvar': np.nan,
            'weighted_rmse': np.nan
        }
    
    total_cells = valid_df[COL_N_CELLS].sum()
    
    return {
        'weighted_mse': (valid_df[COL_MSE_TEST_PROBE] * valid_df[COL_N_CELLS]).sum() / total_cells,
        'weighted_expvar': (valid_df[COL_EXPVAR_TEST_PROBE] * valid_df[COL_N_CELLS]).sum() / total_cells,
        'weighted_rmse': (valid_df[COL_RMSE_TEST_PROBE] * valid_df[COL_N_CELLS]).sum() / total_cells
    }


##############################################################################
# DATA COLLECTION
##############################################################################

def extract_gene_count(dataset_name):
    """
    Extract gene count (100 or 200) from dataset name.
    
    Parameters:
    -----------
    dataset_name : str
        Dataset name (e.g., 'Xenium-Filter_All-Genes_hvg_100')
        
    Returns:
    --------
    gene_count : int or None
        Gene count (100 or 200) or None if not found
    """
    if '_100' in dataset_name:
        return 100
    elif '_200' in dataset_name:
        return 200
    else:
        return None

def simplify_dataset_label(dataset_name):
    """
    Simplify dataset label by removing redundant prefix.
    
    Parameters:
    -----------
    dataset_name : str
        Full dataset name (e.g., 'Xenium-Filter_All-Genes_hvg_100')
        
    Returns:
    --------
    simplified_name : str
        Simplified name (e.g., 'hvg_100' or 'nmf_global_method_b_200')
    """
    # Remove common prefix 'Xenium-Filter_All-Genes_'
    if 'Xenium-Filter_All-Genes_' in dataset_name:
        simplified = dataset_name.replace('Xenium-Filter_All-Genes_', '')
    elif 'Spapros_' in dataset_name:
        # For Spapros datasets, keep the format
        simplified = dataset_name
    else:
        simplified = dataset_name
    
    return simplified

def collect_method_results(base_dir, method_name):
    """
    Collect all results for a specific method across all datasets.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing all dataset folders
    method_name : str
        Method name: 'neural_network', 'LVAE', 'tangram'
        
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with columns: dataset, macro_mse, macro_expvar, weighted_mse, weighted_expvar
    """
    results = []
    
    # Iterate through all dataset directories
    base_path = Path(base_dir)
    for dataset_dir in sorted(base_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        # Skip the comparison_plots directory itself
        if dataset_dir.name.startswith('comparison_plots'):
            continue
        
        # Look for method results
        method_results_dir = dataset_dir / 'results' / 'PerCellType_Reconstruction' / method_name
        
        if not method_results_dir.exists():
            logging.warning(f"No {method_name} results found for {dataset_dir.name}")
            continue
        
        # Find CSV file
        csv_files = list(method_results_dir.glob('*.csv'))
        if len(csv_files) == 0:
            logging.warning(f"No CSV files found in {method_results_dir}")
            continue
        
        csv_file = csv_files[0]  # Take first CSV file
        
        logging.info(f"Reading {method_name} results from {dataset_dir.name}: {csv_file.name}")
        
        try:
            # Read per-cell-type metrics
            df = pd.read_csv(csv_file)
            
            # Calculate macro and weighted metrics
            macro_metrics = calculate_macro_metrics(df)
            weighted_metrics = calculate_weighted_metrics(df)
            
            # Extract dataset name from directory
            dataset_name = dataset_dir.name
            gene_count = extract_gene_count(dataset_name)
            simplified_label = simplify_dataset_label(dataset_name)
            
            # Store results
            results.append({
                'dataset': dataset_name,
                'dataset_label': simplified_label,
                'gene_count': gene_count,
                'macro_mse': macro_metrics['macro_mse'],
                'macro_expvar': macro_metrics['macro_expvar'],
                'macro_rmse': macro_metrics['macro_rmse'],
                'weighted_mse': weighted_metrics['weighted_mse'],
                'weighted_expvar': weighted_metrics['weighted_expvar'],
                'weighted_rmse': weighted_metrics['weighted_rmse']
            })
            
        except Exception as e:
            logging.error(f"Error processing {csv_file}: {e}")
            continue
    
    if len(results) == 0:
        logging.warning(f"No valid results found for method: {method_name}")
        return None
    
    results_df = pd.DataFrame(results)
    
    logging.info(f"Collected {len(results_df)} datasets for {method_name}")
    
    return results_df


##############################################################################
# PLOTTING FUNCTIONS
##############################################################################

def plot_method_mse_comparison(results_df, method_name, gene_count, output_dir, dpi=300):
    """
    Create MSE comparison plot (macro + weighted) for a specific method and gene count.
    Uses horizontal bar plot style matching the evaluation module.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with aggregated metrics per dataset
    method_name : str
        Method name for plot title
    gene_count : int
        Gene count (100 or 200)
    output_dir : str
        Output directory for plots
    dpi : int
        Plot resolution
    """
    if results_df is None or len(results_df) == 0:
        logging.warning(f"No data to plot for {method_name}")
        return
    
    # Create figure with 2 subplots (macro and weighted)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(results_df) * 0.4)))
    
    # Define colors for datasets - use a vibrant color palette
    # Choose between different palettes based on number of datasets
    if len(results_df) <= 10:
        colors = sns.color_palette("Set2", len(results_df))
    else:
        colors = sns.color_palette("tab20", len(results_df))
    
    # ========================================================================
    # Macro MSE
    # ========================================================================
    ax = axes[0]
    df_sorted = results_df.sort_values('macro_mse', ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    
    bars = ax.barh(y_pos, df_sorted['macro_mse'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['dataset_label'], fontsize=10)
    ax.set_xlabel('MSE (lower is better)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('Macro Average MSE', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Auto-scale x-axis
    max_val = df_sorted['macro_mse'].max()
    ax.set_xlim([0, max_val * 1.15])
    
    # Add value labels (positioned intelligently)
    for i, (bar, value) in enumerate(zip(bars, df_sorted['macro_mse'])):
        if not np.isnan(value):
            if value > max_val * 0.15:
                # Inside bar (black text for readability)
                ax.text(value - max_val * 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.1f}', ha='right', va='center', fontsize=8, 
                       color='black', fontweight='bold')
            else:
                # Small values - outside right
                ax.text(value + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.1f}', ha='left', va='center', fontsize=8)
    
    # ========================================================================
    # Weighted MSE
    # ========================================================================
    ax = axes[1]
    df_sorted = results_df.sort_values('weighted_mse', ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    
    bars = ax.barh(y_pos, df_sorted['weighted_mse'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['dataset_label'], fontsize=10)
    ax.set_xlabel('MSE (lower is better)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('Weighted Average MSE', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Auto-scale x-axis
    max_val = df_sorted['weighted_mse'].max()
    ax.set_xlim([0, max_val * 1.15])
    
    # Add value labels (positioned intelligently)
    for i, (bar, value) in enumerate(zip(bars, df_sorted['weighted_mse'])):
        if not np.isnan(value):
            if value > max_val * 0.15:
                # Inside bar (black text for readability)
                ax.text(value - max_val * 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.1f}', ha='right', va='center', fontsize=8, 
                       color='black', fontweight='bold')
            else:
                # Small values - outside right
                ax.text(value + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.1f}', ha='left', va='center', fontsize=8)
    
    plt.suptitle(f'Per-Cell-Type Reconstruction: {method_name} MSE Comparison ({gene_count} genes)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f'{method_name}_{gene_count}genes_mse_comparison.png')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved: {output_file}")


def plot_method_expvar_comparison(results_df, method_name, gene_count, output_dir, dpi=300):
    """
    Create Explained Variance comparison plot (macro + weighted) for a specific method and gene count.
    Uses horizontal bar plot style matching the evaluation module.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with aggregated metrics per dataset
    method_name : str
        Method name for plot title
    gene_count : int
        Gene count (100 or 200)
    output_dir : str
        Output directory for plots
    dpi : int
        Plot resolution
    """
    if results_df is None or len(results_df) == 0:
        logging.warning(f"No data to plot for {method_name}")
        return
    
    # Create figure with 2 subplots (macro and weighted)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(results_df) * 0.4)))
    
    # Define colors for datasets - use a vibrant color palette
    # Choose between different palettes based on number of datasets
    if len(results_df) <= 10:
        colors = sns.color_palette("Set2", len(results_df))
    else:
        colors = sns.color_palette("tab20", len(results_df))
    
    # ========================================================================
    # Macro Explained Variance
    # ========================================================================
    ax = axes[0]
    df_sorted = results_df.sort_values('macro_expvar', ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    
    bars = ax.barh(y_pos, df_sorted['macro_expvar'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['dataset_label'], fontsize=10)
    ax.set_xlabel('Explained Variance (higher is better)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('Macro Average Explained Variance', fontsize=13, fontweight='bold')
    
    # Auto-scale x-axis to accommodate all values (including negative outliers)
    min_val = df_sorted['macro_expvar'].min()
    max_val = df_sorted['macro_expvar'].max()
    if min_val < 0:
        # Extend left for negative values, extend right for positive values
        ax.set_xlim([min_val * 1.1, max(1.0, max_val * 1.15)])
    else:
        ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels (positioned intelligently)
    for i, (bar, value) in enumerate(zip(bars, df_sorted['macro_expvar'])):
        if not np.isnan(value):
            if value > 0.15:
                # Inside bar (black text for readability)
                ax.text(value - 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='right', va='center', fontsize=8, 
                       color='black', fontweight='bold')
            elif value >= 0:
                # Small positive values - outside right
                ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='left', va='center', fontsize=8)
            else:
                # Negative values - outside left of bar
                ax.text(value - 0.05, bar.get_y() + bar.get_height()/2,
                       f'{value:.2f}', ha='right', va='center', fontsize=8, color='red')
    
    # ========================================================================
    # Weighted Explained Variance
    # ========================================================================
    ax = axes[1]
    df_sorted = results_df.sort_values('weighted_expvar', ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    
    bars = ax.barh(y_pos, df_sorted['weighted_expvar'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['dataset_label'], fontsize=10)
    ax.set_xlabel('Explained Variance (higher is better)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('Weighted Average Explained Variance', fontsize=13, fontweight='bold')
    
    # Auto-scale x-axis to accommodate all values (including negative outliers)
    min_val = df_sorted['weighted_expvar'].min()
    max_val = df_sorted['weighted_expvar'].max()
    if min_val < 0:
        # Extend left for negative values, extend right for positive values
        ax.set_xlim([min_val * 1.1, max(1.0, max_val * 1.15)])
    else:
        ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels (positioned intelligently)
    for i, (bar, value) in enumerate(zip(bars, df_sorted['weighted_expvar'])):
        if not np.isnan(value):
            if value > 0.15:
                # Inside bar (black text for readability)
                ax.text(value - 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='right', va='center', fontsize=8, 
                       color='black', fontweight='bold')
            elif value >= 0:
                # Small positive values - outside right
                ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='left', va='center', fontsize=8)
            else:
                # Negative values - outside left of bar
                ax.text(value - 0.05, bar.get_y() + bar.get_height()/2,
                       f'{value:.2f}', ha='right', va='center', fontsize=8, color='red')
    
    plt.suptitle(f'Per-Cell-Type Reconstruction: {method_name} Explained Variance Comparison ({gene_count} genes)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f'{method_name}_{gene_count}genes_expvar_comparison.png')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved: {output_file}")


##############################################################################
# MAIN FUNCTION
##############################################################################

def main():
    parser = argparse.ArgumentParser(
        description='Create comparison plots for NMF-independent evaluation, grouped by method'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Base directory containing all dataset evaluation folders'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for plots (default: base_dir/comparison_plots_by_method)'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['neural_network', 'LVAE', 'tangram'],
        help='Methods to plot (default: neural_network LVAE tangram)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Plot resolution (default: 300)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_dir, 'comparison_plots_by_method')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("="*80)
    logging.info("NMF-INDEPENDENT EVALUATION: PLOTS BY METHOD")
    logging.info("="*80)
    logging.info(f"Base directory: {args.base_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Methods: {', '.join(args.methods)}")
    logging.info("")
    
    # Process each method
    for method_name in args.methods:
        logging.info(f"\nProcessing method: {method_name}")
        logging.info("-"*80)
        
        # Collect results for this method
        results_df = collect_method_results(args.base_dir, method_name)
        
        if results_df is not None:
            # Split by gene count
            for gene_count in [100, 200]:
                df_subset = results_df[results_df['gene_count'] == gene_count].copy()
                
                if len(df_subset) == 0:
                    logging.warning(f"No {gene_count}-gene datasets found for {method_name}")
                    continue
                
                logging.info(f"  Processing {gene_count}-gene datasets ({len(df_subset)} datasets)")
                
                # Create MSE comparison plot
                plot_method_mse_comparison(df_subset, method_name, gene_count, args.output_dir, dpi=args.dpi)
                
                # Create Explained Variance comparison plot
                plot_method_expvar_comparison(df_subset, method_name, gene_count, args.output_dir, dpi=args.dpi)
                
                # Save metrics to CSV
                csv_file = os.path.join(args.output_dir, f'{method_name}_{gene_count}genes_metrics.csv')
                df_subset.to_csv(csv_file, index=False)
                logging.info(f"  Saved metrics: {csv_file}")
        else:
            logging.warning(f"Skipping {method_name} - no valid results found")
    
    logging.info("\n" + "="*80)
    logging.info("PLOTTING COMPLETE")
    logging.info("="*80)
    logging.info(f"Plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
