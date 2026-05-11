"""Gene selection result visualization utilities.

This module contains plotting functions for visualizing gene selection
results, including feature importance, confusion matrices, and gene
source distributions.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

from _constants import (
    COL_ACCURACY,
    COL_FEATURE_IMPORTANCE,
    COL_GENE,
    DEFAULT_PNG_DPI,
)

logger = logging.getLogger(__name__)

__all__ = [
    "plot_confusion_matrix",
    "plot_feature_importances",
    "plot_f1_distribution",
    "plot_final_gene_dotplot",
    "plot_gene_count_comparison",
    "plot_gene_source_distribution",
]

def plot_confusion_matrix(conf_matrix, class_names, seed, fold_idx, results_dir):
    """Plot confusion matrix for a specific fold"""
    plt.figure(figsize=(10, 8))
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    sns.heatmap(conf_matrix_df, annot=True, cmap='Blues', fmt='g')
    plt.title(f'Confusion Matrix - Seed {seed} - Fold {fold_idx+1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_seed{seed}_fold{fold_idx+1}.png'), dpi=DEFAULT_PNG_DPI)
    plt.close()

def plot_feature_importances(feature_importances, seed, fold_idx, results_dir):
    """Plot feature importances for a specific fold"""
    top_n = min(30, len(feature_importances))
    plt.figure(figsize=(12, 8))
    sns.barplot(x=COL_FEATURE_IMPORTANCE, y=COL_GENE, data=feature_importances.head(top_n))
    plt.title(f'Top {top_n} Gene Importances - Seed {seed} - Fold {fold_idx+1}')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'feature_imp_seed{seed}_fold{fold_idx+1}.png'), dpi=DEFAULT_PNG_DPI)
    plt.close()

def plot_f1_distribution(f1_scores, results_dir):
    """Plot F1 score distribution across folds"""
    plt.figure(figsize=(10, 6))
    sns.histplot(f1_scores, kde=True)
    plt.axvline(np.mean(f1_scores), color='red', linestyle='--', 
                label=f'Mean F1 score: {np.mean(f1_scores):.4f}')
    plt.title('Macro F1 Score Distribution Across All Folds')
    plt.xlabel('F1 Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'f1_score_distribution.png'), dpi=DEFAULT_PNG_DPI)
    plt.close()

def plot_final_gene_dotplot(adata, probeset_genes, deg_genes, nmf_genes, gene_scores=None, groupby='original', 
                           output_path=None, figsize=(28, 16), cmap='viridis', grid=True, filename_prefix=None):
    """
    Create an enhanced dotplot of the probeset genes with genes ordered by their combined scores,
    with clear visual indicators for gene sources (NMF-only, DEG-only, or both).
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data matrix.
    probeset_genes : list
        List of final selected probeset genes to display
    deg_genes : list or set
        List of genes selected from the DEG-based decision trees
    nmf_genes : list or set
        List of genes selected from the NMF analysis
    gene_scores : dict, optional
        Dictionary mapping genes to their combined scores (default: None)
    groupby : str, optional
        Column name in adata.obs for grouping cells (default: 'original').
    output_path : str, optional
        Path to save the figure (default: None, which uses results_dir).
    figsize : tuple, optional
        Figure size (width, height) in inches.
    cmap : str, optional
        Colormap for dot expression values.
    grid : bool, optional
        Whether to show grid lines.
    """
    # Convert inputs to sets for easier operations
    probeset_set = set(probeset_genes)
    deg_set = set(deg_genes)
    nmf_set = set(nmf_genes)
    
    # Create sets for each category
    deg_only_set = probeset_set.intersection(deg_set) - nmf_set
    nmf_only_set = probeset_set.intersection(nmf_set) - deg_set
    both_set = probeset_set.intersection(deg_set).intersection(nmf_set)
    
    logging.info(f"Creating dotplot with {len(probeset_set)} probeset genes")
    logging.info(f"- {len(deg_only_set)} from DEG-based decision trees only")
    logging.info(f"- {len(nmf_only_set)} from NMF only")
    logging.info(f"- {len(both_set)} from both DEG and NMF")
    
    # Keep only genes present in the dataset
    available_genes = set(adata.var_names)
    available_probeset_genes = list(probeset_set.intersection(available_genes))
    
    if not available_probeset_genes:
        logging.error("No probeset genes found in the dataset. Cannot create dotplot.")
        return
    
    # Process gene scores if provided
    if gene_scores is not None:
        # Log gene_scores information for debugging
        if isinstance(gene_scores, pd.DataFrame):
            logging.info(f"Gene scores provided as DataFrame with columns: {gene_scores.columns.tolist()}")
            logging.info(f"Gene scores DataFrame shape: {gene_scores.shape}")
            if len(gene_scores) > 0:
                logging.info(f"First row sample: {gene_scores.iloc[0].to_dict()}")
        else:
            logging.info(f"Gene scores provided as dictionary with {len(gene_scores)} entries")
    if gene_scores is not None:
        # Check if gene_scores is a DataFrame (from the error message)
        if isinstance(gene_scores, pd.DataFrame):
            # Create a dictionary from the DataFrame's gene and score columns
            matching_gene_scores = {}
            for _, row in gene_scores.iterrows():
                if row[COL_GENE] in available_probeset_genes:
                    # Try different possible column names for the score
                    if 'selection_score' in row and pd.notna(row['selection_score']):
                        matching_gene_scores[row[COL_GENE]] = row['selection_score']
                    elif 'combined_score' in row and pd.notna(row['combined_score']):
                        matching_gene_scores[row[COL_GENE]] = row['combined_score']
                    elif 'score' in row and pd.notna(row['score']):
                        matching_gene_scores[row[COL_GENE]] = row['score']
                    elif 'metric_value' in row and pd.notna(row['metric_value']):
                        matching_gene_scores[row[COL_GENE]] = row['metric_value']
        else:
            # Original dictionary handling
            matching_gene_scores = {gene: score for gene, score in gene_scores.items() 
                                if gene in available_probeset_genes}
        
        if matching_gene_scores:
            # Sort genes by their combined scores in descending order
            plot_genes = sorted(available_probeset_genes, 
                            key=lambda g: matching_gene_scores.get(g, float('-inf')), 
                            reverse=True)
            logging.info(f"Ordered {len(plot_genes)} genes by their combined scores")
            
            # Log the top 5 genes and their scores for verification
            top_genes = plot_genes[:min(5, len(plot_genes))]
            top_scores = [matching_gene_scores.get(g, 'N/A') for g in top_genes]
            logging.info(f"Top 5 genes by score: {list(zip(top_genes, top_scores))}")
        else:
            plot_genes = available_probeset_genes
            logging.warning("No matching genes found between provided scores and available probeset genes")
            logging.warning(f"Available genes sample: {available_probeset_genes[:5]}")
            if isinstance(gene_scores, pd.DataFrame) and not gene_scores.empty:
                sample_genes = gene_scores[COL_GENE].tolist()[:5]
                logging.warning(f"Gene scores DataFrame sample genes: {sample_genes}")
    else:
        plot_genes = available_probeset_genes
        logging.info("No gene scores provided. Using original gene order.")

    # Try simpler scanpy visualization approach first
    try:

        # Set global font size for matplotlib
        matplotlib.rcParams.update({'font.size': 14})
        matplotlib.rcParams['axes.titlesize'] = 18
        matplotlib.rcParams['axes.labelsize'] = 16
        
        # Create a figure with adequate width for all genes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use scanpy's simpler dotplot with the ordered genes
        sc.pl.dotplot(
            adata, 
            var_names=plot_genes,
            groupby=groupby,
            standard_scale='var',
            cmap=cmap,
            show=False,
            var_group_rotation=90,
            var_group_positions=[]
        )
        
        # Save the figure using standard plt.savefig with explicit path
        scanpy_filename = f'{filename_prefix}_scanpy_final_gene_dotplot.png' if filename_prefix else 'scanpy_final_gene_dotplot.png'
        logging.info(f"Attempting to save scanpy plot to: {os.path.join(output_path, scanpy_filename)}")
        plt.savefig(os.path.join(output_path, scanpy_filename), dpi=300)
        logging.info(f"Successfully saved scanpy plot to: {os.path.join(output_path, scanpy_filename)}")
        
        plt.close()
        
    except Exception as e:
        logging.error(f"Error with simple scanpy visualization: {str(e)}. Trying detailed approach.") 

    ##########
    ##########
    # Manual implementation as fallback
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create temporary AnnData with only relevant genes in the desired order
    # This is important to maintain the gene ordering by score
    ordered_gene_indices = [list(adata.var_names).index(gene) for gene in plot_genes if gene in adata.var_names]
    if scipy.sparse.issparse(adata.X):
        ordered_X = adata.X[:, ordered_gene_indices]
    else:
        ordered_X = adata.X[:, ordered_gene_indices]
    
    # Create a new AnnData object with the ordered genes
    temp_adata = anndata.AnnData(X=ordered_X, obs=adata.obs.copy())
    temp_adata.var_names = [plot_genes[i] for i in range(len(ordered_gene_indices))]
    
    # Compute mean expression and percent expressed for each gene in each group
    mean_exp = pd.DataFrame(index=temp_adata.var_names, columns=adata.obs[groupby].cat.categories)
    pct_exp = pd.DataFrame(index=temp_adata.var_names, columns=adata.obs[groupby].cat.categories)
    
    for group in temp_adata.obs[groupby].cat.categories:
        group_cells = temp_adata[temp_adata.obs[groupby] == group]
        if scipy.sparse.issparse(group_cells.X):
            X = group_cells.X.toarray()
        else:
            X = group_cells.X

        # Calculate mean expression only over expressing cells (Scanpy logic)
        # Mask non-expressing cells as NaN, then take mean (ignoring NaN)
        X_masked = np.where(X > 0, X, np.nan)
        mean_values = np.nanmean(X_masked, axis=0)
        # If a gene is not expressed in any cell, np.nanmean returns nan; set these to 0
        mean_values = np.nan_to_num(mean_values, nan=0.0)
        mean_exp[group] = mean_values

        # Calculate percent of cells expressing each gene (unchanged)
        percent_expressed = np.sum(X > 0, axis=0) / X.shape[0] * 100
        pct_exp[group] = percent_expressed
    
    # After calculating mean expression, apply scaling similar to scanpy's approach
    def scale_genes_scanpy_style(expression_matrix, axis='var'):
        """
        Scale values between 0 and 1 following scanpy's standard_scale approach.
        Parameters:
        -----------
        expression_matrix : pandas.DataFrame
            Matrix to scale
        axis : str
            'var' to scale each column (variable/gene) independently
            'group' to scale each row (group) independently
        Returns:
        --------
        pandas.DataFrame
            Scaled matrix
        """
        result = expression_matrix.copy()
        if axis == 'var':  # Scale each column (variable/gene)
            for col in result.columns:
                result[col] = result[col] - result[col].min()
                max_val = result[col].max()
                if max_val != 0:
                    result[col] = result[col] / max_val
                else:
                    result[col] = 0
        elif axis == 'group':  # Scale each row (group)
            for idx in result.index:
                result.loc[idx] = result.loc[idx] - result.loc[idx].min()
                max_val = result.loc[idx].max()
                if max_val != 0:
                    result.loc[idx] = result.loc[idx] / max_val
                else:
                    result.loc[idx] = 0
        return result

    # Apply scaling to gene expression values
    mean_exp_scaled = scale_genes_scanpy_style(mean_exp)

    # Reshape the scaled data for plotting
    mean_exp_scaled_melted = mean_exp_scaled.reset_index().melt(id_vars='index', var_name=groupby, value_name='mean_expression')
    pct_exp_melted = pct_exp.reset_index().melt(id_vars='index', var_name=groupby, value_name='percent_expressed')

    # Merge with percent expressed data
    plot_data = mean_exp_scaled_melted.merge(pct_exp_melted, on=['index', groupby])

    # Create a mapping of gene categories
    gene_categories = {}
    for gene in temp_adata.var_names:
        if gene in both_set:
            gene_categories[gene] = 'both'
        elif gene in nmf_only_set:
            gene_categories[gene] = 'nmf_only'
        else:
            gene_categories[gene] = 'deg_only'

    # Use viridis colormap to match scanpy's default
    cmap_obj = 'viridis'

    # Calculate dot sizes based on percent expressed values
    # Normalize to a reasonable range for visualization
    min_size = 5  # Define min_size here
    max_size = 200  # Define max_size here
    dot_sizes = min_size + (plot_data['percent_expressed'] / 100.0) * (max_size - min_size)
    
    # Plot all gene expression dots
    scatter = ax.scatter(
        x=plot_data['index'], 
        y=plot_data[groupby],
        s=dot_sizes, 
        c=plot_data['mean_expression'],
        cmap=cmap_obj,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.2,
        zorder=2,
        vmin=0,
        vmax=1
    )
    
    # Add colorbar with improved formatting
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Mean expression in group', fontsize=14, labelpad=10)
    
    # Add grid
    if grid:
        ax.grid(True, color='gray', linewidth=0.1)
        ax.set_axisbelow(True)
    
    # Customize the plot with larger fonts
    ax.set_title('Expression of Probeset Genes across Cell Types', fontsize=18)
    ax.set_xlabel('Genes', fontsize=16)
    ax.set_ylabel('Cell Types', fontsize=16)
    
    # Set y-ticks with larger font
    ax.tick_params(axis='y', which='major', labelsize=14)
    
    # Create custom x-tick labels with colored text
    unique_genes = pd.unique(plot_data['index'])  # Preserves order
    ax.set_xticks(range(len(unique_genes)))
    
    # Create modified x-tick labels
    labels = []
    for gene in unique_genes:
        labels.append(gene)
    
    # Set the labels with improved position to avoid overlap
    ax.set_xticklabels(labels, rotation=90)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
    
    # Apply colors and larger font to the x-tick labels
    for i, label in enumerate(ax.get_xticklabels()):
        gene = unique_genes[i]
        if gene in both_set:
            label.set_color('blue')
            label.set_fontweight('bold')
            label.set_fontsize(14)
        elif gene in nmf_only_set:
            label.set_color('red')
            label.set_fontsize(14)
        else:
            label.set_fontsize(14)
    
    # Add a legend with improved positioning and larger font
    legend_elements = [
        patches.Patch(fill=False, edgecolor='red', linewidth=1.5, label='NMF-based genes'),
        patches.Patch(fill=False, edgecolor='blue', linewidth=1.5, label='Both DEG and NMF'),
        patches.Patch(fill=False, edgecolor='black', linewidth=1.5, label='DEG-analysis')
    ]
    
    # Create a separate legend for dot sizes
    # Figure out position for the dot size legend (right side of plot)
    ax_pos = ax.get_position()
    dot_legend_ax = fig.add_axes([ax_pos.x1 + 0.02, ax_pos.y0 + ax_pos.height * 0.4, 0.05, ax_pos.height * 0.2])
    dot_legend_ax.axis('off')
    
    # Sample a few actual dots from the plot to use their exact sizes in the legend
    pcts = [0, 25, 50, 75, 100]
    dot_legend_elements = []
    
    # The key insight: scatter() uses area, while Line2D uses diameter for markersize
    # So we need to properly convert between these two size measures
    
    for pct in pcts:
        # Calculate exact size using the same formula as in the scatter plot
        scatter_size = min_size + (pct / 100.0) * (max_size - min_size)
        
        # For Line2D, we need to use the square root of the area to get comparable visual size
        # and then calibrate with a scale factor (through trial and error)
        # Scale factor of 0.8 rather than 0.4 gives a better match
        legend_markersize = np.sqrt(scatter_size) * 0.8
        
        dot_legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label=f'{pct}%',
                     markerfacecolor='black', markersize=legend_markersize)
        )
    # Add the dot size legend with clear title
    dot_legend = dot_legend_ax.legend(
        handles=dot_legend_elements,
        loc='center left',
        title='Fraction expressed',
        frameon=True,
        fontsize=12
    )
    dot_legend.get_title().set_fontsize(14)
    
    # Position gene source legend above the dot size legend
    gene_legend_ax = fig.add_axes([ax_pos.x1 + 0.02, ax_pos.y0 + ax_pos.height * 0.65, 0.05, ax_pos.height * 0.2])
    gene_legend_ax.axis('off')
    gene_legend = gene_legend_ax.legend(
        handles=legend_elements,
        loc='center left',
        title="Gene Sources",
        fontsize=12,
        frameon=True
    )
    gene_legend.get_title().set_fontsize(14)
    
    # First apply tight layout to main plot only (exclude the legend axes)
    # This helps avoid the "not compatible with tight_layout" warning
    ax_position = ax.get_position()
    fig.tight_layout(rect=[ax_position.x0, ax_position.y0, ax_position.x1, ax_position.y1])
    
    # Then adjust to make room for legend and labels
    fig.subplots_adjust(right=0.78, bottom=0.25)  # More space for legend and x-labels
    
    # Save figure with extra space for legend and labels
    final_filename = f'{filename_prefix}_final_gene_dotplot.png' if filename_prefix else 'final_gene_dotplot.png'
    plt.savefig(os.path.join(output_path, final_filename), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_gene_count_comparison(celltype_gene_counts, results_dir):
    """
    Plot comparing the number of genes selected by Method A and Method B for each cell type
    
    Parameters:
    -----------
    celltype_gene_counts: dict
        Dictionary with method keys, each containing dict with celltype:count pairs
    results_dir: str
        Directory to save results to
    """
    plt.figure(figsize=(14, 8))
    
    # Extract data for plotting
    celltypes = sorted(set(list(celltype_gene_counts['method_a'].keys()) + 
                           list(celltype_gene_counts['method_b'].keys())))
    
    method_a_counts = [celltype_gene_counts['method_a'].get(ct, 0) for ct in celltypes]
    method_b_counts = [celltype_gene_counts['method_b'].get(ct, 0) for ct in celltypes]
    
    # Set width of bars
    bar_width = 0.35
    x_pos = np.arange(len(celltypes))
    
    # Create bars
    plt.bar(x_pos - bar_width/2, method_a_counts, bar_width, 
            label='Method A: Top 5 per factor', color='steelblue')
    plt.bar(x_pos + bar_width/2, method_b_counts, bar_width,
            label='Method B: Normalized weight > threshold', color='darkorange')
    
    # Add labels, title, and legend
    plt.xlabel('Cell Type', fontsize=12)
    plt.ylabel('Number of Unique Genes Selected', fontsize=12)
    plt.title('Number of Genes Selected by Each Method per Cell Type', fontsize=14)
    plt.xticks(x_pos, celltypes, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig(os.path.join(results_dir, 'gene_count_comparison_by_method.png'), dpi=DEFAULT_PNG_DPI)
    plt.close()
    logging.info(f"Saved gene count comparison plot to {results_dir}")


def create_gene_sharing_analysis_plot(gene_tracking_df, results_dir, method_name):
    """
    Create visualization plots for gene sharing analysis across celltypes.
    
    Parameters:
    -----------
    gene_tracking_df: pd.DataFrame
        DataFrame with gene tracking information
    results_dir: str
        Directory to save plots
    method_name: str
        Method identifier for file naming
    """
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Gene Sharing Analysis Across Cell Types ({method_name})', fontsize=16, fontweight='bold')
    
    # Plot 1: Bar chart of shared vs unique genes
    shared_count = len(gene_tracking_df[gene_tracking_df['is_shared'] == True])
    unique_count = len(gene_tracking_df[gene_tracking_df['is_shared'] == False])
    
    categories = ['Celltype-Specific\nGenes', 'Shared Across\nMultiple Celltypes']
    counts = [unique_count, shared_count]
    colors = ['steelblue', 'darkorange']
    
    bars = ax1.bar(categories, counts, color=colors)
    ax1.set_title('Gene Sharing Distribution', fontsize=14)
    ax1.set_ylabel('Number of Genes', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Plot 2: Histogram of number of celltypes per gene
    celltype_counts = gene_tracking_df['num_celltypes'].value_counts().sort_index()
    
    ax2.bar(celltype_counts.index, celltype_counts.values, color='darkgreen')
    ax2.set_title('Distribution of Genes by Number of Contributing Cell Types', fontsize=14)
    ax2.set_xlabel('Number of Cell Types Contributing Gene', fontsize=12)
    ax2.set_ylabel('Number of Genes', fontsize=12)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, count in enumerate(celltype_counts.values):
        ax2.text(celltype_counts.index[i], count + 0.5, str(count), 
                ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Pie chart of gene sharing
    sizes = [unique_count, shared_count]
    labels = [f'Celltype-Specific\n({unique_count})', f'Shared\n({shared_count})']
    colors_pie = ['steelblue', 'darkorange']
    
    if sum(sizes) > 0:
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie, 
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Gene Sharing Proportions', fontsize=14)
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax3.text(0.5, 0.5, 'No genes to display', ha='center', va='center')
        ax3.set_title('Gene Sharing Proportions', fontsize=14)
    
    # Plot 4: Summary statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Calculate additional statistics
    max_celltypes = gene_tracking_df['num_celltypes'].max()
    avg_celltypes = gene_tracking_df['num_celltypes'].mean()
    total_genes = len(gene_tracking_df)
    
    table_data = [
        ['Metric', 'Value', 'Percentage'],
        ['Total Unique Genes', total_genes, '100.0%'],
        ['Celltype-Specific Genes', unique_count, f'{unique_count/total_genes*100:.1f}%'],
        ['Shared Genes', shared_count, f'{shared_count/total_genes*100:.1f}%'],
        ['Max Celltypes per Gene', max_celltypes, '-'],
        ['Avg Celltypes per Gene', f'{avg_celltypes:.2f}', '-']
    ]
    
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0], 
                     cellLoc='center', loc='center', colWidths=[0.5, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color table header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Summary Statistics', fontsize=14)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'gene_sharing_analysis_{method_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved gene sharing analysis plot to {plot_path}")


def _convert_provenance_to_gene_source(provenance_df, reduction_type='NMF'):
    """
    Convert provenance report format to gene_source format for plotting.
    
    Parameters:
    -----------
    provenance_df : pd.DataFrame
        Provenance report with initial_source, source_details columns
    reduction_type : str
        'NMF' or 'PCA' for dimensionality reduction type
        
    Returns:
    --------
    pd.DataFrame with gene_source column added
    """
    df = provenance_df.copy()
    
    def map_to_gene_source(row):
        """Map provenance columns to gene_source string"""
        initial_source = row['initial_source']
        source_details = row.get('source_details', '')
        
        # Force-include / user-defined markers
        if initial_source == 'force_include':
            return 'user-defined-marker'
        
        # Replacement genes
        if initial_source == 'blacklist_replacement':
            replaced_gene = row.get('blacklist_replacement_for', 'unknown')
            return f'replacement for {replaced_gene} (blacklist)'
        
        if initial_source == 'xenium_replacement':
            replaced_gene = row.get('xenium_replacement_for', 'unknown')
            return f'replacement for {replaced_gene} (xenium)'
        
        if initial_source == 'odt_replacement':
            replaced_gene = row.get('odt_replacement_for', 'unknown')
            return f'replacement for {replaced_gene} (odt)'
        
        # Initial selection genes
        if initial_source == 'initial_selection':
            # Check source_details for specific source
            if source_details in ['DT', 'dt']:
                return 'DT'
            elif source_details.upper() in ['NMF', 'PCA']:
                return source_details.upper()
            elif 'both' in source_details.lower():
                return f'both (DT & {reduction_type})'
            elif 'dimred' in source_details.lower():
                return reduction_type
            elif 'gap-filling' in source_details.lower() or 'cell-type' in source_details.lower():
                if 'cell-type' in source_details.lower() or 'celltype' in source_details.lower():
                    return 'cell-type-specific gap-filling'
                elif 'global' in source_details.lower():
                    return 'global gap-filling'
                elif 'deg' in source_details.lower():
                    return 'DEG-based gap-filling'
                else:
                    return 'gap-filling'
            else:
                return source_details if source_details else 'initial_selection'
        
        return initial_source
    
    df['gene_source'] = df.apply(map_to_gene_source, axis=1)
    return df


def plot_gene_source_distribution(summary_df=None, summary_csv_path=None, results_dir=None, reduction_type='NMF', strategy_name=None, figsize=(12, 8)):
    """
    Create a bar plot showing the distribution of selected genes by source
    (DT-unique, dimred-unique, both, gap-filling, blacklist-replacement, xenium-replacement, odt-replacement).
    
    This function visualizes how many genes came from each source based on the
    gene_source column in the gene_selection_replacement_summary or provenance report.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame, optional
        DataFrame with gene selection summary (if already loaded)
    summary_csv_path : str, optional
        Path to the gene_selection_replacement_summary.csv file (if not loaded)
    results_dir : str
        Directory to save the output plot
    reduction_type : str, optional
        Type of dimensionality reduction used ('PCA' or 'NMF'), for plot labels
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 8))
        
    Returns:
    --------
    str : Path to the saved plot
    """
    try:
        # Load data if not provided
        if summary_df is None:
            # Prefer provenance report over old summary file
            provenance_path = os.path.join(results_dir, 'final_panel_with_provenance.csv') if results_dir else None
            
            if provenance_path and os.path.exists(provenance_path):
                logging.info(f"Loading provenance report: {provenance_path}")
                summary_df = pd.read_csv(provenance_path)
                # Map provenance columns to gene_source format for plotting
                summary_df = _convert_provenance_to_gene_source(summary_df, reduction_type)
            elif summary_csv_path and os.path.exists(summary_csv_path):
                logging.info(f"Loading gene selection summary: {summary_csv_path}")
                summary_df = pd.read_csv(summary_csv_path)
            else:
                logging.warning(f"No summary data found. Tried: {provenance_path}, {summary_csv_path}")
                return None
        
        logging.info(f"Loaded gene selection summary: {len(summary_df)} total genes")
        
        # Filter to only genes in final panel (exclude blacklisted genes)
        # Genes with 'replaced as blacklisted' or 'blacklisted in component' are NOT in final panel
        selected_df = summary_df[
            ~summary_df['gene_source'].str.contains('blacklisted', case=False, na=False)
        ].copy()
        logging.info(f"Final panel genes for plot: {len(selected_df)} genes (excluded blacklisted)")
        
        if len(selected_df) == 0:
            logging.warning("No selected genes found in summary file")
            return None
        
        # Categorize genes by their source (NEW FORMAT)
        def categorize_source(gene_source):
            """Categorize gene source into plot-friendly categories"""
            gene_source_lower = gene_source.lower()
            
            # Check for user-defined markers
            if 'user-defined' in gene_source_lower or 'user-specified' in gene_source_lower:
                return 'User-defined Markers'
            
            # Check for replacement types
            if 'replacement for' in gene_source_lower:
                if 'blacklist' in gene_source_lower:
                    return 'Blacklist Replacement'
                elif 'xenium' in gene_source_lower:
                    return 'Xenium Replacement'
                elif 'odt' in gene_source_lower:
                    return 'ODT Replacement'
                else:
                    return 'Other Replacement'
            
            # Check for gap-filling
            if 'gap-filling' in gene_source_lower or 'gap-fill' in gene_source_lower:
                if 'cell-type' in gene_source_lower:
                    return 'Gap-filling: Cell-type'
                elif 'global' in gene_source_lower:
                    return 'Gap-filling: Global'
                elif 'deg' in gene_source_lower:
                    return 'Gap-filling: DEG'
                else:
                    return 'Gap-filling: Other'
            
            # Check for both DT and dimred
            if 'both' in gene_source_lower:
                return f'Both (DT & {reduction_type})'
            
            # Check for DT-only
            if gene_source == 'DT' or gene_source_lower == 'dt':
                return 'DT-unique'
            
            # Check for dimred-only (NMF or PCA)
            if gene_source.upper() in ['NMF', 'PCA']:
                return f'{reduction_type}-unique'
            
            # Default: unknown
            return 'Other'
        
        selected_df['source_category'] = selected_df['gene_source'].apply(categorize_source)
        
        # Count genes by category
        category_counts = selected_df['source_category'].value_counts()
        
        # Define category order for consistent plotting
        preferred_order = [
            'User-defined Markers',
            'DT-unique',
            f'{reduction_type}-unique',
            f'Both (DT & {reduction_type})',
            'Gap-filling: Cell-type',
            'Gap-filling: Global',
            'Gap-filling: DEG',
            'Gap-filling: Other',
            'Blacklist Replacement',
            'Xenium Replacement',
            'ODT Replacement',
            'Other Replacement',
            'Other'
        ]
        
        # Reorder categories
        ordered_categories = [cat for cat in preferred_order if cat in category_counts.index]
        category_counts = category_counts.reindex(ordered_categories)
        
        # Create color palette
        colors = {
            'User-defined Markers': '#E67E22',  # Orange
            'DT-unique': '#E74C3C',  # Red
            f'{reduction_type}-unique': '#3498DB',  # Blue
            f'Both (DT & {reduction_type})': '#9B59B6',  # Purple
            'Gap-filling: Cell-type': '#2ECC71',  # Green
            'Gap-filling: Global': '#27AE60',  # Dark green
            'Gap-filling: DEG': '#16A085',  # Teal
            'Gap-filling: Other': '#1ABC9C',  # Light teal
            'Blacklist Replacement': '#F39C12',  # Gold
            'Xenium Replacement': '#E8B04B',  # Light gold
            'ODT Replacement': '#F4D03F',  # Yellow
            'Other Replacement': '#F8E27D',  # Pale yellow
            'Other': '#95A5A6'  # Gray
        }
        
        bar_colors = [colors.get(cat, '#95A5A6') for cat in category_counts.index]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(range(len(category_counts)), category_counts.values, color=bar_colors, 
                      edgecolor='black', linewidth=1.5, alpha=0.85)
        
        # Add value labels on top of bars
        for i, (bar, count) in enumerate(zip(bars, category_counts.values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({count/len(selected_df)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize plot
        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels(category_counts.index, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Number of Genes', fontsize=13, fontweight='bold')
        ax.set_xlabel('Gene Source', fontsize=13, fontweight='bold')
        
        # Include strategy name in title if provided
        title_suffix = f' ({strategy_name})' if strategy_name else ''
        ax.set_title(f'Distribution of Selected Genes by Source{title_suffix}\n(Total: {len(selected_df)} genes)', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Add grid for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add a box with summary statistics
        # For contribution calculation, we need to count ORIGINAL sources
        # including blacklisted genes that were replaced (not the replacements themselves)
        
        # Count genes from each original source (excluding blacklist replacements from the count)
        dt_contribution = 0
        dimred_contribution = 0
        
        # Include ALL genes from summary (including blacklisted ones) to get true contribution
        for _, row in summary_df.iterrows():
            gene_source_lower = row['gene_source'].lower()
            
            # Skip user-defined markers and actual replacement entries
            if 'user-defined' in gene_source_lower or 'replacement for' in gene_source_lower:
                continue
            
            # Check if gene came from DT (including blacklisted DT genes)
            if 'dt' in gene_source_lower and 'both' not in gene_source_lower:
                dt_contribution += 1
            # Check if gene came from both DT and dimred
            elif 'both' in gene_source_lower:
                dt_contribution += 1
                dimred_contribution += 1
            # Check if gene came from dimred only
            elif gene_source_lower in ['nmf', 'pca'] or reduction_type.lower() in gene_source_lower:
                dimred_contribution += 1
            # Gap-filling genes: celltype and global come from dimred, deg comes from DT
            elif 'gap-filling' in gene_source_lower:
                if 'deg' in gene_source_lower:
                    dt_contribution += 1
                elif 'cell-type' in gene_source_lower or 'global' in gene_source_lower:
                    dimred_contribution += 1
        
        textstr = '\n'.join([
            f'Total Selected: {len(selected_df)}',
            f'DT contribution: {dt_contribution}',
            f'{reduction_type} contribution: {dimred_contribution}',
            f'Gap-filled: {sum(v for k, v in category_counts.items() if "Gap-filling" in k)}',
            f'Blacklist: {category_counts.get("Blacklist Replacement", 0)}',
            f'Xenium: {category_counts.get("Xenium Replacement", 0)}',
            f'ODT: {category_counts.get("ODT Replacement", 0)}'
        ])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot with strategy name in filename if provided
        filename = f'gene_source_distribution_{strategy_name}.png' if strategy_name else 'gene_source_distribution.png'
        plot_path = os.path.join(results_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved gene source distribution plot to {plot_path}")
        logging.info(f"Category breakdown:\n{category_counts}")
        
        return plot_path
        
    except Exception as e:
        logging.error(f"Failed to create gene source distribution plot: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return None


