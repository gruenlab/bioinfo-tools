"""UMAP visualization for probe panels.

This script generates UMAP visualizations for probe panels and the full
transcriptome, with automatic detection of dimensionality reduction methods.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc

from _clustering_plots import plot_umap_for_representation
from _constants import DEFAULT_PNG_DPI

logger = logging.getLogger(__name__)

__all__ = [
    "find_h5ad_files_matching_panel",
    "main",
]

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def find_h5ad_files_matching_panel(preprocessed_dir, panel_pattern):
    """
    Find h5ad files matching a specific panel pattern.
    
    Parameters:
    -----------
    preprocessed_dir : str
        Directory containing preprocessed h5ad files
    panel_pattern : str
        Pattern to match (e.g., 'dt_nmf', 'dt_pca', 'Spapros')
        
    Returns:
    --------
    list
        List of matching h5ad file paths
    """
    # Convert pattern to match different naming conventions
    patterns_to_try = [
        f"*{panel_pattern}*.h5ad",
        f"*{panel_pattern.replace('_', '-')}*.h5ad",
        f"*{panel_pattern.capitalize()}*.h5ad",
        f"*{panel_pattern.upper()}*.h5ad",
        f"*{panel_pattern.lower()}*.h5ad"
    ]
    
    matching_files = []
    for pattern in patterns_to_try:
        search_pattern = os.path.join(preprocessed_dir, pattern)
        files = glob.glob(search_pattern)
        matching_files.extend(files)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in matching_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    return unique_files


def detect_dimensionality_reduction(adata):
    """
    Detect which dimensionality reduction was used (PCA, NMF, or both).
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object to check
        
    Returns:
    --------
    str
        'pca', 'nmf', or 'both'
    """
    has_pca = 'X_pca' in adata.obsm
    has_nmf = 'X_nmf' in adata.obsm
    
    if has_pca and has_nmf:
        return 'both'
    elif has_nmf:
        return 'nmf'
    elif has_pca:
        return 'pca'
    else:
        logging.warning("No dimensionality reduction found. Defaulting to PCA.")
        return 'pca'


def preprocess_for_umap(adata, n_neighbors=15, n_pcs=30, dimred_type='pca'):
    """
    Preprocess data for UMAP if not already done.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object to preprocess
    n_neighbors : int
        Number of neighbors for neighborhood graph
    n_pcs : int
        Number of principal components to use
    dimred_type : str
        'pca' or 'nmf'
        
    Returns:
    --------
    AnnData
        Preprocessed AnnData object
    """
    # Check if neighbors have been computed
    if 'neighbors' not in adata.uns:
        logging.info(f"Computing neighborhood graph using {dimred_type.upper()}...")
        
        # Determine the representation key
        if dimred_type == 'nmf':
            rep_key = 'X_nmf'
        else:
            rep_key = 'X_pca'
        
        # Check if representation exists
        if rep_key not in adata.obsm:
            logging.warning(f"{rep_key} not found. Computing {dimred_type.upper()}...")
            if dimred_type == 'pca':
                sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1))
            else:
                # For NMF, we need non-negative data
                logging.info("Computing NMF requires non-negative data...")
                # This is a simplified approach - adjust as needed
                from sklearn.decomposition import NMF
                nmf = NMF(n_components=min(n_pcs, adata.n_vars - 1), random_state=42)
                adata.obsm['X_nmf'] = nmf.fit_transform(np.abs(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X))
        
        # Compute neighbors
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=rep_key)
    
    # Check if Leiden clustering has been computed
    if 'leiden' not in adata.obs:
        logging.info("Computing Leiden clustering...")
        sc.tl.leiden(adata, resolution=1.0)
    
    return adata


def plot_panel_umaps(adata_path, panel_name, output_dir, celltype_col, png_dpi=300):
    """
    Plot UMAPs for a specific panel.
    
    Parameters:
    -----------
    adata_path : str
        Path to the h5ad file
    panel_name : str
        Name of the panel for labeling
    output_dir : str
        Directory to save plots
    celltype_col : str
        Column name for cell type annotations
    png_dpi : int
        Resolution for saved figures
        
    Returns:
    --------
    list
        Paths to saved figures
    """
    logging.info(f"Processing panel: {panel_name}")
    logging.info(f"  Loading: {os.path.basename(adata_path)}")
    
    # Load the data
    adata = sc.read_h5ad(adata_path)
    
    # Detect dimensionality reduction
    dimred_type = detect_dimensionality_reduction(adata)
    logging.info(f"  Detected dimensionality reduction: {dimred_type.upper()}")
    
    # Preprocess if needed
    adata = preprocess_for_umap(adata, dimred_type=dimred_type if dimred_type != 'both' else 'pca')
    
    # Create output directory for this panel
    panel_output_dir = os.path.join(output_dir, panel_name)
    os.makedirs(panel_output_dir, exist_ok=True)
    
    saved_plots = []
    
    # Plot UMAPs based on dimensionality reduction type
    if dimred_type == 'both':
        # Plot both PCA and NMF
        for rep in ['pca', 'nmf']:
            logging.info(f"  Generating UMAP for {rep.upper()}...")
            plot_path = plot_umap_for_representation(
                adata=adata,
                rep_name=rep,
                dataset_name=f"{panel_name}_{rep.upper()}",
                output_dir=panel_output_dir,
                dimensionality_reduction='both',
                PNG_DPI=png_dpi,
                celltype_col=celltype_col
            )
            if plot_path:
                saved_plots.append(plot_path)
    else:
        # Plot single dimensionality reduction
        logging.info(f"  Generating UMAP for {dimred_type.upper()}...")
        plot_path = plot_umap_for_representation(
            adata=adata,
            rep_name=dimred_type,
            dataset_name=panel_name,
            output_dir=panel_output_dir,
            dimensionality_reduction=dimred_type,
            PNG_DPI=png_dpi,
            celltype_col=celltype_col
        )
        if plot_path:
            saved_plots.append(plot_path)
    
    logging.info(f"  ✓ Completed {panel_name}: {len(saved_plots)} plot(s) saved")
    return saved_plots


# ===================================================================
# ARGUMENT PARSING
# ===================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Plot UMAPs for probe panels and full transcriptome',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--preprocessed_dir', required=True,
                       help='Directory containing preprocessed h5ad files')
    
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for UMAP plots')
    
    # Panel selection
    parser.add_argument('--panels',
                       help='Comma-separated list of panel patterns to plot (e.g., "dt_nmf,dt_pca,Spapros")')
    
    parser.add_argument('--full_transcriptome',
                       help='Path to full transcriptome h5ad file')
    
    # Optional arguments
    parser.add_argument('--celltype_col', default='cell_type',
                       help='Column name for cell type annotations (default: cell_type)')
    
    parser.add_argument('--png_dpi', type=int, default=300,
                       help='DPI for saved plots (default: 300)')
    
    parser.add_argument('--n_neighbors', type=int, default=15,
                       help='Number of neighbors for UMAP (default: 15)')
    
    parser.add_argument('--n_pcs', type=int, default=30,
                       help='Number of PCs to use (default: 30)')
    
    parser.add_argument('--specific_files', nargs='+',
                       help='Specific h5ad files to plot (full paths)')
    
    return parser.parse_args()


# ===================================================================
# MAIN EXECUTION
# ===================================================================

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info("="*80)
    logging.info("UMAP PLOTTING PIPELINE")
    logging.info("="*80)
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Cell type column: {args.celltype_col}")
    logging.info(f"PNG DPI: {args.png_dpi}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_saved_plots = []
    
    # Process specific files if provided
    if args.specific_files:
        logging.info("\n" + "="*80)
        logging.info("PROCESSING SPECIFIC FILES")
        logging.info("="*80)
        
        for file_path in args.specific_files:
            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}")
                continue
            
            panel_name = os.path.basename(file_path).replace('.h5ad', '')
            plots = plot_panel_umaps(
                adata_path=file_path,
                panel_name=panel_name,
                output_dir=args.output_dir,
                celltype_col=args.celltype_col,
                png_dpi=args.png_dpi
            )
            all_saved_plots.extend(plots)
    
    # Process panels by pattern
    if args.panels:
        logging.info("\n" + "="*80)
        logging.info("PROCESSING PANELS BY PATTERN")
        logging.info("="*80)
        
        panel_patterns = [p.strip() for p in args.panels.split(',')]
        logging.info(f"Panel patterns: {panel_patterns}")
        
        for pattern in panel_patterns:
            logging.info(f"\nSearching for files matching: {pattern}")
            matching_files = find_h5ad_files_matching_panel(args.preprocessed_dir, pattern)
            
            if not matching_files:
                logging.warning(f"  No files found matching pattern: {pattern}")
                continue
            
            logging.info(f"  Found {len(matching_files)} file(s)")
            
            # Plot each matching file
            for file_path in matching_files:
                panel_name = os.path.basename(file_path).replace('.h5ad', '')
                plots = plot_panel_umaps(
                    adata_path=file_path,
                    panel_name=panel_name,
                    output_dir=args.output_dir,
                    celltype_col=args.celltype_col,
                    png_dpi=args.png_dpi
                )
                all_saved_plots.extend(plots)
    
    # Process full transcriptome
    if args.full_transcriptome:
        logging.info("\n" + "="*80)
        logging.info("PROCESSING FULL TRANSCRIPTOME")
        logging.info("="*80)
        
        if os.path.exists(args.full_transcriptome):
            plots = plot_panel_umaps(
                adata_path=args.full_transcriptome,
                panel_name="Full_Transcriptome",
                output_dir=args.output_dir,
                celltype_col=args.celltype_col,
                png_dpi=args.png_dpi
            )
            all_saved_plots.extend(plots)
        else:
            logging.warning(f"Full transcriptome file not found: {args.full_transcriptome}")
    
    # Summary
    logging.info("\n" + "="*80)
    logging.info("UMAP PLOTTING COMPLETE")
    logging.info("="*80)
    logging.info(f"Total plots generated: {len(all_saved_plots)}")
    logging.info(f"All plots saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
