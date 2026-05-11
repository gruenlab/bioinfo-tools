#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Preprocess data for evaluation pipeline.

This script preprocesses datasets for evaluation by:
1. Loading gene lists from the selection pipeline output directory
2. Subsetting data to each gene list
3. Running process_data_for_panel_evaluation() to compute:
   - PCA and/or NMF embeddings
   - Leiden clustering at multiple resolutions (7-60 clusters)
   - K-neighborhood graphs (k=5,10,15,20,30,50)
   - UMAP visualizations
4. Saving preprocessed datasets for fast evaluation

The output is compatible with run_evaluation.py, which can load these
preprocessed datasets instead of recomputing embeddings and clusters.

Usage:
------
# Basic usage with gene lists directory
python preprocess_for_evaluation.py \\
    --input_file /path/to/raw_data.h5ad \\
    --gene_lists_dir /path/to/gene_lists/ \\
    --output_dir /path/to/preprocessed/ \\
    --celltype_column celltypes_v2 \\
    --dimensionality_reduction both \\
    --n_neighbors 15

# With external panels - flexible subsetting
python preprocess_for_evaluation.py \\
    --input_file /path/to/raw_data.h5ad \\
    --output_dir /path/to/preprocessed/ \\
    --external_panels panel1.csv panel2.csv \\
    --external_names Method1 Method2 \\
    --external_probeset_sizes 500 0

# Using a text file listing gene list paths (avoids shell "Argument list too long")
python preprocess_for_evaluation.py \\
    --input_file /path/to/raw_data.h5ad \\
    --gene_list_files_txt /path/to/gene_list_paths.txt \\
    --output_dir /path/to/preprocessed/

Output structure:
-----------------
output_dir/
├── dt_simple_100.h5ad
├── dt_nmf_200.h5ad
├── Method1_500.h5ad
└── logs/
    └── preprocess_YYYYMMDD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import gc
import glob
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc

# Configure pandas to use object dtype for strings instead of ArrowStringArray
# This ensures compatibility with anndata's HDF5 writer (pandas 2.x+ uses PyArrow strings by default)
pd.options.mode.string_storage = "python"

# =============================================================================
# MODULE IMPORTS
# =============================================================================

_SCRIPT_DIR = Path(__file__).parent.absolute()
_MODULES_DIR = _SCRIPT_DIR.parent           # SpatialProbeDesign/Modules/
_PROJECT_DIR = _MODULES_DIR.parent          # SpatialProbeDesign/

# Add Evaluation-module to sys.path for process_data_for_panel_evaluation
_EVAL_MODULE_DIR = _MODULES_DIR / "Evaluation-module"
if str(_EVAL_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_MODULE_DIR))

# Add Utility-module to sys.path for convert_ensembl_to_gene_symbols
_UTILITY_DIR = _MODULES_DIR / "Utility-module"
if str(_UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILITY_DIR))

from _preprocessing import process_data_for_panel_evaluation
from _utils import convert_ensembl_to_gene_symbols

# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ['main']


# =============================================================================
# Setup
# =============================================================================

def setup_logging(log_file: Optional[str] = None, log_level: int = logging.INFO) -> None:
    """Configure logging with both file and console outputs.

    Args:
        log_file: Optional path to a log file. If None, logs only to console.
        log_level: Logging level (default: logging.INFO).
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.root.setLevel(log_level)
    formatter = logging.Formatter(log_format, datefmt=date_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

    logging.info(f"Starting preprocessing at {datetime.now().strftime(date_format)}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Working directory: {os.getcwd()}")


# =============================================================================
# Gene list loading utilities
# =============================================================================

def load_gene_list_from_csv(filepath: str) -> List[str]:
    """Load a gene list from a CSV file.

    Handles multiple CSV formats:
    1. Single column with genes (may or may not have header)
    2. Multi-column with a 'gene' column (e.g., NS-Forest format)
    3. First column contains genes (standard format)
    4. ranked_gene_list.csv with 'selected_final' column (filters to True values)

    Args:
        filepath: Path to CSV file.

    Returns:
        List of gene names.
    """
    try:
        df = pd.read_csv(filepath)

        # Check if this is a ranked_gene_list.csv with selected_final column
        if 'selected_final' in df.columns and 'gene' in df.columns:
            logging.info(f"Found ranked_gene_list.csv format with selected_final column in {os.path.basename(filepath)}")
            # Filter to only genes where selected_final == True
            df_selected = df[df['selected_final'] == True]
            genes = df_selected['gene'].tolist()
            logging.info(f"Filtered to {len(genes)} genes with selected_final=True (from {len(df)} total genes)")
            genes = [g for g in genes if pd.notna(g)]
            return genes

        # Original logic for other formats (selected_genes.csv, simple lists, etc.)
        common_headers = [
            'gene', 'genes', 'gene_name', 'gene_id', 'symbol',
            'gene_symbol', 'feature', 'id', 'name', 'sp_genes'
        ]

        gene_column = None
        for col in df.columns:
            if col.lower() in common_headers:
                gene_column = col
                logging.info(f"Found gene column '{col}' in CSV: {filepath}")
                break

        if gene_column is not None:
            genes = df[gene_column].tolist()
        elif df.columns[0].startswith(('ENS', 'ENSG', 'ENSM')):
            df = pd.read_csv(filepath, header=None)
            genes = df.iloc[:, 0].tolist()
        else:
            genes = df.iloc[:, 0].tolist()

        genes = [g for g in genes if pd.notna(g)]
        logging.info(f"Loaded {len(genes)} genes from {os.path.basename(filepath)}")
        return genes

    except Exception as e:
        logging.error(f"Error loading gene list from {filepath}: {e}")
        return []


def extract_genelist_name_from_path(csv_file: str) -> str:
    """Extract a standardized gene list name from a file path.

    Handles both old and new directory structures:
    OLD: .../Filter/Baseline/Strategy/Size-genes/results/selected_genes.csv
    NEW: .../Filter/Baseline/Strategy/N_factors/Size-genes/results/selected_genes.csv

    Args:
        csv_file: Path to a gene list CSV file.

    Returns:
        Standardized name for the gene list.
    """
    path_parts = csv_file.split(os.sep)

    try:
        results_idx = path_parts.index("results")

        size_dir = path_parts[results_idx - 1]
        potential_factor_dir = path_parts[results_idx - 2]

        if potential_factor_dir.endswith("_factors"):
            factor_dir = potential_factor_dir
            strategy = path_parts[results_idx - 3]
            subset = path_parts[results_idx - 4]
            filter_method = path_parts[results_idx - 5]
            n_factors = factor_dir.replace("_factors", "")
        else:
            n_factors = None
            strategy = path_parts[results_idx - 2]
            subset = path_parts[results_idx - 3]
            filter_method = path_parts[results_idx - 4]

        size = size_dir.replace("-genes", "")
        filename = os.path.basename(csv_file)
        filling_suffix = ""

        if filename.startswith("selected_hvg_"):
            size_match = re.search(r'selected_hvg_(\d+)genes\.csv', filename)
            if size_match:
                size = size_match.group(1)
        elif filename.startswith("selected_random_"):
            size_match = re.search(r'selected_random_(\d+)genes\.csv', filename)
            if size_match:
                size = size_match.group(1)
        elif filename != "selected_genes.csv":
            filling_suffix = filename.replace("selected_genes_", "").replace(".csv", "")

        if n_factors is not None:
            if filling_suffix:
                return f"{filter_method}_{subset}_{strategy}_{n_factors}factors_{size}_{filling_suffix}"
            else:
                return f"{filter_method}_{subset}_{strategy}_{n_factors}factors_{size}"
        else:
            if filling_suffix:
                return f"{filter_method}_{subset}_{strategy}_{size}_{filling_suffix}"
            else:
                return f"{filter_method}_{subset}_{strategy}_{size}"

    except Exception:
        return os.path.splitext(os.path.basename(csv_file))[0]


def load_all_gene_lists(gene_lists_dir: str, adata) -> Dict[str, List[str]]:
    """Load all gene lists from the Selection pipeline output directory.

    Supports both old and new directory structures:
    OLD: .../Filter/Baseline/Strategy/Size-genes/results/selected_genes.csv
    NEW: .../Filter/Baseline/Strategy/N_factors/Size-genes/results/selected_genes.csv

    Files named 'intermediate_panel_before-gap-fill.csv' and bootstrap files
    are automatically excluded.

    Args:
        gene_lists_dir: Base directory containing Selected-panels output.
        adata: Reference dataset (to check gene availability).

    Returns:
        Dictionary mapping gene list names to gene lists.
    """
    gene_lists = {}
    available_genes = set(adata.var_names)

    if not os.path.exists(gene_lists_dir):
        logging.error(f"Gene lists directory not found: {gene_lists_dir}")
        return gene_lists

    logging.info(f"Scanning directory structure: {gene_lists_dir}")

    csv_pattern_standard_old = os.path.join(
        gene_lists_dir, "*", "*", "*", "*-genes", "results", "selected_genes*.csv")
    csv_pattern_hvg_old = os.path.join(
        gene_lists_dir, "*", "*", "*", "*-genes", "results", "selected_hvg_*.csv")
    csv_pattern_random_old = os.path.join(
        gene_lists_dir, "*", "*", "*", "*-genes", "results", "selected_random_*.csv")

    csv_pattern_standard_new = os.path.join(
        gene_lists_dir, "*", "*", "*", "*_factors", "*-genes", "results", "selected_genes*.csv")
    csv_pattern_hvg_new = os.path.join(
        gene_lists_dir, "*", "*", "*", "*_factors", "*-genes", "results", "selected_hvg_*.csv")
    csv_pattern_random_new = os.path.join(
        gene_lists_dir, "*", "*", "*", "*_factors", "*-genes", "results", "selected_random_*.csv")

    csv_files = (
        glob.glob(csv_pattern_standard_old) + glob.glob(csv_pattern_hvg_old)
        + glob.glob(csv_pattern_random_old) + glob.glob(csv_pattern_standard_new)
        + glob.glob(csv_pattern_hvg_new) + glob.glob(csv_pattern_random_new)
    )

    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("intermediate_")]
    logging.info(f"Found {len(csv_files)} gene list files (excluding intermediate files)")

    for csv_file in sorted(csv_files):
        path_parts = csv_file.split(os.sep)

        try:
            results_idx = path_parts.index("results")
            size_dir = path_parts[results_idx - 1]
            potential_factor_dir = path_parts[results_idx - 2]

            if potential_factor_dir.endswith("_factors"):
                n_factors = potential_factor_dir.replace("_factors", "")
                strategy = path_parts[results_idx - 3]
                subset = path_parts[results_idx - 4]
                filter_method = path_parts[results_idx - 5]
            else:
                n_factors = None
                strategy = path_parts[results_idx - 2]
                subset = path_parts[results_idx - 3]
                filter_method = path_parts[results_idx - 4]

            size = size_dir.replace("-genes", "")
            filename = os.path.basename(csv_file)
            filling_suffix = ""

            if filename.startswith("selected_hvg_"):
                size_match = re.search(r'selected_hvg_(\d+)genes\.csv', filename)
                if size_match:
                    size = size_match.group(1)
            elif filename.startswith("selected_random_"):
                size_match = re.search(r'selected_random_(\d+)genes\.csv', filename)
                if size_match:
                    size = size_match.group(1)
                if "_bootstrap" in filename:
                    logging.info(f"Skipping bootstrap file: {filename}")
                    continue
            elif filename != "selected_genes.csv":
                filling_suffix = filename.replace("selected_genes_", "").replace(".csv", "")

            if n_factors is not None:
                full_name = (
                    f"{filter_method}_{subset}_{strategy}_{n_factors}factors_{size}_{filling_suffix}"
                    if filling_suffix
                    else f"{filter_method}_{subset}_{strategy}_{n_factors}factors_{size}"
                )
            else:
                full_name = (
                    f"{filter_method}_{subset}_{strategy}_{size}_{filling_suffix}"
                    if filling_suffix
                    else f"{filter_method}_{subset}_{strategy}_{size}"
                )

            genes = load_gene_list_from_csv(csv_file)
            if len(genes) == 0:
                logging.warning(f"Empty gene list: {full_name}")
                continue

            genes_filtered = [g for g in genes if g in available_genes]
            missing = len(genes) - len(genes_filtered)
            if missing > 0:
                logging.warning(f"{full_name}: {missing}/{len(genes)} genes not found in dataset")

            if len(genes_filtered) > 0:
                gene_lists[full_name] = genes_filtered
                logging.info(f"Loaded {full_name}: {len(genes_filtered)} genes")
            else:
                logging.error(f"No valid genes found for {full_name}")

        except (ValueError, IndexError) as e:
            logging.error(f"Could not parse path {csv_file}: {e}")
            continue

    if len(gene_lists) == 0:
        logging.error("No gene lists loaded! Check directory structure.")
        logging.error(
            f"Expected: {gene_lists_dir}/Filter-Method/Gene-Subset/strategy/size-genes/"
            "results/selected_*.csv"
        )
    else:
        logging.info(f"Successfully loaded {len(gene_lists)} gene lists")

    return gene_lists


# =============================================================================
# External panel loading
# =============================================================================

def load_external_panels(
    external_paths: List[str],
    external_names: List[str],
    probeset_sizes: Optional[List[int]],
    adata
) -> Dict[str, List[str]]:
    """Load external gene panels and optionally subset to specified sizes.

    Args:
        external_paths: Paths to external gene panel CSV files.
        external_names: Names for the external panels.
        probeset_sizes: Target sizes to subset panels to (use 0 or -1 to keep all genes).
        adata: Reference dataset (to check gene availability).

    Returns:
        Dictionary mapping panel names to gene lists.
    """
    external_gene_lists = {}
    available_genes = set(adata.var_names)

    if not external_paths:
        return external_gene_lists

    for i, (path, name) in enumerate(zip(external_paths, external_names)):
        logging.info(f"Loading external panel: {name} from {path}")

        if not os.path.exists(path):
            logging.error(f"External panel file not found: {path}")
            continue

        genes = load_gene_list_from_csv(path)
        if len(genes) == 0:
            logging.error(f"No genes loaded from {path}")
            continue

        genes_filtered = [g for g in genes if g in available_genes]
        missing = len(genes) - len(genes_filtered)
        if missing > 0:
            logging.warning(f"  {missing}/{len(genes)} genes not found in dataset")

        if len(genes_filtered) == 0:
            logging.error(f"  No valid genes found in dataset for {name}")
            continue

        if probeset_sizes and i < len(probeset_sizes):
            target_size = probeset_sizes[i]
            if target_size > 0 and target_size < len(genes_filtered):
                logging.info(f"  Subsetting from {len(genes_filtered)} to {target_size} genes")
                genes_filtered = genes_filtered[:target_size]
                panel_name = f"{name}_{target_size}"
            else:
                panel_name = name
        else:
            panel_name = name

        external_gene_lists[panel_name] = genes_filtered
        logging.info(f"  Added: {panel_name} ({len(genes_filtered)} genes)")

    return external_gene_lists


def load_10x_panel_genes(panel_path: str) -> List[str]:
    """Load genes from a 10x panel metadata CSV file.

    Args:
        panel_path: Path to 10x panel CSV file.

    Returns:
        List of gene names from the panel.
    """
    try:
        df = pd.read_csv(panel_path)
        gene_columns = ['gene', 'genes', 'gene_name', 'gene_id', 'symbol', 'gene_symbol']
        gene_col = next((c for c in gene_columns if c in df.columns), None)
        genes = df[gene_col].dropna().tolist() if gene_col else df.iloc[:, 0].dropna().tolist()
        logging.info(f"Loaded {len(genes)} genes from 10x panel: {panel_path}")
        return genes
    except Exception as e:
        logging.error(f"Error loading 10x panel from {panel_path}: {e}")
        return []


def combine_panels_with_10x(
    gene_lists: Dict[str, List[str]],
    panel_10x_genes_dict: Dict[str, List[str]],
    add_10x_mode: str,
    data_dir: str
) -> Dict[str, List[str]]:
    """Combine gene panels with 10x panels based on the specified mode.

    Args:
        gene_lists: Dictionary of {panel_name: [genes]}.
        panel_10x_genes_dict: Dict with '5k' and 'mMulti_v1' gene lists.
        add_10x_mode: One of 'both', 'mMulti', '5k', 'no-10x-panel', 'all'.
        data_dir: Base data directory (unused here, kept for API compatibility).

    Returns:
        Updated gene_lists dictionary with combined panels.
    """
    if add_10x_mode == 'no-10x-panel':
        return gene_lists

    if not panel_10x_genes_dict:
        logging.warning("No 10x panel genes available for combination")
        return gene_lists

    logging.info(f"Combining gene lists with 10x panels (mode: {add_10x_mode})")

    if add_10x_mode in ['both', 'all']:
        panels_to_add = ['mMulti_v1', '5k']
    elif add_10x_mode == 'mMulti':
        panels_to_add = ['mMulti_v1']
    elif add_10x_mode == '5k':
        panels_to_add = ['5k']
    else:
        panels_to_add = []

    combined_lists: Dict[str, List[str]] = {}
    if add_10x_mode == 'all':
        combined_lists.update(gene_lists)

    panel_name_to_suffix = {'mMulti_v1': 'mMulti-addon', '5k': '5k-addon'}

    for panel_name, genes in gene_lists.items():
        if '10x' in panel_name.lower() or panel_name in ['mMulti_v1', '5k']:
            if add_10x_mode != 'all':
                combined_lists[panel_name] = genes
            continue

        for p10x in panels_to_add:
            if p10x not in panel_10x_genes_dict:
                continue
            panel_10x_genes = panel_10x_genes_dict[p10x]
            combined_genes = list(dict.fromkeys(genes + panel_10x_genes))
            suffix = panel_name_to_suffix[p10x]
            new_name = f"{panel_name}_{suffix}"
            combined_lists[new_name] = combined_genes
            logging.info(
                f"  {new_name}: {len(genes)} + {len(panel_10x_genes)} → "
                f"{len(combined_genes)} unique genes"
            )

    logging.info(f"Total gene lists after 10x combination: {len(combined_lists)}")
    return combined_lists


# =============================================================================
# Per-panel preprocessing
# =============================================================================

def preprocess_single_genelist(
    adata,
    genelist_name: str,
    genes: List[str],
    output_dir: str,
    n_neighbors: int = 15,
    dimensionality_reduction: str = 'pca'
) -> Optional[str]:
    """Preprocess data for a single gene list.

    Args:
        adata: Raw input AnnData.
        genelist_name: Name of the gene list (used as output filename stem).
        genes: List of genes to subset to.
        output_dir: Directory to write the output h5ad file.
        n_neighbors: Number of neighbors for KNN graph construction.
        dimensionality_reduction: 'pca', 'nmf', or 'both'.

    Returns:
        Path to saved h5ad file, or None on failure.
    """
    logging.info("=" * 80)
    logging.info(f"Preprocessing: {genelist_name}")
    logging.info(f"Number of genes: {len(genes)}")
    logging.info("=" * 80)

    try:
        adata_subset = adata.copy()

        adata_processed = process_data_for_panel_evaluation(
            adata=adata_subset,
            probeset=genes,
            n_neighbors=n_neighbors,
            layer="counts",
            hvg=False,
            subset=False,
            scale=False,
            dataset_name=genelist_name,
            dimensionality_reduction=dimensionality_reduction,
            filter_genes=False,   # Preserve exact panel composition
        )

        output_file = os.path.join(output_dir, f"{genelist_name}.h5ad")
        logging.info("Saving preprocessed data with gzip compression...")
        adata_processed.write_h5ad(output_file, compression='gzip', compression_opts=9)

        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logging.info(f"Saved: {output_file} ({file_size_mb:.2f} MB)")

        if 'X_pca' in adata_processed.obsm:
            logging.info(f"  PCA: {adata_processed.obsm['X_pca'].shape[1]} components")
        if 'X_nmf' in adata_processed.obsm:
            logging.info(f"  NMF: {adata_processed.obsm['X_nmf'].shape[1]} components")

        leiden_cols = [
            c for c in adata_processed.obs.columns
            if c.startswith('leiden_') and '_clusters' in c
        ]
        logging.info(f"  Leiden clusterings: {len(leiden_cols)}")

        neighbor_keys = [k for k in adata_processed.uns if k.startswith('neighbors_')]
        logging.info(f"  Neighbor graphs: {len(neighbor_keys)}")

        del adata_subset
        del adata_processed
        gc.collect()

        return output_file

    except Exception as e:
        logging.error(f"Error preprocessing {genelist_name}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


# =============================================================================
# Argument parser
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess data for evaluation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required
    parser.add_argument(
        '--input_file', type=str, required=True,
        help='Path to raw h5ad file'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for preprocessed datasets'
    )

    # Gene list sources (at least one required)
    gene_source = parser.add_mutually_exclusive_group()
    gene_source.add_argument(
        '--gene_lists_dir', type=str,
        help='Directory containing gene list subdirectories (Selection pipeline output)'
    )
    gene_source.add_argument(
        '--gene_list_files_txt', type=str,
        help=(
            'Text file listing gene list file paths (one per line). '
            'Use instead of --gene_lists_dir to avoid "Argument list too long" errors.'
        )
    )

    # External panels
    parser.add_argument(
        '--external_panels', type=str, nargs='+',
        help='Paths to external gene panel CSV files'
    )
    parser.add_argument(
        '--external_names', type=str, nargs='+',
        help='Names for external panels (one per panel)'
    )
    parser.add_argument(
        '--external_probeset_sizes', type=int, nargs='+',
        help=(
            'Probeset sizes to subset external panels to (one per panel). '
            'Use 0 or -1 to keep all genes for a specific panel.'
        )
    )

    # 10x panel combination
    parser.add_argument(
        '--add_10x_panels', type=str, default='no-10x-panel',
        choices=['both', 'mMulti', '5k', 'no-10x-panel', 'all'],
        help=(
            'How to combine gene lists with 10x panels: '
            '"both" adds mMulti_v1 and 5k to each panel; '
            '"mMulti" adds only mMulti_v1; '
            '"5k" adds only 5k panel; '
            '"no-10x-panel" skips 10x panels (default); '
            '"all" keeps originals + creates mMulti and 5k versions.'
        )
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='/home/gruengroup/helene/SpatialProbeDesign_tmp',
        help='Base data directory containing 10x panel files'
    )

    # Processing parameters
    parser.add_argument(
        '--celltype_column', type=str, default='celltypes_v2',
        help='Column name for cell type annotation (default: celltypes_v2)'
    )
    parser.add_argument(
        '--n_neighbors', type=int, default=15,
        help='Number of neighbors for KNN graph (default: 15)'
    )
    parser.add_argument(
        '--dimensionality_reduction', type=str, default='both',
        choices=['pca', 'nmf', 'both'],
        help='Dimensionality reduction method (default: both)'
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    """Main preprocessing pipeline.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"preprocess_{timestamp}.log")
    setup_logging(log_file=log_file, log_level=getattr(logging, args.log_level))

    logging.info("=" * 80)
    logging.info("PREPROCESSING PARAMETERS")
    logging.info("=" * 80)
    logging.info(f"Input file:              {args.input_file}")
    logging.info(f"Output directory:        {args.output_dir}")
    if args.gene_lists_dir:
        logging.info(f"Gene lists directory:    {args.gene_lists_dir}")
    if args.gene_list_files_txt:
        logging.info(f"Gene list files txt:     {args.gene_list_files_txt}")
    logging.info(f"Cell type column:        {args.celltype_column}")
    logging.info(f"Number of neighbors:     {args.n_neighbors}")
    logging.info(f"Dimensionality reduction:{args.dimensionality_reduction}")
    logging.info("=" * 80)

    # Load raw data
    logging.info(f"\nLoading raw data from {args.input_file}")
    try:
        adata = sc.read_h5ad(args.input_file)
        logging.info(f"Loaded dataset: {adata.n_obs} cells × {adata.n_vars} genes")

        if adata.var_names[0].startswith(('ENSMUSG', 'ENSG')):
            logging.info("Detected ENSEMBL IDs - converting to gene symbols...")
            convert_ensembl_to_gene_symbols(adata, inplace=True)
        else:
            logging.info(f"Gene names in symbol format (first: {adata.var_names[0]})")

        if args.celltype_column not in adata.obs.columns:
            logging.error(
                f"Cell type column '{args.celltype_column}' not found in data. "
                f"Available: {', '.join(adata.obs.columns)}"
            )
            return 1

        logging.info(f"Cell types: {adata.obs[args.celltype_column].nunique()} unique types")

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return 1

    # Load gene lists
    gene_lists: Dict[str, List[str]] = {}

    if args.gene_list_files_txt:
        logging.info(f"Loading gene lists from file list: {args.gene_list_files_txt}")
        try:
            with open(args.gene_list_files_txt, 'r') as f:
                gene_list_paths = [line.strip() for line in f if line.strip()]

            logging.info(f"Found {len(gene_list_paths)} gene list file paths")
            available_genes = set(adata.var_names)

            for gene_list_path in gene_list_paths:
                if not os.path.exists(gene_list_path):
                    logging.warning(f"Gene list file not found: {gene_list_path}")
                    continue

                basename = os.path.basename(gene_list_path)
                if not basename.startswith("selected_genes"):
                    continue

                genelist_name = extract_genelist_name_from_path(gene_list_path)

                try:
                    genes = load_gene_list_from_csv(gene_list_path)
                    if len(genes) == 0:
                        logging.warning(f"{genelist_name}: Empty gene list")
                        continue

                    genes_available = [g for g in genes if g in available_genes]
                    if len(genes_available) > 0:
                        gene_lists[genelist_name] = genes_available
                    else:
                        logging.warning(f"{genelist_name}: No genes found in dataset")

                except Exception as e:
                    logging.error(f"Error loading {gene_list_path}: {e}")

            logging.info(f"Loaded {len(gene_lists)} gene lists from file list")

        except Exception as e:
            logging.error(f"Error reading gene list file: {e}")
            return 1

    elif args.gene_lists_dir:
        logging.info("Loading gene lists from directory...")
        gene_lists = load_all_gene_lists(args.gene_lists_dir, adata)

    # Load external panels
    if args.external_panels:
        logging.info("Loading external panels...")

        if not args.external_names:
            logging.error("--external_names is required when using --external_panels")
            return 1

        if len(args.external_panels) != len(args.external_names):
            logging.error(
                f"Panel count ({len(args.external_panels)}) must match "
                f"name count ({len(args.external_names)})"
            )
            return 1

        if (args.external_probeset_sizes
                and len(args.external_probeset_sizes) != len(args.external_panels)):
            logging.error(
                f"Probeset size count ({len(args.external_probeset_sizes)}) must match "
                f"panel count ({len(args.external_panels)})"
            )
            return 1

        external_gene_lists = load_external_panels(
            args.external_panels,
            args.external_names,
            args.external_probeset_sizes,
            adata
        )
        gene_lists.update(external_gene_lists)
        logging.info(f"Added {len(external_gene_lists)} external panels")

    # Combine with 10x panels if requested
    if args.add_10x_panels != 'no-10x-panel':
        logging.info("=" * 80)
        logging.info("LOADING 10X PANELS")
        logging.info("=" * 80)

        panel_10x_paths = {
            'mMulti_v1': os.path.join(
                args.data_dir, 'Data/10x-panels/Xenium_mMulti_v1_metadata_annotations.csv'),
            '5k': os.path.join(
                args.data_dir, 'Data/10x-panels/XeniumPrimeMouse5Kpan_tissue_pathways_metadata.csv')
        }

        panel_10x_genes_dict: Dict[str, List[str]] = {}
        available_genes = set(adata.var_names)

        for panel_name, panel_path in panel_10x_paths.items():
            if not os.path.exists(panel_path):
                logging.error(f"10x panel file not found: {panel_path}")
                return 1
            genes = load_10x_panel_genes(panel_path)
            genes_filtered = [g for g in genes if g in available_genes]
            panel_10x_genes_dict[panel_name] = genes_filtered
            logging.info(f"  {panel_name}: {len(genes_filtered)} available genes")

        if len(panel_10x_genes_dict) == 0:
            logging.error("No 10x panels could be loaded")
            return 1

        # Add standalone 10x panels
        for panel_name, genes in panel_10x_genes_dict.items():
            already_exists = any(
                panel_name in n or n in panel_name or f"{panel_name}_10x" in n
                for n in gene_lists
            )
            if not already_exists:
                gene_lists[panel_name] = genes
                logging.info(f"  Added standalone panel: {panel_name}")

        gene_lists = combine_panels_with_10x(
            gene_lists, panel_10x_genes_dict, args.add_10x_panels, args.data_dir
        )

    if len(gene_lists) == 0:
        logging.error(
            "No gene lists loaded. Provide --gene_lists_dir, --gene_list_files_txt, "
            "or --external_panels."
        )
        return 1

    logging.info(f"\nTotal gene lists to preprocess: {len(gene_lists)}")

    # Preprocess each gene list
    logging.info("=" * 80)
    logging.info(f"PREPROCESSING {len(gene_lists)} GENE LISTS")
    logging.info("=" * 80)

    successful = 0
    failed = 0

    for i, (name, genes) in enumerate(gene_lists.items(), 1):
        logging.info(f"\nProcessing {i}/{len(gene_lists)}: {name}")

        result = preprocess_single_genelist(
            adata=adata,
            genelist_name=name,
            genes=genes,
            output_dir=args.output_dir,
            n_neighbors=args.n_neighbors,
            dimensionality_reduction=args.dimensionality_reduction
        )

        if result is not None:
            successful += 1
        else:
            failed += 1

        gc.collect()

    logging.info("=" * 80)
    logging.info("PREPROCESSING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Successful: {successful}/{len(gene_lists)}")
    logging.info(f"Failed:     {failed}/{len(gene_lists)}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
