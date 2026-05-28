"""Preprocessing functions for evaluation pipeline datasets.

This module provides utilities to:
- Load gene lists from selection pipeline output directories.
- Preprocess probe-panel subsets: normalize, compute PCA/NMF,
  build Leiden clusters and KNN graphs.
- Preprocess the full-transcriptome reference dataset using the same
  pipeline for fair comparison.

Functions:
    load_gene_list_from_csv: Load gene names from a CSV file.
    extract_genelist_name_from_path: Parse a file path into a dataset name.
    load_all_gene_lists: Load all gene lists from a selection output dir.
    process_data_for_panel_evaluation: Full preprocessing for one panel.
    preprocess_reference_dataset: Preprocess the reference transcriptome.
"""

from __future__ import annotations

import logging
import os
import sys
import glob
import gc
import re
from pathlib import Path
from scipy.sparse import issparse
from sklearn.decomposition import NMF

import scanpy as sc
import pandas as pd
import numpy as np

# Configure pandas to use object dtype for strings instead of ArrowStringArray
# This ensures compatibility with anndata's HDF5 writer (pandas 2.x+ uses PyArrow strings by default)
pd.options.mode.string_storage = "python"

logger = logging.getLogger(__name__)

__all__ = [
    "load_gene_list_from_csv",
    "extract_genelist_name_from_path",
    "load_all_gene_lists",
    "process_data_for_panel_evaluation",
    "preprocess_reference_dataset",
]

# ===================================================================
# UTILITY IMPORTS WITH GRACEFUL FALLBACK
# ===================================================================

try:
    from _validation import is_anndata_raw, is_anndata_raw_layer
except ImportError:
    try:
        _util_dir = Path(__file__).parent.parent / "Utility-module"
        if _util_dir.exists() and str(_util_dir) not in sys.path:
            sys.path.insert(0, str(_util_dir))
        from _validation import is_anndata_raw, is_anndata_raw_layer
    except ImportError:
        logger.warning("Could not import validation utilities; using stubs. Pipeline will inject these at runtime.")

        def is_anndata_raw(adata: object) -> bool:
            """Stub: assume raw — conservative default when validation module is unavailable."""
            return True

        def is_anndata_raw_layer(adata: object, layer: str) -> bool:
            """Stub: assume raw."""
            return True

try:
    from _utils import convert_ensembl_to_gene_symbols
except ImportError:
    try:
        _util_dir = Path(__file__).parent.parent / "Utility-module"
        if _util_dir.exists() and str(_util_dir) not in sys.path:
            sys.path.insert(0, str(_util_dir))
        from _utils import convert_ensembl_to_gene_symbols
    except ImportError:
        def convert_ensembl_to_gene_symbols(adata: object, inplace: bool = True) -> None:
            """Stub: no-op."""
            pass


# ===================================================================
# COMPATIBILITY UTILITIES
# ===================================================================

def _convert_arrow_strings_to_object(adata):
    """Convert all pandas string extension arrays to object dtype.

    This ensures compatibility with anndata's HDF5 writer, which doesn't support
    pandas string extension arrays (StringArray, ArrowStringArray) in versions < 0.11.

    Handles:
    - pd.arrays.StringArray (nullable strings with python backend)
    - pd.arrays.ArrowStringArray (nullable strings with pyarrow backend)
    - Any dtype with string representation containing 'string'

    Args:
        adata: AnnData object potentially containing string extension arrays.

    Returns:
        Modified AnnData object with object dtype strings.
    """
    import anndata
    from pandas.api.types import is_string_dtype

    def _is_string_extension_array(obj):
        """Check if object is a pandas string extension array."""
        # Check if it's a StringArray or ArrowStringArray instance
        try:
            if isinstance(obj, (pd.arrays.StringArray, pd.arrays.ArrowStringArray)):
                return True
        except (AttributeError, TypeError):
            pass

        # Check if dtype is string-like
        if hasattr(obj, 'dtype'):
            dtype_str = str(obj.dtype)
            # Check for 'string' dtype (covers both StringArray and ArrowStringArray)
            if dtype_str == 'string' or dtype_str.startswith('string['):
                return True
            # Also check using pandas API
            try:
                if is_string_dtype(obj.dtype) and obj.dtype != object:
                    return True
            except (AttributeError, TypeError):
                pass
        return False

    def _convert_to_object(obj):
        """Safely convert to object dtype."""
        try:
            return obj.astype('object')
        except Exception as e:
            logger.warning(f"Could not convert {type(obj)} to object dtype: {e}")
            return obj

    # Convert var index
    if _is_string_extension_array(adata.var.index):
        adata.var.index = _convert_to_object(adata.var.index)

    # Convert var columns
    for col in adata.var.columns:
        if _is_string_extension_array(adata.var[col]):
            adata.var[col] = _convert_to_object(adata.var[col])

    # Convert obs index
    if _is_string_extension_array(adata.obs.index):
        adata.obs.index = _convert_to_object(adata.obs.index)

    # Convert obs columns
    for col in adata.obs.columns:
        if _is_string_extension_array(adata.obs[col]):
            adata.obs[col] = _convert_to_object(adata.obs[col])

    # Convert raw.var if present
    if adata.raw is not None:
        needs_reconstruction = False
        new_var = adata.raw.var.copy()

        # Check and convert raw.var index
        if _is_string_extension_array(adata.raw.var.index):
            new_var.index = _convert_to_object(new_var.index)
            needs_reconstruction = True

        # Check and convert raw.var columns
        for col in new_var.columns:
            if _is_string_extension_array(new_var[col]):
                new_var[col] = _convert_to_object(new_var[col])
                needs_reconstruction = True

        # Reconstruct raw if any conversions were made
        if needs_reconstruction:
            temp_adata = anndata.AnnData(X=adata.raw.X, var=new_var, varm=adata.raw.varm)
            adata.raw = temp_adata

    return adata


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def _fix_index_column(adata: object) -> object:
    """
    Fix reserved column name '_index' in AnnData var.

    Handles repeated pattern of checking and fixing '_index' column
    that appears in both .var and .raw.var. If '_index' is identical
    to the current index, it is removed. If it differs and the current
    index is numeric, gene names are restored from '_index'. Otherwise,
    it is renamed to 'original_index'.

    Args:
        adata: AnnData object with potentially problematic '_index' column.

    Returns:
        Modified AnnData object.
    """
    import anndata

    # Fix .var if present
    if '_index' in adata.var.columns:
        logger.warning("Found '_index' column in var - checking if it's safe to remove...")
        index_col = adata.var['_index']
        current_index = adata.var.index

        if index_col.equals(current_index):
            logger.info("  '_index' column is identical to current index - safe to remove")
            adata.var = adata.var.drop(columns=['_index'])
        else:
            logger.warning("  '_index' column differs from current index")
            is_numeric_index = all(str(idx).isdigit() for idx in current_index[:10])

            if is_numeric_index:
                logger.warning("  Current index appears to be numeric placeholders - restoring gene names from '_index'")
                new_var = adata.var.drop(columns=['_index'])
                new_var.index = index_col
                new_var.index.name = None
                adata.var = new_var
                logger.info(f"  Restored gene names to var index (first 5: {new_var.index[:5].tolist()})")
            else:
                logger.warning("  Preserving '_index' as 'original_index' column")
                adata.var = adata.var.rename(columns={'_index': 'original_index'})

    # Fix .raw.var if raw is present
    if adata.raw is not None and '_index' in adata.raw.var.columns:
        logger.warning("Found '_index' column in raw.var - checking if it's safe to remove...")
        index_col = adata.raw.var['_index']
        current_index = adata.raw.var.index

        if index_col.equals(current_index):
            logger.info("  '_index' column in raw.var is identical to current index - safe to remove")
            new_var = adata.raw.var.drop(columns=['_index'])
            temp_adata = anndata.AnnData(X=adata.raw.X, var=new_var, varm=adata.raw.varm)
            adata.raw = temp_adata
        else:
            logger.warning("  '_index' column in raw.var differs from current index")
            is_numeric_index = all(str(idx).isdigit() for idx in current_index[:10])

            if is_numeric_index:
                logger.warning("  Current index appears to be numeric placeholders - restoring gene names from '_index'")
                new_var = adata.raw.var.drop(columns=['_index'])
                new_var.index = index_col
                new_var.index.name = None
                temp_adata = anndata.AnnData(X=adata.raw.X, var=new_var, varm=adata.raw.varm)
                adata.raw = temp_adata
                logger.info(f"  Restored gene names to raw.var index (first 5: {new_var.index[:5].tolist()})")
            else:
                logger.warning("  Preserving '_index' as 'original_index' column in raw.var")
                new_var = adata.raw.var.rename(columns={'_index': 'original_index'})
                temp_adata = anndata.AnnData(X=adata.raw.X, var=new_var, varm=adata.raw.varm)
                adata.raw = temp_adata

    return adata


# ===================================================================
# GENE LIST LOADING
# ===================================================================

def load_gene_list_from_csv(filepath: str) -> list[str]:
    """
    Load a gene list from a CSV file.

    Handles multiple CSV formats:
    1. Single column with genes (may or may not have header)
    2. Multi-column with a 'gene' column (e.g., NS-Forest format)
    3. First column contains genes (standard format)
    4. ranked_gene_list.csv with 'final_selection' column (filters to True values)

    Args:
        filepath: Path to CSV file.

    Returns:
        List of gene names.

    Example:
        >>> genes = load_gene_list_from_csv('/path/to/genes.csv')
        >>> len(genes)
        100
    """
    try:
        df = pd.read_csv(filepath)

        # Check if this is a ranked_gene_list.csv with final_selection column
        if 'final_selection' in df.columns and 'gene' in df.columns:
            logger.info(f"Found ranked_gene_list.csv format with final_selection column in {os.path.basename(filepath)}")
            # Filter to only genes where final_selection == True
            df_selected = df[df['final_selection'] == True]
            genes = df_selected['gene'].tolist()
            logger.info(f"Filtered to {len(genes)} genes with final_selection=True (from {len(df)} total genes)")
            genes = [g for g in genes if pd.notna(g)]
            return genes

        # Original logic for other formats (selected_genes.csv, simple lists, etc.)
        common_headers = ['gene', 'genes', 'gene_name', 'gene_id', 'symbol',
                         'gene_symbol', 'feature', 'id', 'name', 'sp_genes']

        gene_column = None
        for col in df.columns:
            if col.lower() in common_headers:
                gene_column = col
                logger.info(f"Found gene column '{col}' in CSV: {filepath}")
                break

        if gene_column is not None:
            genes = df[gene_column].tolist()
        elif df.columns[0].startswith(('ENS', 'ENSG', 'ENSM')):
            df = pd.read_csv(filepath, header=None)
            genes = df.iloc[:, 0].tolist()
        else:
            if df.columns[0].lower() in common_headers:
                genes = df.iloc[:, 0].tolist()
            else:
                genes = df.iloc[:, 0].tolist()

        genes = [g for g in genes if pd.notna(g)]
        logger.info(f"Loaded {len(genes)} genes from {os.path.basename(filepath)}")
        return genes

    except Exception as e:
        logger.error(f"Error loading gene list from {filepath}: {e}")
        return []


def extract_genelist_name_from_path(csv_file: str) -> str:
    """
    Extract a standardized gene list name from a file path.

    Handles both old and new directory structures:
    - OLD: .../Filter/Baseline/Strategy/Size-genes/results/selected_genes.csv
    - NEW: .../Filter/Baseline/Strategy/N_factors/Size-genes/results/selected_genes.csv

    Args:
        csv_file: Path to a gene list CSV file.

    Returns:
        Standardized name for the gene list.

    Example:
        >>> name = extract_genelist_name_from_path('/path/Scanpy-Filter/All-Genes/dt_deg/100-genes/results/selected_genes.csv')
        >>> name
        'Scanpy-Filter_All-Genes_dt_deg_100'
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

    except Exception as e:
        return os.path.splitext(os.path.basename(csv_file))[0]


def load_all_gene_lists(gene_lists_dir: str, adata: object) -> dict[str, list[str]]:
    """
    Load all gene lists from the Selection pipeline output directory.

    Expected directory structure (created by Selection pipeline):
        gene_lists_dir/
        ├── Filter-Method/
        │   ├── Gene-Subset/
        │   │   ├── strategy1/
        │   │   │   ├── 100-genes/
        │   │   │   │   └── results/
        │   │   │   │       ├── selected_genes.csv
        │   │   │   │       ├── selected_genes_DEG-based-filling.csv
        │   │   │   │       └── ...

    Fill-up strategies (when dt_dimred doesn't reach target panel size):
    - DEG-based-filling: Fill up with top DEGs
    - cell-type-specific-filling: Fill up with per-celltype dimred genes
    - global-gene-filling: Fill up with global dimred genes

    Note: Files named 'intermediate_panel_before-gap-fill.csv' and bootstrap files
    are excluded from evaluation as they represent incomplete gene panels.

    Args:
        gene_lists_dir: Base directory containing Selected-panels output.
        adata: Reference dataset (to check gene availability).

    Returns:
        Dictionary mapping gene list names to gene lists.
        Format: {filter}_{subset}_{strategy}_{size}[_{filling_suffix}]

    Example:
        >>> lists = load_all_gene_lists('/path/to/Selected-panels', adata)
        >>> 'Scanpy-Filter_All-Genes_dt_deg_100' in lists
        True
    """
    gene_lists = {}
    available_genes = set(adata.var_names)

    if not os.path.exists(gene_lists_dir):
        logger.error(f"Gene lists directory not found: {gene_lists_dir}")
        return gene_lists

    logger.info(f"Scanning directory structure: {gene_lists_dir}")

    # Priority 1: ranked_gene_list.csv (authoritative source with final_selection column)
    csv_pattern_ranked_old = os.path.join(gene_lists_dir, "*", "*-genes", "ranked_gene_list.csv")
    csv_pattern_ranked_new = os.path.join(gene_lists_dir, "*", "*_factors", "*-genes", "ranked_gene_list.csv")

    # Priority 2: selected_genes*.csv (legacy format, fallback when ranked_gene_list.csv doesn't exist)
    csv_pattern_standard_old = os.path.join(gene_lists_dir, "*", "*-genes", "selected_genes*.csv")
    csv_pattern_standard_new = os.path.join(gene_lists_dir, "*", "*_factors", "*-genes", "selected_genes*.csv")

    # Priority 3: Baseline methods (hvg, random)
    csv_pattern_hvg_old = os.path.join(gene_lists_dir, "*", "*-genes", "selected_hvg_*.csv")
    csv_pattern_hvg_new = os.path.join(gene_lists_dir, "*", "*_factors", "*-genes", "selected_hvg_*.csv")
    csv_pattern_random_old = os.path.join(gene_lists_dir, "*", "*-genes", "selected_random_*.csv")
    csv_pattern_random_new = os.path.join(gene_lists_dir, "*", "*_factors", "*-genes", "selected_random_*.csv")

    # Collect all files by type
    ranked_files = glob.glob(csv_pattern_ranked_old) + glob.glob(csv_pattern_ranked_new)
    standard_files = glob.glob(csv_pattern_standard_old) + glob.glob(csv_pattern_standard_new)
    hvg_files = glob.glob(csv_pattern_hvg_old) + glob.glob(csv_pattern_hvg_new)
    random_files = glob.glob(csv_pattern_random_old) + glob.glob(csv_pattern_random_new)

    # Deduplicate: if ranked_gene_list.csv exists in a directory, exclude selected_genes*.csv from the same directory
    ranked_dirs = {os.path.dirname(f) for f in ranked_files}
    standard_files_filtered = [
        f for f in standard_files
        if os.path.dirname(f) not in ranked_dirs
    ]

    # Combine all files, excluding intermediate files
    csv_files = ranked_files + standard_files_filtered + hvg_files + random_files
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("intermediate_")]

    logger.info(
        f"Found {len(csv_files)} gene list files: "
        f"{len(ranked_files)} ranked_gene_list.csv, "
        f"{len(standard_files_filtered)} selected_genes.csv (no ranked equivalent), "
        f"{len(hvg_files)} HVG, "
        f"{len(random_files)} random"
    )

    for csv_file in sorted(csv_files):
        path_parts = csv_file.split(os.sep)

        try:
            # Find the base directory index by locating gene_lists_dir in the path
            gene_lists_dir_parts = gene_lists_dir.split(os.sep)
            base_idx = len(gene_lists_dir_parts) - 1

            # The file is at: gene_lists_dir/strategy/*-genes/selected_*.csv
            # or: gene_lists_dir/strategy/*_factors/*-genes/selected_*.csv
            size_dir_idx = len(path_parts) - 2  # *-genes is always 2nd from end
            size_dir = path_parts[size_dir_idx]

            # Check if there's a factor directory
            potential_factor_idx = size_dir_idx - 1
            potential_factor_dir = path_parts[potential_factor_idx]

            if potential_factor_dir.endswith("_factors"):
                # Factor-based structure: strategy/*_factors/*-genes/
                factor_dir = potential_factor_dir
                n_factors = factor_dir.replace("_factors", "")
                strategy = path_parts[potential_factor_idx - 1]
                # For the simplified pattern, we don't have filter_method and subset in the path
                # They would be part of gene_lists_dir itself
                subset = "All-Genes"  # Default assumption
                filter_method = "Scanpy-Filter"  # Default assumption
            else:
                # Standard structure: strategy/*-genes/
                factor_dir = None
                n_factors = None
                strategy = potential_factor_dir
                # For the simplified pattern, we don't have filter_method and subset in the path
                subset = "All-Genes"  # Default assumption
                filter_method = "Scanpy-Filter"  # Default assumption

            size = size_dir.replace("-genes", "")
            filename = os.path.basename(csv_file)
            filling_suffix = ""

            if filename == "ranked_gene_list.csv":
                # This is the authoritative panel file, no special handling needed
                logger.info(f"Detected ranked_gene_list.csv (authoritative panel)")

            elif filename.startswith("selected_hvg_"):
                size_match = re.search(r'selected_hvg_(\d+)genes\.csv', filename)
                if size_match:
                    size = size_match.group(1)
                logger.info(f"Detected HVG file with size {size}")

            elif filename.startswith("selected_random_"):
                size_match = re.search(r'selected_random_(\d+)genes\.csv', filename)
                if size_match:
                    size = size_match.group(1)
                logger.info(f"Detected random file with size {size}")
                if "_bootstrap" in filename:
                    logger.info(f"Skipping bootstrap file: {filename}")
                    continue

            elif filename != "selected_genes.csv":
                filling_suffix = filename.replace("selected_genes_", "").replace(".csv", "")
                logger.info(f"Detected fill-up strategy: {filling_suffix}")

            if n_factors is not None:
                if filling_suffix:
                    full_name = f"{filter_method}_{subset}_{strategy}_{n_factors}factors_{size}_{filling_suffix}"
                else:
                    full_name = f"{filter_method}_{subset}_{strategy}_{n_factors}factors_{size}"
            else:
                if filling_suffix:
                    full_name = f"{filter_method}_{subset}_{strategy}_{size}_{filling_suffix}"
                else:
                    full_name = f"{filter_method}_{subset}_{strategy}_{size}"

            genes = load_gene_list_from_csv(csv_file)

            if len(genes) == 0:
                logger.warning(f"Empty gene list: {full_name}")
                continue

            genes_filtered = [g for g in genes if g in available_genes]
            missing = len(genes) - len(genes_filtered)

            if missing > 0:
                logger.warning(f"{full_name}: {missing}/{len(genes)} genes not found in dataset")

            if len(genes_filtered) > 0:
                gene_lists[full_name] = genes_filtered
                logger.info(f"Loaded {full_name}: {len(genes_filtered)} genes from {csv_file}")
            else:
                logger.error(f"No valid genes found for {full_name}")

        except (ValueError, IndexError) as e:
            logger.error(f"Could not parse path {csv_file}: {e}")
            continue

    if len(gene_lists) == 0:
        # ── Flat-file fallback ──────────────────────────────────────────────
        # Support a simple flat layout where CSV files sit directly inside
        # gene_lists_dir (e.g. when called from the Streamlit app runner).
        flat_csv_files = glob.glob(os.path.join(gene_lists_dir, "ranked_gene_list.csv"))
        flat_csv_files += glob.glob(os.path.join(gene_lists_dir, "selected_genes*.csv"))
        flat_csv_files += glob.glob(os.path.join(gene_lists_dir, "selected_hvg_*.csv"))
        flat_csv_files += glob.glob(os.path.join(gene_lists_dir, "selected_random_*.csv"))
        flat_csv_files = [f for f in flat_csv_files
                          if not os.path.basename(f).startswith("intermediate_")
                          and "_bootstrap" not in os.path.basename(f)]

        if flat_csv_files:
            logger.info(f"Nested structure not found; falling back to {len(flat_csv_files)} flat CSV(s) in {gene_lists_dir}")
            for csv_file in sorted(flat_csv_files):
                full_name = os.path.splitext(os.path.basename(csv_file))[0]
                genes = load_gene_list_from_csv(csv_file)
                if not genes:
                    logger.warning(f"Empty gene list: {csv_file}")
                    continue
                genes_filtered = [g for g in genes if g in available_genes]
                missing = len(genes) - len(genes_filtered)
                if missing > 0:
                    logger.warning(f"{full_name}: {missing}/{len(genes)} genes not found in dataset")
                if genes_filtered:
                    gene_lists[full_name] = genes_filtered
                    logger.info(f"Loaded (flat) {full_name}: {len(genes_filtered)} genes")
                else:
                    logger.error(f"No valid genes found for flat file: {csv_file}")

    if len(gene_lists) == 0:
        logger.error(f"No gene lists loaded! Check directory structure.")
        logger.error(f"Expected structure: {gene_lists_dir}/strategy/size-genes/ranked_gene_list.csv")
        logger.error(f"Files can be: ranked_gene_list.csv (recommended), selected_genes.csv, selected_genes_DEG-based-filling.csv, selected_hvg_*genes.csv, selected_random_*genes.csv, etc.")
        logger.error(f"Note: ranked_gene_list.csv takes priority over selected_genes.csv when both exist")
        logger.error(f"Note: Files named 'intermediate_panel_before-gap-fill.csv' and bootstrap files are automatically excluded")
    else:
        logger.info(f"Successfully loaded {len(gene_lists)} gene lists")
        logger.info(f"Gene list names: {sorted(gene_lists.keys())}")

    return gene_lists


# ===================================================================
# PREPROCESSING FUNCTIONS
# ===================================================================

def process_data_for_panel_evaluation(
    adata: object,
    probeset: list[str],
    n_neighbors: int,
    layer: str = "counts",
    hvg: bool = False,
    subset: bool = False,
    scale: bool = False,
    dataset_name: str | None = None,
    dimensionality_reduction: str = "pca",
    filter_genes: bool = True,
    nmf_counts_input: str = "raw",
) -> object:
    """
    Process and preprocess AnnData object for panel evaluation.

    Performs normalization, dimensionality reduction (PCA/NMF), Leiden clustering
    at multiple resolutions (7-60 clusters), and KNN graph construction.

    Args:
        adata: The raw AnnData object.
        probeset: List of genes to keep.
        n_neighbors: Number of neighbors for KNN graph construction.
        layer: Layer to use for normalization and HVG selection (default: "counts").
        hvg: Whether to perform highly variable gene selection (default: False).
        subset: Whether to subset to highly variable genes (default: False).
        scale: Whether to scale the data before dimensionality reduction (default: False).
        dataset_name: Name of the dataset for logging purposes (optional).
        dimensionality_reduction: Which method to use: "pca", "nmf", or "both" (default: "pca").
        filter_genes: Whether to filter lowly expressed genes (default: True).
            Set to True for reference preprocessing, False for panel evaluation.
        nmf_counts_input: Count matrix to use as NMF input. ``"raw"`` (default):
            raw integer counts from ``adata.raw`` or ``adata.layers['counts']``
            (validated with ``is_anndata_raw_layer``). ``"lognorm"``: log-normalised
            counts from ``adata.X``.

    Returns:
        Processed AnnData object with embeddings, clusterings, and neighbor graphs.

    Raises:
        ValueError: If ``nmf_counts_input`` is unknown or requested raw counts are missing.
        Exception: If critical processing steps fail.

    Example:
        >>> adata_proc = process_data_for_panel_evaluation(
        ...     adata, genes, n_neighbors=15, dataset_name="my_panel"
        ... )
        >>> 'X_pca' in adata_proc.obsm
        True
    """
    if not adata.obs_names.is_unique:
        logger.info("Making observation names unique")
        adata.obs_names_make_unique()

    if dataset_name:
        logger.info(f"Checking quality for dataset: {dataset_name}")
        issues = []

        if not issparse(adata.X):
            if (adata.X < 0).any():
                num_neg = (adata.X < 0).sum()
                issues.append(f"Contains {num_neg} negative values")
        else:
            if adata.X.min() < 0:
                num_neg = np.sum(adata.X.data < 0)
                issues.append(f"Contains {num_neg} negative values (sparse)")

        cell_sums = adata.X.sum(axis=1)
        if issparse(cell_sums):
            cell_sums = cell_sums.A1
        zero_cells = (cell_sums == 0).sum()
        if zero_cells > 0:
            issues.append(f"Contains {zero_cells} cells with zero counts")

        if issues:
            logger.warning(f"WARNING: Dataset {dataset_name} has these issues:")
            for issue in issues:
                logger.warning(f" - {issue}")
        else:
            logger.info(f"Dataset {dataset_name} looks good!")

    is_log = not is_anndata_raw(adata)
    if is_log and dataset_name:
        logger.info(f"Dataset {dataset_name} appears to be log-transformed already")
        if 'counts' in adata.layers:
            counts_is_raw = is_anndata_raw_layer(adata, 'counts')
            if counts_is_raw:
                logger.info(f"Counts layer verified as raw data")
            else:
                logger.info(f"Counts layer is not raw data! Be careful with the interpretation of NMF results.")
        else:
            logger.info(f"No counts layer available and .X is log-normalized already. NMF analysis is not possible!")
    else:
        logger.info(f"Dataset {dataset_name} is not log-transformed already")
        if 'counts' in adata.layers:
            counts_is_raw = is_anndata_raw_layer(adata, 'counts')
            if counts_is_raw:
                logger.info(f"Counts layer verified as raw data")
            else:
                logger.info(f"Counts layer is not raw data. Transfer .X to counts layer")
                adata.layers['counts'] = adata.X.copy()
        else:
            logger.info(f"No counts layer available. Transfer .X to counts layer")
            adata.layers['counts'] = adata.X.copy()

    adata = adata[:, probeset].copy()
    # Ensure object is fully in-memory to prevent errno 11 (file locking) issues
    # This is crucial for large files where backing mode may retain file handles
    if hasattr(adata, 'isbacked') and adata.isbacked:
        adata = adata.to_memory()
    print(f"Processing dataset with {len(probeset)} genes")
    print(f"Raw data check: {is_anndata_raw(adata)}")

    if dataset_name:
        adata.uns["dataset_name"] = dataset_name
    else:
        adata.uns["dataset_name"] = f"geneset_{len(probeset)}_genes"

    if filter_genes:
        logger.info("Filtering low-quality cells and genes...")
        logger.info("Filtering cells with < 100 detected genes...")
        sc.pp.filter_cells(adata, min_genes=100)
        logger.info("Filtering genes expressed in < 3 cells...")
        sc.pp.filter_genes(adata, min_cells=3)
        logger.info(f"After filtering: {adata.n_obs} cells × {adata.n_vars} genes")
    else:
        logger.info("Skipping cell and gene filtering to preserve exact panel composition")
        logger.info("Note: Some genes/cells may have low expression but are important for rare cell types")
        logger.info(f"Proceeding with: {adata.n_obs} cells × {adata.n_vars} genes")

    adata.raw = adata

    if not is_log:
        print("Performing normalization and log transformation")
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    else:
        print("Skipping normalization and log1p as data appears to be log-transformed already")

    if hvg and not is_log:
        sc.pp.highly_variable_genes(adata,
                            flavor="seurat_v3",
                            n_top_genes=8000,
                            layer=layer,
                            subset=subset)

    if issparse(adata.X):
        adata.X = adata.X.toarray()

    if scale:
        print("Scaling data")
        if np.isinf(adata.X).any():
            print(f"WARNING: {np.isinf(adata.X).sum()} infinite values detected and replaced")
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)

        sc.pp.scale(adata, max_value=10)

        if np.isnan(adata.X).any():
            print(f"WARNING: {np.isnan(adata.X).sum()} NaN values detected after scaling")
            print("Replacing NaN values with zeros")
            adata.X = np.nan_to_num(adata.X, nan=0.0)

    if dimensionality_reduction in ["pca", "both"]:
        print("Running PCA")
        if np.isnan(adata.X).any():
            print("ERROR: Cannot run PCA - data contains NaN values")
            adata.X = np.nan_to_num(adata.X, nan=0.0)

        try:
            sc.tl.pca(adata)
        except Exception as e:
            print(f"PCA ERROR: {e}")
            print("Attempting to fix data and retry PCA...")
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                sc.tl.pca(adata, n_comps=min(30, adata.X.shape[1]-1))
            except Exception as e2:
                print(f"Second PCA attempt failed: {e2}")
                n_comps = min(50, adata.shape[1]-1)
                adata.obsm['X_pca'] = np.zeros((adata.shape[0], n_comps))
                print("WARNING: Using zeros for PCA results!")

    if dimensionality_reduction in ["nmf", "both"]:
        print("Running NMF")

        if nmf_counts_input == "raw":
            if hasattr(adata, 'raw') and adata.raw is not None:
                if not is_anndata_raw(adata.raw):
                    raise ValueError(
                        "nmf_counts_input='raw': adata.raw.X does not contain raw integer counts."
                    )
                X_nmf_input = adata.raw.X.copy() if hasattr(adata.raw.X, 'copy') else np.array(adata.raw.X)
                data_source = "raw"
                print("Using adata.raw.X for NMF (verified as raw counts)")
            elif 'counts' in adata.layers:
                if not is_anndata_raw_layer(adata, 'counts'):
                    raise ValueError(
                        "nmf_counts_input='raw': adata.layers['counts'] does not contain raw integer counts"
                    )
                X_nmf_input = adata.layers['counts'].copy()
                data_source = "counts_layer"
                print("Using layers['counts'] for NMF (verified as raw counts)")
            else:
                raise ValueError(
                    "nmf_counts_input='raw': no raw counts found in adata.raw or adata.layers['counts']"
                )
        elif nmf_counts_input == "lognorm":
            # Only guard when the data arrived already log-normalised (is_log=True).
            # When is_log=False, normalize_total+log1p ran above and X is guaranteed
            # to be log-normalised — no need to re-check (and the stub always returns
            # True, which would cause a false positive in that case).
            if is_log and is_anndata_raw(adata):
                raise ValueError(
                    "nmf_counts_input='lognorm': adata.X appears to contain raw integer counts, "
                    "not log-normalized data. Normalize adata.X before evaluation."
                )
            X_nmf_input = adata.X.copy() if hasattr(adata.X, 'copy') else np.array(adata.X)
            data_source = "X_lognorm"
            print("Using adata.X (log-normalized, verified) for NMF")
        else:
            raise ValueError(
                f"Unknown nmf_counts_input='{nmf_counts_input}'. Choose 'raw' or 'lognorm'."
            )

        if issparse(X_nmf_input):
            X_nmf_input = X_nmf_input.toarray()

        if np.isnan(X_nmf_input).any() or np.isinf(X_nmf_input).any():
            print("ERROR: Cannot run NMF - data contains NaN or inf values")
            X_nmf_input = np.nan_to_num(X_nmf_input, nan=0.0, posinf=0.0, neginf=0.0)
            if (X_nmf_input < 0).any():
                print("ERROR: Data contains negative values")

        try:
            n_comps = 50
            print(f"Running NMF with {n_comps} components on {X_nmf_input.shape[0]} cells x {X_nmf_input.shape[1]} genes")
            nmf_model = NMF(n_components=n_comps, init='nndsvda', random_state=42, max_iter=1000)
            adata.obsm['X_nmf'] = nmf_model.fit_transform(X_nmf_input)
            adata.varm['nmf_components'] = nmf_model.components_.T
            adata.uns['nmf_data_source'] = data_source
            print(f"NMF completed with {n_comps} components")
            print(f"NMF reconstruction error: {nmf_model.reconstruction_err_:.4f}")
        except Exception as e:
            print(f"NMF ERROR: {e}")
            print("Creating empty NMF results to prevent downstream errors")
            n_comps = 5
            adata.obsm['X_nmf'] = np.zeros((adata.shape[0], n_comps))
            adata.uns['nmf_data_source'] = "failed"
            print("WARNING: Using zeros for NMF results!")


    if dimensionality_reduction == "pca":
        use_reps = ['X_pca']
        rep_names = ['pca']
    elif dimensionality_reduction == "nmf":
        use_reps = ['X_nmf']
        rep_names = ['nmf']
    else:
        use_reps = ['X_pca', 'X_nmf']
        rep_names = ['pca', 'nmf']
        print("Note: Both PCA and NMF computed. Running neighbors computation for both representations.")

    neighbor_params = [5, 10, 15, 20, 30, 50]

    for rep_idx, (use_rep, rep_name) in enumerate(zip(use_reps, rep_names)):
        print(f"\nProcessing neighbors for {rep_name.upper()} representation...")

        for n_neighs in neighbor_params:
            print(f"Computing neighbors with n_neighbors={n_neighs} using {rep_name}")

            if dimensionality_reduction == "both":
                key_added = f"neighbors_{rep_name}_k{n_neighs}"
            else:
                key_added = f"neighbors_k{n_neighs}"

            sc.pp.neighbors(adata, n_neighbors=n_neighs, key_added=key_added, use_rep=use_rep)
            sc.tl.umap(adata, neighbors_key=key_added)

            if dimensionality_reduction == "both":
                adata.obsm[f"X_umap_{rep_name}_k{n_neighs}"] = adata.obsm["X_umap"].copy()
            else:
                adata.obsm[f"X_umap_k{n_neighs}"] = adata.obsm["X_umap"].copy()

        if dimensionality_reduction == "both":
            neighbors_key = f"neighbors_{rep_name}"
        else:
            neighbors_key = "neighbors"

        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, key_added=neighbors_key if dimensionality_reduction == "both" else None)
        sc.tl.umap(adata, neighbors_key=neighbors_key if dimensionality_reduction == "both" else None)

        if dimensionality_reduction == "both":
            adata.obsm[f"X_umap_{rep_name}"] = adata.obsm["X_umap"].copy()

    if dimensionality_reduction == "both":
        clustering_neighbors_keys = ['neighbors_pca', 'neighbors_nmf']
        clustering_rep_names = ['pca', 'nmf']
    else:
        clustering_neighbors_keys = [None]
        clustering_rep_names = [dimensionality_reduction]

    for neighbors_key, rep_name in zip(clustering_neighbors_keys, clustering_rep_names):
        if dimensionality_reduction == "both":
            print(f"\n=== Running Leiden clustering for {rep_name.upper()} representation ===")
        else:
            print(f"\n=== Running Leiden clustering ===")

        target_clusters = list(range(7, 61))
        test_resolutions = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        resolution_to_clusters = {}

        print("Finding resolutions for target cluster numbers...")

        for res in test_resolutions:
            if dimensionality_reduction == "both":
                leiden_key = f"leiden_{rep_name}_res{res}"
            else:
                leiden_key = f"leiden_res{res}"

            sc.tl.leiden(adata, resolution=res, key_added=leiden_key, neighbors_key=neighbors_key,
                        flavor="igraph", n_iterations=2, directed=False)
            n_clusters = len(pd.unique(adata.obs[leiden_key]))
            resolution_to_clusters[res] = n_clusters
            print(f"Resolution {res:.2f} gives {n_clusters} clusters")

        for target in target_clusters:
            if target in resolution_to_clusters.values():
                res = [r for r, n in resolution_to_clusters.items() if n == target][0]
                print(f"Using existing resolution {res:.2f} for {target} clusters")
                if dimensionality_reduction == "both":
                    leiden_key = f"leiden_{target}_clusters_{rep_name}"
                    source_key = f"leiden_{rep_name}_res{res}"
                else:
                    leiden_key = f"leiden_{target}_clusters"
                    source_key = f"leiden_res{res}"
                adata.obs[leiden_key] = adata.obs[source_key]
                continue

            lower_res = None
            upper_res = None
            lower_n = 0
            upper_n = float('inf')

            for res, n_clusters in resolution_to_clusters.items():
                if n_clusters <= target and n_clusters > lower_n:
                    lower_res = res
                    lower_n = n_clusters
                if n_clusters >= target and n_clusters < upper_n:
                    upper_res = res
                    upper_n = n_clusters

            if lower_res is not None and upper_res is not None:
                interp_factor = (target - lower_n) / (upper_n - lower_n)
                new_res = lower_res + interp_factor * (upper_res - lower_res)
            elif lower_res is not None:
                new_res = lower_res * 1.5
            elif upper_res is not None:
                new_res = upper_res * 0.7
            else:
                new_res = 1.0

            attempts = 0
            max_attempts = 10

            while attempts < max_attempts:
                if dimensionality_reduction == "both":
                    leiden_key = f"leiden_{rep_name}_res{new_res}"
                else:
                    leiden_key = f"leiden_res{new_res}"

                sc.tl.leiden(adata, resolution=new_res, key_added=leiden_key, neighbors_key=neighbors_key,
                           flavor="igraph", n_iterations=2, directed=False)
                n_clusters = len(pd.unique(adata.obs[leiden_key]))
                resolution_to_clusters[new_res] = n_clusters
                print(f"Resolution {new_res:.3f} gives {n_clusters} clusters (target: {target})")

                if n_clusters == target:
                    if dimensionality_reduction == "both":
                        adata.obs[f"leiden_{target}_clusters_{rep_name}"] = adata.obs[leiden_key]
                    else:
                        adata.obs[f"leiden_{target}_clusters"] = adata.obs[leiden_key]
                    break

                if n_clusters < target:
                    lower_res = new_res
                    lower_n = n_clusters
                else:
                    upper_res = new_res
                    upper_n = n_clusters

                if lower_res is not None and upper_res is not None:
                    interp_factor = (target - lower_n) / (upper_n - lower_n)
                    new_res = lower_res + interp_factor * (upper_res - lower_res)
                elif lower_res is not None:
                    new_res = lower_res * 1.2
                else:
                    new_res = upper_res * 0.8

                attempts += 1

            if attempts == max_attempts:
                print(f"Could not find exact resolution for {target} clusters after {max_attempts} attempts")
                closest_res = min(resolution_to_clusters.items(), key=lambda x: abs(x[1] - target))
                print(f"Using resolution {closest_res[0]:.3f} which gives {closest_res[1]} clusters")
                if dimensionality_reduction == "both":
                    source_key = f"leiden_{rep_name}_res{closest_res[0]}"
                    target_key = f"leiden_{target}_clusters_{rep_name}"
                else:
                    source_key = f"leiden_res{closest_res[0]}"
                    target_key = f"leiden_{target}_clusters"
                adata.obs[target_key] = adata.obs[source_key]

    if dimensionality_reduction == "both":
        default_neighbors_key = "neighbors_pca"
    else:
        default_neighbors_key = None

    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, neighbors_key=default_neighbors_key)

    print("Computing neighborhood similarity across different neighbor parameters...")

    return adata


def preprocess_reference_dataset(
    adata: object,
    output_file: str,
    dimensionality_reduction: str = "both",
    n_neighbors: int = 15,
    n_nmf_components: int = 5
) -> None:
    """
    Preprocess the full-transcriptome reference dataset for evaluation.

    Applies the same preprocessing pipeline as panel evaluation to the full reference,
    enabling fair comparison between probe panels and the reference transcriptome.

    Args:
        adata: AnnData object with raw data (full transcriptome).
        output_file: Path to save preprocessed reference h5ad file.
        dimensionality_reduction: Type of reduction: "pca", "nmf", or "both" (default: "both").
        n_neighbors: Number of neighbors for UMAP (default: 15).
        n_nmf_components: Number of NMF components (default: 5).

    Returns:
        None (saves to output_file).

    Raises:
        FileNotFoundError: If input data cannot be loaded.
        Exception: If preprocessing fails.

    Example:
        >>> preprocess_reference_dataset(
        ...     adata, '/path/to/reference.h5ad',
        ...     dimensionality_reduction='both', n_neighbors=15
        ... )
    """
    logger.info("Making observation names unique")
    adata.obs_names_make_unique()

    logger.info("Ensuring counts layer exists")
    if 'counts' not in adata.layers:
        if is_anndata_raw(adata):
            logger.info("Creating 'counts' layer from X (verified as raw counts)")
            adata.layers['counts'] = adata.X.copy()
        elif adata.raw is not None:
            import anndata
            raw_X = adata.raw[:, adata.var_names].X
            tmp = anndata.AnnData(X=raw_X)
            if not is_anndata_raw(tmp):
                raise ValueError(
                    "preprocess_reference_dataset: adata.X is not raw counts and adata.raw.X "
                    "also does not appear to contain raw counts. Cannot create layers['counts']."
                )
            logger.info("Restoring 'counts' layer from adata.raw.X (adata.X appears normalized)")
            adata.layers['counts'] = raw_X.copy()
        else:
            raise ValueError(
                "preprocess_reference_dataset: adata.X does not appear to contain raw counts "
                "(is_anndata_raw=False) and adata.raw is None. Cannot create layers['counts'] "
                "from normalized data. Pre-populate adata.layers['counts'] with raw counts "
                "before calling this function."
            )

    all_genes = adata.var_names.tolist()
    logger.info(f"\nPreprocessing reference with {len(all_genes)} genes...")

    adata_processed = process_data_for_panel_evaluation(
        adata=adata,
        probeset=all_genes,
        n_neighbors=n_neighbors,
        layer="counts",
        hvg=False,
        subset=False,
        scale=False,
        dataset_name="full_transcriptome",
        dimensionality_reduction=dimensionality_reduction,
        filter_genes=True,
    )

    if dimensionality_reduction in ["nmf", "both"]:
        if 'X_nmf' not in adata_processed.obsm:
            logger.info(f"Computing NMF with {n_nmf_components} components...")
            from sklearn.decomposition import NMF as sklearn_NMF

            X = adata_processed.X
            if hasattr(X, 'toarray'):
                X = X.toarray()

            X = np.abs(X)

            nmf = sklearn_NMF(
                n_components=n_nmf_components,
                init='nndsvda',
                max_iter=1000,
                random_state=42
            )
            W = nmf.fit_transform(X)

            adata_processed.obsm['X_nmf'] = W
            adata_processed.varm['nmf_components'] = nmf.components_.T
            logger.info(f"NMF complete: {W.shape}")

    has_pca = 'X_pca' in adata_processed.obsm
    has_nmf = 'X_nmf' in adata_processed.obsm
    leiden_cols = [c for c in adata_processed.obs.columns if c.startswith('leiden_')]
    neighbor_keys = [k for k in adata_processed.uns.keys() if k.startswith('neighbors_')]

    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING VERIFICATION")
    logger.info("="*60)
    logger.info(f"PCA: {'✓' if has_pca else '✗'}")
    logger.info(f"NMF: {'✓' if has_nmf else '✗'}")
    logger.info(f"Leiden clusterings: {len(leiden_cols)}")
    logger.info(f"Neighbor graphs: {len(neighbor_keys)}")

    if has_pca:
        logger.info(f"  PCA shape: {adata_processed.obsm['X_pca'].shape}")
    if has_nmf:
        logger.info(f"  NMF shape: {adata_processed.obsm['X_nmf'].shape}")

    logger.info(f"\nSaving preprocessed reference to: {output_file}")

    adata_processed = _fix_index_column(adata_processed)
    adata_processed = _convert_arrow_strings_to_object(adata_processed)

    # Set anndata flag as fallback for any remaining string arrays (requires anndata >= 0.11)
    try:
        import anndata
        if hasattr(anndata.settings, 'allow_write_nullable_strings'):
            anndata.settings.allow_write_nullable_strings = True
            logger.info("Set anndata.settings.allow_write_nullable_strings = True")
    except Exception as e:
        logger.debug(f"Could not set anndata.settings.allow_write_nullable_strings: {e}")

    adata_processed.write_h5ad(output_file, compression='gzip')

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.2f} MB")

    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*80)
