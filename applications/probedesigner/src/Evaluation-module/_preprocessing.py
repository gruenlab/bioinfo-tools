"""Preprocessing functions for evaluation pipeline datasets.

This module provides utilities to:
- Load gene lists from selection pipeline output directories.
- Preprocess probe-panel subsets: normalize, compute PCA,
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
import re
from pathlib import Path
from scipy.sparse import issparse

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
    "filter_reference_blacklist",
]

# ===================================================================
# UTILITY IMPORTS
# ===================================================================

# Canonical raw-count check — hard import (one _validation.py in the cut).
_util_dir = Path(__file__).parent.parent / "Utility-module"
if str(_util_dir) not in sys.path:
    sys.path.insert(0, str(_util_dir))
from _validation import is_anndata_raw, is_anndata_raw_layer

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

try:
    _selection_dir = Path(__file__).parent.parent / "Selection-module"
    if _selection_dir.exists() and str(_selection_dir) not in sys.path:
        sys.path.insert(0, str(_selection_dir))
    from _filtering import apply_blacklist_filter
except ImportError:
    logger.warning("Could not import apply_blacklist_filter from Selection-module; reference blacklist filtering will be unavailable.")
    apply_blacklist_filter = None


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
    4. A single-strategy panel-information CSV with an 'in_panel' (or legacy
       'final_selection'/'selected_final') column (filters to True values)

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

        # Check if this is a panel-information/ranked-gene-list CSV with a panel-membership column
        if 'gene' in df.columns:
            for panel_col in ('final_selection', 'in_panel', 'selected_final'):
                if panel_col in df.columns:
                    logger.info(
                        f"Found panel-information format with '{panel_col}' column in "
                        f"{os.path.basename(filepath)}"
                    )
                    df_selected = df[df[panel_col].fillna(False).astype(bool)]
                    genes = df_selected['gene'].tolist()
                    logger.info(
                        f"Filtered to {len(genes)} genes with {panel_col}=True "
                        f"(from {len(df)} total genes)"
                    )
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

    Handles both directory-layout variants:
    - .../Filter/Baseline/Strategy/Size-genes/results/selected_genes.csv
    - .../Filter/Baseline/Strategy/N_factors/Size-genes/results/selected_genes.csv

    Args:
        csv_file: Path to a gene list CSV file.

    Returns:
        Standardized name for the gene list.

    Example:
        >>> name = extract_genelist_name_from_path('/path/Scanpy-Filter/All-Genes/rf_deg/100-genes/results/selected_genes.csv')
        >>> name
        'Scanpy-Filter_All-Genes_rf_deg_100'
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


def load_all_gene_lists(
    gene_lists_dir: str, adata: object, return_paths: bool = False
) -> dict[str, list[str]] | tuple[dict[str, list[str]], dict[str, str]]:
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

    Fill-up strategies (when dimred selection doesn't reach target panel size):
    - DEG-based-filling: Fill up with top DEGs
    - cell-type-specific-filling: Fill up with per-celltype dimred genes
    - global-gene-filling: Fill up with global dimred genes

    Note: Files named 'intermediate_panel_before-gap-fill.csv' and bootstrap files
    are excluded from evaluation as they represent incomplete gene panels.

    Args:
        gene_lists_dir: Base directory containing Selected-panels output.
        adata: Reference dataset (to check gene availability).
        return_paths: When True, also return a ``{gene_list_name: source_csv_path}``
            mapping (same keys as the gene-list dict), so callers can locate each
            panel's originating ``ranked_gene_list.csv`` (e.g. for the
            ``informative_celltypes`` column). Default False keeps the historical
            single-dict return.

    Returns:
        Dictionary mapping gene list names to gene lists
        (format ``{filter}_{subset}_{strategy}_{size}[_{filling_suffix}]``), or a
        ``(gene_lists, path_map)`` tuple when ``return_paths=True``.

    Example:
        >>> lists = load_all_gene_lists('/path/to/Selected-panels', adata)
        >>> 'Scanpy-Filter_All-Genes_rf_deg_100' in lists
        True
    """
    gene_lists = {}
    path_map: dict[str, str] = {}
    available_genes = set(adata.var_names)

    if not os.path.exists(gene_lists_dir):
        logger.error(f"Gene lists directory not found: {gene_lists_dir}")
        return (gene_lists, path_map) if return_paths else gene_lists

    logger.info(f"Scanning directory structure: {gene_lists_dir}")

    # Priority 1: ranked_gene_list.csv / *_panel_information.csv (authoritative source with an
    # in_panel/final_selection column). *_panel_information.csv covers the renamed
    # single-strategy outputs (rf_deg_panel_information.csv, nmf_panel_information.csv, ...).
    csv_pattern_ranked_old = os.path.join(gene_lists_dir, "*", "*-genes", "ranked_gene_list.csv")
    csv_pattern_ranked_new = os.path.join(gene_lists_dir, "*", "*_factors", "*-genes", "ranked_gene_list.csv")
    csv_pattern_panelinfo_old = os.path.join(gene_lists_dir, "*", "*-genes", "*_panel_information.csv")
    csv_pattern_panelinfo_new = os.path.join(
        gene_lists_dir, "*", "*_factors", "*-genes", "*_panel_information.csv"
    )

    # Priority 2: selected_genes*.csv (legacy format, fallback when ranked_gene_list.csv doesn't exist)
    csv_pattern_standard_old = os.path.join(gene_lists_dir, "*", "*-genes", "selected_genes*.csv")
    csv_pattern_standard_new = os.path.join(gene_lists_dir, "*", "*_factors", "*-genes", "selected_genes*.csv")

    # Priority 3: Baseline methods (hvg, random)
    csv_pattern_hvg_old = os.path.join(gene_lists_dir, "*", "*-genes", "selected_hvg_*.csv")
    csv_pattern_hvg_new = os.path.join(gene_lists_dir, "*", "*_factors", "*-genes", "selected_hvg_*.csv")
    csv_pattern_random_old = os.path.join(gene_lists_dir, "*", "*-genes", "selected_random_*.csv")
    csv_pattern_random_new = os.path.join(gene_lists_dir, "*", "*_factors", "*-genes", "selected_random_*.csv")

    # Collect all files by type
    ranked_files = (
        glob.glob(csv_pattern_ranked_old) + glob.glob(csv_pattern_ranked_new)
        + glob.glob(csv_pattern_panelinfo_old) + glob.glob(csv_pattern_panelinfo_new)
    )
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
        f"{len(ranked_files)} ranked_gene_list.csv/*_panel_information.csv, "
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

            if filename == "ranked_gene_list.csv" or filename.endswith("_panel_information.csv"):
                # This is the authoritative panel file, no special handling needed
                logger.info(f"Detected {filename} (authoritative panel)")

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
                path_map[full_name] = csv_file
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
        flat_csv_files += glob.glob(os.path.join(gene_lists_dir, "*_panel_information.csv"))
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
                    path_map[full_name] = csv_file
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

    return (gene_lists, path_map) if return_paths else gene_lists


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
        dataset_name: Name of the dataset for logging purposes (optional).
        dimensionality_reduction: Ignored; panel preprocessing computes only PCA.
        filter_genes: Whether to filter lowly expressed genes (default: True).
            Set to True for reference preprocessing, False for panel evaluation.
        nmf_counts_input: Ignored; kept only so callers that also drive the NMF
            evaluation can pass a single value through.

    Returns:
        Processed AnnData object with embeddings, clusterings, and neighbor graphs.

    Raises:
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

    var_set = set(adata.var_names)
    missing = [g for g in probeset if g not in var_set]
    if missing:
        logger.info(
            f"NOTE: {len(missing)} gene(s) from the probe panel are not present in the "
            f"single-cell dataset and will be excluded from evaluation: {missing}. "
            f"These genes exist in the 10x Xenium panel but were not captured in the "
            f"scRNA-seq reference (gene name aliases or tissue-specific absence)."
        )
        probeset = [g for g in probeset if g in var_set]
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

    # Only store .raw when there is no counts layer to serve as the raw-count
    # reference.  When layers['counts'] already contains verified raw counts
    # (the normal case for HCA/ST data), .raw would just duplicate the
    # subsetted lognorm matrix and waste memory, especially for the full-
    # transcriptome reference (~167K cells × 28K genes ≈ 18 GB in float64).
    if 'counts' not in adata.layers:
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

    # Keep adata.X sparse here — sc.tl.pca handles sparse input via ARPACK.
    # Densifying the full-transcriptome reference (167K × 28K genes) wastes ~18 GB.
    # The NaN check below is made sparse-aware so we never need to materialise the
    # full dense matrix just to validate the data.

    if dimensionality_reduction in ["pca", "both"]:
        print("Running PCA")
        # Sparse-aware NaN check: inspect only the stored non-zero values (.data)
        # so we never have to densify adata.X just for validation.
        _x_has_nan = (
            np.isnan(adata.X.data).any() if issparse(adata.X) else np.isnan(adata.X).any()
        )
        if _x_has_nan:
            print("ERROR: Cannot run PCA - data contains NaN values")
            if issparse(adata.X):
                adata.X = adata.X.copy()
                adata.X.data[:] = np.nan_to_num(adata.X.data, nan=0.0)
            else:
                adata.X = np.nan_to_num(adata.X, nan=0.0)

        try:
            sc.tl.pca(adata)
        except Exception as e:
            print(f"PCA ERROR: {e}")
            print("Attempting to fix data and retry PCA...")
            # Densify only in this rare fallback path so the retry has a clean array.
            if issparse(adata.X):
                adata.X = adata.X.toarray()
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                sc.tl.pca(adata, n_comps=min(30, adata.X.shape[1]-1))
            except Exception as e2:
                print(f"Second PCA attempt failed: {e2}")
                n_comps = min(50, adata.shape[1]-1)
                adata.obsm['X_pca'] = np.zeros((adata.shape[0], n_comps))
                print("WARNING: Using zeros for PCA results!")

    use_reps = ['X_pca']
    rep_names = ['pca']

    neighbor_params = [5, 10, 15, 20, 30, 50]

    for rep_idx, (use_rep, rep_name) in enumerate(zip(use_reps, rep_names)):
        print(f"\nProcessing neighbors for {rep_name.upper()} representation...")

        for n_neighs in neighbor_params:
            print(f"Computing neighbors with n_neighbors={n_neighs} using {rep_name}")
            key_added = f"neighbors_k{n_neighs}"
            sc.pp.neighbors(adata, n_neighbors=n_neighs, key_added=key_added, use_rep=use_rep)
            # Per-k neighbour graphs feed neighbourhood-preservation. No per-k UMAP is
            # computed here — only the graphs are consumed downstream.

        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
        sc.tl.umap(adata)

    clustering_neighbors_keys = [None]
    clustering_rep_names = ['pca']

    for neighbors_key, rep_name in zip(clustering_neighbors_keys, clustering_rep_names):
        print(f"\n=== Running Leiden clustering ===")

        target_clusters = list(range(7, 61))
        test_resolutions = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        resolution_to_clusters = {}

        print("Finding resolutions for target cluster numbers...")

        for res in test_resolutions:
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
                adata.obs[f"leiden_{target}_clusters"] = adata.obs[f"leiden_res{res}"]
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
                leiden_key = f"leiden_res{new_res}"

                sc.tl.leiden(adata, resolution=new_res, key_added=leiden_key, neighbors_key=neighbors_key,
                           flavor="igraph", n_iterations=2, directed=False)
                n_clusters = len(pd.unique(adata.obs[leiden_key]))
                resolution_to_clusters[new_res] = n_clusters
                print(f"Resolution {new_res:.3f} gives {n_clusters} clusters (target: {target})")

                if n_clusters == target:
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
                # NOTE: the column keeps the `{target}` name even though it holds a
                # different cluster count. Downstream (compute_clustering_similarity)
                # pairs reference/panel columns on the *realized* count, not this name,
                # so the mismatch does not corrupt ARI/NMI.
                adata.obs[f"leiden_{target}_clusters"] = adata.obs[f"leiden_res{closest_res[0]}"]

    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

    print("Computing neighborhood similarity across different neighbor parameters...")

    return adata


def filter_reference_blacklist(
    adata: object,
    blacklist_patterns: list[str] | None = None,
    use_default_blacklist: bool = False,
) -> object:
    """Remove genes matching Selection-module's blacklist patterns from the reference.

    Delegates to :func:`_filtering.apply_blacklist_filter` (Selection-module) so the
    reference dataset can be filtered with the exact same rule (case-insensitive gene
    name prefix match) used when the panel itself was selected, letting fair
    comparisons be made against a panel that was barred from ever including those
    genes (e.g. a ribosomal-gene blacklist).

    Args:
        adata: Reference AnnData (full transcriptome), not yet preprocessed.
        blacklist_patterns: Extra prefix patterns to remove, beyond the default
            (e.g. custom patterns used during selection).
        use_default_blacklist: Also remove genes matching Selection-module's
            ``DEFAULT_BLACKLIST_PATTERNS`` (``mt-``, ``hsp``, ``rps``, ``rpl``).

    Returns:
        The filtered AnnData (unchanged object if no patterns are requested).
    """
    if not blacklist_patterns and not use_default_blacklist:
        return adata
    if apply_blacklist_filter is None:
        raise ImportError(
            "apply_blacklist_filter could not be imported from Selection-module; "
            "cannot apply --reference_blacklist_patterns / --reference_disable_default_blacklist."
        )
    filtered_genes, removed_genes, _ = apply_blacklist_filter(
        gene_list=adata.var_names.tolist(),
        blacklist_patterns=blacklist_patterns or None,
        use_default_blacklist=use_default_blacklist,
    )
    if removed_genes:
        logger.info(
            "Reference blacklist filter removed %d genes (e.g. %s)",
            len(removed_genes), removed_genes[:10],
        )
        adata = adata[:, filtered_genes].copy()
    return adata


def preprocess_reference_dataset(
    adata: object,
    output_file: str,
    dimensionality_reduction: str = "pca",
    n_neighbors: int = 15,
) -> None:
    """
    Preprocess the full-transcriptome reference dataset for evaluation.

    Applies the same preprocessing pipeline as panel evaluation to the full reference,
    enabling fair comparison between probe panels and the reference transcriptome.

    Args:
        adata: AnnData object with raw data (full transcriptome).
        output_file: Path to save preprocessed reference h5ad file.
        dimensionality_reduction: Ignored; reference preprocessing computes only PCA.
        n_neighbors: Number of neighbors for UMAP (default: 15).

    Returns:
        None (saves to output_file).

    Raises:
        FileNotFoundError: If input data cannot be loaded.
        Exception: If preprocessing fails.

    Example:
        >>> preprocess_reference_dataset(
        ...     adata, '/path/to/reference.h5ad',
        ...     n_neighbors=15
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
        dataset_name="full_transcriptome",
        dimensionality_reduction=dimensionality_reduction,
        filter_genes=True,
    )

    has_pca = 'X_pca' in adata_processed.obsm
    leiden_cols = [c for c in adata_processed.obs.columns if c.startswith('leiden_')]
    neighbor_keys = [k for k in adata_processed.uns.keys() if k.startswith('neighbors_')]

    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING VERIFICATION")
    logger.info("="*60)
    logger.info(f"PCA: {'✓' if has_pca else '✗'}")
    logger.info(f"Leiden clusterings: {len(leiden_cols)}")
    logger.info(f"Neighbor graphs: {len(neighbor_keys)}")

    if has_pca:
        logger.info(f"  PCA shape: {adata_processed.obsm['X_pca'].shape}")

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
