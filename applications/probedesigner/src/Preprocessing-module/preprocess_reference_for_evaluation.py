#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Preprocess reference dataset for evaluation pipeline.

This script preprocesses the full transcriptome reference dataset to include:
- PCA embeddings (50 components)
- NMF embeddings (5 components)
- Leiden clustering (resolutions 7-60 clusters)
- K-neighborhood graphs (k=5,10,15,20,30,50)
- UMAP visualizations

The preprocessed reference is required for the evaluation pipeline.

Usage:
    python preprocess_reference_for_evaluation.py \\
        --input_file /path/to/raw_data.h5ad \\
        --output_file /path/to/preprocessed/full_transcriptome.h5ad \\
        --dimensionality_reduction both \\
        --n_neighbors 15
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import NMF as sklearn_NMF

# Configure pandas to use object dtype for strings instead of ArrowStringArray
# This ensures compatibility with anndata's HDF5 writer (pandas 2.x+ uses PyArrow strings by default)
pd.options.mode.string_storage = "python"

# ---------------------------------------------------------------------------
# Set up import paths BEFORE importing any local modules.
#
# Multiple modules (Preprocessing, Evaluation, Utility) each have their own
# _constants.py — relying on sys.path ordering to pick the right one is
# fragile. Instead we load Preprocessing-module/_constants.py explicitly by
# file path using importlib, then add the other module directories for the
# remaining imports (_preprocessing, _utils).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent.absolute()
_MODULES_DIR = _SCRIPT_DIR.parent           # SpatialProbeDesign/Modules/
_PROJECT_DIR = _MODULES_DIR.parent          # SpatialProbeDesign/

# Load Preprocessing-module constants by explicit path to avoid any ambiguity
import importlib.util as _ilu                                                  # noqa: E402
_cspec = _ilu.spec_from_file_location("_preproc_constants", _SCRIPT_DIR / "_constants.py")
_cmod  = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_cmod)
DEFAULT_N_COMPONENTS_NMF = _cmod.DEFAULT_N_COMPONENTS_NMF
DEFAULT_RANDOM_STATE      = _cmod.DEFAULT_RANDOM_STATE
del _ilu, _cspec, _cmod

# Now add Evaluation-module and Utility-module for their unique imports
_EVAL_MODULE_DIR = _MODULES_DIR / "Evaluation-module"
if str(_EVAL_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_MODULE_DIR))

_UTILITY_DIR = _MODULES_DIR / "Utility-module"
if str(_UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILITY_DIR))

from _preprocessing import process_data_for_panel_evaluation  # noqa: E402

# _utils.py uses relative imports (from ._constants import ...) that only work
# when imported as part of its package — importing it via sys.path at module
# level breaks that. We import it lazily inside functions instead.

# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ['main', 'preprocess_reference_for_analysis_scripts']


def setup_logging() -> None:
    """Configure logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def _fix_index_columns(adata_processed) -> None:
    """Remove or rename the reserved '_index' column in var / raw.var if present.

    Extracted into a helper so it can be shared by both main() and
    preprocess_reference_for_analysis_scripts().
    """
    # Fix raw.var
    if adata_processed.raw is not None and '_index' in adata_processed.raw.var.columns:
        import anndata
        index_col    = adata_processed.raw.var['_index']
        current_index = adata_processed.raw.var.index
        if index_col.equals(current_index):
            new_var = adata_processed.raw.var.drop(columns=['_index'])
            adata_processed.raw = anndata.AnnData(
                X=adata_processed.raw.X, var=new_var, varm=adata_processed.raw.varm)
        else:
            is_numeric = all(str(idx).isdigit() for idx in current_index[:10])
            if is_numeric:
                new_var = adata_processed.raw.var.drop(columns=['_index'])
                new_var.index = index_col
                new_var.index.name = None
                adata_processed.raw = anndata.AnnData(
                    X=adata_processed.raw.X, var=new_var, varm=adata_processed.raw.varm)
            else:
                new_var = adata_processed.raw.var.rename(columns={'_index': 'original_index'})
                adata_processed.raw = anndata.AnnData(
                    X=adata_processed.raw.X, var=new_var, varm=adata_processed.raw.varm)

    # Fix var
    if '_index' in adata_processed.var.columns:
        index_col    = adata_processed.var['_index']
        current_index = adata_processed.var.index
        if index_col.equals(current_index):
            adata_processed.var = adata_processed.var.drop(columns=['_index'])
        else:
            adata_processed.var = adata_processed.var.rename(columns={'_index': 'original_index'})


def preprocess_reference_for_analysis_scripts(
    input_file: str,
    output_file: str,
    celltype_column: str = "celltype",
    dimensionality_reduction: str = "both",
    n_neighbors: int = 15,
    n_nmf_components: int = DEFAULT_N_COMPONENTS_NMF,
) -> None:
    """Preprocess raw h5ad as the canonical input for analysis scripts.

    Produces an AnnData with:
    - ``layers["counts"]``  — raw integer counts (for NMF in analysis scripts)
    - ``.X``                — log-normalised values
    - ``obsm["X_umap"]``    — UMAP embedding (needed by stability-analysis feature plots)
    - PCA embeddings and Leiden clusterings

    Reuses the same logic as :func:`main` but is importable as a Python function,
    allowing other modules (e.g. ``preprocess_for_selection.py``) to call it without
    subprocess overhead.

    Parameters
    ----------
    input_file:
        Path to raw h5ad file (raw integer counts + obs[celltype_column]).
    output_file:
        Path where the preprocessed h5ad will be written.
    celltype_column:
        Column in ``.obs`` that contains cell-type labels.
    dimensionality_reduction:
        ``"pca"``, ``"nmf"``, or ``"both"`` (default).
    n_neighbors:
        Number of neighbours for UMAP construction.
    n_nmf_components:
        Number of NMF components to compute.
    """
    logger.info("=" * 80)
    logger.info("PREPROCESSING FOR ANALYSIS SCRIPTS")
    logger.info("=" * 80)
    logger.info(f"Input file:               {input_file}")
    logger.info(f"Output file:              {output_file}")
    logger.info(f"Cell type column:         {celltype_column}")
    logger.info(f"Dimensionality reduction: {dimensionality_reduction}")
    logger.info(f"N neighbours:             {n_neighbors}")
    logger.info(f"N NMF components:         {n_nmf_components}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir_str = os.path.dirname(output_file)
    if output_dir_str:
        os.makedirs(output_dir_str, exist_ok=True)

    logger.info("\nLoading data...")
    adata = sc.read_h5ad(input_file)
    # Ensure data is fully in-memory to prevent errno 11 (file locking) issues
    # This is crucial for large files that may use HDF5 backing mode
    if hasattr(adata, 'isbacked') and adata.isbacked:
        logger.info("Converting backed AnnData to in-memory mode...")
        adata = adata.to_memory()
    logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # Validate cell-type column
    if celltype_column not in adata.obs.columns:
        raise ValueError(
            f"Cell-type column '{celltype_column}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    logger.info(
        f"Cell-type column '{celltype_column}': "
        f"{adata.obs[celltype_column].nunique()} unique types"
    )

    # Convert ENSEMBL IDs to gene symbols if needed.
    # _utils uses relative imports so it must be loaded lazily inside a function,
    # not at module level.
    if adata.var_names[0].startswith('ENSMUSG') or adata.var_names[0].startswith('ENSG'):
        logger.info("\nConverting ENSEMBL IDs to gene symbols...")
        try:
            from _utils import convert_ensembl_to_gene_symbols
            convert_ensembl_to_gene_symbols(adata, inplace=True)
        except ImportError:
            logger.warning(
                "Could not import convert_ensembl_to_gene_symbols — "
                "ENSEMBL IDs will be kept as-is."
            )
    else:
        logger.info(f"Gene names already in symbol format (first gene: {adata.var_names[0]})")

    adata.obs_names_make_unique()

    # Ensure counts layer exists before normalization
    if 'counts' not in adata.layers:
        logger.info("Creating 'counts' layer from X")
        adata.layers['counts'] = adata.X.copy()

    all_genes = adata.var_names.tolist()
    logger.info(f"\nPreprocessing with {len(all_genes)} genes...")

    adata_processed = process_data_for_panel_evaluation(
        adata=adata,
        probeset=all_genes,
        n_neighbors=n_neighbors,
        layer="counts",
        hvg=False,
        subset=False,
        scale=False,
        dataset_name="analysis_input",
        dimensionality_reduction=dimensionality_reduction,
        filter_genes=True,
    )

    # Compute extra NMF components if requested but not yet present
    if dimensionality_reduction in ["nmf", "both"]:
        if 'X_nmf' not in adata_processed.obsm:
            logger.info(f"Computing NMF with {n_nmf_components} components...")
            X = adata_processed.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            X = np.abs(X)
            nmf = sklearn_NMF(
                n_components=n_nmf_components,
                init='nndsvda',
                max_iter=1000,
                random_state=DEFAULT_RANDOM_STATE,
            )
            W = nmf.fit_transform(X)
            adata_processed.obsm['X_nmf'] = W
            adata_processed.varm['nmf_components'] = nmf.components_.T
            logger.info(f"✓ NMF complete: {W.shape}")

    _fix_index_columns(adata_processed)

    logger.info(f"\nSaving to: {output_file}")

    # Write with retry logic and atomic operation to handle errno 11 (file locking)
    # This is especially important for large files (e.g., LCA dataset at 2GB)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Write to temporary file first to avoid partial writes on failure
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=os.path.dirname(output_file),
                suffix='.h5ad'
            ) as tmp:
                tmp_path = tmp.name

            adata_processed.write_h5ad(tmp_path, compression='gzip')

            # Atomic rename - if this succeeds, we're guaranteed a complete file
            shutil.move(tmp_path, output_file)
            logger.info(f"Successfully wrote to: {output_file}")
            break

        except (OSError, BlockingIOError) as e:
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            # Retry on errno 11 (Resource temporarily unavailable - file locking)
            if (hasattr(e, 'errno') and e.errno == 11) and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Write failed with errno 11 (file locked), "
                    f"retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Write failed after {attempt + 1} attempts")
                raise

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.2f} MB")
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"  Cells:   {adata_processed.n_obs}")
    logger.info(f"  Genes:   {adata_processed.n_vars}")
    logger.info(f"  UMAP:    {'✓' if 'X_umap' in adata_processed.obsm else '✗'}")
    logger.info(f"  PCA:     {'✓' if 'X_pca'  in adata_processed.obsm else '✗'}")
    logger.info(f"  NMF:     {'✓' if 'X_nmf'  in adata_processed.obsm else '✗'}")
    logger.info("=" * 80)


def main() -> None:
    """Main function to preprocess reference dataset.

    Loads raw reference data, applies quality control and normalization,
    and computes dimensionality reduction embeddings (PCA, NMF), clustering,
    and neighborhood graphs required for evaluation.

    Delegates all processing to :func:`preprocess_reference_for_analysis_scripts`.

    Raises:
        FileNotFoundError: If input file does not exist.
    """
    setup_logging()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Preprocess reference dataset for evaluation pipeline"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to raw input h5ad file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save preprocessed reference h5ad file"
    )
    parser.add_argument(
        "--dimensionality_reduction",
        type=str,
        default="both",
        choices=["pca", "nmf", "both"],
        help="Type of dimensionality reduction: pca, nmf, or both (default: both)"
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="Number of neighbors for UMAP (default: 15)"
    )
    parser.add_argument(
        "--n_nmf_components",
        type=int,
        default=DEFAULT_N_COMPONENTS_NMF,
        help=f"Number of NMF components (default: {DEFAULT_N_COMPONENTS_NMF})"
    )

    args = parser.parse_args()

    preprocess_reference_for_analysis_scripts(
        input_file=args.input_file,
        output_file=args.output_file,
        dimensionality_reduction=args.dimensionality_reduction,
        n_neighbors=args.n_neighbors,
        n_nmf_components=args.n_nmf_components,
    )


if __name__ == "__main__":
    main()
