"""
Utility functions for data preparation and dimensionality reduction.

This module contains utility functions for data processing, normalization,
and performing dimensionality reduction (NMF, PCA, and Consensus NMF)
analyses on AnnData objects.
"""

from __future__ import annotations

import logging
import math
import gc
import os
import numpy as np
import pandas as pd
import scipy.sparse
import scanpy as sc
from sklearn.decomposition import NMF, PCA
import matplotlib.pyplot as plt
from anndata import AnnData

from ._constants import NORMALIZE_TARGET_SUM, MIN_CELLS_FOR_DIMRED
from ._validation import is_anndata_raw, is_anndata_raw_layer
from ._consensus_nmf import ConsensusNmf, select_optimal_k

__all__ = [
    'convert_ensembl_to_gene_symbols',
    'subset_data_to_gene_lists',
    'perform_nmf_pca_per_celltype',
    'perform_consensus_nmf_per_celltype',
]

logger = logging.getLogger(__name__)


def convert_ensembl_to_gene_symbols(
        adata: AnnData,
        species: str = 'mouse',
        inplace: bool = True,
        handle_duplicates: str = 'keep_first',
        keep_ensembl_if_no_symbol: bool = True,
        batch_size: int = 1000) -> AnnData | None:
    """
    Convert ENSEMBL IDs to gene symbols in an AnnData object using MyGene.info.

    This function queries the MyGene.info database to convert ENSEMBL gene IDs
    to official gene symbols. It handles batch queries for efficiency,
    multiple genes with the same symbol (duplicates), and genes without
    a symbol annotation.

    Args:
        adata: AnnData object with ENSEMBL IDs as var_names.
        species: Species for query: 'mouse', 'human', or a taxonomy ID
            (default: 'mouse').
        inplace: If True, modify adata in place. If False, return a copy
            (default: True).
        handle_duplicates: How to handle duplicate gene symbols:
            - 'keep_first': Keep first occurrence, append ENSEMBL suffix to others
            - 'append_ensembl': Append ENSEMBL ID suffix to all duplicates
            - 'make_unique': Use pandas make_unique to add numeric suffixes
            (default: 'keep_first').
        keep_ensembl_if_no_symbol: If True, keep ENSEMBL ID when symbol
            lookup fails. If False, genes without symbols will be removed
            (default: True).
        batch_size: Number of genes to query per batch. MyGene.info limit
            is 1000 (default: 1000).

    Returns:
        Modified AnnData object (if inplace=False), otherwise None.

    Raises:
        ImportError: If mygene package is not installed.
        ValueError: If invalid handle_duplicates option provided.
    """
    try:
        import mygene
    except ImportError:
        raise ImportError("mygene package is required for ENSEMBL to gene symbol conversion. "
                         "Install it with: pip install mygene")

    # Work on copy if not inplace
    if not inplace:
        adata = adata.copy()

    logger.info("\n=== Converting ENSEMBL IDs to Gene Symbols (MyGene.info) ===")
    logger.info(f"Original var_names (first 5): {adata.var_names[:5].tolist()}")
    logger.info(f"Total genes: {adata.n_vars}")
    logger.info(f"Species: {species}")

    # Store original ENSEMBL IDs
    original_ensembl = adata.var_names.copy()
    adata.var['ENSEMBL'] = original_ensembl

    # Initialize MyGene
    mg = mygene.MyGeneInfo()

    # Query in batches
    ensembl_ids = adata.var_names.tolist()
    n_batches = math.ceil(len(ensembl_ids) / batch_size)

    logger.info(f"Querying MyGene.info in {n_batches} batches of {batch_size} genes...")

    all_results = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(ensembl_ids))
        batch = ensembl_ids[start_idx:end_idx]

        logger.info(f"  Batch {i+1}/{n_batches}: querying {len(batch)} genes...")

        # Query MyGene - specify fields to get symbol and name
        results = mg.querymany(
            batch,
            scopes='ensembl.gene',
            fields='symbol,name',
            species=species,
            returnall=True
        )

        all_results.extend(results['out'])

    logger.info(f"Retrieved {len(all_results)} results for {len(ensembl_ids)} genes")

    # Build symbol mapping - use 'query' field from results to match back to original IDs
    symbols = pd.Series(index=adata.var_names, dtype=str)
    gene_names = pd.Series(index=adata.var_names, dtype=str)
    n_notfound = 0
    processed_ids = set()

    for result in all_results:
        # Get the original query ID from the result
        ensembl_id = result.get('query', None)

        if ensembl_id is None:
            logger.warning(f"  Result missing 'query' field, skipping: {result}")
            continue

        # Skip if we already processed this ID (happens with duplicate hits)
        if ensembl_id in processed_ids:
            continue
        processed_ids.add(ensembl_id)

        if 'symbol' in result:
            symbols[ensembl_id] = result['symbol']
            gene_names[ensembl_id] = result.get('name', '')
        elif 'notfound' in result and result['notfound']:
            n_notfound += 1
            if keep_ensembl_if_no_symbol:
                symbols[ensembl_id] = ensembl_id
                gene_names[ensembl_id] = ''
            else:
                symbols[ensembl_id] = np.nan
                gene_names[ensembl_id] = ''
        else:
            # No symbol returned but not explicitly notfound
            logger.warning(f"  Gene {ensembl_id}: No symbol in result")
            if keep_ensembl_if_no_symbol:
                symbols[ensembl_id] = ensembl_id
                gene_names[ensembl_id] = ''
            else:
                symbols[ensembl_id] = np.nan
                gene_names[ensembl_id] = ''

    # Store full gene names in var
    adata.var['name'] = gene_names
    adata.var['symbol'] = symbols.copy()

    if n_notfound > 0:
        logger.warning(f"Found {n_notfound} genes without gene symbols in MyGene.info")
        if keep_ensembl_if_no_symbol:
            logger.info("  → Keeping ENSEMBL IDs for genes without symbols")
        else:
            logger.warning(f"  → Removing {n_notfound} genes without symbols")
            # Filter out genes without symbols
            keep_genes = ~symbols.isna()
            adata._inplace_subset_var(keep_genes)
            symbols = symbols[keep_genes]
            original_ensembl = original_ensembl[keep_genes]

    # Check for duplicate symbols
    duplicated = symbols.duplicated(keep=False)
    n_duplicated = duplicated.sum()

    if n_duplicated > 0:
        logger.warning(f"Found {n_duplicated} genes with duplicate symbols")

        if handle_duplicates == 'keep_first':
            logger.info("  → Keeping first occurrence, appending ENSEMBL suffix to duplicates")
            dup_mask = symbols.duplicated(keep='first')
            symbols.loc[dup_mask] = symbols.loc[dup_mask] + '_' + original_ensembl[dup_mask]

        elif handle_duplicates == 'append_ensembl':
            logger.info("  → Appending ENSEMBL suffix to all duplicate symbols")
            symbols.loc[duplicated] = symbols.loc[duplicated] + '_' + original_ensembl[duplicated]

        elif handle_duplicates == 'make_unique':
            logger.info("  → Using pandas make_unique to add numeric suffixes")
            symbols = pd.Series(pd.io.parsers.ParserBase({'names': symbols})._maybe_dedup_names(symbols),
                               index=symbols.index)
        else:
            raise ValueError(f"Invalid handle_duplicates option: {handle_duplicates}. "
                           f"Choose from: 'keep_first', 'append_ensembl', 'make_unique'")

    # Update var_names
    adata.var_names = symbols
    adata.var_names.name = 'gene_symbol'

    # Verify uniqueness
    if not adata.var_names.is_unique:
        n_still_dup = adata.var_names.duplicated().sum()
        logger.error(f"ERROR: var_names still has {n_still_dup} duplicates after conversion!")
        logger.error(f"Duplicated gene symbols: {adata.var_names[adata.var_names.duplicated()].tolist()[:10]}")
        raise ValueError("Gene symbol conversion resulted in non-unique var_names")

    # Log success statistics
    n_converted = len(adata.var_names) - n_notfound
    logger.info(f"\nSuccessfully converted {n_converted} ENSEMBL IDs to gene symbols")
    logger.info(f"  Final gene count: {adata.n_vars}")
    logger.info(f"  New var_names (first 5): {adata.var_names[:5].tolist()}")
    logger.info(f"  Duplicate handling: {handle_duplicates}")
    if n_notfound > 0:
        if keep_ensembl_if_no_symbol:
            logger.info(f"  Genes with ENSEMBL IDs (no symbol found): {n_notfound}")
        else:
            logger.info(f"  Genes removed (no symbol found): {n_notfound}")
    if n_duplicated > 0:
        logger.info(f"  Duplicate symbols resolved: {n_duplicated}")

    if not inplace:
        return adata


def subset_data_to_gene_lists(
        ref_data: AnnData,
        gene_lists: dict,
        benchmark_genesets: dict,
        celltype_col: str,
        logger) -> dict:
    """
    Prepare datasets for evaluation by subsetting to gene lists.

    Args:
        ref_data: Reference dataset with all genes.
        gene_lists: Dictionary with gene lists from DT and XGB.
        benchmark_genesets: Dictionary with gene lists from benchmark methods.
        celltype_col: Column name for cell type annotations.
        logger: Logger instance.

    Returns:
        Dictionary with processed datasets.

    Raises:
        ValueError: If celltype column not found.
    """
    # Initialize the dictionary for processed datasets
    datasets = {}

    # Store the reference dataset
    datasets['reference'] = ref_data
    datasets['reference'].obs_names_make_unique()

    # Check required columns
    if celltype_col not in ref_data.obs.columns:
        logger.error(f"Cell type column '{celltype_col}' not found in dataset")
        available_cols = ', '.join(ref_data.obs.columns)
        logger.info(f"Available columns: {available_cols}")
        raise ValueError(f"Cell type column '{celltype_col}' not found in dataset")

    logger.info(f"Preparing datasets using cell type column: {celltype_col}")
    logger.info(f"Reference dataset shape: {ref_data.shape}")

    # Check if data needs log normalization - we want to use normalized data
    is_raw = is_anndata_raw(ref_data)

    if is_raw:
        logger.info("Reference dataset appears to contain raw counts. Applying log normalization...")
        # Store raw counts in counts layer before normalization
        if 'counts' not in ref_data.layers:
            ref_data.layers['counts'] = ref_data.X.copy()
            logger.info("Stored raw counts in 'counts' layer")

        # Apply log normalization (normalize_total + log1p)
        sc.pp.normalize_total(ref_data, target_sum=NORMALIZE_TARGET_SUM)
        sc.pp.log1p(ref_data)
        logger.info("Applied log normalization to reference dataset")
    else:
        logger.info("Reference dataset appears to be already normalized")
        # Check for raw counts in layers
        if 'counts' in ref_data.layers:
            logger.info("Found 'counts' layer with raw data")
        else:
            logger.warning("No 'counts' layer found but data appears processed")

    # Process gene lists to create subset datasets (no h5ad saving)
    for name, genes in gene_lists.items():
        logger.info(f"Processing {name} with {len(genes)} genes")
        # Find intersection with genes in the reference dataset
        available_genes = list(set(genes).intersection(set(ref_data.var_names)))
        if not available_genes:
            logger.warning(f"No genes from {name} found in the reference dataset")
            continue
        logger.info(f"{len(available_genes)}/{len(genes)} genes found in reference dataset")
        # Create a subset dataset
        if len(available_genes) > 0:
            try:
                subset = ref_data[:, available_genes].copy()
                subset.obs_names_make_unique()
                datasets[name] = subset
                logger.info(f"Created {name} dataset with {len(available_genes)} genes")
            except Exception as e:
                logger.error(f"Failed to create subset for {name}: {str(e)}")

    # Process benchmark gene sets (do not save h5ad files)
    for name, genes in benchmark_genesets.items():
        logger.info(f"Processing benchmark {name} with {len(genes)} genes")
        # Find intersection with genes in the reference dataset
        available_genes = list(set(genes).intersection(set(ref_data.var_names)))
        if not available_genes:
            logger.warning(f"No genes from benchmark {name} found in the reference dataset")
            continue
        logger.info(f"{len(available_genes)}/{len(genes)} genes from benchmark {name} found in reference dataset")
        # Create a subset dataset (do not save h5ad)
        if len(available_genes) > 0:
            try:
                subset = ref_data[:, available_genes].copy()
                subset.obs_names_make_unique()
                datasets[name] = subset
            except Exception as e:
                logger.error(f"Failed to create subset for benchmark {name}: {str(e)}")

    return datasets


def perform_nmf_pca_per_celltype(
        adata: AnnData,
        groupby: str = 'original',
        method: str = 'nmf',
        n_components: int = 5,
        random_state: int = 42,
        results_dir: str | None = None) -> dict:
    """
    Perform NMF or PCA analysis for each cell type separately.

    Args:
        adata: Annotated data matrix.
        groupby: Column in adata.obs for grouping by cell type.
        method: Method to use: 'nmf', 'pca', or 'both' (default: 'nmf').
        n_components: Number of components/factors for dimensionality reduction.
        random_state: Random state for reproducibility.
        results_dir: Path to save results. If None, results are not saved to disk.

    Returns:
        Dictionary with cell type names as keys and results as values.
        Each result contains:
        - 'method': 'nmf' or 'pca'
        - 'loadings_df': DataFrame with gene loadings (genes × components)
        - 'explained_variance': variance explained
        - 'n_cells': number of cells in this cell type
        - 'n_genes': number of genes used

    Raises:
        ValueError: If method or groupby column invalid.
    """
    if method not in ['nmf', 'pca', 'both']:
        raise ValueError(f"method must be 'nmf', 'pca', or 'both', got {method}")

    logger.info(f"Performing per-celltype {method.upper()} analysis (n_components={n_components})")
    if results_dir:
        logger.info(f"Results will be saved to: {results_dir}")

    # Validate groupby column
    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    # Ensure the groupby column is categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        logger.info(f"Converting '{groupby}' column to categorical dtype")
        adata.obs[groupby] = adata.obs[groupby].astype('category')

    # Determine which cell types have enough cells
    valid_celltypes = []
    for celltype in adata.obs[groupby].cat.categories:
        subset = adata[adata.obs[groupby] == celltype]
        if subset.shape[0] >= MIN_CELLS_FOR_DIMRED and subset.shape[0] >= n_components * 2:
            valid_celltypes.append(celltype)
        else:
            logger.warning(f"Cell type '{celltype}' skipped: need ≥{MIN_CELLS_FOR_DIMRED} cells and ≥{n_components * 2} (has {subset.shape[0]})")

    num_valid_celltypes = len(valid_celltypes)
    if num_valid_celltypes == 0:
        logger.error("No cell types have enough cells for analysis!")
        return {}

    logger.info(f"Valid cell types for analysis: {num_valid_celltypes}/{len(adata.obs[groupby].cat.categories)}")

    # Initialize results dictionary
    celltype_results = {}

    # Process each valid cell type
    for celltype in valid_celltypes:
        logger.info(f"Processing cell type: {celltype}")

        # Subset data for this cell type
        subset = adata[adata.obs[groupby] == celltype].copy()
        n_cells = subset.shape[0]

        # Perform NMF if requested
        if method in ['nmf', 'both']:
            try:
                logger.info(f"Running NMF for {celltype}")

                # Check for counts layer
                if 'counts' not in subset.layers:
                    logger.error(f"No 'counts' layer found for {celltype}! Skipping NMF.")
                    continue

                # Verify counts layer is raw
                counts_is_raw = is_anndata_raw_layer(subset, 'counts')
                if not counts_is_raw:
                    logger.error(f"Counts layer is not raw for {celltype}! Skipping NMF.")
                    continue

                # Extract raw counts
                raw_counts = subset.layers['counts']

                # Calculate standard deviation and filter genes
                if scipy.sparse.issparse(raw_counts):
                    raw_counts_dense = raw_counts.toarray()
                    std = np.std(raw_counts_dense, axis=0)
                else:
                    std = np.std(raw_counts, axis=0)

                ind = np.where(std > 0)[0].astype(int)
                n_features = len(ind)

                if n_features < n_components:
                    logger.warning(f"Cell type {celltype} has too few features ({n_features}), skipping NMF")
                    continue

                # Filter to non-zero std genes
                raw_counts_filtered = raw_counts[:, ind]
                filtered_gene_names = subset.var_names[ind]

                # Run NMF
                logger.info(f"Running NMF for {celltype} with {n_features} genes and {n_cells} cells")
                nmf_model = NMF(
                    n_components=n_components,
                    init="nndsvda",
                    random_state=random_state,
                    beta_loss="kullback-leibler",
                    solver="mu",
                    max_iter=1000,
                    alpha_W=0.0,
                    alpha_H=0.0,
                    l1_ratio=0
                )

                W = nmf_model.fit_transform(raw_counts_filtered)
                H = nmf_model.components_

                # Create loadings DataFrame
                nmf_df = pd.DataFrame(
                    H.T,
                    index=filtered_gene_names,
                    columns=[f'NMF_{i+1}' for i in range(n_components)]
                )

                # Store results
                key = f"{celltype}_nmf" if method == 'both' else celltype
                celltype_results[key] = {
                    'method': 'nmf',
                    'loadings_df': nmf_df,
                    'n_cells': n_cells,
                    'n_genes': n_features,
                    'model': nmf_model
                }

                logger.info(f"NMF completed for {celltype}")

                # Save results if directory provided
                if results_dir:
                    os.makedirs(results_dir, exist_ok=True)
                    safe_celltype = celltype.replace(' ', '_').replace('/', '_')
                    nmf_df.to_csv(os.path.join(results_dir, f"nmf_loadings_{safe_celltype}.csv"))

                # Clean up
                del W, H, raw_counts_filtered, nmf_model
                gc.collect()

            except Exception as e:
                logger.error(f"NMF failed for cell type {celltype}: {str(e)}")
                continue

        # Perform PCA if requested
        if method in ['pca', 'both']:
            try:
                logger.info(f"Running PCA for {celltype}")

                # Check if data is log-transformed
                is_log = not is_anndata_raw(subset)

                if not is_log:
                    # Need to log-transform first
                    logger.info(f"Log-transforming data for PCA on {celltype}")
                    if 'counts' not in subset.layers:
                        subset.layers['counts'] = subset.X.copy()
                    sc.pp.normalize_total(subset, target_sum=NORMALIZE_TARGET_SUM)
                    sc.pp.log1p(subset)

                # Filter genes with zero variance
                if scipy.sparse.issparse(subset.X):
                    X_dense = subset.X.toarray()
                else:
                    X_dense = subset.X

                gene_var = np.var(X_dense, axis=0)
                ind = np.where(gene_var > 0)[0].astype(int)
                n_features = len(ind)

                if n_features < n_components:
                    logger.warning(f"Cell type {celltype} has too few features ({n_features}), skipping PCA")
                    continue

                # Filter to non-zero variance genes
                X_filtered = X_dense[:, ind]
                filtered_gene_names = subset.var_names[ind]

                # Run PCA
                logger.info(f"Running PCA for {celltype} with {n_features} genes and {n_cells} cells")
                pca_model = PCA(n_components=n_components, random_state=random_state)
                pca_model.fit(X_filtered)

                # Get loadings (components)
                loadings = pca_model.components_.T

                # Create loadings DataFrame
                pca_df = pd.DataFrame(
                    loadings,
                    index=filtered_gene_names,
                    columns=[f'PC_{i+1}' for i in range(n_components)]
                )

                # Store results
                key = f"{celltype}_pca" if method == 'both' else celltype
                celltype_results[key] = {
                    'method': 'pca',
                    'loadings_df': pca_df,
                    'n_cells': n_cells,
                    'n_genes': n_features,
                    'model': pca_model
                }

                logger.info(f"PCA completed for {celltype}")

                # Save results if directory provided
                if results_dir:
                    os.makedirs(results_dir, exist_ok=True)
                    safe_celltype = celltype.replace(' ', '_').replace('/', '_')
                    pca_df.to_csv(os.path.join(results_dir, f"pca_loadings_{safe_celltype}.csv"))

                # Clean up
                del X_filtered, pca_model
                gc.collect()

            except Exception as e:
                logger.error(f"PCA failed for cell type {celltype}: {str(e)}")
                continue

    logger.info(f"Per-celltype {method.upper()} analysis complete. Processed {len(celltype_results)} cell types.")

    return celltype_results


def perform_consensus_nmf_per_celltype(
        adata: AnnData,
        groupby: str = 'original',
        k_values: list = None,
        n_iter: int = 100,
        random_state: int = 42,
        density_threshold: float = 0.5,
        auto_select_k: bool = True,
        k_selection_method: str = 'elbow',
        results_dir: str | None = None) -> dict:
    """
    Perform Consensus NMF analysis for each cell type separately.

    This function runs multiple NMF iterations for each cell type and builds
    consensus gene expression programs through clustering, providing more stable
    and reproducible results than single-run NMF.

    Args:
        adata: Annotated data matrix.
        groupby: Column in adata.obs for grouping by cell type (default: 'original').
        k_values: List of K values (number of components) to test
            (default: [3, 5, 7, 9]).
        n_iter: Number of NMF iterations per K value (default: 100).
        random_state: Random state for reproducibility (default: 42).
        density_threshold: Threshold for filtering outlier spectra in consensus
            (default: 0.5). Lower values = more strict filtering.
        auto_select_k: If True, automatically select optimal K for each cell type
            (default: True).
        k_selection_method: Method for automatic K selection: 'elbow' or
            'silhouette' (default: 'elbow').
        results_dir: Path to save results. If None, results are not saved to disk.

    Returns:
        Dictionary with cell type names as keys and results as values.
        Each result contains:
        - 'method': 'consensus_nmf'
        - 'loadings_df': DataFrame with consensus gene loadings
        - 'cnmf_object': The ConsensusNmf object with all results
        - 'selected_k': The K value used
        - 'k_metrics': DataFrame with metrics for all tested K values
        - 'n_cells': number of cells in this cell type
        - 'n_genes': number of genes used
    """
    if k_values is None:
        k_values = [3, 5, 7, 9]

    logger.info(f"=== Performing Consensus NMF Per-Celltype ===")
    logger.info(f"K values to test: {k_values}")
    logger.info(f"Iterations per K: {n_iter}")
    logger.info(f"Auto-select K: {auto_select_k} (method: {k_selection_method})")

    if results_dir:
        logger.info(f"Results will be saved to: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)

    # Validate groupby column
    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    # Ensure the groupby column is categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        logger.info(f"Converting '{groupby}' column to categorical dtype")
        adata.obs[groupby] = adata.obs[groupby].astype('category')

    # Determine which cell types have enough cells
    max_k = max(k_values) if isinstance(k_values, list) else k_values
    valid_celltypes = []
    for celltype in adata.obs[groupby].cat.categories:
        subset = adata[adata.obs[groupby] == celltype]
        if subset.shape[0] >= MIN_CELLS_FOR_DIMRED and subset.shape[0] >= max_k * 2:
            valid_celltypes.append(celltype)
        else:
            logger.warning(f"Cell type '{celltype}' skipped: need ≥{MIN_CELLS_FOR_DIMRED} cells and ≥{max_k * 2} (has {subset.shape[0]})")

    num_valid_celltypes = len(valid_celltypes)
    if num_valid_celltypes == 0:
        logger.error("No cell types have enough cells for analysis!")
        return {}

    logger.info(f"Valid cell types for analysis: {num_valid_celltypes}/{len(adata.obs[groupby].cat.categories)}")

    # Initialize results dictionary
    celltype_results = {}

    # Process each valid cell type
    for celltype in valid_celltypes:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing cell type: {celltype}")
        logger.info(f"{'='*70}")

        # Subset data for this cell type
        subset = adata[adata.obs[groupby] == celltype].copy()
        n_cells = subset.shape[0]

        try:
            # Check for counts layer
            if 'counts' not in subset.layers:
                logger.error(f"No 'counts' layer found for {celltype}! Skipping.")
                continue

            # Verify counts layer is raw
            counts_is_raw = is_anndata_raw_layer(subset, 'counts')
            if not counts_is_raw:
                logger.error(f"Counts layer is not raw for {celltype}! Skipping.")
                continue

            # Extract raw counts and gene names
            raw_counts = subset.layers['counts']
            gene_names = subset.var_names.to_numpy()

            # Initialize ConsensusNmf with standard NMF parameters
            cnmf = ConsensusNmf(
                k_values=k_values,
                n_iter=n_iter,
                random_state=random_state,
                nmf_init="nndsvda",
                beta_loss="kullback-leibler",
                solver="mu",
                max_iter=1000,
                alpha_W=0.0,
                alpha_H=0.0,
                l1_ratio=0
            )

            # Run factorization (multiple NMF iterations)
            logger.info(f"Running {n_iter} NMF iterations for {celltype}...")
            cnmf.factorize(raw_counts, gene_names=gene_names)

            # Run consensus for each K
            for k in cnmf.k_values:
                if k in cnmf.results:
                    logger.info(f"Computing consensus for K={k}...")
                    cnmf.consensus(k, density_threshold=density_threshold)

            # Get K-selection metrics
            k_metrics = cnmf.k_selection_metrics()

            # Select optimal K
            if auto_select_k:
                selected_k = select_optimal_k(cnmf, method=k_selection_method)
                logger.info(f"Auto-selected optimal K={selected_k} using {k_selection_method} method")
            else:
                selected_k = k_values[0] if isinstance(k_values, list) else k_values
                logger.info(f"Using K={selected_k} (first in list)")

            # Get consensus results for selected K
            if selected_k not in cnmf.results or 'consensus' not in cnmf.results[selected_k]:
                logger.error(f"Consensus not available for K={selected_k}, using first available K")
                # Find first K with consensus
                for k in cnmf.k_values:
                    if k in cnmf.results and 'consensus' in cnmf.results[k]:
                        selected_k = k
                        break

            consensus = cnmf.results[selected_k]['consensus']
            consensus_H = consensus['consensus_H']
            consensus_gene_names = cnmf.results[selected_k]['gene_names']

            # Create loadings DataFrame (genes × factors)
            loadings_df = pd.DataFrame(
                consensus_H.T,
                index=consensus_gene_names,
                columns=[f'NMF_{i+1}' for i in range(selected_k)]
            )

            # Store results
            celltype_results[celltype] = {
                'method': 'consensus_nmf',
                'loadings_df': loadings_df,
                'cnmf_object': cnmf,
                'selected_k': selected_k,
                'k_metrics': k_metrics,
                'consensus_results': consensus,
                'n_cells': n_cells,
                'n_genes': len(consensus_gene_names)
            }

            logger.info(f"Consensus NMF completed for {celltype} (K={selected_k})")
            logger.info(f"  Silhouette score: {consensus.get('silhouette_score', 'N/A')}")
            logger.info(f"  Spectra filtered: {consensus.get('n_filtered', 0)}/{consensus.get('n_filtered', 0) + consensus.get('n_removed', 0)}")

            # Save results if directory provided
            if results_dir:
                safe_celltype = celltype.replace(' ', '_').replace('/', '_')

                # Save consensus loadings
                loadings_path = os.path.join(results_dir, f"consensus_nmf_loadings_{safe_celltype}_K{selected_k}.csv")
                loadings_df.to_csv(loadings_path)
                logger.info(f"  Saved loadings to {loadings_path}")

                # Save K-selection plot
                plot_path = os.path.join(results_dir, f"k_selection_{safe_celltype}.png")
                cnmf.plot_k_selection(save_path=plot_path)
                logger.info(f"  Saved K-selection plot to {plot_path}")

                # Save K metrics
                metrics_path = os.path.join(results_dir, f"k_metrics_{safe_celltype}.csv")
                k_metrics.to_csv(metrics_path, index=False)
                logger.info(f"  Saved K metrics to {metrics_path}")

            # Clean up
            del raw_counts, subset
            gc.collect()

        except Exception as e:
            logger.error(f"Consensus NMF failed for cell type {celltype}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"\n{'='*70}")
    logger.info(f"Consensus NMF per-celltype complete. Processed {len(celltype_results)} cell types.")
    logger.info(f"{'='*70}\n")

    return celltype_results
