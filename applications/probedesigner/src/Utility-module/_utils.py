"""
Utility functions for data preparation.

This module contains utility functions for converting ENSEMBL gene IDs to
gene symbols via MyGene.info.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from anndata import AnnData

__all__ = [
    'convert_ensembl_to_gene_symbols',
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
