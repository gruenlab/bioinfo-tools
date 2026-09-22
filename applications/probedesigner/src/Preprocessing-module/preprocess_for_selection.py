"""Preprocessing for gene selection pipeline.

This module preprocesses raw data for all filter/HVG combinations, creating
preprocessed datasets with normalized expression and PCA embeddings.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData

# Module constants — package-relative when imported as ``preprocessing.…``,
# flat import when this file is run as a script from its own directory.
# --- load THIS directory's _constants.py by path (sibling dirs share the name) ---
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _cpath
_cspec = _ilu.spec_from_file_location("_constants", _cpath(__file__).resolve().parent / "_constants.py")
_sys.modules["_constants"] = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_sys.modules["_constants"])

# Canonical raw-count check from Utility-module (hard import — one implementation in the cut).
_UTILITY_DIR = _cpath(__file__).resolve().parent.parent / "Utility-module"
if str(_UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILITY_DIR))
from _validation import is_anndata_raw

# Gene blacklist filter lives in Selection-module/_filtering.py. It is imported
# lazily inside _process_single_combination (not here) because _filtering re-pins
# sys.modules["_constants"] to Selection-module's copy on import, which would
# shadow this module's own _constants during startup.
_SELECTION_DIR = _cpath(__file__).resolve().parent.parent / "Selection-module"


try:
    from ._constants import (
        DEFAULT_HVG_FLAVOR,
        DEFAULT_MIN_CELLS_PER_GENE,
        DEFAULT_MIN_GENES_PER_CELL,
        DEFAULT_N_COMPONENTS_NMF,
        DEFAULT_N_COMPONENTS_PCA,
        DEFAULT_N_HVG,
        DEFAULT_RANDOM_STATE,
        FILTER_METHODS,
    )
except ImportError:  # pragma: no cover - script / flat-import fallback
    from _constants import (
        DEFAULT_HVG_FLAVOR,
        DEFAULT_MIN_CELLS_PER_GENE,
        DEFAULT_MIN_GENES_PER_CELL,
        DEFAULT_N_COMPONENTS_NMF,
        DEFAULT_N_COMPONENTS_PCA,
        DEFAULT_N_HVG,
        DEFAULT_RANDOM_STATE,
        FILTER_METHODS,
    )

logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = ['preprocess_for_selection', 'preprocess_for_analysis']


def _is_integer_count_matrix(matrix) -> bool:
    """Return True when ``matrix`` looks like raw integer counts.

    Delegates to Utility-module's canonical ``is_anndata_raw`` (which only inspects
    ``.X``) so every raw-count check in the cut shares one implementation.
    """
    return matrix is not None and is_anndata_raw(SimpleNamespace(X=matrix))


def _copy_matrix(matrix):
    """Copy sparse or dense matrices without changing their representation."""
    return matrix.copy() if hasattr(matrix, "copy") else np.array(matrix, copy=True)


def _resolve_raw_counts(adata: AnnData) -> Tuple[object, str]:
    """Find raw integer counts aligned to the current AnnData variables."""
    if "counts" in adata.layers and _is_integer_count_matrix(adata.layers["counts"]):
        return _copy_matrix(adata.layers["counts"]), "layers['counts']"

    if adata.raw is not None and _is_integer_count_matrix(adata.raw.X):
        raw_var_names = adata.raw.var_names
        positions = raw_var_names.get_indexer(adata.var_names)
        missing = adata.var_names[positions < 0].tolist()
        if missing:
            preview = ", ".join(map(str, missing[:5]))
            raise ValueError(
                "Raw counts in adata.raw.X cannot be aligned to filtered genes; "
                f"{len(missing)} genes are missing from adata.raw.var_names "
                f"(examples: {preview})"
            )
        return _copy_matrix(adata.raw.X[:, positions]), "adata.raw.X"

    if _is_integer_count_matrix(adata.X):
        return _copy_matrix(adata.X), "adata.X"

    raise ValueError(
        "No valid raw integer counts found. Expected raw counts in "
        "adata.layers['counts'], adata.raw.X, or adata.X before normalization."
    )


def preprocess_for_analysis(
    input_file: str,
    output_file: str,
    celltype_column: str = "celltype",
    dimensionality_reduction: str = "both",
    n_neighbors: int = 15,
    n_nmf_components: int = DEFAULT_N_COMPONENTS_NMF,
) -> None:
    """Preprocess raw h5ad as canonical input for analysis scripts.

    Public API wrapper — delegates to
    :func:`preprocess_reference_for_analysis_scripts` in the Evaluation module,
    which produces an AnnData with:

    - ``layers["counts"]``  — raw integer counts (for NMF)
    - ``.X``                — log-normalised values
    - ``obsm["X_umap"]``    — UMAP embedding (for stability-analysis feature plots)
    - PCA embeddings and Leiden clusterings

    Parameters
    ----------
    input_file:
        Path to raw h5ad file.
    output_file:
        Path for the preprocessed output h5ad.
    celltype_column:
        Column in ``.obs`` with cell-type labels.
    dimensionality_reduction:
        ``"pca"``, ``"nmf"``, or ``"both"`` (default).
    n_neighbors:
        Number of neighbours for UMAP.
    n_nmf_components:
        Number of NMF components.
    """
    import sys as _sys
    _eval_dir = str(Path(__file__).resolve().parent.parent / "Evaluation-module")
    if _eval_dir not in _sys.path:
        _sys.path.insert(0, _eval_dir)
    from preprocess_reference_for_evaluation import preprocess_reference_for_analysis_scripts
    preprocess_reference_for_analysis_scripts(
        input_file=input_file,
        output_file=output_file,
        celltype_column=celltype_column,
        dimensionality_reduction=dimensionality_reduction,
        n_neighbors=n_neighbors,
        n_nmf_components=n_nmf_components,
    )

# =============================================================================
# Module Configuration
# =============================================================================

# =============================================================================
# Main Preprocessing Function
# =============================================================================

def preprocess_for_selection(
    input_file: str,
    output_dir: str,
    celltype_column: str = 'celltype',
    filter_methods: List[str] = None,
    hvg_option: str = 'both',
    n_components_pca: int = DEFAULT_N_COMPONENTS_PCA,
    n_hvg: int = DEFAULT_N_HVG,
    hvg_flavor: str = DEFAULT_HVG_FLAVOR,
    random_state: int = DEFAULT_RANDOM_STATE,
    blacklist_patterns: List[str] | None = None,
    use_default_blacklist: bool = True,
) -> None:
    """
    Preprocess data for gene selection pipeline.

    Creates preprocessed datasets for specified combinations of:
    - Filter methods (scanpy, no_filter)
    - HVG options (all_genes, hvg_subset, or both)

    Each dataset includes:
    - Filtered and normalized expression data (layers["counts"] = raw, .X = log-norm)
    - PCA embedding

    Args:
        input_file: Path to raw h5ad file
        output_dir: Base output directory for preprocessed datasets
        celltype_column: Column name for cell type annotation
        filter_methods: List of filter methods to use
        hvg_option: HVG processing - 'all_genes', 'hvg', or 'both' (default)
        n_components_pca: Number of PCA components
        n_hvg: Number of highly variable genes to select
        hvg_flavor: Scanpy HVG flavor ('seurat', 'seurat_v3', 'cell_ranger')
        random_state: Random state for reproducibility
        blacklist_patterns: Extra gene-name prefixes to remove before
            normalization (case-insensitive), combined with the default set
            unless ``use_default_blacklist`` is False.
        use_default_blacklist: Remove the default families (mt-, hsp, rps, rpl)
            before normalization. Default True.

    Examples:
        >>> preprocess_for_selection(
        ...     input_file='data/raw.h5ad',
        ...     output_dir='data/preprocessed',
        ...     celltype_column='celltype'
        ... )
    """
    if filter_methods is None:
        filter_methods = FILTER_METHODS

    # Build HVG options list based on parameter
    if hvg_option == 'both':
        hvg_options = [False, True]
    elif hvg_option == 'all_genes':
        hvg_options = [False]
    elif hvg_option == 'hvg':
        hvg_options = [True]
    else:
        raise ValueError(f"Invalid hvg_option: {hvg_option}. Must be 'all_genes', 'hvg', or 'both'")

    logger.info("=" * 80)
    logger.info("PREPROCESSING FOR GENE SELECTION")
    logger.info("=" * 80)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Cell type column: {celltype_column}")
    logger.info(f"Filter methods: {filter_methods}")
    logger.info(f"HVG option: {hvg_option}")
    logger.info(f"HVG count: {n_hvg}")
    logger.info(f"HVG flavor: {hvg_flavor}")
    logger.info(f"PCA components: {n_components_pca}")

    # Load data
    logger.info("")
    logger.info("Loading data...")
    adata = sc.read_h5ad(input_file)
    logger.info(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Validate celltype column
    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Cell type column '{celltype_column}' not found in adata.obs")

    # Process all combinations
    total_combinations = len(filter_methods) * len(hvg_options)
    current = 0

    for filter_method in filter_methods:
        for subset_to_hvg in hvg_options:
            current += 1
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"COMBINATION {current}/{total_combinations}: "
                       f"{filter_method}, HVG={subset_to_hvg}")
            logger.info("=" * 80)
            
            # Process single combination
            try:
                _process_single_combination(
                    adata=adata.copy(),
                    output_dir=output_dir,
                    celltype_column=celltype_column,
                    filter_method=filter_method,
                    subset_to_hvg=subset_to_hvg,
                    n_components_pca=n_components_pca,
                    n_hvg=n_hvg,
                    hvg_flavor=hvg_flavor,
                    random_state=random_state,
                    blacklist_patterns=blacklist_patterns,
                    use_default_blacklist=use_default_blacklist,
                )
            except Exception as e:
                logger.error(f"Failed to process combination: {e}")
                raise
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Processed {total_combinations} combinations")
    logger.info(f"Results saved to: {output_dir}")


def _process_single_combination(
    adata: AnnData,
    output_dir: str,
    celltype_column: str,
    filter_method: str,
    subset_to_hvg: bool,
    n_components_pca: int,
    n_hvg: int,
    hvg_flavor: str,
    random_state: int,
    blacklist_patterns: List[str] | None = None,
    use_default_blacklist: bool = True,
) -> None:
    """Process single filter/HVG combination."""
    
    # Generate output directory name
    filter_name = "Scanpy-Filter" if filter_method == "scanpy" else "No-Filter"
    hvg_name = "HVG" if subset_to_hvg else "All-Genes"
    comb_name = f"{filter_name}_{hvg_name}"
    comb_dir = os.path.join(output_dir, comb_name)
    
    os.makedirs(comb_dir, exist_ok=True)
    
    logger.info(f"Processing: {comb_name}")
    logger.info(f"Output directory: {comb_dir}")
    
    # Step 1: Apply filtering
    logger.info("")
    logger.info("Step 1: Filtering")
    adata = _apply_filtering(adata, filter_method)
    logger.info(f"After filtering: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Step 1b: Gene blacklist filter — BEFORE normalization, so blacklisted
    # families (mt-, hsp, rps, rpl by default) never contribute to
    # normalize_total's per-cell size factors.
    n_blacklist_removed = 0
    if blacklist_patterns or use_default_blacklist:
        logger.info("")
        logger.info("Step 1b: Blacklist filtering (pre-normalization)")
        if str(_SELECTION_DIR) not in sys.path:
            sys.path.insert(0, str(_SELECTION_DIR))
        from _filtering import apply_blacklist_filter  # noqa: E402  (lazy: see import note)
        kept_genes, removed_genes, _ = apply_blacklist_filter(
            gene_list=adata.var_names.tolist(),
            blacklist_patterns=blacklist_patterns,
            use_default_blacklist=use_default_blacklist,
        )
        n_blacklist_removed = len(removed_genes)
        if n_blacklist_removed:
            adata = adata[:, kept_genes].copy()
        _combined = (['mt-', 'hsp', 'rps', 'rpl'] if use_default_blacklist else []) + (blacklist_patterns or [])
        logger.info(
            f"Blacklist patterns {_combined}: {adata.shape[1]} genes "
            f"({n_blacklist_removed} removed)"
        )

    # Step 2: Normalize and log-transform
    logger.info("")
    logger.info("Step 2: Normalization and log-transformation")
    raw_counts, raw_source = _resolve_raw_counts(adata)
    logger.info(f"Using {raw_source} as raw counts for layers['counts']")
    adata.layers["counts"] = raw_counts
    adata.X = _copy_matrix(raw_counts)
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata)  # scanpy default target_sum = per-cell median, matches Evaluation-module's convention
    sc.pp.log1p(adata)

    # Step 3: HVG selection
    if subset_to_hvg:
        logger.info("")
        logger.info("Step 3: Selecting highly variable genes")
        sc.pp.highly_variable_genes(adata, flavor=hvg_flavor, n_top_genes=n_hvg)
        adata = adata[:, adata.var['highly_variable']].copy()
        logger.info(f"After HVG selection: {adata.shape[1]} genes")

    # Step 4: PCA
    logger.info("")
    logger.info(f"Step 4: Running PCA ({n_components_pca} components)")
    sc.pp.pca(adata, n_comps=n_components_pca, random_state=random_state)

    # Step 5: Save preprocessed data
    logger.info("")
    logger.info("Step 5: Saving preprocessed data")
    adata_output = os.path.join(comb_dir, 'preprocessed.h5ad')
    adata.write(adata_output)
    logger.info(f"Saved preprocessed data: {adata_output}")

    # Step 6: Save metadata
    metadata = {
        'filter_method': filter_method,
        'subset_to_hvg': subset_to_hvg,
        'n_cells': adata.shape[0],
        'n_genes': adata.shape[1],
        'n_components_pca': n_components_pca,
        'n_hvg': n_hvg,
        'hvg_flavor': hvg_flavor,
        'celltype_column': celltype_column,
        'blacklist_pre_normalization': bool(blacklist_patterns or use_default_blacklist),
        'blacklist_use_default': use_default_blacklist,
        'blacklist_extra_patterns': blacklist_patterns or [],
        'blacklist_genes_removed': n_blacklist_removed,
        'timestamp': datetime.now().isoformat()
    }
    metadata_output = os.path.join(comb_dir, 'metadata.json')
    with open(metadata_output, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_output}")
    
    logger.info(f"✓ Completed: {comb_name}")


def _apply_filtering(adata: AnnData, filter_method: str) -> AnnData:
    """Apply filtering based on method."""

    if filter_method == "scanpy":
        logger.info(
            f"Applying Scanpy filter (min_genes={DEFAULT_MIN_GENES_PER_CELL}, "
            f"min_cells={DEFAULT_MIN_CELLS_PER_GENE})"
        )

        # Filter cells and genes
        sc.pp.filter_cells(adata, min_genes=DEFAULT_MIN_GENES_PER_CELL)
        sc.pp.filter_genes(adata, min_cells=DEFAULT_MIN_CELLS_PER_GENE)
        
    elif filter_method == "no_filter":
        logger.info("No filtering applied")
    
    else:
        raise ValueError(f"Unknown filter method: {filter_method}")
    
    return adata


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess data for gene selection pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--input_file', type=str, required=True,
        help='Path to raw h5ad file'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Base output directory for preprocessed datasets'
    )
    
    # Optional arguments
    parser.add_argument(
        '--celltype_column', type=str, default='celltype',
        help='Column name for cell type annotation (default: celltype)'
    )
    parser.add_argument(
        '--filter_methods', type=str, nargs='+',
        default=['scanpy', 'no_filter'],
        choices=['scanpy', 'no_filter'],
        help='Filter methods to use (default: both)'
    )
    parser.add_argument(
        '--hvg_option', type=str, default='both',
        choices=['all_genes', 'hvg', 'both'],
        help=(
            'HVG processing option: '
            '"all_genes" (no HVG subsetting), '
            '"hvg" (subset to highly variable genes only), '
            '"both" (write both variants; default)'
        )
    )
    parser.add_argument(
        '--n_components_pca', type=int, default=DEFAULT_N_COMPONENTS_PCA,
        help=f'Number of PCA components (default: {DEFAULT_N_COMPONENTS_PCA})'
    )
    parser.add_argument(
        '--n_hvg', type=int, default=DEFAULT_N_HVG,
        help=f'Number of highly variable genes (default: {DEFAULT_N_HVG})'
    )
    parser.add_argument(
        '--hvg_flavor', type=str, default=DEFAULT_HVG_FLAVOR,
        choices=['seurat', 'seurat_v3', 'cell_ranger'],
        help=f'Scanpy HVG flavor (default: {DEFAULT_HVG_FLAVOR})'
    )
    parser.add_argument(
        '--random_state', type=int, default=DEFAULT_RANDOM_STATE,
        help=f'Random state for reproducibility (default: {DEFAULT_RANDOM_STATE})'
    )
    parser.add_argument(
        '--blacklist_patterns', type=str, nargs='*', default=None,
        help=(
            'Extra gene-name prefixes to remove BEFORE normalization '
            '(case-insensitive). Combined with the default set unless '
            '--disable_default_blacklist is given.'
        )
    )
    parser.add_argument(
        '--disable_default_blacklist', action='store_true',
        help=(
            'Do not remove the default blacklist families (mt-, hsp, rps, rpl) '
            'before normalization. By default they ARE removed pre-normalization.'
        )
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def setup_logging(log_level: str = 'INFO') -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    
    setup_logging(args.log_level)
    
    try:
        preprocess_for_selection(
            input_file=args.input_file,
            output_dir=args.output_dir,
            celltype_column=args.celltype_column,
            filter_methods=args.filter_methods,
            hvg_option=args.hvg_option,
            n_components_pca=args.n_components_pca,
            n_hvg=args.n_hvg,
            hvg_flavor=args.hvg_flavor,
            random_state=args.random_state,
            blacklist_patterns=args.blacklist_patterns,
            use_default_blacklist=not args.disable_default_blacklist,
        )
        return 0
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
