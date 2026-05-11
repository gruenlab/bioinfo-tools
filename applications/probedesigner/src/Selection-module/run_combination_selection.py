"""
Combination gene selection strategies (rf_nmf, rf_pca).

This module orchestrates combination strategies that merge genes from two sources:
1. Random Forest on DEG-filtered genes (rf_deg)
2. Dimensionality reduction (NMF or PCA) - global or per-celltype

The workflow follows the pattern from _selection.py:
- Run both components WITH full filtering (Xenium + ODT)
- Combine filtered results with ratio-based selection
- Handle duplicates by assigning to RF pool and replacing from dimred pool
- Support 3 gap-filling strategies to reach target size
- Maintain factor/component and cell type awareness throughout

Written by: Helene Hemmer
Date: 2026-02-08
Last modified: 2026-02-08
"""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import scanpy as sc
from anndata import AnnData

# Support script execution by adding module directory to sys.path
import sys
MODULE_DIR = Path(__file__).parent.absolute()
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

# Import run_single_selection (sys.path already includes MODULE_DIR above)
from run_single_selection import run_single_selection

# Use absolute imports (for script execution)
from _constants import (
    DEFAULT_PROBESET_SIZE,
    DEFAULT_REDUCTION_TYPES,
    DEFAULT_ANALYSIS_TYPES,
    DEFAULT_PER_FACTOR_SELECTION,
    DEFAULT_DIMRED_PERCENTAGE,
    DEFAULT_BLACKLIST_PATTERNS,
    DEFAULT_RF_PERCENTAGE,
    DEFAULT_RUN_CELLTYPE_FILLING,
    DEFAULT_RUN_GLOBAL_FILLING,
    DEFAULT_RUN_DEG_FILLING,
    COMBINATION_OVERSAMPLE_FACTOR,
    GAP_FILL_STRATEGY_CELLTYPE,
    GAP_FILL_STRATEGY_DEG,
    GAP_FILL_STRATEGY_GLOBAL,
    GENE_SOURCE_DIMRED,
    GENE_SOURCE_DIMRED_REPLACEMENT,
    GENE_SOURCE_FORCE_INCLUDE,
    GENE_SOURCE_GAP_FILL_CELLTYPE,
    GENE_SOURCE_GAP_FILL_DEG,
    GENE_SOURCE_GAP_FILL_GLOBAL,
    GENE_SOURCE_OVERLAP_TO_RF,
    GENE_SOURCE_RF,
    DEFAULT_MIN_XENIUM_EXPRESSION,
    DEFAULT_MAX_XENIUM_EXPRESSION,
    DEFAULT_ODT_METHOD,
    DEFAULT_ODT_MIN_PROBES_THRESHOLD,
)
from _gene_list_builder import GeneListBuilder
from _odt_filtering import apply_odt_filter_with_replacement


def run_combination_selection(
    strategy: str,
    adata: AnnData,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    reduction_type: str = DEFAULT_REDUCTION_TYPES,
    analysis_type: str = DEFAULT_ANALYSIS_TYPES,
    dimred_method: str = DEFAULT_PER_FACTOR_SELECTION,
    n_components: int = 50,
    pool_size_per_celltype: int = 200,  # Pool size per celltype for Phase 1
    pool_size_per_factor: int = 200,      # Pool size per factor for Phase 1
    rf_percentage: float = DEFAULT_RF_PERCENTAGE,
    dimred_percentage: float = DEFAULT_DIMRED_PERCENTAGE,
    force_include_genes: Optional[List[str]] = None,
    blacklist_patterns: Optional[List[str]] = None,
    use_default_blacklist: bool = True,
    apply_xenium_filter: bool = True,
    xenium_celltype_aware: bool = True,
    apply_odt_filter: bool = False,
    xenium_min_expr: float = DEFAULT_MIN_XENIUM_EXPRESSION,
    xenium_max_expr: float = DEFAULT_MAX_XENIUM_EXPRESSION,
    odt_method: str = DEFAULT_ODT_METHOD,
    odt_min_probes_threshold: int = DEFAULT_ODT_MIN_PROBES_THRESHOLD,
    odt_species: str = 'mus_musculus',
    
    # Gap-filling control (three boolean flags)
    run_celltype_filling: bool = DEFAULT_RUN_CELLTYPE_FILLING,
    run_global_filling: bool = DEFAULT_RUN_GLOBAL_FILLING,
    run_deg_filling: bool = DEFAULT_RUN_DEG_FILLING,
    
    results_dir: Optional[str] = None,
    experiment_name: str = 'combination_selection',
    # Caching support (reuse pre-computed component results)
    rf_deg_cache_dir: Optional[str] = None,
    dimred_cache_dir: Optional[str] = None,
    nmf_model_cache_dir: Optional[str] = None,
    force_recompute: bool = False,
    **kwargs
) -> GeneListBuilder:
    """
    Run combination gene selection strategy (rf_nmf or rf_pca).
    
    This function orchestrates a two-phase workflow:
    1. Run RF (rf_deg) and dimred (dimred_only) strategies independently with FULL filtering
       OR load from cache if available
    2. Combine filtered results with ratio-based selection and duplicate resolution
    3. Apply gap-filling if combined panel < target size (duplicate resolution ≠ gap-filling)
    
    IMPORTANT DISTINCTION:
    - **Duplicate resolution**: When a gene is selected by both RF and dimred, assign it to RF pool.
      This may REDUCE the dimred count, creating a shortfall that must be filled from next-best
      dimred candidates (with ODT checking).
    - **Gap-filling**: When the combined panel (force-include + RF + dimred) is SMALLER than target
      size, add genes from additional sources (DEG-based, celltype-specific, or global dimred).
    
    These are separate operations: duplicate resolution maintains the RF/dimred ratio by replacing
    lost dimred genes, while gap-filling increases total panel size when needed.
    
    Force-include genes are added FIRST, then remaining slots are filled with RF/dimred genes
    according to the specified ratio. Duplicates are assigned to the RF pool, and replacement
    dimred genes are selected with factor/celltype awareness and ODT checking.
    
    If the combined panel is smaller than target size, gap-filling strategies are applied.
    
    Parameters
    ----------
    strategy : str
        Combination strategy name: 'rf_nmf' or 'rf_pca'
    adata : AnnData
        Annotated data matrix with expression data
    probeset_size : int
        Target total number of genes in final panel
    reduction_type : str
        Dimensionality reduction type: 'nmf' or 'pca'
    analysis_type : str, default='per_celltype'
        Dimred analysis level: 'per_celltype' or 'global'
    dimred_method : str, default='method_a'
        Dimred gene selection method: 'method_a' (top genes by absolute factor loading)
    n_components : int, default=5
        Number of components for dimensionality reduction
    rf_percentage : float, default=0.25
        Target percentage of genes from RF (e.g., 0.25 = 25%)
    dimred_percentage : float, default=0.75
        Target percentage of genes from dimred (e.g., 0.75 = 75%)
    force_include_genes : List[str], optional
        Genes to force-include (highest priority, added before ratio calculation)
    blacklist_patterns : List[str], optional
        Gene name prefixes to exclude (e.g., ['mt-', 'rps', 'rpl'])
    apply_xenium_filter : bool, default=True
        Whether to apply Xenium expression filtering
    apply_odt_filter : bool, default=True
        Whether to apply ODT probe designability filtering
    xenium_min_expr : float, default=0.1
        Minimum mean expression threshold for Xenium filter
    xenium_max_expr : float, default=100.0
        Maximum mean expression threshold for Xenium filter
    odt_method : str, default='SCRINSHOT'
        ODT probe design method: 'SCRINSHOT', 'MERFISH', or 'SEQFISHPLUS'
    odt_species : str, default='mus_musculus'
        Species name for ODT pipeline
    run_celltype_filling : bool, default=True
        Enable cell-type-specific dimred gap-filling (per-celltype analysis)
    run_global_filling : bool, default=True
        Enable global dimred gap-filling (global analysis)
    run_deg_filling : bool, default=True
        Enable DEG-based gap-filling (works for both analysis types)
    results_dir : str, optional
        Directory to save results
    experiment_name : str, default='combination_selection'
        Name for this experiment (used in result filenames)
    rf_deg_cache_dir : str, optional
        Directory containing cached rf_deg results (selected_genes.csv).
        If provided and exists, will load cached results instead of recomputing.
        Expected structure: rf_deg_cache_dir/selected_genes.csv
    dimred_cache_dir : str, optional
        Directory containing cached dimred_only results (selected_genes.csv).
        If provided and exists, will load cached results instead of recomputing.
        Expected structure: dimred_cache_dir/selected_genes.csv
    force_recompute : bool, default=False
        If True, ignore cached results and recompute both components.
        Use this to regenerate results even when cache exists.
    **kwargs
        Additional parameters passed to component strategies
        
    Returns
    -------
    GeneListBuilder
        Final gene panel with complete metadata including:
        - gene_source: Labels like 'rf_deg', 'dimred', 'overlap→rf_deg', 
          'dimred_replacement', 'gap_fill_*', 'force_include'
        - component: Factor/component ID for dimred genes
        - celltype: Cell type for per-celltype genes
        - All scores and metadata from component strategies
        
    Notes
    -----
    - Force-include genes reduce the pool available for RF/dimred selection
      Example: 100 genes total, 20 force-include → 60 dimred + 20 RF at 75:25 ratio
    - Duplicates between RF and dimred are always assigned to RF pool
    - Replacement dimred genes maintain factor/celltype from original dimred selection
    - Gap-filling only activates if combined panel < target size
    - Output contains ONLY the final target_size genes (not intermediate selections)
    """
    
    logging.info("=" * 80)
    logging.info(f"COMBINATION STRATEGY: {strategy.upper()}")
    logging.info("=" * 80)
    logging.info(f"Target panel size: {probeset_size} genes")
    logging.info(f"Reduction type: {reduction_type.upper()}")
    logging.info(f"Analysis type: {analysis_type}")
    logging.info(f"Target composition: {rf_percentage:.0%} RF + {dimred_percentage:.0%} {reduction_type.upper()}")
    logging.info(f"ODT filter enabled: {apply_odt_filter}")

    if not apply_odt_filter:
        logging.info(
            "ODT filtering disabled globally: component strategies and replacement checks "
            "will run without ODT validation"
        )

    def _read_cache_filter_flags(cache_dir: str) -> tuple[Optional[bool], Optional[bool], Optional[bool], Optional[list]]:
        """Read filter flags from filtering_summary.json if present.
        
        Returns (odt_flag, xenium_flag, blacklist_any_active_flag, blacklist_custom_patterns)
        Any value is None if the summary file is missing or the field is absent.
        """
        summary_file = os.path.join(cache_dir, 'filtering_summary.json')
        if not os.path.exists(summary_file):
            return None, None, None, None

        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            filters = summary.get('filters_applied', {})
            return (
                filters.get('odt'),
                filters.get('xenium'),
                filters.get('blacklist_any_active'),
                filters.get('blacklist_custom_patterns'),
            )
        except Exception as e:
            logging.warning(f"Could not parse cache summary {summary_file}: {e}")
            return None, None, None, None
    
    # Validate strategy
    if strategy not in ['rf_nmf', 'rf_pca']:
        raise ValueError(f"Invalid combination strategy: {strategy}. Must be 'rf_nmf' or 'rf_pca'")
    
    # Validate reduction type matches strategy
    expected_reduction = 'nmf' if 'nmf' in strategy else 'pca'
    if reduction_type.lower() != expected_reduction:
        raise ValueError(
            f"Reduction type '{reduction_type}' doesn't match strategy '{strategy}'. "
            f"Expected '{expected_reduction}'"
        )
    
    # Validate percentages
    if not (0 < rf_percentage < 1 and 0 < dimred_percentage < 1):
        raise ValueError(f"Percentages must be between 0 and 1, got RF={rf_percentage}, dimred={dimred_percentage}")
    
    if abs((rf_percentage + dimred_percentage) - 1.0) > 0.01:
        raise ValueError(
            f"Percentages must sum to 1.0, got RF={rf_percentage} + dimred={dimred_percentage} "
            f"= {rf_percentage + dimred_percentage}"
        )
    
    # Setup results directory
    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), 'results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Process force-include genes (highest priority)
    # =========================================================================
    
    force_include_set = _validate_force_include_genes(
        force_include_genes, adata, probeset_size
    )
    
    # Adjust target size to account for force-included genes
    adjusted_target_size = probeset_size - len(force_include_set)
    
    if adjusted_target_size <= 0:
        raise ValueError(
            f"Force-include genes ({len(force_include_set)}) exceed or equal target size ({probeset_size}). "
            "Reduce force-include list or increase target size."
        )
    
    logging.info(f"Force-include genes: {len(force_include_set)}")
    if force_include_set:
        logging.info(f"  Genes: {', '.join(sorted(force_include_set))}")
    logging.info(f"Adjusted target for RF/dimred: {adjusted_target_size} genes")
    
    # Blacklist is applied once at pipeline level (run_selection_pipeline.py) before
    # any strategy runs. adata arriving here is already filtered; pass disabled flags
    # to sub-strategy calls so the filter is never applied a second time.
    _sub_blacklist_patterns = None
    _sub_use_default_blacklist = False

    # =========================================================================
    # STEP 2: Calculate target counts for RF and dimred (from adjusted target)
    # =========================================================================
    
    n_rf_target = int(adjusted_target_size * rf_percentage)
    n_dimred_target = adjusted_target_size - n_rf_target  # Ensures exact sum
    
    logging.info(f"Target gene counts:")
    logging.info(f"  RF genes: {n_rf_target}")
    logging.info(f"  Dimred genes: {n_dimred_target}")
    logging.info(f"  Force-include: {len(force_include_set)}")
    logging.info(f"  Total: {n_rf_target + n_dimred_target + len(force_include_set)} = {probeset_size}")
    
    # =========================================================================
    # STEP 3: Run RF component (rf_deg) with FULL filtering OR load from cache
    # =========================================================================
    
    logging.info("")
    logging.info("=" * 80)
    logging.info("PHASE 1: Run RF (rf_deg) component with filtering")
    logging.info("=" * 80)
    
    # Oversample to ensure enough candidates for duplicate resolution
    n_rf_oversample = int(n_rf_target * COMBINATION_OVERSAMPLE_FACTOR)
    
    logging.info(f"Requesting {n_rf_oversample} RF genes (oversampled for duplicate resolution)")
    
    # Check for cached RF results
    rf_builder = None
    rf_odt_tested_genes = set()
    
    if rf_deg_cache_dir and not force_recompute:
        # Look for ranked_gene_list.csv (new name) with fallback to the old name
        rf_cache_file = os.path.join(rf_deg_cache_dir, 'ranked_gene_list.csv')
        if not os.path.exists(rf_cache_file):
            rf_cache_file = os.path.join(rf_deg_cache_dir, 'ranked_gene_list_final.csv')

        if os.path.exists(rf_cache_file):
            logging.info(f"Found cached RF ranked list: {rf_cache_file}")
            logging.info("Loading cached rf_deg results instead of recomputing...")
            try:
                # Load ranked list (with filter results)
                rf_cache_df = pd.read_csv(rf_cache_file)

                rf_cache_odt_flag, rf_cache_xenium_flag, rf_cache_blacklist_flag, rf_cache_bl_patterns = _read_cache_filter_flags(rf_deg_cache_dir)
                if apply_odt_filter and rf_cache_odt_flag is False:
                    raise ValueError(
                        "RF cache was generated with ODT disabled, but combination run requires ODT-enabled component cache"
                    )
                if (not apply_odt_filter) and apply_xenium_filter and rf_cache_xenium_flag is False:
                    raise ValueError(
                        "RF cache was generated with Xenium disabled, but no-ODT combination run requires Xenium-cleaned component cache"
                    )
                # Blacklist compatibility: flag if current run uses blacklist but cache did not (or vice versa)
                current_blacklist_active = bool(blacklist_patterns or use_default_blacklist)
                if rf_cache_blacklist_flag is not None and current_blacklist_active != rf_cache_blacklist_flag:
                    raise ValueError(
                        f"RF cache blacklist mismatch: cache was generated with blacklist_any_active={rf_cache_blacklist_flag}, "
                        f"but current run has blacklist_any_active={current_blacklist_active}. "
                        "Use --force_recompute or point to a compatible RF cache."
                    )
                # Warn (don't fail) if custom patterns differ
                if rf_cache_bl_patterns is not None and (blacklist_patterns or []) != rf_cache_bl_patterns:
                    logging.warning(
                        f"RF cache was built with custom blacklist patterns {rf_cache_bl_patterns}, "
                        f"but current run uses {blacklist_patterns or []}. Gene pools may differ."
                    )
                
                # Extract ODT-tested genes (genes marked as selected in final results).
                # Support new column name ('in_panel'/'final_selection') and old ('selected').
                _rf_sel_col = next(
                    (c for c in ('in_panel', 'final_selection', 'selected') if c in rf_cache_df.columns),
                    None,
                )
                if _rf_sel_col is not None:
                    selected_cache_rf = set(rf_cache_df[rf_cache_df[_rf_sel_col] == True]['gene'].tolist())
                    if apply_odt_filter:
                        rf_odt_tested_genes = selected_cache_rf
                        logging.info(f"✓ Loaded {len(rf_odt_tested_genes)} ODT-tested RF genes from ranked list")
                    else:
                        rf_odt_tested_genes = set()
                        if apply_xenium_filter:
                            logging.info(
                                f"✓ Loaded {len(selected_cache_rf)} Xenium-cleaned RF genes from ranked list (ODT disabled mode)"
                            )
                        else:
                            logging.info(
                                f"✓ Loaded {len(selected_cache_rf)} RF genes from ranked list (ODT disabled, Xenium disabled mode)"
                            )
                else:
                    logging.warning(
                        f"Cached RF results missing selection column: {rf_cache_file}\n"
                        "Cache appears to be in old format or incomplete. Will regenerate from scratch."
                    )
                    rf_builder = None
                    rf_odt_tested_genes = set()
                    # Skip remaining cache loading and trigger from-scratch generation
                    raise ValueError("Cache format invalid")  # Caught by outer try/except
                
                # Create builder from cached data
                from ._gene_list_builder import GeneListBuilder
                rf_builder = GeneListBuilder(
                    strategy_name='rf_deg',
                    analysis_type='global',
                )
                
                # Add genes from cache
                for _, row in rf_cache_df.iterrows():
                    # Backward compatibility: Map old DT values to RF
                    gene_source = row.get('gene_source', 'rf_deg')
                    if gene_source == 'DT':
                        gene_source = 'RF'
                    elif gene_source == 'overlap→DT':
                        gene_source = 'overlap→rf_deg'
                    
                    rf_builder.add_gene(
                        gene_name=row['gene'],
                        gene_source=gene_source,
                        rank=row.get('rank', None),
                        selection_score=row.get('selection_score', None),
                        celltype=row.get('celltype', 'global'),
                        component=row.get('component', None),
                        additional_metadata={}
                    )
                
                logging.info(f"✓ Loaded {len(rf_builder.get_all_genes())} RF genes from ranked list")
                
            except Exception as e:
                logging.warning(f"Failed to load RF cache: {e}")
                logging.warning("Will recompute RF component")
                rf_builder = None
                rf_odt_tested_genes = set()
    
    # Run RF selection if not cached
    if rf_builder is None:
        logging.info("Running RF (rf_deg) selection from scratch...")
                
        rf_results_dir = os.path.join(results_dir, 'rf_component')
        os.makedirs(rf_results_dir, exist_ok=True)
        
        rf_builder = run_single_selection(
            strategy='rf_deg',
            adata=adata,
            probeset_size=n_rf_oversample,
            force_include_genes=None,  # Don't double-count force-include in components
            blacklist_patterns=_sub_blacklist_patterns,
            use_default_blacklist=_sub_use_default_blacklist,
            apply_xenium_filter=apply_xenium_filter,
            xenium_celltype_aware=xenium_celltype_aware,
            apply_odt_filter=apply_odt_filter,
            xenium_min_expr=xenium_min_expr,
            xenium_max_expr=xenium_max_expr,
            odt_method=odt_method,
            odt_species=odt_species,
            results_dir=rf_results_dir,
            **kwargs
        )
        
        # Load ODT-tested genes from newly computed results
        rf_results_file = os.path.join(rf_results_dir, 'ranked_gene_list.csv')
        if not os.path.exists(rf_results_file):
            rf_results_file = os.path.join(rf_results_dir, 'ranked_gene_list_final.csv')
        if os.path.exists(rf_results_file):
            rf_results_df = pd.read_csv(rf_results_file)
            # Support both old ('selected') and new ('in_panel'/'final_selection') column names
            _sel_col = next((c for c in ('in_panel', 'final_selection', 'selected')
                             if c in rf_results_df.columns), None)
            if _sel_col is not None:
                selected_rf = set(rf_results_df[rf_results_df[_sel_col] == True]['gene'].tolist())
                if apply_odt_filter:
                    rf_odt_tested_genes = selected_rf
                    logging.info(f"✓ Loaded {len(rf_odt_tested_genes)} ODT-tested RF genes from new run")
                else:
                    rf_odt_tested_genes = set()
                    if apply_xenium_filter:
                        logging.info(
                            f"✓ Loaded {len(selected_rf)} Xenium-cleaned RF genes from new run (ODT disabled mode)"
                        )
                    else:
                        logging.info(
                            f"✓ Loaded {len(selected_rf)} RF genes from new run (ODT disabled, Xenium disabled mode)"
                        )
    
    rf_genes_available = rf_builder.get_all_genes()
    logging.info(f"RF component returned {len(rf_genes_available)} genes after filtering")
    
    # =========================================================================
    # STEP 4: Run dimred component (dimred_only) with FULL filtering OR load from cache
    # =========================================================================
    
    logging.info("")
    logging.info("=" * 80)
    logging.info(f"PHASE 2: Run {reduction_type.upper()} (dimred_only) component with filtering")
    logging.info("=" * 80)
    
    # Oversample for dimred as well
    n_dimred_oversample = int(n_dimred_target * COMBINATION_OVERSAMPLE_FACTOR)
    
    logging.info(f"Requesting {n_dimred_oversample} dimred genes (oversampled for duplicate resolution)")
    
    # Check for cached dimred results
    dimred_builder = None
    dimred_odt_tested_genes = set()
    
    if dimred_cache_dir and not force_recompute:
        # Look for ranked_gene_list.csv (new name) with fallback to the old name
        dimred_cache_file = os.path.join(dimred_cache_dir, 'ranked_gene_list.csv')
        if not os.path.exists(dimred_cache_file):
            dimred_cache_file = os.path.join(dimred_cache_dir, 'ranked_gene_list_final.csv')

        if os.path.exists(dimred_cache_file):
            logging.info(f"Found cached dimred ranked list: {dimred_cache_file}")
            logging.info("Loading cached dimred_only results instead of recomputing...")
            try:
                # Load ranked list (with filter results)
                dimred_cache_df = pd.read_csv(dimred_cache_file)

                dimred_cache_odt_flag, dimred_cache_xenium_flag, dimred_cache_blacklist_flag, dimred_cache_bl_patterns = _read_cache_filter_flags(dimred_cache_dir)
                if apply_odt_filter and dimred_cache_odt_flag is False:
                    raise ValueError(
                        "Dimred cache was generated with ODT disabled, but combination run requires ODT-enabled component cache"
                    )
                if (not apply_odt_filter) and apply_xenium_filter and dimred_cache_xenium_flag is False:
                    raise ValueError(
                        "Dimred cache was generated with Xenium disabled, but no-ODT combination run requires Xenium-cleaned component cache"
                    )
                # Blacklist compatibility: flag if current run uses blacklist but cache did not (or vice versa)
                current_blacklist_active = bool(blacklist_patterns or use_default_blacklist)
                if dimred_cache_blacklist_flag is not None and current_blacklist_active != dimred_cache_blacklist_flag:
                    raise ValueError(
                        f"Dimred cache blacklist mismatch: cache was generated with blacklist_any_active={dimred_cache_blacklist_flag}, "
                        f"but current run has blacklist_any_active={current_blacklist_active}. "
                        "Use --force_recompute or point to a compatible dimred cache."
                    )
                # Warn (don't fail) if custom patterns differ
                if dimred_cache_bl_patterns is not None and (blacklist_patterns or []) != dimred_cache_bl_patterns:
                    logging.warning(
                        f"Dimred cache was built with custom blacklist patterns {dimred_cache_bl_patterns}, "
                        f"but current run uses {blacklist_patterns or []}. Gene pools may differ."
                    )
                
                # Extract ODT-tested genes.
                # Support new column name ('in_panel'/'final_selection') and old ('selected').
                _dim_sel_col = next(
                    (c for c in ('in_panel', 'final_selection', 'selected') if c in dimred_cache_df.columns),
                    None,
                )
                if _dim_sel_col is not None:
                    selected_cache_dimred = set(dimred_cache_df[dimred_cache_df[_dim_sel_col] == True]['gene'].tolist())
                    if apply_odt_filter:
                        dimred_odt_tested_genes = selected_cache_dimred
                        logging.info(f"✓ Loaded {len(dimred_odt_tested_genes)} ODT-tested dimred genes from ranked list")
                    else:
                        dimred_odt_tested_genes = set()
                        if apply_xenium_filter:
                            logging.info(
                                f"✓ Loaded {len(selected_cache_dimred)} Xenium-cleaned dimred genes from ranked list (ODT disabled mode)"
                            )
                        else:
                            logging.info(
                                f"✓ Loaded {len(selected_cache_dimred)} dimred genes from ranked list (ODT disabled, Xenium disabled mode)"
                            )
                else:
                    logging.warning(
                        f"Cached dimred results missing selection column: {dimred_cache_file}\n"
                        "Cache appears to be in old format or incomplete. Will regenerate from scratch."
                    )
                    dimred_builder = None
                    dimred_odt_tested_genes = set()
                    # Skip remaining cache loading and trigger from-scratch generation
                    raise ValueError("Cache format invalid")  # Caught by outer try/except
                
                # Create builder from cached data
                from ._gene_list_builder import GeneListBuilder
                dimred_builder = GeneListBuilder(
                    strategy_name='dimred_only',
                    analysis_type=analysis_type,
                )
                
                # Add genes from cache
                for _, row in dimred_cache_df.iterrows():
                    dimred_builder.add_gene(
                        gene_name=row['gene'],
                        gene_source=row.get('gene_source', 'dimred'),
                        rank=row.get('rank', None),
                        selection_score=row.get('selection_score', None),
                        celltype=row.get('celltype', 'global'),
                        component=row.get('component', None),
                        additional_metadata={'component_loading': row.get('component_loading', None)}
                    )
                
                logging.info(f"✓ Loaded {len(dimred_builder.get_all_genes())} dimred genes from ranked list")
                
            except Exception as e:
                logging.warning(f"Failed to load dimred cache: {e}")
                logging.warning("Will recompute dimred component")
                dimred_builder = None
                dimred_odt_tested_genes = set()
    
    # Run dimred selection if not cached
    if dimred_builder is None:
        logging.info(f"Running {reduction_type.upper()} (dimred_only) selection from scratch...")

        # ---------------------------------------------------------------------------
        # Resolve NMF model cache directory.
        # Priority:
        #   1. Explicitly supplied nmf_model_cache_dir (user override)
        #   2. Auto-detected from dimred_cache_dir/nmf_models/ (reuses a pre-computed
        #      dimred_only fit so the NMF is not recomputed from scratch)
        #   3. None (NMF will be fit fresh and saved inside dimred_results_dir)
        # ---------------------------------------------------------------------------
        _resolved_nmf_cache = nmf_model_cache_dir
        if _resolved_nmf_cache is None and dimred_cache_dir:
            _candidate = os.path.join(dimred_cache_dir, 'nmf_models')
            if os.path.isdir(_candidate):
                _resolved_nmf_cache = _candidate
                logging.info(
                    f"Auto-resolved NMF model cache from dimred_cache_dir: {_resolved_nmf_cache}"
                )
            else:
                logging.debug(
                    f"dimred_cache_dir provided but no nmf_models/ sub-directory found "
                    f"({_candidate}); NMF will be refit from scratch."
                )

        dimred_results_dir = os.path.join(results_dir, f'{reduction_type}_component')
        os.makedirs(dimred_results_dir, exist_ok=True)

        dimred_builder = run_single_selection(
            strategy='dimred_only',
            adata=adata,
            probeset_size=n_dimred_oversample,
            reduction_type=reduction_type,
            analysis_type=analysis_type,
            dimred_method=dimred_method,
            n_components=n_components,
            pool_size_per_celltype=pool_size_per_celltype,
            pool_size_per_factor=pool_size_per_factor,
            force_include_genes=None,  # Don't double-count force-include in components
            blacklist_patterns=_sub_blacklist_patterns,
            use_default_blacklist=_sub_use_default_blacklist,
            apply_xenium_filter=apply_xenium_filter,
            xenium_celltype_aware=xenium_celltype_aware,
            apply_odt_filter=apply_odt_filter,
            xenium_min_expr=xenium_min_expr,
            xenium_max_expr=xenium_max_expr,
            odt_method=odt_method,
            odt_species=odt_species,
            results_dir=dimred_results_dir,
            nmf_model_cache_dir=_resolved_nmf_cache,
            **kwargs
        )
        
        # Load ODT-tested genes from newly computed results
        dimred_results_file = os.path.join(dimred_results_dir, 'ranked_gene_list.csv')
        if not os.path.exists(dimred_results_file):
            dimred_results_file = os.path.join(dimred_results_dir, 'ranked_gene_list_final.csv')
        if os.path.exists(dimred_results_file):
            dimred_results_df = pd.read_csv(dimred_results_file)
            _dim_res_col = next(
                (c for c in ('in_panel', 'final_selection', 'selected') if c in dimred_results_df.columns),
                None,
            )
            if _dim_res_col is not None:
                selected_dimred = set(dimred_results_df[dimred_results_df[_dim_res_col] == True]['gene'].tolist())
                if apply_odt_filter:
                    dimred_odt_tested_genes = selected_dimred
                    logging.info(f"✓ Loaded {len(dimred_odt_tested_genes)} ODT-tested dimred genes from new run")
                else:
                    dimred_odt_tested_genes = set()
                    if apply_xenium_filter:
                        logging.info(
                            f"✓ Loaded {len(selected_dimred)} Xenium-cleaned dimred genes from new run (ODT disabled mode)"
                        )
                    else:
                        logging.info(
                            f"✓ Loaded {len(selected_dimred)} dimred genes from new run (ODT disabled, Xenium disabled mode)"
                        )
    
    dimred_genes_available = dimred_builder.get_all_genes()
    logging.info(f"Dimred component returned {len(dimred_genes_available)} genes after filtering")
    
    # =========================================================================
    # STEP 5: Combine filtered results with duplicate resolution
    # =========================================================================
    
    logging.info("")
    logging.info("=" * 80)
    logging.info("PHASE 3: Combine filtered components with duplicate resolution")
    logging.info("=" * 80)
    
    # Log ODT-tested gene statistics
    all_odt_tested = rf_odt_tested_genes | dimred_odt_tested_genes
    logging.info(f"ODT-tested genes available for reuse:")
    logging.info(f"  RF-tested: {len(rf_odt_tested_genes)}")
    logging.info(f"  Dimred-tested: {len(dimred_odt_tested_genes)}")
    logging.info(f"  Total unique: {len(all_odt_tested)}")
    
    combined_builder = _combine_filtered_components(
        rf_builder=rf_builder,
        dimred_builder=dimred_builder,
        n_rf_target=n_rf_target,
        n_dimred_target=n_dimred_target,
        force_include_set=force_include_set,
        rf_odt_tested_genes=rf_odt_tested_genes,
        dimred_odt_tested_genes=dimred_odt_tested_genes,
        apply_odt_filter=apply_odt_filter,
        reduction_type=reduction_type,
        analysis_type=analysis_type,
        adata=adata,
        odt_method=odt_method,
        odt_min_probes_threshold=odt_min_probes_threshold,
        odt_species=odt_species,
        results_dir=results_dir
    )
    
    # =========================================================================
    # STEP 6: Check if gap-filling is needed
    # =========================================================================
    
    current_size = len(combined_builder.get_all_genes())
    gap_needed = probeset_size - current_size
    
    logging.info("")
    logging.info("=" * 80)
    logging.info("PHASE 4: Gap-filling assessment")
    logging.info("=" * 80)
    logging.info(f"Current panel size: {current_size}")
    logging.info(f"Target size: {probeset_size}")
    logging.info(f"Gap to fill: {gap_needed}")
    
    if gap_needed <= 0:
        logging.info("No gap-filling needed - panel is at target size")
        combined_builder.add_metadata('gap_filling_applied', False)
        combined_builder.add_metadata('gap_filling_strategy', 'none')
    else:
        logging.info(f"Gap-filling required: {gap_needed} genes needed")
        
        # Apply gap-filling strategies
        final_builder = _apply_gap_filling(
            base_builder=combined_builder,
            rf_builder=rf_builder,
            dimred_builder=dimred_builder,
            adata=adata,
            gap_needed=gap_needed,
            target_size=probeset_size,
            run_celltype_filling=run_celltype_filling,
            run_global_filling=run_global_filling,
            run_deg_filling=run_deg_filling,
            reduction_type=reduction_type,
            analysis_type=analysis_type,
            dimred_method=dimred_method,
            n_components=n_components,
            results_dir=results_dir
        )
        
        combined_builder = final_builder
    
    # =========================================================================
    # STEP 7: Final validation and output
    # =========================================================================
    
    final_genes = combined_builder.get_all_genes()
    final_size = len(final_genes)
    
    logging.info("")
    logging.info("=" * 80)
    logging.info(f"COMBINATION STRATEGY COMPLETE: {strategy.upper()}")
    logging.info("=" * 80)
    logging.info(f"Final panel size: {final_size} genes")
    logging.info(f"Target size: {probeset_size} genes")
    
    if final_size != probeset_size:
        logging.warning(
            f"WARNING: Final size ({final_size}) does not match target ({probeset_size}). "
            "Check gap-filling strategies."
        )
    
    # Mark all genes as selected — every gene that made it into the combined
    # builder has passed Xenium (and ODT if enabled) filtering inside its
    # component, so all are final panel members.
    combined_builder.mark_selected(final_genes, 'initial')
    combined_builder.mark_selected(final_genes, 'final')

    # Add combination metadata
    combined_builder.add_metadata('strategy', strategy)
    combined_builder.add_metadata('rf_percentage', rf_percentage)
    combined_builder.add_metadata('dimred_percentage', dimred_percentage)
    combined_builder.add_metadata('reduction_type', reduction_type)
    combined_builder.add_metadata('analysis_type', analysis_type)
    combined_builder.add_metadata('dimred_method', dimred_method)
    combined_builder.add_metadata('n_components', n_components)
    combined_builder.add_metadata('target_size', probeset_size)
    combined_builder.add_metadata('final_size', final_size)
    
    # Save results
    _save_combination_results(
        rf_builder=rf_builder,
        dimred_builder=dimred_builder,
        combined_builder=combined_builder,
        results_dir=results_dir,
        strategy=strategy,
        reduction_type=reduction_type
    )
    
    logging.info(f"Results saved to: {results_dir}")
    logging.info("=" * 80)
    
    return combined_builder


# =============================================================================
# Helper Functions
# =============================================================================


def _validate_force_include_genes(
    force_include_genes: Optional[List[str]],
    adata: AnnData,
    probeset_size: int
) -> Set[str]:
    """
    Validate and filter force-include genes.
    
    Parameters
    ----------
    force_include_genes : List[str], optional
        Genes to force-include
    adata : AnnData
        Annotated data matrix (for validation)
    probeset_size : int
        Target panel size
        
    Returns
    -------
    Set[str]
        Validated set of force-include genes
    """
    if not force_include_genes:
        return set()
    
    valid_genes = []
    invalid_genes = []
    
    for gene in force_include_genes:
        if gene in adata.var_names:
            valid_genes.append(gene)
        else:
            invalid_genes.append(gene)
    
    if invalid_genes:
        logging.warning(
            f"Force-include genes not found in dataset (skipping): {', '.join(invalid_genes)}"
        )
    
    if len(valid_genes) >= probeset_size:
        raise ValueError(
            f"Force-include genes ({len(valid_genes)}) >= target size ({probeset_size}). "
            "Reduce force-include list or increase target size."
        )
    
    return set(valid_genes)


def _combine_filtered_components(
    rf_builder: GeneListBuilder,
    dimred_builder: GeneListBuilder,
    n_rf_target: int,
    n_dimred_target: int,
    force_include_set: Set[str],
    rf_odt_tested_genes: Set[str],
    dimred_odt_tested_genes: Set[str],
    apply_odt_filter: bool,
    reduction_type: str,
    analysis_type: str,
    adata: AnnData,
    odt_method: str,
    odt_min_probes_threshold: int,
    odt_species: str,
    results_dir: str
) -> GeneListBuilder:
    """
    Combine RF and dimred components with duplicate resolution.
    
    This function implements the core combination logic:
    1. Handle force-include genes FIRST (highest priority, add before combination)
    2. Select top N RF genes and M dimred genes
    3. Identify overlapping genes (selected by both methods)
    4. Assign overlaps to RF pool (RF priority)
    5. Replace lost dimred genes with next-best dimred candidates
    6. Apply ODT checking ONLY to NEW genes (reuse ODT results from single runs)
    
    Parameters
    ----------
    rf_builder : GeneListBuilder
        RF component results (filtered)
    dimred_builder : GeneListBuilder
        Dimred component results (filtered)
    n_rf_target : int
        Target number of RF genes
    n_dimred_target : int
        Target number of dimred genes
    force_include_set : Set[str]
        Force-include genes to add (highest priority)
    rf_odt_tested_genes : Set[str]
        Genes already tested with ODT in RF single run
    dimred_odt_tested_genes : Set[str]
        Genes already tested with ODT in dimred single run
    reduction_type : str
        'nmf' or 'pca'
    adata : AnnData
        Annotated data matrix (for force-include validation)
    odt_method : str
        ODT probe design method
    odt_species : str
        Species for ODT (e.g., 'mus_musculus')
    results_dir : str
        Directory to save duplicate resolution report
        
    Returns
    -------
    GeneListBuilder
        Combined gene panel with metadata
    """
    
    logging.info("Combining filtered components...")
    
    # =========================================================================
    # STEP 1: Handle force-include genes FIRST
    # =========================================================================
    
    all_odt_tested = rf_odt_tested_genes | dimred_odt_tested_genes
    force_include_needing_odt = force_include_set - all_odt_tested
    
    if force_include_set:
        logging.info(f"")
        logging.info(f"Processing {len(force_include_set)} force-include genes...")
        if apply_odt_filter:
            logging.info(f"  Already ODT-tested: {len(force_include_set - force_include_needing_odt)}")
            logging.info(f"  Not yet ODT-tested: {len(force_include_needing_odt)}")

        if apply_odt_filter and force_include_needing_odt:
            # Force-include genes are MANDATORY - include without ODT validation
            logging.info(
                f"  Including {len(force_include_needing_odt)} force-include genes WITHOUT ODT validation"
            )
            logging.warning(
                f"  WARNING: These {len(force_include_needing_odt)} genes were not ODT-tested in single strategies.\n"
                f"    They may not be designable. Consider running single strategies with them to validate.\n"
                f"    Genes: {', '.join(sorted(list(force_include_needing_odt)[:10]))}{'...' if len(force_include_needing_odt) > 10 else ''}"
            )
    
    # =========================================================================
    # STEP 2: Select genes from components, removing force-include to avoid duplicates
    # =========================================================================
    
    # Get ranked gene lists
    rf_genes_ranked = rf_builder.get_all_genes()
    dimred_genes_ranked = dimred_builder.get_all_genes()
    
    # Remove force-include genes from component lists to avoid duplicates
    rf_genes_filtered = [g for g in rf_genes_ranked if g not in force_include_set]
    dimred_genes_filtered = [g for g in dimred_genes_ranked if g not in force_include_set]
    
    if force_include_set:
        logging.info(f"Removed {len(rf_genes_ranked) - len(rf_genes_filtered)} force-include genes from RF list")
        logging.info(f"Removed {len(dimred_genes_ranked) - len(dimred_genes_filtered)} force-include genes from dimred list")
    
    # Select top N genes from each method
    rf_genes_selected = set(rf_genes_filtered[:n_rf_target])
    dimred_genes_selected = set(dimred_genes_filtered)  # Use all available dimred genes initially
    
    logging.info(f"Selected {len(rf_genes_selected)} RF genes (target: {n_rf_target})")
    logging.info(f"Selected {len(dimred_genes_selected)} dimred genes (will be trimmed after duplicate resolution)")
    
    # Identify overlapping genes
    overlapping_genes = rf_genes_selected & dimred_genes_selected
    
    # Assign overlaps to RF pool (RF priority for duplicates)
    rf_final = rf_genes_selected  # Includes overlaps
    dimred_unique = dimred_genes_selected - overlapping_genes
    
    logging.info(f"")
    logging.info(f"Duplicate analysis:")
    logging.info(f"  Overlapping genes: {len(overlapping_genes)}")
    logging.info(f"  RF-unique genes: {len(rf_genes_selected - overlapping_genes)}")
    logging.info(f"  Dimred-unique genes: {len(dimred_unique)}")
    
    # Calculate how many dimred genes we lost to duplicates
    dimred_shortfall = n_dimred_target - len(dimred_unique)
    
    if dimred_shortfall > 0:
        logging.info(f"")
        logging.info(f"Dimred shortfall: {dimred_shortfall} genes (lost to duplicates)")
        logging.info(f"Replacing from next-best dimred candidates...")
        
        # Get replacement candidates (dimred genes not in RF or current dimred set)
        replacement_candidates = [
            g for g in dimred_genes_filtered 
            if g not in rf_final and g not in dimred_unique
        ]
        
        logging.info(f"Available replacement candidates: {len(replacement_candidates)}")
        
        if len(replacement_candidates) < dimred_shortfall:
            logging.warning(
                f"Not enough replacement candidates ({len(replacement_candidates)}) "
                f"to fill shortfall ({dimred_shortfall})"
            )
        
        # Replace genes with factor/celltype awareness and selective ODT testing
        dimred_replacements, replacement_report = _replace_dimred_genes_with_odt_check(
            n_needed=dimred_shortfall,
            candidates=replacement_candidates,
            dimred_builder=dimred_builder,
            current_rf_genes=rf_final,
            current_dimred_genes=dimred_unique,
            rf_odt_tested_genes=rf_odt_tested_genes,
            dimred_odt_tested_genes=dimred_odt_tested_genes,
            force_include_genes=force_include_set,
            apply_odt_filter=apply_odt_filter,
            odt_method=odt_method,
            odt_min_probes_threshold=odt_min_probes_threshold,
            odt_species=odt_species,
            results_dir=results_dir
        )
        
        # Add replacements to dimred pool
        dimred_final = dimred_unique | set(dimred_replacements)
        
        logging.info(f"Added {len(dimred_replacements)} replacement dimred genes")
        logging.info(f"Final dimred count: {len(dimred_final)} (target: {n_dimred_target})")
        
    else:
        # No shortfall - just take top N dimred genes
        dimred_final = set(list(dimred_unique)[:n_dimred_target])
        replacement_report = []
        
        logging.info(f"No shortfall - selected top {n_dimred_target} dimred genes")
    
    # Build combined gene list with metadata
    combined_builder = GeneListBuilder(
        strategy_name=f'rf_{reduction_type}',
        analysis_type=analysis_type,
    )
    
    # Add force-include genes (highest priority)
    for gene in force_include_set:
        combined_builder.add_gene(
            gene_name=gene,
            gene_source=GENE_SOURCE_FORCE_INCLUDE,
            rank=None,  # Will be set during finalization
            selection_score=None,
            celltype='global',
            component=None,
            additional_metadata={'force_included': True}
        )
    
    # Add RF genes (including overlaps)
    for gene in rf_final:
        rf_metadata = rf_builder.get_gene_metadata(gene)
        
        gene_source = GENE_SOURCE_RF
        if gene in overlapping_genes:
            gene_source = GENE_SOURCE_OVERLAP_TO_RF
        
        combined_builder.add_gene(
            gene_name=gene,
            gene_source=gene_source,
            rank=rf_metadata.rank if rf_metadata else None,
            selection_score=rf_metadata.selection_score if rf_metadata else None,
            celltype=rf_metadata.celltype if rf_metadata else 'global',
            component=None,  # RF genes don't have components
            additional_metadata={
                'from_rf': True,
                'is_overlap': gene in overlapping_genes,
                'original_rf_rank': rf_metadata.rank if rf_metadata else None
            }
        )
    
    # Add dimred genes (unique + replacements)
    for gene in dimred_final:
        dimred_metadata = dimred_builder.get_gene_metadata(gene)
        
        # Check if this is a replacement gene
        is_replacement = any(r['replacement_gene'] == gene for r in replacement_report)
        
        gene_source = GENE_SOURCE_DIMRED_REPLACEMENT if is_replacement else GENE_SOURCE_DIMRED
        
        combined_builder.add_gene(
            gene_name=gene,
            gene_source=gene_source,
            rank=dimred_metadata.rank if dimred_metadata else None,
            selection_score=dimred_metadata.selection_score if dimred_metadata else None,
            celltype=dimred_metadata.celltype if dimred_metadata else 'global',
            component=dimred_metadata.component if dimred_metadata else None,
            additional_metadata={
                'from_dimred': True,
                'is_replacement': is_replacement,
                'original_dimred_rank': dimred_metadata.rank if dimred_metadata else None,
                'component_loading': dimred_metadata.component_loading if dimred_metadata else None
            }
        )
    
    # Add combination metadata
    combined_builder.add_metadata('n_force_include', len(force_include_set))
    combined_builder.add_metadata('n_rf_final', len(rf_final))
    combined_builder.add_metadata('n_dimred_final', len(dimred_final))
    combined_builder.add_metadata('n_overlapping', len(overlapping_genes))
    combined_builder.add_metadata('n_dimred_replacements', len(replacement_report))
    combined_builder.add_metadata('overlapping_genes', list(overlapping_genes))
    
    total_genes = len(force_include_set) + len(rf_final) + len(dimred_final)
    logging.info(f"")
    logging.info(f"Combined panel summary:")
    logging.info(f"  Force-include: {len(force_include_set)}")
    logging.info(f"  RF genes: {len(rf_final)} (includes {len(overlapping_genes)} overlaps)")
    logging.info(f"  Dimred genes: {len(dimred_final)} (includes {len(replacement_report)} replacements)")
    logging.info(f"  Total: {total_genes}")
    
    return combined_builder


def _replace_dimred_genes_with_odt_check(
    n_needed: int,
    candidates: List[str],
    dimred_builder: GeneListBuilder,
    current_rf_genes: Set[str],
    current_dimred_genes: Set[str],
    rf_odt_tested_genes: Set[str],
    dimred_odt_tested_genes: Set[str],
    force_include_genes: Set[str],
    apply_odt_filter: bool,
    odt_method: str,
    odt_min_probes_threshold: int,
    odt_species: str,
    results_dir: str
) -> Tuple[List[str], List[Dict]]:
    """
    Replace dimred genes lost to duplicates with ODT-checked candidates.
    
    Maintains factor/component and cell type awareness during replacement.
    TRUSTS ODT results from single-strategy runs - only tests NEW genes.
    
    Parameters
    ----------
    n_needed : int
        Number of replacement genes needed
    candidates : List[str]
        Ranked list of replacement candidates
    dimred_builder : GeneListBuilder
        Dimred component results (for metadata)
    current_rf_genes : Set[str]
        Current RF genes (to avoid duplicates)
    current_dimred_genes : Set[str]
        Current dimred genes (to avoid duplicates)
    rf_odt_tested_genes : Set[str]
        Genes already tested with ODT in RF single run
    dimred_odt_tested_genes : Set[str]
        Genes already tested with ODT in dimred single run
    force_include_genes : Set[str]
        Force-include genes (to avoid duplicates)
    odt_method: str
        ODT probe design method to use for checking replacements
    odt_species : str
        Species for ODT (e.g., 'mus_musculus')
    results_dir : str
        Directory for ODT intermediate files
        
    Returns
    -------
    Tuple[List[str], List[Dict]]
        - List of accepted replacement genes
        - List of replacement report dicts
    """
    
    replacements = []
    replacement_report = []
    
    all_odt_tested = rf_odt_tested_genes | dimred_odt_tested_genes
    all_excluded = current_rf_genes | current_dimred_genes | force_include_genes
    
    odt_output_dir = None
    if apply_odt_filter:
        odt_output_dir = os.path.join(results_dir, 'dimred_replacement_odt')
        os.makedirs(odt_output_dir, exist_ok=True)
    
    # Statistics tracking
    n_skipped_already_in_panel = 0
    n_accepted_odt_pretested = 0
    n_accepted_odt_new_test = 0
    n_accepted_odt_skipped = 0
    n_failed_odt = 0
    
    for candidate in candidates:
        if len(replacements) >= n_needed:
            break
        
        # Skip if already in ANY part of the combined panel
        if candidate in all_excluded or candidate in set(replacements):
            n_skipped_already_in_panel += 1
            continue
        
        # Get candidate metadata
        candidate_metadata = dimred_builder.get_gene_metadata(candidate)
        
        if candidate_metadata is None:
            logging.warning(f"No metadata for candidate gene {candidate}, skipping")
            continue

        if not apply_odt_filter:
            replacements.append(candidate)
            replacement_report.append({
                'replacement_gene': candidate,
                'component': candidate_metadata.component,
                'celltype': candidate_metadata.celltype,
                'odt_passed': None,
                'odt_skipped': True,
                'rank_in_candidates': candidates.index(candidate)
            })
            n_accepted_odt_skipped += 1
            logging.info(
                f"  ✓ Replacement {len(replacements)}/{n_needed}: {candidate} "
                f"(ODT disabled, component={candidate_metadata.component}, celltype={candidate_metadata.celltype})"
            )
            continue
        
        # Check if candidate was already ODT-tested in single strategies
        if candidate in all_odt_tested:
            # TRUST single-run ODT results - accept immediately without re-testing
            replacements.append(candidate)
            replacement_report.append({
                'replacement_gene': candidate,
                'component': candidate_metadata.component,
                'celltype': candidate_metadata.celltype,
                'odt_passed': True,
                'odt_pretested': True,  # Mark as pre-tested
                'rank_in_candidates': candidates.index(candidate)
            })
            n_accepted_odt_pretested += 1
            logging.info(
                f"  ✓ Replacement {len(replacements)}/{n_needed}: {candidate} "
                f"(ODT pre-tested, component={candidate_metadata.component}, celltype={candidate_metadata.celltype})"
            )
            continue
        
        # Candidate is NEW - needs ODT testing
        # Build factor/celltype-aware replacement pool
        remaining_candidates = [
            g for g in candidates[candidates.index(candidate)+1:] 
            if g not in all_excluded and g not in set(replacements)
        ]
        
        # Filter by component if available
        if candidate_metadata.component is not None:
            component_candidates = []
            for g in remaining_candidates:
                g_meta = dimred_builder.get_gene_metadata(g)
                if g_meta and g_meta.component == candidate_metadata.component:
                    component_candidates.append(g)
            
            if component_candidates:
                remaining_candidates = component_candidates
        
        # Filter by celltype if not global
        if candidate_metadata.celltype != 'global':
            celltype_candidates = []
            for g in remaining_candidates:
                g_meta = dimred_builder.get_gene_metadata(g)
                if g_meta and g_meta.celltype == candidate_metadata.celltype:
                    celltype_candidates.append(g)
            
            if celltype_candidates:
                remaining_candidates = celltype_candidates
        
        # Build component mapping for ODT
        component_mapping = {}
        if candidate_metadata.component is not None:
            component_mapping[candidate] = candidate_metadata.component
            for g in remaining_candidates:
                g_meta = dimred_builder.get_gene_metadata(g)
                if g_meta and g_meta.component is not None:
                    component_mapping[g] = g_meta.component
        
        # Build celltype mapping for ODT
        celltype_mapping = {candidate: candidate_metadata.celltype}
        for g in remaining_candidates:
            g_meta = dimred_builder.get_gene_metadata(g)
            if g_meta:
                celltype_mapping[g] = g_meta.celltype
        
        # Apply ODT check to NEW gene
        try:
            odt_result = apply_odt_filter_with_replacement(
                selected_genes=[candidate],
                all_genes_ranked=remaining_candidates,
                selected_genes_component_mapping=component_mapping,
                replacement_pool_component_mapping=component_mapping,
                selected_genes_celltype_mapping=celltype_mapping,
                replacement_pool_celltype_mapping=celltype_mapping,
                factor_aware=True,
                celltype_aware=(candidate_metadata.celltype != 'global'),
                odt_method=odt_method,
                odt_output_dir=odt_output_dir,
                species=odt_species,
                min_probes_threshold=odt_min_probes_threshold,
                max_iterations=100,
                results_dir=odt_output_dir
            )
            
            final_genes = odt_result.get('final_genes', [])
            
            if final_genes and candidate in final_genes:
                # Candidate passed ODT
                replacements.append(candidate)
                replacement_report.append({
                    'replacement_gene': candidate,
                    'component': candidate_metadata.component,
                    'celltype': candidate_metadata.celltype,
                    'odt_passed': True,
                    'odt_pretested': False,
                    'rank_in_candidates': candidates.index(candidate)
                })
                n_accepted_odt_new_test += 1
                logging.info(
                    f"  ✓ Replacement {len(replacements)}/{n_needed}: {candidate} "
                    f"(NEW ODT test passed, component={candidate_metadata.component}, celltype={candidate_metadata.celltype})"
                )
            
            elif final_genes and len(final_genes) > 0:
                # ODT replaced candidate with alternative
                replacement_gene = final_genes[0]
                replacements.append(replacement_gene)
                replacement_report.append({
                    'replacement_gene': replacement_gene,
                    'original_candidate': candidate,
                    'component': candidate_metadata.component,
                    'celltype': candidate_metadata.celltype,
                    'odt_passed': True,
                    'odt_replaced': True,
                    'odt_pretested': False,
                    'rank_in_candidates': candidates.index(candidate)
                })
                n_accepted_odt_new_test += 1
                logging.info(
                    f"  ✓ Replacement {len(replacements)}/{n_needed}: {replacement_gene} "
                    f"(NEW ODT replaced {candidate}, component={candidate_metadata.component})"
                )
            
            else:
                # Candidate failed ODT
                n_failed_odt += 1
                logging.info(
                    f"  ✗ Candidate {candidate} failed ODT (no suitable replacement found)"
                )
        
        except Exception as e:
            logging.error(f"Error during ODT check for {candidate}: {e}")
            n_failed_odt += 1
            continue
    
    # Log statistics
    logging.info(f"")
    logging.info(f"Replacement statistics:")
    if apply_odt_filter:
        logging.info(f"  Accepted (pre-tested): {n_accepted_odt_pretested}")
        logging.info(f"  Accepted (new tests): {n_accepted_odt_new_test}")
    else:
        logging.info(f"  Accepted (ODT disabled): {n_accepted_odt_skipped}")
    logging.info(f"  Skipped (already in panel): {n_skipped_already_in_panel}")
    if apply_odt_filter:
        logging.info(f"  Failed ODT: {n_failed_odt}")
    
    return replacements, replacement_report


def _apply_gap_filling(
    base_builder: GeneListBuilder,
    rf_builder: GeneListBuilder,
    dimred_builder: GeneListBuilder,
    adata: AnnData,
    gap_needed: int,
    target_size: int,
    run_celltype_filling: bool,
    run_global_filling: bool,
    run_deg_filling: bool,
    reduction_type: str,
    analysis_type: str,
    dimred_method: str,
    n_components: int,
    results_dir: str
) -> GeneListBuilder:
    """
    Apply gap-filling strategies to reach target panel size.
    
    Tries strategies in priority order until target size is reached.
    Priority order is determined internally based on boolean flags and analysis_type.
    
    Parameters
    ----------
    base_builder : GeneListBuilder
        Combined panel (before gap filling)
    rf_builder : GeneListBuilder
        RF component results
    dimred_builder : GeneListBuilder
        Dimred component results
    adata : AnnData
        Annotated data matrix
    gap_needed : int
        Number of genes needed to reach target
    target_size : int
        Target panel size
    run_celltype_filling : bool
        Enable celltype-specific filling
    run_global_filling : bool
        Enable global gene filling
    run_deg_filling : bool
        Enable DEG-based filling
    reduction_type : str
        'nmf' or 'pca'
    analysis_type : str
        'per_celltype' or 'global'
    dimred_method : str
        'method_a' (top genes by absolute factor loading)
    n_components : int
        Number of components
    results_dir : str
        Results directory
        
    Returns
    -------
    GeneListBuilder
        Final panel after gap filling
    """
    
    # Build gap-filling strategy priority order based on boolean flags and analysis_type
    gap_fill_priority = []
    
    if analysis_type == 'per_celltype':
        # Per-celltype analysis: prefer celltype-specific, then DEG-based
        if run_celltype_filling:
            gap_fill_priority.append(GAP_FILL_STRATEGY_CELLTYPE)
        if run_deg_filling:
            gap_fill_priority.append(GAP_FILL_STRATEGY_DEG)
        # Ignore run_global_filling for per-celltype (incompatible)
        if run_global_filling:
            logging.warning(
                f"Global gap-filling is incompatible with per-celltype analysis (analysis_type='{analysis_type}'). "
                "Ignoring run_global_filling=True."
            )
    elif analysis_type == 'global':
        # Global analysis: prefer global, then DEG-based
        if run_global_filling:
            gap_fill_priority.append(GAP_FILL_STRATEGY_GLOBAL)
        if run_deg_filling:
            gap_fill_priority.append(GAP_FILL_STRATEGY_DEG)
        # Ignore run_celltype_filling for global (incompatible)
        if run_celltype_filling:
            logging.warning(
                f"Celltype-specific gap-filling is incompatible with global analysis (analysis_type='{analysis_type}'). "
                "Ignoring run_celltype_filling=True."
            )
    else:
        raise ValueError(f"Unknown analysis_type: '{analysis_type}'. Expected 'per_celltype' or 'global'.")
    
    if not gap_fill_priority:
        logging.warning("All gap-filling strategies disabled. Panel may be smaller than target size if gaps exist.")
    
    logging.info("Applying gap-filling strategies...")
    logging.info(f"Priority order: {' → '.join(gap_fill_priority)}")
    
    current_builder = base_builder
    current_genes = set(current_builder.get_all_genes())
    genes_still_needed = gap_needed
    
    for strategy in gap_fill_priority:
        if genes_still_needed <= 0:
            break
        
        logging.info("")
        logging.info(f"Trying gap-filling strategy: {strategy}")
        logging.info(f"Genes still needed: {genes_still_needed}")
        
        if strategy == GAP_FILL_STRATEGY_DEG and run_deg_filling:
            # DEG-based filling: Use additional RF genes
            gap_genes = _fill_gap_with_deg(
                current_genes=current_genes,
                rf_builder=rf_builder,
                n_needed=genes_still_needed,
                results_dir=results_dir
            )
            gene_source_label = GENE_SOURCE_GAP_FILL_DEG
        
        elif strategy == GAP_FILL_STRATEGY_CELLTYPE and run_celltype_filling:
            # Cell-type-specific filling: Use per-celltype dimred genes
            gap_genes = _fill_gap_with_celltype_dimred(
                current_genes=current_genes,
                dimred_builder=dimred_builder,
                adata=adata,
                n_needed=genes_still_needed,
                analysis_type=analysis_type,
                reduction_type=reduction_type,
                dimred_method=dimred_method,
                n_components=n_components,
                results_dir=results_dir
            )
            gene_source_label = GENE_SOURCE_GAP_FILL_CELLTYPE
        
        elif strategy == GAP_FILL_STRATEGY_GLOBAL and run_global_filling:
            # Global gene filling: Use global dimred genes
            gap_genes = _fill_gap_with_global_dimred(
                current_genes=current_genes,
                adata=adata,
                n_needed=genes_still_needed,
                reduction_type=reduction_type,
                dimred_method=dimred_method,
                n_components=n_components,
                results_dir=results_dir
            )
            gene_source_label = GENE_SOURCE_GAP_FILL_GLOBAL
        
        else:
            logging.warning(f"Gap-filling strategy '{strategy}' not enabled or not recognized, skipping")
            continue
        
        # Add gap-fill genes to builder
        if gap_genes:
            logging.info(f"Gap-filling strategy '{strategy}' provided {len(gap_genes)} genes")
            
            for gene in gap_genes:
                if gene not in current_genes:
                    current_builder.add_gene(
                        gene_name=gene,
                        gene_source=gene_source_label,
                        rank=None,
                        selection_score=None,
                        celltype='global',  # Gap-fill genes treated as global
                        component=None,
                        additional_metadata={'gap_filled': True, 'gap_fill_strategy': strategy}
                    )
                    current_genes.add(gene)
                    genes_still_needed -= 1
                    
                    if genes_still_needed <= 0:
                        break
            
            logging.info(f"Added {len(gap_genes)} gap-fill genes from '{strategy}'")
            logging.info(f"Genes still needed: {genes_still_needed}")
        else:
            logging.warning(f"Gap-filling strategy '{strategy}' found no suitable genes")
    
    if genes_still_needed > 0:
        logging.warning(
            f"Could not fill entire gap: {genes_still_needed} genes still needed after all strategies. "
            f"Final size: {len(current_genes)} (target: {target_size})"
        )
    
    current_builder.add_metadata('gap_filling_applied', True)
    current_builder.add_metadata('gap_filled_count', gap_needed - genes_still_needed)
    current_builder.add_metadata('gap_remaining', genes_still_needed)
    
    return current_builder


def _fill_gap_with_deg(
    current_genes: Set[str],
    rf_builder: GeneListBuilder,
    n_needed: int,
) -> List[str]:
    """
    DEG-based gap filling: Use next-best RF genes.
    
    This is NOT for force-include genes. It's for filling the gap when the
    combined panel (force-include + RF + dimred) is smaller than target size.
    
    Uses the full ranked list of RF genes from Phase 1, selecting genes that
    weren't already included in the initial RF allocation or as dimred genes.
    """
    
    logging.info("DEG-based gap filling: Using next-best RF genes")
    logging.info("Reusing ranked genes from Phase 1 rf_builder (no recomputation needed)")
    
    # Get all RF genes ranked
    all_rf_genes = rf_builder.get_all_genes()
    
    # Find genes not in current panel
    candidates = [g for g in all_rf_genes if g not in current_genes]
    
    # Take top N candidates
    gap_genes = candidates[:n_needed]
    
    logging.info(f"Found {len(gap_genes)} DEG-based gap-fill genes")
    
    return gap_genes


def _fill_gap_with_celltype_dimred(
    current_genes: Set[str],
    dimred_builder: GeneListBuilder,
    n_needed: int,
    analysis_type: str,
) -> List[str]:
    """
    Cell-type-specific gap filling: Use per-celltype dimred genes.
    
    This reuses genes from dimred_builder (Phase 2 results) that weren't selected
    in the initial combination. No need to re-run dimred since we already have
    the full ranked list from per-celltype analysis.
    """
    
    logging.info("Cell-type-specific gap filling: Using per-celltype dimred genes")
    logging.info("Reusing ranked genes from Phase 2 dimred_builder (no recomputation needed)")
    
    if analysis_type != 'per_celltype':
        logging.warning("Cannot use celltype-specific filling with global analysis type")
        return []
    
    # Get all dimred genes ranked
    all_dimred_genes = dimred_builder.get_all_genes()
    
    # Find genes not in current panel
    candidates = [g for g in all_dimred_genes if g not in current_genes]
    
    # Take top N candidates
    gap_genes = candidates[:n_needed]
    
    logging.info(f"Found {len(gap_genes)} celltype-specific gap-fill genes")
    
    return gap_genes


def _fill_gap_with_global_dimred(
    current_genes: Set[str],
    adata: AnnData,
    n_needed: int,
    reduction_type: str,
    dimred_method: str,
    n_components: int,
    results_dir: str
) -> List[str]:
    """
    Global gene gap filling: Run global dimred on-the-fly.
    
    NOTE: Unlike celltype-specific filling (which reuses the dimred_builder from
    Phase 2), global filling MUST run a new dimred analysis because:
    - If initial analysis was per-celltype, we don't have global dimred results
    - If initial analysis was global, those genes are already in the panel
    - Gap-filling needs NEW genes not yet selected
    
    Future optimization: Cache global dimred results across runs for reuse.
    """
    
    logging.info("Global gene gap filling: Running global dimred analysis")
    logging.info("NOTE: Re-running global dimred because initial analysis may have been per-celltype")
    
    # Import dimred selection
    from ._dimred import run_dimred_and_select_genes
    
    # Run global dimred
    global_results = run_dimred_and_select_genes(
        adata=adata,
        reduction_type=reduction_type,
        analysis_type='global',
        dimred_method=dimred_method,
        n_components=n_components,
        probeset_size=n_needed * 2,  # Oversample
        results_dir=os.path.join(results_dir, 'global_dimred_gap_fill')
    )
    
    # Get candidate genes
    if isinstance(global_results, GeneListBuilder):
        candidates = global_results.get_all_genes()
    elif isinstance(global_results, dict):
        candidates = global_results.get('selected_genes', [])
    else:
        candidates = []
    
    # Filter out genes already in panel
    gap_genes = [g for g in candidates if g not in current_genes][:n_needed]
    
    logging.info(f"Found {len(gap_genes)} global dimred gap-fill genes")
    
    return gap_genes


def _save_combination_results(
    rf_builder: GeneListBuilder,
    dimred_builder: GeneListBuilder,
    combined_builder: GeneListBuilder,
    results_dir: str,
    strategy: str,
    reduction_type: str
) -> None:
    """
    Save combination strategy results.

    Output files:
    1. ranked_gene_list.csv   - All combined genes with full metadata;
                                panel genes (in_panel=True) first, ranked by score.
                                Component genes not selected into the panel follow below.
                                Note: the per-component ranked lists (which include ALL
                                Xenium-passing genes from each component) are already saved
                                by run_single_selection inside rf_component/ and
                                {reduction}_component/ subdirectories.
    2. combination_summary.json - Summary statistics including duplicate-resolution counts.
    """

    os.makedirs(results_dir, exist_ok=True)

    logging.info("")
    logging.info("=" * 80)
    logging.info("SAVING COMBINATION RESULTS")
    logging.info("=" * 80)

    # 1. Unified ranked gene list -------------------------------------------
    #    All genes in the combined builder, panel genes first.
    #    The combination builder contains genes from both components that survived
    #    Xenium (and ODT if enabled) filtering in their respective single runs.
    combined_df = combined_builder.to_dataframe()

    # Add in_panel indicator as the second column
    combined_df.insert(1, 'in_panel', combined_df['final_selection'].fillna(False))

    # Panel genes first, then remaining; within each group sort by rank
    combined_df = combined_df.sort_values(
        ['in_panel', 'rank'],
        ascending=[False, True],
    ).reset_index(drop=True)

    ranked_output = os.path.join(results_dir, 'ranked_gene_list.csv')
    combined_df.to_csv(ranked_output, index=False)
    n_panel  = int(combined_df['in_panel'].sum())
    n_remain = len(combined_df) - n_panel
    logging.info(
        f"✓ Saved ranked_gene_list.csv: "
        f"{n_panel} panel genes + {n_remain} remaining genes ({len(combined_df)} total)"
    )

    # 2. Combination summary JSON -------------------------------------------
    # Count duplicate-resolution events from in-memory replacement report
    # (stored in combined_builder metadata if available).
    n_replacements = combined_builder.metadata.get('n_dimred_replacements', 0)

    summary = {
        'strategy': strategy,
        'reduction_type': reduction_type,
        'rf_component_genes': len(rf_builder.get_all_genes()),
        'dimred_component_genes': len(dimred_builder.get_all_genes()),
        'combined_panel_size': n_panel,
        'n_overlapping_genes': combined_builder.metadata.get('n_overlapping', 0),
        'n_dimred_replacements': n_replacements,
        'gap_filling_applied': combined_builder.metadata.get('gap_filling_applied', False),
        'gap_filling_strategy': combined_builder.metadata.get('gap_filling_strategy', 'none'),
        'metadata': combined_builder.metadata,
    }

    import json
    summary_path = os.path.join(results_dir, 'combination_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"✓ Saved combination_summary.json")

    logging.info(f"✓ All combination results saved to: {results_dir}")


