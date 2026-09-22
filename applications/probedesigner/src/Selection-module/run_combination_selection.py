"""
Combination gene selection strategies (RecoVar, RecoVar_PCA).

This module orchestrates combination strategies that merge genes from two sources:
1. Random Forest on DEG-filtered genes (rf_deg)
2. Dimensionality reduction (NMF or PCA) - global or per-celltype

The workflow:
- Run both components WITH full filtering (Xenium)
- Combine filtered results with ratio-based selection
- Handle duplicates by assigning to RF pool and replacing from dimred pool
- Support 2 gap-filling strategies (celltype-specific dimred, DEG-based) to reach target size
- Maintain factor/component and cell type awareness throughout
"""

from __future__ import annotations

import logging
import os
import json
import re
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
from anndata import AnnData

# Support script execution by adding module directory to sys.path
import sys
MODULE_DIR = Path(__file__).parent.absolute()
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

# Import run_single_selection (sys.path already includes MODULE_DIR above)
from run_single_selection import run_single_selection

# Use absolute imports (for script execution)
# --- load THIS directory's _constants.py by path (sibling dirs share the name) ---
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _cpath
_cspec = _ilu.spec_from_file_location("_constants", _cpath(__file__).resolve().parent / "_constants.py")
_sys.modules["_constants"] = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_sys.modules["_constants"])


from _constants import (
    COL_CELLTYPE,
    DEFAULT_PROBESET_SIZE,
    DEFAULT_REDUCTION_TYPE,
    DEFAULT_DIMRED_PERCENTAGE,
    DEFAULT_RF_PERCENTAGE,
    DEFAULT_RUN_CELLTYPE_FILLING,
    DEFAULT_RUN_DEG_FILLING,
    COMBINATION_OVERSAMPLE_FACTOR,
    GAP_FILL_STRATEGY_CELLTYPE,
    GAP_FILL_STRATEGY_DEG,
    GAP_FILL_STRATEGY_DISPLAY_NAMES,
    GENE_SOURCE_DIMRED,
    GENE_SOURCE_FORCE_INCLUDE,
    GENE_SOURCE_GAP_FILL_CELLTYPE,
    GENE_SOURCE_GAP_FILL_DEG,
    GENE_SOURCE_OVERLAP_TO_RF,
    GENE_SOURCE_RF,
    DEFAULT_MIN_XENIUM_EXPRESSION,
    DEFAULT_MAX_XENIUM_EXPRESSION,
    RF_CONTRIBUTING_CELLTYPE_MIN_SHARE,
)
from _gene_list_builder import GeneListBuilder, panel_information_filename


def run_combination_selection(
    strategy: str,
    adata: AnnData,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    reduction_type: str = DEFAULT_REDUCTION_TYPE,
    n_components: int = 50,
    pool_size_per_celltype: int = 200,  # Pool size per celltype for Phase 1
    dimred_n_jobs: int = 1,  # workers for the dimred component's per-celltype NMF/PCA fits
    rf_percentage: float = DEFAULT_RF_PERCENTAGE,
    # dimred_percentage is DISPLAY-ONLY: the RF/dimred split is
    # driven entirely by rf_percentage — n_dimred_target = adjusted_target_size - n_rf_target.
    # This value is only checked for consistency (must sum to 1.0 ± 0.01 with rf_percentage)
    # and echoed into metadata/logs; it never enters the split maths, so 0.30/0.70 and
    # 0.30/0.6999 produce the identical panel. Kept for a readable, self-documenting CLI.
    dimred_percentage: float = DEFAULT_DIMRED_PERCENTAGE,
    force_include_genes: Optional[List[str]] = None,
    blacklist_patterns: Optional[List[str]] = None,
    use_default_blacklist: bool = True,
    apply_xenium_filter: bool = True,
    xenium_celltype_aware: bool = True,
    xenium_min_expr: float = DEFAULT_MIN_XENIUM_EXPRESSION,
    xenium_max_expr: float = DEFAULT_MAX_XENIUM_EXPRESSION,

    # Gap-filling control (two boolean flags)
    run_celltype_filling: bool = DEFAULT_RUN_CELLTYPE_FILLING,
    run_deg_filling: bool = DEFAULT_RUN_DEG_FILLING,
    
    results_dir: Optional[str] = None,
    experiment_name: str = 'combination_selection',
    # Caching support (reuse pre-computed component results)
    rf_deg_cache_dir: Optional[str] = None,
    dimred_cache_dir: Optional[str] = None,
    nmf_model_cache_dir: Optional[str] = None,
    force_recompute: bool = False,
    # Diagnostic plots -- both opt-in only: skipped entirely (never written into
    # results_dir) when their directory isn't explicitly given.
    dotplot_dir: Optional[str] = None,
    expression_distribution_dir: Optional[str] = None,
    **kwargs
) -> GeneListBuilder:
    """
    Run combination gene selection strategy (RecoVar or RecoVar_PCA).
    
    This function orchestrates a two-phase workflow:
    1. Run RF (rf_deg) and dimred (dimred_only) strategies independently with FULL filtering
       OR load from cache if available
    2. Combine filtered results with ratio-based selection and duplicate resolution
    3. Apply gap-filling if combined panel < target size (duplicate resolution ≠ gap-filling)
    
    IMPORTANT DISTINCTION:
    - **Duplicate resolution**: When a gene is selected by both RF and dimred, assign it to RF pool.
      This may REDUCE the dimred count, creating a shortfall that must be filled from next-best
      dimred candidates.
    - **Gap-filling**: When the combined panel (force-include + RF + dimred) is SMALLER than target
      size, add genes from additional sources (celltype-specific dimred or DEG-based).
    
    These are separate operations: duplicate resolution maintains the RF/dimred ratio by replacing
    lost dimred genes, while gap-filling increases total panel size when needed.
    
    Force-include genes are added FIRST, then remaining slots are filled with RF/dimred genes
    according to the specified ratio. Duplicates are assigned to the RF pool, and replacement
    dimred genes are selected with factor/celltype awareness.
    
    If the combined panel is smaller than target size, gap-filling strategies are applied.
    
    Parameters
    ----------
    strategy : str
        Combination strategy name: 'RecoVar' or 'RecoVar_PCA'
    adata : AnnData
        Annotated data matrix with expression data
    probeset_size : int
        Target total number of genes in final panel
    reduction_type : str
        Dimensionality reduction type: 'nmf' or 'pca'
    n_components : int, default=5
        Number of components for dimensionality reduction
    pool_size_per_celltype : int, default=200
        Phase-1 dimred pool size per cell type. Forwarded to the dimred component.
    dimred_n_jobs : int, default=1
        Workers for the dimred component's per-celltype NMF/PCA fits (1 = sequential,
        -1 = all cores). Forwarded to run_single_selection(strategy='dimred_only').
    rf_percentage : float, default=0.25
        Target fraction of genes from RF (e.g., 0.25 = 25%). This is the *only* knob that
        moves the split: n_rf_target = int(adjusted_target_size * rf_percentage) and dimred
        takes the remainder.
    dimred_percentage : float, default=0.75
        Display-only. Must equal 1 - rf_percentage within 0.01 or
        the run raises; it is echoed to metadata/logs but never enters the split maths, so it
        cannot by itself change the panel. For a 100% RF or 100% dimred panel use the
        `rf_simple` / `rf_deg` or `dimred_only` single strategies instead — this orchestrator
        rejects `rf_percentage`/`dimred_percentage` of 0 or 1.
    force_include_genes : List[str], optional
        Genes to force-include (highest priority, added before ratio calculation)
    blacklist_patterns : List[str], optional
        Gene name prefixes to exclude (e.g., ['mt-', 'rps', 'rpl'])
    use_default_blacklist : bool, default=True
        Also apply the built-in default blacklist patterns alongside
        ``blacklist_patterns``.
    apply_xenium_filter : bool, default=True
        Whether to apply Xenium expression filtering
    xenium_celltype_aware : bool, default=True
        Use per-cell-type mean expression for the Xenium filter; when False, use a
        single pooled global mean.
    xenium_min_expr : float, default=0.1
        Minimum mean expression threshold for Xenium filter
    xenium_max_expr : float, default=100.0
        Maximum mean expression threshold for Xenium filter
    run_celltype_filling : bool, default=True
        Enable cell-type-specific dimred gap-filling (per-celltype analysis)
    run_deg_filling : bool, default=True
        Enable DEG-based gap-filling (works for both analysis types)
    results_dir : str, optional
        Directory to save results
    experiment_name : str, default='combination_selection'
        Name for this experiment (used in result filenames)
    rf_deg_cache_dir : str, optional
        Directory containing cached rf_deg results (ranked_gene_list.csv).
        If provided and exists, will load cached results instead of recomputing.
        Expected structure: rf_deg_cache_dir/ranked_gene_list.csv
    dimred_cache_dir : str, optional
        Directory containing cached dimred_only results (ranked_gene_list.csv).
        If provided and exists, will load cached results instead of recomputing.
        Expected structure: dimred_cache_dir/ranked_gene_list.csv
    nmf_model_cache_dir : str, optional
        Shared directory for NMF/PCA model pkl files, forwarded to the dimred
        component so one fit can be reused across probeset sizes / strategies.
    force_recompute : bool, default=False
        If True, ignore cached results and recompute both components.
        Use this to regenerate results even when cache exists.
    **kwargs
        Additional parameters passed to component strategies (notably
        ``dimred_counts_input`` — 'raw' or 'lognorm' — routed to the dimred
        component's NMF/PCA input).
        
    Returns
    -------
    GeneListBuilder
        Final gene panel with complete metadata including:
        - gene_source: Labels like 'rf_deg', 'dimred', 'overlap→rf_deg',
          'gap_fill_celltype', 'gap_fill_deg', 'force_include'
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
    logging.info(f"Reduction type: {reduction_type.upper()} (per cell type)")
    logging.info(f"Target composition: {rf_percentage:.0%} RF + {dimred_percentage:.0%} {reduction_type.upper()}")
    def _read_cache_filter_flags(cache_dir: str) -> tuple[Optional[bool], Optional[bool], Optional[list]]:
        """Read filter flags from filtering_summary.json if present.

        Returns (xenium_flag, blacklist_any_active_flag, blacklist_custom_patterns)
        Any value is None if the summary file is missing or the field is absent.
        """
        summary_file = os.path.join(cache_dir, 'filtering_summary.json')
        if not os.path.exists(summary_file):
            return None, None, None

        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            filters = summary.get('filters_applied', {})
            return (
                filters.get('xenium'),
                filters.get('blacklist_any_active'),
                filters.get('blacklist_custom_patterns'),
            )
        except Exception as e:
            logging.warning(f"Could not parse cache summary {summary_file}: {e}")
            return None, None, None
    
    # Validate strategy
    if strategy not in ['RecoVar', 'RecoVar_PCA']:
        raise ValueError(f"Invalid combination strategy: {strategy}. Must be 'RecoVar' or 'RecoVar_PCA'")
    
    # Validate reduction type matches strategy.
    # 'RecoVar' is the NMF hybrid (renamed from 'rf_nmf'); 'RecoVar_PCA' the PCA one.
    # (Substring checks like "'nmf' in strategy" broke with the rename -- 'nmf' is
    # not a substring of 'RecoVar'.)
    expected_reduction = 'pca' if strategy == 'RecoVar_PCA' else 'nmf'
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
    
    # Split is a function of rf_percentage ONLY — dimred_percentage is display-only (#26).
    # int() truncates toward zero, so a fractional RF share always rounds *down*
    # (e.g. probeset_size=250, rf_percentage=0.25 -> n_rf_target=62, dimred gets 188).
    # This is deliberate: dimred is the larger, primary component and absorbs the
    # remainder so the two always sum to adjusted_target_size exactly.
    n_rf_target = int(adjusted_target_size * rf_percentage)
    n_dimred_target = adjusted_target_size - n_rf_target  # exact sum by construction
    
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

    if rf_deg_cache_dir and not force_recompute:
        # Look for rf_deg_panel_information.csv (current name), falling back to the
        # older ranked_gene_list.csv / ranked_gene_list_final.csv names for pre-rename
        # cache directories (see docs/doc-pipeline/audit_3.md, "Selection-module CSV
        # output reorganization").
        rf_cache_file = os.path.join(rf_deg_cache_dir, panel_information_filename('rf_deg'))
        if not os.path.exists(rf_cache_file):
            rf_cache_file = os.path.join(rf_deg_cache_dir, 'ranked_gene_list.csv')
        if not os.path.exists(rf_cache_file):
            rf_cache_file = os.path.join(rf_deg_cache_dir, 'ranked_gene_list_final.csv')

        if os.path.exists(rf_cache_file):
            logging.info(f"Found cached RF ranked list: {rf_cache_file}")
            logging.info("Loading cached rf_deg results instead of recomputing...")
            try:
                # Load ranked list (with filter results)
                rf_cache_df = pd.read_csv(rf_cache_file)

                rf_cache_xenium_flag, rf_cache_blacklist_flag, rf_cache_bl_patterns = _read_cache_filter_flags(rf_deg_cache_dir)
                if apply_xenium_filter and rf_cache_xenium_flag is False:
                    raise ValueError(
                        "RF cache was generated with Xenium disabled, but this run requires Xenium-cleaned component cache"
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

                # Support new column name ('in_panel'/'final_selection') and old ('selected').
                _rf_sel_col = next(
                    (c for c in ('in_panel', 'final_selection', 'selected') if c in rf_cache_df.columns),
                    None,
                )
                if _rf_sel_col is not None:
                    selected_cache_rf = set(rf_cache_df[rf_cache_df[_rf_sel_col] == True]['gene'].tolist())
                    logging.info(
                        f"✓ Loaded {len(selected_cache_rf)} RF genes from ranked list"
                        + (" (Xenium-cleaned)" if apply_xenium_filter else "")
                    )
                else:
                    logging.warning(
                        f"Cached RF results missing selection column: {rf_cache_file}\n"
                        "Cache appears to be in old format or incomplete. Will regenerate from scratch."
                    )
                    rf_builder = None
                    # Skip remaining cache loading and trigger from-scratch generation
                    raise ValueError("Cache format invalid")  # Caught by outer try/except
                
                # Create builder from cached data
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
                        additional_metadata={
                            # None-safe for pre-upgrade caches without these columns.
                            'rf_celltype': row.get('rf_celltype', '') or '',
                            'rf_contributing_celltypes': row.get('rf_contributing_celltypes', '') or '',
                            'rf_celltype_scores': row.get('rf_celltype_scores', '{}') or '{}',
                        },
                    )
                
                logging.info(f"✓ Loaded {len(rf_builder.get_all_genes())} RF genes from ranked list")
                
            except Exception as e:
                logging.warning(f"Failed to load RF cache: {e}")
                logging.warning("Will recompute RF component")
                rf_builder = None
    
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
            xenium_min_expr=xenium_min_expr,
            xenium_max_expr=xenium_max_expr,
            results_dir=rf_results_dir,
            **kwargs
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
    
    if dimred_cache_dir and not force_recompute:
        # Look for {nmf,pca}_panel_information.csv (current name), falling back to the
        # older ranked_gene_list.csv / ranked_gene_list_final.csv names for pre-rename
        # cache directories.
        dimred_cache_file = os.path.join(
            dimred_cache_dir, panel_information_filename('dimred_only', reduction_type)
        )
        if not os.path.exists(dimred_cache_file):
            dimred_cache_file = os.path.join(dimred_cache_dir, 'ranked_gene_list.csv')
        if not os.path.exists(dimred_cache_file):
            dimred_cache_file = os.path.join(dimred_cache_dir, 'ranked_gene_list_final.csv')

        if os.path.exists(dimred_cache_file):
            logging.info(f"Found cached dimred ranked list: {dimred_cache_file}")
            logging.info("Loading cached dimred_only results instead of recomputing...")
            try:
                # Load ranked list (with filter results)
                dimred_cache_df = pd.read_csv(dimred_cache_file)

                dimred_cache_xenium_flag, dimred_cache_blacklist_flag, dimred_cache_bl_patterns = _read_cache_filter_flags(dimred_cache_dir)
                if apply_xenium_filter and dimred_cache_xenium_flag is False:
                    raise ValueError(
                        "Dimred cache was generated with Xenium disabled, but this run requires Xenium-cleaned component cache"
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

                # Support new column name ('in_panel'/'final_selection') and old ('selected').
                _dim_sel_col = next(
                    (c for c in ('in_panel', 'final_selection', 'selected') if c in dimred_cache_df.columns),
                    None,
                )
                if _dim_sel_col is not None:
                    selected_cache_dimred = set(dimred_cache_df[dimred_cache_df[_dim_sel_col] == True]['gene'].tolist())
                    logging.info(
                        f"✓ Loaded {len(selected_cache_dimred)} dimred genes from ranked list"
                        + (" (Xenium-cleaned)" if apply_xenium_filter else "")
                    )
                else:
                    logging.warning(
                        f"Cached dimred results missing selection column: {dimred_cache_file}\n"
                        "Cache appears to be in old format or incomplete. Will regenerate from scratch."
                    )
                    dimred_builder = None
                    # Skip remaining cache loading and trigger from-scratch generation
                    raise ValueError("Cache format invalid")  # Caught by outer try/except
                
                # Create builder from cached data
                dimred_builder = GeneListBuilder(
                    strategy_name='dimred_only',
                    analysis_type='per_celltype',
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
                        additional_metadata={
                            'component_loading': row.get('component_loading', None),
                            # Restore per-cell-type attribution so cached dimred reuse
                            # still feeds the coverage diagnostic and gap-fill lookups.
                            'n_celltypes_selected': row.get('n_celltypes_selected', 1),
                            'contributing_celltypes': row.get('contributing_celltypes', '') or '',
                        },
                    )
                
                logging.info(f"✓ Loaded {len(dimred_builder.get_all_genes())} dimred genes from ranked list")
                
            except Exception as e:
                logging.warning(f"Failed to load dimred cache: {e}")
                logging.warning("Will recompute dimred component")
                dimred_builder = None
    
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
            n_components=n_components,
            pool_size_per_celltype=pool_size_per_celltype,
            dimred_n_jobs=dimred_n_jobs,
            force_include_genes=None,  # Don't double-count force-include in components
            blacklist_patterns=_sub_blacklist_patterns,
            use_default_blacklist=_sub_use_default_blacklist,
            apply_xenium_filter=apply_xenium_filter,
            xenium_celltype_aware=xenium_celltype_aware,
            xenium_min_expr=xenium_min_expr,
            xenium_max_expr=xenium_max_expr,
            results_dir=dimred_results_dir,
            nmf_model_cache_dir=_resolved_nmf_cache,
            **kwargs
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
    
    combined_builder = _combine_filtered_components(
        rf_builder=rf_builder,
        dimred_builder=dimred_builder,
        n_rf_target=n_rf_target,
        n_dimred_target=n_dimred_target,
        force_include_set=force_include_set,
        reduction_type=reduction_type,
        adata=adata,
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
            gap_needed=gap_needed,
            target_size=probeset_size,
            run_celltype_filling=run_celltype_filling,
            run_deg_filling=run_deg_filling,
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

    # Combined-panel size status — parity with run_single_selection's filtering_summary.json.
    # The per-component filtering_summary.json files carry panel_size_status for the RF and
    # dimred halves; this records it for the combined panel.
    if final_size >= probeset_size:
        panel_size_status = 'complete'
        short_panel_reason = None
    else:
        panel_size_status = 'short'
        gap_remaining = combined_builder.metadata.get('gap_remaining')
        short_panel_reason = (
            'gap_fill_pool_exhausted' if gap_remaining
            else 'insufficient_component_genes_after_filtering'
        )
    combined_builder.add_metadata('requested_panel_size', probeset_size)
    combined_builder.add_metadata('final_panel_size', final_size)
    combined_builder.add_metadata('panel_size_status', panel_size_status)
    combined_builder.add_metadata('short_panel_reason', short_panel_reason)
    
    # Mark all genes as selected — every gene that made it into the combined
    # builder has passed Xenium filtering inside its component, so all are
    # final panel members.
    combined_builder.mark_selected(final_genes, 'initial')
    combined_builder.mark_selected(final_genes, 'final')

    # Add combination metadata
    combined_builder.add_metadata('strategy', strategy)
    combined_builder.add_metadata('rf_percentage', rf_percentage)
    combined_builder.add_metadata('dimred_percentage', dimred_percentage)
    combined_builder.add_metadata('reduction_type', reduction_type)
    combined_builder.add_metadata('analysis_type', 'per_celltype')
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

    _celltype_column = kwargs.get('celltype_column', COL_CELLTYPE)
    if results_dir and dotplot_dir:
        _generate_combination_dotplot(
            adata=adata,
            results_dir=results_dir,
            celltype_column=_celltype_column,
            output_dir=dotplot_dir,
            reduction_type=reduction_type,
        )
    elif results_dir:
        logging.info(
            "Skipping final-gene dotplot: no dotplot_dir given (this plot is opt-in "
            "only -- it is never written into results_dir)"
        )

    if results_dir and expression_distribution_dir:
        _generate_expression_distribution_plot(
            adata=adata,
            results_dir=results_dir,
            output_dir=expression_distribution_dir,
        )
    elif results_dir:
        logging.info(
            "Skipping panel expression-distribution plot: no expression_distribution_dir "
            "given (this plot is opt-in only -- it is never written into results_dir)"
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
    reduction_type: str,
    adata: AnnData,
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
    reduction_type : str
        'nmf' or 'pca'
    adata : AnnData
        Annotated data matrix (for force-include validation)
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

    if force_include_set:
        logging.info(f"")
        logging.info(f"Processing {len(force_include_set)} force-include genes...")
    
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
    dimred_genes_selected = set(dimred_genes_filtered[:n_dimred_target])

    logging.info(f"Selected {len(rf_genes_selected)} RF genes (target: {n_rf_target})")
    logging.info(f"Selected {len(dimred_genes_selected)} dimred genes (target: {n_dimred_target})")

    # Identify overlapping genes; RF has priority
    overlapping_genes = rf_genes_selected & dimred_genes_selected
    rf_final = rf_genes_selected
    dimred_unique = dimred_genes_selected - overlapping_genes

    logging.info(f"")
    logging.info(f"Duplicate analysis:")
    logging.info(f"  Overlapping genes: {len(overlapping_genes)}")
    logging.info(f"  RF-unique genes: {len(rf_genes_selected - overlapping_genes)}")
    logging.info(f"  Dimred-unique genes: {len(dimred_unique)}")

    # Genes lost to RF overlap create a gap; gap-filling (Step 6) fills it.
    # No replacement here — gap-fill prefers multi-CT genes via (n_celltypes, loading) order.
    dimred_final = dimred_unique
    if len(dimred_final) < n_dimred_target:
        logging.info(
            f"Dimred shortfall: {n_dimred_target - len(dimred_final)} genes lost to RF overlaps. "
            f"Will be filled by gap-filling (Step 6)."
        )
    
    # Build combined gene list with metadata.
    # 'nmf' -> 'RecoVar', 'pca' -> 'RecoVar_PCA' (matches the --strategy choices).
    combined_strategy_name = 'RecoVar' if reduction_type.lower() == 'nmf' else f'RecoVar_{reduction_type.upper()}'
    combined_builder = GeneListBuilder(
        strategy_name=combined_strategy_name,
        analysis_type='per_celltype',
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
        rf_record = rf_builder.gene_records.get(gene, {})

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
                'original_rf_rank': rf_metadata.rank if rf_metadata else None,
                # Per-class RF attribution ("from the random forest itself").
                'rf_celltype': rf_record.get('rf_celltype', ''),
                'rf_contributing_celltypes': rf_record.get('rf_contributing_celltypes', ''),
                'rf_celltype_scores': rf_record.get('rf_celltype_scores', '{}'),
            }
        )
    
    # Add dimred genes
    for gene in dimred_final:
        dimred_metadata = dimred_builder.get_gene_metadata(gene)
        dimred_record = dimred_builder.gene_records.get(gene, {})

        combined_builder.add_gene(
            gene_name=gene,
            gene_source=GENE_SOURCE_DIMRED,
            rank=dimred_metadata.rank if dimred_metadata else None,
            selection_score=dimred_metadata.selection_score if dimred_metadata else None,
            celltype=dimred_metadata.celltype if dimred_metadata else 'global',
            component=dimred_metadata.component if dimred_metadata else None,
            additional_metadata={
                'from_dimred': True,
                'original_dimred_rank': dimred_metadata.rank if dimred_metadata else None,
                'component_loading': dimred_metadata.component_loading if dimred_metadata else None,
                'n_celltypes_selected': dimred_record.get('n_celltypes_selected', 1),
                'contributing_celltypes': dimred_record.get('contributing_celltypes', ''),
            }
        )
    
    # Add combination metadata
    combined_builder.add_metadata('n_force_include', len(force_include_set))
    combined_builder.add_metadata('n_rf_final', len(rf_final))
    combined_builder.add_metadata('n_dimred_final', len(dimred_final))
    combined_builder.add_metadata('n_overlapping', len(overlapping_genes))
    combined_builder.add_metadata('overlapping_genes', list(overlapping_genes))

    total_genes = len(force_include_set) + len(rf_final) + len(dimred_final)
    logging.info(f"")
    logging.info(f"Combined panel summary:")
    logging.info(f"  Force-include: {len(force_include_set)}")
    logging.info(f"  RF genes: {len(rf_final)} (includes {len(overlapping_genes)} overlaps with dimred)")
    logging.info(f"  Dimred genes: {len(dimred_final)}")
    logging.info(f"  Total: {total_genes} (gap to target filled in Step 6)")
    
    return combined_builder


def _apply_gap_filling(
    base_builder: GeneListBuilder,
    rf_builder: GeneListBuilder,
    dimred_builder: GeneListBuilder,
    gap_needed: int,
    target_size: int,
    run_celltype_filling: bool,
    run_deg_filling: bool,
) -> GeneListBuilder:
    """
    Apply gap-filling strategies to reach target panel size.

    Tries strategies in priority order (celltype-specific, then DEG-based) until
    the target size is reached.

    Parameters
    ----------
    base_builder : GeneListBuilder
        Combined panel (before gap filling)
    rf_builder : GeneListBuilder
        RF component results
    dimred_builder : GeneListBuilder
        Dimred component results
    gap_needed : int
        Number of genes needed to reach target
    target_size : int
        Target panel size
    run_celltype_filling : bool
        Enable celltype-specific filling
    run_deg_filling : bool
        Enable DEG-based filling

    Returns
    -------
    GeneListBuilder
        Final panel after gap filling
    """

    # Build gap-filling strategy priority order: celltype-specific, then DEG-based
    gap_fill_priority = []
    if run_celltype_filling:
        gap_fill_priority.append(GAP_FILL_STRATEGY_CELLTYPE)
    if run_deg_filling:
        gap_fill_priority.append(GAP_FILL_STRATEGY_DEG)

    if not gap_fill_priority:
        logging.warning("All gap-filling strategies disabled. Panel may be smaller than target size if gaps exist.")
    
    logging.info("Applying gap-filling strategies...")
    logging.info(f"Priority order: {' → '.join(gap_fill_priority)}")
    
    current_builder = base_builder
    current_genes = set(current_builder.get_all_genes())
    genes_still_needed = gap_needed
    strategies_used: List[str] = []  # gap-fill strategies that actually contributed genes

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
            )
            gene_source_label = GENE_SOURCE_GAP_FILL_DEG
        
        elif strategy == GAP_FILL_STRATEGY_CELLTYPE and run_celltype_filling:
            # Cell-type-specific filling: Use per-celltype dimred genes
            gap_genes = _fill_gap_with_celltype_dimred(
                current_genes=current_genes,
                dimred_builder=dimred_builder,
                n_needed=genes_still_needed,
            )
            gene_source_label = GENE_SOURCE_GAP_FILL_CELLTYPE

        else:
            logging.warning(f"Gap-filling strategy '{strategy}' not enabled or not recognized, skipping")
            continue
        
        # Add gap-fill genes to builder
        if gap_genes:
            logging.info(f"Gap-filling strategy '{strategy}' provided {len(gap_genes)} genes")
            n_before = genes_still_needed

            for gene in gap_genes:
                if gene not in current_genes:
                    # Carry the cell-type attribution the gene had in its source
                    # component so downstream diagnostics can count gap-fill genes
                    # per cell type (celltype-fill -> dimred pool, DEG-fill -> RF).
                    gap_celltype = 'global'
                    gap_meta = {
                        'gap_filled': True,
                        'gap_fill_strategy': GAP_FILL_STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
                    }
                    if gene_source_label == GENE_SOURCE_GAP_FILL_CELLTYPE:
                        rec = dimred_builder.gene_records.get(gene, {})
                        gap_celltype = rec.get('celltype') or 'global'
                        gap_meta['n_celltypes_selected'] = rec.get('n_celltypes_selected', 1)
                        gap_meta['contributing_celltypes'] = rec.get('contributing_celltypes', '')
                    elif gene_source_label == GENE_SOURCE_GAP_FILL_DEG:
                        rec = rf_builder.gene_records.get(gene, {})
                        gap_meta['rf_celltype'] = rec.get('rf_celltype', '')
                        gap_meta['rf_contributing_celltypes'] = rec.get('rf_contributing_celltypes', '')
                        gap_meta['rf_celltype_scores'] = rec.get('rf_celltype_scores', '{}')

                    current_builder.add_gene(
                        gene_name=gene,
                        gene_source=gene_source_label,
                        rank=None,
                        selection_score=None,
                        celltype=gap_celltype,
                        component=None,
                        additional_metadata=gap_meta,
                    )
                    current_genes.add(gene)
                    genes_still_needed -= 1
                    
                    if genes_still_needed <= 0:
                        break

            if n_before - genes_still_needed > 0:
                strategies_used.append(GAP_FILL_STRATEGY_DISPLAY_NAMES.get(strategy, strategy))
            logging.info(f"Added {n_before - genes_still_needed} gap-fill genes from '{strategy}'")
            logging.info(f"Genes still needed: {genes_still_needed}")
        else:
            logging.warning(f"Gap-filling strategy '{strategy}' found no suitable genes")

    if genes_still_needed > 0:
        logging.warning(
            f"Could not fill entire gap: {genes_still_needed} genes still needed after all strategies. "
            f"Final size: {len(current_genes)} (target: {target_size})"
        )
    
    current_builder.add_metadata('gap_filling_applied', True)
    current_builder.add_metadata(
        'gap_filling_strategy', '+'.join(strategies_used) if strategies_used else 'none'
    )
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
    
    # Use only the precomputed RF-ranked list, and never re-add genes already
    # present from RF, dimred, force-include, or earlier gap-fill strategies.
    candidates = [g for g in all_rf_genes if g not in current_genes]
    
    # Take top N candidates
    gap_genes = candidates[:n_needed]
    
    logging.info(f"Found {len(gap_genes)} DEG-based gap-fill genes")
    
    return gap_genes


def _fill_gap_with_celltype_dimred(
    current_genes: Set[str],
    dimred_builder: GeneListBuilder,
    n_needed: int,
) -> List[str]:
    """
    Cell-type-specific gap filling: Use per-celltype dimred genes.

    This reuses genes from dimred_builder (Phase 2 results) that weren't selected
    in the initial combination. No need to re-run dimred since we already have
    the full ranked list from per-celltype analysis.
    """

    logging.info("Cell-type-specific gap filling: Using per-celltype dimred genes")
    logging.info("Reusing ranked genes from Phase 2 dimred_builder (no recomputation needed)")

    # Get all dimred genes ranked
    all_dimred_genes = dimred_builder.get_all_genes()
    
    # Find genes not in current panel
    candidates = [g for g in all_dimred_genes if g not in current_genes]
    
    # Take top N candidates
    gap_genes = candidates[:n_needed]
    
    logging.info(f"Found {len(gap_genes)} celltype-specific gap-fill genes")
    
    return gap_genes


# Priority order for the panel-row (in_panel=True) section of RecoVar_panel_information.csv's
# sort key: RF-derived sources first, then dimred, then force-include, then the two gap-fill
# variants. Any gene_source not listed here (shouldn't happen) sorts last.
_PANEL_GENE_SOURCE_PRIORITY = {
    GENE_SOURCE_RF: 0,
    GENE_SOURCE_OVERLAP_TO_RF: 0,
    GENE_SOURCE_DIMRED: 1,
    GENE_SOURCE_FORCE_INCLUDE: 2,
    GENE_SOURCE_GAP_FILL_DEG: 3,
    GENE_SOURCE_GAP_FILL_CELLTYPE: 3,
}

# Priority order for the non-panel-row (in_panel=False) section: random-forest-pool candidates
# before NMF/PCA-pool candidates.
_NON_PANEL_POOL_PRIORITY = {'random forest': 0, 'NMF': 1}

# Display-only rename applied to RecoVar_panel_information.csv's gene_source column as
# the very last step of _build_panel_information(), after sorting -- internal identifiers
# (GENE_SOURCE_* constants, combined_builder.gene_records, gap-fill logic) are untouched;
# only the written CSV's string values change. force_include is intentionally left as-is.
_GENE_SOURCE_DISPLAY_RENAME = {
    GENE_SOURCE_RF: 'random forest',
    GENE_SOURCE_DIMRED: 'NMF',
    GENE_SOURCE_OVERLAP_TO_RF: 'overlap',
    GENE_SOURCE_GAP_FILL_CELLTYPE: 'gap-fill (NMF)',
    GENE_SOURCE_GAP_FILL_DEG: 'gap-fill (random forest)',
}


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
    1. RecoVar_panel_information.csv - Every gene considered during combination (union of the
                                RF and dimred candidate pools plus the final panel), with
                                panel membership (`in_panel`) and per-gene RF/NMF provenance.
                                Panel genes are listed first. Note: the per-component ranked
                                lists (which include ALL Xenium-passing candidates from each
                                component) are already saved by run_single_selection inside
                                rf_component/ and {reduction}_component/ subdirectories.
    2. combination_summary.json - Summary statistics including duplicate-resolution counts
                                and the combined-panel size status (`panel_size_status`
                                `complete`/`short` + `short_panel_reason`), matching the
                                per-component `filtering_summary.json` fields.
    """

    os.makedirs(results_dir, exist_ok=True)

    logging.info("")
    logging.info("=" * 80)
    logging.info("SAVING COMBINATION RESULTS")
    logging.info("=" * 80)

    # 1. Panel information -----------------------------------------------------
    panel_info_df = _build_panel_information(rf_builder, dimred_builder, combined_builder)
    panel_info_output = os.path.join(results_dir, 'RecoVar_panel_information.csv')
    panel_info_df.to_csv(panel_info_output, index=False)
    n_panel = int(panel_info_df['in_panel'].sum())
    logging.info(
        f"✓ Saved RecoVar_panel_information.csv: {n_panel} panel genes, "
        f"{len(panel_info_df)} genes tracked total"
    )

    # 2. Combination summary JSON -------------------------------------------
    summary = {
        'strategy': strategy,
        'reduction_type': reduction_type,
        'rf_component_genes': len(rf_builder.get_all_genes()),
        'dimred_component_genes': len(dimred_builder.get_all_genes()),
        'combined_panel_size': n_panel,
        'requested_panel_size': combined_builder.metadata.get('requested_panel_size'),
        'final_panel_size': combined_builder.metadata.get('final_panel_size', n_panel),
        'panel_size_status': combined_builder.metadata.get('panel_size_status'),
        'short_panel_reason': combined_builder.metadata.get('short_panel_reason'),
        'n_overlapping_genes': combined_builder.metadata.get('n_overlapping', 0),
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


def _build_panel_information(
    rf_builder: GeneListBuilder,
    dimred_builder: GeneListBuilder,
    combined_builder: GeneListBuilder,
) -> pd.DataFrame:
    """
    Build the RecoVar_panel_information.csv table: every gene considered during combination
    (union of the RF candidate pool, the dimred candidate pool, and the final panel), with
    panel membership and per-gene RF/NMF provenance. This is the sole combination-strategy
    output — it replaces both the former panel-only `ranked_gene_list.csv` and the former
    `gene_provenance.csv`.

    Covers the union of the RF candidate pool, the dimred candidate pool, and the final panel
    (the last is a superset guard for force-include / global-dimred-gap-fill genes that never
    passed through the RF or dimred component builders).

    `primary_celltype` uses a value-based fallback: a gene's real RF cell-type attribution
    (`rf_celltype`) wins whenever present, falling back to NMF's `celltype` only when RF has
    nothing usable for that gene — not merely when the gene isn't in the RF pool at all.
    `secondary_celltypes_nmf` / `n_celltypes_nmf` / `mean_expression` all follow whichever
    builder `primary_celltype` was actually resolved from.

    Args:
        rf_builder: RF component results (filtered).
        dimred_builder: Dimred component results (filtered).
        combined_builder: Final combined panel builder.

    Returns:
        DataFrame with columns: gene, in_panel, gene_source, informative_celltypes,
        primary_celltype, secondary_celltypes_nmf, n_celltypes_nmf, gap_filled,
        gap_fill_strategy, mean_expression. Panel genes (in_panel=True) sorted first by
        gene_source priority then gap_filled; non-panel genes sorted by candidate-pool
        priority then gene name.
    """
    rf_genes = set(rf_builder.get_all_genes())
    dimred_genes = set(dimred_builder.get_all_genes())
    panel_genes = set(combined_builder.get_all_genes())

    all_genes = sorted(rf_genes | dimred_genes | panel_genes)

    records = []
    for gene in all_genes:
        in_rf = gene in rf_genes
        in_dimred = gene in dimred_genes
        in_panel = gene in panel_genes

        rf_rec = rf_builder.gene_records.get(gene, {})
        dimred_rec = dimred_builder.gene_records.get(gene, {})
        panel_rec = combined_builder.gene_records.get(gene, {})

        if in_panel:
            gene_source = panel_rec.get('gene_source')
        elif in_rf or in_dimred:
            gene_source = 'random forest' if in_rf else 'NMF'
        else:
            gene_source = ''  # panel_only candidates are always in_panel=True; unreachable

        rf_celltype = rf_rec.get('rf_celltype') or None
        if rf_celltype:
            primary_celltype = rf_celltype
            resolved_from = 'rf'
        else:
            primary_celltype = dimred_rec.get('celltype') or None
            resolved_from = 'dimred' if primary_celltype else None

        if resolved_from == 'dimred':
            secondary_celltypes_nmf = dimred_rec.get('contributing_celltypes') or ''
            n_celltypes_nmf = dimred_rec.get('n_celltypes_selected')
            mean_expression = dimred_rec.get('mean_expression')
        elif resolved_from == 'rf':
            secondary_celltypes_nmf = ''
            n_celltypes_nmf = None
            mean_expression = rf_rec.get('mean_expression')
        else:
            secondary_celltypes_nmf = ''
            n_celltypes_nmf = None
            mean_expression = None

        informative_parts = [primary_celltype] if primary_celltype else []
        if secondary_celltypes_nmf:
            informative_parts += [
                ct for ct in secondary_celltypes_nmf.split(', ') if ct
            ]
        informative_celltypes = '|'.join(dict.fromkeys(informative_parts))

        gap_filled = bool(panel_rec.get('gap_filled', False)) if in_panel else False
        gap_fill_strategy = (panel_rec.get('gap_fill_strategy') or '') if in_panel else ''

        records.append({
            'gene': gene,
            'in_panel': in_panel,
            'gene_source': gene_source,
            'informative_celltypes': informative_celltypes,
            'primary_celltype': primary_celltype,
            'secondary_celltypes_nmf': secondary_celltypes_nmf,
            'n_celltypes_nmf': n_celltypes_nmf,
            'gap_filled': gap_filled,
            'gap_fill_strategy': gap_fill_strategy,
            'mean_expression': mean_expression,
        })

    df = pd.DataFrame.from_records(records)

    panel_mask = df['in_panel']
    df_panel = df[panel_mask].copy()
    df_other = df[~panel_mask].copy()

    df_panel['_source_rank'] = df_panel['gene_source'].map(_PANEL_GENE_SOURCE_PRIORITY).fillna(99)
    df_panel = df_panel.sort_values(
        ['_source_rank', 'gap_filled'], ascending=[True, True]
    ).drop(columns='_source_rank')

    df_other['_pool_rank'] = df_other['gene_source'].map(_NON_PANEL_POOL_PRIORITY).fillna(99)
    df_other = df_other.sort_values(
        ['_pool_rank', 'gene'], ascending=[True, True]
    ).drop(columns='_pool_rank')

    result = pd.concat([df_panel, df_other], ignore_index=True)
    # Display-only rename, applied last so it never affects the sort above (which keys
    # off the real internal GENE_SOURCE_* identifiers). Non-panel rows' pool labels
    # ("random forest" / "NMF") are already display strings and pass through unchanged.
    result['gene_source'] = result['gene_source'].replace(_GENE_SOURCE_DISPLAY_RENAME)
    return result


# gene_source values (panel rows only) attributed to each side of the RF/dimred split,
# for the dotplot's source-colored gene labels -- matches the DEG_SOURCES/NMF_SOURCES
# categorization Analysis-scripts/pipeline/plot_recovar_selection.py used to apply
# separately, after evaluation preprocessing. rf_simple is not included: combination
# strategies always use rf_deg for their RF component, never rf_simple.
# NOTE: these are the DISPLAY strings (post _GENE_SOURCE_DISPLAY_RENAME), since this
# reads the just-written RecoVar_panel_information.csv back from disk, not the internal
# GENE_SOURCE_* identifiers.
_DOTPLOT_RF_SOURCES = {'random forest', 'overlap', 'gap-fill (random forest)'}
_DOTPLOT_DIMRED_SOURCES = {'NMF', 'gap-fill (NMF)'}


def _build_per_celltype_gene_lists(
    panel_df: pd.DataFrame,
    adata: AnnData,
    celltype_column: str,
    rf_component_df: Optional[pd.DataFrame],
    dimred_component_df: Optional[pd.DataFrame],
) -> tuple[dict[str, list[str]], list[str]]:
    """For each cell type, list every panel gene whose own selection round -- random
    forest's or NMF's, per the gene's ``gene_source`` -- implicated that cell type, as
    a flat list (no primary/secondary tiering).

    Cell-type membership is read from the RF/NMF component files (``rf_celltype`` /
    ``rf_celltype_scores`` in ``rf_component_df``, ``celltype`` / ``informative_celltypes``
    in ``dimred_component_df``) rather than the combined CSV's ``primary_celltype`` /
    ``secondary_celltypes_nmf`` columns -- those apply a value-based fallback where RF's
    attribution always overwrites NMF's for any gene that also happens to appear in RF's
    full (oversampled) candidate pool, which silently discards NMF's often much richer,
    genuinely-per-celltype-fit attribution. NMF is fit *independently per cell type*
    (see ``Selection-module/CLAUDE.md``), so every cell type in a gene's
    ``informative_celltypes`` is a cell type whose own fit selected that gene -- not a
    lesser "secondary" relationship, just one that didn't happen to win the shared
    factor slot during duplicate resolution.

    Routing by ``gene_source``:
    - ``'random forest'`` / ``'gap-fill (random forest)'``: RF's own membership set.
    - ``'NMF'`` / ``'gap-fill (NMF)'``: NMF's own membership set.
    - ``'overlap'``: the union of both -- selected independently by both methods, so
      both methods' evidence counts. A gene can then legitimately land on more than
      one cell type's plot for reasons neither method alone would explain.
    - ``'force_include'``, or a gene missing from the relevant component file(s):
      falls back to the combined CSV's own ``informative_celltypes`` (e.g. an
      older-format results dir missing a component file); if that's also empty, the
      gene is unattributed.

    RF's membership set = ``{rf_celltype}`` (the Gini-importance argmax) plus every
    other cell type in ``rf_celltype_scores`` (a JSON dict) with share >=
    ``RF_CONTRIBUTING_CELLTYPE_MIN_SHARE`` -- usually empty in practice, since real
    RF-selected genes' shares are typically sharply peaked on the argmax. NMF's
    membership set = ``{celltype}`` union ``informative_celltypes.split('|')``.

    Cell types are ordered by descending cell count (matching the convention used by
    the cells-per-celltype bar chart); a cell-type name not in that vocabulary is
    appended after. Gene order within each cell type's list follows ``panel_df``'s own
    row order (already ``gene_source``-priority sorted).

    Returns a ``(per_celltype, unattributed)`` pair: ``per_celltype`` maps each cell
    type with >=1 gene to its flat gene list; ``unattributed`` is the list of panel
    genes with no resolvable cell-type membership at all. Returns ``({}, [])`` if
    ``panel_df`` doesn't have a ``gene_source`` column (e.g. an older-format CSV).
    """
    if 'gene_source' not in panel_df.columns:
        return {}, []

    celltype_order = adata.obs[celltype_column].value_counts().index.tolist()

    rf_by_gene = rf_component_df.set_index('gene') if rf_component_df is not None else None
    dimred_by_gene = dimred_component_df.set_index('gene') if dimred_component_df is not None else None

    def _rf_membership(gene: str) -> set:
        if rf_by_gene is None or gene not in rf_by_gene.index:
            return set()
        row = rf_by_gene.loc[gene]
        rf_celltype = str(row.get('rf_celltype') or '').strip()
        members = {rf_celltype} if rf_celltype else set()
        scores_raw = row.get('rf_celltype_scores')
        if isinstance(scores_raw, str) and scores_raw.strip():
            try:
                scores = json.loads(scores_raw)
                members |= {
                    ct for ct, share in scores.items()
                    if share >= RF_CONTRIBUTING_CELLTYPE_MIN_SHARE
                }
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        return members

    def _dimred_membership(gene: str) -> set:
        if dimred_by_gene is None or gene not in dimred_by_gene.index:
            return set()
        row = dimred_by_gene.loc[gene]
        celltype = str(row.get('celltype') or '').strip()
        members = {celltype} if celltype else set()
        informative_raw = str(row.get('informative_celltypes') or '').strip()
        if informative_raw:
            members |= {c.strip() for c in informative_raw.split('|') if c.strip()}
        return members

    membership: dict[str, list[str]] = {ct: [] for ct in celltype_order}
    unattributed: list[str] = []

    for _, row in panel_df.iterrows():
        gene = row['gene']
        gene_source = str(row.get('gene_source') or '')

        if gene_source in ('random forest', 'gap-fill (random forest)'):
            members = _rf_membership(gene)
        elif gene_source in ('NMF', 'gap-fill (NMF)'):
            members = _dimred_membership(gene)
        elif gene_source == 'overlap':
            members = _rf_membership(gene) | _dimred_membership(gene)
        else:
            members = set()

        if not members:
            # Graceful fallback to the combined CSV's own informative_celltypes
            # (covers force_include, missing component files, older-format CSVs).
            informative_raw = str(row.get('informative_celltypes') or '').strip()
            if informative_raw:
                members = {c.strip() for c in informative_raw.split('|') if c.strip()}

        if not members:
            unattributed.append(gene)
            continue

        for ct in members:
            membership.setdefault(ct, []).append(gene)

    result = {ct: genes for ct, genes in membership.items() if genes}
    return result, unattributed


def _generate_combination_dotplot(
    adata: AnnData,
    results_dir: str,
    celltype_column: str,
    output_dir: str,
    reduction_type: str,
) -> None:
    """Generate one final-gene expression dotplot per cell type, right after panel
    construction.

    Reads back the just-written ``RecoVar_panel_information.csv`` from
    ``results_dir``, plus the RF and dimred component files
    (``rf_component/rf_deg_panel_information.csv``,
    ``{reduction_type}_component/{reduction_type}_panel_information.csv``), and calls
    ``Plotting-module``'s ``plot_final_gene_dotplot()`` once per cell type (via
    ``_build_per_celltype_gene_lists``), directly from the selection pipeline, using
    the same ``adata`` already in memory -- no dependency on Evaluation-module
    preprocessing (which this used to wait on, via the standalone
    ``Analysis-scripts/pipeline/plot_recovar_selection.py`` driver run after a full
    evaluation cell completed). Each cell type's plot shows a flat list of every gene
    whose own selection round (RF's or NMF's) implicated that cell type -- replacing
    the old single whole-panel dotplot that mechanically chunked by gene count. Plots
    are written to ``output_dir``, which may differ from ``results_dir`` (e.g. a
    caller-managed ``plots/`` tree, kept separate from the gene-list CSVs). Non-fatal:
    a plotting failure is logged and swallowed rather than failing the selection run;
    a missing/unreadable component file degrades gracefully (see
    ``_build_per_celltype_gene_lists``) rather than aborting the whole plot.
    """
    panel_info_path = os.path.join(results_dir, 'RecoVar_panel_information.csv')
    if not os.path.exists(panel_info_path):
        return
    if celltype_column not in adata.obs.columns:
        logging.warning(
            f"Skipping final-gene dotplot: celltype column '{celltype_column}' "
            f"not found in adata.obs"
        )
        return

    try:
        import sys as _sys2
        _plotting_dir = str(Path(__file__).parent.parent / "Plotting-module")
        if _plotting_dir not in _sys2.path:
            _sys2.path.insert(0, _plotting_dir)
        from _selection_plots import plot_final_gene_dotplot, _order_genes_by_celltype_contribution

        panel_df = pd.read_csv(panel_info_path)
        panel_df = panel_df[panel_df['in_panel'].fillna(False).astype(bool)]
        src = panel_df['gene_source'].astype(str)
        deg_gene_set = set(panel_df.loc[src.isin(_DOTPLOT_RF_SOURCES), 'gene'])
        nmf_gene_set = set(panel_df.loc[src.isin(_DOTPLOT_DIMRED_SOURCES), 'gene'])

        rf_component_df = None
        rf_component_path = os.path.join(results_dir, 'rf_component', 'rf_deg_panel_information.csv')
        try:
            if os.path.exists(rf_component_path):
                rf_component_df = pd.read_csv(rf_component_path)
        except Exception as exc:
            logging.warning(f"Could not read RF component file for dotplot attribution (non-fatal): {exc}")

        dimred_component_df = None
        try:
            dimred_filename = panel_information_filename('dimred_only', reduction_type)
            dimred_component_path = os.path.join(
                results_dir, f'{reduction_type}_component', dimred_filename
            )
            if os.path.exists(dimred_component_path):
                dimred_component_df = pd.read_csv(dimred_component_path)
        except Exception as exc:
            logging.warning(f"Could not read dimred component file for dotplot attribution (non-fatal): {exc}")

        per_celltype, unattributed = _build_per_celltype_gene_lists(
            panel_df, adata, celltype_column, rf_component_df, dimred_component_df
        )

        os.makedirs(output_dir, exist_ok=True)
        n_plotted = 0
        for ct, ct_genes in per_celltype.items():
            ct_genes = _order_genes_by_celltype_contribution(adata, ct_genes, celltype_column, ct)
            plot_final_gene_dotplot(
                adata=adata,
                probeset_genes=ct_genes,
                deg_genes=[g for g in ct_genes if g in deg_gene_set],
                nmf_genes=[g for g in ct_genes if g in nmf_gene_set],
                groupby=celltype_column,
                output_path=output_dir,
                filename_prefix=re.sub(r'[^A-Za-z0-9]+', '_', ct).strip('_') or 'unknown',
                title=f"Panel genes for {ct}",
            )
            n_plotted += 1

        if unattributed:
            plot_final_gene_dotplot(
                adata=adata,
                probeset_genes=unattributed,
                deg_genes=[g for g in unattributed if g in deg_gene_set],
                nmf_genes=[g for g in unattributed if g in nmf_gene_set],
                groupby=celltype_column,
                output_path=output_dir,
                filename_prefix='unattributed',
                title="Panel genes with no cell-type attribution",
            )
            n_plotted += 1

        logging.info(f"✓ Saved {n_plotted} per-celltype final-gene dotplots to: {output_dir}")
    except Exception as exc:  # never let a diagnostic plot fail the selection run
        logging.warning(f"final-gene dotplot generation failed (non-fatal): {exc}")


def _generate_expression_distribution_plot(
    adata: AnnData,
    results_dir: str,
    output_dir: str,
) -> None:
    """Generate the panel-gene raw-vs-lognorm expression distribution histogram.

    Reads back the just-written ``RecoVar_panel_information.csv`` from ``results_dir``
    to get the panel gene list, and calls ``Plotting-module``'s
    ``plot_panel_expression_distribution()`` directly from the selection pipeline,
    using the same ``adata`` already in memory (both ``adata.layers['counts']`` and
    ``adata.X`` are needed -- raw and log-normalized respectively). Non-fatal: a
    plotting failure is logged and swallowed rather than failing the selection run.
    """
    panel_info_path = os.path.join(results_dir, 'RecoVar_panel_information.csv')
    if not os.path.exists(panel_info_path):
        return

    try:
        import sys as _sys3
        _plotting_dir = str(Path(__file__).parent.parent / "Plotting-module")
        if _plotting_dir not in _sys3.path:
            _sys3.path.insert(0, _plotting_dir)
        from _selection_plots import plot_panel_expression_distribution

        panel_df = pd.read_csv(panel_info_path)
        panel_genes = panel_df.loc[
            panel_df['in_panel'].fillna(False).astype(bool), 'gene'
        ].tolist()

        os.makedirs(output_dir, exist_ok=True)
        plot_panel_expression_distribution(
            adata=adata,
            panel_genes=panel_genes,
            output_dir=output_dir,
        )
    except Exception as exc:  # never let a diagnostic plot fail the selection run
        logging.warning(f"panel expression-distribution plot generation failed (non-fatal): {exc}")

