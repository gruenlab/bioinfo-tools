"""
Single gene selection strategy wrapper with the standardized filtering pipeline.

FILTERING ORDER:
----------------
1. BLACKLIST FILTER (PRE-selection): Remove genes from adata before selection
2. RUN SELECTION STRATEGY: Work on filtered adata
3. XENIUM FILTER (POST-selection): Apply to ranked gene list, celltype-aware
4. SELECT TOP N: From Xenium-filtered list

All strategies share this filtering logic for consistency and reproducibility.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from anndata import AnnData

# Support script execution by adding module directory to sys.path
import sys
MODULE_DIR = Path(__file__).parent.absolute()
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

# Import refactored selection modules - use absolute imports
from _gene_list_builder import GeneListBuilder, derive_informative_celltypes, panel_information_filename
from _filtering import (
    apply_blacklist_filter,
    apply_xenium_filter_to_genelist,
    apply_xenium_filter_global_to_genelist,
)
# --- load THIS directory's _constants.py by path (sibling dirs share the name) ---
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _cpath
_cspec = _ilu.spec_from_file_location("_constants", _cpath(__file__).resolve().parent / "_constants.py")
_sys.modules["_constants"] = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_sys.modules["_constants"])


from _constants import (
    COL_SELECTED_INITIAL,
    COL_PASSED_XENIUM,
    COL_GENE,
    COL_SELECTION_SCORE,
)

from _deg_selection import select_degs
from _baseline_selection import (
    select_highly_variable_genes,
    select_random_genes,
)
from _rf_selection import select_genes_with_rf, format_rf_ranked_df
from _dimred_selection import (
    select_genes_from_nmf,
    select_genes_from_pca,
    format_dimred_ranked_df,
)
# Use absolute imports (for script execution)
from _constants import (
    DEFAULT_PROBESET_SIZE,
    DEFAULT_MIN_CELLS_PER_CELLTYPE,
    DEFAULT_RANDOM_STATE,
    COL_CELLTYPE,
    DEFAULT_MIN_XENIUM_EXPRESSION,
    DEFAULT_MAX_XENIUM_EXPRESSION,
    DEFAULT_REDUCTION_TYPE,
)

logger = logging.getLogger(__name__)


def _resolve_gene_mean_expression(
    builder: GeneListBuilder,
    mean_expr_per_ct: Optional[dict],
    global_mean_expr: Optional[dict],
) -> dict:
    """Build a ``{gene: mean_expression}`` map for every gene the builder tracks.

    Prefers the per-celltype means (each gene scored in its assigned celltype;
    genes without a specific celltype get their highest per-celltype mean) and
    falls back to the pooled global means. Returns an empty dict when neither
    source is available (Xenium filtering disabled).
    """
    df = builder.to_dataframe()
    if df.empty:
        return {}

    resolved: dict = {}
    if mean_expr_per_ct:
        for _, row in df.iterrows():
            gene = row[COL_GENE]
            ct_expr = mean_expr_per_ct.get(row.get(COL_CELLTYPE, 'global'))
            if ct_expr is not None and gene in ct_expr:
                resolved[gene] = ct_expr[gene]
            else:
                vals = [d[gene] for d in mean_expr_per_ct.values() if gene in d]
                if vals:
                    resolved[gene] = max(vals)
    elif global_mean_expr:
        for gene in df[COL_GENE]:
            if gene in global_mean_expr:
                resolved[gene] = global_mean_expr[gene]
    return resolved


def _json_safe_panel_metadata(builder: GeneListBuilder) -> dict:
    """Return JSON-safe run-level metadata, excluding private cache payloads."""
    return {
        key: value
        for key, value in builder.metadata.items()
        if not str(key).startswith("_")
    }


def run_single_selection(
    adata: AnnData,
    strategy: str,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    celltype_column: str = COL_CELLTYPE,
    min_cells_per_celltype: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
    random_state: int = DEFAULT_RANDOM_STATE,
    # Blacklist filter (PRE-selection)
    blacklist_patterns: Optional[list[str]] = None,
    use_default_blacklist: bool = True,
    force_include_genes: Optional[list[str]] = None,
    # Xenium filter (POST-selection)
    apply_xenium_filter: bool = False,
    xenium_celltype_aware: bool = True,
    xenium_min_expr: float = DEFAULT_MIN_XENIUM_EXPRESSION,
    xenium_max_expr: float = DEFAULT_MAX_XENIUM_EXPRESSION,
    mean_expr_per_ct: Optional[dict] = None,
    global_mean_expr: Optional[dict] = None,
    # Strategy-specific parameters
    reduction_type: Optional[str] = DEFAULT_REDUCTION_TYPE,  # 'nmf' or 'pca'
    n_components: int = 50,
    pool_size_per_celltype: int = 200,  # Pool size per celltype for Phase 1
    dimred_n_jobs: int = 1,  # workers for per-celltype NMF/PCA fits (1=seq, -1=all cores)
    # NOTE: Dimred strategies (dimred_only) handle gap-filling internally via factor-aware
    # duplicate resolution. NMF/PCA is always fitted per cell type.
    # Caching (optional pre-computed results)
    deg_results: Optional[GeneListBuilder] = None,
    rf_simple_results: Optional[GeneListBuilder] = None,
    nmf_loadings_per_celltype: Optional[dict] = None,
    pca_loadings_per_celltype: Optional[dict] = None,
    dimred_counts_input: str = "raw",
    require_nmf_model_cache: bool = False,
    nmf_objective: str = "auto",
    rf_score_cache_file: Optional[str] = None,
    # Output
    results_dir: Optional[str] = None,
    # Shared NMF/PCA model cache (independent of results_dir, reused across probeset sizes)
    nmf_model_cache_dir: Optional[str] = None,
) -> GeneListBuilder:
    """Run a single gene selection strategy with standardized filtering pipeline.

    Filtering order:
    1. Blacklist filter (PRE-selection): Remove blacklisted genes from adata
    2. Run selection strategy: Work on filtered adata
    3. Xenium filter (POST-selection): Apply to ranked genes, celltype-aware
    4. Select top N: From Xenium-filtered list

    Args:
        adata: Annotated data matrix (will be modified if blacklist applied)
        strategy: Selection strategy name
            - 'deg_only': Differential expression genes
            - 'rf_simple': Random forest on all genes
            - 'rf_deg': Random forest on DEG-filtered genes
            - 'hvg': Highly variable genes
            - 'random': Random selection
            - 'dimred_only': NMF/PCA only (requires reduction_type)
        probeset_size: Target number of genes
        celltype_column: Column with cell type annotations
        min_cells_per_celltype: Minimum cells per cell type
        random_state: Random seed
        blacklist_patterns: List of gene patterns to exclude (e.g., ['mt-', 'Rps'])
        use_default_blacklist: Also apply the built-in default blacklist patterns.
        force_include_genes: Genes always kept, bypassing the blacklist.
        apply_xenium_filter: Whether to apply Xenium expression filter
        xenium_celltype_aware: Use per-cell-type mean expression for the Xenium filter
            (uses ``mean_expr_per_ct``); when False use the pooled ``global_mean_expr``.
        xenium_min_expr: Min expression threshold for Xenium
        xenium_max_expr: Max expression threshold for Xenium
        mean_expr_per_ct: Pre-computed per-celltype mean expression (celltype-aware mode).
        global_mean_expr: Pre-computed pooled global mean expression (global Xenium mode).
        reduction_type: 'nmf' or 'pca' (for dimred_only strategy)
        n_components: Number of NMF/PCA components
        pool_size_per_celltype: Phase-1 dimred pool size per cell type.
        dimred_n_jobs: Workers for per-celltype NMF/PCA fitting.
            1 (default) = sequential; -1 = all cores. Results are identical to sequential.
        deg_results: Pre-computed DEG results (for caching)
        rf_simple_results: Pre-computed rf_simple GeneListBuilder reused as the rf_deg
            candidate source (caching).
        nmf_loadings_per_celltype: Pre-computed per-celltype NMF loadings
        pca_loadings_per_celltype: Pre-computed per-celltype PCA loadings
        dimred_counts_input: Matrix for the dimred fit — 'raw' (default) or 'lognorm'.
        require_nmf_model_cache: If True, raise instead of recomputing when a required
            NMF/PCA model cache is missing.
        nmf_objective: NMF factorization objective — 'auto' (default) derives the
            solver/beta_loss from dimred_counts_input; 'frobenius'/'kl' force that
            objective regardless of input. NMF only (ignored for reduction_type='pca').
        rf_score_cache_file: Explicit pickled RF score cache to reuse (skips retraining).
        results_dir: Directory to save results
        nmf_model_cache_dir: Shared NMF/PCA model cache directory, independent of
            ``results_dir`` and reused across probeset sizes.

    Returns:
        GeneListBuilder with final selected genes after all filtering

    Raises:
        ValueError: If strategy is invalid or required parameters missing

    Examples:
        >>> # DEG selection with blacklist and Xenium filtering
        >>> builder = run_single_selection(
        ...     adata,
        ...     strategy='deg_only',
        ...     probeset_size=500,
        ...     blacklist_patterns=['mt-', 'Rps', 'Rpl'],
        ...     apply_xenium_filter=True,
        ...     mean_expr_per_ct=mean_expr_dict
        ... )

        >>> # NMF selection (per-celltype)
        >>> builder = run_single_selection(
        ...     adata,
        ...     strategy='dimred_only',
        ...     reduction_type='nmf',
        ...     blacklist_patterns=['mt-']
        ... )
    """
    logger.info("=" * 80)
    logger.info(f"RUNNING SELECTION STRATEGY: {strategy.upper()}")
    logger.info("=" * 80)
    logger.info(f"Dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    logger.info(f"Target genes: {probeset_size}")

    # ========================================================================
    # PRE-SELECTION BLACKLIST FILTER
    # ========================================================================
    # When called via run_selection_pipeline.py the blacklist has already been
    # applied at pipeline level (blacklist_patterns=None, use_default_blacklist=False).
    # This block handles direct calls to run_single_selection where the caller
    # wants in-place blacklist filtering.
    # ========================================================================

    original_n_genes = adata.n_vars
    blacklisted_genes = []

    if blacklist_patterns or use_default_blacklist:
        # Apply blacklist filter using modular function
        all_genes = adata.var_names.tolist()
        filtered_genes, removed_genes, force_included = apply_blacklist_filter(
            gene_list=all_genes,
            blacklist_patterns=blacklist_patterns,
            use_default_blacklist=use_default_blacklist,
            force_include_genes=force_include_genes,
        )

        blacklisted_genes = removed_genes

        if blacklisted_genes:
            adata = adata[:, filtered_genes].copy()
            logger.info(f"Blacklist filter: removed {len(blacklisted_genes)} genes "
                        f"({original_n_genes} → {adata.n_vars} remaining)")

    # ========================================================================
    # STEP 2: RUN SELECTION STRATEGY
    # ========================================================================
    # Run the selected strategy on the blacklist-filtered adata.
    # Each strategy returns a GeneListBuilder with all genes ranked by score.
    # ========================================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 2: RUN SELECTION STRATEGY")
    logger.info("=" * 80)

    if strategy == "deg_only":
        logger.info("Strategy: Differential Expression Genes (DEG)")
        builder = select_degs(
            adata=adata,
            probeset_size=probeset_size,
            celltype_column=celltype_column,
            min_cells_per_celltype=min_cells_per_celltype,
            results_dir=results_dir,
        )

    elif strategy == "rf_simple":
        logger.info("Strategy: Random Forest (all genes)")
        builder = select_genes_with_rf(
            adata=adata,
            probeset_size=probeset_size,
            celltype_column=celltype_column,
            min_cells_per_celltype=min_cells_per_celltype,
            random_state=random_state,
            use_deg_prefilter=False,
            rf_score_cache_file=rf_score_cache_file,
            results_dir=results_dir,
        )

    elif strategy == "rf_deg":
        logger.info("Strategy: Random Forest (DEG-filtered genes)")
        builder = select_genes_with_rf(
            adata=adata,
            probeset_size=probeset_size,
            celltype_column=celltype_column,
            min_cells_per_celltype=min_cells_per_celltype,
            random_state=random_state,
            use_deg_prefilter=True,
            deg_results=deg_results,  # Use cached DEGs if available
            rf_score_cache_file=rf_score_cache_file,
            results_dir=results_dir,
        )

    elif strategy == "hvg":
        logger.info("Strategy: Highly Variable Genes (HVG)")
        builder = select_highly_variable_genes(
            adata=adata,
            probeset_size=probeset_size,
            results_dir=results_dir,
        )

    elif strategy == "random":
        logger.info(f"Strategy: Random Selection (seed={random_state})")
        builder = select_random_genes(
            adata=adata,
            probeset_size=probeset_size,
            random_state=random_state,
            results_dir=results_dir,
        )


    elif strategy == "dimred_only":
        if reduction_type is None:
            raise ValueError("reduction_type required for dimred_only strategy (must be 'nmf' or 'pca')")

        logger.info(f"Strategy: Dimensionality Reduction ({reduction_type.upper()}, per cell type)")

        if reduction_type == "nmf":
            builder = select_genes_from_nmf(
                adata=adata,
                probeset_size=probeset_size,
                celltype_column=celltype_column,
                n_components=n_components,
                pool_size_per_celltype=pool_size_per_celltype,
                nmf_n_jobs=dimred_n_jobs,
                min_cells_per_celltype=min_cells_per_celltype,
                random_state=random_state,
                nmf_loadings_per_celltype=nmf_loadings_per_celltype,
                results_dir=results_dir,
                nmf_model_cache_dir=nmf_model_cache_dir,
                mean_expr_per_ct=mean_expr_per_ct,
                xenium_min_expr=xenium_min_expr,
                xenium_max_expr=xenium_max_expr,
                dimred_counts_input=dimred_counts_input,
                require_nmf_model_cache=require_nmf_model_cache,
                nmf_objective=nmf_objective,
            )
        elif reduction_type == "pca":
            builder = select_genes_from_pca(
                adata=adata,
                probeset_size=probeset_size,
                celltype_column=celltype_column,
                n_components=n_components,
                pool_size_per_celltype=pool_size_per_celltype,
                pca_n_jobs=dimred_n_jobs,
                min_cells_per_celltype=min_cells_per_celltype,
                random_state=random_state,
                pca_loadings_per_celltype=pca_loadings_per_celltype,
                results_dir=results_dir,
                nmf_model_cache_dir=nmf_model_cache_dir,
                mean_expr_per_ct=mean_expr_per_ct,
                xenium_min_expr=xenium_min_expr,
                xenium_max_expr=xenium_max_expr,
                dimred_counts_input=dimred_counts_input,
            )
        else:
            raise ValueError(f"Invalid reduction_type: {reduction_type} (must be 'nmf' or 'pca')")

    else:
        raise ValueError(
            f"Invalid strategy: {strategy}. Must be one of: "
            "deg_only, rf_simple, rf_deg, hvg, random, dimred_only"
        )

    logger.info(f"✓ Selection complete: {len(builder.get_selected_genes('initial'))} genes selected")

    # ========================================================================
    # STEP 3: POST-SELECTION XENIUM FILTER (CELLTYPE-AWARE)
    # ========================================================================
    # Apply Xenium filter to the ranked gene list BEFORE selecting top N.
    # For dimred_only, this step is intentionally skipped because Xenium is
    # already applied earlier during per-celltype pool construction.
    # ========================================================================

    skip_post_xenium_for_dimred = (strategy == "dimred_only")

    if apply_xenium_filter and not skip_post_xenium_for_dimred:
        logger.info("")
        logger.info("=" * 80)
        use_global = not xenium_celltype_aware and global_mean_expr is not None
        logger.info(f"STEP 3: XENIUM FILTER ({'GLOBAL' if use_global else 'CELLTYPE-AWARE'})")
        logger.info("=" * 80)
        logger.info(f"Expression range: [{xenium_min_expr}, {xenium_max_expr}]")

        if use_global:
            builder = apply_xenium_filter_global_to_genelist(
                gene_list_builder=builder,
                global_mean_expr=global_mean_expr,
                min_expr=xenium_min_expr,
                max_expr=xenium_max_expr,
            )
        elif mean_expr_per_ct is None:
            logger.warning("Xenium filter requested but no expression data available, skipping")
        else:
            builder = apply_xenium_filter_to_genelist(
                gene_list_builder=builder,
                mean_expr_per_ct=mean_expr_per_ct,
                min_expr=xenium_min_expr,
                max_expr=xenium_max_expr,
                celltype_aware=xenium_celltype_aware,
            )

        logger.info(f"✓ Xenium filter complete: {len(builder.get_selected_genes('initial'))} genes remain")
    elif skip_post_xenium_for_dimred:
        logger.info("Skipping Step 3 Xenium filter for dimred_only (already applied in Phase 1 pool filtering)")
    else:
        logger.info("Xenium filter disabled (apply_xenium_filter=False)")

    # Record per-gene mean expression (from the same data the Xenium filter uses)
    # into the builder's `mean_expression` column, when that data is available.
    mean_expr_map = _resolve_gene_mean_expression(builder, mean_expr_per_ct, global_mean_expr)
    if mean_expr_map:
        builder.set_mean_expression(mean_expr_map)
        logger.info(f"✓ Recorded mean expression for {len(mean_expr_map)} genes")

    # ========================================================================
    # STEP 4: SELECT TOP N FROM FILTERED LIST
    # ========================================================================
    # Builder already has genes marked as selected/not-selected.
    # Xenium filter may have removed some selected genes, so we need to
    # re-select from the remaining genes.
    # ========================================================================

    n_total_ranked = len(builder.get_all_genes())

    # Count genes that are BOTH initially selected AND passed the Xenium filter.
    # This is the true effective panel size after filtering.  Using only
    # get_selected_genes('initial') can return the full original selection (before
    # Xenium removed some), causing the gap-fill condition to never trigger even
    # though the final panel would be smaller than requested.
    all_genes_df = builder.to_dataframe()
    passed_filter = all_genes_df[COL_PASSED_XENIUM].fillna(True)
    selected_and_passing = all_genes_df[all_genes_df[COL_SELECTED_INITIAL] & passed_filter]
    n_effective = len(selected_and_passing)

    if n_effective < probeset_size:
        logger.warning(
            f"Only {n_effective} genes remain after Xenium filter "
            f"(requested {probeset_size}). "
            f"Full ranked pool has {n_total_ranked} genes — attempting gap-fill from "
            f"non-selected Xenium-passing candidates."
        )

        # Candidates: not initially selected AND passed Xenium
        candidate_genes = all_genes_df[
            (~all_genes_df[COL_SELECTED_INITIAL]) & passed_filter
        ].sort_values(COL_SELECTION_SCORE, ascending=False)

        genes_needed = probeset_size - n_effective
        gap_fill_genes = candidate_genes.head(genes_needed)[COL_GENE].tolist()

        for gene in gap_fill_genes:
            builder.mark_selected(gene)

        logger.info(
            f"✓ Gap-filled: added {len(gap_fill_genes)} genes from ranked pool "
            f"({n_effective} post-filter + {len(gap_fill_genes)} gap-fill = "
            f"{n_effective + len(gap_fill_genes)} total)"
        )
        if len(gap_fill_genes) < genes_needed:
            logger.warning(
                f"Gap-fill incomplete: only {len(gap_fill_genes)}/{genes_needed} genes available "
                f"after Xenium filter. Panel will be smaller than requested."
            )

    # Promote initial selection to final
    selected_initial = builder.get_selected_genes('initial')
    if selected_initial:
        builder.mark_selected(selected_initial, selection_type='final')
        logger.info(f"✓ Promoted {len(selected_initial)} genes from initial to final")

    # ========================================================================
    # SAVE FINAL RESULTS
    # ========================================================================
    # Output files:
    # 1. {strategy}_panel_information.csv (see panel_information_filename()) - Every gene
    #      the builder tracked (all candidates, including Xenium failures for the
    #      strategies that still expose that column). deg_only/hvg/random keep the full
    #      schema (panel genes at the top, ranked by score); rf_deg/rf_simple/dimred_only
    #      are trimmed/reordered by their own format_rf_ranked_df()/format_dimred_ranked_df()
    #      (see docs/doc-pipeline/audit_3.md, "Selection-module CSV output reorganization").
    # 2. filtering_summary.json - Statistics
    # ========================================================================

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

        logger.info("")
        logger.info("=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)

        # 1. Build unified ranked gene list -----------------------------------
        #    Scope: every gene the builder tracked, including Xenium failures (kept with
        #    passed_xenium=False + xenium_failure_reason rather than dropped).
        #    Layout: panel genes (in_panel=True) first, rest below; both groups
        #    sorted by rank ascending (rank 1 = highest scoring gene).
        full_df = builder.to_dataframe()

        # Unified per-gene cell-type coverage column (genes-per-cell-type is then
        # just full_df['informative_celltypes'].str.split('|').explode().value_counts()).
        full_df['informative_celltypes'] = derive_informative_celltypes(full_df)

        # Insert in_panel as the second column for readability
        full_df.insert(1, 'in_panel', full_df['final_selection'].fillna(False))

        # Strategy-owned formatters trim/reorder/sort the columns each strategy's own
        # module knows are non-redundant for its output (see docs/doc-pipeline/audit_3.md,
        # "Selection-module CSV output reorganization"). deg_only/hvg/random keep today's
        # full schema, sorted panel-genes-first then by rank ascending, unchanged.
        if strategy == "dimred_only":
            full_df = format_dimred_ranked_df(full_df)
        elif strategy in ("rf_deg", "rf_simple"):
            full_df = format_rf_ranked_df(full_df)
        else:
            full_df = full_df.sort_values(
                ['in_panel', 'rank'],
                ascending=[False, True],
            ).reset_index(drop=True)

        ranked_output = os.path.join(
            results_dir, panel_information_filename(strategy, reduction_type)
        )
        full_df.to_csv(ranked_output, index=False)
        n_panel    = int(full_df['in_panel'].sum())
        n_remain   = len(full_df) - n_panel
        panel_metadata = _json_safe_panel_metadata(builder)
        panel_metadata['requested_panel_size'] = probeset_size
        panel_metadata['final_panel_size'] = n_panel
        if n_panel >= probeset_size:
            panel_metadata['panel_size_status'] = 'complete'
            panel_metadata['short_panel_reason'] = None
        elif panel_metadata.get('rf_short_panel_allowed'):
            panel_metadata['panel_size_status'] = 'short'
            if not panel_metadata.get('short_panel_reason'):
                panel_metadata['short_panel_reason'] = 'rf_deg_candidate_pool_exhausted_after_retries'
        else:
            panel_metadata['panel_size_status'] = 'short'
            if not panel_metadata.get('short_panel_reason'):
                panel_metadata['short_panel_reason'] = 'insufficient_ranked_genes_after_filtering'

        for key, value in panel_metadata.items():
            builder.add_metadata(key, value)

        logger.info(
            f"✓ Saved {os.path.basename(ranked_output)}: "
            f"{n_panel} panel genes + {n_remain} remaining candidate genes "
            f"({len(full_df)} total)"
        )

        # 2. Save filtering summary statistics
        summary = {
            'strategy': strategy,
            'analysis_type': builder.analysis_type,
            'target_size': probeset_size,
            'requested_panel_size': probeset_size,
            'final_panel_size': n_panel,
            'panel_size_status': panel_metadata.get('panel_size_status'),
            'short_panel_reason': panel_metadata.get('short_panel_reason'),
            'rf_deg_recompute_attempts': panel_metadata.get('rf_deg_recompute_attempts'),
            'rf_deg_candidate_pool_sizes': panel_metadata.get('rf_deg_candidate_pool_sizes'),
            'rf_deg_cache_used': panel_metadata.get('rf_deg_cache_used'),
            'rf_deg_cache_rejected_reason': panel_metadata.get('rf_deg_cache_rejected_reason'),
            'initial_selected': len(builder.get_genes_by_stage('initial')),
            'after_xenium': len(builder.get_genes_by_stage('post_xenium')) if apply_xenium_filter else len(builder.get_genes_by_stage('initial')),
            'final_selected': len(builder.get_selected_genes('final')),
            'xenium_failed': builder.count_filter_failures('xenium') if apply_xenium_filter else 0,
            'filters_applied': {
                'xenium': apply_xenium_filter,
                'blacklist_custom_patterns': blacklist_patterns or [],
                'blacklist_use_default': use_default_blacklist,
                'blacklist_any_active': bool(blacklist_patterns or use_default_blacklist),
            },
            'metadata': panel_metadata,
        }
        summary_output = os.path.join(results_dir, "filtering_summary.json")
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved filtering_summary.json")
        
        logger.info(f"✓ All results saved to: {results_dir}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("SELECTION PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Final panel size: {len(builder.get_selected_genes('final'))} genes")

    return builder
