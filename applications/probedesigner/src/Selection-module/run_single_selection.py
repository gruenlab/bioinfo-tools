"""
Refactored Gene Selection Wrapper with Filtering Pipeline
==========================================================

This module provides a clean wrapper for running single gene selection strategies
with the standardized filtering pipeline:

FILTERING ORDER (EXACT MATCH TO ORIGINAL):
------------------------------------------
1. BLACKLIST FILTER (PRE-selection): Remove genes from adata before selection
2. RUN SELECTION STRATEGY: Work on filtered adata
3. XENIUM FILTER (POST-selection): Apply to ranked gene list, celltype-aware
4. SELECT TOP N: From Xenium-filtered list
5. ODT FILTER (POST-selection): Apply to top N genes with replacement

This ensures all strategies use the exact same filtering logic as the original
pipeline, maintaining consistency and reproducibility.

Author: Refactored from run-selection.py
Date: 2026-02-08
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from anndata import AnnData

# Support script execution by adding module directory to sys.path
import sys
MODULE_DIR = Path(__file__).parent.absolute()
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

# Import refactored selection modules - use absolute imports
from _gene_list_builder import GeneListBuilder
from _filtering import (
    apply_blacklist_filter,
    apply_xenium_filter_to_genelist,
    apply_xenium_filter_global_to_genelist,
)
from _constants import (
    COL_SELECTED_INITIAL,
    COL_PASSED_XENIUM,
    COL_PASSED_ODT,
    COL_GENE,
    COL_SELECTION_SCORE,
)

from _odt_filtering import apply_odt_filter_with_replacement
from _design_probes import CompleteProbeDesignPipeline

from _deg_selection import select_DEGs
from _baseline_selection import (
    select_highly_variable_genes,
    select_random_genes,
    select_random_genes_bootstrap,
)
from _rf_selection import select_genes_with_rf
from _dimred_selection import (
    select_genes_from_nmf,
    select_genes_from_pca,
)
# Use absolute imports (for script execution)
from _constants import (
    DEFAULT_PROBESET_SIZE,
    DEFAULT_MIN_CELLS_PER_CELLTYPE,
    DEFAULT_RANDOM_STATE,
    COL_CELLTYPE,
    DEFAULT_MIN_XENIUM_EXPRESSION,
    DEFAULT_MAX_XENIUM_EXPRESSION,
    DEFAULT_ODT_METHOD,
    DEFAULT_ODT_MIN_PROBES_THRESHOLD,
    DEFAULT_REDUCTION_TYPES,
    DEFAULT_ANALYSIS_TYPES,
    DEFAULT_PER_FACTOR_SELECTION,
    DEFAULT_BLACKLIST_PATTERNS
    
)

logger = logging.getLogger(__name__)


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
    # ODT filter (POST-selection)
    apply_odt_filter: bool = False,
    odt_gtf_file: Optional[str] = None,
    odt_genome_file: Optional[str] = None,
    odt_species: str = "mus_musculus",
    odt_method: str = DEFAULT_ODT_METHOD,
    odt_annotation_source: str = "ensembl",
    odt_annotation_release: str = "110",
    odt_min_probes_threshold: int = DEFAULT_ODT_MIN_PROBES_THRESHOLD,
    odt_n_jobs: int = 4,
    odt_output_dir: Optional[str] = None,
    odt_reference_dir: Optional[str] = None,
    # Strategy-specific parameters
    reduction_type: Optional[str] = DEFAULT_REDUCTION_TYPES,  # 'nmf' or 'pca'
    analysis_type: str = DEFAULT_ANALYSIS_TYPES,  # 'global' or 'per_celltype'
    dimred_method: str = DEFAULT_PER_FACTOR_SELECTION,  # only 'method_a'
    n_components: int = 50,
    pool_size_per_celltype: int = 200,  # Pool size per celltype for Phase 1 in per_celltype mode
    pool_size_per_factor: int = 200,      # Pool size per factor for Phase 1 in global mode
    # NOTE: Dimred strategies (dimred_only) handle gap-filling internally via factor-aware
    # duplicate resolution. Gap-filling is automatic for both global and per-celltype analysis.
    # Caching (optional pre-computed results)
    deg_results: Optional[GeneListBuilder] = None,
    rf_simple_results: Optional[GeneListBuilder] = None,
    nmf_loadings_global: Optional[pd.DataFrame] = None,
    nmf_loadings_per_celltype: Optional[dict] = None,
    pca_loadings_global: Optional[pd.DataFrame] = None,
    pca_loadings_per_celltype: Optional[dict] = None,
    # cNMF options (only used when reduction_type == 'nmf')
    use_consensus_nmf: bool = False,
    k_min: int = 3,
    k_max: int = 15,
    k_step: int = 2,
    cnmf_n_iter: int = 20,
    k_selection_method: str = "silhouette",
    use_consensus_H: bool = False,
    # Output
    results_dir: Optional[str] = None,
    # Shared NMF/PCA model cache (independent of results_dir, reused across probeset sizes)
    nmf_model_cache_dir: Optional[str] = None,
) -> GeneListBuilder:
    """Run a single gene selection strategy with standardized filtering pipeline.

    This function implements the EXACT filtering order from the original pipeline:
    1. Blacklist filter (PRE-selection): Remove blacklisted genes from adata
    2. Run selection strategy: Work on filtered adata
    3. Xenium filter (POST-selection): Apply to ranked genes, celltype-aware
    4. Select top N: From Xenium-filtered list
    5. ODT filter (POST-selection): Apply to top N with replacement

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
        apply_xenium_filter: Whether to apply Xenium expression filter
        xenium_min_expr: Min expression threshold for Xenium
        xenium_max_expr: Max expression threshold for Xenium
        mean_expr_per_ct: Pre-computed per-celltype mean expression
        apply_odt_filter: Whether to apply ODT designability filter
        odt_gtf_file: Path to GTF file for ODT
        odt_genome_file: Path to genome FASTA for ODT
        odt_species: Species name for ODT (default: 'mus_musculus')
        odt_method: ODT probe design method ('SCRINSHOT', 'MERFISH', 'SEQFISHPLUS')
        odt_annotation_source: Annotation source ('ensembl' or 'ncbi')
        odt_annotation_release: Annotation release version (default: '110')
        odt_min_probes_threshold: Minimum successful probe sets required (default: 3)
        odt_n_jobs: Number of parallel jobs for ODT (default: 4)
        odt_output_dir: Custom ODT output directory (default: results_dir/odt_filtering)
        odt_reference_dir: Shared ODT reference directory (default: parent of results_dir/odt_references_shared)
        reduction_type: 'nmf' or 'pca' (for dimred_only strategy)
        analysis_type: 'global' or 'per_celltype'
        dimred_method: Gene selection method (only 'method_a' — top genes by absolute factor loading)
        n_components: Number of NMF/PCA components
        deg_results: Pre-computed DEG results (for caching)
        rf_simple_results: Pre-computed RF_simple results (for caching)
        nmf_loadings_global: Pre-computed global NMF loadings
        nmf_loadings_per_celltype: Pre-computed per-celltype NMF loadings
        pca_loadings_global: Pre-computed global PCA loadings
        pca_loadings_per_celltype: Pre-computed per-celltype PCA loadings
        results_dir: Directory to save results

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

        >>> # Random forest with DEG caching
        >>> builder = run_single_selection(
        ...     adata,
        ...     strategy='rf_deg',
        ...     deg_results=cached_deg_builder,  # Reuse DEGs
        ...     apply_odt_filter=True,
        ...     odt_gtf_file='genome.gtf'
        ... )

        >>> # NMF selection (per-celltype, Method A)
        >>> builder = run_single_selection(
        ...     adata,
        ...     strategy='dimred_only',
        ...     reduction_type='nmf',
        ...     analysis_type='per_celltype',
        ...     dimred_method='method_a',
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
        builder = select_DEGs(
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
            use_deg_prefilter=False,
            results_dir=results_dir,
        )

    elif strategy == "rf_deg":
        logger.info("Strategy: Random Forest (DEG-filtered genes)")
        builder = select_genes_with_rf(
            adata=adata,
            probeset_size=probeset_size,
            celltype_column=celltype_column,
            min_cells_per_celltype=min_cells_per_celltype,
            use_deg_prefilter=True,
            deg_results=deg_results,  # Use cached DEGs if available
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

    elif strategy == "random_bootstrap":
        logger.info(f"Strategy: Random Bootstrap (seed={random_state})")
        builder = select_random_genes_bootstrap(
            adata=adata,
            probeset_size=probeset_size,
            n_bootstrap=10,
            random_state=random_state,
            results_dir=results_dir,
        )

    elif strategy == "dimred_only":
        if reduction_type is None:
            raise ValueError("reduction_type required for dimred_only strategy (must be 'nmf' or 'pca')")

        logger.info(f"Strategy: Dimensionality Reduction ({reduction_type.upper()})")
        logger.info(f"  Analysis: {analysis_type}, Method: {dimred_method}")

        if reduction_type == "nmf":
            builder = select_genes_from_nmf(
                adata=adata,
                probeset_size=probeset_size,
                analysis_type=analysis_type,
                method=dimred_method,
                celltype_column=celltype_column,
                n_components=n_components,
                pool_size_per_celltype=pool_size_per_celltype,
                pool_size_per_factor=pool_size_per_factor,
                min_cells_per_celltype=min_cells_per_celltype,
                random_state=random_state,
                nmf_loadings_global=nmf_loadings_global,
                nmf_loadings_per_celltype=nmf_loadings_per_celltype,
                results_dir=results_dir,
                nmf_model_cache_dir=nmf_model_cache_dir,
                mean_expr_per_ct=mean_expr_per_ct,
                use_consensus_nmf=use_consensus_nmf,
                k_min=k_min,
                k_max=k_max,
                k_step=k_step,
                cnmf_n_iter=cnmf_n_iter,
                k_selection_method=k_selection_method,
                use_consensus_H=use_consensus_H,
            )
        elif reduction_type == "pca":
            builder = select_genes_from_pca(
                adata=adata,
                probeset_size=probeset_size,
                analysis_type=analysis_type,
                method=dimred_method,
                celltype_column=celltype_column,
                n_components=n_components,
                pool_size_per_celltype=pool_size_per_celltype,
                pool_size_per_factor=pool_size_per_factor,
                min_cells_per_celltype=min_cells_per_celltype,
                random_state=random_state,
                pca_loadings_global=pca_loadings_global,
                pca_loadings_per_celltype=pca_loadings_per_celltype,
                results_dir=results_dir,
                nmf_model_cache_dir=nmf_model_cache_dir,
                mean_expr_per_ct=mean_expr_per_ct,
            )
        else:
            raise ValueError(f"Invalid reduction_type: {reduction_type} (must be 'nmf' or 'pca')")

    else:
        raise ValueError(
            f"Invalid strategy: {strategy}. Must be one of: "
            "deg_only, rf_simple, rf_deg, hvg, random, random_bootstrap, dimred_only"
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

    selected_genes = builder.get_selected_genes('initial')

    # ========================================================================
    # STEP 5: ODT DESIGNABILITY FILTER WITH REPLACEMENT
    # ========================================================================
    # Apply ODT filter to final selected genes, replacing failed genes
    # with next-best candidates from the filtered list.
    # ========================================================================

    odt_filter_applied = False
    if apply_odt_filter:
        # Auto-download references if not provided or don't exist
        download_needed = False
        if odt_gtf_file is None or odt_genome_file is None:
            logger.info("Reference files not provided, will attempt to download...")
            download_needed = True
        elif not os.path.exists(odt_gtf_file) or not os.path.exists(odt_genome_file):
            logger.warning(
                f"Provided reference files not found:\n"
                f"  GTF: {odt_gtf_file} (exists: {os.path.exists(odt_gtf_file)})\n"
                f"  Genome: {odt_genome_file} (exists: {os.path.exists(odt_genome_file)})\n"
                f"Will attempt to download..."
            )
            download_needed = True
        
        if download_needed:
            try:
                if odt_reference_dir is None and results_dir:
                    odt_reference_dir = os.path.join(
                        os.path.dirname(os.path.abspath(results_dir)),
                        "odt_references_shared",
                    )
                elif odt_reference_dir is None:
                    odt_reference_dir = "./odt_references"

                os.makedirs(odt_reference_dir, exist_ok=True)

                logger.info(
                    f"Downloading reference genome for {odt_species} "
                    f"(source: {odt_annotation_source}, release: {odt_annotation_release})"
                )
                # Initialize pipeline for downloading only
                download_pipeline = CompleteProbeDesignPipeline(
                    pipeline_type="merfish",  # Type doesn't matter for downloads
                    output_dir=odt_reference_dir,
                    species=odt_species,
                    annotation_source=odt_annotation_source,
                    annotation_release=odt_annotation_release,
                    n_jobs=odt_n_jobs,
                )
                # Download references
                odt_gtf_file, odt_genome_file = download_pipeline.download_references()
                logger.info(
                    f"✓ Successfully downloaded references:\n"
                    f"  GTF: {odt_gtf_file}\n"
                    f"  Genome: {odt_genome_file}"
                )
            except Exception as e:
                logger.error(f"Failed to download reference files: {e}")
                logger.warning("Skipping ODT filter due to missing references")
                odt_gtf_file = None
                odt_genome_file = None
        
        if odt_gtf_file and odt_genome_file:
            # Prepare replacement pools and mappings for ODT (with factor awareness)
            df_all = builder.to_dataframe()
            
            # Get all non-selected genes sorted by score as replacement pool
            passed_filter = (df_all[COL_PASSED_XENIUM].fillna(True))
            replacement_candidates = df_all[
                (~df_all[COL_SELECTED_INITIAL]) & passed_filter
            ].sort_values(COL_SELECTION_SCORE, ascending=False)
            replacement_pool = replacement_candidates["gene"].tolist()
            replacement_pool_celltype_mapping = dict(
                zip(replacement_candidates["gene"], replacement_candidates["celltype"])
            )
            
            # FACTOR AWARENESS: Extract component mappings if available
            replacement_pool_component_mapping = None
            if 'component' in df_all.columns:
                replacement_pool_component_mapping = dict(
                    zip(replacement_candidates["gene"], replacement_candidates["component"])
                )
                logger.info(f"Factor-aware ODT replacement enabled: {len(set(replacement_pool_component_mapping.values()))} components tracked")
            
            # For combination strategies, prepare separate RF replacement pool
            rf_replacement_pool = None
            gene_source_mapping = None
            if "source" in df_all.columns:
                gene_source_mapping = dict(zip(df_all["gene"], df_all["source"]))
                
                # Build RF-specific replacement pool
                passed_filter = (df_all[COL_PASSED_XENIUM].fillna(True))
                rf_candidates = df_all[
                    (~df_all[COL_SELECTED_INITIAL]) & 
                    passed_filter & 
                    (df_all["source"].isin(["RF", "rf_deg"]))
                ].sort_values(COL_SELECTION_SCORE, ascending=False)
                
                if len(rf_candidates) > 0:
                    rf_replacement_pool = rf_candidates["gene"].tolist()
                    logger.info(f"Prepared RF replacement pool: {len(rf_replacement_pool)} genes")
            
            # Apply factor-aware ODT filter
            selected_genes = builder.get_selected_genes('initial')
            odt_results = apply_odt_filter_with_replacement(
                gene_list_builder=builder,
                selected_genes=selected_genes,
                replacement_pool=replacement_pool,
                replacement_pool_celltype_mapping=replacement_pool_celltype_mapping,
                replacement_pool_component_mapping=replacement_pool_component_mapping,  # FACTOR AWARENESS
                rf_replacement_pool=rf_replacement_pool,
                gene_source_mapping=gene_source_mapping,
                odt_method=odt_method,
                odt_output_dir=odt_output_dir,
                species=odt_species,
                annotation_source=odt_annotation_source,
                annotation_release=odt_annotation_release,
                gtf_file=odt_gtf_file,
                genome_file=odt_genome_file,
                min_probes_threshold=odt_min_probes_threshold,
                max_iterations=1000,
                n_jobs=odt_n_jobs,
                results_dir=results_dir,
                factor_aware=True  # Enable factor-aware replacement (Problem 3)
            )
            
            # Update builder with ODT results
            builder = odt_results['gene_list_builder']
            odt_filter_applied = True
            
            # Log factor balance if available
            if 'factor_balance_report' in odt_results:
                logger.info(f"✓ ODT filter complete with factor balance maintained")
            else:
                logger.info(f"✓ ODT filter complete: {len(builder.get_selected_genes())} genes in final panel")
    else:
        logger.info("ODT filter disabled (apply_odt_filter=False)")

    # If ODT is disabled (or could not be executed), keep final panel populated
    # by promoting the current initial selection to final.
    if (not apply_odt_filter) or (apply_odt_filter and not odt_filter_applied):
        selected_initial = builder.get_selected_genes('initial')
        if selected_initial:
            builder.mark_selected(selected_initial, selection_type='final')
            if not apply_odt_filter:
                logger.info(
                    f"✓ ODT disabled: promoted {len(selected_initial)} genes from initial to final"
                )
            else:
                logger.warning(
                    f"ODT requested but skipped/failed; promoted {len(selected_initial)} genes from initial to final"
                )

    # ========================================================================
    # SAVE FINAL RESULTS
    # ========================================================================
    # Output files:
    # 1. ranked_gene_list.csv  - Xenium-passing genes only:
    #      • Panel genes (final_selection=True) at the top, ranked by score
    #      • Remaining Xenium-passing genes below, ranked by score
    #      • in_panel column added for clear membership indicator
    # 2. filtering_summary.json - Statistics
    # ========================================================================

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

        logger.info("")
        logger.info("=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)

        # 1. Build unified ranked gene list -----------------------------------
        #    Scope: genes that passed Xenium (or all genes when Xenium disabled).
        #    Genes that failed Xenium are excluded — they cannot be in any panel.
        #    Layout: panel genes (in_panel=True) first, rest below; both groups
        #    sorted by rank ascending (rank 1 = highest scoring gene).
        full_df = builder.to_dataframe()

        if apply_xenium_filter:
            # Keep genes where passed_xenium is True or None (None → not tested,
            # e.g. force-include genes that bypass the filter).
            xenium_df = full_df[full_df['passed_xenium'].isin([True]) |
                                full_df['passed_xenium'].isna()].copy()
        else:
            xenium_df = full_df.copy()

        # Insert in_panel as the second column for readability
        xenium_df.insert(1, 'in_panel', xenium_df['final_selection'].fillna(False))

        # Panel genes first, within each group sort by rank (ascending)
        xenium_df = xenium_df.sort_values(
            ['in_panel', 'rank'],
            ascending=[False, True],
        ).reset_index(drop=True)

        ranked_output = os.path.join(results_dir, "ranked_gene_list.csv")
        xenium_df.to_csv(ranked_output, index=False)
        n_panel    = int(xenium_df['in_panel'].sum())
        n_remain   = len(xenium_df) - n_panel
        logger.info(
            f"✓ Saved ranked_gene_list.csv: "
            f"{n_panel} panel genes + {n_remain} remaining Xenium-passing genes "
            f"({len(xenium_df)} total)"
        )

        # 2. Save filtering summary statistics
        summary = {
            'strategy': strategy,
            'analysis_type': builder.analysis_type,
            'target_size': probeset_size,
            'initial_selected': len(builder.get_genes_by_stage('initial')),
            'after_xenium': len(builder.get_genes_by_stage('post_xenium')) if apply_xenium_filter else len(builder.get_genes_by_stage('initial')),
            'after_odt': len(builder.get_genes_by_stage('final')) if apply_odt_filter else len(builder.get_genes_by_stage('post_xenium') if apply_xenium_filter else builder.get_genes_by_stage('initial')),
            'final_selected': len(builder.get_selected_genes('final')),
            'xenium_failed': builder.count_filter_failures('xenium') if apply_xenium_filter else 0,
            'odt_failed': builder.count_filter_failures('odt') if apply_odt_filter else 0,
            'replacements': len(builder.get_replacement_genes()),
            'filters_applied': {
                'xenium': apply_xenium_filter,
                'odt': apply_odt_filter,
                'blacklist_custom_patterns': blacklist_patterns or [],
                'blacklist_use_default': use_default_blacklist,
                'blacklist_any_active': bool(blacklist_patterns or use_default_blacklist),
            }
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
