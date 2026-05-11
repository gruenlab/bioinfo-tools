"""
ODT (Oligo Designer Toolsuite) probe designability filtering with factor-aware replacement.

This module provides factor-aware and source-aware gene filtering using the
ODT pipeline to test probe designability. Genes that fail probe design are
replaced iteratively while maintaining factor balance and source composition.

Key Features:
- Factor-aware replacement: Maintains dimred component representation
- Source-aware replacement: Preserves RF/dimred composition in combination strategies
- Cell-type-aware replacement: Matches cell type for per-celltype methods
- Iterative testing: Replaces failed genes until convergence or max iterations
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Use absolute imports (for script execution)
from _gene_list_builder import GeneListBuilder
from _design_probes import CompleteProbeDesignPipeline

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

# ODT configuration defaults
DEFAULT_ODT_METHOD = 'SCRINSHOT'
DEFAULT_SPECIES = 'mus_musculus'
DEFAULT_ANNOTATION_SOURCE = 'ensembl'
DEFAULT_ANNOTATION_RELEASE = '110'
DEFAULT_MIN_PROBES_THRESHOLD = 3
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_N_JOBS = 4

# Replacement strategy constants
MIN_REPLACEMENT_CANDIDATES = 10  # Minimum candidates needed per pool
FACTOR_BALANCE_WARNING_THRESHOLD = 0.2  # Warn if factor imbalance > 20%


# ==============================================================================
# PUBLIC API
# ==============================================================================


def apply_odt_filter_with_replacement(
    gene_list_builder: GeneListBuilder,
    selected_genes: List[str],
    replacement_pool: List[str],
    replacement_pool_celltype_mapping: Optional[Dict[str, str]] = None,
    replacement_pool_component_mapping: Optional[Dict[str, str]] = None,
    rf_replacement_pool: Optional[List[str]] = None,
    gene_source_mapping: Optional[Dict[str, str]] = None,
    odt_method: str = DEFAULT_ODT_METHOD,
    odt_output_dir: Optional[str] = None,
    species: str = DEFAULT_SPECIES,
    annotation_source: str = DEFAULT_ANNOTATION_SOURCE,
    annotation_release: str = DEFAULT_ANNOTATION_RELEASE,
    gtf_file: Optional[str] = None,
    genome_file: Optional[str] = None,
    min_probes_threshold: int = DEFAULT_MIN_PROBES_THRESHOLD,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    n_jobs: int = DEFAULT_N_JOBS,
    results_dir: Optional[str] = None,
    factor_aware: bool = True,
) -> Dict[str, any]:
    """
    Apply ODT probe designability filtering with factor-aware iterative replacement.
    
    Tests genes for probe designability using the ODT pipeline. Genes that fail
    are replaced with next-best candidates while maintaining:
    - **Factor balance**: For dimred strategies, preserve component representation
    - **Source composition**: For combination strategies (rf_nmf/rf_pca), maintain RF/dimred ratio
    - **Cell type matching**: For per-celltype methods, replace within same cell type
    
    Args:
        gene_list_builder: Gene list with metadata including component assignments.
        selected_genes: Genes to test (ordered by rank/score).
        replacement_pool: Ranked list of replacement candidates.
            - For single strategies: ALL genes ranked by method score
            - For combination strategies: dimred genes only (RF uses separate pool)
        replacement_pool_celltype_mapping: Cell type mapping for replacement pool.
            Format: {gene: celltype}. Required for per-celltype methods.
        replacement_pool_component_mapping: Component/factor mapping for replacement pool.
            Format: {gene: component} (e.g., {'Sox2': 'NMF3', 'Gapdh': 'NMF1'}).
            Required for factor-aware replacement.
        rf_replacement_pool: Ranked list of RF genes for combination strategies.
            Required to maintain RF/dimred composition in rf_nmf/rf_pca.
        gene_source_mapping: Maps each gene to selection source.
            Format: {gene: source} where source is 'RF', 'dimred', or 'force_include'.
            Required for combination strategies.
        odt_method: ODT method ('SCRINSHOT', 'MERFISH', 'SEQFISHPLUS').
        odt_output_dir: Directory for ODT intermediate files.
        species: Species name (default: 'mus_musculus').
        annotation_source: 'ensembl' or 'ncbi' (default: 'ensembl').
        annotation_release: Annotation release version (default: '110').
        gtf_file: Path to GTF file (if None, will download).
        genome_file: Path to genome FASTA file (if None, will download).
        min_probes_threshold: Minimum successful probe sets required (default: 3).
        max_iterations: Maximum replacement iterations (default: 1000).
        n_jobs: Number of parallel jobs for ODT (default: 4).
        results_dir: Directory to save filtering report.
        factor_aware: If True, maintain factor balance during replacement (default: True).
        
    Returns:
        Dictionary containing:
            - 'final_genes': List of genes passing ODT or replaced
            - 'removed_genes': List of genes that failed ODT
            - 'replacement_genes': List of replacement genes added
            - 'replacement_mapping': Dict {replacement_gene: removed_gene}
            - 'factor_balance_report': Dict showing per-factor success/failure rates
            - 'filtering_report': Dict with detailed statistics
            - 'gene_list_builder': Updated builder with ODT results
            
    Raises:
        FileNotFoundError: If GTF/genome files not found and download fails.
        ValueError: If replacement_pool_component_mapping required but not provided.
        RuntimeError: If ODT pipeline initialization fails.
        
    Examples:
        >>> # Single strategy (NMF) with factor awareness
        >>> result = apply_odt_filter_with_replacement(
        ...     gene_list_builder=builder,
        ...     selected_genes=['Sox2', 'Gapdh', 'Actb'],
        ...     replacement_pool=all_nmf_genes,
        ...     replacement_pool_component_mapping={
        ...         'Sox2': 'NMF1', 'Gapdh': 'NMF2', ...
        ...     },
        ...     gtf_file='mouse.gtf',
        ...     genome_file='mouse.fa',
        ...     factor_aware=True
        ... )
        >>> print(result['factor_balance_report'])
        {'NMF1': {'initial': 10, 'passed': 8, 'replaced': 2}, ...}
        
        >>> # Combination strategy (rf_nmf) with source + factor awareness
        >>> result = apply_odt_filter_with_replacement(
        ...     gene_list_builder=builder,
        ...     selected_genes=rf_nmf_genes,
        ...     replacement_pool=nmf_ranked_genes,
        ...     replacement_pool_component_mapping=nmf_component_map,
        ...     rf_replacement_pool=rf_ranked_genes,
        ...     gene_source_mapping={'Gapdh': 'RF', 'Sox2': 'dimred'},
        ...     factor_aware=True
        ... )
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"APPLYING ODT PROBE DESIGNABILITY FILTERING ({odt_method})")
    logger.info("=" * 80)
    logger.info(f"Selected genes to test: {len(selected_genes)}")
    logger.info(f"Replacement pool size: {len(replacement_pool)}")
    if rf_replacement_pool:
        logger.info(f"RF replacement pool size: {len(rf_replacement_pool)}")
    logger.info(f"Factor-aware replacement: {factor_aware}")
    logger.info(f"Min probes threshold: {min_probes_threshold}")
    logger.info(f"Max iterations: {max_iterations}")
    
    # Validate inputs
    if factor_aware and not replacement_pool_component_mapping:
        logger.warning(
            "Factor-aware replacement requested but replacement_pool_component_mapping "
            "not provided. Falling back to non-factor-aware replacement."
        )
        factor_aware = False
    
    # Setup ODT output directory
    if odt_output_dir is None:
        odt_output_dir = os.path.join(results_dir, 'odt_filtering') if results_dir else './odt_filtering'
    os.makedirs(odt_output_dir, exist_ok=True)
    
    # Initialize ODT pipeline
    logger.info(f"Initializing {odt_method} probe designer...")
    try:
        pipeline = CompleteProbeDesignPipeline(
            pipeline_type=odt_method.lower(),
            output_dir=odt_output_dir,
            species=species,
            annotation_source=annotation_source,
            annotation_release=annotation_release,
            design_mode='filter_only',  # Fast mode - only test designability
            write_intermediate_steps=False,
            n_jobs=n_jobs
        )
        logger.info("✓ ODT pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ODT pipeline: {e}")
        raise RuntimeError(f"ODT pipeline initialization failed: {e}") from e
    
    # Download or use provided references
    if gtf_file is None or genome_file is None:
        logger.info("Downloading reference genome and annotations...")
        try:
            gtf_file, genome_file = pipeline.download_references()
            logger.info(f"✓ References downloaded: GTF={gtf_file}, Genome={genome_file}")
        except Exception as e:
            logger.error(f"Failed to download references: {e}")
            raise FileNotFoundError(f"Could not download references: {e}") from e
    else:
        pipeline.gtf_file = gtf_file
        pipeline.genome_file = genome_file
        logger.info(f"Using provided references: GTF={gtf_file}, Genome={genome_file}")
    
    # Extract metadata from builder
    df = gene_list_builder.to_dataframe()
    gene_celltype_mapping = dict(zip(df['gene'], df['celltype']))
    gene_component_mapping = dict(zip(df['gene'], df.get('component', ['global'] * len(df))))
    
    # Initialize tracking
    current_genes = list(selected_genes)  # Work with a copy
    tested_genes: Set[str] = set()
    
    # Build replacement pools by factor and source
    replacement_pools = _build_factor_aware_replacement_pools(
        replacement_pool=replacement_pool,
        replacement_pool_celltype_mapping=replacement_pool_celltype_mapping,
        replacement_pool_component_mapping=replacement_pool_component_mapping,
        rf_replacement_pool=rf_replacement_pool,
        factor_aware=factor_aware
    )
    
    # Initialize result tracking
    results = {
        'final_genes': [],
        'removed_genes': [],
        'replacement_genes': [],
        'replacement_mapping': {},
        'factor_balance_report': {},
        'filtering_report': {},
        'odt_results': {}
    }
    
    # Track factor representation
    if factor_aware:
        factor_tracker = _initialize_factor_tracker(
            selected_genes=selected_genes,
            gene_component_mapping=gene_component_mapping
        )
        logger.info(f"Tracking {len(factor_tracker)} factors/components")
    
    # Iterative testing and replacement
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        genes_to_test = [g for g in current_genes if g not in tested_genes]
        
        if len(genes_to_test) == 0:
            logger.info(f"Iteration {iteration}: No new genes to test, filtering complete")
            break
        
        logger.info(f"Iteration {iteration}: Testing {len(genes_to_test)} genes for designability...")
        
        # Test genes with ODT
        successful_genes, failed_genes = _test_genes_with_odt(
            pipeline=pipeline,
            genes_to_test=genes_to_test,
            odt_method=odt_method,
            min_probes_threshold=min_probes_threshold
        )
        
        # Mark all tested genes
        tested_genes.update(genes_to_test)
        
        if len(failed_genes) == 0:
            logger.info(f"Iteration {iteration}: All {len(genes_to_test)} genes passed ODT")
            break
        
        logger.info(f"Iteration {iteration}: {len(failed_genes)} genes failed probe design")
        
        # Update factor tracker if factor-aware
        if factor_aware:
            for gene in failed_genes:
                component = gene_component_mapping.get(gene, 'global')
                if component in factor_tracker:
                    factor_tracker[component]['failed'] += 1
        
        # Replace failed genes
        for gene in failed_genes:
            current_genes.remove(gene)
            results['removed_genes'].append(gene)
            
            gene_ct = gene_celltype_mapping.get(gene, 'global')
            gene_component = gene_component_mapping.get(gene, 'global')
            # For single strategies (no source mapping), default to 'dimred'
            if gene_source_mapping:
                gene_source = gene_source_mapping.get(gene, 'dimred')
            else:
                gene_source = 'dimred'  # Single strategies are all dimred-based
            
            # Find replacement using factor/source-aware strategy
            replacement = _find_replacement_gene(
                failed_gene=gene,
                gene_celltype=gene_ct,
                gene_component=gene_component,
                gene_source=gene_source,
                replacement_pools=replacement_pools,
                tested_genes=tested_genes,
                current_genes=set(current_genes),
                factor_aware=factor_aware
            )
            
            if replacement:
                current_genes.append(replacement)
                results['replacement_genes'].append(replacement)
                results['replacement_mapping'][replacement] = gene
                
                # Assign same metadata to replacement
                gene_celltype_mapping[replacement] = gene_ct
                gene_component_mapping[replacement] = gene_component
                if gene_source_mapping and gene_source:
                    gene_source_mapping[replacement] = gene_source
                
                logger.debug(
                    f"  Replaced {gene} (component={gene_component}, source={gene_source}) "
                    f"with {replacement}"
                )
            else:
                source_info = f", source={gene_source}" if gene_source else ""
                logger.warning(
                    f"  No replacement found for {gene} "
                    f"(celltype={gene_ct}, component={gene_component}{source_info})"
                )
    
    # Finalize results
    results['final_genes'] = current_genes
    
    # Generate factor balance report
    if factor_aware:
        results['factor_balance_report'] = _generate_factor_balance_report(
            factor_tracker=factor_tracker,
            final_genes=current_genes,
            gene_component_mapping=gene_component_mapping
        )
        _log_factor_balance_report(results['factor_balance_report'])
    
    # Update gene_list_builder with ODT results
    _update_builder_with_odt_results(
        gene_list_builder=gene_list_builder,
        removed_genes=results['removed_genes'],
        final_genes=results['final_genes'],
        replacement_mapping=results['replacement_mapping']
    )
    
    # Add updated GeneListBuilder to results
    results['gene_list_builder'] = gene_list_builder
    
    # Create summary report
    report = {
        'initial_size': len(selected_genes),
        'final_size': len(current_genes),
        'n_removed': len(results['removed_genes']),
        'n_replaced': len(results['replacement_genes']),
        'n_iterations': iteration,
        'odt_method': odt_method,
        'min_probes_threshold': min_probes_threshold,
        'factor_aware': factor_aware
    }
    results['filtering_report'] = report
    
    # Save report
    if results_dir:
        _save_filtering_report(report, results_dir, odt_method, results.get('factor_balance_report'))
    
    logger.info("=" * 80)
    logger.info("ODT FILTERING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Removed: {report['n_removed']} genes (failed probe design)")
    logger.info(f"Replaced: {report['n_replaced']} genes")
    logger.info(f"Final panel: {report['final_size']} genes")
    logger.info(f"Iterations: {report['n_iterations']}")
    logger.info("=" * 80)
    
    return results


# ==============================================================================
# PRIVATE HELPER FUNCTIONS
# ==============================================================================


def _build_factor_aware_replacement_pools(
    replacement_pool: List[str],
    replacement_pool_celltype_mapping: Optional[Dict[str, str]],
    replacement_pool_component_mapping: Optional[Dict[str, str]],
    rf_replacement_pool: Optional[List[str]],
    factor_aware: bool
) -> Dict[str, Dict[any, List[str]]]:
    """
    Build replacement pools organized by source, cell-type, and factor/component.
    
    Creates a nested structure for efficient replacement lookup:
    
    For per-celltype strategies:
    {
        'dimred': {
            ('T_cells', 'NMF1'): ['Gene1', 'Gene2', ...],  # Sorted by rank
            ('T_cells', 'NMF2'): ['Gene3', 'Gene4', ...],
            ('B_cells', 'NMF1'): ['Gene5', 'Gene6', ...],
            ...
        },
        'RF': {
            'global': ['Gene10', 'Gene11', ...]
        }
    }
    
    For global strategies:
    {
        'dimred': {
            'NMF1': ['Gene1', 'Gene2', ...],
            'NMF2': ['Gene3', 'Gene4', ...],
            ...
        },
        'RF': {
            'global': ['Gene10', 'Gene11', ...]
        }
    }
    
    Args:
        replacement_pool: List of ranked dimred genes.
        replacement_pool_celltype_mapping: Cell type assignments.
        replacement_pool_component_mapping: Factor/component assignments.
        rf_replacement_pool: List of ranked RF genes (optional).
        factor_aware: Whether to organize by factor.
        
    Returns:
        Nested dict: {source: {key: [genes]}} where key is either component
        or (celltype, component) tuple depending on strategy.
    """
    # Detect per-celltype mode
    per_celltype_mode = replacement_pool_celltype_mapping is not None
    
    logger.debug(
        f"Building {'celltype + factor' if per_celltype_mode else 'factor'}-aware "
        f"replacement pools..."
    )
    
    pools: Dict[str, Dict[any, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    # Build dimred pool (organized by factor and optionally celltype)
    for gene in replacement_pool:
        if factor_aware and replacement_pool_component_mapping:
            component = replacement_pool_component_mapping.get(gene, 'global')
        else:
            component = 'global'
        
        # Use (celltype, component) key for per-celltype, component only for global
        if per_celltype_mode:
            celltype = replacement_pool_celltype_mapping.get(gene, 'unknown')
            pool_key = (celltype, component)
        else:
            pool_key = component
        
        pools['dimred'][pool_key].append(gene)
    
    # Build RF pool (always global - no factor structure for RF genes)
    if rf_replacement_pool:
        pools['RF']['global'] = list(rf_replacement_pool)
    
    # Log pool sizes
    total_dimred_genes = sum(len(genes) for genes in pools['dimred'].values())
    total_rf_genes = sum(len(genes) for genes in pools.get('RF', {}).values())
    logger.info(
        f"Built replacement pools: {total_dimred_genes} dimred genes "
        f"across {len(pools['dimred'])} groups"
        + (f", {total_rf_genes} RF genes" if total_rf_genes > 0 else "")
    )
    for source, component_pools in pools.items():
        for pool_key, genes in component_pools.items():
            logger.debug(f"  Pool {source}/{pool_key}: {len(genes)} genes")
    
    return pools


def _initialize_factor_tracker(
    selected_genes: List[str],
    gene_component_mapping: Dict[str, str]
) -> Dict[str, Dict[str, int]]:
    """
    Initialize tracker for monitoring factor balance during ODT filtering.
    
    Args:
        selected_genes: Initial gene selection.
        gene_component_mapping: Gene-to-component assignments.
        
    Returns:
        Dict tracking per-factor counts: {component: {'initial': N, 'failed': 0}}
    """
    factor_counts = defaultdict(lambda: {'initial': 0, 'failed': 0})
    
    for gene in selected_genes:
        component = gene_component_mapping.get(gene, 'global')
        factor_counts[component]['initial'] += 1
    
    logger.debug(f"Initialized factor tracker for {len(factor_counts)} factors")
    for component, counts in factor_counts.items():
        logger.debug(f"  {component}: {counts['initial']} genes")
    
    return dict(factor_counts)


def _test_genes_with_odt(
    pipeline: CompleteProbeDesignPipeline,
    genes_to_test: List[str],
    odt_method: str,
    min_probes_threshold: int
) -> Tuple[List[str], List[str]]:
    """
    Test genes for probe designability using ODT pipeline.
    
    Args:
        pipeline: Initialized ODT pipeline.
        genes_to_test: Genes to test.
        odt_method: ODT method name.
        min_probes_threshold: Minimum probes required.
        
    Returns:
        Tuple of (successful_genes, failed_genes)
    """
    try:
        # Extract gene sequences
        pipeline.extract_gene_sequences(
            gene_list=genes_to_test,
            region_type="transcript",
            auto_convert_symbols=True
        )
        
        # Run probe design
        if odt_method.upper() == 'SCRINSHOT':
            odt_output = pipeline.run_scrinshot_design(
                gene_ids=pipeline.current_gene_ids,
                set_size_min=min_probes_threshold
            )
        elif odt_method.upper() == 'MERFISH':
            odt_output = pipeline.run_merfish_design(
                gene_ids=pipeline.current_gene_ids,
                set_size_min=min_probes_threshold
            )
        elif odt_method.upper() == 'SEQFISHPLUS':
            odt_output = pipeline.run_seqfishplus_design(
                gene_ids=pipeline.current_gene_ids,
                set_size_min=min_probes_threshold
            )
        else:
            raise ValueError(f"Unknown ODT method: {odt_method}")
        
        # Get successful genes
        successful_symbols = set(odt_output['successful_gene_symbols'])
        successful_genes = [g for g in genes_to_test if g in successful_symbols]
        failed_genes = [g for g in genes_to_test if g not in successful_symbols]
        
        return successful_genes, failed_genes
        
    except Exception as e:
        logger.error(f"ODT testing failed: {e}")
        # Treat all genes as failed if ODT crashes
        return [], genes_to_test


def _find_replacement_gene(
    failed_gene: str,
    gene_celltype: str,
    gene_component: str,
    gene_source: Optional[str],
    replacement_pools: Dict[str, Dict[any, List[str]]],
    tested_genes: Set[str],
    current_genes: Set[str],
    factor_aware: bool
) -> Optional[str]:
    """
    Find replacement gene using factor/source/celltype-aware strategy.
    
    Replacement priority:
    1. Match source (RF vs dimred)
    2. Match cell type (if per-celltype mode detected)
    3. Match factor/component (if factor_aware)
    4. Not already tested or selected
    
    Args:
        failed_gene: Gene that failed ODT.
        gene_celltype: Cell type assignment.
        gene_component: Factor/component assignment.
        gene_source: Source ('RF' or 'dimred').
        replacement_pools: Organized replacement candidates.
        tested_genes: Genes already tested.
        current_genes: Genes currently selected.
        factor_aware: Whether to match factor.
        
    Returns:
        Replacement gene name or None if no suitable candidate found.
    """
    # Determine source pool
    if gene_source == 'RF' or gene_source == 'rf_deg':
        source_pools = replacement_pools.get('RF', {})
        pool_key = 'global'  # RF genes don't have factor structure
    elif gene_source == 'force_include':
        logger.warning(f"Force-include gene {failed_gene} failed ODT - not replacing")
        return None
    else:
        source_pools = replacement_pools.get('dimred', {})
        
        # Detect per-celltype mode by checking if any key is a tuple
        per_celltype_mode = any(isinstance(k, tuple) for k in source_pools.keys())
        
        # Build pool key based on strategy
        if per_celltype_mode:
            # Per-celltype: use (celltype, component) key
            component = gene_component if factor_aware else 'global'
            pool_key = (gene_celltype, component)
        else:
            # Global: use component only
            pool_key = gene_component if factor_aware else 'global'
    
    # Get candidate pool for this factor/celltype combination
    candidate_pool = source_pools.get(pool_key, [])
    
    if not candidate_pool:
        logger.warning(
            f"No candidates in pool for source={gene_source}, key={pool_key}"
        )
        return None
    
    # Find first suitable candidate
    for candidate in candidate_pool:
        if candidate in tested_genes or candidate in current_genes:
            continue
        
        # All checks passed
        return candidate
    
    # No suitable candidate found
    return None


def _generate_factor_balance_report(
    factor_tracker: Dict[str, Dict[str, int]],
    final_genes: List[str],
    gene_component_mapping: Dict[str, str]
) -> Dict[str, Dict[str, int]]:
    """
    Generate report showing factor representation before and after ODT filtering.
    
    Args:
        factor_tracker: Initial counts and failures per factor.
        final_genes: Final selected genes after ODT.
        gene_component_mapping: Gene-to-component assignments.
        
    Returns:
        Report dict: {component: {'initial': N, 'failed': M, 'final': K, 'retention': %}}
    """
    report = {}
    
    # Count final genes per factor
    final_counts = defaultdict(int)
    for gene in final_genes:
        component = gene_component_mapping.get(gene, 'global')
        final_counts[component] += 1
    
    # Build report
    for component, counts in factor_tracker.items():
        initial = counts['initial']
        failed = counts['failed']
        final = final_counts.get(component, 0)
        
        report[component] = {
            'initial': initial,
            'failed': failed,
            'final': final,
            'retention_pct': (final / initial * 100) if initial > 0 else 0
        }
    
    return report


def _log_factor_balance_report(report: Dict[str, Dict[str, int]]) -> None:
    """Log factor balance report in readable format."""
    logger.info("")
    logger.info("Factor Balance Report:")
    logger.info("=" * 60)
    logger.info(f"{'Component':<15} {'Initial':>8} {'Failed':>8} {'Final':>8} {'Retention':>10}")
    logger.info("-" * 60)
    
    for component, counts in sorted(report.items()):
        logger.info(
            f"{component:<15} {counts['initial']:>8} {counts['failed']:>8} "
            f"{counts['final']:>8} {counts['retention_pct']:>9.1f}%"
        )
    
    logger.info("=" * 60)


def _update_builder_with_odt_results(
    gene_list_builder: GeneListBuilder,
    removed_genes: List[str],
    final_genes: List[str],
    replacement_mapping: Dict[str, str]
) -> None:
    """
    Update GeneListBuilder with ODT filter results.
    
    Args:
        gene_list_builder: Builder to update.
        removed_genes: Genes that failed ODT.
        final_genes: Genes in final panel.
        replacement_mapping: {replacement_gene: removed_gene}
    """
    # Mark ODT filter results
    for gene in removed_genes:
        gene_list_builder.mark_filter_result(
            gene=gene,
            filter_name='odt',
            passed=False,
            failure_reason='insufficient_probes'
        )
    
    for gene in final_genes:
        if gene in replacement_mapping:
            # This is a replacement gene
            original_gene = replacement_mapping[gene]
            gene_list_builder.record_replacement(
                failed_gene=original_gene,
                replacement_gene=gene,
                reason='odt_filter'
            )
        
        gene_list_builder.mark_filter_result(
            gene=gene,
            filter_name='odt',
            passed=True
        )
    
    # Mark final selections
    gene_list_builder.mark_selected(final_genes, selection_type='final')
    
    logger.debug(f"Updated builder with ODT results for {len(final_genes)} genes")


def _save_filtering_report(
    report: Dict[str, any],
    results_dir: str,
    odt_method: str,
    factor_balance_report: Optional[Dict[str, Dict[str, int]]]
) -> None:
    """Save ODT filtering report to JSON file."""
    import json
    
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, f'odt_{odt_method.lower()}_filtering_report.json')
    
    # Add factor balance if available
    if factor_balance_report:
        report['factor_balance'] = factor_balance_report
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✓ Saved ODT filtering report to {report_path}")
