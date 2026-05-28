"""
GeneListBuilder for unified gene list tracking across selection pipeline.

This module provides the GeneListBuilder class, which standardizes gene list
output across all selection strategies. It tracks comprehensive metadata
through the entire pipeline: selection → filtering → replacement → final.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

# Use absolute imports (for script execution)
from _constants import (
    COL_GENE,
    COL_RANK,
    COL_SELECTION_SCORE,
    COL_SELECTION_STRATEGY,
    COL_ANALYSIS_TYPE,
    COL_CELLTYPE,
    COL_MEAN_EXPRESSION,
    COL_SELECTED_INITIAL,
    COL_PASSED_XENIUM,

    COL_FINAL_SELECTION,
    COL_XENIUM_FAILURE_REASON,

    COL_REPLACED_BY,
    COL_REPLACES_GENE,
    COL_REPLACEMENT_REASON,
)

logger = logging.getLogger(__name__)


class GeneListBuilder:
    """
    Builds unified gene lists with comprehensive metadata tracking.
    
    This class standardizes output across all gene selection strategies,
    maintaining complete provenance information for every gene regardless
    of whether it was selected, filtered, or replaced.
    
    Key features:
    - Tracks selection scores and rankings
    - Records filter pass/fail with reasons
    - Maintains replacement history
    - Supports incremental updates (selection → filtering → replacement)
    - Exports to pandas DataFrame or CSV
    
    Attributes:
        strategy_name: Name of selection strategy (e.g., 'deg_only', 'rf_nmf').
        analysis_type: Type of analysis ('global' or 'per_celltype').
        gene_records: Dict mapping gene names to metadata dicts.
        
    Examples:
        >>> builder = GeneListBuilder('deg_only', 'per_celltype')
        >>> builder.add_genes(
        ...     genes=['Gene1', 'Gene2'],
        ...     ranks=[1, 2],
        ...     scores=[0.001, 0.002],
        ...     celltype_mapping={'Gene1': 'T_cells', 'Gene2': 'B_cells'}
        ... )
        >>> builder.mark_selected(['Gene1'], selection_type='initial')
        >>> df = builder.to_dataframe()
        >>> df.shape[0]
        2
    """
    
    def __init__(
        self,
        strategy_name: str,
        analysis_type: str
    ) -> None:
        """
        Initialize GeneListBuilder.
        
        Args:
            strategy_name: Selection strategy name.
            analysis_type: 'global' or 'per_celltype'.
            
        Raises:
            ValueError: If analysis_type not 'global' or 'per_celltype'.
        """
        if analysis_type not in ['global', 'per_celltype']:
            raise ValueError(
                f"analysis_type must be 'global' or 'per_celltype', "
                f"got '{analysis_type}'"
            )
        
        self.strategy_name = strategy_name
        self.analysis_type = analysis_type
        self.gene_records: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}

        logger.debug(
            f"Initialized GeneListBuilder: strategy={strategy_name}, "
            f"analysis={analysis_type}"
        )
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Store a pipeline-level metadata key/value pair.

        Args:
            key: Metadata key name.
            value: Value to store.

        Examples:
            >>> builder.add_metadata('strategy', 'rf_nmf')
            >>> builder.metadata['strategy']
            'rf_nmf'
        """
        self.metadata[key] = value

    def add_genes(
        self,
        genes: List[str],
        ranks: List[int],
        scores: List[float],
        celltype_mapping: Optional[Dict[str, str]] = None,
        component_mapping: Optional[Dict[str, str]] = None,
        **extra_cols: Dict[str, List[Any]]
    ) -> None:
        """
        Add genes with metadata to the builder.
        
        This is the primary method for populating the gene list. Call this
        once with ALL genes from the selection strategy (not just top N),
        along with their rankings and scores.
        
        Args:
            genes: List of gene names.
            ranks: Rank of each gene (1 = best).
            scores: Selection score for each gene (strategy-specific).
            celltype_mapping: Optional dict mapping gene to celltype.
            component_mapping: Optional dict mapping gene to component/factor.
            **extra_cols: Additional metadata columns (e.g., pvalue, log2fc).
                Each value should be a list matching length of genes.
                
        Raises:
            ValueError: If genes, ranks, scores have different lengths.
            ValueError: If extra_cols lists don't match genes length.
            
        Examples:
            >>> builder.add_genes(
            ...     genes=['A', 'B'],
            ...     ranks=[1, 2],
            ...     scores=[8.5, 7.2],
            ...     celltype_mapping={'A': 'T_cells', 'B': 'T_cells'},
            ...     component_mapping={'A': 'NMF1', 'B': 'NMF2'},
            ...     pvalue=[0.001, 0.002],
            ...     log2fc=[2.5, 2.1]
            ... )
        """
        # Validate input lengths
        if not (len(genes) == len(ranks) == len(scores)):
            raise ValueError(
                f"genes, ranks, scores must have same length. "
                f"Got: genes={len(genes)}, ranks={len(ranks)}, "
                f"scores={len(scores)}"
            )
        
        for col_name, col_values in extra_cols.items():
            if len(col_values) != len(genes):
                raise ValueError(
                    f"Extra column '{col_name}' has length {len(col_values)}, "
                    f"expected {len(genes)}"
                )
        
        logger.debug(f"Adding {len(genes)} genes to builder")
        
        for idx, gene in enumerate(genes):
            # Initialize record with essential columns
            record = {
                COL_GENE: gene,
                COL_RANK: ranks[idx],
                COL_SELECTION_SCORE: scores[idx],
                COL_SELECTION_STRATEGY: self.strategy_name,
                COL_ANALYSIS_TYPE: self.analysis_type,
                COL_CELLTYPE: (
                    celltype_mapping.get(gene, 'global')
                    if celltype_mapping else 'global'
                ),
                COL_MEAN_EXPRESSION: None,  # Set later if available
                COL_SELECTED_INITIAL: False,
                COL_PASSED_XENIUM: None,
                COL_FINAL_SELECTION: False,
                COL_XENIUM_FAILURE_REASON: None,
                COL_REPLACED_BY: None,
                COL_REPLACES_GENE: None,
                COL_REPLACEMENT_REASON: None,
            }
            
            # Add component if provided
            if component_mapping:
                record['component'] = component_mapping.get(gene, None)
            
            # Add extra columns
            for col_name, col_values in extra_cols.items():
                record[col_name] = col_values[idx]
            
            # Store record
            self.gene_records[gene] = record
        
        logger.debug(
            f"✓ Added {len(genes)} genes to builder "
            f"(total: {len(self.gene_records)})"
        )
    
    def add_gene(
        self,
        gene: Optional[str] = None,
        selection_score: Optional[float] = None,
        rank: Optional[int] = None,
        celltype: Optional[str] = None,
        component: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # Aliases used by combination pipeline
        gene_name: Optional[str] = None,
        gene_source: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a single gene with metadata (convenience wrapper for add_genes).

        Accepts both the canonical parameter names (``gene``, ``metadata``) and
        the aliases used by the combination pipeline (``gene_name``,
        ``additional_metadata``, ``gene_source``).

        Args:
            gene: Gene name (canonical).
            selection_score: Selection score (strategy-specific).
            rank: Rank of gene (1 = best).
            celltype: Optional celltype assignment.
            component: Optional component/factor assignment.
            metadata: Optional dict of additional metadata fields.
            gene_name: Alias for ``gene``.
            gene_source: Optional source label stored in metadata.
            additional_metadata: Alias for ``metadata``.

        Examples:
            >>> builder.add_gene(
            ...     gene='GeneA',
            ...     selection_score=8.5,
            ...     rank=1,
            ...     celltype='T_cells',
            ...     component='NMF1',
            ...     metadata={'pvalue': 0.001}
            ... )
        """
        # Resolve aliases
        resolved_gene = gene if gene is not None else gene_name
        if resolved_gene is None:
            raise ValueError("Either 'gene' or 'gene_name' must be provided")
        resolved_meta = metadata if metadata is not None else additional_metadata

        # Merge gene_source into the metadata dict so it is persisted in the record
        if gene_source is not None:
            resolved_meta = dict(resolved_meta) if resolved_meta else {}
            resolved_meta.setdefault('gene_source', gene_source)

        # Prepare mappings
        celltype_mapping = {resolved_gene: celltype} if celltype else None
        component_mapping = {resolved_gene: component} if component else None

        # Prepare extra columns from metadata
        extra_cols = {}
        if resolved_meta:
            for key, value in resolved_meta.items():
                extra_cols[key] = [value]

        # Call add_genes with single-element lists
        self.add_genes(
            genes=[resolved_gene],
            ranks=[rank],
            scores=[selection_score],
            celltype_mapping=celltype_mapping,
            component_mapping=component_mapping,
            **extra_cols
        )
    
    def mark_selected(
        self,
        genes: Union[str, List[str]],
        selection_type: str = 'initial',
        rank: Optional[int] = None
    ) -> None:
        """
        Mark genes as selected at a specific stage.
        
        Args:
            genes: Gene name (str) or list of gene names (List[str]) to mark as selected.
            selection_type: Stage of selection ('initial', 'post_xenium', 'final').
            rank: Optional rank to update for the gene(s).
            
        Raises:
            ValueError: If selection_type is invalid.
            KeyError: If gene not found in gene_records.
            
        Examples:
            >>> builder.mark_selected(['Gene1', 'Gene2'], 'initial')
            >>> builder.mark_selected('Gene1', 'final')
            >>> builder.mark_selected('Gene1', 'initial', rank=5)
        """
        valid_types = ['initial', 'post_xenium', 'final']
        if selection_type not in valid_types:
            raise ValueError(
                f"selection_type must be one of {valid_types}, "
                f"got '{selection_type}'"
            )
        
        # Convert single gene to list
        if isinstance(genes, str):
            genes = [genes]
        
        logger.debug(
            f"Marking {len(genes)} genes as selected ({selection_type})"
        )
        
        for gene in genes:
            if gene not in self.gene_records:
                raise KeyError(
                    f"Gene '{gene}' not found in gene_records. "
                    "Call add_genes() first."
                )
            
            if selection_type == 'initial':
                self.gene_records[gene][COL_SELECTED_INITIAL] = True
            elif selection_type == 'final':
                self.gene_records[gene][COL_FINAL_SELECTION] = True
            
            # Update rank if provided
            if rank is not None:
                self.gene_records[gene][COL_RANK] = rank
        
        if len(genes) == 1:
            logger.info(f"✓ Marked '{genes[0]}' as {selection_type}")
        else:
            logger.info(f"✓ Marked {len(genes)} genes as {selection_type}")
    
    def mark_filter_result(
        self,
        gene: str,
        filter_name: str,
        passed: bool,
        failure_reason: Optional[str] = None
    ) -> None:
        """
        Record filter pass/fail for a gene.
        
        Args:
            gene: Gene name.
            filter_name: Name of filter ('xenium').
            passed: Whether gene passed the filter.
            failure_reason: Reason for failure if passed=False.

        Raises:
            ValueError: If filter_name is invalid.
            KeyError: If gene not found in gene_records.

        Examples:
            >>> builder.mark_filter_result('Gene1', 'xenium', True)
        """
        valid_filters = ['xenium']
        if filter_name not in valid_filters:
            raise ValueError(
                f"filter_name must be one of {valid_filters}, "
                f"got '{filter_name}'"
            )

        if gene not in self.gene_records:
            raise KeyError(
                f"Gene '{gene}' not found in gene_records. "
                "Call add_genes() first."
            )

        if filter_name == 'xenium':
            self.gene_records[gene][COL_PASSED_XENIUM] = passed
            if not passed and failure_reason:
                self.gene_records[gene][COL_XENIUM_FAILURE_REASON] = failure_reason
        
        logger.debug(
            f"Gene {gene}: {filter_name} filter "
            f"{'PASSED' if passed else 'FAILED'}"
            f"{' (' + failure_reason + ')' if failure_reason else ''}"
        )
    
    def record_replacement(
        self,
        failed_gene: str,
        replacement_gene: str,
        reason: str
    ) -> None:
        """
        Track gene replacements.
        
        Args:
            failed_gene: Gene that failed filter.
            replacement_gene: Gene that replaced it.
            reason: Reason for replacement (e.g., 'xenium_filter', 'duplicate').
            
        Raises:
            KeyError: If either gene not found in gene_records.
            
        Examples:
            >>> builder.record_replacement('Gene1', 'Gene2', 'xenium_filter')
        """
        if failed_gene not in self.gene_records:
            raise KeyError(
                f"Failed gene '{failed_gene}' not found in gene_records"
            )
        
        if replacement_gene not in self.gene_records:
            raise KeyError(
                f"Replacement gene '{replacement_gene}' not found in gene_records"
            )
        
        # Update failed gene
        self.gene_records[failed_gene][COL_REPLACED_BY] = replacement_gene
        self.gene_records[failed_gene][COL_REPLACEMENT_REASON] = reason
        
        # Update replacement gene
        self.gene_records[replacement_gene][COL_REPLACES_GENE] = failed_gene
        self.gene_records[replacement_gene][COL_REPLACEMENT_REASON] = reason
        
        logger.debug(
            f"Replacement recorded: {failed_gene} → {replacement_gene} "
            f"(reason: {reason})"
        )
    
    def set_mean_expression(
        self,
        mean_expr_dict: Dict[str, float]
    ) -> None:
        """
        Set mean expression values for genes.
        
        Args:
            mean_expr_dict: Dict mapping gene names to mean expression values.
            
        Examples:
            >>> builder.set_mean_expression({'Gene1': 12.5, 'Gene2': 8.3})
        """
        logger.debug(f"Setting mean expression for {len(mean_expr_dict)} genes")
        
        for gene, expr in mean_expr_dict.items():
            if gene in self.gene_records:
                self.gene_records[gene][COL_MEAN_EXPRESSION] = expr
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export gene list to pandas DataFrame.
        
        Returns:
            DataFrame with all gene records and metadata.
            Sorted by rank (ascending).
            
        Examples:
            >>> df = builder.to_dataframe()
            >>> df.columns.tolist()
            ['gene', 'rank', 'selection_score', ...]
        """
        if not self.gene_records:
            logger.warning("No genes in builder, returning empty DataFrame")
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(self.gene_records, orient='index')
        df = df.sort_values(COL_RANK)
        df = df.reset_index(drop=True)
        
        logger.debug(f"Exported DataFrame: {df.shape[0]} genes × {df.shape[1]} columns")
        
        return df
    
    def to_csv(
        self,
        output_path: Path | str,
        include_timestamp: bool = False
    ) -> None:
        """
        Save gene list to CSV file.
        
        Args:
            output_path: Path to output CSV file.
            include_timestamp: If True, add timestamp column.
            
        Examples:
            >>> builder.to_csv('results/ranked_gene_list.csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        
        if include_timestamp:
            df['timestamp'] = datetime.now().isoformat()
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"✓ Saved gene list to: {output_path}")
    
    def get_selected_genes(
        self,
        selection_type: str = 'final'
    ) -> List[str]:
        """
        Get list of selected genes at a specific stage.
        
        Args:
            selection_type: Which selection to retrieve ('initial', 'final').
            
        Returns:
            List of gene names, sorted by rank.
            
        Raises:
            ValueError: If selection_type is invalid.
            
        Examples:
            >>> initial_genes = builder.get_selected_genes('initial')
            >>> final_genes = builder.get_selected_genes('final')
        """
        valid_types = ['initial', 'final']
        if selection_type not in valid_types:
            raise ValueError(
                f"selection_type must be one of {valid_types}, "
                f"got '{selection_type}'"
            )
        
        col = COL_SELECTED_INITIAL if selection_type == 'initial' else COL_FINAL_SELECTION
        
        selected_genes = [
            gene for gene, record in self.gene_records.items()
            if record.get(col, False)
        ]
        
        # Sort by rank (handle None values by putting them last)
        selected_genes.sort(
            key=lambda g: (self.gene_records[g][COL_RANK] is None, self.gene_records[g][COL_RANK] or float('inf'))
        )
        
        logger.debug(f"Retrieved {len(selected_genes)} {selection_type} genes")
        
        return selected_genes
    
    def get_all_genes(self) -> List[str]:
        """
        Get all genes in the builder (both selected and non-selected).
        
        Returns:
            List of all gene names, sorted by rank.
            
        Examples:
            >>> builder = GeneListBuilder('deg_only', 'global')
            >>> builder.add_genes(['A', 'B', 'C'], [1, 2, 3], [0.1, 0.2, 0.3])
            >>> builder.get_all_genes()
            ['A', 'B', 'C']
        """
        all_genes = list(self.gene_records.keys())
        
        # Sort by rank (handle None values by putting them last)
        all_genes.sort(
            key=lambda g: (self.gene_records[g][COL_RANK] is None, self.gene_records[g][COL_RANK] or float('inf'))
        )
        
        logger.debug(f"Retrieved {len(all_genes)} total genes from builder")

        return all_genes

    def get_gene_metadata(self, gene: str) -> Optional[SimpleNamespace]:
        """
        Return metadata for a single gene as a SimpleNamespace.

        Provides convenient attribute-style access to the stored record fields
        needed when combining results across selection components.

        Args:
            gene: Gene name to look up.

        Returns:
            SimpleNamespace with attributes rank, selection_score, celltype,
            component, and component_loading, or None if the gene is not found.

        Examples:
            >>> meta = builder.get_gene_metadata('GeneA')
            >>> meta.rank
            1
            >>> meta.celltype
            'T_cells'
        """
        record = self.gene_records.get(gene)
        if record is None:
            return None
        return SimpleNamespace(
            rank=record.get(COL_RANK),
            selection_score=record.get(COL_SELECTION_SCORE),
            celltype=record.get(COL_CELLTYPE),
            component=record.get('component'),
            component_loading=record.get('component_loading'),
        )

    def get_filter_statistics(self) -> Dict[str, int]:
        """
        Get statistics about filter results.
        
        Returns:
            Dict with counts of passed/failed genes for each filter.
            
        Examples:
            >>> stats = builder.get_filter_statistics()
            >>> stats['xenium_passed']
            45
        """
        stats = {
            'total_genes': len(self.gene_records),
            'selected_initial': sum(
                1 for r in self.gene_records.values()
                if r.get(COL_SELECTED_INITIAL, False)
            ),
            'selected_final': sum(
                1 for r in self.gene_records.values()
                if r.get(COL_FINAL_SELECTION, False)
            ),
            'xenium_passed': sum(
                1 for r in self.gene_records.values()
                if r.get(COL_PASSED_XENIUM, False)
            ),
            'xenium_failed': sum(
                1 for r in self.gene_records.values()
                if r.get(COL_PASSED_XENIUM) is False
            ),
            'replacements_made': sum(
                1 for r in self.gene_records.values()
                if r.get(COL_REPLACES_GENE) is not None
            ),
        }
        
        return stats
    
    def summary(self) -> str:
        """
        Get a human-readable summary of the gene list.
        
        Returns:
            Multi-line string with key statistics.
            
        Examples:
            >>> print(builder.summary())
            GeneListBuilder Summary
            =======================
            Strategy: deg_only
            Analysis: per_celltype
            Total genes: 1000
            ...
        """
        stats = self.get_filter_statistics()
        
        summary_lines = [
            "GeneListBuilder Summary",
            "=======================",
            f"Strategy: {self.strategy_name}",
            f"Analysis: {self.analysis_type}",
            f"Total genes: {stats['total_genes']}",
            f"",
            "Selection:",
            f"  Initial: {stats['selected_initial']}",
            f"  Final: {stats['selected_final']}",
            f"",
            "Filtering:",
            f"  Xenium passed: {stats['xenium_passed']}",
            f"  Xenium failed: {stats['xenium_failed']}",
            f"",
            f"Replacements made: {stats['replacements_made']}",
        ]
        
        return "\n".join(summary_lines)
    
    def __repr__(self) -> str:
        """String representation of GeneListBuilder."""
        return (
            f"GeneListBuilder(strategy='{self.strategy_name}', "
            f"analysis='{self.analysis_type}', "
            f"n_genes={len(self.gene_records)})"
        )
    
    def __len__(self) -> int:
        """Return number of genes in builder."""
        return len(self.gene_records)
    
    def get_genes_by_stage(self, stage: str) -> List[str]:
        """
        Get genes that were selected at a specific pipeline stage.
        
        Args:
            stage: Pipeline stage ('initial', 'post_xenium', 'final').
            
        Returns:
            List of gene names at that stage.
            
        Raises:
            ValueError: If stage is invalid.
            
        Examples:
            >>> initial = builder.get_genes_by_stage('initial')
            >>> final = builder.get_genes_by_stage('final')
        """
        valid_stages = ['initial', 'post_xenium', 'final']
        if stage not in valid_stages:
            raise ValueError(
                f"stage must be one of {valid_stages}, got '{stage}'"
            )
        
        if stage == 'initial':
            return self.get_selected_genes('initial')
        elif stage == 'final':
            return self.get_selected_genes('final')
        else:  # post_xenium
            # Genes that passed xenium and were selected
            genes = [
                gene for gene, record in self.gene_records.items()
                if record.get(COL_SELECTED_INITIAL, False) and
                   record.get(COL_PASSED_XENIUM, True)  # True if not tested
            ]
            genes.sort(key=lambda g: (self.gene_records[g][COL_RANK] is None,
                                       self.gene_records[g][COL_RANK] or 0))
            return genes
    
    def count_filter_failures(self, filter_name: str) -> int:
        """
        Count how many genes failed a specific filter.
        
        Args:
            filter_name: Name of filter ('xenium').

        Returns:
            Number of genes that failed the filter.

        Raises:
            ValueError: If filter_name is invalid.

        Examples:
            >>> xenium_failures = builder.count_filter_failures('xenium')
        """
        valid_filters = ['xenium']
        if filter_name not in valid_filters:
            raise ValueError(
                f"filter_name must be one of {valid_filters}, got '{filter_name}'"
            )

        col = COL_PASSED_XENIUM
        
        return sum(
            1 for record in self.gene_records.values()
            if record.get(col) is False
        )
    
    def get_replacement_genes(self) -> List[str]:
        """
        Get genes that were added as replacements.
        
        Returns:
            List of replacement gene names.
            
        Examples:
            >>> replacements = builder.get_replacement_genes()
            >>> len(replacements)
            5
        """
        return [
            gene for gene, record in self.gene_records.items()
            if record.get(COL_REPLACES_GENE) is not None
        ]
