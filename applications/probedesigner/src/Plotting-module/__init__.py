"""Plotting utilities for spatial probe design pipeline.

Provides visualization functions for evaluation metrics (clustering,
neighborhood preservation, cell-type classification, NMF variability)
and selection results (gene dotplots, provenance charts).

CLI entry points:
    plot_evaluation.py        — Evaluation results visualization
    plot_nmf_independent.py   — NMF reconstruction comparison plots
    plot_umaps.py             — UMAP visualization of preprocessed panels
    plot_pipeline_results.py  — Comprehensive plotting for all analysis types
    plot_raw_vs_log_factor_umaps.py — Raw-vs-log NMF factor UMAP grids

Internal modules:
    _clustering_plots   — Clustering, kNN, cell-type classification plots
    _variability_plots  — NMF variability metric plots
    _selection_plots    — Gene selection result plots
    _stability_plots    — Stability analysis plots (gene frequency, overlap, metrics)
    _k_varying_plots    — K-varying analysis plots (reconstruction quality, gene stability)
    _comparison_umaps   — Raw-vs-log factor UMAP comparison plots
    _constants          — Shared column names and plot settings
"""

from __future__ import annotations

from ._clustering_plots import (
    extract_display_name_from_dataset,
    extract_detailed_display_name_for_heatmap,
    extract_factor_number_from_dataset_name,
    extract_strategy_from_dataset_name,
    generate_category_9_label,
    generate_method_specific_colors,
    generate_method_specific_colors_and_markers,
    generate_panel_size_colors_and_strategy_markers,
    plot_celltype_accuracy_barchart,
    plot_celltype_f1_heatmap,
    plot_clustering_quality_ari,
    plot_clustering_quality_nmi,
    plot_confusion_matrix,
    plot_decision_tree_visualization,
    plot_feature_importance_heatmap,
    plot_neighborhood_preservation_by_k,
    plot_neighborhood_preservation_heatmap,
    plot_optimal_neighborhood_preservation,
    plot_umap_for_representation,
    resolve_marker_strategy,
    create_feature_plots,
)
from ._variability_plots import (
    create_combined_plot_with_info,
    extract_display_name_for_factor_range,
    extract_strategy_and_display_name,
    get_category_colors,
    get_factor_specific_colors,
    is_factor_range_mode,
    plot_aggregated_celltype_metrics,
    plot_celltype_evaluation_results,
)
from ._selection_plots import (
    plot_confusion_matrix as plot_confusion_matrix_selection,
    plot_f1_distribution,
    plot_feature_importances,
    plot_final_gene_dotplot,
    plot_gene_count_comparison,
    plot_gene_source_distribution,
)
from ._stability_plots import (
    plot_aggregate_metrics_summary,
    plot_feature_umaps,
    plot_gene_frequency,
    plot_gene_overlap,
    plot_metrics_summary,
)
from ._k_varying_plots import (
    plot_reconstruction_quality,
    plot_gene_stability,
    plot_aggregate_metrics,
    plot_reconstruction_metrics_grid,
    plot_baseline_comparison,
    plot_per_celltype_grid,
)
from ._comparison_umaps import (
    ensure_umap,
    infer_celltype_column,
    plot_celltype_umap,
    plot_raw_vs_log_factor_umap_grid,
)

__all__ = [
    # Clustering / kNN / cell-type plots
    "extract_display_name_from_dataset",
    "extract_detailed_display_name_for_heatmap",
    "extract_factor_number_from_dataset_name",
    "extract_strategy_from_dataset_name",
    "generate_category_9_label",
    "generate_method_specific_colors",
    "generate_method_specific_colors_and_markers",
    "generate_panel_size_colors_and_strategy_markers",
    "plot_celltype_accuracy_barchart",
    "plot_celltype_f1_heatmap",
    "plot_clustering_quality_ari",
    "plot_clustering_quality_nmi",
    "plot_confusion_matrix",
    "plot_decision_tree_visualization",
    "plot_feature_importance_heatmap",
    "plot_neighborhood_preservation_by_k",
    "plot_neighborhood_preservation_heatmap",
    "plot_optimal_neighborhood_preservation",
    "plot_umap_for_representation",
    "resolve_marker_strategy",
    "create_feature_plots",
    # Variability plots
    "create_combined_plot_with_info",
    "extract_display_name_for_factor_range",
    "extract_strategy_and_display_name",
    "get_category_colors",
    "get_factor_specific_colors",
    "is_factor_range_mode",
    "plot_aggregated_celltype_metrics",
    "plot_celltype_evaluation_results",
    # Selection plots
    "plot_f1_distribution",
    "plot_feature_importances",
    "plot_final_gene_dotplot",
    "plot_gene_count_comparison",
    "plot_gene_source_distribution",
    # Stability analysis plots
    "plot_aggregate_metrics_summary",
    "plot_feature_umaps",
    "plot_gene_frequency",
    "plot_gene_overlap",
    "plot_metrics_summary",
    # K-varying analysis plots
    "plot_reconstruction_quality",
    "plot_gene_stability",
    "plot_aggregate_metrics",
    "plot_reconstruction_metrics_grid",
    "plot_baseline_comparison",
    "plot_per_celltype_grid",
    # Raw-vs-log factor UMAP comparison plots
    "ensure_umap",
    "infer_celltype_column",
    "plot_celltype_umap",
    "plot_raw_vs_log_factor_umap_grid",
]

__version__ = "2.0.0"
