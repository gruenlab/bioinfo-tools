"""Constants for the plotting pipeline.

Defines CSV column names read from evaluation/selection results,
shared plot settings, and file naming conventions.
"""

from __future__ import annotations

# =============================================================================
# EVALUATION RESULT CSV COLUMN NAMES
# (These must match exactly what Evaluation-module writes to disk)
# =============================================================================

# --- Shared identifier columns ---
COL_DATASET: str = "dataset"
COL_REPRESENTATION: str = "representation"
COL_N_CLUSTERS: str = "n_clusters"

# --- Neighborhood preservation ---
COL_K: str = "k"
COL_PRESERVATION_SCORE: str = "preservation_score"
COL_OPTIMAL_K: str = "optimal_k"

# --- Clustering quality ---
COL_ARI: str = "ARI"
COL_NMI: str = "NMI"

# --- Cell-type classification ---
COL_CELLTYPE: str = "celltype"
COL_F1_SCORE: str = "f1-score"
COL_ACCURACY: str = "accuracy"
COL_FEATURE_IMPORTANCE: str = "importance"
COL_GENE: str = "gene"

# --- NMF variability ---
COL_MSE_TEST_PROBE: str = "mse_test_probe"
COL_EXPVAR_TEST_PROBE: str = "expvar_test_probe"
COL_RMSE_TEST_PROBE: str = "rmse_test_probe"
COL_WEIGHTED_MSE: str = "weighted_mse_test_probe"
COL_WEIGHTED_EXPVAR: str = "weighted_expvar_test_probe"
COL_MACRO_MSE: str = "macro_mse_test_probe"
COL_MACRO_EXPVAR: str = "macro_expvar_test_probe"
COL_N_CELLS: str = "n_cells"
COL_SKIPPED: str = "skipped"

# =============================================================================
# SELECTION RESULT CSV COLUMN NAMES
# =============================================================================

COL_SELECTION_SCORE: str = "selection_score"
COL_COMBINED_SCORE: str = "combined_score"
COL_GENE_SOURCE: str = "gene_source"
COL_INITIAL_SOURCE: str = "initial_source"

# =============================================================================
# METRIC AND DIMENSION REDUCTION COLUMNS
# =============================================================================

COL_METRIC: str = "metric"
COL_DIM_REDUCTION: str = "dim_reduction"
COL_SCORE: str = "score"
COL_CLUSTERING_QUALITY_ARI: str = "clustering_quality_ari"
COL_CLUSTERING_QUALITY_NMI: str = "clustering_quality_nmi"
COL_DISPLAY_NAME: str = "display_name"

# =============================================================================
# PLOT SETTINGS
# =============================================================================

# DPI Settings (Publication Quality)
DEFAULT_PNG_DPI: int = 600  # Increased from 300 for publication quality
ANALYSIS_PNG_DPI: int = 600  # For analysis plots (previously 200 in various scripts)

DEFAULT_FIGURE_FORMAT: str = "png"
DEFAULT_COLORMAP: str = "viridis"

# =============================================================================
# PUBLICATION-QUALITY FONT SIZES
# =============================================================================

# Font sizes optimized for publication in high-impact journals
PUB_TITLE_SIZE: int = 20        # Main plot titles
PUB_SUPTITLE_SIZE: int = 22     # Figure super-titles
PUB_LABEL_SIZE: int = 18        # Axis labels (x and y)
PUB_TICK_SIZE: int = 16         # Tick labels
PUB_LEGEND_SIZE: int = 16       # Legend text
PUB_LEGEND_TITLE_SIZE: int = 18 # Legend titles
PUB_VALUE_SIZE: int = 14        # Value labels on bars/points
PUB_COLORBAR_SIZE: int = 16     # Colorbar labels

# =============================================================================
# PUBLICATION-QUALITY LINE/MARKER SIZES
# =============================================================================

PUB_LINE_WIDTH: int = 3         # Plot line width (increased from 2)
PUB_MARKER_SIZE: int = 8        # Marker size (increased from 6)
PUB_BAR_EDGE_WIDTH: float = 2.5 # Bar edge width (increased from 2)

# =============================================================================
# ANALYSIS-SPECIFIC COLORS
# =============================================================================

# Stability analysis colors
STABILITY_RF_COLOR: str = "#FF9800"    # Orange (RF pool)
STABILITY_NMF_COLOR: str = "#4CAF50"   # Green (NMF pool)
STABILITY_METRIC_COLOR: str = "#2196F3" # Blue (for metric bars)

# K-varying analysis colors
K_VARYING_NMF_COLOR: str = "#1f77b4"   # Blue (NMF method)
K_VARYING_CNMF_COLOR: str = "#ff7f0e"  # Orange (cNMF method)

# General analysis colors
ANALYSIS_NMF_COLOR: str = "#2196F3"     # Blue (NMF reconstruction)
ANALYSIS_TANGRAM_COLOR: str = "#FF9800" # Orange (Tangram reconstruction)

# =============================================================================
# FILE NAMING
# =============================================================================

RESULTS_DIRNAME: str = "results"
FIGURES_DIRNAME: str = "figures"
