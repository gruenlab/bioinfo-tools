"""Constants for the evaluation pipeline.

Organized into two categories:
1. USER-CONFIGURABLE PARAMETERS — exposed via shell script, have fallback defaults
2. INTERNAL ALGORITHM PARAMETERS — control internal behavior, not exposed to users
"""

from __future__ import annotations

# =============================================================================
# USER-CONFIGURABLE PARAMETERS
# =============================================================================

# -------------------- Data & Analysis --------------------

DEFAULT_CELLTYPE_COLUMN: str = "new_annot"
DEFAULT_EVALUATION_TYPE: str = "both"          # Options: "baseline", "variability", "both"
DEFAULT_DIMENSIONALITY_REDUCTION: str = "pca"  # Options: "pca", "nmf", "both"
DEFAULT_DIM_REDUCTION_PREPROCESS: str = "both"

# -------------------- Numerical Parameters --------------------

DEFAULT_N_COMPONENTS: int = 5       # NMF components
DEFAULT_N_NEIGHBORS: int = 15       # kNN graph construction
DEFAULT_TEST_SIZE: float = 0.3      # Train/test split ratio
DEFAULT_RANDOM_STATE: int = 42      # Reproducibility seed
DEFAULT_TANGRAM_N_EPOCHS: int = 500 # Tangram mapping epochs

# =============================================================================
# INTERNAL ALGORITHM PARAMETERS
# =============================================================================

# -------------------- Clustering / kNN --------------------

# k values evaluated for neighborhood preservation
DEFAULT_KNN_K_VALUES: list[int] = [5, 10, 15, 20, 30, 50]

# -------------------- Cell-type Classification --------------------

# Number of cells per cell type used for training the classifier
DEFAULT_SUBSAMPLE_SIZE: int = 500

# Train/test split ratio expressed as inverse (4 → 80/20 split)
DEFAULT_SPLIT_RATIO: int = 4

# -------------------- Tangram Reconstruction --------------------

# Minimum cells per cell type to run per-cell-type Tangram mapping
MIN_CELLS_PER_CELLTYPE: int = 30

# =============================================================================
# FILE AND DIRECTORY NAMING
# =============================================================================

RESULTS_DIRNAME: str = "results"
LOGS_DIRNAME: str = "logs"
PREPROCESSED_DIRNAME: str = "preprocessed"

# Standard output filenames
CLUSTERING_RESULTS_FILENAME: str = "clustering_evaluation_results.csv"
VARIABILITY_RESULTS_FILENAME: str = "variability_evaluation_results.csv"
TANGRAM_RESULTS_FILENAME: str = "tangram_reconstruction_results.csv"

# Plot settings
DEFAULT_PNG_DPI: int = 300

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
