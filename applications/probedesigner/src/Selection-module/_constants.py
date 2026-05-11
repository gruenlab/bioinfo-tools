"""
Constants for gene selection pipeline.

This module defines configuration constants organized into TWO categories:

1. USER-CONFIGURABLE PARAMETERS - Exposed via shell script, have fallback defaults
2. INTERNAL ALGORITHM PARAMETERS - Not exposed, control internal behavior

The shell script (Run-Pipeline-Combined.sh) forwards user-specified values 
to override the defaults defined here.
"""

from __future__ import annotations

# =============================================================================
# USER-CONFIGURABLE PARAMETERS
# =============================================================================
# These parameters are exposed to users via the shell script and command-line.
# Defaults defined here are used as fallbacks if not specified by the user.

# -------------------- Data & Strategy Selection --------------------

# Strategy lists (can be overridden by STRATEGIES in shell script)
DEFAULT_SINGLE_STRATEGIES = ['deg_only', 'rf_simple', 'rf_deg', 'hvg', 'random', 'dimred_only']
DEFAULT_COMBINATION_STRATEGIES = ['rf_nmf', 'rf_pca']
DEFAULT_ALL_STRATEGIES = DEFAULT_SINGLE_STRATEGIES + DEFAULT_COMBINATION_STRATEGIES

# Cell type filtering
DEFAULT_MIN_CELLS_PER_CELLTYPE = 10

# Target panel size
DEFAULT_PROBESET_SIZE = 100

# Random state for reproducibility
DEFAULT_RANDOM_STATE = 42

# -------------------- Dimensionality Reduction --------------------

# Number of components
DEFAULT_N_COMPONENTS_NMF = 5
DEFAULT_N_COMPONENTS_PCA = 50

# Analysis configuration
DEFAULT_REDUCTION_TYPE = "nmf"  # Options: 'nmf', 'pca'
DEFAULT_ANALYSIS_TYPE = "per_celltype"  # Options: 'global', 'per_celltype'
DEFAULT_DIMRED_METHOD = "method_a"  # Only option: 'method_a' (absolute factor weights)

# Legacy aliases (for backward compatibility)
DEFAULT_REDUCTION_TYPES = ("nmf",)
DEFAULT_ANALYSIS_TYPES = ("per_celltype",)
DEFAULT_PER_FACTOR_SELECTION = ("method_a",)

# -------------------- Combination Strategy Ratios --------------------

DEFAULT_RF_PERCENTAGE = 0.25  # 25% from Random Forest
DEFAULT_DIMRED_PERCENTAGE = 0.75  # 75% from dimensionality reduction

# Gap-filling strategies (enabled/disabled via shell script)
DEFAULT_RUN_CELLTYPE_FILLING = True
DEFAULT_RUN_GLOBAL_FILLING = True
DEFAULT_RUN_DEG_FILLING = True

# -------------------- Filtering Parameters --------------------

# Blacklist patterns
DEFAULT_BLACKLIST_PATTERNS = ['mt-', 'hsp']
DEFAULT_DISABLE_DEFAULT_BLACKLIST = False

# Xenium expression filter
DEFAULT_XENIUM_ENABLED = True  # Enabled by default
DEFAULT_MIN_XENIUM_EXPRESSION = 0.1
DEFAULT_MAX_XENIUM_EXPRESSION = 100.0

# ODT probe design filter
DEFAULT_ODT_ENABLED = False  # Disabled by default (computationally expensive)
DEFAULT_ODT_METHOD = "SCRINSHOT"  # Options: 'SCRINSHOT', 'MERFISH', 'SEQFISHPLUS'
DEFAULT_ODT_MIN_PROBES_THRESHOLD = 8
DEFAULT_ODT_SPECIES = "mus_musculus"  

# Aliases for backward compatibility
MIN_ODT_PROBES_THRESHOLD = DEFAULT_ODT_MIN_PROBES_THRESHOLD

# =============================================================================
# INTERNAL ALGORITHM PARAMETERS
# =============================================================================
# These control internal algorithm behavior and are NOT exposed to users.
# Do not modify these unless you understand the algorithm internals.

# -------------------- Internal Constraints --------------------

# Minimum genes per cell type in selection (hard constraint)
MIN_GENES_PER_CELLTYPE = 2

# Minimum cells for mean expression calculation
DEFAULT_MIN_CELLS_FOR_MEAN_EXPR = 3

# -------------------- External API Parameters --------------------

# MyGene.info batch size
MYGENE_BATCH_SIZE = 1000

# MyGene.info species mapping
MYGENE_SPECIES = {
    'mouse': 'mm',
    'human': 'hs',
    'mus_musculus': 'mm',
    'homo_sapiens': 'hs',
}

# =============================================================================
# FILE NAMING CONVENTIONS
# =============================================================================

# -------------------- Dimensionality Reduction (NMF/PCA) --------------------

# NMF/PCA default components (aliases for backward compatibility)
DEFAULT_NMF_COMPONENTS = DEFAULT_N_COMPONENTS_NMF
DEFAULT_PCA_COMPONENTS = DEFAULT_N_COMPONENTS_PCA

# Genes per component in dimred
DEFAULT_DIMRED_GENES_PER_COMPONENT = 20

# Pool-based architecture: Pool size per celltype/factor
DEFAULT_POOL_SIZE_PER_CELLTYPE = 200  # Genes per celltype in Phase 1 pool
DEFAULT_POOL_SIZE_PER_FACTOR = 200     # Genes per factor in Phase 1 pool (global mode)

# NMF iteration parameters
DEFAULT_NMF_MAX_ITER = 1000
DEFAULT_NMF_BETA_LOSS = "kullback-leibler"
DEFAULT_NMF_SOLVER = "mu"
DEFAULT_NMF_INIT = "nndsvda"

# -------------------- Consensus NMF --------------------

# K values to test for consensus NMF
DEFAULT_CONSENSUS_K_VALUES = [5, 10, 15, 20]

# Number of iterations for consensus NMF
DEFAULT_CONSENSUS_ITERATIONS = 100

# Density threshold percentile for consensus NMF
DENSITY_THRESHOLD_PERCENTILE = 2.0

# -------------------- Factor-Aware Selection --------------------

# Minimum genes per factor (warn if any factor empty)
MIN_FACTOR_CONTRIBUTION = 1

# Tolerance for factor imbalance (10% deviation allowed)
FACTOR_BALANCE_TOLERANCE = 0.1

# -------------------- Combination Strategies (rf_nmf, rf_pca) --------------------

# Oversampling factor for component strategies
COMBINATION_OVERSAMPLE_FACTOR = 1.5

# Gap-filling strategy names (internal constants)
GAP_FILL_STRATEGY_CELLTYPE = 'celltype-specific-filling'
GAP_FILL_STRATEGY_GLOBAL = 'global-gene-filling'
GAP_FILL_STRATEGY_DEG = 'deg-based-filling'

# -------------------- Gene Provenance Tracking --------------------

# Gene source labels for metadata tracking
GENE_SOURCE_FORCE_INCLUDE = 'force_include'
GENE_SOURCE_RF = 'rf_deg'
GENE_SOURCE_DIMRED = 'dimred'
GENE_SOURCE_OVERLAP_TO_RF = 'overlap→rf_deg'
GENE_SOURCE_DIMRED_REPLACEMENT = 'dimred_replacement'
GENE_SOURCE_GAP_FILL_CELLTYPE = 'gap_fill_celltype'
GENE_SOURCE_GAP_FILL_GLOBAL = 'gap_fill_global'
GENE_SOURCE_GAP_FILL_DEG = 'gap_fill_deg'

# -------------------- Filtering --------------------

# Maximum iterations for ODT filtering with replacement
MAX_ODT_FILTER_ITERATIONS = 1000

# -------------------- Random Forest (RF) --------------------

# Cross-validation folds
DEFAULT_RF_N_FOLDS = 5

# Random seeds for stability across multiple CV runs
DEFAULT_RF_RANDOM_SEEDS = [42, 43, 44, 45, 46]

# Number of trees
DEFAULT_RF_N_ESTIMATORS = 100

# Tree parameters
DEFAULT_RF_MAX_DEPTH = 3
DEFAULT_RF_MIN_SAMPLES_LEAF = 10

# Feature importance threshold
DEFAULT_RF_FEATURE_IMPORTANCE_THRESHOLD = 0.0

# -------------------- DEG Selection --------------------

# Statistical test method
DEFAULT_DEG_METHOD = 'wilcoxon'

# Filtering thresholds
DEFAULT_DEG_MIN_FOLD_CHANGE = 1.5
DEFAULT_DEG_MAX_PVAL = 0.05

# Standard output filenames
GENE_PANEL_FILENAME = 'selected_genes.csv'
RANKED_GENE_LIST_INITIAL = 'ranked_gene_list_initial.csv'
RANKED_GENE_LIST_FINAL = 'ranked_gene_list_final.csv'
GENE_PROVENANCE_REPORT = 'gene_provenance_report.csv'
FILTERING_SUMMARY_JSON = 'filtering_summary.json'

# Combination strategy filenames
RF_COMPONENT_RANKED_LIST = 'rf_component_ranked_gene_list.csv'
DIMRED_COMPONENT_RANKED_LIST = '{reduction}_component_ranked_gene_list.csv'
COMBINED_RANKED_LIST = 'combined_selected_genes_ranked_list.csv'
DUPLICATE_RESOLUTION_REPORT = 'duplicate_resolution_report.csv'
COMBINATION_SUMMARY_JSON = 'combination_summary.json'

# Directory names
RESULTS_DIRNAME = 'results'
DATA_DIRNAME = 'data'
PLOTS_DIRNAME = 'plots'
MODELS_DIRNAME = 'models'

# =============================================================================
# Column Names for Unified Gene List Output
# =============================================================================

# Essential columns (all strategies)
COL_GENE = 'gene'
COL_RANK = 'rank'
COL_SELECTION_SCORE = 'selection_score'
COL_SELECTION_STRATEGY = 'selection_strategy'
COL_ANALYSIS_TYPE = 'analysis_type'
COL_CELLTYPE = 'celltype'
COL_MEAN_EXPRESSION = 'mean_expression'
COL_SELECTED_INITIAL = 'selected_initial'
COL_PASSED_XENIUM = 'passed_xenium'
COL_PASSED_ODT = 'passed_odt'
COL_FINAL_SELECTION = 'final_selection'
COL_XENIUM_FAILURE_REASON = 'xenium_failure_reason'
COL_ODT_FAILURE_REASON = 'odt_failure_reason'
COL_REPLACED_BY = 'replaced_by'
COL_REPLACES_GENE = 'replaces_gene'
COL_REPLACEMENT_REASON = 'replacement_reason'

# DEG-specific columns
COL_PVALUE = 'pvalue'
COL_PADJ = 'padj'
COL_LOG2FC = 'log2fc'
COL_PCT_IN_GROUP = 'pct_in_group'
COL_PCT_OUT_GROUP = 'pct_out_group'

# Random forest columns
COL_FEATURE_IMPORTANCE = 'feature_importance'
COL_SELECTION_FREQUENCY = 'selection_frequency'

# Dimensionality reduction columns
COL_COMPONENT = 'component'
COL_COMPONENT_LOADING = 'component_loading'
COL_MAX_LOADING = 'max_loading'
COL_SUM_LOADING = 'sum_loading'
COL_N_COMPONENTS_REPRESENTED = 'n_components_represented'

# Combination strategy columns
COL_GENE_SOURCE = 'gene_source'
COL_INITIAL_DUPLICATE = 'initial_duplicate'
COL_DUPLICATE_RESOLUTION_ACTION = 'duplicate_resolution_action'

# =============================================================================
# Logging Configuration
# =============================================================================

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Log levels
LOG_LEVEL_DEFAULT = 'INFO'
LOG_LEVEL_DEBUG = 'DEBUG'
LOG_LEVEL_WARNING = 'WARNING'
LOG_LEVEL_ERROR = 'ERROR'
