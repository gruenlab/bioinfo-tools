"""
Constants for gene selection pipeline.

This module defines configuration constants organized into TWO categories:

1. USER-CONFIGURABLE PARAMETERS - have a corresponding CLI flag on
   run_selection_pipeline.py; the value here is only the fallback default.
2. INTERNAL ALGORITHM PARAMETERS - no CLI flag, changeable only from Python.

Note: not every constant under category 1 has a flag. The RF hyper-parameters
(DEFAULT_RF_N_ESTIMATORS, DEFAULT_RF_MAX_DEPTH, DEFAULT_RF_N_FOLDS,
DEFAULT_RF_RANDOM_SEEDS) live under category 2 and are Python-API-only; only
DEFAULT_RF_PERCENTAGE / DEFAULT_DIMRED_PERCENTAGE are CLI-exposed.
"""

from __future__ import annotations

# =============================================================================
# USER-CONFIGURABLE PARAMETERS
# =============================================================================
# These parameters have a corresponding CLI flag on run_selection_pipeline.py.
# Defaults defined here are used as fallbacks if not specified by the user.

# -------------------- Data & Strategy Selection --------------------

# Strategy lists (each entry maps to a --strategy value on run_selection_pipeline.py)
DEFAULT_SINGLE_STRATEGIES = ['deg_only', 'rf_simple', 'rf_deg', 'hvg', 'random', 'dimred_only']
DEFAULT_COMBINATION_STRATEGIES = ['RecoVar', 'RecoVar_PCA']
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
# NMF/PCA is always fitted independently within each cell type — there is no global
# (all-cells-pooled) analysis mode, and no user-selectable gene-ranking method
# (top genes by absolute factor/PC loading is the only ranking).

# -------------------- Combination Strategy Ratios --------------------

DEFAULT_RF_PERCENTAGE = 0.25  # 25% from Random Forest
DEFAULT_DIMRED_PERCENTAGE = 0.75  # 75% from dimensionality reduction

# Gap-filling strategies (enabled/disabled via --disable_*_filling CLI flags)
DEFAULT_RUN_CELLTYPE_FILLING = True
DEFAULT_RUN_DEG_FILLING = True

# -------------------- Filtering Parameters --------------------

# Blacklist patterns
DEFAULT_BLACKLIST_PATTERNS = ['mt-', 'hsp', 'rps', 'rpl']

# Xenium expression filter
DEFAULT_MIN_XENIUM_EXPRESSION = 0.1
DEFAULT_MAX_XENIUM_EXPRESSION = 100.0

# =============================================================================
# INTERNAL ALGORITHM PARAMETERS
# =============================================================================
# These control internal algorithm behavior and are NOT exposed to users.
# Do not modify these unless you understand the algorithm internals.

# -------------------- Dimensionality Reduction (NMF/PCA) --------------------

# NMF/PCA default components (aliases for backward compatibility)
DEFAULT_NMF_COMPONENTS = DEFAULT_N_COMPONENTS_NMF

# --- NMF: fixed kwargs forwarded to nico2_lib NmfPredictor (selection + evaluation) ---
# KEEP IN SYNC WITH Evaluation-module/_constants.py. This is the Frobenius/cd fallback
# default only -- the actual solver/beta_loss/init/max_iter used by every NMF call is now
# resolved per-run by Utility-module/_nmf_objective.py (data-space-driven: raw counts ->
# mu/Kullback-Leibler, lognorm -> this cd/Frobenius set), overridable via --nmf_objective /
# nmf_objective=. NmfPredictor.fit()/.predict() honour solver + beta_loss + init + max_iter;
# alpha_*/l1_ratio remain no-ops, so those entries document intent only.
NMF_SOLVER = "cd"          # sklearn's Frobenius default
NMF_BETA_LOSS = "frobenius"
NMF_INIT = "nndsvd"
NMF_MAX_ITER = 1000
NMF_ALPHA_W = 0.0
NMF_ALPHA_H = 0.0
NMF_L1_RATIO = 0.0
NMF_PREDICTOR_FIXED_KWARGS = dict(
    solver=NMF_SOLVER,
    beta_loss=NMF_BETA_LOSS,
    init=NMF_INIT,
    max_iter=NMF_MAX_ITER,
    alpha_W=NMF_ALPHA_W,
    alpha_H=NMF_ALPHA_H,
    l1_ratio=NMF_L1_RATIO,
)

# -------------------- Factor-Aware Selection --------------------

# Minimum genes per factor (warn if any factor empty)
MIN_FACTOR_CONTRIBUTION = 1

# Tolerance for factor imbalance (10% deviation allowed)
FACTOR_BALANCE_TOLERANCE = 0.1

# -------------------- Combination Strategies (RecoVar, RecoVar_PCA) --------------------

# Oversampling factor for component strategies
COMBINATION_OVERSAMPLE_FACTOR = 1.5

# Gap-filling strategy names (internal constants)
GAP_FILL_STRATEGY_CELLTYPE = 'celltype-specific-filling'
GAP_FILL_STRATEGY_DEG = 'deg-based-filling'

# Display labels for gap_fill_strategy in output CSVs — reflect the actual gene source each
# internal strategy pulls from (not the internal naming, which is misleading for DEG-based-filling:
# it reuses the rf_deg ranked list, not a fresh DEG test).
GAP_FILL_STRATEGY_DISPLAY_NAMES = {
    GAP_FILL_STRATEGY_CELLTYPE: 'dimred',
    GAP_FILL_STRATEGY_DEG: 'rf_deg',
}

# -------------------- Gene Provenance Tracking --------------------

# Gene source labels for metadata tracking
GENE_SOURCE_FORCE_INCLUDE = 'force_include'
GENE_SOURCE_RF = 'rf_deg'
GENE_SOURCE_DIMRED = 'dimred'
GENE_SOURCE_OVERLAP_TO_RF = 'overlap→rf_deg'
GENE_SOURCE_GAP_FILL_CELLTYPE = 'gap_fill_celltype'
GENE_SOURCE_GAP_FILL_DEG = 'gap_fill_deg'

# -------------------- Random Forest (RF) --------------------

# Cross-validation folds
DEFAULT_RF_N_FOLDS = 5

# Random seeds for stability across multiple CV runs
DEFAULT_RF_RANDOM_SEEDS = [42, 43, 44, 45, 46]

# Number of trees
DEFAULT_RF_N_ESTIMATORS = 100

# Tree parameters
DEFAULT_RF_MAX_DEPTH = 3

# -------------------- DEG Selection --------------------

# Statistical test method
DEFAULT_DEG_METHOD = 'wilcoxon'

# Filtering thresholds
DEFAULT_DEG_MAX_PVAL = 0.05

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
COL_FINAL_SELECTION = 'final_selection'
COL_XENIUM_FAILURE_REASON = 'xenium_failure_reason'

# Dimensionality reduction columns
COL_COMPONENT = 'component'

# Random-forest per-cell-type attribution columns.
# The RF path trains one global multiclass model, so its feature_importances_ is a
# single scalar per gene and every RF gene is written celltype='global'. These
# columns carry a per-class decomposition of that importance (see
# _rf_selection._forest_per_class_gini_importance) so downstream diagnostics can
# attribute RF genes to cell types "from the random forest itself".
COL_RF_CELLTYPE = 'rf_celltype'                              # argmax cell type ('' if never split)
COL_RF_CONTRIBUTING_CELLTYPES = 'rf_contributing_celltypes'  # RF_CONTRIBUTING_CELLTYPE_SEP-joined
COL_RF_CELLTYPE_SCORES = 'rf_celltype_scores'                # json {celltype: share}

# Unified per-gene "which cell types is this gene informative for" column on
# ranked_gene_list.csv. Merges rf_contributing_celltypes (RF genes),
# contributing_celltypes (NMF/dimred + celltype gap-fill) and celltype (DEG) into
# one RF_CONTRIBUTING_CELLTYPE_SEP-joined string, so
# df[COL_INFORMATIVE_CELLTYPES].str.split(sep).explode().value_counts() gives the
# genes-per-cell-type coverage without a separate file. Built by
# _gene_list_builder.derive_informative_celltypes().
COL_INFORMATIVE_CELLTYPES = 'informative_celltypes'

# A class counts as "contributing" when its share of a gene's own RF importance is
# at least this; the argmax class is used as a fallback so the field is never empty
# for an attributed gene.
RF_CONTRIBUTING_CELLTYPE_MIN_SHARE = 0.20
# Separator for COL_RF_CONTRIBUTING_CELLTYPES — '|' not ',', because cell-type
# names routinely contain commas (e.g. "CD16-negative, CD56-bright natural killer
# cell, human").
RF_CONTRIBUTING_CELLTYPE_SEP = '|'
