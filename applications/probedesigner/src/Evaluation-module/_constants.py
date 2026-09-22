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

DEFAULT_CELLTYPE_COLUMN: str = "cluster"       # wired to run_evaluation.py --celltype_col
DEFAULT_EVALUATION_TYPE: str = "all"           # Options: "baseline", "variability", "biology", "all", "tangram_only"
DEFAULT_DIM_REDUCTION_PREPROCESS: str = "both"  # which embeddings preprocessing computes: "pca"/"nmf"/"both"

# -------------------- Numerical Parameters --------------------

DEFAULT_N_COMPONENTS: int = 5        # NMF components
DEFAULT_N_NEIGHBORS: int = 15        # kNN graph construction
DEFAULT_RANDOM_STATE: int = 42       # Reproducibility seed
DEFAULT_TANGRAM_N_EPOCHS: int = 1000 # Tangram mapping epochs (wired to --tangram_n_epochs)

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

# Depth cap for the DecisionTreeClassifier in evaluate_celltype_identification, applied
# identically to every panel (not tuned per panel). Set to 15 after the eval-tree depth
# sweep (docs/doc-pipeline/eval-celltype-clf-maxdepth-sweep.md): at depth 10 a cluster of
# lineage-adjacent cell types (T/NK/ILC1s/cDC1s/cDC2s/Mig.cDCs/Monocytes) collapsed to
# F1 ~ 0 on genuinely good panels because the tree ran out of splits before isolating
# their leaves; depth 15 recovers them (panel macro_f1 0.61 -> 0.94) while depth 20 begins
# to overfit small panels. The metric row still records the actual depth as
# `classifier_max_depth`.
DEFAULT_CELLTYPE_CLF_MAX_DEPTH: int = 15

# -------------------- Tangram Reconstruction --------------------

# Minimum cells per cell type to run per-cell-type Tangram mapping
MIN_CELLS_PER_CELLTYPE: int = 30

# -------------------- NMF reconstruction --------------------

# Fixed kwargs forwarded to nico2_lib NmfPredictor (selection + evaluation).
# KEEP IN SYNC WITH Selection-module/_constants.py. This is the Frobenius/cd fallback
# default only -- the actual solver/beta_loss/init/max_iter used by every NMF call is now
# resolved per-run by Utility-module/_nmf_objective.py (data-space-driven: raw counts ->
# mu/Kullback-Leibler, lognorm -> this cd/Frobenius set), overridable via --nmf_objective /
# nmf_objective=. NmfPredictor.fit()/.predict() honour solver + beta_loss + init + max_iter;
# alpha_*/l1_ratio remain no-ops, so those entries document intent only.
NMF_SOLVER: str = "cd"
NMF_BETA_LOSS: str = "frobenius"
NMF_INIT: str = "nndsvd"
NMF_MAX_ITER: int = 1000
NMF_ALPHA_W: float = 0.0
NMF_ALPHA_H: float = 0.0
NMF_L1_RATIO: float = 0.0
NMF_PREDICTOR_FIXED_KWARGS: dict = dict(
    solver=NMF_SOLVER,
    beta_loss=NMF_BETA_LOSS,
    init=NMF_INIT,
    max_iter=NMF_MAX_ITER,
    alpha_W=NMF_ALPHA_W,
    alpha_H=NMF_ALPHA_H,
    l1_ratio=NMF_L1_RATIO,
)
