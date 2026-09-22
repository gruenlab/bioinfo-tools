"""Evaluation pipeline for spatial probe design.

This package provides tools to evaluate the quality of selected probe panels
using several complementary approaches:

- **Baseline evaluation**: clustering quality (ARI, NMI), neighbourhood
  preservation (kNN overlap), and cell-type identification accuracy.
- **Variability evaluation**: NMF reconstruction metrics (absolute MSE / explained
  variance for the probe reconstruction and for a full-gene NMF baseline).
- **Reconstruction check** (optional): Tangram-based full-transcriptome
  reconstruction from probe-panel expression.

The module directory name is hyphenated (``Evaluation-module``), so this cannot be
imported as ``import evaluation``. Add the directory to ``sys.path`` and import the
flat module names, e.g.::

    import sys
    sys.path.insert(0, ".../Code/RecoVar/Evaluation-module")
    from _clustering import evaluate_clustering_quality
    from nmf import nmf_reconstruction
    from metrics import calculate_mse

This ``__init__.py`` re-exports the same names for convenience once the directory
is on ``sys.path``; the sibling modules use flat imports internally, so the line
below guarantees that.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from _clustering import (  # noqa: E402
    compute_clustering_similarity,
    compute_neighborhood_preservation,
    evaluate_celltype_identification,
    evaluate_clustering_quality,
    evaluate_neighborhood_preservation,
)
from metrics import (  # noqa: E402
    calculate_explained_variance,
    calculate_macro_explained_variance,
    calculate_macro_mse,
    calculate_mse,
    calculate_weighted_explained_variance,
    calculate_weighted_mse,
)
from nmf import (  # noqa: E402
    nmf_reconstruction,
    nmf_reconstruction_by_celltype,
)

try:
    from pca import pca_reconstruction, pca_reconstruction_by_celltype  # noqa: E402
    from ridge import ridge_reconstruction, ridge_reconstruction_by_celltype  # noqa: E402
    from ica import ica_reconstruction, ica_reconstruction_by_celltype  # noqa: E402
    _reconstruction_available = True
except ImportError:  # pragma: no cover - optional deps
    _reconstruction_available = False

__all__ = [
    # Baseline metrics
    "compute_clustering_similarity",
    "compute_neighborhood_preservation",
    "evaluate_celltype_identification",
    "evaluate_clustering_quality",
    "evaluate_neighborhood_preservation",
    # Variability metrics
    "calculate_explained_variance",
    "calculate_macro_explained_variance",
    "calculate_macro_mse",
    "calculate_mse",
    "calculate_weighted_explained_variance",
    "calculate_weighted_mse",
    "nmf_reconstruction",
    "nmf_reconstruction_by_celltype",
    # PCA reconstruction
    "pca_reconstruction",
    "pca_reconstruction_by_celltype",
    # Ridge regression reconstruction
    "ridge_reconstruction",
    "ridge_reconstruction_by_celltype",
    # ICA reconstruction
    "ica_reconstruction",
    "ica_reconstruction_by_celltype",
]

__version__ = "2.0.0"
