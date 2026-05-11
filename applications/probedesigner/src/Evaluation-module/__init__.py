"""Evaluation pipeline for spatial probe design.

This package provides tools to evaluate the quality of selected probe panels
using three complementary approaches:

- **Baseline evaluation**: clustering quality (ARI, NMI), neighbourhood
  preservation (kNN overlap), and cell-type identification accuracy.
- **Variability evaluation**: NMF representation metrics (MSE, explained variance).
- **Reconstruction check** (optional): Tangram-based full-transcriptome
  reconstruction from probe-panel expression.

Typical workflow::

    from evaluation import run_evaluation
    run_evaluation.main()

Or import individual metric functions directly::

    from evaluation._clustering import evaluate_clustering_quality
    from evaluation._variability import nmf_reconstruction
"""

from __future__ import annotations

from ._clustering import (
    compute_clustering_similarity,
    compute_neighborhood_preservation,
    evaluate_celltype_identification,
    evaluate_clustering_quality,
    evaluate_neighborhood_preservation,
)
from ._variability import (
    calculate_explained_variance,
    calculate_macro_explained_variance,
    calculate_macro_mse,
    calculate_mse,
    calculate_weighted_explained_variance,
    calculate_weighted_mse,
    nmf_reconstruction,
    nmf_reconstruction_by_celltype,
)

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
]

__version__ = "2.0.0"
