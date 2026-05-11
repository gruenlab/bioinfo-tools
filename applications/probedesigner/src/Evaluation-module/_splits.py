"""Train/test split generation for the evaluation pipeline.

Splits are generated once and forwarded identically to both NMF and Tangram
evaluation so that both methods operate on the same cell partitions.

Functions:
    generate_evaluation_splits: Generate global + per-celltype splits.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)

__all__ = ["generate_evaluation_splits"]


def _build_per_celltype_splits(
    adata,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    celltype_col: str,
    min_cells: int = 10,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Derive per-celltype train/test splits from a global fold split.

    Intersects global train/test indices with each cell type's positions.
    Cell types with fewer than ``min_cells`` in either partition are skipped.

    Args:
        adata: AnnData with cell-type annotations.
        train_idx: Integer position array of training cells.
        test_idx: Integer position array of test cells.
        celltype_col: obs column containing cell-type labels.
        min_cells: Minimum cells required in train AND test for a cell type to be included.

    Returns:
        Dict mapping each qualifying cell type to ``(train_ct_idx, test_ct_idx)``.
    """
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ct in adata.obs[celltype_col].unique():
        ct_pos = np.where(adata.obs[celltype_col] == ct)[0]
        train_ct = np.intersect1d(train_idx, ct_pos)
        test_ct = np.intersect1d(test_idx, ct_pos)
        if len(train_ct) < min_cells or len(test_ct) < min_cells:
            continue
        splits[ct] = (train_ct, test_ct)
    return splits


def generate_evaluation_splits(
    adata,
    celltype_col: str,
    split_mode: Literal["simple", "kfold"] = "kfold",
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = 42,
    min_cells_per_celltype: int = 10,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    """Generate train/test splits for NMF and Tangram evaluation.

    Returns a list of ``(train_idx, test_idx, per_celltype_splits)`` tuples that
    should be passed *identically* to both ``run_variability_evaluation`` and
    ``run_tangram_stage`` so that NMF and Tangram operate on the same partitions.

    Args:
        adata: AnnData whose cells are to be split.
        celltype_col: obs column containing cell-type labels.
        split_mode: ``"kfold"`` (default) for stratified k-fold; ``"simple"`` for
            a single random split.
        test_size: Fraction of cells held out for testing (used in ``"simple"`` mode).
        n_splits: Number of folds for stratified k-fold (used in ``"kfold"`` mode).
        random_state: Seed for reproducibility.
        min_cells_per_celltype: Minimum cells required in both train and test partitions
            for a cell type to appear in ``per_celltype_splits``.

    Returns:
        List of ``(train_idx, test_idx, per_celltype_splits)`` tuples.
        ``"simple"`` mode returns a one-element list.
        ``"kfold"`` mode returns ``n_splits`` elements.
    """
    if celltype_col not in adata.obs.columns:
        raise ValueError(f"Celltype column '{celltype_col}' not found in adata.obs")

    if split_mode == "simple":
        all_idx = np.arange(adata.n_obs)
        train_idx, test_idx = train_test_split(
            all_idx, test_size=test_size, random_state=random_state
        )
        per_ct = _build_per_celltype_splits(
            adata, train_idx, test_idx, celltype_col, min_cells=min_cells_per_celltype
        )
        return [(train_idx, test_idx, per_ct)]

    # kfold mode
    ct_labels = adata.obs[celltype_col].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = []
    for fold, (train_idx, test_idx) in enumerate(
        skf.split(np.arange(adata.n_obs), ct_labels)
    ):
        per_ct = _build_per_celltype_splits(
            adata, train_idx, test_idx, celltype_col, min_cells=min_cells_per_celltype
        )
        logger.debug("Fold %d: %d train, %d test cells", fold, len(train_idx), len(test_idx))
        result.append((train_idx, test_idx, per_ct))
    return result
