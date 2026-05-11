"""UMAP comparison plots for raw-vs-log NMF factor values.

This module provides reusable plotting helpers for visualizing NMF factor
activations on a shared UMAP embedding, comparing raw-count and log-normalized
variants side by side.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

try:
    from ._constants import ANALYSIS_PNG_DPI
except ImportError:
    # Support direct script execution where package-relative imports are unavailable.
    from _constants import ANALYSIS_PNG_DPI

logger = logging.getLogger(__name__)


def _transform_values(values: np.ndarray, transform: str, shift: float) -> np.ndarray:
    """Apply optional value transform for improved visual contrast."""
    if transform == "none":
        return values
    if transform == "log1p":
        return np.log1p(np.maximum(values + shift, 0.0))
    raise ValueError(f"Unsupported transform '{transform}'. Use 'none' or 'log1p'.")


def _robust_limits(values: np.ndarray, lower_percentile: float, upper_percentile: float) -> tuple[float, float]:
    """Compute robust plotting limits from finite values."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError(
            "Percentiles must satisfy 0 <= lower < upper <= 100. "
            f"Got lower={lower_percentile}, upper={upper_percentile}."
        )

    vmin = float(np.percentile(finite_values, lower_percentile))
    vmax = float(np.percentile(finite_values, upper_percentile))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    return vmin, vmax


def infer_celltype_column(adata, requested_column: Optional[str] = None) -> str:
    """Infer a reasonable cell-type annotation column from adata.obs."""
    if requested_column is not None:
        if requested_column not in adata.obs.columns:
            raise KeyError(
                f"Requested celltype column '{requested_column}' not found in adata.obs."
            )
        return requested_column

    candidates = ["annot", "cell_type", "celltype", "cluster", "celltypes_v2"]
    for col in candidates:
        if col in adata.obs.columns:
            return col

    available = ", ".join(map(str, adata.obs.columns.tolist()))
    raise KeyError(
        "Could not infer cell-type annotation column. "
        f"Tried {candidates}. Available columns: {available}"
    )


def ensure_umap(
    adata,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    random_state: int = 42,
) -> None:
    """Ensure ``adata.obsm['X_umap']`` exists, computing it when missing.

    Parameters
    ----------
    adata
        AnnData object.
    n_neighbors : int, default 15
        Number of neighbors used for neighborhood graph.
    n_pcs : int, default 30
        Number of principal components for neighbor graph.
    random_state : int, default 42
        Seed used by UMAP for deterministic embedding.
    """
    if "X_umap" in adata.obsm:
        return

    if "X_pca" not in adata.obsm:
        max_pca = max(2, min(int(n_pcs), int(adata.n_vars - 1)))
        logger.info("Computing PCA with n_comps=%d", max_pca)
        sc.pp.pca(adata, n_comps=max_pca)

    pca_comps = int(adata.obsm["X_pca"].shape[1])
    used_pcs = max(2, min(int(n_pcs), pca_comps))

    logger.info("Computing neighbors (n_neighbors=%d, n_pcs=%d)", n_neighbors, used_pcs)
    sc.pp.neighbors(adata, n_neighbors=int(n_neighbors), n_pcs=used_pcs, use_rep="X_pca")

    logger.info("Computing UMAP (random_state=%d)", random_state)
    sc.tl.umap(adata, random_state=int(random_state))


def _validate_factor_arrays(
    w_raw: np.ndarray,
    w_log: np.ndarray,
    probeset_size: int,
) -> int:
    """Validate factor matrices and return number of factors."""
    if w_raw.ndim != 2 or w_log.ndim != 2:
        raise ValueError(
            f"Expected 2D factor matrices for probeset {probeset_size}, "
            f"got shapes {w_raw.shape} and {w_log.shape}."
        )

    if w_raw.shape != w_log.shape:
        raise ValueError(
            f"Raw/log factor matrices must have matching shapes for probeset {probeset_size}, "
            f"got {w_raw.shape} vs {w_log.shape}."
        )

    n_factors = int(w_raw.shape[1])
    if n_factors < 1:
        raise ValueError(f"No factors found for probeset {probeset_size}.")

    return n_factors


def plot_raw_vs_log_factor_umap_grid(
    adata,
    probeset_size: int,
    output_dir: Path,
    raw_key: Optional[str] = None,
    log_key: Optional[str] = None,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    random_state: int = 42,
    point_size: float = 4.0,
    cmap: str = "magma",
    png_dpi: int = ANALYSIS_PNG_DPI,
    figure_format: str = "png",
    title_prefix: str = "NMF Factor UMAP Comparison",
    factor_names: Optional[Sequence[str]] = None,
    value_transform: str = "log1p",
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> Path:
    """Plot a 2xN UMAP grid for raw/log NMF factor values of one probeset.

    Rows correspond to data variants (raw, log) and columns to factor indices.
    For each factor column, raw and log panels share one value scale to make
    colors directly comparable between the two rows.

    Parameters
    ----------
    adata
        AnnData object containing factor matrices in ``obsm``.
    probeset_size : int
        Probeset size used in factor matrix key naming.
    output_dir : Path
        Directory where the figure is saved.
    raw_key : str, optional
        Custom ``obsm`` key for raw-count factors.
    log_key : str, optional
        Custom ``obsm`` key for log-normalized factors.
    n_neighbors : int, default 15
        UMAP neighbors parameter when embedding is computed.
    n_pcs : int, default 30
        Number of PCs for neighbor graph construction.
    random_state : int, default 42
        Seed for UMAP.
    point_size : float, default 4.0
        Point size in scatter plots.
    cmap : str, default "viridis"
        Matplotlib colormap used for factor values.
    png_dpi : int, default ANALYSIS_PNG_DPI
        Figure DPI.
    figure_format : str, default "png"
        Figure file format.
    title_prefix : str, default "NMF Factor UMAP Comparison"
        Prefix shown in figure suptitle.
    factor_names : sequence of str, optional
        Column labels for factors; length must match number of factors.
    value_transform : str, default "log1p"
        Value transform for improved dynamic-range readability.
        Use "log1p" or "none".
    lower_percentile : float, default 2.0
        Lower percentile for robust color clipping.
    upper_percentile : float, default 98.0
        Upper percentile for robust color clipping.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_raw_key = raw_key or f"W_nmf_factors_raw_{probeset_size}"
    resolved_log_key = log_key or f"W_nmf_factors_log_{probeset_size}"

    if resolved_raw_key not in adata.obsm:
        raise KeyError(
            f"Missing raw factor key '{resolved_raw_key}' in adata.obsm for probeset {probeset_size}."
        )
    if resolved_log_key not in adata.obsm:
        raise KeyError(
            f"Missing log factor key '{resolved_log_key}' in adata.obsm for probeset {probeset_size}."
        )

    w_raw = np.asarray(adata.obsm[resolved_raw_key])
    w_log = np.asarray(adata.obsm[resolved_log_key])
    n_factors = _validate_factor_arrays(w_raw=w_raw, w_log=w_log, probeset_size=probeset_size)

    if factor_names is not None and len(factor_names) != n_factors:
        raise ValueError(
            f"factor_names length ({len(factor_names)}) must match number of factors ({n_factors})."
        )

    ensure_umap(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state)
    umap = np.asarray(adata.obsm["X_umap"])

    if umap.ndim != 2 or umap.shape[1] < 2:
        raise ValueError(f"Expected X_umap with at least 2 columns, got shape {umap.shape}.")

    fig_width = max(14, n_factors * 4.2)
    fig, axes = plt.subplots(
        2,
        n_factors,
        figsize=(fig_width, 8),
        squeeze=False,
        constrained_layout=True,
    )

    for factor_idx in range(n_factors):
        raw_values = w_raw[:, factor_idx]
        log_values = w_log[:, factor_idx]
        both_values = np.concatenate([raw_values, log_values])
        finite_original = both_values[np.isfinite(both_values)]
        shift = 0.0
        if finite_original.size > 0:
            min_original = float(np.min(finite_original))
            if min_original < 0.0:
                shift = -min_original

        raw_plot_values = _transform_values(raw_values, transform=value_transform, shift=shift)
        log_plot_values = _transform_values(log_values, transform=value_transform, shift=shift)
        both_plot_values = np.concatenate([raw_plot_values, log_plot_values])
        vmin, vmax = _robust_limits(
            both_plot_values,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )

        raw_ax = axes[0, factor_idx]
        log_ax = axes[1, factor_idx]

        raw_mask = np.isfinite(raw_plot_values)
        raw_idx = np.where(raw_mask)[0]
        if raw_idx.size > 0:
            raw_idx = raw_idx[np.argsort(raw_plot_values[raw_idx])]

        log_mask = np.isfinite(log_plot_values)
        log_idx = np.where(log_mask)[0]
        if log_idx.size > 0:
            log_idx = log_idx[np.argsort(log_plot_values[log_idx])]

        raw_scatter = raw_ax.scatter(
            umap[raw_idx, 0],
            umap[raw_idx, 1],
            c=raw_plot_values[raw_idx],
            s=point_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            alpha=0.9,
        )
        log_ax.scatter(
            umap[log_idx, 0],
            umap[log_idx, 1],
            c=log_plot_values[log_idx],
            s=point_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            alpha=0.9,
        )

        if factor_names is None:
            factor_label = f"Factor {factor_idx + 1}"
        else:
            factor_label = str(factor_names[factor_idx])

        raw_ax.set_title(factor_label, fontsize=12, pad=10)
        for ax in (raw_ax, log_ax):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#f7f7f7")
            for spine in ax.spines.values():
                spine.set_visible(False)

        colorbar = fig.colorbar(
            raw_scatter,
            ax=[raw_ax, log_ax],
            location="right",
            fraction=0.035,
            pad=0.01,
        )
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label("Enrichment", fontsize=9)

    axes[0, 0].set_ylabel("Raw", fontsize=12)
    axes[1, 0].set_ylabel("Log", fontsize=12)

    fig.suptitle(
        f"{title_prefix} ({probeset_size} genes)",
        fontsize=16,
        y=1.02,
    )

    ext = figure_format.lower().lstrip(".")
    out_path = output_dir / f"nmf_factor_umap_grid_raw_vs_log_{probeset_size}.{ext}"
    fig.savefig(out_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved raw-vs-log factor UMAP grid: %s", out_path)
    return out_path


def plot_celltype_umap(
    adata,
    output_path: Path,
    celltype_col: Optional[str] = None,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    random_state: int = 42,
    png_dpi: int = ANALYSIS_PNG_DPI,
) -> Path:
    """Plot one UMAP colored by cell-type annotations."""
    ensure_umap(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state)
    resolved_celltype_col = infer_celltype_column(adata, requested_column=celltype_col)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    sc.pl.umap(
        adata,
        color=resolved_celltype_col,
        ax=ax,
        show=False,
        title=f"Cell type UMAP ({resolved_celltype_col})",
        legend_loc="right margin",
    )

    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved celltype UMAP: %s", output_path)
    return output_path
