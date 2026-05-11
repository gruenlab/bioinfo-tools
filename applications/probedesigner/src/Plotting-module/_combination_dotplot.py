"""Custom dotplot for combination-strategy probe panels.

Reimplements the scanpy DotPlot calculations exactly:
  - **Dot colour**  = mean log-normalised expression per group, mapped through
    a single ``"Reds"`` colourmap (global min/max across all genes/groups).
  - **Dot size**    = fraction of cells expressing (> cutoff), scaled with the
    same power-law as scanpy:
        size = frac ** size_exponent * (largest_dot - smallest_dot) + smallest_dot
  - **Gene-name labels** on the x-axis are coloured by the gene's selection
    source (e.g. red for DEG, blue for NMF/PCA).
  - All other stylistic choices (colorbar, size legend, source legend) follow
    the scanpy dot-plot conventions.

Main entry point
----------------
``plot_combination_dotplot(adata, var_names, groupby, gene_sources, ...)``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse

logger = logging.getLogger(__name__)

__all__ = ["plot_combination_dotplot"]

# ---------------------------------------------------------------------------
# Default colour palette for gene-source labels
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_COLORS: Dict[str, str] = {
    "rf_deg":              "#e31a1c",   # Red   — random-forest DEGs
    "dimred":              "#1f78b4",   # Blue  — dimensionality-reduction genes
    "overlap→rf_deg":      "#ff7f00",   # Orange — overlapping genes kept as RF
    "dimred_replacement":  "#6a3d9a",   # Purple
    "force_include":       "#33a02c",   # Green
    "gap_fill_celltype":   "#b15928",   # Brown
    "gap_fill_global":     "#a6cee3",   # Light blue
    "gap_fill_deg":        "#fb9a99",   # Light red
    "other":               "#888888",   # Grey
}

DEFAULT_SOURCE_LABEL_MAP: Dict[str, str] = {
    "rf_deg":              "DEG",
    "dimred":              "Dim-reduction",
    "overlap→rf_deg":      "Overlap → RF",
    "dimred_replacement":  "Dimred replacement",
    "force_include":       "Force-included",
    "gap_fill_celltype":   "Gap fill (cell-type)",
    "gap_fill_global":     "Gap fill (global)",
    "gap_fill_deg":        "Gap fill (DEG)",
    "other":               "Other",
}


# ---------------------------------------------------------------------------
# Scanpy-identical data preparation
# ---------------------------------------------------------------------------

def _prepare_dot_data(
    adata,
    var_names: List[str],
    groupby: str,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    standard_scale: Optional[str] = "var",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dot_color_df, dot_size_df) exactly as scanpy DotPlot does.

    Parameters
    ----------
    adata:
        AnnData with log-normalised counts in ``.X``.
    var_names:
        Genes to include (order preserved).
    groupby:
        Column in ``adata.obs`` defining the groups.
    expression_cutoff:
        A cell is considered *expressing* if ``X > expression_cutoff``
        (scanpy default: 0.0).
    mean_only_expressed:
        If True, mean is taken only over expressing cells (scanpy default:
        False).
    standard_scale:
        ``"var"`` normalises each gene to [0, 1] across groups;
        ``"group"`` normalises each group to [0, 1];
        ``None`` leaves values as-is.
    """
    valid_genes = [g for g in var_names if g in adata.var_names]
    if not valid_genes:
        raise ValueError("None of var_names found in adata.var_names.")
    if len(valid_genes) < len(var_names):
        missing = set(var_names) - set(valid_genes)
        logger.warning("Genes missing from adata (skipped): %s", missing)

    adata_sub = adata[:, valid_genes]
    X = adata_sub.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.array(X, copy=True)

    groups = adata_sub.obs[groupby].astype(str)
    obs_tidy = pd.DataFrame(X, index=groups, columns=valid_genes)
    obs_tidy.index.name = groupby

    # Fraction expressing
    obs_bool = obs_tidy > expression_cutoff
    dot_size_df = (
        obs_bool.groupby(level=0, observed=True).sum()
        / obs_bool.groupby(level=0, observed=True).count()
    )

    # Mean expression
    if mean_only_expressed:
        obs_expressed = obs_tidy.where(obs_bool)
        dot_color_df = obs_expressed.groupby(level=0, observed=True).mean().fillna(0.0)
    else:
        dot_color_df = obs_tidy.groupby(level=0, observed=True).mean()

    # Optional standard scale
    if standard_scale == "var":
        col_min = dot_color_df.min(axis=0)
        col_max = dot_color_df.max(axis=0)
        denom = (col_max - col_min).replace(0, 1)
        dot_color_df = (dot_color_df - col_min) / denom
    elif standard_scale == "group":
        row_min = dot_color_df.min(axis=1)
        row_max = dot_color_df.max(axis=1)
        denom = (row_max - row_min).replace(0, 1)
        dot_color_df = dot_color_df.sub(row_min, axis=0).div(denom, axis=0)

    dot_color_df = dot_color_df.reindex(columns=valid_genes)
    dot_size_df  = dot_size_df.reindex(columns=valid_genes)
    return dot_color_df, dot_size_df


def _size_to_scatter(
    frac: float,
    smallest_dot: float = 0.0,
    largest_dot: float = 200.0,
    size_exponent: float = 1.5,
) -> float:
    """Scanpy-identical dot-size mapping.

    ``size = frac ** size_exponent * (largest_dot - smallest_dot) + smallest_dot``
    """
    return frac ** size_exponent * (largest_dot - smallest_dot) + smallest_dot


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def plot_combination_dotplot(
    adata,
    var_names: List[str],
    groupby: str,
    gene_sources: Dict[str, str],
    output_path: Union[str, Path],
    *,
    source_colors: Optional[Dict[str, str]] = None,
    source_label_overrides: Optional[Dict[str, str]] = None,
    expression_cutoff: float = 0.0,
    mean_only_expressed: bool = False,
    standard_scale: Optional[str] = "var",
    smallest_dot: float = 0.0,
    largest_dot: float = 200.0,
    size_exponent: float = 1.5,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    colorbar_title: str = "Mean expression\n(scaled per gene)",
    dpi: int = 300,
    gene_label_fontsize: int = 9,
    group_label_fontsize: int = 10,
    swap_axes: bool = False,
) -> None:
    """Plot a combination dotplot with per-gene source label colouring.

    Dot rendering is 100 % custom (no scanpy), using exact scanpy calculations:

    * **Dot colour** — mean expression mapped through a single ``Reds``
      colourmap (normalised over the global min/max after any
      ``standard_scale`` transform).
    * **Dot size**   — fraction of expressing cells, scaled with the same
      power-law as scanpy (``frac ** 1.5 * 200``).
    * **X-axis tick labels** (gene names) are coloured by selection source.

    Parameters
    ----------
    adata:
        AnnData with log-normalised expression in ``.X``.
    var_names:
        Ordered list of genes to display.
    groupby:
        Column in ``adata.obs`` defining groups (e.g. cell types).
    gene_sources:
        Mapping ``gene → source_label``, e.g.
        ``{"GENE1": "dimred", "GENE2": "rf_deg"}``.
    output_path:
        Where to save the PNG.
    source_colors:
        Override the default source → colour mapping.
    source_label_overrides:
        Override legend display labels, e.g.
        ``{"dimred": "NMF", "rf_deg": "DEG"}``.
    expression_cutoff:
        Threshold for expressing cells (scanpy default 0.0).
    mean_only_expressed:
        Average only over expressing cells (scanpy default False).
    standard_scale:
        ``"var"`` (default) normalises each gene to [0, 1]; ``None`` for raw.
    smallest_dot, largest_dot, size_exponent:
        Dot-size scaling (scanpy defaults: 0, 200, 1.5).
    figsize:
        Override auto figsize.
    title:
        Figure suptitle.
    colorbar_title:
        Label on the expression colorbar.
    dpi:
        PNG resolution (default 300).
    gene_label_fontsize, group_label_fontsize:
        Axis tick-label font sizes.
    swap_axes:
        If True, genes on y-axis, groups on x-axis.
    """
    _src_colors = {**DEFAULT_SOURCE_COLORS, **(source_colors or {})}
    _label_map  = {**DEFAULT_SOURCE_LABEL_MAP}
    if source_label_overrides:
        _label_map.update(source_label_overrides)

    # ------------------------------------------------------------------ #
    # 1.  Compute dot data (scanpy-identical)                              #
    # ------------------------------------------------------------------ #
    dot_color_df, dot_size_df = _prepare_dot_data(
        adata,
        var_names=var_names,
        groupby=groupby,
        expression_cutoff=expression_cutoff,
        mean_only_expressed=mean_only_expressed,
        standard_scale=standard_scale,
    )

    valid_genes = list(dot_color_df.columns)
    groups      = list(dot_color_df.index)
    n_genes     = len(valid_genes)
    n_groups    = len(groups)

    # Global expression range → single Reds colormap
    global_vmin = float(dot_color_df.values.min())
    global_vmax = float(dot_color_df.values.max())
    _norm        = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)
    _cmap        = plt.cm.Reds

    # ------------------------------------------------------------------ #
    # 2.  Figure layout                                                    #
    # ------------------------------------------------------------------ #
    if figsize is None:
        if swap_axes:
            w = max(6,  n_groups * 0.45 + 3)
            h = max(5,  n_genes  * 0.22 + 2.5)
        else:
            w = max(12, n_genes  * 0.22 + 3)
            h = max(5,  n_groups * 0.45 + 2.5)
        figsize = (w, h)

    fig = plt.figure(figsize=figsize)

    # GridSpec: [main | gap | colorbar | gap | size-legend]
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1, 0.02, 0.03, 0.02, 0.10],
        left=0.14, right=0.95, top=0.88, bottom=0.24,
        wspace=0.05,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_sleg = fig.add_subplot(gs[0, 4])

    # ------------------------------------------------------------------ #
    # 3.  Draw dots                                                        #
    # ------------------------------------------------------------------ #
    for gi, gene in enumerate(valid_genes):
        for gri, group in enumerate(groups):
            frac     = float(dot_size_df.loc[group, gene])
            mean_exp = float(dot_color_df.loc[group, gene])

            color = _cmap(_norm(mean_exp))
            size  = _size_to_scatter(frac, smallest_dot, largest_dot, size_exponent)

            x, y = (gri, gi) if swap_axes else (gi, gri)
            ax_main.scatter(x, y, s=size, c=[color],
                            linewidths=0.3, edgecolors="0.5", zorder=3)

    # ------------------------------------------------------------------ #
    # 4.  Axes cosmetics                                                   #
    # ------------------------------------------------------------------ #
    if swap_axes:
        ax_main.set_xticks(range(n_groups))
        ax_main.set_xticklabels(groups, rotation=45, ha="right",
                                fontsize=group_label_fontsize)
        ax_main.set_yticks(range(n_genes))
        ax_main.set_yticklabels(valid_genes, fontsize=gene_label_fontsize)
        for tick, gene in zip(ax_main.get_yticklabels(), valid_genes):
            tick.set_color(_src_colors.get(gene_sources.get(gene, "other"),
                                           _src_colors["other"]))
        ax_main.set_xlim(-0.7, n_groups - 0.3)
        ax_main.set_ylim(-0.7, n_genes  - 0.3)
    else:
        ax_main.set_xticks(range(n_genes))
        ax_main.set_xticklabels(valid_genes, rotation=90, ha="center",
                                fontsize=gene_label_fontsize)
        ax_main.set_yticks(range(n_groups))
        ax_main.set_yticklabels(groups, fontsize=group_label_fontsize)
        for tick, gene in zip(ax_main.get_xticklabels(), valid_genes):
            tick.set_color(_src_colors.get(gene_sources.get(gene, "other"),
                                           _src_colors["other"]))
        ax_main.set_xlim(-0.7, n_genes  - 0.3)
        ax_main.set_ylim(-0.7, n_groups - 0.3)

    ax_main.grid(True, linewidth=0.4, color="0.88", zorder=0)
    ax_main.set_axisbelow(True)
    ax_main.tick_params(length=0)
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=11, y=0.93)

    # ------------------------------------------------------------------ #
    # 5.  Colorbar                                                         #
    # ------------------------------------------------------------------ #
    sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar)
    cb.set_label(colorbar_title, fontsize=9, labelpad=4)
    cb.ax.tick_params(labelsize=8)
    ax_cbar.yaxis.set_label_position("right")
    ax_cbar.yaxis.tick_right()

    # ------------------------------------------------------------------ #
    # 6.  Dot-size legend                                                  #
    # ------------------------------------------------------------------ #
    ax_sleg.set_xlim(0, 1)
    ax_sleg.set_ylim(-0.5, 5)
    ax_sleg.axis("off")
    ax_sleg.set_title("Fraction\nexpressing", fontsize=9, loc="center", pad=2)

    for i, frac in enumerate([0.25, 0.50, 0.75, 1.0]):
        sz = _size_to_scatter(frac, smallest_dot, largest_dot, size_exponent)
        ax_sleg.scatter(0.4, i, s=sz, c="grey", linewidths=0.3, edgecolors="0.5", zorder=3)
        ax_sleg.text(0.65, i, f"{int(frac * 100)}%", va="center", ha="left", fontsize=9)

    # ------------------------------------------------------------------ #
    # 7.  Source-colour legend (gene-label colours)                        #
    # ------------------------------------------------------------------ #
    present_sources = sorted({gene_sources.get(g, "other") for g in valid_genes})
    patches = [
        mpatches.Patch(
            facecolor=_src_colors.get(src, _src_colors["other"]),
            edgecolor="0.4", linewidth=0.6,
            label=_label_map.get(src, src),
        )
        for src in present_sources
    ]
    if patches:
        fig.legend(
            handles=patches,
            title="Gene source",
            title_fontsize=11,
            fontsize=10,
            loc="lower center",
            ncol=min(4, len(patches)),
            bbox_to_anchor=(0.44, 0.0),
            frameon=True,
            edgecolor="0.8",
        )

    # ------------------------------------------------------------------ #
    # 8.  Save                                                             #
    # ------------------------------------------------------------------ #
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Combination dotplot saved → %s", output_path)
