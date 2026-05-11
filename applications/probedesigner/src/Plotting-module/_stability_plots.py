"""Stability analysis plotting functions.

This module contains publication-quality plotting functions for visualizing
gene selection stability and metric variability across pipeline iterations.

Functions moved from Analysis-scripts/run_stability_analysis.py and
optimized for publication quality with increased font sizes and DPI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

try:
    from ._constants import (
        ANALYSIS_PNG_DPI,
        PUB_COLORBAR_SIZE,
        PUB_LABEL_SIZE,
        PUB_LEGEND_SIZE,
        PUB_SUPTITLE_SIZE,
        PUB_TICK_SIZE,
        PUB_TITLE_SIZE,
        PUB_VALUE_SIZE,
        STABILITY_METRIC_COLOR,
        STABILITY_NMF_COLOR,
        STABILITY_RF_COLOR,
    )
except ImportError:
    # Support direct script execution where package-relative imports are unavailable.
    from _constants import (
        ANALYSIS_PNG_DPI,
        PUB_COLORBAR_SIZE,
        PUB_LABEL_SIZE,
        PUB_LEGEND_SIZE,
        PUB_SUPTITLE_SIZE,
        PUB_TICK_SIZE,
        PUB_TITLE_SIZE,
        PUB_VALUE_SIZE,
        STABILITY_METRIC_COLOR,
        STABILITY_NMF_COLOR,
        STABILITY_RF_COLOR,
    )

if TYPE_CHECKING:
    from typing import Optional

logger = logging.getLogger(__name__)


def plot_gene_frequency(
    gene_df: pd.DataFrame,
    output_dir: Path,
    n_iterations: int,
    top_n: Optional[int] = None,
    png_dpi: int = ANALYSIS_PNG_DPI,
    output_filename: str = "gene_frequency.png",
    title_suffix: str = "",
) -> None:
    """Horizontal bar chart of gene selection frequency across stability iterations.

    **Publication-optimized** square plot with larger fonts for clarity.

    Bars are coloured on a sequential blue scale: darker = selected in more iterations.
    Only genes selected in more than one iteration are shown; at most *top_n* genes
    are plotted (or all if top_n is None).

    Parameters
    ----------
    gene_df : pd.DataFrame
        DataFrame with columns: 'gene', 'n_selected', 'fraction_selected'
    output_dir : Path
        Output directory (plots/ subdirectory will be created)
    n_iterations : int
        Total number of stability iterations
    top_n : int, optional
        Maximum number of genes to display (None = all)
    png_dpi : int, default ANALYSIS_PNG_DPI (600)
        Resolution for saved PNG
    output_filename : str, default "gene_frequency.png"
        File name used inside ``{output_dir}/plots/``
    title_suffix : str, default ""
        Extra text appended to the plot title

    Outputs
    -------
    Saves to ``{output_dir}/plots/gene_frequency.png``
    """
    try:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        df = gene_df[gene_df["n_selected"] > 1].copy()
        if df.empty:
            logger.warning("No genes were selected in >1 iteration — skipping gene frequency plot.")
            return

        # Limit to top_n if specified, sorted ascending so the most-selected gene appears at the top
        if top_n is not None:
            df = df.nlargest(top_n, "n_selected").sort_values("n_selected", ascending=True)
        else:
            df = df.sort_values("n_selected", ascending=True)

        # PUBLICATION CHANGE: Square aspect ratio with reduced width
        fig, ax = plt.subplots(figsize=(10, 10))

        # Map fraction_selected → colour using Blues colormap
        cmap = plt.get_cmap("Blues")
        colors = [cmap(0.35 + 0.65 * f) for f in df["fraction_selected"].values]

        bars = ax.barh(df["gene"].values, df["n_selected"].values, color=colors, height=0.7)

        # Add colorbar indicating fraction
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
        cbar.set_label("Fraction of iterations selected", fontsize=PUB_COLORBAR_SIZE)
        cbar.ax.tick_params(labelsize=PUB_TICK_SIZE)

        # PUBLICATION CHANGES: Larger fonts
        ax.set_xlabel("Number of iterations selected", fontsize=PUB_LABEL_SIZE, labelpad=10)
        ax.set_title(
            f"Gene selection frequency ({n_iterations} iterations){title_suffix}",
            fontsize=PUB_TITLE_SIZE,
            pad=20,
        )
        ax.set_xlim(0, n_iterations + 0.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)
        ax.tick_params(axis="x", labelsize=PUB_TICK_SIZE)
        ax.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        out = plots_dir / output_filename
        fig.savefig(out, dpi=png_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved gene frequency plot: %s", out)
    except Exception as exc:
        logger.warning("Gene frequency plot failed: %s", exc, exc_info=True)


def plot_gene_overlap(
    gene_df: pd.DataFrame,
    output_dir: Path,
    n_iterations: int,
    png_dpi: int = ANALYSIS_PNG_DPI,
    output_filename: str = "gene_overlap.png",
    title_suffix: str = "",
    first_pool_col: str = "n_rf_pool",
    second_pool_col: Optional[str] = "n_nmf_pool",
    first_pool_label: str = "RF pool",
    second_pool_label: str = "NMF pool",
    show_legend: bool = True,
) -> None:
    """Grouped bar chart showing gene stability distribution by selection method.

    Parameters
    ----------
    gene_df : pd.DataFrame
        DataFrame with count columns for the two pools being compared.
    output_dir : Path
        Output directory (plots/ subdirectory will be created)
    n_iterations : int
        Total number of stability iterations
    png_dpi : int, default ANALYSIS_PNG_DPI (600)
        Resolution for saved PNG
    output_filename : str, default "gene_overlap.png"
        File name used inside ``{output_dir}/plots/``
    title_suffix : str, default ""
        Extra text appended to the plot title
    first_pool_col : str, default "n_rf_pool"
        Column containing counts for the first pool/method
    second_pool_col : str | None, default "n_nmf_pool"
        Column containing counts for the second pool/method. If None, a
        single-series plot is generated using only ``first_pool_col``.
    first_pool_label : str, default "RF pool"
        Legend label for the first pool/method
    second_pool_label : str, default "NMF pool"
        Legend label for the second pool/method
    show_legend : bool, default True
        Whether to draw the legend.
    """
    try:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if gene_df.empty:
            logger.warning("Gene table is empty — skipping gene overlap plot.")
            return

        if first_pool_col not in gene_df.columns:
            logger.warning("No pool count information available — skipping gene overlap plot.")
            return

        has_second_pool = second_pool_col is not None and second_pool_col in gene_df.columns

        stability_data = []
        for freq in range(1, n_iterations + 1):
            first_count = int((gene_df[first_pool_col] == freq).sum())
            second_count = int((gene_df[second_pool_col] == freq).sum()) if has_second_pool else 0
            stability_data.append(
                {
                    "frequency": freq,
                    "first_count": first_count,
                    "second_count": second_count,
                }
            )

        if not stability_data:
            logger.warning("No stability data available for gene overlap plot.")
            return

        stab_df = pd.DataFrame(stability_data)

        total_first_genes = int((gene_df[first_pool_col] > 0).sum())
        total_second_genes = int((gene_df[second_pool_col] > 0).sum()) if has_second_pool else 0
        stab_df["first_pct"] = (stab_df["first_count"] / total_first_genes * 100) if total_first_genes > 0 else 0
        stab_df["second_pct"] = (stab_df["second_count"] / total_second_genes * 100) if total_second_genes > 0 else 0

        x = np.arange(len(stab_df))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 10))

        if has_second_pool:
            bars_first = ax.bar(
                x - width / 2,
                stab_df["first_count"],
                width,
                label=first_pool_label,
                color=STABILITY_RF_COLOR,
                alpha=0.9,
                edgecolor="white",
                linewidth=1.0,
            )
            bars_second = ax.bar(
                x + width / 2,
                stab_df["second_count"],
                width,
                label=second_pool_label,
                color=STABILITY_NMF_COLOR,
                alpha=0.9,
                edgecolor="white",
                linewidth=1.0,
            )
        else:
            bars_first = ax.bar(
                x,
                stab_df["first_count"],
                width=0.6,
                label=first_pool_label,
                color=STABILITY_RF_COLOR,
                alpha=0.9,
                edgecolor="white",
                linewidth=1.0,
            )
            bars_second = []

        def autolabel(bars, percentages):
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{int(height)}\n({pct:.1f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=PUB_VALUE_SIZE,
                        fontweight="bold",
                    )

        autolabel(bars_first, stab_df["first_pct"])
        if has_second_pool:
            autolabel(bars_second, stab_df["second_pct"])

        ax.set_xlabel("Number of iterations gene was selected", fontsize=PUB_LABEL_SIZE, labelpad=10)
        ax.set_ylabel("Number of genes", fontsize=PUB_LABEL_SIZE, labelpad=10)
        ax.set_title(
            f"Gene Stability by Selection Frequency ({n_iterations} iterations){title_suffix}",
            fontsize=PUB_TITLE_SIZE,
            fontweight="bold",
            pad=20,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{freq}" for freq in stab_df["frequency"]], fontsize=PUB_TICK_SIZE)
        ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)
        if show_legend and has_second_pool:
            ax.legend(fontsize=PUB_LEGEND_SIZE, loc="upper right", framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        out = plots_dir / output_filename
        fig.savefig(out, dpi=png_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved gene overlap plot: %s", out)

    except Exception as exc:
        logger.warning("Gene overlap plot failed: %s", exc, exc_info=True)


def plot_metrics_summary(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    png_dpi: int = ANALYSIS_PNG_DPI,
):
    """Bar charts of mean ± std MSE and ExpVar per cell type across stability iterations.

    **Publication-optimized** with larger fonts and improved layout.

    Reads the ``summary_mean`` / ``summary_std`` rows already present in *metrics_df*
    and saves:

    - ``plots/stability_metrics_summary.png``        — per-celltype MSE / ExpVar
    - ``plots/stability_metrics_global_summary.png`` — global MSE / ExpVar

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metric table with summary rows.
    output_dir : Path
        Output directory (plots/ subdirectory will be created)
    png_dpi : int, default ANALYSIS_PNG_DPI (600)
        Resolution for saved PNG
    """
    try:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        mean_rows = metrics_df[metrics_df["iteration"] == "summary_mean"].copy()
        std_rows = metrics_df[metrics_df["iteration"] == "summary_std"].copy()

        if mean_rows.empty:
            logger.warning(
                "No summary_mean rows found in metrics — skipping metrics summary plot."
            )
            return

        # ── Per-celltype panel ───────────────────────────────────────────────
        ct_mean = mean_rows[mean_rows["analysis_type"] == "per_celltype"].copy()
        ct_std = std_rows[std_rows["analysis_type"] == "per_celltype"].copy()

        if not ct_mean.empty:
            # Sort cell types by expvar mean descending
            if "expvar" in ct_mean.columns:
                ct_mean = ct_mean.sort_values("expvar", ascending=False)

            celltypes = ct_mean["celltype"].values
            n_ct = len(celltypes)
            y = np.arange(n_ct)
            std_map = ct_std.set_index("celltype") if not ct_std.empty else pd.DataFrame()

            def _get_std(ct: str, col: str) -> float:
                if not std_map.empty and ct in std_map.index and col in std_map.columns:
                    v = std_map.loc[ct, col]
                    return float(v) if pd.notna(v) else 0.0
                return 0.0

            n_iter = (
                int(
                    metrics_df["iteration"]
                    .apply(lambda x: x if str(x).isdigit() else -1)
                    .astype(int)
                    .max()
                )
                + 1
                if any(str(x).isdigit() for x in metrics_df["iteration"])
                else "?"
            )

            # PUBLICATION CHANGE: Larger figure, improved spacing
            fig, axes = plt.subplots(1, 2, figsize=(16, max(8, n_ct * 0.4)))

            for ax, col, xlabel in [
                (axes[0], "mse", "MSE"),
                (axes[1], "expvar", "Explained Variance"),
            ]:
                if col not in ct_mean.columns:
                    ax.set_visible(False)
                    continue
                vals = ct_mean[col].values.astype(float)
                errs = np.array([_get_std(ct, col) for ct in celltypes])

                ax.barh(
                    y,
                    vals,
                    xerr=errs,
                    color=STABILITY_METRIC_COLOR,
                    alpha=0.85,
                    capsize=4,
                    height=0.7,
                )
                ax.set_yticks(y)
                ax.set_yticklabels(celltypes, fontsize=PUB_TICK_SIZE)
                ax.set_xlabel(xlabel, fontsize=PUB_LABEL_SIZE, labelpad=10)
                ax.set_title(
                    f"{xlabel} per cell type (mean ± std)", fontsize=PUB_TITLE_SIZE, pad=15
                )
                ax.tick_params(axis="x", labelsize=PUB_TICK_SIZE)
                ax.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            fig.suptitle(
                f"NMF reconstruction stability ({n_iter} iterations)",
                fontsize=PUB_SUPTITLE_SIZE,
                y=1.01,
            )
            fig.tight_layout()
            out = plots_dir / "stability_metrics_summary.png"
            fig.savefig(out, dpi=png_dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved stability metrics summary plot: %s", out)

        # ── Global panel (side-by-side MSE and ExpVar) ───────────────────────
        g_mean = mean_rows[mean_rows["analysis_type"] == "global_reconstruction"].copy()
        g_std = std_rows[std_rows["analysis_type"] == "global_reconstruction"].copy()

        if not g_mean.empty:
            # PUBLICATION CHANGE: Larger figure
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # MSE subplot
            if "mse" in g_mean.columns:
                mse_mean = float(g_mean.iloc[0]["mse"])
                mse_std = (
                    float(g_std.iloc[0]["mse"])
                    if not g_std.empty
                    and "mse" in g_std.columns
                    and pd.notna(g_std.iloc[0]["mse"])
                    else 0.0
                )
                axes[0].bar(
                    [0],
                    [mse_mean],
                    yerr=[mse_std],
                    color=STABILITY_METRIC_COLOR,
                    alpha=0.85,
                    capsize=5,
                    width=0.5,
                )
                axes[0].set_xticks([0])
                axes[0].set_xticklabels(["MSE"], fontsize=PUB_TICK_SIZE)
                axes[0].set_ylabel("MSE", fontsize=PUB_LABEL_SIZE, labelpad=10)
                axes[0].set_title("Mean Squared Error", fontsize=PUB_TITLE_SIZE, pad=15)
                axes[0].tick_params(axis="y", labelsize=PUB_TICK_SIZE)
                axes[0].grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
                axes[0].spines["top"].set_visible(False)
                axes[0].spines["right"].set_visible(False)
            else:
                axes[0].set_visible(False)

            # ExpVar subplot
            if "expvar" in g_mean.columns:
                expvar_mean = float(g_mean.iloc[0]["expvar"])
                expvar_std = (
                    float(g_std.iloc[0]["expvar"])
                    if not g_std.empty
                    and "expvar" in g_std.columns
                    and pd.notna(g_std.iloc[0]["expvar"])
                    else 0.0
                )
                axes[1].bar(
                    [0],
                    [expvar_mean],
                    yerr=[expvar_std],
                    color=STABILITY_METRIC_COLOR,
                    alpha=0.85,
                    capsize=5,
                    width=0.5,
                )
                axes[1].set_xticks([0])
                axes[1].set_xticklabels(["ExpVar"], fontsize=PUB_TICK_SIZE)
                axes[1].set_ylabel("Explained Variance", fontsize=PUB_LABEL_SIZE, labelpad=10)
                axes[1].set_title("Explained Variance", fontsize=PUB_TITLE_SIZE, pad=15)
                axes[1].tick_params(axis="y", labelsize=PUB_TICK_SIZE)
                axes[1].grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
                axes[1].spines["top"].set_visible(False)
                axes[1].spines["right"].set_visible(False)
            else:
                axes[1].set_visible(False)

            fig.suptitle(
                "Global NMF reconstruction (mean ± std)", fontsize=PUB_SUPTITLE_SIZE, y=1.02
            )
            fig.tight_layout()
            out = plots_dir / "stability_metrics_global_summary.png"
            fig.savefig(out, dpi=png_dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved global metrics summary plot: %s", out)

    except Exception as exc:
        logger.warning("Metrics summary plot failed: %s", exc, exc_info=True)


def plot_aggregate_metrics_summary(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    png_dpi: int = ANALYSIS_PNG_DPI,
) -> None:
    """Bar chart of aggregate metrics (macro and weighted) with mean ± std across iterations.

    **Publication-optimized** with larger fonts and improved layout.

    Shows macro and weighted MSE/ExpVar for per-celltype analysis in a side-by-side layout.
    Reads summary statistics from the ``summary_mean`` / ``summary_std`` rows.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with aggregate metrics columns
    output_dir : Path
        Output directory (plots/ subdirectory will be created)
    png_dpi : int, default ANALYSIS_PNG_DPI (600)
        Resolution for saved PNG

    Outputs
    -------
    Saves to ``{output_dir}/plots/stability_aggregate_metrics.png``
    """
    try:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        mean_rows = metrics_df[metrics_df["iteration"] == "summary_mean"].copy()
        std_rows = metrics_df[metrics_df["iteration"] == "summary_std"].copy()

        if mean_rows.empty:
            logger.warning(
                "No summary_mean rows found in metrics — skipping aggregate metrics plot."
            )
            return

        # Filter for per-celltype analysis (aggregate metrics are only meaningful for per-celltype)
        ct_mean = mean_rows[mean_rows["analysis_type"] == "per_celltype"].copy()
        ct_std = std_rows[std_rows["analysis_type"] == "per_celltype"].copy()

        if ct_mean.empty:
            logger.warning(
                "No per-celltype analysis found — skipping aggregate metrics plot."
            )
            return

        # Check if aggregate metric columns exist
        required_cols = ["macro_mse", "macro_expvar", "weighted_mse", "weighted_expvar"]
        missing_cols = [c for c in required_cols if c not in ct_mean.columns]
        if missing_cols:
            logger.warning(
                "Missing aggregate metric columns %s — skipping aggregate metrics plot.",
                missing_cols,
            )
            return

        # Extract unique aggregate values (they're the same across all celltypes in summary)
        # Just take the first row since macro/weighted are duplicated
        mean_vals = ct_mean.iloc[0]
        std_vals = ct_std.iloc[0] if not ct_std.empty else None

        # Prepare data for plotting
        macro_mse = float(mean_vals["macro_mse"])
        macro_expvar = float(mean_vals["macro_expvar"])
        weighted_mse = float(mean_vals["weighted_mse"])
        weighted_expvar = float(mean_vals["weighted_expvar"])

        # Get standard deviations
        if std_vals is not None and not pd.isna(std_vals["macro_mse"]):
            macro_mse_std = float(std_vals["macro_mse"])
            macro_expvar_std = float(std_vals["macro_expvar"])
            weighted_mse_std = float(std_vals["weighted_mse"])
            weighted_expvar_std = float(std_vals["weighted_expvar"])
        else:
            macro_mse_std = macro_expvar_std = weighted_mse_std = weighted_expvar_std = 0.0

        # Determine number of iterations for title
        n_iter = (
            int(
                metrics_df["iteration"]
                .apply(lambda x: x if str(x).isdigit() else -1)
                .astype(int)
                .max()
            )
            + 1
            if any(str(x).isdigit() for x in metrics_df["iteration"])
            else "?"
        )

        # PUBLICATION CHANGE: Larger figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # MSE subplot (left)
        mse_labels = ["Macro", "Weighted"]
        mse_means = [macro_mse, weighted_mse]
        mse_stds = [macro_mse_std, weighted_mse_std]
        x_pos = np.arange(len(mse_labels))

        axes[0].bar(
            x_pos,
            mse_means,
            yerr=mse_stds,
            color=STABILITY_METRIC_COLOR,
            alpha=0.85,
            capsize=5,
            width=0.6,
        )
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(mse_labels, fontsize=PUB_TICK_SIZE)
        axes[0].set_ylabel("MSE", fontsize=PUB_LABEL_SIZE, labelpad=10)
        axes[0].set_title("Aggregate MSE (mean ± std)", fontsize=PUB_TITLE_SIZE, pad=15)
        axes[0].tick_params(axis="y", labelsize=PUB_TICK_SIZE)
        axes[0].grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # ExpVar subplot (right)
        expvar_labels = ["Macro", "Weighted"]
        expvar_means = [macro_expvar, weighted_expvar]
        expvar_stds = [macro_expvar_std, weighted_expvar_std]

        axes[1].bar(
            x_pos,
            expvar_means,
            yerr=expvar_stds,
            color=STABILITY_METRIC_COLOR,
            alpha=0.85,
            capsize=5,
            width=0.6,
        )
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(expvar_labels, fontsize=PUB_TICK_SIZE)
        axes[1].set_ylabel("Explained Variance", fontsize=PUB_LABEL_SIZE, labelpad=10)
        axes[1].set_title("Aggregate ExpVar (mean ± std)", fontsize=PUB_TITLE_SIZE, pad=15)
        axes[1].tick_params(axis="y", labelsize=PUB_TICK_SIZE)
        axes[1].grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.8)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        fig.suptitle(
            f"Aggregate metrics stability ({n_iter} iterations)",
            fontsize=PUB_SUPTITLE_SIZE,
            y=1.02,
            fontweight="bold",
        )
        fig.tight_layout()
        out = plots_dir / "stability_aggregate_metrics.png"
        fig.savefig(out, dpi=png_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved aggregate metrics summary plot: %s", out)

    except Exception as exc:
        logger.warning("Aggregate metrics plot failed: %s", exc, exc_info=True)


def plot_feature_umaps(
    adata: sc.AnnData,
    all_genes: list[str],
    celltype_col: str,
    output_dir: Path,
    png_dpi: int = ANALYSIS_PNG_DPI,
) -> None:
    """UMAP feature plots for every gene that was selected in at least one iteration.

    **Publication-optimized** with higher DPI default (600 instead of 150).

    A cell-type overview UMAP is saved first, followed by batched gene-expression
    feature plots (16 genes per grid figure).  A UMAP embedding is computed on the
    fly if one is not already present in ``adata.obsm``.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object with expression data
    all_genes : list[str]
        List of gene names to plot
    celltype_col : str
        Column name in adata.obs for cell types
    output_dir : Path
        Output directory (plots/feature_plots/ subdirectory will be created)
    png_dpi : int, default ANALYSIS_PNG_DPI (600)
        Resolution for saved PNG (increased from 150 for publication quality)
    """
    try:
        feature_dir = output_dir / "plots" / "feature_plots"
        feature_dir.mkdir(parents=True, exist_ok=True)

        # Filter to genes present in the AnnData
        genes_in = [g for g in all_genes if g in adata.var_names]
        if not genes_in:
            logger.warning(
                "None of the selected genes are in adata.var_names — skipping feature plots."
            )
            return

        logger.info("Generating UMAP feature plots for %d selected genes ...", len(genes_in))

        # ── Compute UMAP if not already present ──────────────────────────────
        if "X_umap" not in adata.obsm:
            logger.info("No precomputed UMAP found — computing PCA → neighbours → UMAP ...")
            sc.pp.pca(adata, n_comps=30)
            sc.pp.neighbors(adata, n_neighbors=15)
            sc.tl.umap(adata)
        else:
            logger.info("Using precomputed UMAP from adata.obsm['X_umap'].")

        # ── Cell-type overview ────────────────────────────────────────────────
        try:
            fig = sc.pl.umap(adata, color=celltype_col, show=False, return_fig=True)
            ct_path = feature_dir / "celltype_umap.png"
            fig.savefig(ct_path, dpi=png_dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved cell-type UMAP: %s", ct_path)
        except Exception as exc:
            logger.warning("Cell-type UMAP failed: %s", exc)

        # ── Batched gene feature plots (16 per figure) ────────────────────────
        batch_size = 16
        n_batches = (len(genes_in) + batch_size - 1) // batch_size
        for b in range(n_batches):
            batch = genes_in[b * batch_size : (b + 1) * batch_size]
            try:
                fig = sc.pl.umap(
                    adata,
                    color=batch,
                    ncols=4,
                    show=False,
                    return_fig=True,
                )
                grid_path = feature_dir / f"feature_grid_{b:02d}.png"
                fig.savefig(grid_path, dpi=png_dpi, bbox_inches="tight")
                plt.close(fig)
                logger.info(
                    "Saved feature grid %d/%d (%d genes): %s",
                    b + 1,
                    n_batches,
                    len(batch),
                    grid_path,
                )
            except Exception as exc:
                logger.warning("Feature grid %d failed: %s", b, exc, exc_info=True)

    except Exception as exc:
        logger.warning("Feature UMAP plotting failed: %s", exc, exc_info=True)
