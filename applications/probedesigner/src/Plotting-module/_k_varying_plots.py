"""K-varying analysis plotting functions.

This module contains publication-quality plotting functions for visualizing
reconstruction quality and gene selection stability across different K values
(number of NMF/cNMF components).

Functions moved from Analysis-scripts/plot_k_varying_results.py and
plot_per_celltype_grid.py, optimized for publication quality with configurable
DPI, formats, colors, and font sizes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from ._constants import (
        ANALYSIS_PNG_DPI,
        K_VARYING_CNMF_COLOR,
        K_VARYING_NMF_COLOR,
        PUB_LABEL_SIZE,
        PUB_LEGEND_SIZE,
        PUB_SUPTITLE_SIZE,
        PUB_TICK_SIZE,
        PUB_TITLE_SIZE,
        PUB_VALUE_SIZE,
    )
except ImportError:
    # Support direct script execution where package-relative imports are unavailable.
    from _constants import (
        ANALYSIS_PNG_DPI,
        K_VARYING_CNMF_COLOR,
        K_VARYING_NMF_COLOR,
        PUB_LABEL_SIZE,
        PUB_LEGEND_SIZE,
        PUB_SUPTITLE_SIZE,
        PUB_TICK_SIZE,
        PUB_TITLE_SIZE,
        PUB_VALUE_SIZE,
    )

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Set seaborn style
sns.set_style("whitegrid")


def plot_reconstruction_quality(
    df: pd.DataFrame,
    output_dir: Path,
    metric_type: str = "global",
    png_dpi: int = ANALYSIS_PNG_DPI,
    fig_format: str = "png",
    colors: Optional[Dict[str, str]] = None,
    metric_filter: Optional[List[str]] = None,
) -> None:
    """Plot reconstruction quality metrics (MSE/ExpVar) vs K for different methods.

    Creates separate plots for MSE and Explained Variance, with subplots for
    different evaluation methods (NMF-eval and cNMF-eval).

    Parameters
    ----------
    df : pd.DataFrame
        Reconstruction quality dataframe with columns: k_value, iteration,
        selection_method, eval_method, and metric columns
    output_dir : Path
        Directory to save plots (plots/ subdirectory will be created)
    metric_type : str, default "global"
        Type of metrics to plot: 'global', 'macro', or 'weighted'
    png_dpi : int, default ANALYSIS_PNG_DPI (600)
        Resolution for saved plots
    fig_format : str, default "png"
        Output format: 'png', 'pdf', or 'svg'
    colors : dict, optional
        Custom colors for selection methods. Default: NMF=blue, cNMF=orange
    metric_filter : list, optional
        Subset of metrics to render from {'mse', 'expvar'}. If None, renders both.

    Outputs
    -------
    Saves plots to ``{output_dir}/plots/``

:
        - ``{eval_method}_eval_mse_{metric_type}.{format}``
        - ``{eval_method}_eval_expvar_{metric_type}.{format}``
    """
    logger.info(f"Generating reconstruction quality plots ({metric_type} metrics)")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Define metric column names based on type
    if metric_type == "global":
        mse_col = "mse_global"
        expvar_col = "expvar_global"
    elif metric_type == "macro":
        mse_col = "macro_mse"
        expvar_col = "macro_expvar"
    elif metric_type == "weighted":
        mse_col = "weighted_mse"
        expvar_col = "weighted_expvar"
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")

    # Check if columns exist
    if mse_col not in df.columns or expvar_col not in df.columns:
        logger.warning(f"Columns {mse_col} or {expvar_col} not found. Skipping {metric_type} plots.")
        return

    # Use default or custom colors
    if colors is None:
        colors = {"nmf": K_VARYING_NMF_COLOR, "cnmf": K_VARYING_CNMF_COLOR}

    selected_metrics = {"mse", "expvar"}
    if metric_filter:
        selected_metrics = {m.lower().strip() for m in metric_filter if m.lower().strip() in {"mse", "expvar"}}
        if not selected_metrics:
            logger.warning("metric_filter did not contain valid entries ('mse', 'expvar'). Rendering both.")
            selected_metrics = {"mse", "expvar"}

    # Plot configurations: (eval_method, metric_col, ylabel, filename_suffix)
    plot_configs = [
        ("nmf", mse_col, f"MSE (lower is better)", "mse"),
        ("nmf", expvar_col, f"Explained Variance (higher is better)", "expvar"),
        ("cnmf", mse_col, f"MSE (lower is better)", "mse"),
        ("cnmf", expvar_col, f"Explained Variance (higher is better)", "expvar"),
    ]

    for eval_method, metric_col, ylabel, suffix in plot_configs:
        if suffix not in selected_metrics:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for selection_method in ["nmf", "cnmf"]:
            # Filter data
            mask = (
                (df["eval_method"] == eval_method)
                & (df["selection_method"] == selection_method)
            )
            subset = df[mask]

            if len(subset) == 0:
                logger.warning(f"No data for eval={eval_method}, selection={selection_method}")
                continue

            # Aggregate: mean ± std per K
            agg = subset.groupby("k_value")[metric_col].agg(["mean", "std"]).reset_index()

            # Plot
            label = f"{selection_method.upper()}-selection"
            ax.errorbar(
                agg["k_value"],
                agg["mean"],
                yerr=agg["std"],
                marker="o",
                label=label,
                capsize=5,
                linewidth=2,
                markersize=8,
                color=colors.get(selection_method, K_VARYING_NMF_COLOR if selection_method == "nmf" else K_VARYING_CNMF_COLOR),
                alpha=0.8,
            )

        ax.set_xlabel("Number of Components (K)", fontsize=PUB_LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=PUB_LABEL_SIZE)
        title_metric = suffix.upper()
        ax.set_title(
            f"{eval_method.upper()}-evaluation: {title_metric} vs K ({metric_type} metrics)",
            fontsize=PUB_TITLE_SIZE,
            fontweight="bold",
        )
        ax.legend(fontsize=PUB_LEGEND_SIZE)

        # Set x-axis ticks to exact k-values
        k_values = sorted(df["k_value"].unique())
        ax.set_xticks(k_values)
        ax.set_xticklabels([int(k) for k in k_values], fontsize=PUB_TICK_SIZE)
        ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)

        ax.grid(True, alpha=0.3)

        output_path = plots_dir / f"{eval_method}_eval_{suffix}_{metric_type}.{fig_format}"
        fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {output_path}")


def plot_gene_stability(
    overlap_df: pd.DataFrame,
    output_dir: Path,
    png_dpi: int = ANALYSIS_PNG_DPI,
    fig_format: str = "png",
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """Plot gene selection overlap/stability across iterations.

    Generates two plots:
    1. Jaccard similarity vs K
    2. Intersection percentage vs K

    Parameters
    ----------
    overlap_df : pd.DataFrame
        Gene overlap summary dataframe with columns: k_value, method,
        pairwise_jaccard, intersection_pct
    output_dir : Path
        Directory to save plots (plots/ subdirectory will be created)
    png_dpi : int, default ANALYSIS_PNG_DPI (600)
        Resolution for saved plots
    fig_format : str, default "png"
        Output format: 'png', 'pdf', or 'svg'
    colors : dict, optional
        Custom colors for selection methods. Default: NMF=blue, cNMF=orange

    Outputs
    -------
    Saves plots to ``{output_dir}/plots/``:
        - ``gene_stability_jaccard.{format}``
        - ``gene_stability_intersection_pct.{format}``
    """
    logger.info("Generating gene overlap plots")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if "pairwise_jaccard" not in overlap_df.columns:
        logger.warning("Gene overlap columns not found. Skipping overlap plots.")
        return

    # Use default or custom colors
    if colors is None:
        colors = {"nmf": K_VARYING_NMF_COLOR, "cnmf": K_VARYING_CNMF_COLOR}

    # Plot Jaccard similarity vs K
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in overlap_df["method"].unique():
        subset = overlap_df[overlap_df["method"] == method]

        ax.plot(
            subset["k_value"],
            subset["pairwise_jaccard"],
            marker="o",
            label=f"{method.upper()}-selection",
            linewidth=2,
            markersize=8,
            color=colors.get(method, "#333333"),
            alpha=0.8,
        )

    ax.set_xlabel("Number of Components (K)", fontsize=PUB_LABEL_SIZE)
    ax.set_ylabel("Pairwise Jaccard Similarity", fontsize=PUB_LABEL_SIZE)
    ax.set_title(
        "Gene Selection Stability vs K",
        fontsize=PUB_TITLE_SIZE,
        fontweight="bold",
    )
    ax.legend(fontsize=PUB_LEGEND_SIZE)

    # Set x-axis ticks to exact k-values
    k_values = sorted(overlap_df["k_value"].unique())
    ax.set_xticks(k_values)
    ax.set_xticklabels([int(k) for k in k_values], fontsize=PUB_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)

    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    output_path = plots_dir / f"gene_stability_jaccard.{fig_format}"
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")

    # Plot intersection percentage vs K
    if "intersection_pct" in overlap_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        for method in overlap_df["method"].unique():
            subset = overlap_df[overlap_df["method"] == method]

            ax.plot(
                subset["k_value"],
                subset["intersection_pct"],
                marker="o",
                label=f"{method.upper()}-selection",
                linewidth=2,
                markersize=8,
                color=colors.get(method, "#333333"),
                alpha=0.8,
            )

        ax.set_xlabel("Number of Components (K)", fontsize=PUB_LABEL_SIZE)
        ax.set_ylabel("Intersection Percentage (%)", fontsize=PUB_LABEL_SIZE)
        ax.set_title(
            "Gene Selection Overlap vs K",
            fontsize=PUB_TITLE_SIZE,
            fontweight="bold",
        )
        ax.legend(fontsize=PUB_LEGEND_SIZE)

        # Set x-axis ticks to exact k-values
        k_values = sorted(overlap_df["k_value"].unique())
        ax.set_xticks(k_values)
        ax.set_xticklabels([int(k) for k in k_values], fontsize=PUB_TICK_SIZE)
        ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)

        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        output_path = plots_dir / f"gene_stability_intersection_pct.{fig_format}"
        fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {output_path}")


def plot_aggregate_metrics(
    results_df: pd.DataFrame,
    output_dir: Path,
    png_dpi: int = ANALYSIS_PNG_DPI,
    fig_format: str = "png",
    colors: Optional[Dict[str, str]] = None,
    metric_filter: Optional[List[str]] = None,
) -> None:
    """Plot comparison of all available metric types (global, macro, weighted).

    Generates comparison plots showing global, macro-averaged, and weighted
    MSE and ExpVar on the same plot for each eval/selection method combination.

    Parameters
    ----------
    results_df : pd.DataFrame
        Reconstruction quality dataframe
    output_dir : Path
        Directory to save plots (plots/ subdirectory will be created)
    png_dpi : int, default ANALYSIS_PNG_DPI (600)
        Resolution for saved plots
    fig_format : str, default "png"
        Output format: 'png', 'pdf', or 'svg'
    colors : dict, optional
        Custom colors for metric types. Default: global=blue, macro=orange, weighted=green
    metric_filter : list, optional
        Subset of metrics to render from {'mse', 'expvar'}. If None, renders both.

    Outputs
    -------
    Saves plots to ``{output_dir}/plots/``:
        - ``comparison_mse_{selection_method}sel_{eval_method}eval.{format}``
        - ``comparison_expvar_{selection_method}sel_{eval_method}eval.{format}``
    """
    logger.info("Generating metric comparison plots")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Define metric groups
    mse_metrics = ["mse_global", "macro_mse", "weighted_mse"]
    expvar_metrics = ["expvar_global", "macro_expvar", "weighted_expvar"]

    # Check which metrics are available
    available_mse = [m for m in mse_metrics if m in results_df.columns]
    available_expvar = [m for m in expvar_metrics if m in results_df.columns]

    selected_metrics = {"mse", "expvar"}
    if metric_filter:
        selected_metrics = {m.lower().strip() for m in metric_filter if m.lower().strip() in {"mse", "expvar"}}
        if not selected_metrics:
            logger.warning("metric_filter did not contain valid entries ('mse', 'expvar'). Rendering both.")
            selected_metrics = {"mse", "expvar"}

    if ("mse" not in selected_metrics or not available_mse) and ("expvar" not in selected_metrics or not available_expvar):
        logger.warning("No metrics available for comparison plots")
        return

    metric_labels = {
        "mse_global": "Global MSE",
        "macro_mse": "Macro-avg MSE",
        "weighted_mse": "Weighted-avg MSE",
        "expvar_global": "Global ExpVar",
        "macro_expvar": "Macro-avg ExpVar",
        "weighted_expvar": "Weighted-avg ExpVar",
    }

    # Use default or custom colors
    if colors is None:
        colors_by_metric = {
            "mse_global": "#1f77b4",
            "macro_mse": "#ff7f0e",
            "weighted_mse": "#2ca02c",
            "expvar_global": "#1f77b4",
            "macro_expvar": "#ff7f0e",
            "weighted_expvar": "#2ca02c",
        }
    else:
        colors_by_metric = colors

    # Create plots for each eval_method and selection_method combination
    for eval_method in results_df["eval_method"].unique():
        for selection_method in results_df["selection_method"].unique():
            mask = (
                (results_df["eval_method"] == eval_method)
                & (results_df["selection_method"] == selection_method)
            )
            subset = results_df[mask]

            if len(subset) == 0:
                continue

            # Plot MSE metrics
            if "mse" in selected_metrics and available_mse:
                fig, ax = plt.subplots(figsize=(10, 6))

                for metric in available_mse:
                    agg = subset.groupby("k_value")[metric].agg(["mean", "std"]).reset_index()
                    ax.errorbar(
                        agg["k_value"],
                        agg["mean"],
                        yerr=agg["std"],
                        marker="o",
                        label=metric_labels[metric],
                        capsize=5,
                        linewidth=2,
                        markersize=8,
                        color=colors_by_metric.get(metric, "#333333"),
                        alpha=0.8,
                    )

                ax.set_xlabel("Number of Components (K)", fontsize=PUB_LABEL_SIZE)
                ax.set_ylabel("MSE (lower is better)", fontsize=PUB_LABEL_SIZE)
                ax.set_title(
                    f"{selection_method.upper()}-selection, {eval_method.upper()}-eval: MSE Comparison",
                    fontsize=PUB_TITLE_SIZE,
                    fontweight="bold",
                )
                ax.legend(fontsize=PUB_LEGEND_SIZE)

                # Set x-axis ticks to exact k-values
                k_values = sorted(results_df["k_value"].unique())
                ax.set_xticks(k_values)
                ax.set_xticklabels([int(k) for k in k_values], fontsize=PUB_TICK_SIZE)
                ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)

                ax.grid(True, alpha=0.3)

                output_path = plots_dir / f"comparison_mse_{selection_method}sel_{eval_method}eval.{fig_format}"
                fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Saved: {output_path}")

            # Plot ExpVar metrics
            if "expvar" in selected_metrics and available_expvar:
                fig, ax = plt.subplots(figsize=(10, 6))

                for metric in available_expvar:
                    agg = subset.groupby("k_value")[metric].agg(["mean", "std"]).reset_index()
                    ax.errorbar(
                        agg["k_value"],
                        agg["mean"],
                        yerr=agg["std"],
                        marker="o",
                        label=metric_labels[metric],
                        capsize=5,
                        linewidth=2,
                        markersize=8,
                        color=colors_by_metric.get(metric, "#333333"),
                        alpha=0.8,
                    )

                ax.set_xlabel("Number of Components (K)", fontsize=PUB_LABEL_SIZE)
                ax.set_ylabel("Explained Variance (higher is better)", fontsize=PUB_LABEL_SIZE)
                ax.set_title(
                    f"{selection_method.upper()}-selection, {eval_method.upper()}-eval: ExpVar Comparison",
                    fontsize=PUB_TITLE_SIZE,
                    fontweight="bold",
                )
                ax.legend(fontsize=PUB_LEGEND_SIZE)

                # Set x-axis ticks to exact k-values
                k_values = sorted(results_df["k_value"].unique())
                ax.set_xticks(k_values)
                ax.set_xticklabels([int(k) for k in k_values], fontsize=PUB_TICK_SIZE)
                ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)

                ax.grid(True, alpha=0.3)

                output_path = plots_dir / f"comparison_expvar_{selection_method}sel_{eval_method}eval.{fig_format}"
                fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Saved: {output_path}")


def plot_reconstruction_metrics_grid(
    results_df: pd.DataFrame,
    global_df: pd.DataFrame,
    output_dir: Path,
    png_dpi: int = ANALYSIS_PNG_DPI,
    fig_format: str = "png",
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """Create unified 2×3 grid plot showing all reconstruction metrics.

    Allows direct visual comparison of NMF vs cNMF stability across:
    - Global, Macro, and Weighted MSE (row 1)
    - Global, Macro, and Weighted ExpVar (row 2)

    Automatically detects and removes extreme outliers using the IQR method
    (threshold: 10×IQR) to prevent scale distortion.

    Parameters
    ----------
    results_df : pd.DataFrame
        Reconstruction quality results (macro/weighted metrics)
    global_df : pd.DataFrame
        Global reconstruction quality results
    output_dir : Path
        Directory to save plots (plots/ subdirectory will be created)
    png_dpi : int, default ANALYSIS_PNG_DPI (600)
        Resolution for saved plots
    fig_format : str, default "png"
        Output format: 'png', 'pdf', or 'svg'
    colors : dict, optional
        Custom colors for methods. Default: nmf-nmf=blue, cnmf-cnmf=orange

    Outputs
    -------
    Saves to ``{output_dir}/plots/reconstruction_metrics_grid.{format}``
    """
    logger.info("Creating unified reconstruction metrics grid plot...")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # OUTLIER DETECTION AND REMOVAL
    # =========================================================================

    # Detect and remove extreme outliers using IQR method (threshold: 10×IQR)
    outliers_removed = []
    results_df_clean = results_df.copy()

    for metric in ["macro_mse", "weighted_mse", "macro_expvar", "weighted_expvar"]:
        if metric not in results_df_clean.columns:
            continue

        Q1 = results_df_clean[metric].quantile(0.25)
        Q3 = results_df_clean[metric].quantile(0.75)
        IQR = Q3 - Q1

        # Use 10×IQR threshold for extreme outliers only
        lower_bound = Q1 - 10 * IQR
        upper_bound = Q3 + 10 * IQR

        # Identify outliers
        outlier_mask = (results_df_clean[metric] < lower_bound) | (results_df_clean[metric] > upper_bound)

        if outlier_mask.any():
            outlier_rows = results_df_clean[outlier_mask]
            for _, row in outlier_rows.iterrows():
                outlier_info = (
                    f"k={int(row['k_value'])}, "
                    f"{row['selection_method'].upper()}-{row['eval_method'].upper()}, "
                    f"iter={int(row['iteration'])}, "
                    f"{metric}={row[metric]:.2e}"
                )
                outliers_removed.append(outlier_info)
                logger.warning(f"Removing outlier: {outlier_info}")

            # Remove outliers
            results_df_clean = results_df_clean[~outlier_mask]

    if outliers_removed:
        logger.info(f"Removed {len(outliers_removed)} outlier data points")

    # Create 2×3 grid (rows: MSE/ExpVar, cols: Global/Macro/Weighted)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Use default or custom colors
    if colors is None:
        colors = {
            "nmf-nmf": K_VARYING_NMF_COLOR,
            "cnmf-cnmf": K_VARYING_CNMF_COLOR,
        }

    # Get k-values for x-axis
    k_values = sorted(results_df["k_value"].unique())

    # =========================================================================
    # ROW 1: MSE METRICS
    # =========================================================================

    # --- Panel 1,1: Global MSE ---
    ax = axes[0, 0]
    for method in ["nmf-nmf", "cnmf-cnmf"]:
        sel_method, eval_method = method.split("-")
        mask = (
            (global_df["selection_method"] == sel_method)
            & (global_df["eval_method"] == eval_method)
        )
        subset = global_df[mask]

        if "mse_global" in subset.columns:
            agg = subset.groupby("k_value")["mse_global"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                agg["k_value"],
                agg["mean"],
                yerr=agg["std"],
                marker="o",
                label=method.upper(),
                color=colors[method],
                capsize=5,
                linewidth=2,
            )

    ax.set_title("Global MSE", fontsize=PUB_TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("K value", fontsize=PUB_LABEL_SIZE)
    ax.set_ylabel("MSE (lower is better)", fontsize=PUB_LABEL_SIZE)
    ax.set_xticks(k_values)
    ax.set_xticklabels([int(k) for k in k_values], fontsize=PUB_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)
    ax.legend(fontsize=PUB_LEGEND_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Panel 1,2: Macro MSE ---
    ax = axes[0, 1]
    for method in ["nmf-nmf", "cnmf-cnmf"]:
        sel_method, eval_method = method.split("-")
        mask = (
            (results_df_clean["selection_method"] == sel_method)
            & (results_df_clean["eval_method"] == eval_method)
        )
        subset = results_df_clean[mask]

        if "macro_mse" in subset.columns:
            agg = subset.groupby("k_value")["macro_mse"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                agg["k_value"],
                agg["mean"],
                yerr=agg["std"],
                marker="o",
                label=method.upper(),
                color=colors[method],
                capsize=5,
                linewidth=2,
            )

    ax.set_title("Macro-averaged MSE", fontsize=PUB_TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("K value", fontsize=PUB_LABEL_SIZE)
    ax.set_ylabel("MSE (lower is better)", fontsize=PUB_LABEL_SIZE)
    ax.set_xticks(k_values)
    ax.set_xticklabels([int(k) for k in k_values], fontsize=PUB_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)
    ax.legend(fontsize=PUB_LEGEND_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Panel 1,3: Weighted MSE ---
    ax = axes[0, 2]
    for method in ["nmf-nmf", "cnmf-cnmf"]:
        sel_method, eval_method = method.split("-")
        mask = (
            (results_df_clean["selection_method"] == sel_method)
            & (results_df_clean["eval_method"] == eval_method)
        )
        subset = results_df_clean[mask]

        if "weighted_mse" in subset.columns:
            agg = subset.groupby("k_value")["weighted_mse"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                agg["k_value"],
                agg["mean"],
                yerr=agg["std"],
                marker="o",
                label=method.upper(),
                color=colors[method],
                capsize=5,
                linewidth=2,
            )

    ax.set_title("Weighted MSE", fontsize=PUB_TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("K value", fontsize=PUB_LABEL_SIZE)
    ax.set_ylabel("MSE (lower is better)", fontsize=PUB_LABEL_SIZE)
    ax.set_xticks(k_values)
    ax.set_xticklabels([int(k) for k in k_values], fontsize=PUB_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)
    ax.legend(fontsize=PUB_LEGEND_SIZE)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # ROW 2: EXPLAINED VARIANCE METRICS
    # =========================================================================

    # --- Panel 2,1: Global ExpVar ---
    ax = axes[1, 0]
    for method in ["nmf-nmf", "cnmf-cnmf"]:
        sel_method, eval_method = method.split("-")
        mask = (
            (global_df["selection_method"] == sel_method)
            & (global_df["eval_method"] == eval_method)
        )
        subset = global_df[mask]

        if "expvar_global" in subset.columns:
            agg = subset.groupby("k_value")["expvar_global"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                agg["k_value"],
                agg["mean"],
                yerr=agg["std"],
                marker="o",
                label=method.upper(),
                color=colors[method],
                capsize=5,
                linewidth=2,
            )

    ax.set_title("Global Explained Variance", fontsize=PUB_TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("K value", fontsize=PUB_LABEL_SIZE)
    ax.set_ylabel("Explained Variance (higher is better)", fontsize=PUB_LABEL_SIZE)
    ax.set_xticks(k_values)
    ax.set_xticklabels([int(k) for k in k_values], fontsize=PUB_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)
    ax.legend(fontsize=PUB_LEGEND_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Panel 2,2: Macro ExpVar ---
    ax = axes[1, 1]
    for method in ["nmf-nmf", "cnmf-cnmf"]:
        sel_method, eval_method = method.split("-")
        mask = (
            (results_df_clean["selection_method"] == sel_method)
            & (results_df_clean["eval_method"] == eval_method)
        )
        subset = results_df_clean[mask]

        if "macro_expvar" in subset.columns:
            agg = subset.groupby("k_value")["macro_expvar"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                agg["k_value"],
                agg["mean"],
                yerr=agg["std"],
                marker="o",
                label=method.upper(),
                color=colors[method],
                capsize=5,
                linewidth=2,
            )

    ax.set_title("Macro-averaged Explained Variance", fontsize=PUB_TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("K value", fontsize=PUB_LABEL_SIZE)
    ax.set_ylabel("Explained Variance (higher is better)", fontsize=PUB_LABEL_SIZE)
    ax.set_xticks(k_values)
    ax.set_xticklabels([int(k) for k in k_values], fontsize=PUB_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)
    ax.legend(fontsize=PUB_LEGEND_SIZE)
    ax.grid(True, alpha=0.3)

    # --- Panel 2,3: Weighted ExpVar ---
    ax = axes[1, 2]
    for method in ["nmf-nmf", "cnmf-cnmf"]:
        sel_method, eval_method = method.split("-")
        mask = (
            (results_df_clean["selection_method"] == sel_method)
            & (results_df_clean["eval_method"] == eval_method)
        )
        subset = results_df_clean[mask]

        if "weighted_expvar" in subset.columns:
            agg = subset.groupby("k_value")["weighted_expvar"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                agg["k_value"],
                agg["mean"],
                yerr=agg["std"],
                marker="o",
                label=method.upper(),
                color=colors[method],
                capsize=5,
                linewidth=2,
            )

    ax.set_title("Weighted Explained Variance", fontsize=PUB_TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("K value", fontsize=PUB_LABEL_SIZE)
    ax.set_ylabel("Explained Variance (higher is better)", fontsize=PUB_LABEL_SIZE)
    ax.set_xticks(k_values)
    ax.set_xticklabels([int(k) for k in k_values], fontsize=PUB_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)
    ax.legend(fontsize=PUB_LEGEND_SIZE)
    ax.grid(True, alpha=0.3)

    # Overall title
    title_text = "Reconstruction Quality: NMF vs cNMF Comparison Across All Metrics\n(Mean ± Std across iterations)"
    if outliers_removed:
        n_outliers = len(outliers_removed)
        title_text += f"\nNote: {n_outliers} extreme outlier(s) removed (>10×IQR)"

    fig.suptitle(
        title_text,
        fontsize=PUB_SUPTITLE_SIZE,
        fontweight="bold",
        y=0.995,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_path = plots_dir / f"reconstruction_metrics_grid.{fig_format}"
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_baseline_comparison(
    global_df: pd.DataFrame,
    output_dir: Path,
    png_dpi: int = ANALYSIS_PNG_DPI,
    fig_format: str = "png",
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """Plot reconstruction quality relative to baseline.

    Shows MSE improvement percentage over baseline for each evaluation method.

    Parameters
    ----------
    global_df : pd.DataFrame
        Global reconstruction quality dataframe with baseline columns
    output_dir : Path
        Directory to save plots (plots/ subdirectory will be created)
    png_dpi : int, default ANALYSIS_PNG_DPI (600)
        Resolution for saved plots
    fig_format : str, default "png"
        Output format: 'png', 'pdf', or 'svg'
    colors : dict, optional
        Custom colors for selection methods. Default: NMF=blue, cNMF=orange

    Outputs
    -------
    Saves plots to ``{output_dir}/plots/``:
        - ``{eval_method}_eval_mse_baseline_comparison.{format}``
    """
    logger.info("Generating baseline comparison plots")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Check if baseline columns exist
    if "mse_baseline_global" not in global_df.columns:
        logger.warning("Baseline columns not found. Skipping baseline comparison plots.")
        return

    # Calculate relative improvement
    df = global_df.copy()
    df["mse_improvement"] = (df["mse_baseline_global"] - df["mse_global"]) / df["mse_baseline_global"] * 100
    df["expvar_improvement"] = (df["expvar_global"] - df["expvar_baseline_global"]) / np.abs(df["expvar_baseline_global"]) * 100

    # Use default or custom colors
    if colors is None:
        colors = {"nmf": K_VARYING_NMF_COLOR, "cnmf": K_VARYING_CNMF_COLOR}

    # Plot MSE improvement
    for eval_method in df["eval_method"].unique():
        fig, ax = plt.subplots(figsize=(10, 6))

        for selection_method in ["nmf", "cnmf"]:
            mask = (
                (df["eval_method"] == eval_method)
                & (df["selection_method"] == selection_method)
            )
            subset = df[mask]

            if len(subset) == 0:
                continue

            agg = subset.groupby("k_value")["mse_improvement"].agg(["mean", "std"]).reset_index()

            label = f"{selection_method.upper()}-selection"
            ax.errorbar(
                agg["k_value"],
                agg["mean"],
                yerr=agg["std"],
                marker="o",
                label=label,
                capsize=5,
                linewidth=2,
                markersize=8,
                color=colors[selection_method],
                alpha=0.8,
            )

        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
        ax.set_xlabel("Number of Components (K)", fontsize=PUB_LABEL_SIZE)
        ax.set_ylabel("MSE Improvement over Baseline (%)", fontsize=PUB_LABEL_SIZE)
        ax.set_title(
            f"{eval_method.upper()}-evaluation: MSE Improvement vs K",
            fontsize=PUB_TITLE_SIZE,
            fontweight="bold",
        )
        ax.legend(fontsize=PUB_LEGEND_SIZE)

        # Set x-axis ticks to exact k-values
        k_values = sorted(df["k_value"].unique())
        ax.set_xticks(k_values)
        ax.set_xticklabels([int(k) for k in k_values], fontsize=PUB_TICK_SIZE)
        ax.tick_params(axis="y", labelsize=PUB_TICK_SIZE)

        ax.grid(True, alpha=0.3)

        output_path = plots_dir / f"{eval_method}_eval_mse_baseline_comparison.{fig_format}"
        fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {output_path}")


def plot_per_celltype_grid(
    aggregated_data: Dict[str, pd.DataFrame],
    celltypes: List[str],
    k_values: List[int],
    metric: str,
    output_file: Path,
    figsize_per_plot: Tuple[float, float] = (3, 2.5),
    png_dpi: int = ANALYSIS_PNG_DPI,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """Create grid plot showing metric vs k-value for each celltype.

    Each subplot shows the variation of MSE or ExpVar across K values for
    a specific cell type, with separate lines for NMF-NMF and cNMF-cNMF methods.

    Parameters
    ----------
    aggregated_data : dict
        Dictionary mapping celltype to aggregated DataFrame
    celltypes : list
        List of celltypes (already ordered by sample size)
    k_values : list
        List of k-values
    metric : str
        Either "mse" or "expvar"
    output_file : Path
        Path to save figure
    figsize_per_plot : tuple, default (3, 2.5)
        Size of each subplot (width, height)
    png_dpi : int, default ANALYSIS_PNG_DPI (600)
        Resolution for saved plots
    colors : dict, optional
        Custom colors for methods. Default: nmf-nmf=blue, cnmf-cnmf=orange

    Outputs
    -------
    Saves to ``output_file``
    """
    logger.info(f"Creating {metric.upper()} grid plot...")

    n_celltypes = len(celltypes)

    # Determine grid dimensions (prefer wider grids)
    n_cols = int(np.ceil(np.sqrt(n_celltypes * 1.5)))
    n_rows = int(np.ceil(n_celltypes / n_cols))

    logger.info(f"Grid dimensions: {n_rows} rows × {n_cols} columns")

    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
        squeeze=False
    )

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Use default or custom colors
    if colors is None:
        colors = {
            "nmf-nmf": K_VARYING_NMF_COLOR,
            "cnmf-cnmf": K_VARYING_CNMF_COLOR,
        }
    else:
        # Accept either explicit method keys or selection-method aliases.
        colors = {
            "nmf-nmf": colors.get("nmf-nmf", colors.get("nmf", K_VARYING_NMF_COLOR)),
            "cnmf-cnmf": colors.get("cnmf-cnmf", colors.get("cnmf", K_VARYING_CNMF_COLOR)),
        }

    # Plot each celltype
    for idx, celltype in enumerate(celltypes):
        ax = axes_flat[idx]
        ct_df = aggregated_data[celltype]

        # Plot both methods
        for method in ["nmf-nmf", "cnmf-cnmf"]:
            method_df = ct_df[ct_df["method"] == method].sort_values("k_value")

            if method_df.empty:
                continue

            k_vals = method_df["k_value"].values
            mean_vals = method_df[f"{metric}_mean"].values
            std_vals = method_df[f"{metric}_std"].values

            # Plot line with error band
            ax.plot(k_vals, mean_vals, marker='o', label=method.upper(),
                   color=colors[method], linewidth=2, markersize=5)
            ax.fill_between(k_vals, mean_vals - std_vals, mean_vals + std_vals,
                           alpha=0.3, color=colors[method])

        # Format subplot
        n_cells = ct_df["n_cells"].iloc[0] if "n_cells" in ct_df.columns else 0
        n_successful_runs = len(ct_df) // 2  # Divide by 2 methods
        ax.set_title(f"{celltype}\n(n={n_cells:.0f}, {n_successful_runs} runs)", fontsize=9)
        ax.set_xlabel("K value", fontsize=9)

        ylabel = "MSE" if metric == "mse" else "Explained Variance"
        ax.set_ylabel(ylabel, fontsize=9)

        # Set x-axis ticks to full k-value range for consistent scale
        ax.set_xticks(k_values)
        ax.set_xticklabels([int(k) for k in k_values], fontsize=8)
        ax.tick_params(axis='y', labelsize=8)

        # Set x-axis limits to show full range
        ax.set_xlim(min(k_values) - 1, max(k_values) + 1)

        # Add grid for readability
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Add legend only to first subplot
        if idx == 0:
            ax.legend(fontsize=8, loc='best')

    # Hide unused subplots
    for idx in range(n_celltypes, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Overall title
    metric_name = "Mean Squared Error (MSE)" if metric == "mse" else "Explained Variance"
    fig.suptitle(
        f"Per-Celltype {metric_name} Variation Across K-Values\n"
        f"(Mean ± Std across iterations)",
        fontsize=PUB_SUPTITLE_SIZE, fontweight='bold', y=0.995
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=png_dpi, bbox_inches='tight')
    logger.info(f"Saved figure: {output_file}")

    plt.close()
