"""Tangram vs NMF reconstruction comparison plotting utilities.

Provides bar-chart visualisations of reconstruction quality metrics from
Tangram (stratified k-fold CV) and NMF (repeated random-split CV), enabling
a fair side-by-side comparison with mean ± std error bars.

Functions:
    plot_reconstruction_per_celltype: Per-celltype MSE and ExpVar bar charts.
    plot_reconstruction_aggregated_metrics: Macro/weighted aggregate bar charts.
    load_tangram_per_celltype_csv: Helper to read Tangram per-celltype CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _constants import DEFAULT_PNG_DPI

logger = logging.getLogger(__name__)

__all__ = [
    "plot_reconstruction_per_celltype",
    "plot_reconstruction_aggregated_metrics",
    "load_tangram_per_celltype_csv",
]

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

_TANGRAM_COLOR =  "#FF9800"   # orange
_NMF_COLOR     =  "#2196F3"   # blue


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def load_tangram_per_celltype_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a Tangram per-celltype result CSV.

    Args:
        csv_path: Path to the per-celltype CSV produced by
            ``_reconstruction.py``.

    Returns:
        DataFrame with cell types as rows (``__summary__`` row excluded).
        Returns an empty DataFrame if the file does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning("Tangram CSV not found: %s", csv_path)
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Drop the aggregated summary row if present
    df = df[df["celltype"] != "__summary__"].copy()
    # Drop skipped cell types
    if "skipped" in df.columns:
        df = df[~df["skipped"].astype(bool)]
    return df


def _load_summary_row(csv_path: str | Path) -> pd.Series | None:
    """Return the ``__summary__`` row from a Tangram per-celltype CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    summary = df[df["celltype"] == "__summary__"]
    if summary.empty:
        return None
    return summary.iloc[0]


# ---------------------------------------------------------------------------
# Per-celltype bar chart
# ---------------------------------------------------------------------------


def plot_reconstruction_per_celltype(
    tangram_csv: str | Path,
    nmf_celltype_results: dict[str, dict[str, Any]] | None,
    output_path: str | Path,
    dataset_name: str = "",
    png_dpi: int = DEFAULT_PNG_DPI,
) -> None:
    """Side-by-side bar chart of per-celltype MSE and ExpVar (Tangram vs NMF).

    Each cell type gets two grouped bars: Tangram (blue) and NMF (orange).
    Error bars show the std across CV folds.  Cell types are ordered by
    Tangram expvar_mean descending.

    Args:
        tangram_csv: Path to Tangram per-celltype CSV.
        nmf_celltype_results: Per-celltype NMF CV results dict
            (``evaluate_mechanistic_representation_by_celltype_cv`` output,
            keyed by cell-type → metrics dict).  Pass ``None`` to plot
            Tangram only.
        output_path: Where to save the PNG.
        dataset_name: Used in the plot title.
        png_dpi: Output resolution.
    """
    tg_df = load_tangram_per_celltype_csv(tangram_csv)
    if tg_df.empty:
        logger.warning("No Tangram per-celltype data — skipping plot.")
        return

    # Determine which metric columns exist (k-fold CV adds _std suffix)
    tg_mse_col    = "mse_std"    if "mse_std"    in tg_df.columns else None
    tg_expvar_col = "expvar_std" if "expvar_std" in tg_df.columns else None

    # Sort cell types by Tangram expvar (mean) descending
    sort_col = "expvar" if "expvar" in tg_df.columns else tg_df.columns[0]
    tg_df = tg_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    cell_types = tg_df["celltype"].tolist()
    n_ct = len(cell_types)
    x = np.arange(n_ct)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n_ct * 0.9), 6))
    title_suffix = f" — {dataset_name}" if dataset_name else ""

    for ax, metric, metric_std_col, ylabel, title_metric in [
        (axes[0], "mse",    tg_mse_col,    "MSE",              "MSE"),
        (axes[1], "expvar", tg_expvar_col, "Explained Variance", "Explained Variance"),
    ]:
        tg_vals  = tg_df[metric].values if metric in tg_df.columns else np.full(n_ct, np.nan)
        tg_errs  = tg_df[metric_std_col].values if metric_std_col and metric_std_col in tg_df.columns else np.zeros(n_ct)

        has_nmf = nmf_celltype_results is not None
        if has_nmf:
            nmf_vals = np.array([
                float(nmf_celltype_results.get(ct, {}).get(f"mse_test_probe" if metric == "mse" else "expvar_test_probe", np.nan))
                for ct in cell_types
            ])
            nmf_errs = np.array([
                float(nmf_celltype_results.get(ct, {}).get(f"mse_test_probe_std" if metric == "mse" else "expvar_test_probe_std", 0.0))
                for ct in cell_types
            ])
            bars_tg = ax.bar(x - width / 2, tg_vals,  width, yerr=tg_errs,  label="Tangram", color=_TANGRAM_COLOR, capsize=3, alpha=0.85)
            ax.bar(x + width / 2, nmf_vals, width, yerr=nmf_errs, label="NMF",     color=_NMF_COLOR,     capsize=3, alpha=0.85)
        else:
            ax.bar(x, tg_vals, width * 1.4, yerr=tg_errs, label="Tangram", color=_TANGRAM_COLOR, capsize=3, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(cell_types, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_metric} per cell type{title_suffix}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-celltype reconstruction plot: %s", output_path)


# ---------------------------------------------------------------------------
# Aggregated metrics bar chart
# ---------------------------------------------------------------------------


def plot_reconstruction_aggregated_metrics(
    tangram_csv: str | Path,
    nmf_summary: dict[str, Any] | None,
    output_path: str | Path,
    dataset_name: str = "",
    png_dpi: int = DEFAULT_PNG_DPI,
) -> None:
    """Grouped bar chart of macro/weighted MSE and ExpVar (Tangram vs NMF).

    Shows four metric groups: macro MSE, weighted MSE, macro ExpVar, weighted
    ExpVar.  Each group has two bars (Tangram and NMF) with error bars.

    Args:
        tangram_csv: Path to Tangram per-celltype CSV (summary row is read
            from here).
        nmf_summary: NMF CV summary dict (``"summary"`` sub-dict from
            ``evaluate_mechanistic_representation_by_celltype_cv`` output).
            Pass ``None`` to plot Tangram only.
        output_path: Where to save the PNG.
        dataset_name: Used in the plot title.
        png_dpi: Output resolution.
    """
    tg_summary = _load_summary_row(tangram_csv)
    if tg_summary is None:
        logger.warning("No Tangram summary row found in %s — skipping aggregated plot.", tangram_csv)
        return

    # ── Extract Tangram aggregate metrics ─────────────────────────────────
    def _tg(key: str, default: float = np.nan) -> float:
        return float(tg_summary.get(key, default))

    tg_metrics = {
        "macro_mse":     (_tg("macro_mse"),     _tg("macro_mse_std",     0.0)),
        "weighted_mse":  (_tg("weighted_mse"),  _tg("weighted_mse_std",  0.0)),
        "macro_expvar":  (_tg("macro_expvar"),  _tg("macro_expvar_std",  0.0)),
        "weighted_expvar":(_tg("weighted_expvar"),_tg("weighted_expvar_std",0.0)),
    }

    # ── Extract NMF aggregate metrics ─────────────────────────────────────
    has_nmf = nmf_summary is not None

    def _nmf(key: str, default: float = np.nan) -> float:
        return float((nmf_summary or {}).get(key, default))

    nmf_metrics = {
        "macro_mse":     (_nmf("macro_mse_test_probe"),      _nmf("macro_mse_test_probe_std",      0.0)),
        "weighted_mse":  (_nmf("weighted_mse_test_probe"),   _nmf("weighted_mse_test_probe_std",   0.0)),
        "macro_expvar":  (_nmf("macro_expvar_test_probe"),   _nmf("macro_expvar_test_probe_std",   0.0)),
        "weighted_expvar":(_nmf("weighted_expvar_test_probe"),_nmf("weighted_expvar_test_probe_std",0.0)),
    }

    metric_labels = {
        "macro_mse":      "Macro MSE",
        "weighted_mse":   "Weighted MSE",
        "macro_expvar":   "Macro ExpVar",
        "weighted_expvar":"Weighted ExpVar",
    }
    metric_keys = list(metric_labels.keys())
    n_groups = len(metric_keys)
    x = np.arange(n_groups)
    width = 0.35

    # Use two sub-figures: one for MSE metrics, one for ExpVar metrics
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    title_suffix = f" — {dataset_name}" if dataset_name else ""

    for ax, group_keys, group_title in [
        (axes[0], ["macro_mse", "weighted_mse"],     f"MSE{title_suffix}"),
        (axes[1], ["macro_expvar", "weighted_expvar"], f"Explained Variance{title_suffix}"),
    ]:
        g_x = np.arange(len(group_keys))
        tg_vals = np.array([tg_metrics[k][0] for k in group_keys])
        tg_errs = np.array([tg_metrics[k][1] for k in group_keys])
        xticklabels = [metric_labels[k] for k in group_keys]

        if has_nmf:
            nmf_vals = np.array([nmf_metrics[k][0] for k in group_keys])
            nmf_errs = np.array([nmf_metrics[k][1] for k in group_keys])
            ax.bar(g_x - width / 2, tg_vals,  width, yerr=tg_errs,  label="Tangram", color=_TANGRAM_COLOR, capsize=4, alpha=0.85)
            ax.bar(g_x + width / 2, nmf_vals, width, yerr=nmf_errs, label="NMF",     color=_NMF_COLOR,     capsize=4, alpha=0.85)
        else:
            ax.bar(g_x, tg_vals, width * 1.4, yerr=tg_errs, label="Tangram", color=_TANGRAM_COLOR, capsize=4, alpha=0.85)

        ax.set_xticks(g_x)
        ax.set_xticklabels(xticklabels, fontsize=9)
        ax.set_title(group_title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"Aggregate reconstruction metrics{title_suffix}", fontsize=11)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved aggregated reconstruction plot: %s", output_path)
