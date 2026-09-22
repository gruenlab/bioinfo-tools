"""Tangram vs NMF reconstruction comparison plotting utilities.

Provides bar-chart visualisations of reconstruction quality metrics from
Tangram and NMF (both stratified k-fold CV), enabling a fair side-by-side
comparison with mean ± std error bars.

Functions:
    plot_reconstruction_per_celltype: Per-celltype MSE and ExpVar bar charts.
    plot_reconstruction_aggregated_metrics: Macro/weighted/global aggregate bar charts.
    load_tangram_per_celltype_csv: Helper to read Tangram per-celltype CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- load THIS directory's _constants.py by path (sibling dirs share the name) ---
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _cpath
_cspec = _ilu.spec_from_file_location("_constants", _cpath(__file__).resolve().parent / "_constants.py")
_sys.modules["_constants"] = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_sys.modules["_constants"])


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

# Explicit sizes: _variability_plots / _clustering_plots set 20/18/16pt rcParams at
# import time, which makes these (wide, many-tick) figures overflow.
_RC = {
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
}


def _series(df: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (value, std) columns for ``name`` from a results DataFrame.

    Aggregated CSVs store ``<name>_mean`` / ``<name>_std``; older per-fold or
    unaggregated CSVs store the bare ``<name>``. Missing columns give NaN / 0.
    """
    n = len(df)
    for val_col in (f"{name}_mean", name):
        if val_col in df.columns:
            vals = pd.to_numeric(df[val_col], errors="coerce").to_numpy(dtype=float)
            std_col = f"{name}_std"
            errs = (
                pd.to_numeric(df[std_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                if std_col in df.columns else np.zeros(n)
            )
            return vals, errs
    return np.full(n, np.nan), np.zeros(n)


def _scalar(row: Any, name: str) -> tuple[float, float]:
    """Scalar analogue of :func:`_series` for a dict / Series (one CSV row)."""
    if row is None:
        return np.nan, 0.0
    get = row.get
    for val_key in (f"{name}_mean", name):
        v = get(val_key, None)
        if v is not None and pd.notna(v):
            s = get(f"{name}_std", 0.0)
            return float(v), float(s) if pd.notna(s) else 0.0
    return np.nan, 0.0


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

    Each cell type gets two grouped bars: Tangram (orange) and NMF (blue).
    Error bars show the std across CV folds.  Cell types are ordered by
    Tangram expvar descending.

    Args:
        tangram_csv: Path to Tangram per-celltype CSV.
        nmf_celltype_results: Per-celltype NMF results keyed by cell type →
            row dict (``mse_test_probe[_mean/_std]`` /
            ``expvar_test_probe[_mean/_std]``).  ``None`` plots Tangram only.
        output_path: Where to save the PNG.
        dataset_name: Panel name, used once in the figure title.
        png_dpi: Output resolution.
    """
    tg_df = load_tangram_per_celltype_csv(tangram_csv)
    if tg_df.empty:
        logger.warning("No Tangram per-celltype data — skipping plot.")
        return

    # Sort cell types by Tangram expvar descending
    expvar_vals, _ = _series(tg_df, "expvar")
    if np.isnan(expvar_vals).all():
        logger.warning("Tangram CSV %s has no expvar column — cell types left unsorted.", tangram_csv)
    else:
        tg_df = tg_df.iloc[np.argsort(-np.nan_to_num(expvar_vals, nan=-np.inf), kind="stable")]
    tg_df = tg_df.reset_index(drop=True)
    cell_types = tg_df["celltype"].tolist()
    n_ct = len(cell_types)
    x = np.arange(n_ct)
    width = 0.38
    has_nmf = nmf_celltype_results is not None

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(max(12, n_ct * 1.0), 6))

        for ax, tg_name, nmf_name, ylabel in [
            (axes[0], "mse",    "mse_test_probe",    "MSE"),
            (axes[1], "expvar", "expvar_test_probe", "Explained variance"),
        ]:
            tg_vals, tg_errs = _series(tg_df, tg_name)
            if np.isnan(tg_vals).all():
                logger.warning("Tangram '%s' is all-NaN in %s — Tangram bars omitted.", tg_name, tangram_csv)
            nmf_vals = nmf_errs = None
            if has_nmf:
                pairs = [_scalar(nmf_celltype_results.get(ct), nmf_name) for ct in cell_types]
                nmf_vals = np.array([p[0] for p in pairs])
                nmf_errs = np.array([p[1] for p in pairs])
                if np.isnan(nmf_vals).all():
                    logger.warning("NMF '%s' is all-NaN — NMF bars omitted.", nmf_name)
                    nmf_vals = None

            if nmf_vals is not None:
                ax.bar(x - width / 2, tg_vals,  width, yerr=tg_errs,  label="Tangram", color=_TANGRAM_COLOR, capsize=3, alpha=0.85)
                ax.bar(x + width / 2, nmf_vals, width, yerr=nmf_errs, label="NMF",     color=_NMF_COLOR,     capsize=3, alpha=0.85)
            else:
                ax.bar(x, tg_vals, width * 1.4, yerr=tg_errs, label="Tangram", color=_TANGRAM_COLOR, capsize=3, alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(cell_types, rotation=45, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} per cell type")
            ax.legend(loc="best")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        if dataset_name:
            fig.suptitle(dataset_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
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
    tangram_global_csv: str | Path | None = None,
    nmf_global: dict[str, Any] | None = None,
) -> None:
    """Grouped bar chart of macro / weighted / global MSE and ExpVar.

    Two axes (MSE, Explained variance), each with up to three groups (Macro,
    Weighted, Global) and two bars per group (Tangram, NMF) with error bars.

    Args:
        tangram_csv: Tangram per-celltype CSV (its ``__summary__`` row holds the
            macro/weighted aggregates).
        nmf_summary: NMF ``per_celltype``/``summary`` row (``macro_*_test_probe``,
            ``weighted_*_test_probe``); ``None`` plots Tangram only.
        output_path: Where to save the PNG.
        dataset_name: Panel name, used once in the figure title.
        png_dpi: Output resolution.
        tangram_global_csv: Optional Tangram ``global/<panel>.csv`` (adds the
            "Global" group).
        nmf_global: Optional NMF ``global`` row (adds the "Global" group).
    """
    tg_summary = _load_summary_row(tangram_csv)
    if tg_summary is None:
        logger.warning("No Tangram summary row found in %s — skipping aggregated plot.", tangram_csv)
        return

    tg_global = None
    if tangram_global_csv is not None and Path(tangram_global_csv).exists():
        gdf = pd.read_csv(tangram_global_csv)
        tg_global = gdf.iloc[0] if not gdf.empty else None

    # (label, tangram row, tangram key, nmf row, nmf key)
    groups = {
        "mse": [
            ("Macro",    tg_summary, "macro_mse",    nmf_summary, "macro_mse_test_probe"),
            ("Weighted", tg_summary, "weighted_mse", nmf_summary, "weighted_mse_test_probe"),
            ("Global",   tg_global,  "mse",          nmf_global,  "mse_test_probe"),
        ],
        "expvar": [
            ("Macro",    tg_summary, "macro_expvar",    nmf_summary, "macro_expvar_test_probe"),
            ("Weighted", tg_summary, "weighted_expvar", nmf_summary, "weighted_expvar_test_probe"),
            ("Global",   tg_global,  "expvar",          nmf_global,  "expvar_test_probe"),
        ],
    }
    width = 0.38

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, (metric, ylabel) in zip(axes, [("mse", "MSE"), ("expvar", "Explained variance")]):
            rows = [r for r in groups[metric] if r[1] is not None or r[3] is not None]
            labels = [r[0] for r in rows]
            g_x = np.arange(len(rows))
            tg = np.array([_scalar(r[1], r[2]) for r in rows]).reshape(-1, 2)
            nm = np.array([_scalar(r[3], r[4]) for r in rows]).reshape(-1, 2)
            has_tg = not np.isnan(tg[:, 0]).all()
            has_nmf = nmf_summary is not None and not np.isnan(nm[:, 0]).all()
            if not has_tg:
                logger.warning("Tangram '%s' aggregates are all-NaN — Tangram bars omitted.", metric)
            if nmf_summary is not None and not has_nmf:
                logger.warning("NMF '%s' aggregates are all-NaN — NMF bars omitted.", metric)

            if has_nmf:
                ax.bar(g_x - width / 2, tg[:, 0], width, yerr=tg[:, 1], label="Tangram", color=_TANGRAM_COLOR, capsize=4, alpha=0.85)
                ax.bar(g_x + width / 2, nm[:, 0], width, yerr=nm[:, 1], label="NMF",     color=_NMF_COLOR,     capsize=4, alpha=0.85)
            else:
                ax.bar(g_x, tg[:, 0], width * 1.4, yerr=tg[:, 1], label="Tangram", color=_TANGRAM_COLOR, capsize=4, alpha=0.85)

            ax.set_xticks(g_x)
            ax.set_xticklabels(labels)
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend(loc="best")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.suptitle(dataset_name if dataset_name else "Aggregate reconstruction metrics")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved aggregated reconstruction plot: %s", output_path)
