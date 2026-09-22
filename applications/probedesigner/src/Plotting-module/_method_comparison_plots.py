"""Cross-method reconstruction comparison (NMF / PCA / ICA / Ridge / Tangram).

One figure compares how well each method reconstructs the full transcriptome from
a probe panel, for several panels side by side. Rows are gene scopes (all genes,
non-panel genes only); columns are explained variance (global mean and
variance-weighted) and MSE.

Data spaces differ per method and are part of the label: NMF and Tangram run on raw
counts, PCA and ICA on log-normalised data, Ridge in both (two entries). ExpVar is
scale-free; MSE is only comparable *within* one data space.

Functions:
    load_method_comparison_table: Read every method's global results into a long table.
    plot_method_comparison: Draw the figure from that table.
"""

from __future__ import annotations

import logging
from pathlib import Path

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

__all__ = ["METHODS", "SUBSETS", "MODES", "load_method_comparison_table", "plot_method_comparison"]

# (label, results subdir, data space, layout, column suffix). "probe" layout = the
# NMF/PCA/ICA/Ridge CSVs (``*_test_probe_*``, analysis_type == "global" row);
# "tangram" layout = ``tangram/<panel>/global/<panel>.csv``.
METHODS: list[tuple[str, str, str, str, str]] = [
    ("NMF",     "nmf",              "raw",     "probe",   ""),
    ("PCA",     "PCA",              "lognorm", "probe",   ""),
    ("ICA",     "ICA",              "lognorm", "probe",   ""),
    ("Ridge",   "ridge-regression", "raw",     "probe",   "_raw"),
    ("Ridge",   "ridge-regression", "lognorm", "probe",   "_lognorm"),
    ("Tangram", "tangram",          "raw",     "tangram", ""),
]
SUBSETS = ("all_genes", "panel_genes_only", "non_panel_genes_only")
MODES = ("global_mean", "variance_weighted_sum")

_SUBSET_TITLE = {
    "all_genes": "All genes",
    "panel_genes_only": "Panel genes only",
    "non_panel_genes_only": "Non-panel genes only",
}
_MODE_TITLE = {"global_mean": "ExpVar (global mean)", "variance_weighted_sum": "ExpVar (variance-weighted)"}
_PANEL_COLORS = ["#9933FF", "#BB88FF", "#957d06", "#2196F3", "#4CAF50"]  # RecoVar, RecoVar light, Spapros, ...

_RC = {
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
}


def _cols(layout: str, suffix: str, subset: str, mode: str | None) -> tuple[str, str]:
    """Return the (mean, std) column names for ``mode`` (``None`` → MSE)."""
    if layout == "probe":
        base = f"mse_test_probe_{subset}{suffix}" if mode is None else f"expvar_test_probe_{subset}_{mode}{suffix}"
    else:
        base = f"mse_{subset}" if mode is None else f"expvar_{subset}_{mode}"
    return f"{base}_mean", f"{base}_std"


def _read_row(results_root: Path, panel: str, subdir: str, layout: str) -> pd.Series:
    if layout == "tangram":
        path = results_root / subdir / panel / "global" / f"{panel}.csv"
    else:
        path = results_root / subdir / f"{panel}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    df = pd.read_csv(path)
    if layout == "probe":
        df = df[df["analysis_type"] == "global"]
    if df.empty:
        raise ValueError(f"No global row in {path}")
    return df.iloc[0]


def load_method_comparison_table(
    results_roots: dict[str, Path],
    panel_labels: dict[str, str],
    subsets: tuple[str, ...] = SUBSETS,
) -> pd.DataFrame:
    """Collect every method's global reconstruction scores into one long table.

    Args:
        results_roots: ``{panel_name: <...>/Variability-Evaluation/results}``.
        panel_labels: ``{panel_name: label shown in the legend}`` (order = bar order).
        subsets: Gene scopes to read.

    Returns:
        Long DataFrame: ``panel, panel_label, method, space, subset, metric, mean, std``
        where ``metric`` is ``mse`` or ``expvar_<mode>``.

    Raises:
        FileNotFoundError: A method's results file is missing.
        KeyError: A required column is missing (never silently plotted as NaN).
    """
    rows: list[dict] = []
    for panel, label in panel_labels.items():
        for method, subdir, space, layout, suffix in METHODS:
            row = _read_row(Path(results_roots[panel]), panel, subdir, layout)
            for subset in subsets:
                for mode in (None, *MODES):
                    mean_col, std_col = _cols(layout, suffix, subset, mode)
                    if mean_col not in row.index:
                        raise KeyError(f"{method} ({space}) / {panel}: column '{mean_col}' not found")
                    rows.append({
                        "panel": panel, "panel_label": label, "method": method, "space": space,
                        "subset": subset, "metric": "mse" if mode is None else f"expvar_{mode}",
                        "mean": float(row[mean_col]),
                        "std": float(row[std_col]) if std_col in row.index and pd.notna(row[std_col]) else 0.0,
                    })
    df = pd.DataFrame(rows)
    if df["mean"].isna().any():
        bad = df[df["mean"].isna()][["panel", "method", "space", "subset", "metric"]]
        raise ValueError(f"NaN scores in comparison table:\n{bad}")
    return df


def plot_method_comparison(
    table: pd.DataFrame,
    output_path: str | Path,
    subsets: tuple[str, ...] = ("all_genes", "non_panel_genes_only"),
    png_dpi: int = DEFAULT_PNG_DPI,
) -> None:
    """Grouped-bar comparison: rows = gene scope, columns = ExpVar (2 modes) and MSE.

    Args:
        table: Output of :func:`load_method_comparison_table`.
        output_path: PNG path.
        subsets: Gene scopes to draw as rows.
        png_dpi: Output resolution.
    """
    panels = list(dict.fromkeys(table["panel_label"]))
    methods = list(dict.fromkeys(zip(table["method"], table["space"])))
    metrics = [f"expvar_{m}" for m in MODES] + ["mse"]
    n_p = len(panels)
    width = 0.8 / n_p
    x = np.arange(len(methods))
    xlabels = [f"{m}\n({s})" for m, s in methods]

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(len(subsets), 3, figsize=(19, 5.2 * len(subsets)), squeeze=False)
        for r, subset in enumerate(subsets):
            for c, metric in enumerate(metrics):
                ax = axes[r][c]
                sub = table[(table["subset"] == subset) & (table["metric"] == metric)]
                for i, panel in enumerate(panels):
                    vals, errs = [], []
                    for method, space in methods:
                        hit = sub[(sub["panel_label"] == panel) & (sub["method"] == method) & (sub["space"] == space)]
                        vals.append(float(hit["mean"].iloc[0]))
                        errs.append(float(hit["std"].iloc[0]))
                    pos = x + (i - (n_p - 1) / 2) * width
                    bars = ax.bar(pos, vals, width, yerr=errs, capsize=2, alpha=0.9,
                                  color=_PANEL_COLORS[i % len(_PANEL_COLORS)], label=panel)
                    for b, v in zip(bars, vals):
                        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                                    textcoords="offset points", ha="center", va="bottom",
                                    fontsize=7, rotation=90)
                ax.set_xticks(x)
                ax.set_xticklabels(xlabels)
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                if metric == "mse":
                    ax.set_yscale("log")
                    ax.set_ylabel("MSE (log scale)")
                    ax.set_title(f"{_SUBSET_TITLE[subset]} — MSE")
                else:
                    ax.set_ylim(min(0.0, sub["mean"].min() - 0.05), 1.15)
                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.set_ylabel("Explained variance")
                    ax.set_title(f"{_SUBSET_TITLE[subset]} — {_MODE_TITLE[metric[len('expvar_'):]]}")
        handles, labs = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(labs), frameon=False)
        fig.suptitle("Reconstruction of the full transcriptome from a probe panel — 5-fold CV, mean ± std across folds")
        fig.text(0.5, 0.005,
                 "Bracket = data space the method was run in (raw counts / log-normalised). "
                 "ExpVar is scale-free; MSE is only comparable within one data space. "
                 "Tangram/NMF/Ridge(raw) are scored on raw counts, PCA/ICA/Ridge(lognorm) on log-normalised values.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=[0, 0.02, 1, 0.925])
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved method-comparison plot: %s", output_path)
