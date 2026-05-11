#!/usr/bin/env python3
"""Comprehensive plotting CLI for all pipeline analysis types.

This unified CLI provides plotting functionality for:
- K-varying analysis results
- Stability analysis results
- Evaluation results (via plot_evaluation.py)
- Selection results

Usage:
    python plot_pipeline_results.py \\
        --analysis_type {k_varying,stability,evaluation,selection} \\
        --input_dir /path/to/results \\
        --output_dir /path/to/plots \\
        --plot_types reconstruction,stability,metrics \\
        [--dpi 600] \\
        [--format png] \\
        [--color_scheme nmf:#1f77b4,cnmf:#ff7f0e] \\
        [--font_sizes title:20,label:18,tick:16] \\
        [--metric_filter mse,expvar]

Examples:
    # Plot all K-varying results
    python plot_pipeline_results.py \\
        --analysis_type k_varying \\
        --input_dir results/k_varying_analysis \\
        --output_dir plots/k_varying

    # Plot stability analysis with custom DPI
    python plot_pipeline_results.py \\
        --analysis_type stability \\
        --input_dir results/stability_5iter \\
        --output_dir plots/stability \\
        --dpi 600 --format pdf

    # Plot only MSE metrics
    python plot_pipeline_results.py \\
        --analysis_type k_varying \\
        --input_dir results/k_varying \\
        --output_dir plots/k_varying_mse \\
        --metric_filter mse
"""
from __future__ import annotations


import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

_MODULE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(_MODULE_DIR))

# Import plotting functions from centralized modules
from _k_varying_plots import (
    plot_aggregate_metrics,
    plot_baseline_comparison,
    plot_gene_stability,
    plot_per_celltype_grid,
    plot_reconstruction_metrics_grid,
    plot_reconstruction_quality,
)
from _stability_plots import (
    plot_aggregate_metrics_summary,
    plot_gene_frequency,
    plot_gene_overlap,
    plot_metrics_summary,
)
from _selection_plots import plot_gene_source_distribution
from plot_evaluation import create_baseline_plots
from _constants import ANALYSIS_PNG_DPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def normalize_metric_filter(metric_filter: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize metric_filter values to a validated list.

    Parameters
    ----------
    metric_filter : list, optional
        Candidate filter values.

    Returns
    -------
    list or None
        Normalized list containing values from {'mse', 'expvar'}, or None.
    """
    if not metric_filter:
        return None

    normalized = sorted({m.lower().strip() for m in metric_filter if m and m.lower().strip() in {"mse", "expvar"}})
    return normalized or None


def parse_color_scheme(color_string: str) -> Dict[str, str]:
    """Parse color scheme from command line argument.

    Parameters
    ----------
    color_string : str
        Color specification as 'key1:value1,key2:value2'

    Returns
    -------
    dict
        Dictionary mapping keys to color hex codes

    Examples
    --------
    >>> parse_color_scheme('nmf:#1f77b4,cnmf:#ff7f0e')
    {'nmf': '#1f77b4', 'cnmf': '#ff7f0e'}
    """
    colors = {}
    for pair in color_string.split(','):
        if ':' in pair:
            key, value = pair.split(':', 1)
            colors[key.strip()] = value.strip()
    return colors


def parse_font_sizes(font_string: str) -> Dict[str, int]:
    """Parse font sizes from command line argument.

    Parameters
    ----------
    font_string : str
        Font size specification as 'key1:value1,key2:value2'

    Returns
    -------
    dict
        Dictionary mapping font type to size

    Examples
    --------
    >>> parse_font_sizes('title:20,label:18,tick:16')
    {'title': 20, 'label': 18, 'tick': 16}
    """
    fonts = {}
    for pair in font_string.split(','):
        if ':' in pair:
            key, value = pair.split(':', 1)
            try:
                fonts[key.strip()] = int(value.strip())
            except ValueError:
                logger.warning(f"Invalid font size '{value}' for '{key}' - skipping")
    return fonts


def load_k_varying_data(input_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """Load K-varying analysis CSV files.

    Parameters
    ----------
    input_dir : Path
        Directory containing K-varying result CSVs

    Returns
    -------
    dict
        Dictionary with keys: 'reconstruction', 'global', 'overlap'
    """
    logger.info("Loading K-varying data files...")

    data = {}

    # Load reconstruction quality (main results)
    recon_file = input_dir / "reconstruction_quality.csv"
    if recon_file.exists():
        data['reconstruction'] = pd.read_csv(recon_file)
        logger.info(f"Loaded {recon_file.name}: {len(data['reconstruction'])} rows")
    else:
        logger.warning(f"File not found: {recon_file}")
        data['reconstruction'] = None

    # Load global reconstruction quality (with baselines)
    global_file = input_dir / "global_reconstruction_quality.csv"
    if global_file.exists():
        data['global'] = pd.read_csv(global_file)
        logger.info(f"Loaded {global_file.name}: {len(data['global'])} rows")
    else:
        logger.warning(f"File not found: {global_file}")
        data['global'] = None

    # Load gene overlap summary
    overlap_file = input_dir / "gene_overlap_summary.csv"
    if overlap_file.exists():
        data['overlap'] = pd.read_csv(overlap_file)
        logger.info(f"Loaded {overlap_file.name}: {len(data['overlap'])} rows")
    else:
        logger.warning(f"File not found: {overlap_file}")
        data['overlap'] = None

    return data


def load_k_varying_per_celltype_data(
    detailed_results_dir: Path,
    k_values: List[int],
    n_iterations: int,
) -> pd.DataFrame:
    """Load per-celltype K-varying data from detailed per-iteration CSV outputs.

    Parameters
    ----------
    detailed_results_dir : Path
        Path to detailed_results directory.
    k_values : list
        K values to load.
    n_iterations : int
        Number of iterations to scan per K.

    Returns
    -------
    pd.DataFrame
        Combined per-celltype rows.
    """
    all_data: List[pd.DataFrame] = []

    for k in k_values:
        k_dir = detailed_results_dir / f"k_{int(k):02d}"
        if not k_dir.exists():
            continue

        for iter_idx in range(n_iterations):
            iter_dir = k_dir / f"iter_{iter_idx}"
            if not iter_dir.exists():
                continue

            for sel_method, eval_method in [("nmf", "nmf"), ("cnmf", "cnmf")]:
                csv_file = iter_dir / f"{sel_method}_sel_{eval_method}_eval_per_celltype.csv"
                if not csv_file.exists():
                    continue

                try:
                    all_data.append(pd.read_csv(csv_file))
                except Exception as exc:  # pragma: no cover - defensive read path
                    logger.warning(f"Skipping unreadable per-celltype file {csv_file}: {exc}")

    if not all_data:
        raise ValueError("No per-celltype detailed_results CSV files found")

    return pd.concat(all_data, ignore_index=True)


def filter_valid_celltypes(df: pd.DataFrame, min_cells: int = 20) -> List[str]:
    """Filter celltypes with enough non-skipped observations and order by sample size."""
    if "skipped" in df.columns:
        valid_df = df[(~df["skipped"]) & (df.get("n_cells", 0) >= min_cells)].copy()
    else:
        valid_df = df[df.get("n_cells", 0) >= min_cells].copy()

    if valid_df.empty:
        raise ValueError(f"No celltypes with >= {min_cells} cells found")

    celltype_stats = valid_df.groupby("celltype").agg({"n_cells": "mean"}).reset_index()
    celltype_stats = celltype_stats.sort_values("n_cells", ascending=False)
    return celltype_stats["celltype"].tolist()


def aggregate_per_celltype(
    df: pd.DataFrame,
    celltypes: List[str],
) -> Dict[str, pd.DataFrame]:
    """Aggregate per-celltype metrics by K and method (mean ± std)."""
    aggregated: Dict[str, pd.DataFrame] = {}

    for celltype in celltypes:
        ct_df = df[df["celltype"] == celltype].copy()
        ct_df["method"] = ct_df["selection_method"] + "-" + ct_df["eval_method"]
        ct_df = ct_df[ct_df["method"].isin(["nmf-nmf", "cnmf-cnmf"])]

        agg_df = ct_df.groupby(["k_value", "method"]).agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            expvar_mean=("expvar", "mean"),
            expvar_std=("expvar", "std"),
            n_cells=("n_cells", "first"),
        ).reset_index()
        aggregated[celltype] = agg_df

    return aggregated


def load_baseline_triplet_data(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load baseline evaluation CSV triplet from a directory or its baseline_results subdir."""
    base_dir = input_dir / "baseline_results" if (input_dir / "baseline_results").exists() else input_dir

    def _find_first(prefixes: List[str]) -> Optional[Path]:
        for file in sorted(base_dir.glob("*.csv")):
            if any(file.name.startswith(prefix) for prefix in prefixes):
                return file
        return None

    clustering_file = _find_first(["clustering_quality", "clustering_results"])
    neighborhood_file = _find_first(["neighborhood_preservation", "neighborhood_results"])
    celltype_file = _find_first(["celltype_classification", "celltype_results"])

    if not clustering_file and not neighborhood_file and not celltype_file:
        raise FileNotFoundError(f"No baseline evaluation CSV triplet found in {base_dir}")

    results: Dict[str, pd.DataFrame] = {
        "clustering": pd.DataFrame(),
        "neighborhood": pd.DataFrame(),
        "celltype": pd.DataFrame(),
    }

    if clustering_file:
        cdf = pd.read_csv(clustering_file)
        if "dataset" not in cdf.columns and "dataset_name" in cdf.columns:
            cdf["dataset"] = cdf["dataset_name"]
        if "dataset_name" not in cdf.columns and "dataset" in cdf.columns:
            cdf["dataset_name"] = cdf["dataset"]
        results["clustering"] = cdf

    if neighborhood_file:
        ndf = pd.read_csv(neighborhood_file)
        if "dataset" not in ndf.columns and "dataset_name" in ndf.columns:
            ndf = ndf.rename(columns={"dataset_name": "dataset"})
        results["neighborhood"] = ndf

    if celltype_file:
        tdf = pd.read_csv(celltype_file)
        if "dataset" not in tdf.columns and "dataset_name" in tdf.columns:
            tdf = tdf.rename(columns={"dataset_name": "dataset"})
        results["celltype"] = tdf

    return results


def plot_baseline_evaluation_results(
    input_dir: Path,
    output_dir: Path,
    dpi: int = ANALYSIS_PNG_DPI,
) -> None:
    """Generate baseline evaluation plots from a detected baseline-results signature."""
    logger.info("=" * 80)
    logger.info(f"GENERATING BASELINE EVALUATION PLOTS: {input_dir}")
    logger.info("=" * 80)

    results = load_baseline_triplet_data(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_name = input_dir.name
    create_baseline_plots(
        results=results,
        output_dir=str(output_dir),
        group_name=group_name,
        color_map=None,
        png_dpi=dpi,
        plot_clustering=True,
        plot_neighborhood=True,
        plot_celltype=True,
        use_hardcoded_colors=False,
        external_names=None,
    )


def plot_selection_results(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Generate selection plots from ranked gene list or provenance files."""
    logger.info("=" * 80)
    logger.info(f"GENERATING SELECTION PLOTS: {input_dir}")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = [
        input_dir / "selection" / "ranked_gene_list.csv",
        input_dir / "ranked_gene_list.csv",
        input_dir / "selection" / "final_panel_with_provenance.csv",
        input_dir / "final_panel_with_provenance.csv",
    ]
    input_file = next((p for p in candidate_files if p.exists()), None)

    if input_file is None:
        raise FileNotFoundError(f"No selection summary CSV found in {input_dir}")

    summary_df = pd.read_csv(input_file)
    if summary_df.empty:
        logger.warning(f"Selection summary is empty: {input_file}")
        return

    if "gene_source" not in summary_df.columns:
        if "selection_strategy" in summary_df.columns:
            summary_df["gene_source"] = summary_df["selection_strategy"].fillna("Unknown")
        elif "celltype" in summary_df.columns:
            summary_df["gene_source"] = "celltype:" + summary_df["celltype"].fillna("Unknown").astype(str)
        else:
            summary_df["gene_source"] = "Other"

    result = plot_gene_source_distribution(
        summary_df=summary_df,
        results_dir=str(output_dir),
        reduction_type="NMF",
        strategy_name=input_dir.name,
    )
    if result is None:
        logger.warning("Selection plot generation returned no output (input may be missing required columns)")


def detect_directory_signatures(input_dir: Path) -> Set[str]:
    """Detect supported analysis signatures for a directory."""
    signatures: Set[str] = set()

    k_triplet = [
        input_dir / "reconstruction_quality.csv",
        input_dir / "global_reconstruction_quality.csv",
        input_dir / "gene_overlap_summary.csv",
    ]
    if all(path.exists() for path in k_triplet):
        signatures.add("k_varying")

    baseline_dir = input_dir / "baseline_results" if (input_dir / "baseline_results").exists() else input_dir
    csv_names = {p.name for p in baseline_dir.glob("*.csv")}
    has_clustering = any(name.startswith("clustering_quality") or name == "clustering_results.csv" for name in csv_names)
    has_neighborhood = any(name.startswith("neighborhood_preservation") or name == "neighborhood_results.csv" for name in csv_names)
    has_celltype = any(name.startswith("celltype_classification") or name == "celltype_results.csv" for name in csv_names)
    if has_clustering and has_neighborhood and has_celltype:
        signatures.add("evaluation")

    selection_candidates = [
        input_dir / "selection" / "ranked_gene_list.csv",
        input_dir / "ranked_gene_list.csv",
        input_dir / "selection" / "final_panel_with_provenance.csv",
        input_dir / "final_panel_with_provenance.csv",
    ]
    if any(path.exists() for path in selection_candidates):
        signatures.add("selection")

    return signatures


def collect_auto_targets(root_input_dir: Path) -> List[Path]:
    """Collect candidate directories for auto-detection.

    Includes root and immediate subdirectories to support mixed layouts
    such as raw/log branches.
    """
    targets: List[Path] = [root_input_dir]
    for child in sorted(root_input_dir.iterdir()):
        if child.is_dir():
            # Avoid double-processing baseline_results when parent directory already
            # represents that signature.
            if child.name in {"baseline_results", "selection"}:
                continue
            targets.append(child)
    return targets


def load_stability_data(input_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """Load stability analysis CSV files.

    Parameters
    ----------
    input_dir : Path
        Directory containing stability result CSVs

    Returns
    -------
    dict
        Dictionary with keys: 'genes', 'metrics'
    """
    logger.info("Loading stability data files...")

    data = {}

    # Load gene stability
    gene_file = input_dir / "gene_stability.csv"
    if gene_file.exists():
        data['genes'] = pd.read_csv(gene_file)
        logger.info(f"Loaded {gene_file.name}: {len(data['genes'])} rows")
    else:
        logger.warning(f"File not found: {gene_file}")
        data['genes'] = None

    # Load metrics stability
    metrics_file = input_dir / "metrics_stability.csv"
    if metrics_file.exists():
        data['metrics'] = pd.read_csv(metrics_file)
        logger.info(f"Loaded {metrics_file.name}: {len(data['metrics'])} rows")
    else:
        logger.warning(f"File not found: {metrics_file}")
        data['metrics'] = None

    return data


def plot_k_varying_results(
    input_dir: Path,
    output_dir: Path,
    plot_types: Optional[List[str]] = None,
    dpi: int = ANALYSIS_PNG_DPI,
    fig_format: str = "png",
    colors: Optional[Dict[str, str]] = None,
    metric_filter: Optional[List[str]] = None,
) -> None:
    """Generate all K-varying analysis plots.

    Parameters
    ----------
    input_dir : Path
        Directory containing K-varying result CSVs
    output_dir : Path
        Directory to save plots
    plot_types : list, optional
        List of plot types to generate. Default: all
    dpi : int
        Plot resolution
    fig_format : str
        Output format ('png', 'pdf', 'svg')
    colors : dict, optional
        Custom color scheme
    metric_filter : list, optional
        Only plot specific metrics ('mse', 'expvar')
    """
    logger.info("GENERATING K-VARYING ANALYSIS PLOTS")
    logger.info("=" * 80)

    metric_filter = normalize_metric_filter(metric_filter)

    # Load data
    data = load_k_varying_data(input_dir)

    if plot_types is None:
        plot_types = ['reconstruction', 'stability', 'metrics', 'baseline', 'grid', 'per_celltype']

    # Determine metric types to plot
    metric_types = ['global', 'macro', 'weighted']

    # Generate reconstruction quality plots
    if 'reconstruction' in plot_types and data['reconstruction'] is not None:
        for metric_type in metric_types:
            plot_reconstruction_quality(
                data['reconstruction'],
                output_dir,
                metric_type=metric_type,
                png_dpi=dpi,
                fig_format=fig_format,
                colors=colors,
                metric_filter=metric_filter,
            )

    # Generate gene stability plots
    if 'stability' in plot_types and data['overlap'] is not None:
        plot_gene_stability(
            data['overlap'],
            output_dir,
            png_dpi=dpi,
            fig_format=fig_format,
            colors=colors,
        )

    # Generate aggregate metrics comparison
    if 'metrics' in plot_types and data['reconstruction'] is not None:
        plot_aggregate_metrics(
            data['reconstruction'],
            output_dir,
            png_dpi=dpi,
            fig_format=fig_format,
            colors=colors,
            metric_filter=metric_filter,
        )

    # Generate baseline comparison
    if 'baseline' in plot_types and data['global'] is not None:
        plot_baseline_comparison(
            data['global'],
            output_dir,
            png_dpi=dpi,
            fig_format=fig_format,
            colors=colors,
        )

    # Generate unified reconstruction metrics grid
    if 'grid' in plot_types and data['reconstruction'] is not None and data['global'] is not None:
        if metric_filter and set(metric_filter) != {"mse", "expvar"}:
            logger.info("Skipping unified grid plot because it requires both mse and expvar metrics")
        else:
            plot_reconstruction_metrics_grid(
                data['reconstruction'],
                data['global'],
                output_dir,
                png_dpi=dpi,
                fig_format=fig_format,
                colors=colors,
            )

    # Generate per-celltype grids from detailed_results
    if 'per_celltype' in plot_types:
        detailed_results_dir = input_dir / "detailed_results"
        if not detailed_results_dir.exists():
            logger.info(f"Skipping per-celltype plots: detailed_results not found at {detailed_results_dir}")
        else:
            try:
                if data['reconstruction'] is not None and 'k_value' in data['reconstruction'].columns:
                    k_values = sorted(int(k) for k in data['reconstruction']['k_value'].dropna().unique())
                else:
                    k_values = sorted(int(p.name.split("_")[1]) for p in detailed_results_dir.glob("k_*") if p.name.startswith("k_"))

                n_iterations = 5
                if data['reconstruction'] is not None and 'iteration' in data['reconstruction'].columns:
                    n_iterations = int(data['reconstruction']['iteration'].nunique())

                per_ct_df = load_k_varying_per_celltype_data(
                    detailed_results_dir=detailed_results_dir,
                    k_values=k_values,
                    n_iterations=n_iterations,
                )
                celltypes = filter_valid_celltypes(per_ct_df, min_cells=20)
                aggregated_data = aggregate_per_celltype(per_ct_df, celltypes)

                plots_dir = output_dir / "plots"
                if metric_filter is None or "mse" in metric_filter:
                    plot_per_celltype_grid(
                        aggregated_data=aggregated_data,
                        celltypes=celltypes,
                        k_values=k_values,
                        metric="mse",
                        output_file=plots_dir / f"per_celltype_mse_grid.{fig_format}",
                        png_dpi=dpi,
                        colors=colors,
                    )

                if metric_filter is None or "expvar" in metric_filter:
                    plot_per_celltype_grid(
                        aggregated_data=aggregated_data,
                        celltypes=celltypes,
                        k_values=k_values,
                        metric="expvar",
                        output_file=plots_dir / f"per_celltype_expvar_grid.{fig_format}",
                        png_dpi=dpi,
                        colors=colors,
                    )
            except Exception as exc:
                logger.warning(f"Failed to generate per-celltype plots: {exc}")

    logger.info("K-VARYING PLOTS COMPLETE")
    logger.info(f"Output directory: {output_dir / 'plots'}")
    logger.info("=" * 80)


def run_auto_detection_plotting(
    input_dir: Path,
    output_dir: Path,
    plot_types: Optional[List[str]] = None,
    dpi: int = ANALYSIS_PNG_DPI,
    fig_format: str = "png",
    colors: Optional[Dict[str, str]] = None,
    metric_filter: Optional[List[str]] = None,
) -> None:
    """Auto-detect supported analysis signatures and run all compatible plotters."""
    logger.info("=" * 80)
    logger.info("AUTO-DETECTION MODE")
    logger.info("=" * 80)

    targets = collect_auto_targets(input_dir)
    any_detected = False

    for target in targets:
        signatures = detect_directory_signatures(target)
        if not signatures:
            continue

        any_detected = True
        rel_path = target.relative_to(input_dir)
        target_output = output_dir if str(rel_path) == "." else output_dir / rel_path
        target_output.mkdir(parents=True, exist_ok=True)

        logger.info(f"Detected {sorted(signatures)} in {target}")

        if "k_varying" in signatures:
            plot_k_varying_results(
                target,
                target_output,
                plot_types=plot_types,
                dpi=dpi,
                fig_format=fig_format,
                colors=colors,
                metric_filter=metric_filter,
            )

        if "evaluation" in signatures:
            plot_baseline_evaluation_results(
                target,
                target_output / "evaluation",
                dpi=dpi,
            )

        if "selection" in signatures:
            plot_selection_results(
                target,
                target_output / "selection",
            )

    if not any_detected:
        raise ValueError(
            "No supported analysis signatures detected. "
            "Expected K-varying triplet files or baseline evaluation CSV triplets."
        )
    logger.info("AUTO-DETECTION COMPLETE")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)


def plot_stability_results(
    input_dir: Path,
    output_dir: Path,
    plot_types: Optional[List[str]] = None,
    dpi: int = ANALYSIS_PNG_DPI,
    fig_format: str = "png",
) -> None:
    """Generate all stability analysis plots.

    Parameters
    ----------
    input_dir : Path
        Directory containing stability result CSVs
    output_dir : Path
        Directory to save plots
    plot_types : list, optional
        List of plot types to generate. Default: all
    dpi : int
        Plot resolution
    fig_format : str
        Output format ('png', 'pdf', 'svg')
    """
    logger.info("=" * 80)
    logger.info("GENERATING STABILITY ANALYSIS PLOTS")
    logger.info("=" * 80)

    # Load data
    data = load_stability_data(input_dir)

    if plot_types is None:
        plot_types = ['frequency', 'overlap', 'metrics', 'aggregate']

    # Determine number of iterations
    if data['genes'] is not None and 'n_selected' in data['genes'].columns:
        n_iterations = int(data['genes']['n_selected'].max())
    else:
        n_iterations = 5  # Default

    # Generate gene frequency plot
    if 'frequency' in plot_types and data['genes'] is not None:
        plot_gene_frequency(
            data['genes'],
            output_dir,
            n_iterations=n_iterations,
            png_dpi=dpi,
        )

    # Generate gene overlap plot
    if 'overlap' in plot_types and data['genes'] is not None:
        plot_gene_overlap(
            data['genes'],
            output_dir,
            n_iterations=n_iterations,
            png_dpi=dpi,
        )

    # Generate metrics summary plots
    if 'metrics' in plot_types and data['metrics'] is not None:
        plot_metrics_summary(
            data['metrics'],
            output_dir,
            png_dpi=dpi,
        )

    # Generate aggregate metrics summary
    if 'aggregate' in plot_types and data['metrics'] is not None:
        plot_aggregate_metrics_summary(
            data['metrics'],
            output_dir,
            png_dpi=dpi,
        )

    logger.info("=" * 80)
    logger.info("STABILITY PLOTS COMPLETE")
    logger.info(f"Output directory: {output_dir / 'plots'}")
    logger.info("=" * 80)


def main():
    """Main CLI entry point."""
    analysis_choices = ["auto", "k_varying", "stability", "evaluation", "selection"]

    parser = argparse.ArgumentParser(
        description="Comprehensive plotting for pipeline analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Backward-compatible positional arguments
    parser.add_argument(
        "analysis_type_pos",
        nargs="?",
        choices=analysis_choices,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "input_dir_pos",
        nargs="?",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "output_dir_pos",
        nargs="?",
        type=Path,
        help=argparse.SUPPRESS,
    )

    # Required arguments
    parser.add_argument(
        "--analysis_type",
        type=str,
        choices=analysis_choices,
        help="Type of analysis results to plot. Use 'auto' to detect supported layouts.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Directory containing analysis result files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save output plots",
    )

    # Optional arguments
    parser.add_argument(
        "--plot_types",
        type=str,
        help="Comma-separated list of plot types to generate (default: all)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=ANALYSIS_PNG_DPI,
        help=f"Plot resolution (default: {ANALYSIS_PNG_DPI})",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--color_scheme",
        type=str,
        help="Custom colors as 'key1:color1,key2:color2' (e.g., 'nmf:#1f77b4,cnmf:#ff7f0e')",
    )
    parser.add_argument(
        "--font_sizes",
        type=str,
        help="Custom font sizes as 'key1:size1,key2:size2' (e.g., 'title:20,label:18,tick:16')",
    )
    parser.add_argument(
        "--metric_filter",
        type=str,
        help="Only plot specific metrics: 'mse', 'expvar', or 'mse,expvar'",
    )

    args = parser.parse_args()

    analysis_type = args.analysis_type or args.analysis_type_pos
    input_dir = args.input_dir or args.input_dir_pos
    output_dir = args.output_dir or args.output_dir_pos

    if analysis_type is None or input_dir is None or output_dir is None:
        parser.error(
            "the following arguments are required: --analysis_type, --input_dir, --output_dir "
            "(or positional: analysis_type input_dir output_dir)"
        )

    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse optional arguments
    plot_types = args.plot_types.split(',') if args.plot_types else None
    colors = parse_color_scheme(args.color_scheme) if args.color_scheme else None
    font_sizes = parse_font_sizes(args.font_sizes) if args.font_sizes else None
    metric_filter = args.metric_filter.split(',') if args.metric_filter else None

    # Log configuration
    logger.info(f"Analysis type: {analysis_type}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"DPI: {args.dpi}")
    logger.info(f"Format: {args.format}")
    if plot_types:
        logger.info(f"Plot types: {plot_types}")
    if colors:
        logger.info(f"Custom colors: {colors}")
    if metric_filter:
        logger.info(f"Metric filter: {metric_filter}")

    # Route to appropriate plotting function
    try:
        if analysis_type == "k_varying":
            plot_k_varying_results(
                input_dir,
                output_dir,
                plot_types=plot_types,
                dpi=args.dpi,
                fig_format=args.format,
                colors=colors,
                metric_filter=metric_filter,
            )
        elif analysis_type == "auto":
            run_auto_detection_plotting(
                input_dir,
                output_dir,
                plot_types=plot_types,
                dpi=args.dpi,
                fig_format=args.format,
                colors=colors,
                metric_filter=metric_filter,
            )
        elif analysis_type == "stability":
            plot_stability_results(
                input_dir,
                output_dir,
                plot_types=plot_types,
                dpi=args.dpi,
                fig_format=args.format,
            )
        elif analysis_type == "evaluation":
            plot_baseline_evaluation_results(
                input_dir,
                output_dir,
                dpi=args.dpi,
            )
        elif analysis_type == "selection":
            plot_selection_results(
                input_dir,
                output_dir,
            )

        logger.info("All plots generated successfully!")

    except Exception as e:
        logger.error(f"Error generating plots: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
