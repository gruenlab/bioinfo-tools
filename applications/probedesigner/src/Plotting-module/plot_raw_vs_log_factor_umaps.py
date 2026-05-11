#!/usr/bin/env python3
"""Create raw-vs-log NMF factor UMAP comparison grids.

For each probeset size, this script loads
``adata_with_W_matrix_{size}.h5ad`` and creates a grid plot with:
- columns: NMF factors
- rows: raw-count factors (top) and log-normalized factors (bottom)

Color scales are shared per factor column across the two rows to support
direct visual comparison between raw and log variants.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List

import scanpy as sc

_MODULE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(_MODULE_DIR))

from _comparison_umaps import plot_celltype_umap, plot_raw_vs_log_factor_umap_grid
from _constants import ANALYSIS_PNG_DPI

logger = logging.getLogger(__name__)


def _parse_probeset_sizes(raw_sizes: Iterable[str]) -> List[int]:
    """Parse probeset sizes from CLI values.

    Supports values as separate args (``100 200 500``) or comma-separated
    groups (``100,200,500``).
    """
    sizes: List[int] = []
    for token in raw_sizes:
        for piece in token.split(","):
            cleaned = piece.strip()
            if not cleaned:
                continue
            try:
                sizes.append(int(cleaned))
            except ValueError as exc:
                raise ValueError(f"Invalid probeset size '{cleaned}'.") from exc

    if not sizes:
        raise ValueError("No probeset sizes provided.")

    return sorted(set(sizes))


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot raw-vs-log NMF factor UMAP grids for one or more probeset sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("Experiments/LCA/Dimred-NMF-Raw-vs-Log"),
        help="Root directory containing {size}-genes subfolders.",
    )
    parser.add_argument(
        "--probeset_sizes",
        nargs="+",
        default=["100", "200", "500"],
        help="Probeset sizes as space-separated and/or comma-separated values.",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="n_neighbors used for UMAP computation when X_umap is absent.",
    )
    parser.add_argument(
        "--n_pcs",
        type=int,
        default=30,
        help="n_pcs used for neighbor graph when computing UMAP.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for UMAP computation.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=6.0,
        help="Point size for UMAP scatter plots.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Colormap used for factor values.",
    )
    parser.add_argument(
        "--value_transform",
        type=str,
        default="log1p",
        choices=["none", "log1p"],
        help="Transform used before color mapping.",
    )
    parser.add_argument(
        "--lower_percentile",
        type=float,
        default=2.0,
        help="Lower percentile for robust color clipping.",
    )
    parser.add_argument(
        "--upper_percentile",
        type=float,
        default=98.0,
        help="Upper percentile for robust color clipping.",
    )
    parser.add_argument(
        "--celltype_col",
        type=str,
        default=None,
        help="Celltype column name for annotated UMAP. If omitted, a common name is inferred.",
    )
    parser.add_argument(
        "--skip_celltype_umap",
        action="store_true",
        help="Skip creating the separate celltype-annotated UMAP plot.",
    )
    parser.add_argument(
        "--celltype_umap_path",
        type=Path,
        default=None,
        help="Optional output path for the separate celltype UMAP plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=ANALYSIS_PNG_DPI,
        help="Output DPI.",
    )
    parser.add_argument(
        "--figure_format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output figure format.",
    )

    return parser.parse_args()


def main() -> None:
    """Run plotting workflow for all requested probeset sizes."""
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    input_root = args.input_root.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    probeset_sizes = _parse_probeset_sizes(args.probeset_sizes)
    logger.info("Input root: %s", input_root)
    logger.info("Probeset sizes: %s", probeset_sizes)

    saved_paths: List[Path] = []
    celltype_umap_done = False

    for size in probeset_sizes:
        size_dir = input_root / f"{size}-genes"
        adata_path = size_dir / f"adata_with_W_matrix_{size}.h5ad"
        output_dir = size_dir / "plots"

        if not adata_path.exists():
            logger.warning("Skipping size %d: missing file %s", size, adata_path)
            continue

        logger.info("Processing probeset size %d", size)
        adata = sc.read_h5ad(adata_path)

        saved_plot = plot_raw_vs_log_factor_umap_grid(
            adata=adata,
            probeset_size=size,
            output_dir=output_dir,
            n_neighbors=args.n_neighbors,
            n_pcs=args.n_pcs,
            random_state=args.random_state,
            point_size=args.point_size,
            cmap=args.cmap,
            value_transform=args.value_transform,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
            png_dpi=args.dpi,
            figure_format=args.figure_format,
        )
        saved_paths.append(saved_plot)

        if not args.skip_celltype_umap and not celltype_umap_done:
            if args.celltype_umap_path is None:
                ext = args.figure_format.lower().lstrip(".")
                celltype_out = input_root / f"umap_celltypes.{ext}"
            else:
                celltype_out = args.celltype_umap_path.expanduser().resolve()

            saved_celltype = plot_celltype_umap(
                adata=adata,
                output_path=celltype_out,
                celltype_col=args.celltype_col,
                n_neighbors=args.n_neighbors,
                n_pcs=args.n_pcs,
                random_state=args.random_state,
                png_dpi=args.dpi,
            )
            saved_paths.append(saved_celltype)
            celltype_umap_done = True

    if not saved_paths:
        logger.warning("No plots were generated.")
    else:
        logger.info("Generated %d plot(s):", len(saved_paths))
        for plot_path in saved_paths:
            logger.info("  %s", plot_path)


if __name__ == "__main__":
    main()
