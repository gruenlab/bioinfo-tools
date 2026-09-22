"""Materialize a gene-blacklist-filtered copy of a raw h5ad dataset.

Removes genes matching the standard selection-pipeline blacklist (default
mt-/hsp patterns, plus any extra patterns such as rps/rpl) directly from the
raw AnnData, so downstream selection methods that have no blacklist hook of
their own still operate on a gene set with those families fully absent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad

_SELECTION_MODULE_DIR = Path(__file__).parent.parent / "Selection-module"
sys.path.insert(0, str(_SELECTION_MODULE_DIR))

from _filtering import apply_blacklist_filter  # noqa: E402

logger = logging.getLogger(__name__)


def filter_blacklist_genes(
    input_file: str,
    output_file: str,
    blacklist_patterns: list[str] | None = None,
    use_default_blacklist: bool = True,
) -> None:
    logger.info(f"Loading raw data: {input_file}")
    adata = ad.read_h5ad(input_file)
    n_genes_before = adata.n_vars

    filtered_genes, removed_genes, _ = apply_blacklist_filter(
        gene_list=adata.var_names.tolist(),
        blacklist_patterns=blacklist_patterns,
        use_default_blacklist=use_default_blacklist,
    )

    adata = adata[:, filtered_genes].copy()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_file)

    combined_patterns = (["mt-", "hsp", "rps", "rpl"] if use_default_blacklist else []) + (blacklist_patterns or [])
    logger.info(
        f"Blacklist patterns: {combined_patterns} | "
        f"{n_genes_before} genes -> {adata.n_vars} genes "
        f"({len(removed_genes)} removed)"
    )

    params = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "blacklist_patterns": combined_patterns,
        "use_default_blacklist": use_default_blacklist,
        "n_genes_before": n_genes_before,
        "n_genes_after": adata.n_vars,
        "n_genes_removed": len(removed_genes),
        "removed_genes_example": sorted(removed_genes)[:20],
        "timestamp": datetime.now().isoformat(),
    }
    params_file = Path(output_file).with_suffix("").with_name(
        Path(output_file).stem + "_filter_parameters.json"
    )
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Parameters written to: {params_file}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--blacklist_patterns", nargs="*", default=None)
    parser.add_argument("--disable_default_blacklist", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    filter_blacklist_genes(
        input_file=args.input_file,
        output_file=args.output_file,
        blacklist_patterns=args.blacklist_patterns,
        use_default_blacklist=not args.disable_default_blacklist,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
