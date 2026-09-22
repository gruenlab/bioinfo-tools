#!/usr/bin/env python3
"""CLI: side-by-side reconstruction comparison of NMF / PCA / ICA / Ridge / Tangram.

Reads each method's aggregated global results for every panel and writes

  * ``reconstruction_method_comparison.png`` — the figure,
  * ``reconstruction_method_comparison_values.csv`` — every panel x method x gene-scope
    x metric (MSE, ExpVar global-mean, ExpVar variance-weighted) as a long table,
  * ``plot_parameters.json`` — settings and input paths.

Example:
    python plot_method_comparison.py \\
        --evaluation_root Experiments/RecoVar/LCA/audit3_validation/evaluation \\
        --panels RecoVar_raw_rf0.5_nmf0.5_LCA_500genes_seed42,RecoVar_lognorm_rf0.5_nmf0.5_LCA_500genes_seed42,Spapros_LCA_500genes_seed42 \\
        --panel_labels "RecoVar (raw sel.),RecoVar (lognorm sel.),Spapros" \\
        --output_dir Experiments/RecoVar/LCA/audit3_validation/plots/method_comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _method_comparison_plots import (  # noqa: E402
    METHODS,
    load_method_comparison_table,
    plot_method_comparison,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evaluation_root", required=True,
                   help="Directory holding <panel><eval_suffix>/ evaluation folders.")
    p.add_argument("--panels", required=True, help="Comma-separated panel names (bar order).")
    p.add_argument("--panel_labels", default=None,
                   help="Comma-separated legend labels, same order as --panels (default: panel names).")
    p.add_argument("--eval_suffix", default="__eval_raw", help="Suffix of each panel's evaluation folder.")
    p.add_argument("--results_subpath", default="all_raw/Variability-Evaluation/results",
                   help="Path from <panel><eval_suffix>/ to the shared per-method results root.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--png_dpi", type=int, default=600)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    panels = [s.strip() for s in args.panels.split(",") if s.strip()]
    labels = [s.strip() for s in args.panel_labels.split(",")] if args.panel_labels else panels
    if len(labels) != len(panels):
        p.error("--panel_labels must have the same number of entries as --panels")

    roots = {
        panel: Path(args.evaluation_root) / f"{panel}{args.eval_suffix}" / args.results_subpath
        for panel in panels
    }
    table = load_method_comparison_table(roots, dict(zip(panels, labels)))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "reconstruction_method_comparison_values.csv", index=False)
    plot_method_comparison(table, out / "reconstruction_method_comparison.png", png_dpi=args.png_dpi)

    (out / "plot_parameters.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "args": vars(args),
        "panels": dict(zip(panels, labels)),
        "results_roots": {k: str(v) for k, v in roots.items()},
        "methods": [dict(zip(("label", "subdir", "space", "layout", "column_suffix"), m)) for m in METHODS],
        "outputs": ["reconstruction_method_comparison.png", "reconstruction_method_comparison_values.csv"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
