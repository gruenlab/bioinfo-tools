#!/usr/bin/env python
"""
CLI wrapper for Tangram reconstruction check.

Loads preprocessed H5AD files from a completed pipeline run and calls
run_tangram_reconstruction_check() for each panel, then writes a JSON summary.

Exit codes:
    0  - success
    1  - runtime error
    2  - tangram-sc not installed

Usage:
    python run_tangram_check.py \\
        --preprocessed_dir <output_dir>/preprocessing/ \\
        --output_dir <output_dir>/evaluation/tangram/ \\
        [--celltype_col cluster] \\
        [--n_epochs 1000] \\
        [--mode both]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure this module directory is importable
sys.path.insert(0, str(Path(__file__).parent.absolute()))


def main() -> int:
    # Check package availability before parsing args so the error is immediate
    try:
        import tangram  # noqa: F401
    except ImportError:
        print("ERROR: tangram-sc is not installed.", file=sys.stderr)
        print("Install it with: pip install tangram-sc", file=sys.stderr)
        return 2

    from _tangram import run_tangram_reconstruction_check, _TANGRAM_AVAILABLE
    if not _TANGRAM_AVAILABLE:
        print("ERROR: tangram-sc import succeeded but _TANGRAM_AVAILABLE is False.", file=sys.stderr)
        return 2

    import scanpy as sc

    parser = argparse.ArgumentParser(
        description="Tangram reconstruction check on preprocessed H5AD files"
    )
    parser.add_argument(
        "--preprocessed_dir", required=True,
        help="Directory containing full_transcriptome.h5ad and per-panel .h5ad files",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where Tangram results will be written",
    )
    parser.add_argument(
        "--celltype_col", default="cluster",
        help="obs column holding cell-type labels (default: cluster)",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1000,
        help="Tangram optimisation epochs (default: 1000)",
    )
    parser.add_argument(
        "--mode", default="both",
        choices=["global", "per_celltype", "both"],
        help="Reconstruction mode (default: both)",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.3,
        help="Fraction of cells held out for testing (default: 0.3)",
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for train/test split (default: 42)",
    )
    parser.add_argument(
        "--no_split", action="store_true",
        help="Disable train/test split and use the full dataset (original behaviour)",
    )
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full-transcriptome reference
    full_h5ad = preprocessed_dir / "full_transcriptome.h5ad"
    if not full_h5ad.exists():
        print(f"ERROR: full_transcriptome.h5ad not found in {preprocessed_dir}", file=sys.stderr)
        return 1

    print(f"Loading full transcriptome from {full_h5ad}...")
    adata_full = sc.read_h5ad(full_h5ad)

    # Generate train/test splits once so they can be reused across all panels
    # and shared with NMF-based evaluation (same cells guaranteed).
    if not args.no_split:
        from _splits import generate_evaluation_splits
        splits = generate_evaluation_splits(
            adata_full,
            celltype_col=args.celltype_col,
            split_mode="simple",
            test_size=args.test_size,
            random_state=args.random_state,
        )
        train_idx, test_idx, per_celltype_splits = splits[0]
        print(
            f"Train/test split: {len(train_idx)} training cells, "
            f"{len(test_idx)} test cells "
            f"(test_size={args.test_size}, random_state={args.random_state})"
        )
    else:
        train_idx = test_idx = per_celltype_splits = None
        print("Train/test split disabled – using full dataset.")

    # Collect panel H5AD files (every .h5ad that is NOT full_transcriptome.h5ad)
    panel_files = sorted(
        p for p in preprocessed_dir.glob("*.h5ad")
        if p.name != "full_transcriptome.h5ad"
    )
    if not panel_files:
        print(f"ERROR: No panel .h5ad files found in {preprocessed_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(panel_files)} panel(s): {[p.stem for p in panel_files]}")

    run_global = args.mode in ("global", "both")
    run_per_celltype = args.mode in ("per_celltype", "both")

    summary: dict = {}
    for panel_file in panel_files:
        panel_name = panel_file.stem
        print(f"PANEL: {panel_name}")
        print(f"  Loading {panel_file}...")
        adata_panel = sc.read_h5ad(panel_file)

        panel_output_dir = output_dir / panel_name
        panel_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = run_tangram_reconstruction_check(
                adata_full=adata_full,
                adata_subset=adata_panel,
                output_dir=panel_output_dir,
                dataset_name=panel_name,
                celltype_col=args.celltype_col,
                num_epochs=args.n_epochs,
                run_global=run_global,
                run_per_celltype=run_per_celltype,
                train_idx=train_idx,
                test_idx=test_idx,
                per_celltype_splits=per_celltype_splits,
            )
            summary[panel_name] = result
            if "global" in result and not result.get("skipped"):
                gm = result["global"]
                if not gm.get("skipped"):
                    print(
                        f"  ✓ Global  — MSE={gm['mse']:.4f}  ExpVar={gm['expvar']:.4f}"
                    )
            if "per_celltype" in result and not result.get("skipped"):
                n_ct = len(result["per_celltype"])
                n_ok = sum(
                    1 for m in result["per_celltype"].values() if not m.get("skipped")
                )
                print(f"  ✓ Per-celltype — {n_ok}/{n_ct} cell types reconstructed")
        except Exception as exc:
            import traceback
            print(f"  ERROR processing {panel_name}: {exc}", file=sys.stderr)
            traceback.print_exc()
            summary[panel_name] = {"skipped": True, "skip_reason": str(exc)}

    # Write JSON summary consumed by App
    summary_path = output_dir / "tangram_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n✓ Tangram check complete. Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())