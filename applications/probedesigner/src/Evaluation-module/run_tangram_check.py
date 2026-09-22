#!/usr/bin/env python
"""
CLI wrapper for Tangram reconstruction check.

Loads preprocessed H5AD files from a completed pipeline run and calls
run_tangram_reconstruction_check() for each panel, then writes a JSON summary.

Exit codes:
    0  - success
    1  - runtime error
    2  - tangram-sc not installed

By default (unless ``--no_split``), the check loops over all ``--n_splits`` stratified
k-folds (matching the same ``_splits.generate_evaluation_splits()`` convention used by
NMF/PCA/ICA/Ridge), writing each fold's raw results to ``per_fold/`` and an aggregated
(mean ± std across folds) result to the standard ``global/``/``per_celltype/`` location.

Usage:
    python run_tangram_check.py \\
        --preprocessed_dir <output_dir>/preprocessing/ \\
        --output_dir <output_dir>/evaluation/tangram/ \\
        [--celltype_col cluster] \\
        [--n_epochs 1000] \\
        [--mode both] \\
        [--n_splits 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure this module directory is importable
sys.path.insert(0, str(Path(__file__).parent.absolute()))


# ---------------------------------------------------------------------------
# Fold aggregation
# ---------------------------------------------------------------------------

_GLOBAL_SKIP_KEYS = {"skipped", "skip_reason"}
_PER_CELLTYPE_SKIP_KEYS = {"skipped", "skip_reason"}


def _aggregate_and_save_tangram_folds(
    output_dir: Path,
    dataset_name: str,
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-fold Tangram results (mean ± std) and write to disk.

    Mirrors the ``_aggregate_fold_results`` pattern used by
    ``run_pca_evaluation.py``/``run_ica_evaluation.py``/``run_ridge_evaluation.py``:
    numeric columns get ``_mean``/``_std`` suffixes, aggregated across folds. Per-fold
    raw results are already written to ``output_dir/per_fold/`` by
    :func:`_tangram.run_tangram_reconstruction_check` (called with ``fold=``); this
    function only writes the aggregated view, to ``output_dir/global/{dataset_name}.csv``
    and ``output_dir/per_celltype/{dataset_name}.csv`` (the same paths a ``fold=None``
    call would use).

    Args:
        output_dir: Root Tangram output directory for this panel.
        dataset_name: Panel identifier (CSV filename stem).
        fold_results: One :func:`_tangram.run_tangram_reconstruction_check` return
            dict per fold, in fold order.

    Returns:
        Aggregated result dict with ``"global"`` and/or ``"per_celltype_summary"``
        keys (each holding ``{metric}_mean``/``{metric}_std`` values) plus
        ``"n_folds"``.
    """
    output_dir = Path(output_dir)
    aggregated: dict[str, Any] = {"n_folds": len(fold_results)}

    # ── Global metrics across folds ────────────────────────────────────────────
    global_rows = [
        {**{k: v for k, v in fr["global"].items() if k not in _GLOBAL_SKIP_KEYS}, "fold": i}
        for i, fr in enumerate(fold_results)
        if "global" in fr and not fr["global"].get("skipped", False)
    ]
    if global_rows:
        df_global = pd.DataFrame(global_rows)
        numeric_cols = [c for c in df_global.select_dtypes(include=[np.number]).columns if c != "fold"]
        agg = {c: [df_global[c].mean(), df_global[c].std()] for c in numeric_cols}
        agg_row = {"dataset": dataset_name, "mode": "global"}
        for c, (mean_v, std_v) in agg.items():
            agg_row[f"{c}_mean"] = mean_v
            agg_row[f"{c}_std"] = std_v
        global_dir = output_dir / "global"
        global_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([agg_row]).to_csv(global_dir / f"{dataset_name}.csv", index=False)
        aggregated["global"] = {k: v for k, v in agg_row.items() if k not in ("dataset", "mode")}

    # ── Per-celltype metrics across folds ───────────────────────────────────────
    ct_rows: list[dict[str, Any]] = []
    for i, fr in enumerate(fold_results):
        for ct, m in fr.get("per_celltype", {}).items():
            if m.get("skipped", False):
                continue
            ct_rows.append({
                "celltype": ct, "fold": i,
                **{k: v for k, v in m.items() if not k.startswith("_") and k not in _PER_CELLTYPE_SKIP_KEYS},
            })
        summary = fr.get("per_celltype_summary")
        if summary:
            ct_rows.append({"celltype": "__summary__", "fold": i, **summary})

    if ct_rows:
        df_ct = pd.DataFrame(ct_rows)
        numeric_cols = [c for c in df_ct.select_dtypes(include=[np.number]).columns if c != "fold"]
        agg_dict = {c: ["mean", "std"] for c in numeric_cols}
        df_agg = df_ct.groupby("celltype", dropna=False).agg(agg_dict).reset_index()
        df_agg.columns = [
            "_".join(col).rstrip("_") if col[1] else col[0]
            for col in df_agg.columns.values
        ]
        df_agg.insert(1, "dataset", dataset_name)
        df_agg.insert(2, "mode", "per_celltype")
        ct_dir = output_dir / "per_celltype"
        ct_dir.mkdir(parents=True, exist_ok=True)
        df_agg.to_csv(ct_dir / f"{dataset_name}.csv", index=False)

        summary_row = df_agg[df_agg["celltype"] == "__summary__"]
        if not summary_row.empty:
            aggregated["per_celltype_summary"] = {
                c: summary_row.iloc[0][c]
                for c in summary_row.columns
                if c not in ("celltype", "dataset", "mode")
            }

    return aggregated


def main() -> int:
    """Run the standalone Tangram reconstruction check.

    Parses CLI args, loads ``full_transcriptome.h5ad`` and the panel ``.h5ad``
    files from ``--preprocessed_dir``, runs
    :func:`_tangram.run_tangram_reconstruction_check` for each panel, writes
    ``{output_dir}/tangram_summary.json`` plus per-panel CSVs, and returns the
    process exit code (0 success, 1 runtime error, 2 tangram-sc missing).
    """
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
        "--random_state", type=int, default=42,
        help="Random seed for the stratified train/test split (default: 42)",
    )
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of stratified k-fold splits to loop over and aggregate (default: 5).",
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

    # Generate the shared stratified splits once so they can be reused across all
    # panels and match NMF/PCA/ICA/Ridge evaluation (same cells guaranteed, same
    # n_splits/random_state convention). Every fold is evaluated and the results
    # aggregated (mean ± std) — see _aggregate_and_save_tangram_folds.
    if not args.no_split:
        from _splits import generate_evaluation_splits
        splits = generate_evaluation_splits(
            adata_full,
            celltype_col=args.celltype_col,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        print(
            f"Generated {len(splits)} stratified fold(s) (random_state={args.random_state})"
        )
    else:
        splits = None
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
            if splits is not None:
                fold_results = []
                for fold, (train_idx, test_idx, per_celltype_splits) in enumerate(splits):
                    print(f"  Fold {fold + 1}/{len(splits)}...")
                    fold_result = run_tangram_reconstruction_check(
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
                        fold=fold,
                    )
                    fold_results.append(fold_result)
                result = _aggregate_and_save_tangram_folds(panel_output_dir, panel_name, fold_results)
                summary[panel_name] = result
                if "global" in result:
                    gm = result["global"]
                    print(
                        f"  ✓ Global  (n_folds={result['n_folds']}) — "
                        f"MSE={gm['mse_mean']:.4f}±{gm['mse_std']:.4f}  "
                        f"ExpVar={gm['expvar_mean']:.4f}±{gm['expvar_std']:.4f}"
                    )
                if "per_celltype_summary" in result:
                    pcs = result["per_celltype_summary"]
                    print(
                        f"  ✓ Per-celltype (n_folds={result['n_folds']}) — "
                        f"weighted_expvar={pcs.get('weighted_expvar_mean', float('nan')):.4f}"
                        f"±{pcs.get('weighted_expvar_std', float('nan')):.4f}"
                    )
            else:
                result = run_tangram_reconstruction_check(
                    adata_full=adata_full,
                    adata_subset=adata_panel,
                    output_dir=panel_output_dir,
                    dataset_name=panel_name,
                    celltype_col=args.celltype_col,
                    num_epochs=args.n_epochs,
                    run_global=run_global,
                    run_per_celltype=run_per_celltype,
                    train_idx=None,
                    test_idx=None,
                    per_celltype_splits=None,
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