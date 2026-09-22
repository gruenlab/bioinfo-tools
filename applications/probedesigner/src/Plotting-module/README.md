# Plotting-module

Visualisation functions for evaluation and selection results. See `CLAUDE.md` in this
directory for the full catalogue of plot types, colour schemes, and figure sizes;
this README is the quick map.

## Entry points

| Script | What it plots |
|---|---|
| `plot_evaluation.py` | Baseline + variability + optional Tangram-comparison plots from pre-computed evaluation CSVs. |
| `plot_nmf_independent.py` | NMF-independent reconstruction comparison (neural network / LVAE / Tangram). Recomputes macro/weighted metrics locally — see `docs/doc-pipeline/audit_2.md` §2. |
| `plot_umaps.py` | UMAPs from preprocessed panel `.h5ad` files. |
| `plot_pipeline_results.py` | Unified CLI for `k_varying` / `stability` / `evaluation` / `selection` result directories. |
| `plot_raw_vs_log_factor_umaps.py` | Raw-vs-log NMF factor UMAP grids. |

## Internal modules

| File | Contents |
|---|---|
| `_clustering_plots.py` | Clustering / kNN / cell-type plots; dataset-name → strategy / display-name parsing; colour + marker maps. |
| `_variability_plots.py` | NMF variability metric plots; factor-range mode; category colours. |
| `_selection_plots.py` | Gene dotplot, feature-importance, F1 distribution, confusion matrix, gene-source distribution. |
| `_reconstruction_plots.py` | Tangram-vs-NMF per-celltype + aggregated bar charts. Imported directly by `plot_evaluation.py`; not in `__init__.py`. |
| `_stability_plots.py` | Gene frequency / overlap, metric summaries, feature-UMAP grids. |
| `_k_varying_plots.py` | K-varying reconstruction / stability / baseline / grid plots. |
| `_comparison_umaps.py` | Raw-vs-log NMF factor UMAP-grid helpers. |
| `_constants.py` | Column names, DPI / font / line-size constants, colour schemes. |

## Known caveats

- `--png_dpi` is currently ignored: every `savefig` in `_clustering_plots.py` /
  `_variability_plots.py` hard-codes `dpi=DEFAULT_PNG_DPI` (600); `_selection_plots.py`
  hard-codes 300. (`docs/doc-pipeline/audit_2.md` B2.)
- `_clustering_plots.py` / `_variability_plots.py` mutate global `plt.rcParams` at
  import; `_k_varying_plots.py` calls `sns.set_style` at import.

## Public API

`__init__.py` re-exports ~45 functions. Import directly under the flat scheme:

```python
import sys
sys.path.insert(0, ".../Code/RecoVar/Plotting-module")
from _clustering_plots import plot_clustering_quality_ari
from _variability_plots import plot_aggregated_celltype_metrics
```

## Outputs

PNG (and PDF/SVG via `--format` in `plot_pipeline_results.py`). Two functions also emit
a CSV: `plot_celltype_f1_heatmap` → `celltype_f1score_matrix.csv`;
`plot_nmf_independent.main` → `{method}_{gene_count}genes_metrics.csv`. No parameters
JSON is written next to plotting output.
