# Plotting-Module Documentation

> **Stale sections (2026 audit).** `_combination_dotplot.py` / `plot_combination_dotplot`
> were removed as dead code — every "Combination Dotplot" section below is obsolete.
> `global-gene-filling` no longer exists in the RecoVar Selection module. The
> Modules_v2-era strategy vocabulary (`dt_nmf`/`dt_pca`/`dt_simple`/`dt_deg`, the
> ratio-in-name `dt_nmf_DT…_Dimred…`, the `method_a`/`method_b` loading-weight split,
> and the `category-9` comparison mode) was retired: parsers and colour tables now use
> `RecoVar` / `RecoVar_PCA` / `rf_simple` / `rf_deg` and key hybrid colours by dimred
> share only. See `docs/doc-pipeline/audit_2.md` Q15. **DPI: the module standard is
> `DEFAULT_PNG_DPI` / `ANALYSIS_PNG_DPI` = 600.** `plot_evaluation.py --png_dpi` is wired
> through (Q6); `_selection_plots.py`, `plot_nmf_independent.py --dpi` and
> `plot_umaps.py --png_dpi` all default to 600 (Q13-adjacent). Prose below that still says
> "300 (default)" is stale. See `docs/doc-pipeline/audit_2.md` §8 / §11.

## Overview

The Plotting-module provides comprehensive visualization functions for spatial transcriptomics probe panel evaluation and selection results. It supports visualization of clustering quality, neighborhood preservation, cell-type classification, NMF variability metrics, gene selection outcomes, and reconstruction comparisons.

**Location:** `Code/RecoVar/Plotting-module`

**Core Libraries:**
- **matplotlib** - Primary plotting engine
- **seaborn** - Statistical visualizations (heatmaps, barplots)
- **scanpy** - Single-cell specific plots (dotplots, UMAPs)

---

## Module Structure

### Python Files

1. **`__init__.py`** - Module initialization and exports
2. **`_constants.py`** - Shared constants and configuration
3. **`_clustering_plots.py`** - Clustering, kNN, and cell-type classification plots
4. **`_variability_plots.py`** - NMF variability metrics plots
5. **`_selection_plots.py`** - Gene selection result plots
6. **`_reconstruction_plots.py`** - Tangram-vs-NMF reconstruction plots
7. **`_stability_plots.py`** - Stability analysis visualization (gene frequency, overlap, metric summaries)
8. **`_k_varying_plots.py`** - K-varying analysis visualization (reconstruction quality, gene stability, baseline comparison)
9. **`_comparison_umaps.py`** - Raw-vs-log NMF factor UMAP comparison helpers
10. **`plot_evaluation.py`** - Main CLI for evaluation result plots
11. **`plot_nmf_independent.py`** - NMF reconstruction comparison plots
12. **`plot_umaps.py`** - UMAP visualization
13. **`plot_raw_vs_log_factor_umaps.py`** - CLI for factor-grid UMAP comparison
14. **`plot_pipeline_results.py`** - Unified CLI for all analysis types (k_varying, stability, evaluation, selection)

(`_combination_dotplot.py` was removed — it was dead code, imported by nothing.)

---

## Input Data Formats

### 1. Baseline Evaluation Results (CSV)

**Used by:** `plot_evaluation.py`, `_clustering_plots.py`

#### Clustering Results (`clustering_results.csv`)
```
dataset_name,n_clusters,ARI,NMI,representation
panel_A,10,0.85,0.78,pca
panel_B,10,0.72,0.65,nmf
```

**Columns:**
- `dataset_name` - Name of the probe panel or dataset
- `n_clusters` - Number of clusters used
- `ARI` - Adjusted Rand Index score
- `NMI` - Normalized Mutual Information score
- `representation` - Dimensionality reduction method (pca/nmf)

#### Neighborhood Preservation Results (`neighborhood_results.csv`)
```
dataset,k,preservation_score,optimal_k
panel_A,5,0.65,15
panel_A,10,0.72,15
```

**Columns:**
- `dataset` - Panel/dataset name
- `k` - Number of nearest neighbors
- `preservation_score` - kNN preservation score
- `optimal_k` - Optimal k value for this dataset

#### Cell-Type Classification Results (`celltype_results.csv`)
```
dataset,celltype,f1-score,accuracy
panel_A,Neuron,0.92,0.89
panel_A,Astrocyte,0.85,0.88
```

**Columns:**
- `dataset` - Panel/dataset name
- `celltype` - Cell type name
- `f1-score` - F1 classification score
- `accuracy` - Classification accuracy

**Data source:** Evaluation-module baseline metrics

---

### 2. Variability Evaluation Results (CSV)

**Used by:** `plot_evaluation.py`, `_variability_plots.py`

#### NMF Representation (`nmf_representation.csv` or per-dataset CSVs)
```
gene_list,celltype,analysis_type,mse_test_probe,expvar_test_probe,n_cells,skipped
panel_A,Neuron,per_celltype,0.0023,0.85,1250,False
panel_A,global,global,0.0035,0.78,5000,False
```

**Columns:**
- `gene_list` - Panel name
- `celltype` - Cell type or "global"
- `analysis_type` - "per_celltype" or "global"
- `mse_test_probe` - Mean squared error on test set
- `expvar_test_probe` - Explained variance on test set
- `n_cells` - Number of cells analyzed
- `skipped` - Whether analysis was skipped (True/False)

#### Mapping Performance (`mapping_performance.csv` or per-dataset CSVs)

**Same structure as NMF representation**

#### Aggregation Levels

- **Per-celltype:** Individual cell type metrics
- **Global:** Whole-dataset metrics
- **Summary:** Weighted and macro-averaged across cell types

**Data source:** Evaluation-module variability metrics

---

### 3. Selection Results (CSV)

**Used by:** `_selection_plots.py`

#### Gene Provenance (`final_panel_with_provenance.csv`)
```
gene,source,celltype,importance
Gene1,rf_deg,Neuron,0.95
Gene2,dimred,global,0.88
Gene3,overlap→rf_deg,Astrocyte,0.82
```

**Columns:**
- `gene` - Gene symbol
- `source` - Selection source category
- `celltype` - Associated cell type (or "global")
- `importance` - Feature importance score (optional)

#### Feature Importance (`feature_importance.csv`)
```
gene,fold,importance,celltype
Gene1,0,0.95,Neuron
Gene1,1,0.93,Neuron
```

**Columns:**
- `gene` - Gene symbol
- `fold` - Cross-validation fold number
- `importance` - Random forest feature importance
- `celltype` - Cell type

#### Confusion Matrix (`confusion_matrix.csv`)
```
true_label,predicted_label,count,seed,fold
Neuron,Neuron,450,42,0
Neuron,Astrocyte,15,42,0
```

**Columns:**
- `true_label` - True cell type
- `predicted_label` - Predicted cell type
- `count` - Number of cells
- `seed` - Random seed
- `fold` - Cross-validation fold

---

### 4. AnnData Objects (H5AD)

**Used by:** `plot_umaps.py`, `_clustering_plots.py`

#### Required Contents

**Core matrices:**
- `.X` - Expression matrix (log-normalized, genes × cells)
- `.obsm['X_pca']` or `.obsm['X_nmf']` - Dimensionality reduction embeddings

**Annotations:**
- `.obs[celltype_col]` - Cell type annotations (categorical)
- `.obs['leiden']` - Leiden clustering results (optional, categorical)

**Variables:**
- `.var_names` - Gene symbols
- `.var['highly_variable']` - HVG markers (optional, boolean)

**Example structure:**
```python
adata.X  # shape: (n_cells, n_genes)
adata.obsm['X_pca']  # shape: (n_cells, n_components)
adata.obs['cell_type']  # categorical with cell type labels
adata.obs['leiden']  # categorical with cluster assignments
```

### 5. Raw-vs-Log NMF Factor Comparison (H5AD)

**Used by:** `plot_raw_vs_log_factor_umaps.py`, `_comparison_umaps.py`

#### Required Contents

**Core matrices:**
- `.X` - Expression matrix used to compute UMAP if `.obsm['X_umap']` is absent
- `.obsm['W_nmf_factors_raw_{probeset_size}']` - Raw-count NMF W matrix, shape `(n_cells, n_factors)`
- `.obsm['W_nmf_factors_log_{probeset_size}']` - Log-normalized NMF W matrix, shape `(n_cells, n_factors)`

**Optional:**
- `.obsm['X_umap']` - If present, reused directly; otherwise computed from PCA/neighbors

**Grid layout:**
- Rows: `Raw` (top), `Log` (bottom)
- Columns: `Factor 1 ... Factor N`
- Color scaling: shared *per factor column* across raw/log rows for direct comparison

---

## Plot Types and Layouts

### 1. Clustering Plots (`_clustering_plots.py`)

#### UMAP Visualizations
- **Layout:** 2×2 or 3×2 grid
- **Subplots:** Clusters, cell types, Leiden assignments
- **Purpose:** Visual inspection of dimensionality reduction quality

#### ARI/NMI Line Plots
- **X-axis:** Number of clusters (k)
- **Y-axis:** ARI or NMI score
- **Lines:** One per strategy/panel
- **Purpose:** Compare clustering quality across strategies

#### F1 Heatmap
- **Rows:** Cell types
- **Columns:** Datasets/panels (x-axis labels are the shortened method name via
  `extract_display_name_from_dataset`, e.g. `"RecoVar"`, not the full settings string)
- **Color:** F1 score (viridis colormap)
- **Output:** PNG only (`celltype_f1score_heatmap.png`) + the pivot matrix as CSV — no
  PDF export
- **Purpose:** Cell-type classification performance overview

#### Celltype Classification Diagnostics — Global (`plot_celltype_classification_diagnostics_global`)
- **Layout:** One figure, three side-by-side bar-chart panels — Macro F1, Weighted F1,
  Accuracy — one bar per dataset in each, all three sharing the same dataset ordering
  (sorted by Macro F1 descending)
- **Colors:** Macro F1 blue (`#42a5f5`), Weighted F1 teal (`#00838f`), Accuracy dark
  blue (`#1565c0`) with a magenta (`#ad1457`) dashed max-accuracy reference line —
  matched to `Experiments/Benchmark_seed-analysis/aggregated/Evaluation/plots/LCA/seedvar_clustering_100g_LCA_ari.svg`'s
  palette
- **Output:** `celltype_classification_diagnostics_global.png`
- **Purpose:** Single-number, dataset-wide classification summary (accuracy plus both
  F1 averaging conventions) complementing the per-cell-type F1 heatmap above

#### Confusion Matrices
- **Layout:** One heatmap per call (raw counts, and a separate row-normalised variant)
- **Title:** Shortened method name (`extract_display_name_from_dataset`), not the raw
  settings-laden panel name
- **Filename:** `confusion_matrix.png` / `confusion_matrix_normalized.png` — no panel
  name in the filename (settings/seed/size live in the output folder path)
- **Axes:** True vs. predicted cell types
- **Color:** Sequential colormap
- **Purpose:** Detailed classification error analysis

---

### 2. Variability Plots (`_variability_plots.py`)

#### Aggregated Metrics Comparison
- **Layout:** Side-by-side grouped bar charts
- **Groups:** Weighted average vs. macro average
- **Metrics:** MSE (lower is better) or Explained Variance (higher is better)
- **Baseline:** Red dashed line for full NMF baseline
- **Purpose:** Compare global performance across strategies

#### Per-Celltype Metrics
- **Layout:** Horizontal bar charts
- **Sorting:** By performance (ascending for MSE, descending for ExpVar)
- **Colors:** Light blue bars with dark blue edges
- **Labels:** Numeric values on each bar
- **Purpose:** Identify cell-type-specific performance

#### Global Combined Plots
- **Layout:** All methods on single plot
- **Metrics:** MSE and Explained Variance
- **Grouping:** By analysis type (global vs. per-celltype)
- **Purpose:** Comprehensive strategy comparison

---

### 3. Selection Plots (`_selection_plots.py`)

#### Gene Source Distribution
- **Layout:** Stacked or grouped bar chart
- **Colors:** Color-coded by gene source category
- **X-axis:** Strategies or panels
- **Y-axis:** Gene count or fraction
- **Purpose:** Understand gene selection provenance

#### Genes Per Cell Type (`plot_genes_per_celltype`)
- **Layout:** Single-panel vertical bar chart, one bar per cell type + a trailing
  `(unattributed)` bar
- **Y-axis:** number of distinct panel genes informative for that cell type (a gene
  informative for *k* cell types counts once toward each)
- **Reads:** the `informative_celltypes` column on a `{strategy}_panel_information.csv` /
  `RecoVar_panel_information.csv` (combination or single-strategy) — written by
  `Selection-module`; the plot is just
  `df['informative_celltypes'].str.split('|').explode().value_counts()` drawn as bars.
  Falls back to merging `contributing_celltypes` / `rf_contributing_celltypes` /
  `rf_celltype` / `celltype` for files without the column (e.g. the trimmed
  `rf_deg_panel_information.csv`/`rf_simple_panel_information.csv` schema, whose only
  cell-type column is `rf_celltype`).
- **Colour:** plain (`#4c72b0`); `(unattributed)` grey; optional `highlight_celltypes`
  list drawn `#c0392b` with a legend
- **Writes:** the PNG only — **no CSV** (the counts live in the panel-information CSV)
- **Purpose:** diagnostic of per-cell-type gene coverage for the selection module

#### Gene Expression Dotplot (`plot_final_gene_dotplot`)
- **Style:** Scanpy-style dotplot
- **Dot color:** Mean expression (`viridis` colormap by default)
- **Dot size:** Fraction of cells expressing
- **Rows:** All cell types (`groupby`), so cross-reactivity/specificity of every
  shown gene is visible regardless of which genes are on the x-axis
- **Called once per cell type** by `run_combination_selection.py::_generate_combination_dotplot`
  (via `_build_per_celltype_gene_lists`) — one dotplot per cell type with ≥1
  member gene, instead of one dotplot for the whole panel. Each cell type's plot is a
  flat gene list (no bracket grouping) built from that gene's OWN selecting
  component, keyed by `gene_source`, rather than the combined
  `RecoVar_panel_information.csv`'s `primary_celltype`/`secondary_celltypes_nmf`
  columns (those apply a value-based fallback where RF's attribution always
  overwrites NMF's, discarding NMF's often much richer per-celltype signal — NMF is
  fit *independently per cell type*, so there's no real "primary vs. secondary"
  tiering to begin with, just "which cell type's fit selected this gene"):
  - `"random forest"` / `"gap-fill (random forest)"` genes: membership = RF's Gini
    argmax (`rf_celltype`) plus any other cell type crossing
    `RF_CONTRIBUTING_CELLTYPE_MIN_SHARE` in `rf_celltype_scores` (usually just the
    argmax in practice).
  - `"NMF"` / `"gap-fill (NMF)"` genes: membership = the dimred component's
    `celltype` union its `informative_celltypes`.
  - `"overlap"` genes: the union of both — selected independently by both methods, so
    a gene can legitimately land on more than one cell type's plot.
  - `"force_include"` genes, or a gene missing from the relevant component file(s):
    unattributed — go into one extra flat `unattributed_final_gene_dotplot.png`.
- **Gene order (x-axis):** hierarchical clustering via
  `_selection_plots._order_genes_by_celltype_contribution()`, called per cell type
  right before plotting (not applied to the unattributed fallback plot, which has no
  single target cell type to order against). Each gene's mean expression per
  `groupby` category is z-scored per-row (relative pattern, not absolute magnitude);
  genes are clustered on those z-scored profiles (`scipy.cluster.hierarchy.linkage`,
  average linkage, correlation distance); the resulting tree is walked from the root,
  ordering each merge's two children by mean z-scored value in the *target* cell type
  (higher first). Net effect: genes most specific to the target cell type lead,
  tapering to least-specific, while genes with similar expression patterns stay
  clustered adjacently. Falls back to input order (non-fatal) if fewer than 3 genes,
  the target cell type isn't in `adata.obs[groupby]`, or clustering fails.
- **Title:** `"Panel genes for {celltype}"` (via the `title` parameter, rendered
  with `plt.suptitle`) when called this way; `title=None` (no title) otherwise, e.g.
  the legacy `Analysis-scripts/pipeline/plot_recovar_selection.py` driver, which
  still calls this function with the whole panel in one flat, unbracketed dotplot.
- **Chunking:** within any one call (whole-panel or per-cell-type), gene lists of
  more than 100 genes are still split into multiple figures of at most 100 genes
  each (`..._chunk1.png`, `..._chunk2.png`, ..., title suffixed `" (part i/N)"` for
  chunks after the first) so each dotplot stays legible; 100 genes or fewer keep the
  single unsuffixed filename.
- **Purpose:** Visualize expression patterns of selected genes, broken down by which
  cell type they mark

#### Feature Importance Plots
- **Layout:** Horizontal bar chart (top N genes per fold)
- **Sorting:** By importance (descending)
- **Purpose:** Identify most discriminative genes

#### F1 Distribution
- **Layout:** Histogram
- **X-axis:** F1 score bins
- **Y-axis:** Frequency across folds
- **Purpose:** Assess classification stability

---

### 4. Reconstruction Plots (`_reconstruction_plots.py`)

#### Per-Celltype Comparison
- **Layout:** Grouped bar chart
- **Groups:** Cell types
- **Bars:** Tangram (orange) vs. NMF (blue)
- **Metrics:** MSE and Explained Variance
- **Purpose:** Compare reconstruction methods per cell type

#### Aggregated Metrics
- **Layout:** Bar chart with macro/weighted averages
- **Colors:** Tangram (orange) vs. NMF (blue)
- **Metrics:** MSE and Explained Variance
- **Purpose:** Overall reconstruction comparison

---

### 4b. Cross-method reconstruction comparison (`_method_comparison_plots.py`, CLI `plot_method_comparison.py`)
- **Layout:** 2 rows (all genes / non-panel genes only) x 3 columns (ExpVar global-mean, ExpVar variance-weighted, MSE, log scale). x = NMF, PCA, ICA, Ridge (raw), Ridge (lognorm), Tangram, each tagged with the data space it was run in; bars = one per panel (RecoVar raw sel., RecoVar lognorm sel., Spapros); error bars = std across CV folds.
- **Inputs:** the shared per-method results root (`nmf/`, `PCA/`, `ICA/`, `ridge-regression/`, `tangram/<panel>/global/`); a missing file or column raises instead of drawing NaN.
- **Outputs:** `reconstruction_method_comparison.png`, `..._values.csv` (long table, all subsets x modes x MSE), `plot_parameters.json`.
- **Caveat:** MSE is only comparable within a data space (raw vs lognorm). scVI is not in the method table yet (its evaluation is blocked, see `docs/doc-pipeline/scvi-failure-report_audit3-validation.md`).

---

### 5. Combination Dotplot (`_combination_dotplot.py`)

#### Multi-Strategy Expression Dotplot
- **Axes:** Genes × Cell types (or swapped)
- **Dot color:** Mean expression (Reds colormap, 0-100%)
- **Dot size:** Fraction expressing (power-law scaled)
- **Gene labels:** Colored by selection source
- **Legends:**
  - Color bar for mean expression
  - Size legend for fraction expressing (0%, 25%, 50%, 75%, 100%)
  - Source legend for gene label colors
- **Purpose:** Comprehensive expression overview with provenance

---

## Color Schemes and Symbol Assignments

### Strategy-Specific Colors

#### Baseline Strategies
```python
'Random':    '#808080'  # Gray
'HVG':       '#f0dd0d'  # Yellow
```

#### Simple Strategies
```python
'deg_only':   '#fa9c4a'  # Orange
'rf_simple':  '#d56a0d'  # Dark orange
'rf_deg':     '#875223'  # Brown
```

#### Dimred-Only Strategies
```python
'nmf':  '#cc0066'  # Dark pink
'pca':  '#006600'  # Dark forest green
```

#### Hybrid RecoVar Strategies (RF + NMF, PURPLE Family)

Keyed by dimred (NMF) share percentage; lighter = smaller share, darker = larger:
- `RecoVar_10`: `#DDAAFF` (very light purple)
- `RecoVar_25`: `#BB88FF` (light purple)
- `RecoVar_50`: `#9933FF` (vivid purple)
- `RecoVar_75`: `#7700CC` (deep purple)
- `RecoVar_90`: `#550099` (very deep purple)

#### Hybrid RecoVar_PCA Strategies (RF + PCA, LIME-GREEN Family)

Keyed by dimred (PCA) share percentage:
- `RecoVar_PCA_10`: `#CCFF88` (very light lime)
- `RecoVar_PCA_25`: `#AAFF55` (light lime)
- `RecoVar_PCA_50`: `#77DD22` (vivid lime)
- `RecoVar_PCA_75`: `#55AA00` (deep lime green)
- `RecoVar_PCA_90`: `#338800` (very deep lime green)

(`_clustering_plots.py` keys these dicts by integer share; `_variability_plots.py`'s
`get_category_colors` uses the string keys shown above.)

#### External Reference Panels
```python
'Spapros':    '#957d06'  # Gold/brown
'mMulti_v1':  '#000000'  # Black
'5k':         '#000000'  # Black
'NSForest':   (gradient)  # From external gradient
'scMER':      (gradient)  # From external gradient
```

**External panel gradient:**
```python
["#76EFED", "#55AFAE", "#3E7C7B", "#203F3E", "#4C8098", "#1C4F66"]
# Cyan → Teal → Dark teal → Very dark teal → Blue-gray → Dark blue-gray
```

---

### Gene Source Colors

There is **no** module-level `DEFAULT_SOURCE_COLORS` constant. The gene-source colour
dict is local to `plot_gene_source_distribution` in `_selection_plots.py` and keys on
display labels (`'DT-unique'`, `'{reduction_type}-unique'`,
`'Both (DT & {reduction_type})'`, `'Gap-filling: Cell-type'`, …), not the raw
`gene_source` values the Selection module writes. The `gene_source` vocabulary for panel
rows in `RecoVar_panel_information.csv` is: `"random forest"`, `"NMF"`, `"overlap"`
(selected by both, credited to random forest), `force_include`, `"gap-fill (NMF)"`,
`"gap-fill (random forest)"` (non-panel rows carry `"random forest"` / `"NMF"` too, for
which candidate pool the gene came from). These are display strings — Selection-module's
internal identifiers (`rf_deg`, `dimred`, `overlap→rf_deg`, `gap_fill_celltype`,
`gap_fill_deg`) are renamed only at CSV-write time, via
`run_combination_selection.py::_GENE_SOURCE_DISPLAY_RENAME`. (`dimred_replacement` and
`gap_fill_global` are not produced — those features were removed from the Selection
module.)

---

### Reconstruction Comparison Colors

```python
TANGRAM_COLOR = "#FF9800"  # Orange
NMF_COLOR     = "#2196F3"  # Blue
```

---

### Marker Symbols

For line plots with multiple strategies:

```python
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
# Circle, Square, Triangle-up, Diamond, Triangle-down,
# Triangle-left, Triangle-right, Pentagon
```

Markers cycle through this list when more than 8 strategies are plotted.

---

## Plot Dimensions, Fonts, and Styling

### Figure Sizes

#### Clustering/Neighborhood Plots
- **Line plots:** `(10, 6)` or `(12, 6)` inches
- **Heatmaps:** Dynamic based on data
  - Width: `max(8, n_datasets × 0.6)`
  - Height: `max(6, n_celltypes × 0.4)`
- **UMAP grids:** `(16, 10)` for 2×2, `(24, 10)` for 3×2

#### Variability Plots
- **Aggregated metrics:** `(20, max(8, len(data) × 0.5))`
- **Per-celltype:** `(12, max(6, len(data) × 0.4))`
- **Combined global:** `(14, 8)`

#### Selection Plots
- **Source distribution:** `(10, 6)`
- **Dotplot:** Dynamic based on genes/celltypes
- **Feature importance:** `(10, max(6, n_genes × 0.3))`

#### Reconstruction Plots
- **Per-celltype:** `(max(10, n_celltypes × 0.9), 6)`
- **Aggregated:** `(10, 5)`

#### Combination Dotplot
Auto-calculated based on content:
```python
if swap_axes:
    width = max(6, n_groups × 0.45 + 3)
    height = max(5, n_genes × 0.22 + 2.5)
else:
    width = max(12, n_genes × 0.22 + 3)
    height = max(5, n_groups × 0.45 + 2.5)
```

---

### Font Sizes

#### Combination Dotplot
- **Gene labels:** 9 pt (default)
- **Group labels:** 10 pt (default)
- **Title:** 11 pt
- **Colorbar label:** 9 pt
- **Legend title:** 11 pt
- **Legend text:** 10 pt

#### Selection Dotplot
- **Title:** 18 pt
- **Axis labels:** 16 pt
- **Tick labels:** 14 pt
- **Legend title:** 14 pt
- **Legend text:** 12 pt

#### Variability Plots
- **Title:** 14-15 pt (bold)
- **Axis labels:** 12-13 pt (bold)
- **Tick labels:** 8-10 pt
- **Value labels on bars:** 8-10 pt (bold)

#### Clustering Plots
- **Title:** 14-16 pt
- **Axis labels:** 12-14 pt
- **Tick labels:** 10-12 pt
- **Legend:** 10-11 pt

---

### Line Widths and Sizes

#### Dot Sizes (Combination Dotplot)
```python
# Power-law scaling for fraction expressing
size = frac ** size_exponent × (largest_dot - smallest_dot) + smallest_dot

# Defaults:
smallest_dot = 0.0
largest_dot = 200.0
size_exponent = 1.5
```

**Example sizes:**
- 0% expressing: 0.0
- 25% expressing: ~31
- 50% expressing: ~88
- 75% expressing: ~163
- 100% expressing: 200

#### Bar Plot Styling
- **Edge color:** `'#00008B'` (dark blue)
- **Edge width:** 2 pt
- **Fill color:** `'lightblue'`
- **Alpha:** 0.8 (semi-transparent)

#### Line Plot Styling
- **Line width:** 2-2.5 pt
- **Marker size:** 6-8 pt
- **Marker edge width:** 1.5 pt

#### Grid Lines
- **Width:** 0.1-0.4 pt
- **Color:** `'0.88'` or `'gray'`
- **Alpha:** 0.3-0.7
- **Style:** Solid or dashed

#### Baseline Reference Lines
- **Style:** `'--'` (dashed)
- **Color:** `'red'`
- **Width:** 1.5-2 pt
- **Alpha:** 0.7

---

## Titles and Legends

### Title Formats

#### Clustering Quality
```
"Clustering Quality (ARI) - {dim_reduction.upper()}"
"Clustering Quality (NMI) - {dim_reduction.upper()}"
```
Examples:
- `"Clustering Quality (ARI) - PCA"`
- `"Clustering Quality (NMI) - NMF"`

#### Neighborhood Preservation
```
"Neighborhood Preservation by k"
```
(`plot_optimal_neighborhood_preservation` / `plot_neighborhood_preservation_heatmap`
were removed — see "Removed" note under "Plot Types and Layouts" above.)

#### Cell-Type Classification
```
"Celltype Identification Accuracy by Geneset"
"Celltype Identification F1 (Macro vs. Weighted) by Geneset"
"Confusion Matrix - {method_name}\nAccuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}"
"Confusion Matrix (row-normalised) - {method_name}\nAccuracy: ... | Macro F1: ..."
```
`{method_name}` is the shortened display name from `extract_display_name_from_dataset`
(e.g. `"RecoVar"`), not the raw settings-laden panel name.

#### Variability Metrics
```
"{method.title()} MSE - Weighted vs Macro Comparison{title_suffix}"
"{method.title()} Explained Variance - Weighted vs Macro Comparison{title_suffix}"
"{method.title()} MSE - {celltype} (lower is better){title_suffix}"
"{method.title()} Explained Variance - {celltype} (higher is better){title_suffix}"
```
Examples:
- `"Mechanistic MSE - Weighted vs Macro Comparison (100 genes)"`
- `"Mapping Explained Variance - Neuron (higher is better)"`

#### Reconstruction Comparison
```
"{panel_name} - Reconstruction Quality (per cell type)"
"{panel_name} - Reconstruction Quality (aggregated)"
```

#### Combination Dotplot
- User-defined or auto-generated from panel name
- Example: `"Gene Expression - Final Panel (500 genes)"`

---

### Legend Configurations

#### Color Legends (Strategy Lines)
- **Position:** `'lower right'`, `'lower center'`, or `'best'`
- **Frame:** `frameon=True`
- **Edge color:** `'0.8'` (light gray) or `'0.4'` (medium gray)
- **Edge width:** 1 pt
- **Columns:** Auto-calculated, usually 1-2, max 4 for many strategies
- **Font size:** 10-11 pt
- **Marker scale:** 1.0-1.2

#### Size Legends (Dotplot)
- **Title:** `"Fraction\nexpressing"`
- **Values shown:** `[0%, 25%, 50%, 75%, 100%]`
- **Marker sizes:** Scaled identically to plot dots
- **Position:** Right side of plot, separate from color legend
- **Frame:** Visible with light gray edge

#### Source Legends (Gene Labels)
- **Title:** `"Gene Source"` or `"Selection Source"`
- **Position:** `'lower right'` or `'center left'`
- **Columns:** Auto-calculated (max 4)
- **Markers:** Colored rectangles matching label colors
- **Font size:** 10-12 pt

#### Baseline Reference Lines
- **Label format:** `"Full NMF Baseline ({value:.2e})"`
- **Example:** `"Full NMF Baseline (2.34e-03)"`
- **Position:** In main legend with line style shown

---

## Output Formats

### Default Settings

From `_constants.py`:
```python
DEFAULT_PNG_DPI = 600
ANALYSIS_PNG_DPI = 600
DEFAULT_FIGURE_FORMAT = "png"
DEFAULT_COLORMAP = "viridis"
```

**Note:** all save paths now use `DEFAULT_PNG_DPI` (600) as the baseline.
`plot_evaluation.py --png_dpi` is wired through to `_clustering_plots.py` /
`_variability_plots.py` (Q6); `_selection_plots.py`, `plot_nmf_independent.py --dpi`,
`plot_umaps.py --png_dpi` and `plot_pipeline_results.py --dpi` all default to 600.
See `docs/doc-pipeline/audit_2.md` §11 (Q6, Q13).

### File Naming Conventions

#### Clustering Plots
```
clustering_quality_ari_{dim_reduction}.png
clustering_quality_nmi_{dim_reduction}.png
celltype_f1score_heatmap.png                      # PNG only, no PDF; + celltype_f1score_matrix.csv
celltype_classification_diagnostics_global.png    # 3 panels: Macro F1, Weighted F1, Accuracy
neighborhood_preservation_by_k.png
neighborhood_jaccard_by_celltype_pca.png          # 2-decimal annotations
confusion_matrix.png                              # no panel name in the filename
confusion_matrix_normalized.png
umap_visualization_{panel_name}.png
```
(`optimal_neighborhood_preservation.png` / `neighborhood_preservation_heatmap.png` and
their producing functions were removed — see the audit_3.md "Plotting-module polish"
addendum.)

#### Variability Plots
```
aggregated_{method}_mse_weighted_macro_comparison{suffix}.png
aggregated_{method}_expvar_weighted_macro_comparison{suffix}.png
celltype_{method}_mse_{celltype}_combined{suffix}.png
celltype_{method}_expvar_{celltype}_combined{suffix}.png
global_{method}_mse_comparison.png
global_{method}_expvar_comparison.png
```

**Method:** `nmf` or `mapping`
**Suffix:** Optional, e.g., `_100genes`

#### Selection Plots
```
gene_source_distribution.png                # no strategy suffix; settings live in the folder path
genes_per_celltype.png                      # PNG only; counts live in the panel-information CSV
{Celltype}_final_gene_dotplot.png           # one per cell type (run_combination_selection.py caller); <= 100 genes
{Celltype}_final_gene_dotplot_chunk1.png, _chunk2.png, ...  # same, > 100 genes for that cell type
unattributed_final_gene_dotplot.png         # panel genes with no informative_celltypes at all
final_gene_dotplot.png / _chunk1.png, ...   # legacy plot_recovar_selection.py caller: whole panel, one flat dotplot
confusion_matrix_seed{seed}_fold{fold}.png
feature_importance_seed{seed}_fold{fold}.png
f1_score_distribution.png
```

#### Reconstruction Plots
```
{panel_name}_reconstruction_per_celltype.png
{panel_name}_reconstruction_aggregated.png
nmf_independent_reconstruction_{dataset}_per_celltype.png
nmf_independent_reconstruction_{dataset}_aggregated.png
```

#### Combination Dotplot
```
combination_dotplot_{panel_name}.png
# Or user-specified filename
```

---

### Save Options

All plots are saved using:
```python
plt.savefig(path, dpi=png_dpi, bbox_inches='tight')
```

**Parameters:**
- `dpi`: 600 (default = `DEFAULT_PNG_DPI`) or user-specified
- `bbox_inches='tight'`: Removes excess whitespace
- `transparent=False`: White background (default)
- `facecolor='white'`: Explicit white background

**Output quality:**
- **300 DPI:** Publication-quality, suitable for papers
- **150 DPI:** Screen/presentation quality (optional)
- **PNG format:** Lossless compression, widely compatible

---

## CLI Entry Points

### 1. `plot_evaluation.py`

**Purpose:** Main CLI for plotting pre-computed evaluation results

**Usage:**
```bash
python plot_evaluation.py \
    --evaluation_type {baseline,variability,all} \
    --panels panel1,panel2,panel3 \
    --output_dir /path/to/output \
    --results_dir /path/to/results \
    [--group_name my_comparison] \
    [--png_dpi 300]
```

### 4. `plot_raw_vs_log_factor_umaps.py`

**Purpose:** Create side-by-side UMAP factor grids comparing raw vs log NMF factors.

**Usage:**
```bash
python plot_raw_vs_log_factor_umaps.py \
  --input_root /home/gruengroup/helene/SpatialProbeDesign_tmp/Experiments/LCA/Dimred-NMF-Raw-vs-Log \
  --probeset_sizes 100 200 500 \
  --n_neighbors 15 \
  --n_pcs 30 \
  --random_state 42 \
  --dpi 600
```

**Output:**
- `{input_root}/{size}-genes/plots/nmf_factor_umap_grid_raw_vs_log_{size}.png`

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--evaluation_type` | choice | Yes | Type of evaluation: `baseline`, `variability`, or `all` |
| `--panels` | str | Yes | Comma-separated list of panel names |
| `--output_dir` | path | Yes | Directory for output plots |
| `--results_dir` | path | No | Single result directory (use for `baseline` or `variability`, not `all`) |
| `--baseline_results_dir` | path | No* | Baseline result directory — required when `--evaluation_type all` |
| `--variability_results_dir` | path | No* | Variability result directory — required when `--evaluation_type all` |
| `--allow_mixed_sizes` | flag | No | Allow panels with different gene counts in one figure |
| `--group_name` | str | No | Name for this comparison group |
| `--png_dpi` | int | No | Plot resolution (default: 600; wired through to the plot functions since Q6) |

**Baseline-specific flags:**
| Argument | Type | Description |
|----------|------|-------------|
| `--plot_clustering` | flag | Plot clustering quality (ARI/NMI) |
| `--plot_neighborhood` | flag | Plot neighborhood preservation |
| `--plot_celltype` | flag | Plot cell-type classification |

**Variability-specific flags:**
| Argument | Type | Description |
|----------|------|-------------|
| `--plot_nmf` | flag | Plot NMF representation metrics |
| `--plot_mapping` | flag | Plot mapping performance metrics |
| `--plot_celltype_specific` | flag | Plot per-celltype breakdowns |

**Advanced options:**
| Argument | Type | Description |
|----------|------|-------------|
| `--external_names` | str | Comma-separated external panel names for special coloring |
| `--tangram_results_dir` | path | Directory with Tangram results for reconstruction comparison |

**Example:**
```bash
python plot_evaluation.py \
    --evaluation_type all \
    --panels RecoVar_RF_0.5_Dimred_0.5,RecoVar_RF_0.25_Dimred_0.75,HVG \
    --output_dir ./plots/comparison_v1 \
    --results_dir ./results/eval_20240115 \
    --group_name factor_comparison \
    --plot_clustering \
    --plot_nmf \
    --plot_celltype_specific \
    --png_dpi 300
```

---

### 5. `plot_nmf_independent.py`

**Purpose:** Plot NMF-independent reconstruction comparisons (Tangram, neural networks, LVAE)

**Usage:**
```bash
python plot_nmf_independent.py \
    --base_dir /path/to/reconstruction_results \
    --output_dir /path/to/output \
    [--methods neural_network,LVAE,tangram] \
    [--dpi 300]
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--base_dir` | path | Yes | Base directory containing dataset subdirectories |
| `--output_dir` | path | Yes | Directory for output plots |
| `--methods` | str | No | Comma-separated methods to plot (default: neural_network,LVAE,tangram) |
| `--dpi` | int | No | Plot resolution (default: 600) |

**Directory structure expected:**
```
base_dir/
├── dataset1/
│   ├── nmf_representation.csv
│   └── mapping_performance.csv
├── dataset2/
│   ├── nmf_representation.csv
│   └── mapping_performance.csv
```

**Example:**
```bash
python plot_nmf_independent.py \
    --base_dir ./results/nmf_independent \
    --output_dir ./plots/nmf_independent \
    --methods neural_network,tangram \
    --dpi 300
```

---

### 2. `plot_umaps.py`

**Purpose:** Generate UMAP visualizations for probe panels

**Usage:**
```bash
python plot_umaps.py \
    --preprocessed_dir /path/to/h5ad_files \
    --output_dir /path/to/output \
    --panels "panel1*.h5ad,panel2*.h5ad" \
    [--full_transcriptome /path/to/full.h5ad] \
    [--celltype_col cell_type] \
    [--png_dpi 300]
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--preprocessed_dir` | path | Yes | Directory containing preprocessed h5ad files |
| `--output_dir` | path | Yes | Directory for output plots |
| `--panels` | str | Yes | Comma-separated glob patterns for panel files |
| `--full_transcriptome` | path | No | Path to full transcriptome h5ad (for comparison) |
| `--celltype_col` | str | No | Column name for cell types (default: `cell_type`) |
| `--png_dpi` | int | No | Plot resolution (default: 600) |
| `--n_neighbors` | int | No | UMAP n_neighbors parameter (default: 15) |
| `--n_pcs` | int | No | Number of PCs to use (default: 30) |

**Example:**
```bash
python plot_umaps.py \
    --preprocessed_dir ./data/preprocessed \
    --output_dir ./plots/umaps \
    --panels "RecoVar_*_100genes.h5ad,HVG_100genes.h5ad" \
    --full_transcriptome ./data/full_transcriptome.h5ad \
    --celltype_col cell_type \
    --n_neighbors 30 \
    --png_dpi 300
```

---

### 3. `plot_pipeline_results.py`

**Purpose:** Unified CLI for all analysis types — k-varying, stability, evaluation, and selection results

**Usage:**
```bash
python plot_pipeline_results.py \
    --analysis_type {k_varying,stability,evaluation,selection} \
    --input_dir /path/to/results \
    --output_dir /path/to/plots \
    [--plot_types reconstruction,stability,metrics] \
    [--dpi 600] \
    [--format {png,pdf,svg}] \
    [--color_scheme "nmf:#1f77b4"] \
    [--metric_filter mse,expvar]
```

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--analysis_type` | choice | Yes | Type of results to plot: `k_varying`, `stability`, `evaluation`, `selection` |
| `--input_dir` | path | Yes | Directory containing results from Analysis-scripts |
| `--output_dir` | path | Yes | Directory for output plots |
| `--plot_types` | str | No | Comma-separated plot types to generate |
| `--dpi` | int | No | Plot resolution (default: 600 = `ANALYSIS_PNG_DPI`) |
| `--format` | choice | No | Output format: `png`, `pdf`, `svg` (default: `png`) |
| `--color_scheme` | str | No | Custom colors as `key:color` pairs |
| `--metric_filter` | str | No | Subset of metrics to plot (e.g., `mse,expvar`) |

---

## Special Features

### 1. Factor-Range Mode

**Trigger:** When dataset names contain `_Nfactors_` pattern (e.g., `panel_2factors`, `panel_15factors`)

**Behavior:**
- Automatically detects factor numbers (2-15)
- Assigns factor-specific gradient colors:
  ```python
  factor_colors = {
      2: "#D32F2F",   # Red
      3: "#E64A19",   # Deep Orange
      4: "#F57C00",   # Orange
      5: "#FFA000",   # Amber
      # ... through ...
      15: "#808080"   # Gray
  }
  ```
- Display names formatted as: `"2-factors"`, `"15-factors"`
- Allows comparison of NMF component number effects

**Example datasets:**
- `RecoVar_RF_0.5_Dimred_0.5_2factors`
- `RecoVar_RF_0.5_Dimred_0.5_5factors`
- `RecoVar_RF_0.5_Dimred_0.5_15factors`

---

### 2. Multi-Size Mode

**Trigger:** When multiple panel sizes detected (e.g., 100, 200, 500 genes)

**Behavior:**
- Includes size in strategy labels: `"RecoVar 75% (100)"`
- Separate legend entries for each size
- Allows direct comparison of same strategy at different panel sizes
- Color remains consistent, marker/linestyle varies

**Example:**
```
RecoVar_RF_0.25_Dimred_0.75_100genes  → "RecoVar 75% (100)"
RecoVar_RF_0.25_Dimred_0.75_200genes  → "RecoVar 75% (200)"
RecoVar_RF_0.25_Dimred_0.75_500genes  → "RecoVar 75% (500)"
```

---

### 3. Gap-Filling Variants

**Detection:** Automatic based on panel naming

**Variants and labels:**
| Original Name | Display Label | Color Assignment |
|---------------|---------------|------------------|
| `DEG-based-filling` | `"DEG-fill"` | As per base strategy + suffix |
| `cell-type-specific-filling` | `"CT-fill"` | As per base strategy + suffix |

(`global-gene-filling` / `"Global-fill"` no longer exists — global fill was removed from
the Selection module.)

**Purpose:** Distinguish panels with different gap-filling strategies

**Example:**
```
RecoVar_RF_0.5_Dimred_0.5_DEG-based-filling → "RecoVar 50% (DEG-fill)"
```

---

### 4. Category-9 Special Formatting

**Removed (2026 audit, Q15).** The `category-9` comparison mode (`group_name='category-9'`
labels, filter+dimred colour keys, and the heatmap skip) was unreachable in the current
CLI and has been deleted.

**Purpose:** Handle special comparison category with different layout requirements

---

### 5. External Panel Integration

**Trigger:** Panel names in `external_names` argument

**Behavior:**
- Uses special colors from external gradient
- Different marker styles (stars, pentagons)
- Legend section separates external from internal panels
- Labeled as "(External)" in legends

**Common external panels:**
- **Spapros:** Gold/brown color
- **mMulti_v1:** Black
- **NSForest:** Cyan-teal gradient
- **scMER:** Cyan-blue gradient
- **5k:** Full 5000-gene panel (black)

---

## Key Functions Reference

### From `_clustering_plots.py`

#### `plot_clustering_quality_ari()`
**Purpose:** Plot ARI scores across number of clusters
**Inputs:** `clustering_results.csv`, strategy colors
**Outputs:** Line plot with legend

#### `plot_clustering_quality_nmi()`
**Purpose:** Plot NMI scores across number of clusters
**Inputs:** `clustering_results.csv`, strategy colors
**Outputs:** Line plot with legend

#### `plot_neighborhood_preservation_by_k()`
**Purpose:** Plot kNN preservation across k values
**Inputs:** `neighborhood_results.csv`, strategy colors
**Outputs:** Multi-line plot

#### `plot_neighborhood_preservation_celltype_heatmap()`
**Purpose:** Per-cell-type kNN Jaccard preservation heatmap (2-decimal annotations)
**Inputs:** `neighborhood_results.csv`
**Outputs:** `neighborhood_jaccard_by_celltype_pca.png`

(`plot_optimal_neighborhood_preservation()` and `plot_neighborhood_preservation_heatmap()`
were removed — see the audit_3.md "Plotting-module polish" addendum.)

#### `plot_celltype_f1_heatmap()`
**Purpose:** Heatmap of F1 scores; x-axis labels use the shortened method name
**Inputs:** `celltype_results.csv`
**Outputs:** `celltype_f1score_heatmap.png` (PNG only, no PDF) + matrix CSV

#### `plot_celltype_classification_diagnostics_global()`
**Purpose:** Dataset-wide classification summary — three side-by-side bar-chart panels
(Macro F1, Weighted F1, Accuracy), one bar per dataset each, sharing one dataset
ordering (by Macro F1 descending); the Accuracy panel keeps the dashed max-accuracy
reference line
**Inputs:** `celltype_results.csv` (dataset-level rows; requires `weighted_f1`, added
alongside `macro_f1` by `Evaluation-module/_clustering.py`)
**Outputs:** `celltype_classification_diagnostics_global.png`

#### `plot_umap_for_representation()`
**Purpose:** UMAP visualization grid
**Inputs:** AnnData object
**Outputs:** Multi-panel UMAP plot

---

### From `_variability_plots.py`

#### `plot_aggregated_celltype_metrics()`
**Purpose:** Weighted vs. macro average comparison
**Inputs:** Summary CSV with aggregated metrics
**Outputs:** Grouped bar chart with baseline reference

#### `plot_celltype_evaluation_results()`
**Purpose:** Per-celltype metric bars
**Inputs:** Per-celltype CSV
**Outputs:** Sorted horizontal bar chart

#### `create_combined_plot_with_info()`
**Purpose:** Global comparison of all methods
**Inputs:** All evaluation CSVs
**Outputs:** Comprehensive multi-strategy plot

---

### From `_selection_plots.py`

#### `plot_gene_source_distribution()`
**Purpose:** Show gene provenance breakdown; title uses the shortened method name
(no ODT category — dead code removed)
**Inputs:** `final_panel_with_provenance.csv`
**Outputs:** `gene_source_distribution.png` (no strategy/settings suffix)

#### `plot_genes_per_celltype()`
**Purpose:** Per-cell-type gene coverage (how many panel genes are informative for each
cell type) from the `informative_celltypes` column
**Inputs:** a combination `RecoVar_panel_information.csv` or single-strategy
`{strategy}_panel_information.csv` (or a passed df)
**Outputs:** `genes_per_celltype.png` (PNG only — no CSV, no strategy suffix)
**Wired into:** `plot_pipeline_results.py --analysis_type selection`

#### `plot_final_gene_dotplot()`
**Purpose:** Expression dotplot, all cell types as rows; called once per cell type by
`run_combination_selection.py` (see "Gene Expression Dotplot" above), or once for the
whole panel by the legacy `plot_recovar_selection.py` driver
**Inputs:** AnnData object + provenance CSV
**Outputs:** `{Celltype}_final_gene_dotplot.png` (or `_chunk1.png`/`_chunk2.png`/...
when that cell type has more than 100 informative genes) per cell type, plus
`unattributed_final_gene_dotplot.png` when applicable; `final_gene_dotplot.png` for
the legacy whole-panel caller

#### `plot_feature_importances()`
**Purpose:** Top feature importance per fold
**Inputs:** `feature_importance.csv`
**Outputs:** Multi-panel horizontal bar charts

#### `plot_confusion_matrix()` (`_clustering_plots.py`)
**Purpose:** Classification confusion matrices; title uses the shortened method name
**Inputs:** `y_test`/`y_pred` arrays (reproduces the eval classifier directly, not a CSV)
**Outputs:** `confusion_matrix.png` / `confusion_matrix_normalized.png` (no panel name
in the filename)

Note: `_selection_plots.py` has a *different*, unrelated `plot_confusion_matrix()`
(seed/fold-based, reads a pre-computed `confusion_matrix.csv`) — the two are not the
same function; only the `_clustering_plots.py` one changed here.

---

### From `_reconstruction_plots.py`

#### `plot_reconstruction_per_celltype()`
**Purpose:** Compare Tangram vs. NMF per cell type
**Inputs:** Tangram + NMF evaluation CSVs
**Outputs:** Grouped bar chart (orange vs. blue)

#### `plot_reconstruction_aggregated_metrics()`
**Purpose:** Overall reconstruction comparison
**Inputs:** Aggregated metrics from both methods
**Outputs:** Bar chart with macro/weighted averages

---

### From `_combination_dotplot.py`

#### `plot_combination_dotplot()`
**Purpose:** Multi-strategy expression overview
**Inputs:** AnnData objects + provenance information
**Outputs:** Combined dotplot with:
  - Color: Mean expression (Reds colormap)
  - Size: Fraction expressing (power-law scaled)
  - Gene labels: Colored by source

**Key parameters:**
- `swap_axes`: Transpose genes/celltypes
- `smallest_dot`, `largest_dot`: Size range
- `size_exponent`: Power-law exponent (default: 1.5)
- `gene_fontsize`, `group_fontsize`: Label sizes

---

### From `_k_varying_plots.py`

#### `plot_reconstruction_quality()`
**Purpose:** Reconstruction quality (MSE/ExpVar) as a function of panel size K
**Inputs:** DataFrame with `k`, `metric_type`, per-strategy columns
**Outputs:** Line plots per strategy

#### `plot_gene_stability()`
**Purpose:** Gene selection stability across different panel sizes
**Inputs:** DataFrame with gene overlap statistics per K
**Outputs:** Line plot showing fraction of genes retained

#### `plot_aggregate_metrics()`
**Purpose:** Macro and weighted aggregate metrics across K values
**Inputs:** DataFrame with `k`, `macro_mse`, `weighted_mse`, etc.
**Outputs:** Multi-panel line plot

#### `plot_reconstruction_metrics_grid()`
**Purpose:** Grid of reconstruction quality plots (one panel per metric/strategy combination)
**Inputs:** Aggregated results DataFrame
**Outputs:** Multi-panel grid figure

#### `plot_baseline_comparison()`
**Purpose:** Compare strategies against random/HVG baselines across K
**Inputs:** Results DataFrame with baseline columns
**Outputs:** Relative-improvement line plot

#### `plot_per_celltype_grid()`
**Purpose:** Per-cell-type metric grids across K values
**Inputs:** Per-celltype results DataFrame
**Outputs:** Grid of subplots (one per cell type)

---

## Usage Tips

### 1. Comparing Strategies
- Use consistent `--group_name` for related comparisons
- Include baseline strategies (Random, HVG) for context
- Use `--external_names` to highlight reference panels

### 2. Handling Large Datasets
- Per-celltype plots auto-scale height based on number of cell types
- Use `head_limit` and filtering for exploratory analysis
- Consider splitting into multiple plot groups if >20 strategies

### 3. Publication-Quality Plots
- 600 DPI is the default (publication quality); override per journal if needed
- Adjust font sizes if needed for specific journals
- Consider manually editing labels for very long strategy names
- Use `bbox_inches='tight'` (automatic) to remove excess whitespace

### 4. Troubleshooting

**Issue:** Colors not assigned correctly
**Solution:** Check panel naming follows convention: `{Filter}_{Baseline}_{strategy}_{N-genes}`,
where hybrid strategies are `RecoVar_RF_<rf>_Dimred_<dimred>` / `RecoVar_PCA_RF_<rf>_Dimred_<dimred>`

**Issue:** Missing data in plots
**Solution:** Verify CSV files have expected columns (see Input Data Formats)

**Issue:** Overlapping labels
**Solution:** Increase figure size or reduce font size in source code

**Issue:** Too many strategies in legend
**Solution:** Use multiple plot calls with strategy subsets

---

## Dependencies

```python
# Core
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors, patches
import seaborn as sns
import scanpy as sc

# Data handling
import pandas as pd
import numpy as np
from scipy.sparse import issparse

# Utilities
from pathlib import Path
import warnings
```

**Version requirements:**
- matplotlib ≥ 3.5
- seaborn ≥ 0.11
- scanpy ≥ 1.9
- pandas ≥ 1.4
- numpy ≥ 1.21

---

## Stability Analysis Plots

### Overview

Functions for visualizing gene selection stability and metric variability across pipeline iterations, with publication-quality improvements.

**Module:** `_stability_plots.py`

### Input Data Formats

#### Gene Stability Data (`gene_df`)
```csv
gene,n_selected,fraction_selected,n_rf_pool,n_nmf_pool
Gene1,5,1.0,5,0
Gene2,3,0.6,2,1
Gene3,2,0.4,0,2
```

**Columns:**
- `gene` - Gene symbol
- `n_selected` - Number of iterations gene was selected
- `fraction_selected` - Fraction of iterations (0.0-1.0)
- `n_rf_pool` - Times selected from RF pool
- `n_nmf_pool` - Times selected from NMF pool

#### Metrics Stability Data (`metrics_df`)
```csv
iteration,analysis_type,celltype,method,mse,expvar,macro_mse,macro_expvar,weighted_mse,weighted_expvar
0,per_celltype,Neuron,nmf,0.0023,0.85,0.0025,0.82,0.0024,0.83
1,per_celltype,Neuron,nmf,0.0021,0.86,0.0023,0.84,0.0022,0.85
summary_mean,per_celltype,Neuron,nmf,0.0022,0.855,0.0024,0.83,0.0023,0.84
summary_std,per_celltype,Neuron,nmf,0.0001,0.005,0.0001,0.01,0.0001,0.01
```

**Columns:**
- `iteration` - Iteration number or "summary_mean"/"summary_std"
- `analysis_type` - "per_celltype" or "global_reconstruction"
- `celltype` - Cell type name
- `method` - "nmf" or "mapping"
- `mse`, `expvar` - Per-celltype metrics
- `macro_mse`, `macro_expvar` - Macro-averaged aggregate metrics
- `weighted_mse`, `weighted_expvar` - Weighted aggregate metrics

---

### Plot Types

#### 1. Gene Frequency Plot
**Function:** `plot_gene_frequency(gene_df, output_dir, n_iterations, top_n=None, png_dpi=600)`

- **Layout:** Square (10×10 inches) horizontal bar chart
- **Purpose:** Shows how many iterations each gene was selected
- **Colors:** Blues colormap (darker = selected in more iterations)
  - Scale: 0.35-1.0 mapped to fraction_selected
- **Features:**
  - Colorbar showing "Fraction of iterations selected"
  - Only shows genes selected in >1 iteration
  - Optional `top_n` parameter to limit display
  - Sorted ascending (most-selected gene at top)
- **Output:** `{output_dir}/plots/gene_frequency.png`

**Font Sizes:**
- Title: 20pt
- Axis labels: 18pt
- Tick labels: 16pt
- Colorbar label: 16pt

---

#### 2. Gene Overlap Plot
**Function:** `plot_gene_overlap(gene_df, output_dir, n_iterations, png_dpi=600)`

- **Layout:** Square (10×10 inches) grouped bar chart
- **Purpose:** Compares gene stability by selection method (RF vs NMF pool)
- **Colors:**
  - RF pool: Orange (`#FF9800`)
  - NMF pool: Green (`#4CAF50`)
- **Features:**
  - X-axis: Selection frequency (1, 2, ..., n_iterations)
  - Y-axis: Number of genes
  - Bar labels show count + percentage (e.g., "45\n(22.5%)")
  - Side-by-side bars for each frequency
- **Output:** `{output_dir}/plots/gene_overlap.png`

**Font Sizes:**
- Title: 20pt (bold)
- Axis labels: 18pt
- Tick labels: 16pt
- Bar value labels: 14pt (bold)
- Legend: 16pt

**Optimization:** Square aspect ratio (10×10) as requested for publication quality.

---

#### 3. Metrics Summary Plots
**Function:** `plot_metrics_summary(metrics_df, output_dir, png_dpi=600)`

- **Layout:** Side-by-side horizontal bar charts (MSE left, ExpVar right)
- **Purpose:** Shows mean ± std per cell type across iterations
- **Color:** Blue (`#2196F3`)
- **Features:**
  - Error bars showing standard deviation
  - Sorted by ExpVar descending
  - Separate plots for per-celltype and global analysis
- **Outputs:**
  - `{output_dir}/plots/stability_metrics_summary.png` (per-celltype)
  - `{output_dir}/plots/stability_metrics_global_summary.png` (global)

**Figure sizes:**
- Per-celltype: `(16, max(8, n_celltypes × 0.4))` inches
- Global: `(12, 6)` inches

**Font Sizes:**
- Suptitle: 22pt
- Subplot titles: 20pt
- Axis labels: 18pt
- Tick labels: 16pt

---

#### 4. Aggregate Metrics Plot
**Function:** `plot_aggregate_metrics_summary(metrics_df, output_dir, png_dpi=600)`

- **Layout:** Side-by-side bar charts (14×7 inches)
- **Purpose:** Shows macro and weighted average metrics with mean ± std
- **Color:** Blue (`#2196F3`)
- **Features:**
  - Left panel: MSE (Macro, Weighted)
  - Right panel: ExpVar (Macro, Weighted)
  - Error bars with capsize=5
- **Output:** `{output_dir}/plots/stability_aggregate_metrics.png`

**Font Sizes:**
- Suptitle: 22pt (bold)
- Subplot titles: 20pt
- Axis labels: 18pt
- Tick labels: 16pt

---

#### 5. Feature UMAPs
**Function:** `plot_feature_umaps(adata, all_genes, celltype_col, output_dir, png_dpi=600)`

- **Layout:** Grid of UMAP plots (16 genes per grid, 4 columns)
- **Purpose:** Visualize expression patterns of selected genes
- **Features:**
  - Cell-type overview UMAP (first)
  - Batched gene feature plots
  - Auto-computes UMAP if not present (PCA → neighbors → UMAP)
- **Outputs:**
  - `{output_dir}/plots/feature_plots/celltype_umap.png`
  - `{output_dir}/plots/feature_plots/feature_grid_00.png`
  - `{output_dir}/plots/feature_plots/feature_grid_01.png`
  - ...

**DPI:** Increased from 150 → 600 for publication quality

---

### Publication-Quality Improvements

All stability plots have been optimized for publication in high-impact journals:

| Aspect | Original | Optimized | Increase |
|--------|----------|-----------|----------|
| **DPI** | 200 | 600 | 200% |
| **Titles** | 10-12pt | 20pt | 67-100% |
| **Axis Labels** | 9-11pt | 18pt | 64-100% |
| **Tick Labels** | 7-10pt | 16pt | 60-129% |
| **Legends** | 10pt | 16pt | 60% |
| **Value Labels** | 8pt | 14pt | 75% |
| **Colorbar Labels** | 8pt | 16pt | 100% |

**Special optimizations:**
- `plot_gene_frequency`: Square (10×10), reduced width from original (7×variable)
- `plot_gene_overlap`: Square (10×10), increased from original (12×6)

---

### Usage Example

```python
from _stability_plots import (
    plot_gene_frequency,
    plot_gene_overlap,
    plot_metrics_summary,
    plot_aggregate_metrics_summary,
    plot_feature_umaps,
)

# After running stability analysis
plot_gene_frequency(gene_df, output_dir, n_iterations=5)
plot_gene_overlap(gene_df, output_dir, n_iterations=5)
plot_metrics_summary(metrics_df, output_dir)
plot_aggregate_metrics_summary(metrics_df, output_dir)

# Optional: UMAP feature plots for selected genes
selected_genes = gene_df[gene_df["n_selected"] >= 1]["gene"].tolist()
plot_feature_umaps(adata, selected_genes, "cell_type", output_dir)
```

---

### Usage

Callers import these functions directly from `_stability_plots.py` (see the Usage Example above)
and pass in their own `gene_df`/`metrics_df` tables — there is no dedicated CLI wrapper for this
module; any `Analysis-scripts/` script producing stability data in the expected formats can call
these functions to generate the plots.

---

### Color Constants

Stability-specific colors defined in `_constants.py`:

```python
STABILITY_RF_COLOR = "#FF9800"      # Orange (RF pool)
STABILITY_NMF_COLOR = "#4CAF50"     # Green (NMF pool)
STABILITY_METRIC_COLOR = "#2196F3" # Blue (for metric bars)
```

---

### Key Functions Reference

#### `plot_gene_frequency()`
**Purpose:** Horizontal bar chart of gene selection frequency
**Input:** gene_df with n_selected, fraction_selected
**Output:** gene_frequency.png
**Styling:** Blues colormap, square (10×10)

#### `plot_gene_overlap()`
**Purpose:** Grouped bar chart by selection method
**Input:** gene_df with n_rf_pool, n_nmf_pool
**Output:** gene_overlap.png
**Styling:** Orange/green bars, square (10×10)

#### `plot_metrics_summary()`
**Purpose:** Mean ± std MSE/ExpVar per celltype
**Input:** metrics_df with iteration, mse, expvar
**Output:** stability_metrics_summary.png, stability_metrics_global_summary.png
**Styling:** Blue bars with error bars

#### `plot_aggregate_metrics_summary()`
**Purpose:** Macro/weighted aggregate metrics
**Input:** metrics_df with macro_mse, weighted_mse, etc.
**Output:** stability_aggregate_metrics.png
**Styling:** Blue bars, side-by-side layout (14×7)

#### `plot_feature_umaps()`
**Purpose:** UMAP visualization of selected genes
**Input:** AnnData object, gene list
**Output:** celltype_umap.png, feature_grid_*.png
**Styling:** Scanpy defaults, 600 DPI

---

## Contact and Support

For issues, questions, or contributions related to the Plotting-module, please refer to the main project documentation.

**Module version:** 2.0.0
