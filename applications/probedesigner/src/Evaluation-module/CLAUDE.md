# Evaluation Module

Version: 2.0.0

Panel-quality evaluation. Entry point: `run_evaluation.py` (see "Command-Line Interface"
below); the other six scripts (`run_pca_evaluation.py`, `run_ridge_evaluation.py`,
`run_ica_evaluation.py`, `run_lognorm_evaluation.py`, `run_scvi_evaluation.py`,
`run_tangram_check.py`) are independent, standalone alternatives. NMF fitting delegates to `nico2_lib`'s `NmfPredictor` (RecoVar pins the kwargs
it forwards — identical to the Selection module's, see §NMF); the core metric formulas
(`calculate_mse`, `calculate_explained_variance`) delegate to `nico2_lib.metrics`. "nico2" is
that dependency, not a module directory. There is **no** `Evaluation-module-nico2/`; this is
the only evaluation module.

This module provides comprehensive tools to evaluate the quality of selected probe panels for spatial transcriptomics experiments using four complementary approaches.

## Overview

The Evaluation module assesses probe panel quality through:

1. **Baseline Evaluation**: Clustering quality (ARI/NMI), neighborhood preservation (kNN overlap), and cell-type identification (decision tree + F1 variants) — all computed together in one stage (`run_baseline_evaluation()`), gated by `--evaluation_type baseline`/`all`.
2. **Variability Evaluation**: NMF reconstruction metrics (global + per-celltype), gated by `--evaluation_type variability`/`all`.
3. **Biology Evaluation**: Enrichr pathway enrichment (GO Biological Process, KEGG, Reactome) per panel, gated by `--evaluation_type biology`/`all`.
4. **Reconstruction Check**: Tangram-based full-transcriptome reconstruction — gated **independently** by `--include_tangram`, not by `--evaluation_type` (see "Two independent gating axes" below).

Six *separate, standalone* scripts (`run_pca_evaluation.py`, `run_ridge_evaluation.py`, `run_ica_evaluation.py`, `run_lognorm_evaluation.py`, `run_scvi_evaluation.py`, `run_tangram_check.py`) provide alternative/supplementary reconstruction-evaluation methods; see "Standalone scripts" below.

## Directory Structure

```
Evaluation-module/
├── __init__.py                 # Package init — re-exports flat module names (fixed #59)
├── run_evaluation.py           # Main CLI orchestrator (preprocess / baseline / biology / variability / tangram)
├── _preprocessing.py           # Panel + reference preprocessing (subset, normalize, PCA, Leiden, kNN)
├── _splits.py                  # Train/test split generation (generate_evaluation_splits)
├── _filters.py                 # Dataset filename parsing / filtering (--strategies, --probeset_sizes, ...)
├── _constants.py               # Module constants and defaults (some not wired to CLI — see below)
├── _clustering.py              # Baseline metrics: clustering, kNN preservation, cell-type ID
├── nmf.py                      # Built-in NMF reconstruction (nmf_reconstruction, nmf_reconstruction_by_celltype)
├── _tangram.py                 # Tangram reconstruction (used by both run_evaluation.py and run_tangram_check.py)
├── _pathway.py                 # Enrichr pathway enrichment (biology evaluation type)
├── metrics.py                  # Shared metric calculations (calculate_*)
│
├── pca.py                      # Standalone: PCA reconstruction evaluation
├── run_pca_evaluation.py       # CLI for pca.py — NOT called by run_evaluation.py
├── ridge.py                    # Standalone: Ridge-regression reconstruction evaluation
├── run_ridge_evaluation.py     # CLI for ridge.py — NOT called by run_evaluation.py
├── ica.py                      # Standalone: ICA (FastICA) reconstruction evaluation
├── run_ica_evaluation.py       # CLI for ica.py — NOT called by run_evaluation.py
├── nmf_lognorm.py              # Standalone: NMF reconstruction compared in log-normalized space
├── run_lognorm_evaluation.py   # CLI for nmf_lognorm.py — NOT called by run_evaluation.py
├── scvi_eval.py                # Standalone: scVI reconstruction evaluation (global only, raw counts)
├── run_scvi_evaluation.py      # CLI for scvi_eval.py — NOT called by run_evaluation.py
└── run_tangram_check.py        # Standalone Tangram CLI (own exit codes, own split default) — NOT called by run_evaluation.py
```

## Key Features

### Baseline Metrics
- **Clustering Quality**: ARI (Adjusted Rand Index) and NMI (Normalized Mutual Information)
  - Per-cell-type Jaccard/purity vs. best-matching Leiden cluster (`compute_celltype_jaccard_vs_leiden`)
- **Neighborhood Preservation**: kNN overlap between full transcriptome and probe panel
  - Per-cell-type mean Jaccard scores (via `cell_annotations` parameter)
- **Cell-type Identification**: Accuracy of cell-type assignments (decision tree classifier)
  - Overall accuracy and macro-averaged F1 (per-class precision/recall/F1/support also stored)
  - Uses its **own** internal 80/20 stratified split (`split_train_test_sets`, ported from Spapros, seed 42) — independent of `generate_evaluation_splits()`
  - Classifier is a single `DecisionTreeClassifier(max_depth=celltype_clf_max_depth, random_state=42)` — `max_depth` (default **15**, `DEFAULT_CELLTYPE_CLF_MAX_DEPTH`, overridable via `--celltype_clf_max_depth` / `evaluate_celltype_identification(celltype_clf_max_depth=...)`) is **fixed for every panel in a run**, not tuned per panel; the metric row still carries the depth as `classifier_max_depth` for the record. Raised from 10 after the eval-tree depth sweep (`docs/doc-pipeline/eval-celltype-clf-maxdepth-sweep.md`) found depth 10 too shallow to separate several lineage-adjacent cell types — treat it as a measuring instrument and hold it constant across any panels being compared.

### Variability Metrics (built into `run_evaluation.py`)
- **MSE** (Mean Squared Error) and **Explained Variance** (custom `1 - MSE/Var` formula, see Implementation Notes) — reported **absolutely** for the probe-constrained reconstruction (`*_test_probe`) and for an independent full-gene NMF "oracle" baseline (`*_test_baseline`), on train and test splits. There are no probe/baseline ratio columns (`mse_ratio`, `expvar_ratio`, `generalization_gap`) — compare `*_test_probe` against `*_test_baseline` directly; the baseline still drives the "Full NMF Baseline" reference lines in the plots.
- Supports both **macro-averaged** (per cell-type) and **weighted** aggregation
- H (gene loadings) is fixed from a full-gene NMF fit on training data; only cell scores W are re-solved (NNLS) from probe genes — never a fresh unconstrained NMF fit on the probe subset

### Reconstruction Metrics (Tangram, optional)
- Probabilistic cell-mapping reconstruction, gated by `--include_tangram` (independent of `--evaluation_type`)
- No baseline ratio — reports absolute MSE/ExpVar/RMSE/MAE/R²/Pearson
- The **only** evaluation method where a held-out train/test split is optional rather than enforced (module docstring calls full-dataset mapping "a method-level limitation")
- Requires `tangram-sc`; degrades to a skip if not installed

### Biology Metrics (Pathway Enrichment)
- **Enrichr-based**: Runs each panel's gene list against GO Biological Process, KEGG, and
  Reactome (`_pathway.py`, requires `gseapy` — not installed by default, degrades to a
  warning + skip if missing)
- Saves significant pathways (adjusted p < 0.05) and top-12-per-library bar charts
- `--pathway_organism` (`"human"` or `"mouse"`) now selects the default library set
  (`_pathway.PATHWAY_LIBRARIES_BY_ORGANISM`): GO BP 2023 + Reactome 2022 for both, with
  the organism-matched KEGG (`KEGG_2021_Human` vs `KEGG_2019_Mouse`). `--pathway_libraries`
  overrides the whole set. "Significant" uses Enrichr's own per-library BH `Adjusted P-value`
  (read directly, not recomputed), pooled across libraries at p < 0.05.
- **Per cell type (on by default; `--no-pathway_per_celltype` to disable)**: in addition to
  the whole-panel run, runs Enrichr once per cell type into
  `<panel>/per_celltype/<safe_celltype>/`. Two gene-set modes via
  `--pathway_celltype_gene_source`:
  - `informative` (default): the panel genes whose gene-list CSV `informative_celltypes`
    column lists that cell type. **Needs `--gene_lists_dir` or `--gene_list_files_txt`**
    so each preprocessed panel can be mapped back to its source CSV
    (`load_all_gene_lists(..., return_paths=True)`); panels with no mapping are logged
    and skipped for the per-cell-type pass only. A single-strategy `rf_deg`/`rf_simple`
    file (which has no `informative_celltypes` or `contributing_celltypes` column) falls
    back to its `rf_celltype` column instead; older gene lists without any of these fall
    back to `rf_contributing_celltypes` + `celltype` (NMF genes' `contributing_celltypes`
    is not parsed — regenerate the list for full coverage).
  - `all_panel`: the full panel gene list for every cell type in `adata.obs[--celltype_col]`.
  Cell types with fewer than `--pathway_celltype_min_genes` (default 5) genes are recorded
  in the summary with a `skipped_reason` and not queried. Enrichr background is the default
  (unchanged from the global run) in both modes.

## Usage

### `__init__.py`

`__init__.py` prepends its own directory to `sys.path` and re-exports from the real flat
module names (`from _clustering import …`, `from metrics import …`, `from nmf import …`,
`from pca`/`ridge import …`), matching how every sibling module imports.

The directory is still hyphenated (`Evaluation-module`), so `import evaluation` is not
possible without a packaging rename (`src/recovar/…`). The flat-import pattern below is the
recommended way in; `__init__.py` makes the package self-consistent and importable via
`spec_from_file_location` / an added `sys.path` entry.

### Actually-working import pattern

```python
import sys
sys.path.insert(0, "/path/to/Code/RecoVar/Evaluation-module")

from _clustering import evaluate_clustering_quality, evaluate_neighborhood_preservation
from nmf import nmf_reconstruction, nmf_reconstruction_by_celltype
from metrics import calculate_mse, calculate_explained_variance
from _splits import generate_evaluation_splits
```

### Command-Line Interface

The main script [run_evaluation.py](run_evaluation.py) has **two independent gating axes**, not one:

- `--mode {preprocess, evaluate, both}` (default `evaluate`) — `preprocess` writes `full_transcriptome.h5ad` + per-panel `.h5ad` files and **returns immediately without evaluating anything**, even if other evaluation flags are set; `both` runs preprocessing then evaluation in the same invocation.
- `--evaluation_type {baseline, variability, biology, all, tangram_only}` (default `all`) — `baseline`/`variability`/`biology` each independently trigger their stage (for `all`, all three run — they're independent `if` checks, not `elif`). **`tangram_only` matches none of the three baseline/variability/biology checks**, so by itself it runs nothing.
- `--include_tangram` (flag, default off) — gates the Tangram stage **independently of `--evaluation_type`**. This means `--evaluation_type tangram_only` alone does nothing; you need `--evaluation_type tangram_only --include_tangram` to run *only* Tangram, or `--evaluation_type all --include_tangram` to run everything including Tangram.

#### 1. Preprocessing Only
```bash
python run_evaluation.py --mode preprocess \
    --input_file data.h5ad \
    --gene_lists_dir Selected-panels/ \
    --preprocessed_dir preprocessed/ \
    --output_dir Evaluation-Results/
```

#### 2. Evaluation Only
```bash
python run_evaluation.py --mode evaluate \
    --input_file preprocessed/full_transcriptome.h5ad \
    --preprocessed_dir preprocessed/ \
    --output_dir Evaluation-Results/ \
    --evaluation_type all
```

#### 3. Both (Preprocessing + Evaluation, including Tangram)
```bash
python run_evaluation.py --mode both \
    --input_file data.h5ad \
    --gene_lists_dir Selected-panels/ \
    --preprocessed_dir preprocessed/ \
    --output_dir Evaluation-Results/ \
    --evaluation_type all \
    --include_tangram
```

### Dry-Run Mode
```bash
python run_evaluation.py --mode both --dry_run ...
```

Prints resolved configuration without executing the pipeline.

### `run_evaluation.py` argparse defaults

Only `--celltype_col` (→ `_constants.DEFAULT_CELLTYPE_COLUMN` = `"cluster"`),
`--celltype_clf_max_depth` (→ `_constants.DEFAULT_CELLTYPE_CLF_MAX_DEPTH` = `15`), and
`--tangram_n_epochs` (→ `_constants.DEFAULT_TANGRAM_N_EPOCHS` = `1000`) are wired to
`_constants.py`. `--n_components` (`5`), `--random_state` (`42`), `--n_neighbors` (`15`),
`--dim_reduction_preprocess` (`"both"`), `--n_splits` (`5`) are hard-coded literals in the
argparse block, *not* read from the matching `DEFAULT_*` constants. There is no
`--split_mode` / `--test_size`. Pass `--celltype_col` explicitly if your data's cell-type
column is not literally `"cluster"`.

### Additional `run_evaluation.py` flags not shown in the examples above

- `--celltype_clf_max_depth` (default `DEFAULT_CELLTYPE_CLF_MAX_DEPTH` = 15) — `max_depth`
  of the cell-type-classification decision tree, fixed identically for every panel in the
  run. This is a measuring instrument, not a panel property — only override it
  deliberately, and hold it constant across any panels/benchmarks being compared.
- `--nmf_objective` (choices `auto`/`frobenius`/`kl`, default `auto`) — NMF factorization
  objective for the variability stage's `NmfPredictor` calls; `auto` derives from
  `--nmf_counts_input` (raw → mu/Kullback-Leibler, lognorm → cd/Frobenius, see "NMF" below).
  This is a deliberate production default: raw-count runs (the default `--nmf_counts_input`)
  now factorize with `mu`/KL unless overridden.
- `--expvar_mode` (choices = `metrics.EXPVAR_MODES`, default `global_mean`) — the
  explained-variance aggregation formula for the unsuffixed `expvar_*` columns.
- `--expvar_modes` (list) — extra modes to *additionally* report as
  `expvar_*_<mode>` columns.
- `--pathway_libraries` / `--pathway_organism` (`human` / `mouse`) / `--pathway_sleep` /
  `--pathway_top_n` — the biology (Enrichr) stage.
- `--pathway_per_celltype` (on by default; `--no-pathway_per_celltype` disables) /
  `--pathway_celltype_gene_source` (`informative` / `all_panel`) /
  `--pathway_celltype_min_genes` — per-cell-type Enrichr (see "Biology Metrics" above).
- `--reference_blacklist_patterns` / `--reference_disable_default_blacklist` — blacklist
  filtering applied to the reference transcriptome before evaluation.

### Standalone scripts — not called by, and don't call, `run_evaluation.py`

`run_pca_evaluation.py`, `run_ridge_evaluation.py`, `run_ica_evaluation.py`, `run_lognorm_evaluation.py`, and `run_tangram_check.py` are fully independent CLI entry points. None of them import `run_evaluation.py`, and it imports none of them — confirmed by grep in both directions. Each duplicates its own config dataclass, argparse block, and `_load_preprocessed_datasets` helper, but all five **require `run_evaluation.py --mode preprocess` to have already run** (they load `preprocessed_dir/full_transcriptome.h5ad`, which only `--mode preprocess`/`both` produce) and all five reuse the shared `_splits.generate_evaluation_splits()`.

| Script | Alternative to | Baseline convention | Input space requirement |
|---|---|---|---|
| `run_pca_evaluation.py` | the built-in NMF variability stage | absolute probe MSE/ExpVar vs. an absolute full-gene PCA baseline at the same rank — an **oracle** fit independently on each split (train baseline: fit+evaluate on train; test baseline: a *separate* PCA fit directly on test data, not a transform of the train-fitted basis), matching `nmf.py`'s test-baseline convention (no ratio — #38) | requires pre-normalized `adata.X` (errors if it looks like raw counts) |
| `run_ridge_evaluation.py` | the built-in NMF variability stage | absolute probe MSE/ExpVar vs. an absolute per-gene training-mean-predictor baseline (no ratio — #38) | requires **both** `layers['counts']` (raw) and pre-normalized `adata.X` — evaluates in both spaces simultaneously |
| `run_ica_evaluation.py` | the built-in NMF variability stage | absolute probe MSE/ExpVar vs. an absolute full-gene FastICA baseline at the same rank, same independent-oracle-per-split convention as `run_pca_evaluation.py` (no ratio — #38); `n_components` is a hard hyperparameter, not a variance-ranked truncation point like PCA's | requires pre-normalized `adata.X` (errors if it looks like raw counts); a cell type whose FastICA fit doesn't converge is skipped (`skip_reason="ica_failed_to_converge"`) rather than failing the run |
| `run_lognorm_evaluation.py` | the built-in NMF variability stage | same NMF procedure as `nmf.py`, but MSE/ExpVar computed after log-normalizing reference + reconstruction + baseline with a shared `target_sum` (no ratio — #38) | requires `layers['counts']` (raw) |
| `run_scvi_evaluation.py` | the built-in NMF variability stage | absolute probe MSE/ExpVar vs. an absolute full-gene scVI baseline at the same latent dimension — an **oracle** fit independently on each split (train/test), matching the other standalone scripts' convention (no ratio — #38). **Global only — no per-celltype variant**: `nico2_lib`'s `ScviPredictor` retrains a full VAE from scratch on every `.predict()` call with no caching, so per-celltype evaluation would multiply an already-expensive cost by the number of cell types. Results are **not reproducible run-to-run** — `ScviPredictor` exposes no seed control. | requires `layers['counts']` (raw) — scVI is a negative-binomial count model, not a pre-normalized-expression method; expect `n_folds * (2 + 2*n_panels)` independent VAE trainings, by far the most expensive of the five standalone methods |
| `run_tangram_check.py` | `run_evaluation.py --include_tangram` | none (absolute metrics) | own exit codes (0/1/2); uses a single held-out partition (stratified fold 0) unless `--no_split` |

`run_ridge_evaluation.py --use_ridgecv` is an `argparse.BooleanOptionalAction` (default
enabled) — pass `--no-use_ridgecv --alpha <value>` to fit a single fixed-alpha `Ridge`
instead of `RidgeCV`.

## API Reference

### Clustering Functions ([_clustering.py](_clustering.py))

```python
from _clustering import (
    evaluate_clustering_quality,
    evaluate_neighborhood_preservation,
    evaluate_celltype_identification,
)

# Each function takes a dict of {name: preprocessed_adata} and a reference key.
# 'sets' is built by preprocessing each panel + the full transcriptome reference.
sets = {
    "full_transcriptome": adata_full,
    "panel_A": adata_panel_A,
}

# Compute ARI/NMI for clustering; pass celltype_col for per-CT Jaccard/purity rows
results = evaluate_clustering_quality(
    sets, reference_key="full_transcriptome", celltype_col="new_annot"
)
# Returns DataFrame with ARI/NMI rows + optional per-CT rows (columns: celltype, jaccard, purity)

# Compute kNN overlap at k=5,10,15,20,30,50; pass celltype_col for per-CT Jaccard rows
overlap = evaluate_neighborhood_preservation(
    sets, reference_key="full_transcriptome", celltype_col="new_annot"
)

# Evaluate cell-type accuracy (decision tree classifier); reports overall accuracy + macro F1
accuracy = evaluate_celltype_identification(sets, reference_key="full_transcriptome", celltype_col="new_annot")
```

#### Per-cell-type Jaccard vs. Leiden
```python
from _clustering import compute_celltype_jaccard_vs_leiden

# For each true cell type, finds the best-matching Leiden cluster and reports
# Jaccard similarity and purity.
results = compute_celltype_jaccard_vs_leiden(true_labels, cluster_labels)
# Returns: {celltype: {"jaccard": float, "purity": float, "n_cells": int, "best_cluster": str}}
```

### NMF Functions ([nmf.py](nmf.py))

```python
from nmf import nmf_reconstruction, nmf_reconstruction_by_celltype
from _splits import generate_evaluation_splits

# Generate splits once — shared by NMF and Tangram
splits = generate_evaluation_splits(adata, celltype_col='celltype')
train_idx, test_idx, per_celltype_splits = splits[0]

# Global evaluation
results = nmf_reconstruction(
    adata_full, probeset_genes, n_components=10,
    train_idx=train_idx, test_idx=test_idx,
)

# Per cell-type evaluation (per_celltype_splits required)
results = nmf_reconstruction_by_celltype(
    adata_full, probeset_genes, celltype_column='celltype', n_components=5,
    per_celltype_splits=per_celltype_splits,
)
```

### Metric Functions ([metrics.py](metrics.py))

```python
from metrics import (
    calculate_mse,
    calculate_explained_variance,
    calculate_macro_mse,
    calculate_macro_explained_variance,
    calculate_weighted_mse,
    calculate_weighted_explained_variance,
)
```

### Split Generation ([_splits.py](_splits.py))

```python
from _splits import generate_evaluation_splits

# Returns a list of n_splits (train_idx, test_idx, per_celltype_splits) tuples —
# always a cell-type-stratified StratifiedKFold. There is no --split_mode / --test_size.
# A caller wanting one held-out split takes fold 0.
splits = generate_evaluation_splits(
    adata, celltype_col='celltype', n_splits=5, random_state=42,
)
```

## Input Requirements

### AnnData Files

- **AnnData format** (`.h5ad`)
- **Gene expression data**: Stored in `adata.X` or `adata.layers['counts']`
- **Cell-type annotations**: Stored in `adata.obs[celltype_col]`
- **Gene names**: Stored in `adata.var_names`

### Gene List Formats

The evaluation module supports multiple CSV formats for gene lists (in priority order):

1. **`{strategy}_panel_information.csv` / `RecoVar_panel_information.csv`** (recommended,
   current Selection-module format)
   - **Generated by**: Selection-module — single strategies write
     `deg_only_panel_information.csv`, `hvg_panel_information.csv`,
     `random_panel_information.csv`, `rf_simple_panel_information.csv`,
     `rf_deg_panel_information.csv`, or `nmf_panel_information.csv`/
     `pca_panel_information.csv` (`dimred_only`); combination strategies (`RecoVar`,
     `RecoVar_PCA`) write `RecoVar_panel_information.csv`.
   - **All variants carry an `in_panel` column and are handled by
     `load_gene_list_from_csv()`'s `('final_selection', 'in_panel', 'selected_final')`
     fallback chain**: only rows where the first present column of that chain is `True`
     are loaded. Single-strategy files list every candidate gene the builder tracked;
     `RecoVar_panel_information.csv` lists every gene considered during combination
     (union of the RF and dimred candidate pools plus the final panel) — panel genes are
     listed first but the file is **not** panel-only, unlike the historical combination
     output.
   - **Example (`RecoVar_panel_information.csv`)**:
     ```csv
     gene,in_panel,gene_source,informative_celltypes,primary_celltype,secondary_celltypes_nmf,n_celltypes_nmf,gap_filled,gap_fill_strategy,mean_expression
     Cd3d,True,rf_deg,T cells,T cells,,,False,,1.2
     Cd79a,True,dimred,B cells|T cells|NK cells,B cells,"B cells, T cells, NK cells",3,False,,0.9
     Mt-Co1,False,random forest,,,,,,False,,2.1
     ```
   - **Example (`rf_deg_panel_information.csv`)**:
     ```csv
     gene,in_panel,rank,rf_celltype,rf_celltype_scores,mean_expression
     Cd3d,True,1,T cells,"{""T cells"": 0.82, ""NK cells"": 0.18}",1.2
     ```
   - Full per-column tables for every file: `Selection-module/CLAUDE.md`'s "Output Files"
     section.

2. **Simple list** (basic format)
   - Single column of gene names
   - May or may not have header
   - First column is used as gene names

## Output Files

### Preprocessing Mode
- `full_transcriptome.h5ad`: Reference dataset with PCA embeddings, Leiden clusterings, and kNN graphs
- `panel_<name>.h5ad`: Subsetted probe panels with PCA, Leiden clusterings, and kNN graphs
  - **Note:** NMF preprocessing is not performed; only PCA embeddings are computed.

### Evaluation Mode
Each stage writes to its own top-level directory: `Baseline-Evaluation/`, `Biology-Evaluation/`, `Variability-Evaluation/`, `Tangram-Evaluation/` (the latter only if `--include_tangram`).
- `Baseline-Evaluation/`: per-dataset clustering (ARI/NMI), kNN overlap, and cell-type
  accuracy CSVs (rows with a `celltype` column hold per-cell-type Jaccard/purity or kNN Jaccard)
- `Variability-Evaluation/results/nmf/[per_fold/]<panel>.csv`: per-fold and fold-aggregated
  absolute probe + baseline MSE / explained-variance columns (no ratio columns — #38)
- The standalone scripts (`run_pca_evaluation.py`/`run_ica_evaluation.py`/`run_ridge_evaluation.py`/
  `run_scvi_evaluation.py`) all share the **same** `Variability-Evaluation/results/` root NMF
  writes to, differing only by leaf subdir name: `PCA/`, `ICA/`, `ridge-regression/`, `SCVI/`
  respectively (e.g. `Variability-Evaluation/results/SCVI/[per_fold/]<panel>.csv` — global rows
  only, no `celltype` column, since no per-celltype variant exists for scVI; see the
  standalone-scripts table above for the cost/reproducibility caveats). `run_lognorm_evaluation.py`
  is the one exception, writing to its own separate `Variability-Evaluation-Lognorm/` root.
- `Biology-Evaluation/results/`: per panel `<panel>/significant_pathways.csv` +
  `<panel>/top_pathways_<library>.svg`, plus `pathway_summary.csv` (dataset, n_genes,
  n_significant). With `--pathway_per_celltype`: also `<panel>/per_celltype/<safe_celltype>/`
  (same two file types) and `pathway_per_celltype_summary.csv` (dataset, celltype_col,
  gene_source, celltype, n_genes, n_significant, skipped_reason)
- `Tangram-Evaluation/`: per-panel `global/<panel>.csv` + `per_celltype/<panel>.csv`
  (+ `per_fold/…`) reconstruction CSVs, if `--include_tangram`. **No JSON summary** —
  `tangram_summary.json` is written only by the standalone `run_tangram_check.py`, not by
  `run_evaluation.py`.

## Dependencies

- `scanpy >= 1.9`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `anndata`
- `tqdm`
- `nico2_lib` — provides `NmfPredictor` (NMF fit/solve wrapper; RecoVar pins its kwargs, see §NMF), `PcaPredictor`/`FastIcaPredictor` (used by `pca.py`/`ica.py`), `ScviPredictor` (used by `scvi_eval.py` — requires `scvi-tools`/`torch`, pulled in transitively via `nico2_lib`), `nico2_lib.metrics` (MSE / explained-variance formulas), and Tangram unfiltered wrappers
- `psutil` (optional, for memory monitoring)
- `tangram-sc` (optional, for reconstruction)
- `gseapy` (optional, for biology/pathway enrichment — not installed in `SpaprosProbeDesign` by default; `--evaluation_type biology`/`all` logs a warning and skips if missing)

## Implementation Notes

### Train/Test Split Discipline

Splits are generated **once** via `generate_evaluation_splits()` in `_splits.py` and forwarded to both NMF and Tangram. This ensures both methods evaluate on the same held-out cells.

- `generate_evaluation_splits()` returns `[(train_idx, test_idx, per_celltype_splits), ...]`
- `per_celltype_splits` contains position arrays derived by intersecting the global fold indices with per-cell-type positions — they are strict subsets of the global partition
- `nmf_reconstruction_by_celltype` requires `per_celltype_splits`; passing `None` raises `ValueError`
- No internal fallback splitting exists anywhere in the module — all splits must come from `generate_evaluation_splits()`, **except** `evaluate_celltype_identification()`, which uses its own separate internal 80/20 stratified split (Spapros-derived), and Tangram, where a split is optional at all

### NMF
- Uses `NmfPredictor` from `nico2_lib.predictors._nmf._nmf_pred` — a `.fit()` / `.predict()` wrapper around `sklearn.decomposition.NMF` (Frobenius loss)
- **Config is data-space-driven (`--nmf_objective`), shared with the Selection module.** Every `NmfPredictor(...)` here is built from `Utility-module/_nmf_objective.resolve_nmf_objective(nmf_counts_input, nmf_objective)`: `--nmf_objective auto` (default) derives the objective from `--nmf_counts_input` — `raw` (default) → `solver="mu"`, `beta_loss="kullback-leibler"`, `init="nndsvda"`, `max_iter=2000`; `lognorm` → `solver="cd"`, `beta_loss="frobenius"`, `init="nndsvd"`, `max_iter=1000` (byte-identical to the historical pinned `_constants.NMF_PREDICTOR_FIXED_KWARGS`). `--nmf_objective frobenius`/`kl` force either objective regardless of input. Only `n_components` and `random_state` vary further per call. Selection resolves identically via its own `--nmf_objective` flag, so the two sides factorize the same way for a given count-input choice; `_constants.NMF_PREDICTOR_FIXED_KWARGS` remains the byte-identical Frobenius/cd fallback between `Selection-module/_constants.py` and `Evaluation-module/_constants.py`. (`NmfPredictor.fit()`/`.predict()` honour `solver`/`beta_loss`/`init`/`max_iter`; `alpha_*` remain no-ops, so those entries document intent only.)
- Full fit: `predictor = NmfPredictor(...).fit(A)` — `predictor.ref_embedding` ≡ `W`, `predictor.h_reference` ≡ `H`
- Constrained-W solve (probe genes only, fixed H): `W, A_recon = predictor.predict(A_P, indexer=probe_indices)` — equivalent to the old `update_H=False` fixed-H solve, and returns the reconstructed matrix directly (no manual `W @ H` step)
- The built-in variability stage's **test-set baseline** is obtained from an independently fresh NMF fit on the test cells (not a transform of the train fit). `pca.py` and `ica.py` match this same oracle convention (a separate PCA/ICA fit directly on test data for the test baseline) so `expvar_test_baseline`/`mse_test_baseline` mean the same thing across the NMF, PCA, and ICA evaluation scripts — the probe-based reconstruction columns are unaffected and always use the train-fixed basis, as they must.

### Tangram
- Imports unfiltered wrappers from `nico2_lib.predictors._tangram._tangram_pred`:
  - `pp_adatas_unfiltered` — skips `filter_genes` so zero-expressed genes are retained
  - `map_cells_to_space` (aliased as `_map_cells_to_space_unfiltered`) — removes the zero-value gene assertion
  - `project_genes_unfiltered` — skips `filter_genes` on the sc reference
- Effect: **all shared genes** (including zero-expressed ones) are reconstructed, unlike standard `tg.pp_adatas`
- Tangram supports **optional train/test splits** via `train_idx`/`test_idx` parameters (same indices from `generate_evaluation_splits()` as used by NMF — not a separate fit on the full dataset); if omitted, all cells are used for both mapping and evaluation
- **Scoring conventions (fixed 2026-09-20; earlier Tangram results were invalid):**
  - *Target space follows `nmf_counts_input`* (default `"raw"` → `layers['counts']`, `"lognorm"` → `.X`), via `_reference_matrix()`. Tangram itself always maps on raw counts, but only on **private copies** (`.X` is swapped to `layers['counts']` inside `reconstruct_with_tangram`); the caller's AnnData keeps log-normalised `.X`. The old code built the reference from the original object's `.X`, i.e. scored raw-scale predictions against log-scale targets (ExpVar ≈ −1000).
  - *Per-cell scale:* the mapping is a per-training-cell softmax over spots, so each test spot receives ≈ `n_train/n_test` cells' worth of counts. Projected expression is divided by that factor ("one cell per spot"); the factor is stored as `ad_ge.uns["scale_divisor"]` and written as `scale_divisor` in the metrics (1.0 without a split). `mean_total_ref` / `mean_total_pred` (mean counts per cell) are written as sanity checks and should agree.
  - Any Tangram result produced before this fix (raw mode) is wrong and must be rerun.
  - *Gene-subset x mode grid (2026-09-21):* besides the legacy `mse` / `expvar` (= all genes, global mean, unchanged), `_calculate_reconstruction_metrics` writes `mse_<subset>` and `expvar_<subset>_<mode>` for subsets `all_genes` / `panel_genes_only` / `non_panel_genes_only` and modes `global_mean` / `variance_weighted_sum` (same naming as PCA/ICA/NMF/Ridge, minus the `test_probe` token), plus `macro_`/`weighted_` versions in the per-celltype `__summary__` row. Panel genes = `adata_subset` genes (case-insensitive).

### Preprocessing
- Only PCA is computed during panel preprocessing (`process_data_for_panel_evaluation`); NMF embeddings are no longer produced at this stage
- `adata.raw` is only set when `'counts'` is not already in `adata.layers` (memory optimization for large reference datasets)
- `preprocess_reference_dataset` handles three input cases: raw-count `.X` → creates `layers['counts']`; normalized `.X` with `adata.raw` present → restores counts from `adata.raw`; neither → raises `ValueError`
- This is distinct from the repo-wide `Preprocessing-module` (QC/normalization for selection/analysis) — this module's `_preprocessing.py` only subsets to a gene panel and computes evaluation-specific embeddings/clusterings on already-QC'd data

### Metrics
- **MSE**: delegates to `nico2_lib.metrics.mse_metric` — equivalent to `mean((X - X_recon)^2)` over all elements
- **Explained variance**: custom v2 formula `1 - MSE / Var(X_original)` (`nico2_lib.metrics.explained_variance_metric_v2`) — **not** `sklearn.metrics.explained_variance_score` (which uses variance of residuals, not MSE). Used identically across PCA, Ridge, NMF, NMF-lognorm, and Tangram for cross-method comparability.

## Notes

- **Memory Efficiency**: The module uses sparse matrices where possible and includes memory monitoring
- **String Handling**: Configured for pandas 2.x compatibility with ArrowStringArray
- **Caching**: Preprocessing results can be cached to avoid redundant computation

## Related Modules

- [Selection-module](../Selection-module/): Gene selection strategies
- [Preprocessing-module](../Preprocessing-module/): Panel and reference preprocessing

## Author

Helene Hemmer

## License

See project root for license information.
