# Selection Module

Version: 2.0.0

This module provides eight different gene selection strategies for spatial transcriptomics probe design, from simple baselines to sophisticated hybrid approaches.

## Overview

The Selection module implements a comprehensive suite of gene selection strategies:

- **deg_only**: Differential expression analysis
- **rf_simple**: Random forest classifier on all genes
- **rf_deg**: Random forest on DEG-filtered genes
- **RecoVar**: Hybrid random forest + NMF with gap-filling
- **RecoVar_PCA**: Hybrid random forest + PCA with gap-filling
- **dimred_only**: Pure dimensionality reduction (NMF/PCA)
- **hvg**: Highly variable genes (baseline)
- **random**: Random selection (baseline)

## Directory Structure

```
Selection-module/
├── __init__.py                      # Package initialization with public API
├── _constants.py                    # Strategy definitions and defaults
├── _gene_list_builder.py            # Core data structure for tracking gene provenance
├── _filtering.py                    # Blacklist / Xenium filtering
├── _deg_selection.py                # Differential expression analysis
├── _baseline_selection.py           # HVG and random selection
├── _rf_selection.py                 # Random forest-based selection
├── _dimred_selection.py             # NMF/PCA dimensionality reduction
├── _factor_aware.py                 # Factor-aware duplicate resolution
├── run_selection_pipeline.py        # Main pipeline orchestrator (CLI entry point)
├── run_single_selection.py          # Single-strategy executor
└── run_combination_selection.py     # Combination-strategy executor (RecoVar, RecoVar_PCA)
```

## Key Features

### Selection Strategies

#### Simple Strategies
1. **deg_only**: Select genes with highest differential expression across cell types
2. **hvg**: Select highly variable genes (Seurat method)
3. **random**: Random gene selection (baseline control)

#### Machine Learning
4. **rf_simple**: Random forest feature importance on all genes
5. **rf_deg**: Random forest on DEG-prefiltered genes

#### Dimensionality Reduction
6. **dimred_only**: Select genes with highest loadings in NMF/PCA factors

#### Hybrid Strategies
7. **RecoVar**: Random forest (25%) + NMF (75%) using strict-then-fill per-celltype selection
8. **RecoVar_PCA**: Random forest (25%) + PCA (75%) using strict-then-fill per-celltype selection

### Core Components

#### GeneListBuilder ([_gene_list_builder.py](./_gene_list_builder.py))
Unified data structure for tracking gene provenance through:
- Initial selection
- Filtering (blacklist, Xenium compatibility)
- Duplicate resolution

`mean_expression` is populated by `set_mean_expression()`, called from
`run_single_selection.py` with the same per-cell-type / global mean-expression data the
Xenium filter uses (empty when Xenium filtering is disabled). Combination-strategy
gap-filling is tracked outside `GeneListBuilder`, via the `gap_filled` /
`gap_fill_strategy` columns added in `run_combination_selection.py`. (The former
`replaced_by` / `replaces_gene` / `replacement_reason` columns and their dead setter were
removed — they were never populated and are superseded by `gene_source` /
`gap_fill_strategy`.)

Every selection function in this module (`select_degs`, `select_genes_with_rf`, `select_genes_from_nmf`/`select_genes_from_pca`, `run_single_selection`, `run_combination_selection`) returns a populated `GeneListBuilder` — it is the shared "lingua franca" object passed between strategies and the filtering/output layer.

#### Factor-Aware Processing ([_factor_aware.py](./_factor_aware.py))
Intelligent handling of duplicates for NMF/PCA-based selection:
- Within-celltype/within-factor duplicate resolution: assigns a gene to the factor where its `abs(loading)` is highest; backfills the vacated slot with the next-best candidate gene from that factor's pool
- Cross-celltype shared genes are **tracked but not resolved** — `gene_celltype_mapping` records every celltype that selected a gene in Phase 1, and `n_celltypes_per_gene` lets downstream steps prefer broadly-selected genes

#### Filtering Pipeline ([_filtering.py](./_filtering.py))
- **Blacklist filtering**: Remove unwanted gene patterns (default: `mt-`, `hsp`, `rps`, `rpl`, case-insensitive prefix match)
- **Xenium compatibility**: Filter to genes within an experimentally usable mean-expression range (celltype-aware or global)

## Dataset Preprocessing & Filtering

Selection-module does **not** perform normalization or QC itself. Full preprocessing (quality filtering, TPM/log-normalization, HVG selection, PCA/NMF embedding, Leiden clustering) happens upstream in [Preprocessing-module](../Preprocessing-module/). By the time an `AnnData` reaches this module it is expected to already have:
- Raw counts preserved in `adata.layers['counts']` (or `adata.raw`)
- Log-normalized expression in `adata.X`
- Cell-type labels in `adata.obs[celltype_column]`

What this module *does* do to the data before/while selecting genes:

1. **Blacklist filter (once, pre-selection).** `run_selection_pipeline.py` calls `apply_blacklist_filter()` on `adata` exactly once, before any strategy runs, then passes `blacklist_patterns=None, use_default_blacklist=False` down to every strategy/component call so the filter is never reapplied. Matching is case-insensitive gene-name-prefix matching; default patterns are `['mt-', 'hsp', 'rps', 'rpl']` (`_constants.DEFAULT_BLACKLIST_PATTERNS`). `--force_include_genes` bypasses the blacklist for those specific genes.
2. **`min_cells_per_celltype` filtering** (default 10, `filter_celltypes_by_min_cells()`): cell types with too few cells are dropped before RF training or DEG testing, so small groups don't destabilize cross-validation folds or statistical tests.
3. **Xenium mean-expression computation.** If Xenium filtering is enabled (the default), `run_selection_pipeline.py` computes either a per-celltype mean-expression dict (`mean_expr_per_ct`, used when `xenium_celltype_aware=True`, the default) or a single pooled global mean-expression dict (`compute_global_mean_expression()`, used only when `--disable_xenium_celltype_aware` is set). Both are always derived from log-normalized `adata.X`, regardless of what the dimred strategies use for their own input (see next point). These feed the post-selection Xenium filter — see "Filtering Pipeline Order" below.
4. **`dimred_counts_input` resolution** (`_resolve_dimred_input()`, dimred strategies only): NMF/PCA default to **raw integer counts** — `adata.layers['counts']` if present and verified integer-valued, else `adata.raw.X` aligned to the current gene set (raises if any current gene is missing from `adata.raw`) — via `--dimred_counts_input raw` (the default). This is deliberate: NMF (Frobenius, coordinate-descent solver) expects non-negative, count-like input. Every other strategy (RF, DEG, HVG) and the Xenium filter always use log-normalized `adata.X`. Pass `--dimred_counts_input lognorm` to run NMF/PCA on `adata.X` instead (this raises if `adata.X` still looks like raw integer counts).

## Random Forest Gene Selection

Implemented in [_rf_selection.py](./_rf_selection.py), used by `rf_simple` (no pre-filter) and `rf_deg` (DEG-prefiltered candidate genes).

**Cross-validation design:** for each of `random_seeds` (5 seeds, see "Seeding" below) a `StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)` (default `n_folds=5`) is run — **25 fold-runs total**. Each fold trains a fresh `RandomForestClassifier(n_estimators=100, max_depth=3, class_weight="balanced", random_state=seed+fold_idx, n_jobs=-1)` (class imbalance handled once, via `class_weight="balanced"` — the former extra `sample_weight=compute_sample_weight("balanced", …)` was removed so the correction is not applied twice), then scores it on the held-out fold with `f1_score(..., average="macro")`.

**Seeding:** `random_seeds` is not itself CLI-exposed, but it is derived from the pipeline's single `--random_state` flag via `_derive_rf_random_seeds(random_state)`, so `--random_state` is a true single knob controlling both this CV and the NMF/PCA fit's own `random_state` in the same run — pass `random_seeds=[...]` directly to `select_genes_with_rf()` to override the derivation entirely. At the default `random_state=42`, this reproduces the historical hardcoded `[42, 43, 44, 45, 46]` exactly (unchanged behavior for every existing default-seed run). For any other `random_state`, 5 seeds are drawn from a `random_state`-seeded `np.random.default_rng` rather than a simple `[random_state + i for i in range(5)]` offset — the offset scheme would make two runs with adjacent `random_state` values (e.g. 42 and 43) share 4 of their 5 RF CV seeds, so the RNG draw is used instead to keep different `random_state` runs properly independent.

**Ranking metric — this is the part not obvious from the API:** the final ranking score is **not** raw feature importance. For every (gene, fold) appearance, `weighted_importance = importance × macro_F1_of_that_fold`; these are then averaged per gene across all 25 fold-runs to give `weighted_importance` (aliased as `final_score`), and genes are ranked by that descending. The intent (per the source docstring) is to weight a gene's importance by how good the model was when it produced that importance — "genes important in good models" rank higher than genes only important in a poorly-performing fold. `avg_importance`, `max_importance`, and `avg_f1_score` are also tracked as metadata but are not what determines rank.

**Per-cell-type attribution (`rf_celltype` / `rf_contributing_celltypes` / `rf_celltype_scores`):** the RF is one global multiclass model, so `feature_importances_` is a single scalar per gene and `celltype` stays `'global'`. To attribute an RF gene to cell types "from the random forest itself", `_forest_per_class_gini_importance()` decomposes each fitted forest's Gini importance by target class using the identity `G = Σ_c p_c(1-p_c)` (each split's ΔG splits exactly into per-class parts), F1-weighted and summed over all 25 fold-runs. Row sums equal the ordinary importance, so **rank / `final_score` / the selected-gene set are unchanged**. The per-gene class *shares* are written as `rf_celltype` (argmax), `rf_contributing_celltypes` (`|`-joined classes with share ≥ `RF_CONTRIBUTING_CELLTYPE_MIN_SHARE`=0.20, argmax fallback), `rf_celltype_scores` (json). The full genes × cell-types share matrix is also stored in `rf_models/rf_gene_scores.pkl` under `gene_class_importance`.

**`rf_deg` DEG-prefilter retry ladder:** when `use_deg_prefilter=True` and no cached DEG results are supplied, `_select_genes_with_rf_deg_retries` tries up to **3 attempts** with progressively larger DEG candidate pools — `[base_per_group, max(base_per_group*2, probeset_size), len(all_genes)]`, where `base_per_group = n_deg_per_group or max(50, probeset_size // n_groups * 3)`. Each per-cell-type candidate list is first restricted to genes with a Benjamini–Hochberg adjusted p-value `<= DEFAULT_DEG_MAX_PVAL` (0.05) in `_get_deg_genes_for_classification`, then cut to the per-group size; no fold-change cut-off is applied. The **final** attempt drops the significance filter (`max_pval=1.0`) so the "all genes" fallback is truly unfiltered and RF-DEG never hard-fails. It stops as soon as an attempt yields `>= probeset_size` selected genes; otherwise it keeps the best-performing attempt and marks the panel `panel_size_status="short"` with reason `rf_deg_candidate_pool_exhausted_after_retries`.

**Caching:** if `results_dir` is given, RF results are pickled to `rf_models/rf_gene_scores.pkl` (gene scores, candidate genes, and all training hyperparameters) and reused on a later call with a compatible gene set. An explicit cache can also be supplied via `--rf_score_cache_file` / `rf_score_cache_file=...`, checked before any DEG prefiltering runs — this lets benchmark runs reuse a previous RF ranking without recomputing DEGs or retraining.

**Important:** `n_estimators`, `max_depth`, `n_folds`, and an explicit `random_seeds` override are **not exposed as CLI flags** on `run_selection_pipeline.py` — they can only be changed by calling `select_genes_with_rf()` directly from Python. `random_seeds`' *default* value, however, does respond to the CLI's `--random_state` (see "Seeding" above).

## NMF/PCA Gene Selection: Per-Celltype (only mode)

Implemented in [_dimred_selection.py](./_dimred_selection.py) (`select_genes_from_nmf` / `select_genes_from_pca`), used by `dimred_only` directly and by the dimred half of `RecoVar`/`RecoVar_PCA`. NMF/PCA is **always fitted independently within each cell type** — there is no global, all-cells-pooled mode. A **3-phase pool-based architecture** builds the panel:

**NMF configuration is data-space-driven, via `--nmf_objective`.** Every NMF fit goes through `nico2_lib`'s `NmfPredictor`, with the solver/beta_loss/init/max_iter resolved per run by `Utility-module/_nmf_objective.resolve_nmf_objective(dimred_counts_input, nmf_objective)`: `--nmf_objective auto` (the default) derives the objective from `--dimred_counts_input` — `raw` (the default count input) → `solver="mu"`, `beta_loss="kullback-leibler"`, `init="nndsvda"`, `max_iter=2000` (the Poisson-appropriate objective for UMI counts); `lognorm` → `solver="cd"`, `beta_loss="frobenius"`, `init="nndsvd"`, `max_iter=1000` — byte-identical to the historical pinned `_constants.NMF_PREDICTOR_FIXED_KWARGS` set. Pass `--nmf_objective frobenius` / `--nmf_objective kl` to force either objective regardless of count input. Only `n_components` and `random_state` vary further per call. The Evaluation module resolves the same way for reconstruction scoring (its own `--nmf_objective` flag), and `_constants.NMF_PREDICTOR_FIXED_KWARGS` remains the byte-identical Frobenius/cd fallback between the two `_constants.py` copies (`Selection-module/_constants.py` ↔ `Evaluation-module/_constants.py`) for any caller that doesn't go through `_nmf_objective.py`. `NmfPredictor.fit()`/`.predict()` honour `solver`/`beta_loss`/`init`/`max_iter`; `alpha_*` remain no-ops, so those entries document intent only.

**Phase 1 — Pool creation.** NMF/PCA is fit **independently per celltype** (each celltype's factors are `{celltype}_NMF_1`, `{celltype}_NMF_2`, ... and are **not comparable across celltypes**, since they come from separate fits). Cell types with fewer than `min_cells_per_celltype` cells, and within-celltype zero-variance genes, are excluded before the fit. The pool is built per (celltype, factor) combo, with `pool_size_per_celltype` genes distributed across that celltype's factors (`pool_size_per_celltype // n_components` per combo; default pool size 200, auto-resolved from panel size when not explicitly set — see below). If a Xenium mean-expression dict is available, it is applied **inside this phase** (using the celltype's own means) — this is why `dimred_only` skips the separate post-selection Xenium filter step that every other strategy goes through (see "Filtering Pipeline Order").

**Phase 2 — Duplicate resolution.** `_factor_aware.resolve_duplicates_factor_aware_per_celltype()`: **within-celltype** duplicates (a gene selected by more than one factor of the *same* celltype) are assigned to the factor where the gene's `abs(loading)` is highest, with the vacated pool slot backfilled from the next-best candidate for that factor; **cross-celltype** duplicates (the same gene independently selected by different celltypes' NMF) are deliberately left unresolved and tracked via `gene_celltype_mapping`/`n_celltypes_per_gene`, to be preferred later as broadly-informative genes.

**Phase 3 — Final selection (strict → fill → trim):**
  1. **Strict**: for each (celltype, factor) combo, take the top `ceil(probeset_size / n_celltypes / n_components)` pool genes by `abs(loading)` (a valid comparison — same celltype, same NMF run). Because genes shared across celltypes count once, the unique total can fall short of `probeset_size`.
  2. **Fill** (if short): remaining pool genes are sorted by `n_celltypes_per_gene` **descending only** — `abs(loading)` is deliberately *not* used here, since loadings from independent per-celltype NMF fits aren't on a comparable scale.
  3. **Trim** (if over, from ceiling rounding): remove the lowest-`abs(loading)`-scoring genes, each compared within its own assigned celltype.

  Every gene's `n_celltypes_selected` / `contributing_celltypes` is recorded in the output regardless of which celltype "won" it in Step 1.

**All** Phase 2 pool genes (not just the Phase 3 final selection) are added to the `GeneListBuilder` as tracked candidates — only the final genes are marked selected. The non-selected pool genes become the replacement/gap-fill pool used later (by the post-selection Xenium filter's gap-fill, and by combination-strategy gap-filling).

**Pool size auto-resolution:** if `--pool_size_per_celltype` is not explicitly set, `run_selection_pipeline.py` computes `min(5000, max(200, ceil(probeset_size / 3)))` so the pool scales with the target panel size instead of always defaulting to 200.

**Parallelism (`--dimred_n_jobs`):** each cell type's NMF/PCA fit is independent, so `--dimred_n_jobs N` (default `1`; `-1` = all cores, capped at the cell-type count) runs them across `N` worker processes. Results are **byte-identical to sequential** — the per-celltype data slices are cut on the main process, the RNG seed is threaded per task, and results are keyed by cell-type name. Each worker pins its BLAS/OpenMP thread count to 1 (`_pin_blas_threads`) so the process pool doesn't oversubscribe the CPU. The flag threads `run_selection_pipeline.py` → `run_single_selection` / `run_combination_selection` → `select_genes_from_nmf`/`select_genes_from_pca` (`nmf_n_jobs`/`pca_n_jobs`).

## Gene Ranking Mechanics

Each strategy computes its ranking score differently — the scores are **not comparable across strategies**:

| Strategy | Ranking score | Notes |
|---|---|---|
| `deg_only` | scanpy `rank_genes_groups` `scores` column (Wilcoxon Z-statistic by default), used directly — not log2FC or a derived composite | Genes are excluded only by adjusted p-value (`pvals_adj > max_pval`, default 0.05); fold change is not used to exclude or rank genes, so enough candidates remain for Xenium filtering. Top `n_genes_per_group` genes are taken per celltype group, groups processed in categorical order; a gene claimed by an earlier group is not re-selected by a later one (later groups fill their quota from their next-best unclaimed genes), and each gene's stored `celltype`/score is that of its highest-scoring group. |
| `rf_simple` / `rf_deg` | `weighted_importance` = mean of (feature importance × macro-F1) across 25 CV fold-runs | See "Random Forest Gene Selection" above — this is *not* raw feature importance. |
| `dimred_only` (nmf/pca) | `abs(loading)` within the gene's assigned (celltype, factor) | Loadings are **not comparable across celltypes** (independent NMF/PCA fits per celltype) — only used to rank within the same fit. |
| `hvg` | `dispersions_norm` (falls back to `variances_norm` → `variances` → uniform score if none are present) | Reuses `adata.var['highly_variable']` if it already covers `>= probeset_size` genes, else recomputes via `sc.pp.highly_variable_genes`. |
| `random` | uniform score (`1.0` for every gene); rank order is a random shuffle | Genes are drawn with `np.random.choice(replace=False)`. |
| `RecoVar` / `RecoVar_PCA` (combination) | **no unified score** in `RecoVar_panel_information.csv` | RF's `weighted_importance` and dimred's `abs(loading)` are on non-comparable scales, so no score column is written for the combined output; the per-component `rf_component`/`nmf_component` files retain each side's own rank/score for diagnostics. |

## Combination of Random Forest and NMF/PCA (`RecoVar`, `RecoVar_PCA`)

Implemented in [run_combination_selection.py](./run_combination_selection.py).

**1. Target split.** `rf_percentage`/`dimred_percentage` (defaults 0.25/0.75, must sum to 1.0 within 0.01, and each strictly between 0 and 1 — so this path can't make a 100% RF or 100% dimred panel; use `rf_simple`/`rf_deg` or `dimred_only` for those) split the target size (after subtracting any `--force_include_genes`, which are added first and always kept) into `n_rf_target = int(adjusted_target_size * rf_percentage)` and `n_dimred_target = adjusted_target_size - n_rf_target` (guarantees an exact sum). Only `rf_percentage` drives the split — `dimred_percentage` is **display-only**: it's validated for consistency and echoed to metadata/logs but never enters the arithmetic, so `0.30/0.70` and `0.30/0.6999` give the identical panel.

**2. Component runs, oversampled.** RF (`strategy='rf_deg'`) and dimred (`strategy='dimred_only'`) are each run through the **full** `run_single_selection` pipeline (including their own Xenium filtering) requesting `COMBINATION_OVERSAMPLE_FACTOR × target` genes (1.5×, so there's a surplus for duplicate resolution) — or loaded from `--rf_deg_cache_dir` / `--dimred_cache_dir` if provided and compatible (cache compatibility is checked against `filtering_summary.json`: incompatible Xenium/blacklist settings raise an error; differing custom blacklist patterns only warn).

**3. Combine with overlap resolution — RF wins.** `rf_genes_selected = top n_rf_target` of the RF list; `dimred_genes_selected = top n_dimred_target` of the dimred list. Any gene selected by both keeps its RF slot (internal identifier `gene_source='overlap→rf_deg'`, written into `RecoVar_panel_information.csv` as `"overlap"` — see the display-rename note in "Output Files" below); it is removed from the dimred side (`dimred_unique = dimred_genes_selected - overlap`) with **no immediate replacement** — the resulting dimred shortfall is simply left to be closed in the gap-filling step below, which prefers multi-celltype genes rather than just the next-ranked dimred gene.

**4. Gap-filling — a single sequential fallback, not parallel outputs.** If the combined panel (force-include + RF + dimred-unique) is still short of the target, `_apply_gap_filling` tries the two strategies **in priority order, stopping as soon as the gap reaches zero** — celltype-fill (if `run_celltype_filling`, default on) → DEG-fill (if `run_deg_filling`, default on). It does *not* run both and write them to separate subdirectories.

   Celltype-fill and DEG-fill both reuse the already-computed dimred/RF builders' full ranked lists (no recomputation) — celltype-fill pulls the next-best genes from the dimred component's Phase-2 pool, DEG-fill pulls the next-best genes from the RF component's full ranked list. Added genes are tagged with the internal `gene_source` identifier `gap_fill_celltype` / `gap_fill_deg` (written into `RecoVar_panel_information.csv` as `"gap-fill (NMF)"` / `"gap-fill (random forest)"`), and `gap_fill_strategy` = the display name (`dimred` / `rf_deg` respectively) from `GAP_FILL_STRATEGY_DISPLAY_NAMES`.

   > There is no third, *global dimred fill* gap-fill path — only celltype-fill and DEG-fill
   > exist. Dimred selection itself has no `analysis_type='global'` (all-cells-pooled) mode
   > either; NMF/PCA is always fit independently per cell type (see above).

## Usage

### Command-Line Interface

#### Basic Usage
```bash
python run_selection_pipeline.py \
    --strategy RecoVar \
    --input_file data.h5ad \
    --output_dir results/ \
    --probeset_size 100
```

#### Advanced Options
Blacklist and Xenium filtering are **on by default** — there is no `--apply_*` flag to turn them on; use the `--disable_*` flags to turn them off. `--reduction_type` is required for `dimred_only`, `RecoVar`, and `RecoVar_PCA`.

```bash
python run_selection_pipeline.py \
    --strategy RecoVar \
    --input_file data.h5ad \
    --output_dir results/ \
    --probeset_size 100 \
    --reduction_type nmf \
    --n_components 5 \
    --rf_percentage 0.25 \
    --dimred_percentage 0.75 \
    --blacklist_patterns ig- tcr- \
    --xenium_min_expr 0.1 \
    --xenium_max_expr 100.0 \
    --random_state 42
```

### Strategy-Specific Parameters

#### DEG Selection
```bash
--strategy deg_only \
--min_cells_per_celltype 50
```
(There is no `--log2fc_threshold` CLI flag — DEG exclusion is p-value-only; see "Gene Ranking Mechanics".)

#### Random Forest
```bash
--strategy rf_deg \
--min_cells_per_celltype 10
```
RF hyperparameters (`n_estimators`, `max_depth`, `n_folds`, an explicit `random_seeds` override) are not CLI-exposed — use the Python API (below) to change them. `random_seeds`' default already tracks the shared `--random_state` flag (see "Random Forest Gene Selection" → "Seeding" above). To reuse a previous RF ranking instead of retraining: `--rf_score_cache_file path/to/rf_gene_scores.pkl`.

#### Dimensionality Reduction
```bash
--strategy dimred_only \
--reduction_type nmf \
--n_components 10 \
--pool_size_per_celltype 300 \
--dimred_counts_input raw
```
Other useful dimred flags: `--dimred_n_jobs` (parallel per-celltype NMF/PCA fits — `1` default, `-1` = all cores), `--nmf_model_cache_dir` / `--require_nmf_model_cache` (share one NMF fit across runs), `--nmf_objective {auto,frobenius,kl}` (NMF only; default `auto` — data-space-driven, see "NMF/PCA Gene Selection" above).

#### Hybrid Strategies
```bash
--strategy RecoVar \
--rf_percentage 0.25 \
--dimred_percentage 0.75 \
--reduction_type nmf \
--n_components 5 \
--rf_deg_cache_dir results/prior_run/rf_component \
--dimred_cache_dir results/prior_run/nmf_component \
--force_recompute \
--disable_deg_filling
```

### Python API

There is no installed `selection` package — module directory names are hyphenated, so there is no clean `import selection` form. Every entry-point script (and any calling code) adds the module directory to `sys.path` and imports the underscore-prefixed modules directly, e.g.:

```python
import sys
sys.path.insert(0, "/path/to/Code/RecoVar/Selection-module")

from _gene_list_builder import GeneListBuilder
from _deg_selection import select_degs
from _rf_selection import select_genes_with_rf
from _dimred_selection import select_genes_from_nmf

# DEG selection — returns GeneListBuilder
deg_builder = select_degs(
    adata,
    probeset_size=100,
    celltype_column='celltype',
    min_cells_per_celltype=10,
)

# Random forest — returns GeneListBuilder
rf_builder = select_genes_with_rf(
    adata,
    probeset_size=100,
    celltype_column='celltype',
    n_folds=5,
    random_seeds=[42, 43, 44, 45, 46],
)

# NMF selection (always per cell type) — returns GeneListBuilder
nmf_builder = select_genes_from_nmf(
    adata,
    probeset_size=100,
    n_components=5,
    celltype_column='celltype',
)

# Retrieve final gene list
selected_genes = nmf_builder.get_selected_genes()
```

## Input Requirements

- **AnnData format** (`.h5ad`)
- **Gene expression data**: Log-normalized in `adata.X`; raw counts preserved in `adata.layers['counts']` or `adata.raw` (required by dimred strategies' default `dimred_counts_input='raw'`)
- **Cell-type annotations**: Stored in `adata.obs[celltype_col]`
- **Gene names**: Stored in `adata.var_names`

## Output Files

### Gene Lists

**Filenames.** Every `run_single_selection.py` output (`deg_only`, `hvg`, `random`,
`rf_simple`, `rf_deg`, `dimred_only`) is named `{strategy}_panel_information.csv` —
`deg_only_panel_information.csv`, `hvg_panel_information.csv`,
`random_panel_information.csv`, `rf_simple_panel_information.csv`,
`rf_deg_panel_information.csv`, and (for `dimred_only`, matching `--reduction_type`)
`nmf_panel_information.csv` / `pca_panel_information.csv`. This is the sole naming
authority: `_gene_list_builder.panel_information_filename(strategy, reduction_type=None)`.
A `RecoVar`/`RecoVar_PCA` combination run's RF/dimred component subdirectories use the
same per-strategy names (`rf_component/rf_deg_panel_information.csv`,
`nmf_component/nmf_panel_information.csv`); the combination's own top-level output is
always named `RecoVar_panel_information.csv` regardless of `--strategy` value
(`RecoVar` or `RecoVar_PCA`). There is no `selected_genes.csv` / `consensus_genes.csv`,
and combination strategies no longer produce a `ranked_gene_list.csv` or
`gene_provenance.csv` — `RecoVar_panel_information.csv` is the only combination output.

**Single-strategy `deg_only`/`hvg`/`random`.** Full, unchanged column set — every
candidate the builder tracked, including genes that failed the Xenium filter (kept with
`passed_xenium=False` and `xenium_failure_reason` rather than dropped), sorted
`['in_panel', 'rank']` descending/ascending:

| Column | Meaning |
|---|---|
| `gene` | Gene symbol |
| `in_panel` | Final panel membership |
| `rank` | 1-indexed position by `selection_score` descending |
| `selection_score` | Strategy's own ranking score (see "Gene Ranking Mechanics") |
| `selection_strategy` | The strategy name (`deg_only`/`hvg`/`random`) |
| `analysis_type` | `global` for all three |
| `celltype` | Cell type the gene's score/selection is attributed to (`deg_only`); `global` for `hvg`/`random` |
| `mean_expression` | Populated by the Xenium filter's mean-expression computation |
| `selected_initial` / `final_selection` | Legacy panel-membership bookkeeping (identical to `in_panel`) |
| `passed_xenium` / `xenium_failure_reason` | Xenium filter outcome per gene |
| `informative_celltypes` | See below |
| *(strategy-specific extras)* | `deg_only`: `logfoldchanges`, `pvals`, `pvals_adj`. `hvg`: `means`, `dispersions`, `dispersions_norm`, `highly_variable`. `random`: `random_state`. |

**RF component — `rf_deg_panel_information.csv` / `rf_simple_panel_information.csv`**
(standalone `rf_deg`/`rf_simple`, or `rf_component/` inside a combination run). Sorted by
`rank` ascending:

| Column | Meaning |
|---|---|
| `gene` | Gene symbol |
| `in_panel` | Final panel membership |
| `rank` | 1-indexed position by `weighted_importance` (`final_score`) descending |
| `rf_celltype` | Cell type with the highest Gini-importance share for this gene (argmax) |
| `rf_celltype_scores` | Per-class Gini-importance share, JSON dict summing to ≈1.0 |
| `mean_expression` | Populated by the Xenium filter's mean-expression computation |

**Dimred component — `nmf_panel_information.csv` / `pca_panel_information.csv`**
(standalone `dimred_only`, or `nmf_component`/`pca_component` inside a combination run).
Sorted by `['in_panel', 'loading', 'n_celltypes_selected']`, all descending:

| Column | Meaning |
|---|---|
| `gene` | Gene symbol |
| `in_panel` | Final panel membership |
| `selection_strategy` | Display value `nmf` or `pca` (mapped from the internal `dimred_only_{nmf,pca}_per_celltype` identifier) |
| `analysis_type` | `per_celltype` |
| `celltype` | Cell type whose (celltype, factor) slot the gene was assigned to |
| `informative_celltypes` | See below |
| `n_celltypes_selected` | How many cell types' independent NMF/PCA fits selected this gene |
| `component` | The `{celltype}_NMF_{k}` / `{celltype}_PCA_{k}` factor the gene was assigned to |
| `loading` | `abs(loading)` within that (celltype, factor) — **not comparable across celltypes** |
| `mean_expression` | Populated by the Xenium filter (applied inside Phase 1 pool construction for this strategy) |

**`informative_celltypes`** (single-strategy files only — for the combination file's
version, see below): unified per-gene `|`-joined list of every cell type the gene is
informative for, built by `_gene_list_builder.derive_informative_celltypes()`. Feeds
`Plotting-module.plot_genes_per_celltype()`'s coverage diagnostic and
`Evaluation-module/_pathway.py`'s `--pathway_celltype_gene_source informative` mode.

**Combination strategies — `RecoVar_panel_information.csv`** (`RecoVar`/`RecoVar_PCA`
only, written by `run_combination_selection.py`): the **sole** combination output,
covering every gene considered during combination (union of the RF and dimred candidate
pools, plus the final panel) — not panel-only. Panel genes (`in_panel=True`) are listed
first, ordered by `gene_source` priority (RF-derived → dimred → force-include → the two
gap-fill variants) then `gap_filled` (`False` before `True`); non-panel candidates follow,
ordered by pool (`"random forest"` before `"NMF"`) then gene name alphabetically:

| Column | Meaning |
|---|---|
| `gene` | Gene symbol |
| `in_panel` | Final panel membership — **the whole file is not panel-only**, unlike the old `ranked_gene_list.csv` |
| `gene_source` | Panel rows: `"random forest"`, `"NMF"`, `"overlap"` (selected by both, credited to random forest), `force_include`, `"gap-fill (NMF)"`, `"gap-fill (random forest)"`. Non-panel rows: `"random forest"` or `"NMF"` (which candidate pool the gene came from). These are display strings written by `_GENE_SOURCE_DISPLAY_RENAME` — the internal identifiers used elsewhere in the code (`rf_deg`, `dimred`, `overlap→rf_deg`, `gap_fill_celltype`, `gap_fill_deg`) are unchanged |
| `informative_celltypes` | `primary_celltype` + `secondary_celltypes_nmf`, deduplicated and `|`-joined — for RF-resolved rows this is just `primary_celltype` |
| `primary_celltype` | The cell type the gene is attributed to — **value-based fallback**: the gene's real `rf_celltype` if it has one (regardless of which pool/gene_source it was actually selected through), else its dimred `celltype`, else blank |
| `secondary_celltypes_nmf` | Every cell type whose NMF/PCA fit also selected this gene — populated only when `primary_celltype` was resolved from the dimred side |
| `n_celltypes_nmf` | Count of `secondary_celltypes_nmf` — populated only when `primary_celltype` was resolved from the dimred side |
| `gap_filled` | Whether this panel gene was added by gap-filling (always `False` for non-panel rows) |
| `gap_fill_strategy` | `dimred` / `rf_deg` when `gap_filled=True`, blank otherwise |
| `mean_expression` | From whichever builder (RF or NMF) `primary_celltype` was resolved from |

`candidate_pool`, `component`, `rf_rank`, `dimred_rank` (from the old `gene_provenance.csv`)
and `selection_strategy`/`analysis_type`/`n_celltypes_selected`/`contributing_celltypes`/
`rf_contributing_celltypes`/`rf_celltype_scores` (from the old panel-only
`ranked_gene_list.csv`) are all dropped — their useful signal is covered by
`gene_source`/`primary_celltype`/`informative_celltypes` above. No unified
`selection_score` is written (see "Gene Ranking Mechanics" table).

### Metadata
- `selection_parameters.json`: `vars(args)` + `timestamp` + `script`, enriched with
  panel metadata from `result.metadata`. Written by every `run_selection_pipeline.py`
  run. (There is no `parameters_<timestamp>.json`.)
- `filtering_summary.json`: per single-strategy run — panel-size status, Xenium/blacklist
  filter tallies, RF-DEG retry info.
- `combination_summary.json`: Combination strategy summary (RecoVar/RecoVar_PCA only) —
  duplicate-resolution counts plus the combined-panel `panel_size_status`
  (`complete`/`short`) + `short_panel_reason` and `requested_`/`final_panel_size`, mirroring
  each component's `filtering_summary.json`
- `{experiment_name}.log` (default `gene_selection.log`): detailed execution log (also
  echoed to stderr). There is no `selection_log.txt`.

## Filtering Pipeline Order

The filtering pipeline in `run_single_selection.py` executes in this exact order — this order is critical for reproducibility:

1. **Blacklist filter** (PRE-selection) — removes unwanted gene patterns (e.g., `mt-`, `hsp`, `rps`, `rpl`) from `adata` before any selection runs
2. **Run selection strategy** — operates on the blacklist-filtered `adata`
3. **Xenium filter** (POST-selection) — applied to the ranked gene list for every strategy *except* `dimred_only`, which already applied it inside its Phase 1 pool construction (see "NMF/PCA Gene Selection" above); celltype-aware mode checks each gene against its assigned cell type, global mode rejects a gene only if it fails in ALL cell types
4. **Select top N** — picks the top `probeset_size` genes from the xenium-filtered ranked list, gap-filling from remaining Xenium-passing candidates (sorted by selection score) if the strict top-N pass came up short

> **Design rationale — Xenium filter is applied POST-selection (intentional)**: NMF runs on the full transcriptome (or HVG subset) to capture true biological gene programs without constraining the feature space to experimentally detectable genes. The Xenium filter is then applied as a final gate to ensure only experimentally useful genes reach the panel. This is a deliberate improvement over the old pipeline (`Modules/`), which applied the Xenium filter *before* NMF — artificially limiting what the NMF could discover to genes that happen to be within Xenium detection range.

## Strategy Selection Guide

| Strategy | Use Case | Complexity | Interpretability |
|----------|----------|------------|------------------|
| **random** | Baseline control | Low | High |
| **hvg** | Simple baseline | Low | High |
| **deg_only** | Cell-type specific markers | Medium | High |
| **rf_simple** | Feature importance ranking | Medium | Medium |
| **rf_deg** | Refined feature importance | Medium | Medium |
| **dimred_only** | Mechanistic representation | Medium | Medium |
| **RecoVar** | Best overall performance (25% RF + 75% NMF) | High | Low |
| **RecoVar_PCA** | Linear factor structure (25% RF + 75% PCA) | High | Medium |

## Best Practices

1. **Start with baselines**: Run `random` and `hvg` to establish performance floor
2. **Use hybrid strategies**: `RecoVar` typically provides best evaluation metrics
3. **Filtering is on by default**: blacklist and Xenium filtering both run unless explicitly disabled (`--disable_default_blacklist`, `--disable_xenium_filter`) — there is no `--apply_*` flag to turn them on
4. **Set random seeds**: Use `--random_state` for reproducibility
5. **Cache expensive components for combination runs**: use `--rf_deg_cache_dir` / `--dimred_cache_dir` / `--rf_score_cache_file` / `--nmf_model_cache_dir` to reuse previously computed RF or NMF/PCA results instead of recomputing them; pass `--force_recompute` to ignore a cache

## Dependencies

- `scanpy >= 1.9`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `anndata`

## Related Modules

- [Preprocessing-module](../Preprocessing-module/): Upstream normalization/QC that produces the AnnData this module consumes
- [Evaluation-module](../Evaluation-module/): Quality assessment of selected panels

## Author

Helene Hemmer

## License

See project root for license information.
