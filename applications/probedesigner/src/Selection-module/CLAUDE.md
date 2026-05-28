# Selection Module

Version: 2.0.0

This module provides eight different gene selection strategies for spatial transcriptomics probe design, from simple baselines to sophisticated hybrid approaches.

## Overview

The Selection module implements a comprehensive suite of gene selection strategies:

- **deg_only**: Differential expression analysis
- **rf_simple**: Random forest classifier on all genes
- **rf_deg**: Random forest on DEG-filtered genes
- **rf_nmf**: Hybrid random forest + NMF with gap-filling
- **rf_pca**: Hybrid random forest + PCA with gap-filling
- **dimred_only**: Pure dimensionality reduction (NMF/PCA)
- **hvg**: Highly variable genes (baseline)
- **random**: Random selection (baseline)

## Directory Structure

```
Selection-module/
├── __init__.py                      # Package initialization with public API
├── _constants.py                    # Strategy definitions and defaults
├── _gene_list_builder.py            # Core data structure for tracking gene provenance
├── _filtering.py                    # Post-selection filtering (blacklist, Xenium)
├── _deg_selection.py                # Differential expression analysis
├── _baseline_selection.py           # HVG and random selection
├── _rf_selection.py                 # Random forest-based selection
├── _dimred_selection.py             # NMF/PCA dimensionality reduction
├── _factor_aware.py                 # Factor-aware duplicate resolution & gap-filling
├── run_selection_pipeline.py        # Main pipeline orchestrator
├── run_single_selection.py          # Single-strategy executor
└── run_combination_selection.py     # Combination-strategy executor (rf_nmf, rf_pca)
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
7. **rf_nmf**: Random forest (25%) + NMF (75%) using strict-then-fill per-celltype selection
8. **rf_pca**: Random forest (25%) + PCA (75%) using strict-then-fill per-celltype selection

### Core Components

#### GeneListBuilder ([_gene_list_builder.py](._gene_list_builder.py))
Unified data structure for tracking gene provenance through:
- Initial selection
- Filtering (blacklist, Xenium compatibility)
- Duplicate resolution
- Gap-filling and replacement

#### Factor-Aware Processing ([_factor_aware.py](._factor_aware.py))
Intelligent handling of duplicates and gap-filling:
- Within-celltype duplicate resolution: assigns a gene to the factor where its `abs(loading)` is highest; replaces it with the next-best candidate in the other factor
- Cross-celltype shared genes are **tracked but not resolved** — `gene_celltype_mapping` records all celltypes that selected a gene in Phase 1
- Provides `n_celltypes_per_gene` for downstream preference of broadly-selected genes
- Per-cell-type or global strategies supported

#### Filtering Pipeline ([_filtering.py](._filtering.py))
- **Blacklist filtering**: Remove unwanted gene patterns (MT-, RP-, etc.)
- **Xenium compatibility**: Filter to Xenium-compatible genes

## Usage

### Command-Line Interface

#### Basic Usage
```bash
python run_selection_pipeline.py \
    --strategy rf_nmf \
    --input_file data.h5ad \
    --output_dir results/ \
    --probeset_size 100
```

#### Advanced Options
```bash
python run_selection_pipeline.py \
    --strategy rf_nmf \
    --input_file data.h5ad \
    --output_dir results/ \
    --probeset_size 100 \
    --reduction_type nmf \
    --n_components 5 \
    --analysis_type per_celltype \
    --rf_percentage 0.60 \
    --dimred_percentage 0.40 \
    --apply_blacklist \
    --blacklist_patterns MT- RP- \
    --apply_xenium_filter \
    --random_state 42
```

### Strategy-Specific Parameters

#### DEG Selection
```bash
--strategy deg_only \
--min_cells_per_celltype 50 \
--log2fc_threshold 1.0
```

#### Random Forest
```bash
--strategy rf_simple \
--n_estimators 100 \
--max_depth 10 \
--rf_n_jobs 4
```

#### Dimensionality Reduction
```bash
--strategy dimred_only \
--reduction_type nmf \
--n_components 10 \
--analysis_type per_celltype
```

#### Hybrid Strategies
```bash
--strategy rf_nmf \
--rf_percentage 0.25 \
--dimred_percentage 0.75 \
--reduction_type nmf \
--n_components 5
```

### Python API

```python
from selection import GeneListBuilder
from selection import select_DEGs, select_genes_with_rf, select_genes_from_nmf

# DEG selection — returns GeneListBuilder
deg_builder = select_DEGs(
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

# NMF selection — returns GeneListBuilder
nmf_builder = select_genes_from_nmf(
    adata,
    probeset_size=100,
    n_components=5,
    analysis_type='per_celltype',
    celltype_column='celltype',
)

# Retrieve final gene list
selected_genes = nmf_builder.get_selected_genes()
```

## Input Requirements

- **AnnData format** (`.h5ad`)
- **Gene expression data**: Stored in `adata.X` or `adata.layers['counts']`
- **Cell-type annotations**: Stored in `adata.obs[celltype_col]`
- **Gene names**: Stored in `adata.var_names`

## Output Files

### Gene Lists
- `ranked_gene_list.csv`: Full ranked gene list with provenance metadata (combination strategies)
  - Key columns: `gene`, `rank`, `selection_score`, `selection_strategy`, `celltype`, `final_selection`, `passed_xenium`, `replaced_by`, `replaces_gene`, `replacement_reason`, `component`, `component_loading`
  - NMF/PCA dimred columns: `n_celltypes_selected` (int — how many celltypes selected this gene in Phase 1), `contributing_celltypes` (comma-separated list of all contributing celltypes)
  - `final_selection == True` marks genes in the final panel; downstream tools filter on this column
- `selected_genes.csv`: Simple gene list (single strategies)

### Metadata
- `parameters_<timestamp>.json`: Full parameter settings for reproducibility
- `selection_log.txt`: Detailed execution log

### Filtering Reports
- `blacklist_removed.txt`: Genes removed by blacklist filter
- `xenium_incompatible.txt`: Genes removed by Xenium filter

## Filtering Pipeline Order

The filtering pipeline in `run_single_selection.py` executes in this exact order — this order is critical for reproducibility:

1. **Blacklist filter** (PRE-selection) — removes unwanted gene patterns (e.g., `mt-`, `hsp`) from `adata` before any selection runs
2. **Run selection strategy** — operates on the blacklist-filtered `adata`
3. **Xenium filter** (POST-selection) — applied to the ranked gene list; celltype-aware mode checks each gene against its assigned cell type, global mode rejects a gene only if it fails in ALL cell types
4. **Select top N** — picks the top `probeset_size` genes from the xenium-filtered ranked list

> **Design rationale — Xenium filter is applied POST-selection (intentional)**: NMF runs on the full transcriptome (or HVG subset) to capture true biological gene programs without constraining the feature space to experimentally detectable genes. The Xenium filter is then applied as a final gate to ensure only experimentally useful genes reach the panel. This is a deliberate improvement over the old pipeline (`Modules/`), which applied the Xenium filter *before* NMF — artificially limiting what the NMF could discover to genes that happen to be within Xenium detection range.

## Combination Strategy: Per-Celltype NMF/PCA Selection

For `rf_nmf` and `rf_pca` with `analysis_type='per_celltype'`, the dimred gene selection follows a **strict-then-fill** approach:

1. **Phase 1 — Pool building**: Run NMF/PCA independently per celltype; collect top genes per (celltype, factor) combo into a shared pool. Each celltype's NMF is an independent run — loadings are **not comparable across celltypes**.

2. **Phase 2 — Within-celltype duplicate resolution**: If a gene appears in multiple factors of the same celltype, assign it to the factor where its `abs(loading)` is highest; fill the vacated slot with the next-best candidate gene from that factor's pool.

3. **Phase 3 — Strict-then-fill final selection**:
   - **Step 1 (strict)**: For each (celltype, factor) combo, take the top `ceil(probeset_size / n_celltypes / n_factors)` genes by `abs(loading)`. Cross-celltype shared genes may cause some combos to have fewer available genes, making the unique total fall below `probeset_size`.
   - **Step 2 (fill)**: If unique count < `probeset_size`, fill the gap from remaining pool genes sorted by `n_celltypes` only — `abs(loading)` is NOT used here because loadings from different per-celltype NMF runs are not on the same scale.
   - **Step 3 (trim)**: If unique count > `probeset_size` (ceiling rounding over-count), trim the lowest-loading genes (within-CT comparison — valid since these are the same NMF run).

Cross-celltype shared genes are always tracked via `contributing_celltypes` / `n_celltypes_selected` in the output, regardless of which celltype "owns" the gene in Step 1.

## Combination Strategy Gap-Filling

For hybrid strategies (`rf_nmf`, `rf_pca`), three gap-filling variants are generated after the initial RF + dimred combination:

| Variant | Key | Description |
|---------|-----|-------------|
| Cell-type-specific | `run_celltype_filling` | Fills gaps with genes from the same cell-type pool as the missing slot |
| Global-gene | `run_global_filling` | Fills gaps with globally ranked genes from the dimred pool |
| DEG-based | `run_deg_filling` | Fills gaps with DEG-ranked genes |

All three variants are produced in the same run (controlled by `run_celltype_filling`, `run_global_filling`, `run_deg_filling` flags, all `True` by default). Each variant is written to a separate output subdirectory.

## Strategy Selection Guide

| Strategy | Use Case | Complexity | Interpretability |
|----------|----------|------------|------------------|
| **random** | Baseline control | Low | High |
| **hvg** | Simple baseline | Low | High |
| **deg_only** | Cell-type specific markers | Medium | High |
| **rf_simple** | Feature importance ranking | Medium | Medium |
| **rf_deg** | Refined feature importance | Medium | Medium |
| **dimred_only** | Mechanistic representation | Medium | Medium |
| **rf_nmf** | Best overall performance (25% RF + 75% NMF) | High | Low |
| **rf_pca** | Linear factor structure (25% RF + 75% PCA) | High | Medium |

## Best Practices

1. **Start with baselines**: Run `random` and `hvg` to establish performance floor
2. **Use hybrid strategies**: `rf_nmf` typically provides best evaluation metrics
3. **Apply filtering**: Always use `--apply_blacklist` to remove MT-/RP- genes
4. **Set random seeds**: Use `--random_state` for reproducibility
5. **Cache results**: Use `--use_cache` for large datasets with repeated runs

## Dependencies

- `scanpy >= 1.9`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `anndata`

## Related Modules

- [Evaluation-module](../Evaluation-module/): Quality assessment of selected panels
- [Analysis-scripts](../Analysis-scripts/): Stability and comparison analyses

## Author

Helene Hemmer

## License

See project root for license information.
