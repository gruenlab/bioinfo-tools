# Evaluation Module

Version: 2.0.0

This module provides comprehensive tools to evaluate the quality of selected probe panels for spatial transcriptomics experiments using three complementary approaches.

## Overview

The Evaluation module assesses probe panel quality through:

1. **Baseline Evaluation**: Clustering quality, neighborhood preservation, and cell-type identification
2. **Variability Evaluation**: NMF representation metrics
3. **Reconstruction Check** (optional): Tangram-based full-transcriptome reconstruction

## Directory Structure

```
Evaluation-module/
├── __init__.py                 # Package initialization with public API
├── _clustering.py              # Clustering and neighborhood metrics
├── nmf.py                      # NMF representation (nmf_reconstruction, nmf_reconstruction_by_celltype)
├── metrics.py                  # Metric calculations (calculate_*, compute_standardized_score)
├── _splits.py                  # Split generation (generate_evaluation_splits)
├── _tangram.py                 # Tangram reconstruction evaluation
├── _preprocessing.py           # Data preprocessing utilities
├── _filters.py                 # Dataset filtering and grouping
├── _constants.py               # Module constants and defaults
├── run_evaluation.py           # Main CLI entry point
└── run_tangram_check.py        # Tangram-specific evaluation script
```

## Key Features

### Baseline Metrics
- **Clustering Quality**: ARI (Adjusted Rand Index) and NMI (Normalized Mutual Information)
- **Neighborhood Preservation**: kNN overlap between full transcriptome and probe panel
- **Cell-type Identification**: Accuracy of cell-type assignments

### Variability Metrics
- **MSE** (Mean Squared Error): Reconstruction error in NMF factor space
- **Explained Variance**: Proportion of variance captured by probe panel
- Supports both **macro-averaged** (per cell-type) and **weighted** aggregation

### Reconstruction Metrics (Optional)
- **Tangram-based**: Full-transcriptome prediction from probe panel expression
- Requires Tangram installation

## Usage

### Quick Start

```python
from evaluation import run_evaluation
run_evaluation.main()
```

### Command-Line Interface

The main script [run_evaluation.py](run_evaluation.py) supports three modes:

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
    --evaluation_type both
```

#### 3. Both (Preprocessing + Evaluation)
```bash
python run_evaluation.py --mode both \
    --input_file data.h5ad \
    --gene_lists_dir Selected-panels/ \
    --preprocessed_dir preprocessed/ \
    --output_dir Evaluation-Results/ \
    --evaluation_type both \
    --include_tangram
```

### Dry-Run Mode
```bash
python run_evaluation.py --mode both --dry_run ...
```

Prints resolved configuration without executing the pipeline.

## API Reference

### Clustering Functions ([_clustering.py](._clustering.py))

```python
from evaluation import (
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

# Compute ARI/NMI for clustering
results = evaluate_clustering_quality(sets, reference_key="full_transcriptome", dimensionality_reduction="pca")

# Compute kNN overlap at k=5,10,15,20,30,50
overlap = evaluate_neighborhood_preservation(sets, reference_key="full_transcriptome", dimensionality_reduction="pca")

# Evaluate cell-type accuracy (decision tree classifier)
accuracy = evaluate_celltype_identification(sets, reference_key="full_transcriptome", celltype_col="new_annot")
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
    compute_standardized_score,
)
```

### Split Generation ([_splits.py](_splits.py))

```python
from _splits import generate_evaluation_splits

# Returns list of (train_idx, test_idx, per_celltype_splits) tuples
# split_mode: "kfold" (default) or "simple"
splits = generate_evaluation_splits(
    adata, celltype_col='celltype',
    split_mode='kfold', n_splits=5, random_state=42,
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

1. **ranked_gene_list.csv** (recommended, Selection-module v2 format)
   - **Requires columns**: `gene`, `final_selection`
   - **Behavior**: Only genes where `final_selection == True` are evaluated
   - **Generated by**: Selection-module combination strategies (rf_nmf, rf_pca)
   - **Example**:
     ```csv
     gene,selection_score,final_selection,gene_source,celltype
     Cd3d,0.95,True,rf_deg,T cells
     Cd79a,0.88,True,dimred,B cells
     Mt-Co1,0.75,False,filtered_out,All
     ```
   - **Logging**: Reports filtering (e.g., "Filtered to 100 genes with final_selection=True from 150 total")

2. **selected_genes.csv** (standard format)
   - Multi-column CSV with a 'gene' column
   - All genes in the file are included
   - Compatible with NS-Forest and other tools

3. **Simple list** (basic format)
   - Single column of gene names
   - May or may not have header
   - First column is used as gene names

## Output Files

### Preprocessing Mode
- `full_transcriptome.h5ad`: Reference dataset with PCA/NMF/clustering
- `panel_<name>.h5ad`: Subsetted probe panels with computed metrics

### Evaluation Mode
- `baseline_metrics_*.csv`: Clustering, kNN, and cell-type metrics
- `variability_metrics_*.csv`: MSE and explained variance metrics
- `tangram_reconstruction_*.csv`: Reconstruction metrics (if enabled)
- `evaluation_summary.json`: Combined summary statistics

## Dependencies

- `scanpy >= 1.9`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `anndata`
- `tqdm`
- `psutil` (optional, for memory monitoring)
- `tangram-sc` (optional, for reconstruction)

## Implementation Notes

### Train/Test Split Discipline

Splits are generated **once** via `generate_evaluation_splits()` in `_splits.py` and forwarded to both NMF and Tangram. This ensures both methods evaluate on the same held-out cells.

- `generate_evaluation_splits()` returns `[(train_idx, test_idx, per_celltype_splits), ...]`
- `per_celltype_splits` contains position arrays derived by intersecting the global fold indices with per-cell-type positions — they are strict subsets of the global partition
- `nmf_reconstruction_by_celltype` requires `per_celltype_splits`; passing `None` raises `ValueError`
- No internal fallback splitting exists anywhere in the module — all splits must come from `generate_evaluation_splits()`

### NMF
- Uses `sklearn.decomposition.non_negative_factorization` (public API) — **not** the `NMF` class and never `NMF._fit_transform` (private)
- Full fit: `W, H, _ = non_negative_factorization(A, n_components=k, max_iter=..., random_state=...)`
- Constrained-W solve (probe genes only, fixed H): `W, _, _ = non_negative_factorization(A_P, H=H_P, n_components=k, init="custom", update_H=False, ...)`

### Tangram
- Uses custom unfiltered wrappers defined in `_tangram.py` (adapted from colleague's nico2_lib):
  - `pp_adatas_unfiltered` — skips `filter_genes` so zero-expressed genes are retained
  - `_map_cells_to_space_unfiltered` — removes the zero-value gene assertion
  - `project_genes_unfiltered` — skips `filter_genes` on the sc reference
- Effect: **all shared genes** (including zero-expressed ones) are reconstructed, unlike standard `tg.pp_adatas`
- Tangram supports **optional train/test splits** via `train_idx`/`test_idx` parameters (same indices from `generate_evaluation_splits()` as used by NMF — not a separate fit on the full dataset)

### Metrics
- **MSE**: `sklearn.metrics.mean_squared_error(X_original, X_reconstructed)` — equivalent to `mean((X - X_recon)^2)` over all elements
- **Explained variance**: custom v2 formula `1 - MSE / Var(X_original)` — **not** `sklearn.metrics.explained_variance_score` (which uses variance of residuals, not MSE)

## Notes

- **Memory Efficiency**: The module uses sparse matrices where possible and includes memory monitoring
- **String Handling**: Configured for pandas 2.x compatibility with ArrowStringArray
- **Caching**: Preprocessing results can be cached to avoid redundant computation

## Related Modules

- [Selection-module](../Selection-module/): Gene selection strategies
- [Analysis-scripts](../Analysis-scripts/): Downstream analysis pipelines

## Author

Helene Hemmer

## License

See project root for license information.
