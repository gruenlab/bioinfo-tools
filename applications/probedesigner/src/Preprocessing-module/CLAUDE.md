# Preprocessing Module

Version: 2.0.0

This module provides data preprocessing utilities that serve three distinct pipeline contexts: gene selection, panel evaluation, and downstream analysis scripts.

## Overview

Four scripts cover the full preprocessing needs of the pipeline:

| Script | Context | Role |
|--------|---------|------|
| `preprocess_for_selection.py` | Selection pipeline | Importable function + CLI; produces filter/HVG combinations for selection |
| `preprocess_for_evaluation.py` | Evaluation pipeline | CLI; preprocesses all selected gene-list panels for evaluation |
| `preprocess_reference_for_evaluation.py` | Evaluation + Analysis | Importable function + CLI; produces reference h5ad with UMAP/Leiden |
| `preprocess_for_analysis.py` | Analysis scripts | CLI wrapper; delegates to `preprocess_reference_for_evaluation.py` |

## Directory Structure

```
Preprocessing-module/
├── __init__.py                            # Exports preprocess_for_selection, preprocess_for_analysis
├── _constants.py                          # Module constants (n_components, normalization targets, etc.)
├── preprocess_for_selection.py            # Selection pipeline preprocessing
├── preprocess_for_evaluation.py           # Evaluation panel preprocessing (CLI)
├── preprocess_reference_for_evaluation.py # Reference data preprocessing (importable + CLI)
└── preprocess_for_analysis.py             # Analysis script preprocessing (CLI wrapper)
```

## Critical Rule: Raw Counts Preservation

Raw counts **must** be preserved before any normalization:

```python
# Always done first, before normalize_total / log1p
adata.layers["counts"] = adata.X.copy()
```

Check before running any downstream pipeline:
```python
assert "counts" in adata.layers, "Raw counts missing — run preprocessing first"
```

## Normalization Pipeline

```
Raw counts (.X or .layers["counts"])
  ↓
sc.pp.normalize_total(target_sum=1e4)   # CPM normalization
  ↓
sc.pp.log1p()                           # Log-normalization
  ↓
[Optional] sc.pp.highly_variable_genes() # HVG selection
  ↓
sc.pp.pca() / NMF                       # Dimensionality reduction
  ↓
sc.pp.neighbors() → sc.tl.umap()       # Graph + UMAP
  ↓
sc.tl.leiden()                          # Leiden clustering
```

## Key Functions

### `preprocess_for_selection()`

**File:** `preprocess_for_selection.py`

**Importable via:** `from preprocessing import preprocess_for_selection`

**Signature:**
```python
def preprocess_for_selection(
    input_file: str,
    output_dir: str,
    celltype_column: str = "celltype",
    filter_methods: List[str] = None,        # default: ["scanpy", "no_filter"]
    hvg_option: str = "both",               # "all_genes", "hvg", or "both"
    n_components_pca: int = 50,
    n_components_nmf: int = 5,
    random_state: int = 42,
) -> None
```

**Output structure:**
```
output_dir/
└── {filter_name}_{hvg_name}/
    ├── preprocessed.h5ad     # Normalized + embedded AnnData
    ├── pca_loadings.csv      # Gene × PC loading matrix
    ├── nmf_loadings.csv      # Gene × NMF-component loading matrix
    └── metadata.json         # Parameters used
```

**Supported filter methods:**
- `"scanpy"` — standard QC filtering (min cells/genes thresholds from `_constants.py`)
- `"no_filter"` — skip filtering, use all cells/genes

**Supported HVG options:**
- `"all_genes"` — use all genes
- `"hvg"` — use only highly variable genes (n=2000 default)
- `"both"` — produce both outputs in separate subdirectories

---

### `preprocess_reference_for_analysis_scripts()`

**File:** `preprocess_reference_for_evaluation.py`

**Importable via:** Called by `preprocess_for_analysis.py`; can also be imported directly.

**Signature:**
```python
def preprocess_reference_for_analysis_scripts(
    input_file: str,
    output_file: str,
    celltype_column: str = "celltype",
    dimensionality_reduction: str = "both",   # "pca", "nmf", or "both"
    n_neighbors: int = 15,
    n_nmf_components: int = 5,
) -> None
```

**Produces a single output h5ad containing:**
- `layers["counts"]` — preserved raw counts
- `.X` — log-normalized expression
- `.obsm["X_pca"]` — PCA embedding
- `.obsm["X_nmf"]` — NMF embedding (if `dimensionality_reduction` is `"nmf"` or `"both"`)
- `.obsm["X_umap"]` — UMAP coordinates
- `.obs["leiden"]` — Leiden cluster assignments

**Note:** Uses `importlib.util` to load its own `_constants.py` by absolute path — this avoids a name collision with `Evaluation-module/_constants.py` and is an acknowledged exception to the no-importlib rule in the project.

---

### `preprocess_for_evaluation.py` (CLI)

Preprocesses all gene-list panels produced by the Selection-module so they are ready for evaluation. It discovers gene lists automatically from the selection output directory, then calls `process_data_for_panel_evaluation()` from the Evaluation-module for each panel.

**Cross-module dependencies:**
- `Evaluation-module._preprocessing.process_data_for_panel_evaluation` — core preprocessing logic
- `Utility-module._utils.convert_ensembl_to_gene_symbols` — ENSEMBL → gene symbol conversion

**CLI Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--input_file` | path | Yes | Raw h5ad file |
| `--output_dir` | path | Yes | Output directory for preprocessed panel h5ad files |
| `--gene_lists_dir` | path | No* | Selection pipeline output directory (auto-discovers gene lists) |
| `--gene_list_files_txt` | path | No* | Text file listing gene list paths (use when arg list too long) |
| `--external_panels` | path(s) | No | External panel CSV files |
| `--external_names` | str(s) | No | Names for external panels |
| `--external_probeset_sizes` | int(s) | No | Sizes to subset external panels to (0 or -1 = use all) |
| `--add_10x_panels` | choice | No | Combine with 10x panels: `both`, `mMulti`, `5k`, `no-10x-panel`, `all` |
| `--celltype_column` | str | No | Cell type column in obs (default: `celltypes_v2`) |
| `--n_neighbors` | int | No | KNN neighbors for UMAP (default: 15) |
| `--dimensionality_reduction` | choice | No | `pca`, `nmf`, or `both` (default: `both`) |
| `--log_level` | choice | No | `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`) |

*At least one of `--gene_lists_dir` or `--gene_list_files_txt` is required.

**Gene list discovery priority** (within each subdirectory):
1. `ranked_gene_list.csv` — filters rows where `final_selection == True`
2. `selected_genes.csv` — uses all rows
3. HVG/random fallback

**Output:** One `{panel_name}.h5ad` per gene list in `--output_dir`, plus `full_transcriptome.h5ad`.

---

### `preprocess_for_analysis.py` (CLI wrapper)

Thin CLI wrapper around `preprocess_reference_for_analysis_scripts()`. Has no additional logic.

**Usage:**
```bash
python preprocess_for_analysis.py \
    --input_file data.h5ad \
    --output_file analysis_input.h5ad \
    [--dimensionality_reduction both] \
    [--n_neighbors 15] \
    [--n_nmf_components 5]
```

## Constants (`_constants.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_N_COMPONENTS_PCA` | 50 | PCA components |
| `DEFAULT_N_COMPONENTS_NMF` | 5 | NMF components |
| `NORMALIZE_TARGET_SUM` | 1e4 | CPM normalization target |
| `DEFAULT_N_HVG` | 8000 | HVG count |
| `DEFAULT_HVG_FLAVOR` | `"cell_ranger"` | Scanpy HVG flavor |
| `DEFAULT_MIN_GENES_PER_CELL` | 100 | Min genes per cell for QC filter |
| `DEFAULT_MIN_CELLS_PER_GENE` | 3 | Min cells per gene for QC filter |
| `FILTER_METHODS` | `["scanpy", "no_filter"]` | Available filter methods |

## Input Requirements

- **Format:** AnnData `.h5ad`
- **Expression:** Raw integer counts in `.X` or `adata.layers["counts"]`
- **Cell-type labels:** `adata.obs[celltype_column]`
- **Gene names:** `adata.var_names` (gene symbols or ENSEMBL IDs — ENSEMBL IDs are auto-detected and converted)

## Usage Examples

### Selection pipeline preprocessing
```bash
python preprocess_for_selection.py \
    --input_file data.h5ad \
    --output_dir preprocessed_for_selection/ \
    --celltype_column celltype \
    --filter_methods scanpy no_filter \
    --hvg_option both \
    --n_components_pca 50 \
    --n_components_nmf 5
```

### Reference preprocessing for analysis scripts
```bash
python preprocess_for_analysis.py \
    --input_file data.h5ad \
    --output_file analysis_input.h5ad \
    --dimensionality_reduction both \
    --n_neighbors 15 \
    --n_nmf_components 5
```

### Evaluation panel preprocessing
```bash
python preprocess_for_evaluation.py \
    --input_file data.h5ad \
    --output_dir preprocessed_panels/ \
    --gene_lists_dir Selected-panels/ \
    --celltype_column celltypes_v2 \
    --dimensionality_reduction both \
    --n_neighbors 15
```

## Dependencies

- `scanpy >= 1.9`
- `anndata`
- `numpy`
- `pandas`
- `scikit-learn` (NMF, PCA)
- `scipy`

## Related Modules

- [Evaluation-module](../Evaluation-module/): Uses `process_data_for_panel_evaluation()` for panel h5ad creation
- [Selection-module](../Selection-module/): Consumes `preprocess_for_selection.py` output
- [Utility-module](../Utility-module/): Provides `convert_ensembl_to_gene_symbols()`

## Author

Helene Hemmer
