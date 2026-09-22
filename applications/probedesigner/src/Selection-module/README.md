# Selection-module

Gene-panel selection strategies for spatial transcriptomics probe design.

For the full narrative (strategy internals, RF cross-validation design, the 3-phase
NMF/PCA pool architecture, gap-filling, filtering order) see `CLAUDE.md` in this
directory. This README is the quick map.

## Entry points

| Script | Kind | What it does |
|---|---|---|
| `run_selection_pipeline.py` | CLI (`--help`) | The orchestrator. Routes single strategies to `run_single_selection` and `RecoVar` / `RecoVar_PCA` to `run_combination_selection`; applies the pre-selection blacklist filter once, computes Xenium mean-expression, writes `selection_parameters.json` and the log. |
| `run_single_selection.py` | library module | `run_single_selection()` — one strategy + blacklist → selection → Xenium filter → top-N. No argparse / `__main__` despite the `run_` prefix. |
| `run_combination_selection.py` | library module | `run_combination_selection()` — runs the RF (`rf_deg`) and dimred (`dimred_only`) components, combines with overlap resolution (RF wins), gap-fills. No argparse / `__main__`. |

## Strategies (`--strategy`)

`deg_only`, `rf_simple`, `rf_deg`, `dimred_only`, `hvg`, `random`, `RecoVar`,
`RecoVar_PCA`.

## Internal modules

| File | Contents |
|---|---|
| `_constants.py` | Defaults + `NMF_PREDICTOR_FIXED_KWARGS` (kept byte-identical to `Evaluation-module/_constants.py`). |
| `_gene_list_builder.py` | `GeneListBuilder` — the provenance object every strategy returns. |
| `_deg_selection.py` | `select_degs`, `filter_celltypes_by_min_cells`. |
| `_baseline_selection.py` | `select_highly_variable_genes`, `select_random_genes`. |
| `_rf_selection.py` | `select_genes_with_rf` + the `rf_deg` retry ladder + RF score cache. |
| `_dimred_selection.py` | `select_genes_from_nmf` / `select_genes_from_pca` — NMF/PCA is always fit independently per cell type (no all-cells-pooled "global" mode). Imports `calculate_explained_variance` from `Evaluation-module/metrics.py` with an inline fallback. |
| `_factor_aware.py` | Factor-aware duplicate resolution (`resolve_duplicates_factor_aware_*`). |
| `_filtering.py` | `apply_blacklist_filter`, `apply_xenium_filter_to_genelist`, `compute_global_mean_expression`. |

## Public API

`__init__.py` re-exports 19 symbols but uses package-relative imports, so it is not
importable under the flat-import scheme everyone actually uses. Import the internal
modules directly:

```python
import sys
sys.path.insert(0, ".../Code/RecoVar/Selection-module")
from _deg_selection import select_degs
from _rf_selection import select_genes_with_rf
from _dimred_selection import select_genes_from_nmf
```

## Outputs

Written into `--output_dir`: a `{strategy}_panel_information.csv` gene-list file (name via
`_gene_list_builder.panel_information_filename(strategy, reduction_type)` —
`deg_only_panel_information.csv`, `hvg_panel_information.csv`,
`random_panel_information.csv`, `rf_simple_panel_information.csv`,
`rf_deg_panel_information.csv`, `nmf_panel_information.csv`/`pca_panel_information.csv`
for `dimred_only`, or `RecoVar_panel_information.csv` for `RecoVar`/`RecoVar_PCA`), plus
`filtering_summary.json` (single strategies) or `combination_summary.json`
(`RecoVar`/`RecoVar_PCA`), `selection_parameters.json`, `{experiment_name}.log`, and
DEG/HVG/dimred-pool diagnostics + pkl caches.

`RecoVar_panel_information.csv` is the **only** combination-strategy output — it covers
every gene considered during combination (`in_panel` marks final membership), not just the
panel:

| Column | Meaning |
|---|---|
| `gene` | Gene symbol |
| `in_panel` | Final panel membership |
| `gene_source` | Panel rows: `"random forest"`/`"NMF"`/`"overlap"`/`force_include`/`"gap-fill (NMF)"`/`"gap-fill (random forest)"`. Non-panel rows: `"random forest"` / `"NMF"` |
| `informative_celltypes` | `primary_celltype` + `secondary_celltypes_nmf`, deduplicated, `|`-joined |
| `primary_celltype` | Value-based fallback: the gene's `rf_celltype` if it has one, else its dimred `celltype` |
| `secondary_celltypes_nmf` / `n_celltypes_nmf` | Other cell types whose NMF/PCA fit selected this gene, and their count — populated only when `primary_celltype` came from the dimred side |
| `gap_filled` / `gap_fill_strategy` | Whether/how a panel gene was added by gap-filling |
| `mean_expression` | From whichever builder `primary_celltype` was resolved from |

Full column schemas for every file (single-strategy `deg_only`/`hvg`/`random`, the RF and
dimred component files, and `RecoVar_panel_information.csv`): `Selection-module/CLAUDE.md`
"Output Files" section.
