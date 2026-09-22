# Preprocessing-module

Data preprocessing for the selection, evaluation, and analysis pipelines. See
`CLAUDE.md` in this directory for detail; this README is the quick map.

## Entry points

| Script | Kind | What it does |
|---|---|---|
| `preprocess_for_selection.py` | CLI (`--help`) + importable `preprocess_for_selection()` | Per `filter` × `hvg` combination: optional Scanpy QC filter → `normalize_total` (per-cell median) → `log1p` → optional HVG subset (`cell_ranger`, `DEFAULT_N_HVG=8000`) → PCA. Writes `layers["counts"]`. No TPM/variance normalisation, no NMF, no Leiden at this stage. |
| `preprocess_reference_for_evaluation.py` | CLI + importable `preprocess_reference_for_analysis_scripts()` | Reference h5ad with PCA/NMF/UMAP/Leiden + per-k kNN graphs (the NMF/Leiden/UMAP work is delegated to `Evaluation-module._preprocessing`). Writes atomically. Has no `--celltype_column` flag (hard-pinned to `"celltype"`). |
| `preprocess_for_evaluation.py` | CLI | Discovers Selection gene lists and builds one panel `.h5ad` each via `Evaluation-module._preprocessing.process_data_for_panel_evaluation`. **Currently incompatible with `run_selection_pipeline.py` output** — see `docs/doc-pipeline/audit_2.md` §7.3. |
| `preprocess_for_analysis.py` | CLI | Thin wrapper around `preprocess_reference_for_analysis_scripts()`. Configures logging at import (→ stdout). |
| `filter_blacklist_genes.py` | CLI | Writes a blacklist-filtered copy of a raw h5ad + `{stem}_filter_parameters.json`. Imports `apply_blacklist_filter` from `Selection-module`. Not covered by `CLAUDE.md`. |

## Internal module

`_constants.py` — `DEFAULT_N_COMPONENTS_PCA`, `DEFAULT_N_COMPONENTS_NMF`,
`DEFAULT_N_HVG` (8000), `DEFAULT_HVG_FLAVOR` (`cell_ranger`), QC thresholds,
`FILTER_METHODS`. (`LOG_FORMAT` / `LOG_DATE_FORMAT` are exported but unused — every
script hard-codes the format string.)

## Public API

`__init__.py` exports `preprocess_for_selection` and `preprocess_for_analysis` via
package-relative imports (not importable flat). The `preprocess_for_analysis` function
in `preprocess_for_selection.py` has no in-cut caller.

`preprocess_for_selection.py` guards its constants import as
`try: from ._constants import … except ImportError: from _constants import …`;
`preprocess_reference_for_evaluation.py` and `preprocess_for_analysis.py` load their
`_constants.py` by explicit path via `importlib.util` (deliberate, to avoid the
name collision with `Evaluation-module/_constants.py`).

## Outputs

`preprocess_for_selection.py`: per `{filter_name}_{hvg_name}/` dir — `preprocessed.h5ad`
+ `metadata.json` (**not** `pca_loadings.csv` / `nmf_loadings.csv`).
Others: a single h5ad at `--output_file`. Details: `docs/doc-pipeline/audit_2.md` §7.3.
