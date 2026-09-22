# Utility-module — shared helpers

Version: 2.0.0

Cross-cutting helpers for AnnData validation and Ensembl ID conversion. Small now: after
the 2026 audit, everything that had no caller in this cut was removed
(`_panel_utils.py`, `X_is_raw`, `subset_data_to_gene_lists`, `perform_nmf_pca_per_celltype`).

## Files

### `_validation.py`
- `is_anndata_raw(adata)` — heuristic: does `.X` (or `adata.raw`) look like raw integer
  counts? Samples up to `SAMPLE_SIZE_FOR_RAW_CHECK` non-zero values and checks integrality.
- `is_anndata_raw_layer(adata, layer_name)` — same check on a named layer.

The one canonical raw-count check in the cut. Every user (`Selection-module/_rf_selection.py`
& `_dimred_selection.py`, `Preprocessing-module/preprocess_for_selection.py`, and 8
`Evaluation-module` files) now **hard-imports** it (`sys.path` entry + `from _validation
import …`) — the old per-file `except ImportError:` reimplementations / `return True`
stubs were removed (audit_2.md Q13). A missing `_validation.py` therefore surfaces as a
loud `ImportError` at load, not a silent guess. `X_is_raw` (a third, divergent
implementation) was deleted — zero callers.

### `_utils.py`
- `convert_ensembl_to_gene_symbols(adata, ...)` — MyGene.info lookup; the only function
  here with an in-cut caller (`Preprocessing-module/preprocess_for_evaluation.py`).
- `subset_data_to_gene_lists(...)` and `perform_nmf_pca_per_celltype(...)` were removed —
  both were dead in this cut (no caller), and the latter's NMF kwargs
  (`init="nndsvda"`, `beta_loss="kullback-leibler"`, `solver="mu"`) contradicted the pinned
  `NMF_PREDICTOR_FIXED_KWARGS` (`solver="cd"`, Frobenius, `nndsvd`) that `Selection-module`
  and `Evaluation-module` are required to share.

### `_resource_tracker.py`
`ResourceTracker` / `ResourceMetrics` — memory/CPU/wall-time monitoring context object.
**Unused anywhere in this cut and not re-exported by `__init__.py`.** Kept only because
`Modules_v2/Analysis-scripts/` instantiates it directly — do not add new in-cut callers.

### `_constants.py`
`SAMPLE_SIZE_FOR_RAW_CHECK = 10_000`, `MIN_CELLS_FOR_DIMRED = 10`,
`DEFAULT_NMF_ALPHA_W/H/L1_RATIO = 0.0` (duplicated in `Selection-module/_constants.py`
and `Evaluation-module/_constants.py`), `LOG_FORMAT` / `LOG_DATE_FORMAT` (exported, unused
— each CLI hard-codes the format string in its own `setup_logging`).

### Removed: `_panel_utils.py`
~1000 lines of selection-pipeline plumbing (`save_experiment_parameters`,
`save_gene_details`, `GeneProvenanceTracker`, `get_component_panel_path`,
`get_preprocessing_name`, `get_panel_cache_prefix`, `load_cached_panel`,
`save_panel_to_cache`, plus a generic `setup_logging` / `log_memory_usage`). Deleted in the
2026 audit (audit_2.md Q14 / B5): nothing in this cut imported it — the CLIs that need
logging define their own `setup_logging` locally, and the panel-name / cache helpers still
referenced pre-rename strategy names (`dt_nmf`, `args.dt_percentage`). `Modules_v2` keeps
its own separate copy, unaffected.

## Public API

`__init__.py` re-exports (package-relative): from `_validation` — `is_anndata_raw`,
`is_anndata_raw_layer`; from `_utils` — `convert_ensembl_to_gene_symbols`.

Because the directory is hyphenated, import the internal modules directly under a
`sys.path` entry rather than `import utility`.

## Conventions

- Type annotations + Google-style docstrings on public functions.
- `from __future__ import annotations` first.
- Module-level `log = logging.getLogger(__name__)`.
- No `importlib.util` for internal modules; no hardcoded absolute paths; no
  module-level mutable globals.
