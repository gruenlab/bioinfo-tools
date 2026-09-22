# Utility-module

Shared helpers for AnnData validation and Ensembl ID conversion. See `CLAUDE.md` in this
directory for detail.

## Files

| File | Contents | In-cut status |
|---|---|---|
| `_validation.py` | `is_anndata_raw`, `is_anndata_raw_layer` — the one canonical "does this matrix hold raw counts?" check; hard-imported by Selection / Preprocessing / Evaluation. | live (`X_is_raw`, a divergent third implementation, was removed — no callers) |
| `_utils.py` | `convert_ensembl_to_gene_symbols` (MyGene.info). | live (`subset_data_to_gene_lists` and `perform_nmf_pca_per_celltype` removed — no in-cut caller; the latter's NMF kwargs contradicted the pinned `NMF_PREDICTOR_FIXED_KWARGS`) |
| `_resource_tracker.py` | `ResourceTracker` / `ResourceMetrics` — memory/CPU/time monitoring. | unused **within this cut**, kept only because `Modules_v2/Analysis-scripts/` instantiates it directly |
| `_constants.py` | `SAMPLE_SIZE_FOR_RAW_CHECK`, `MIN_CELLS_FOR_DIMRED`, NMF alpha defaults, `LOG_FORMAT` / `LOG_DATE_FORMAT` (exported, unused — each CLI hard-codes the format inline). | live |

**Removed — `_panel_utils.py`** (~1000 lines): selection-pipeline plumbing
(`save_experiment_parameters`, `save_gene_details`, `GeneProvenanceTracker`,
`get_component_panel_path` / `get_preprocessing_name` / `get_panel_cache_prefix`,
`load_cached_panel` / `save_panel_to_cache`) plus a generic `setup_logging` /
`log_memory_usage`. Nothing in this cut imported it — the CLIs that need logging define
their own `setup_logging` — and the name/cache helpers still used pre-rename strategy
names (`dt_nmf`, `args.dt_percentage`). Deleted (audit_2.md Q14 / B5); `Modules_v2` keeps
its own separate copy.

## Public API

`__init__.py` re-exports from `_validation` (`is_anndata_raw`, `is_anndata_raw_layer`) and
`_utils` (`convert_ensembl_to_gene_symbols`). `_resource_tracker` symbols are not exported.
Import directly:

```python
import sys
sys.path.insert(0, ".../Code/RecoVar/Utility-module")
from _validation import is_anndata_raw, is_anndata_raw_layer
```

## Notes

- `NMF_ALPHA_W/H/L1_RATIO` here duplicate the same constants in
  `Selection-module/_constants.py` and `Evaluation-module/_constants.py` ("keep in sync").

See `docs/doc-pipeline/audit_2.md` §2 / §3 (Q13, Q14) for the full picture.
