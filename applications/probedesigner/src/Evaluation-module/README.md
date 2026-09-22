# Evaluation-module

Probe-panel quality evaluation. See `CLAUDE.md` in this directory for the full detail
(the two gating axes, the pinned NMF config, per-metric conventions); this README is
the quick map.

## Entry points

| Script | Kind | Notes |
|---|---|---|
| `run_evaluation.py` | CLI (`--help`) | The orchestrator. `--mode preprocess/evaluate/both`; `--evaluation_type baseline/variability/biology/all/tangram_only`; `--include_tangram` gates Tangram **independently** of `--evaluation_type`. Writes `evaluation_parameters.json` and `logs/`. |
| `run_pca_evaluation.py` | CLI | Standalone PCA-space variability evaluation. |
| `run_ridge_evaluation.py` | CLI | Standalone Ridge-regression variability evaluation (raw + lognorm spaces). |
| `run_lognorm_evaluation.py` | CLI | Standalone NMF-in-lognorm-space variability evaluation. |
| `run_tangram_check.py` | CLI | Standalone Tangram reconstruction check. Own exit codes (0/1/2). Configures **no** logging — prints to stdout, errors to stderr. |

The four standalone scripts neither call nor are called by `run_evaluation.py`; each
requires `run_evaluation.py --mode preprocess` to have produced
`preprocessed_dir/full_transcriptome.h5ad` first, and each reuses
`_splits.generate_evaluation_splits`.

## Internal modules

| File | Contents |
|---|---|
| `_clustering.py` | Baseline metrics: kNN preservation, ARI/NMI, decision-tree cell-type ID, `compute_celltype_jaccard_vs_leiden`. Also carries a Spapros-ported `split_train_test_sets` / `uniform_samples` used only by `evaluate_celltype_identification`. |
| `nmf.py` | Built-in NMF variability metrics (`nmf_reconstruction`, `nmf_reconstruction_by_celltype`). |
| `metrics.py` | `calculate_mse`, `calculate_explained_variance` (5 aggregation modes via `mode=`), macro / weighted helpers. Delegates the core formulas to `nico2_lib.metrics`. |
| `pca.py`, `ridge.py`, `nmf_lognorm.py` | Alternative reconstruction implementations, each used only by its own standalone CLI. |
| `_tangram.py` | Tangram reconstruction wrappers (`reconstruct_with_tangram`, `run_tangram_reconstruction_check`), imported from `nico2_lib.predictors._tangram`. |
| `_preprocessing.py` | Panel subsetting + PCA embedding + a Leiden resolution search + reference preprocessing + `filter_reference_blacklist`. Reaches into `Selection-module/_filtering.apply_blacklist_filter` via `sys.path`. |
| `_splits.py` | `generate_evaluation_splits` — stratified k-fold, shared by NMF and Tangram. |
| `_filters.py` | Dataset-name parsing / panel filtering (`--strategies`, `--probeset_sizes`, …). |
| `_pathway.py` | Enrichr pathway enrichment (biology evaluation type; needs `gseapy`). |
| `_constants.py` | Defaults + `NMF_PREDICTOR_FIXED_KWARGS`. |

## `nico2_lib`

Not vendored into this cut. It is an external editable install at
`Code/bioinfo-tools/applications/nico2_lib/src/nico2_lib/`. Every entry point here
hard-fails at import without it. It provides `NmfPredictor`, `nico2_lib.metrics`, and
the Tangram unfiltered wrappers.

## Public API

`__init__.py` self-adds its directory to `sys.path` and re-exports flat names from
`_clustering`, `metrics`, `nmf`, and (best-effort) `pca` / `ridge`. Or import directly:

```python
import sys
sys.path.insert(0, ".../Code/RecoVar/Evaluation-module")
from _clustering import evaluate_clustering_quality, evaluate_neighborhood_preservation
from nmf import nmf_reconstruction, nmf_reconstruction_by_celltype
from _splits import generate_evaluation_splits
```

## Outputs

`{output_dir}/` = `evaluation_parameters.json`, `logs/`, `Baseline-Evaluation/`,
`Biology-Evaluation/`, `Variability-Evaluation/`, `Tangram-Evaluation/`. Standalones add
`Variability-Evaluation-{PCA,Ridge,Lognorm}/` and their own `*_evaluation_parameters.json`.
`run_tangram_check.py` writes `tangram_summary.json`. Full column schemas:
`docs/doc-pipeline/audit_2.md` §7.2.
