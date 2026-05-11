"""CLI wrapper — preprocess raw h5ad as canonical input for analysis scripts.

Delegates all logic to :func:`preprocess_reference_for_analysis_scripts` in
``preprocess_reference_for_evaluation.py``, which produces an AnnData with:

  - ``layers["counts"]``  — raw integer counts (for NMF in analysis scripts)
  - ``.X``                — log-normalised values
  - ``obsm["X_umap"]``    — UMAP embedding (required by stability-analysis feature plots)
  - PCA embeddings and Leiden clusterings

Usage
-----
python preprocess_for_analysis.py \\
    --input_file  /path/to/raw.h5ad \\
    --output_file /path/to/analysis_input.h5ad \\
    --celltype_column celltype

# Override dimensionality reduction / neighbourhood size:
python preprocess_for_analysis.py \\
    --input_file  /path/to/raw.h5ad \\
    --output_file /path/to/analysis_input.h5ad \\
    --celltype_column celltype \\
    --dimensionality_reduction pca \\
    --n_neighbors 15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make both the Preprocessing-module and the Evaluation-module
# importable regardless of the caller's working directory.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MODULES_DIR = _HERE.parent
_EVAL_MODULE = _MODULES_DIR / "Evaluation-module"

for _p in [str(_HERE), str(_EVAL_MODULE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preprocess_reference_for_evaluation import preprocess_reference_for_analysis_scripts  # noqa: E402

# Load DEFAULT_N_COMPONENTS_NMF by explicit path — avoids collision with
# Evaluation-module/_constants.py and Utility-module/_constants.py.
import importlib.util as _ilu
_cspec = _ilu.spec_from_file_location("_preproc_constants", _HERE / "_constants.py")
_cmod  = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_cmod)
DEFAULT_N_COMPONENTS_NMF = _cmod.DEFAULT_N_COMPONENTS_NMF
del _ilu, _cspec, _cmod

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess raw h5ad as canonical input for analysis scripts "
            "(stability analysis, k-fold evaluation, subset pipeline). "
            "Produces layers['counts'], normalised .X, UMAP, and Leiden clusters."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to raw h5ad file (raw counts + obs[celltype_column]).",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path for the preprocessed output h5ad.",
    )
    parser.add_argument(
        "--celltype_column",
        default="celltype",
        help="Column in .obs that contains cell-type labels.",
    )
    parser.add_argument(
        "--dimensionality_reduction",
        default="both",
        choices=["pca", "nmf", "both"],
        help="Dimensionality reduction to compute (used for UMAP).",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="Number of neighbours for UMAP construction.",
    )
    parser.add_argument(
        "--n_nmf_components",
        type=int,
        default=DEFAULT_N_COMPONENTS_NMF,
        help=f"Number of NMF components (default: {DEFAULT_N_COMPONENTS_NMF}).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        preprocess_reference_for_analysis_scripts(
            input_file=args.input_file,
            output_file=args.output_file,
            celltype_column=args.celltype_column,
            dimensionality_reduction=args.dimensionality_reduction,
            n_neighbors=args.n_neighbors,
            n_nmf_components=args.n_nmf_components,
        )
        return 0
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
