"""Dimension reduction (NMF/PCA) gene selection with factor-aware resolution.

This module provides gene selection based on NMF or PCA loadings. Dimensionality
reduction is always fitted independently within each cell type (there is no global,
all-cells-pooled mode). Integrates factor-aware duplicate resolution from
_factor_aware.py.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from sklearn.decomposition import PCA
from nico2_lib.predictors._nmf._nmf_pred import NmfPredictor

# Import the canonical raw-count check from Utility-module (hard import — one _validation.py
# in the cut, only numpy/scipy/anndata deps; a failure here means a broken checkout).
SCRIPT_DIR = Path(__file__).parent.absolute()
UTILITY_DIR = SCRIPT_DIR.parent / "Utility-module"
sys.path.insert(0, str(UTILITY_DIR))
from _validation import is_anndata_raw, is_anndata_raw_layer
from _nmf_objective import resolve_nmf_objective, describe_nmf_objective

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
# --- load THIS directory's _constants.py by path (sibling dirs share the name) ---
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _cpath
_cspec = _ilu.spec_from_file_location("_constants", _cpath(__file__).resolve().parent / "_constants.py")
_sys.modules["_constants"] = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_sys.modules["_constants"])


from _constants import (
    COL_CELLTYPE,
    DEFAULT_MIN_CELLS_PER_CELLTYPE,
    DEFAULT_PROBESET_SIZE,
    DEFAULT_NMF_COMPONENTS,
    DEFAULT_N_COMPONENTS_PCA,
    DEFAULT_RANDOM_STATE,
    NMF_PREDICTOR_FIXED_KWARGS,
    DEFAULT_MIN_XENIUM_EXPRESSION,
    DEFAULT_MAX_XENIUM_EXPRESSION,
)
from _factor_aware import resolve_duplicates_factor_aware_per_celltype
from _gene_list_builder import GeneListBuilder, panel_information_filename

logger = logging.getLogger(__name__)


# ============================================================================
# NMF/PCA COMPUTATION FUNCTIONS
# ============================================================================


def _resolve_n_jobs(n_jobs: int, n_tasks: int) -> int:
    """Normalise a requested worker count for the per-celltype dimred pools.

    ``-1`` → all cores (sklearn convention); anything < 1 (other than -1) falls
    back to sequential; never spawn more workers than there are cell-type tasks.
    """
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs < 1:
        logger.warning(f"Invalid n_jobs={n_jobs}; falling back to sequential execution")
        n_jobs = 1
    return max(1, min(n_jobs, n_tasks))


def _pin_blas_threads() -> None:
    """ProcessPool initializer: pin BLAS/OpenMP to 1 thread per worker.

    Each ``NmfPredictor.fit`` / ``sklearn.PCA`` call is itself multithreaded via
    numpy's BLAS; without this, ``n_jobs`` worker processes each spawning a full
    BLAS thread pool oversubscribes the CPU and is often *slower* than sequential.
    """
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_var] = "1"

def _resolve_raw_dimred_input(adata: AnnData, context: str) -> Tuple[object, object, str]:
    """Resolve raw-count input for dimred, aligned to the current gene set."""
    if 'counts' in adata.layers and is_anndata_raw_layer(adata, 'counts'):
        return adata.layers['counts'], adata.var_names, "adata.layers['counts']"

    if hasattr(adata, 'raw') and adata.raw is not None and is_anndata_raw(adata.raw):
        positions = adata.raw.var_names.get_indexer(adata.var_names)
        missing = adata.var_names[positions < 0].tolist()
        if missing:
            preview = ", ".join(map(str, missing[:5]))
            raise ValueError(
                f"Cannot align adata.raw.X to current genes for {context}; "
                f"{len(missing)} genes are missing from adata.raw.var_names "
                f"(examples: {preview})"
            )
        return adata.raw.X[:, positions], adata.var_names, "aligned adata.raw.X"

    checked = []
    if 'counts' in adata.layers:
        checked.append("adata.layers['counts']")
    if hasattr(adata, 'raw') and adata.raw is not None:
        checked.append("adata.raw.X")
    checked_msg = ", ".join(checked) if checked else "no raw-count containers"
    raise ValueError(
        f"dimred_counts_input='raw': no valid raw integer counts found for {context} "
        f"(checked {checked_msg})"
    )


def _resolve_dimred_input(
    adata: AnnData,
    dimred_counts_input: str,
    context: str,
) -> Tuple[object, object, str]:
    """Resolve the matrix used for NMF/PCA without mutating adata.X."""
    if dimred_counts_input == "raw":
        matrix, gene_names, source = _resolve_raw_dimred_input(adata, context)
        logger.info(f"Using {source} for {context} (verified as raw counts)")
        return matrix, gene_names, source

    if dimred_counts_input == "lognorm":
        if adata.X is None:
            raise ValueError(f"dimred_counts_input='lognorm': adata.X is None for {context}")
        if is_anndata_raw(adata):
            raise ValueError(
                f"dimred_counts_input='lognorm': adata.X appears to contain raw integer counts, "
                f"not log-normalized data. Normalize before selection."
            )
        logger.info(f"Using adata.X (log-normalized, verified) for {context}")
        return adata.X, adata.var_names, "adata.X"

    raise ValueError(
        f"Unknown dimred_counts_input='{dimred_counts_input}'. Choose 'raw' or 'lognorm'."
    )


def compute_nmf_per_celltype(
    adata: AnnData,
    celltype_column: str = COL_CELLTYPE,
    n_components: int = DEFAULT_NMF_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    min_cells: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
    n_jobs: int = 1,
    parallel_backend: str = 'process',
    cache_dir: Optional[str] = None,
    dimred_counts_input: str = "raw",
    require_existing_cache: bool = False,
    nmf_objective: str = "auto",
) -> dict[str, pd.DataFrame]:
    """Compute per-celltype NMF on the selected count matrix.

    Args:
        adata: Annotated data matrix.
        celltype_column: Column with cell type labels.
        n_components: Number of NMF factors per celltype.
        random_state: Random seed.
        min_cells: Minimum cells per celltype.
        n_jobs: Number of workers for per-celltype fits. 1 (default) keeps sequential
            behavior; -1 uses all cores; capped at the number of cell types.
        parallel_backend: Parallel backend when n_jobs > 1 ('process' or 'thread').
        cache_dir: Directory to save/load cached models. When given, fits are loaded
            from ``{cache_dir}/nmf_models/per_celltype_nmf_{dimred_counts_input}_{nmf_objective}.pkl``
            if present and matching ``n_components``, otherwise recomputed and written.
        dimred_counts_input: Which count matrix to use as NMF input.
            ``"raw"`` (default): raw integer counts from ``adata.raw.X`` or
            ``adata.layers["counts"]`` (validated with ``is_anndata_raw_layer``).
            ``"lognorm"``: log-normalised counts from ``adata.X``.
        require_existing_cache: If ``True``, raise instead of recomputing when the
            cache file is missing, unreadable, or built with a different
            ``n_components``.
        nmf_objective: NMF factorization objective — ``"auto"`` (default) derives the
            solver/beta_loss from ``dimred_counts_input`` (see ``_nmf_objective.py``);
            ``"frobenius"``/``"kl"`` force that objective regardless of input.

    Returns:
        Dict mapping celltype → loadings DataFrame (genes × factors).

    Raises:
        ValueError: If counts not available or celltype column missing
    """
    logger.info(f"Computing per-celltype NMF (n_components={n_components})...")
    _pred_kwargs = resolve_nmf_objective(dimred_counts_input, nmf_objective)
    logger.info(describe_nmf_objective(dimred_counts_input, nmf_objective))

    # Check for cached model
    if cache_dir:
        cache_file = os.path.join(cache_dir, 'nmf_models', f'per_celltype_nmf_{dimred_counts_input}_{nmf_objective}.pkl')
        if os.path.exists(cache_file):
            logger.info(f"Loading cached per-celltype NMF models from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                # Validate cache
                if cached_data['n_components'] == n_components:
                    logger.info(f"✓ Successfully loaded cached per-celltype NMF: {len(cached_data['loadings'])} celltypes")
                    return cached_data['loadings']
                else:
                    if require_existing_cache:
                        raise ValueError(
                            f"Required per-celltype NMF cache has n_components={cached_data['n_components']}, "
                            f"requested n_components={n_components}: {cache_file}"
                        )
                    logger.warning(
                        f"Cache n_components mismatch: cached={cached_data['n_components']}, "
                        f"requested={n_components}. Recomputing..."
                    )
            except Exception as e:
                if require_existing_cache:
                    raise RuntimeError(
                        f"Required per-celltype NMF cache exists but could not be loaded: {cache_file}"
                    ) from e
                logger.warning(f"Failed to load cached model: {e}. Recomputing...")
        elif require_existing_cache:
            raise FileNotFoundError(
                f"Required per-celltype NMF cache not found: {cache_file}"
            )

    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_column}' not in adata.obs")

    counts_full, gene_names, _ = _resolve_dimred_input(adata, dimred_counts_input, "per-celltype NMF")

    celltype_loadings = {}

    # Build task list — one independent NMF fit per celltype. NMF config is the
    # pinned set in _constants.NMF_* (shared with Evaluation-module); only
    # n_components and random_state vary per call.
    nmf_tasks = []
    for celltype in adata.obs[celltype_column].unique():
        mask = (adata.obs[celltype_column] == celltype).values
        n_cells_ct = int(mask.sum())

        if n_cells_ct < min_cells:
            logger.warning(f"Skipping {celltype}: only {n_cells_ct} cells (need >={min_cells})")
            continue

        logger.info(f"  {celltype}: {n_cells_ct} cells, n_components={n_components}")
        counts_ct = counts_full[mask, :]
        nmf_tasks.append((celltype, n_cells_ct, counts_ct, gene_names, n_components, random_state, _pred_kwargs))

    n_jobs = _resolve_n_jobs(n_jobs, len(nmf_tasks))

    if n_jobs == 1:
        for task in nmf_tasks:
            celltype, n_cells_ct, nmf_result, error_msg = _compute_single_celltype_nmf_task(task)
            if error_msg is not None:
                logger.warning(f"Skipping {celltype}: NMF task failed with error: {error_msg}")
                continue
            if nmf_result is None:
                continue

            celltype_loadings[celltype] = nmf_result
            logger.info(f"  ✓ {celltype}: {nmf_result.shape}")
    else:
        backend = parallel_backend.lower()
        if backend not in {'process', 'thread'}:
            logger.warning(
                f"Unsupported parallel_backend='{parallel_backend}', defaulting to 'process'"
            )
            backend = 'process'

        logger.info(
            f"Running per-celltype NMF in parallel with n_jobs={n_jobs}, backend={backend}"
        )

        _kw = {"initializer": _pin_blas_threads} if backend == 'process' else {}
        executor_cls = ProcessPoolExecutor if backend == 'process' else ThreadPoolExecutor
        with executor_cls(max_workers=n_jobs, **_kw) as executor:
            # executor.map preserves input order → results are deterministic and
            # identical to the sequential path (tasks are pure, seed is threaded).
            for celltype, n_cells_ct, nmf_result, error_msg in executor.map(
                _compute_single_celltype_nmf_task,
                nmf_tasks,
            ):
                if error_msg is not None:
                    logger.warning(f"Skipping {celltype}: NMF task failed with error: {error_msg}")
                    continue
                if nmf_result is None:
                    continue

                celltype_loadings[celltype] = nmf_result
                logger.info(f"  ✓ {celltype}: {nmf_result.shape}")

    logger.info(f"✓ Per-celltype NMF complete: {len(celltype_loadings)} celltypes")
    
    # Cache models if directory provided
    if cache_dir:
        model_dir = os.path.join(cache_dir, 'nmf_models')
        os.makedirs(model_dir, exist_ok=True)
        cache_file = os.path.join(model_dir, f'per_celltype_nmf_{dimred_counts_input}_{nmf_objective}.pkl')
        try:
            cache_data = {
                'loadings': celltype_loadings,
                'n_components': n_components,
                'celltypes': list(celltype_loadings.keys()),
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"✓ Saved per-celltype NMF models to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")

    return celltype_loadings


def _compute_single_celltype_nmf(
    celltype: str,
    counts_ct,
    gene_names: pd.Index,
    n_components: int,
    random_state: int,
    predictor_kwargs: Optional[dict] = None,
) -> Optional[pd.DataFrame]:
    """Run NMF for a single celltype and return its gene x factor loadings."""
    if scipy.sparse.issparse(counts_ct):
        counts_dense = counts_ct.toarray()
    else:
        counts_dense = np.asarray(counts_ct)

    std = np.std(counts_dense, axis=0)
    nonzero_index = np.where(std > 0)[0].astype(int)
    n_features = len(nonzero_index)

    if n_features < n_components:
        logger.warning(
            f"Skipping {celltype}: only {n_features} features with non-zero std "
            f"(need >= {n_components})"
        )
        return None

    counts_dense = counts_dense[:, nonzero_index]

    # NMF config resolved via _nmf_objective.resolve_nmf_objective (data-space-driven,
    # overridable via --nmf_objective); falls back to the pinned Frobenius default
    # (see _constants.NMF_*) when called without predictor_kwargs.
    _predictor = NmfPredictor(
        n_components=n_components,
        embedding_size=n_components,
        random_state=random_state,
        **(predictor_kwargs if predictor_kwargs is not None else dict(NMF_PREDICTOR_FIXED_KWARGS)),
    ).fit(counts_dense)
    H = _predictor.h_reference

    factor_names = [f"{celltype}_NMF_{i+1}" for i in range(n_components)]
    loadings_df = pd.DataFrame(
        H.T,
        index=gene_names[nonzero_index],
        columns=factor_names,
    )
    return loadings_df


def _compute_single_celltype_nmf_task(
    task: tuple[str, int, object, pd.Index, int, int, Optional[dict]],
) -> tuple[str, int, Optional[pd.DataFrame], Optional[str]]:
    """Task wrapper for optional parallel execution of per-celltype NMF."""
    celltype, n_cells_ct, counts_ct, gene_names, n_components, random_state, predictor_kwargs = task
    try:
        nmf_result = _compute_single_celltype_nmf(
            celltype=celltype,
            counts_ct=counts_ct,
            gene_names=gene_names,
            n_components=n_components,
            random_state=random_state,
            predictor_kwargs=predictor_kwargs,
        )
        return celltype, n_cells_ct, nmf_result, None
    except Exception as exc:
        return celltype, n_cells_ct, None, str(exc)


def compute_pca_per_celltype(
    adata: AnnData,
    celltype_column: str = COL_CELLTYPE,
    n_components: int = DEFAULT_NMF_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    min_cells: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
    n_jobs: int = 1,
    parallel_backend: str = 'process',
    cache_dir: Optional[str] = None,
    dimred_counts_input: str = "raw",
) -> dict[str, pd.DataFrame]:
    """Compute per-celltype PCA on the selected dimred input matrix.

    Args:
        adata: Annotated data matrix
        celltype_column: Column with cell type labels
        n_components: Number of PCs per celltype
        random_state: Random seed
        min_cells: Minimum cells per celltype
        n_jobs: Number of workers for per-celltype fits. 1 (default) keeps sequential
            behavior; -1 uses all cores; capped at the number of cell types.
        parallel_backend: Parallel backend when n_jobs > 1 ('process' or 'thread').
        cache_dir: Directory to save/load cached models (optional)
        dimred_counts_input: Which count matrix to use as PCA input.

    Returns:
        Dict mapping celltype → loadings DataFrame (genes × PCs).
    """
    logger.info(f"Computing per-celltype PCA (n_components={n_components})...")
    
    # Check for cached model
    if cache_dir:
        cache_file = os.path.join(cache_dir, 'pca_models', f'per_celltype_pca_{dimred_counts_input}.pkl')
        if os.path.exists(cache_file):
            logger.info(f"Loading cached per-celltype PCA models from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"✓ Successfully loaded cached per-celltype PCA: {len(cached_data['loadings'])} celltypes")
                return cached_data['loadings']
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}. Recomputing...")

    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_column}' not in adata.obs")

    X_full, gene_names, _ = _resolve_dimred_input(adata, dimred_counts_input, "per-celltype PCA")

    celltype_loadings = {}

    pca_tasks = []
    for celltype in adata.obs[celltype_column].unique():
        mask = (adata.obs[celltype_column] == celltype).values
        n_cells_ct = int(mask.sum())

        if n_cells_ct < min_cells:
            logger.warning(f"Skipping {celltype}: only {n_cells_ct} cells (need >={min_cells})")
            continue

        logger.info(f"  {celltype}: {n_cells_ct} cells")
        X_ct = X_full[mask, :]
        pca_tasks.append((celltype, n_cells_ct, X_ct, gene_names, n_components, random_state))

    n_jobs = _resolve_n_jobs(n_jobs, len(pca_tasks))

    if n_jobs == 1:
        for task in pca_tasks:
            celltype, n_cells_ct, pca_result, error_msg = _compute_single_celltype_pca_task(task)
            if error_msg is not None:
                logger.warning(f"Skipping {celltype}: PCA task failed with error: {error_msg}")
                continue
            if pca_result is None:
                continue

            loadings_df, total_marginal_r2 = pca_result
            celltype_loadings[celltype] = loadings_df
            logger.info(
                f"  ✓ {celltype}: {loadings_df.shape}, "
                f"marginal explained variance (sklearn)={total_marginal_r2:.4f}"
            )
    else:
        backend = parallel_backend.lower()
        if backend not in {'process', 'thread'}:
            logger.warning(
                f"Unsupported parallel_backend='{parallel_backend}', defaulting to 'process'"
            )
            backend = 'process'

        logger.info(
            f"Running per-celltype PCA in parallel with n_jobs={n_jobs}, backend={backend}"
        )

        _kw = {"initializer": _pin_blas_threads} if backend == 'process' else {}
        executor_cls = ProcessPoolExecutor if backend == 'process' else ThreadPoolExecutor
        with executor_cls(max_workers=n_jobs, **_kw) as executor:
            for celltype, n_cells_ct, pca_result, error_msg in executor.map(
                _compute_single_celltype_pca_task,
                pca_tasks,
            ):
                if error_msg is not None:
                    logger.warning(f"Skipping {celltype}: PCA task failed with error: {error_msg}")
                    continue
                if pca_result is None:
                    continue

                loadings_df, total_marginal_r2 = pca_result
                celltype_loadings[celltype] = loadings_df
                logger.info(
                    f"  ✓ {celltype}: {loadings_df.shape}, "
                    f"marginal explained variance (sklearn)={total_marginal_r2:.4f}"
                )

    logger.info(f"✓ Per-celltype PCA complete: {len(celltype_loadings)} celltypes")
    
    # Cache models if directory provided
    if cache_dir:
        model_dir = os.path.join(cache_dir, 'pca_models')
        os.makedirs(model_dir, exist_ok=True)
        cache_file = os.path.join(model_dir, f'per_celltype_pca_{dimred_counts_input}.pkl')
        try:
            cache_data = {
                'loadings': celltype_loadings,
                'n_components': n_components,
                'celltypes': list(celltype_loadings.keys()),
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"✓ Saved per-celltype PCA models to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")

    return celltype_loadings


def _compute_single_celltype_pca(
    celltype: str,
    X_ct,
    gene_names: pd.Index,
    n_components: int,
    random_state: int,
) -> Tuple[pd.DataFrame, float]:
    """Run PCA for a single celltype and return loadings + the sklearn marginal EV ratio."""
    if scipy.sparse.issparse(X_ct):
        X = X_ct.toarray()
    else:
        X = np.asarray(X_ct)

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    H = pca.components_

    component_names = [f"{celltype}_PC_{i+1}" for i in range(n_components)]
    total_marginal_r2 = float(pca.explained_variance_ratio_.sum())
    loadings_df = pd.DataFrame(
        H.T,
        index=gene_names,
        columns=component_names,
    )
    return loadings_df, total_marginal_r2


def _compute_single_celltype_pca_task(
    task: tuple[str, int, object, pd.Index, int, int],
) -> tuple[str, int, Optional[Tuple[pd.DataFrame, float]], Optional[str]]:
    """Task wrapper for optional parallel execution of per-celltype PCA."""
    celltype, n_cells_ct, X_ct, gene_names, n_components, random_state = task
    try:
        pca_result = _compute_single_celltype_pca(
            celltype=celltype,
            X_ct=X_ct,
            gene_names=gene_names,
            n_components=n_components,
            random_state=random_state,
        )
        return celltype, n_cells_ct, pca_result, None
    except Exception as exc:
        return celltype, n_cells_ct, None, str(exc)


# ============================================================================
# GENE SELECTION FUNCTIONS
# ============================================================================


def select_genes_from_nmf(
    adata: AnnData,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    celltype_column: str = COL_CELLTYPE,
    n_components: int = DEFAULT_NMF_COMPONENTS,
    pool_size_per_celltype: int = 200,
    min_cells_per_celltype: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
    random_state: int = DEFAULT_RANDOM_STATE,
    nmf_n_jobs: int = 1,
    nmf_parallel_backend: str = 'process',
    nmf_loadings_per_celltype: Optional[dict[str, pd.DataFrame]] = None,
    results_dir: Optional[str] = None,
    nmf_model_cache_dir: Optional[str] = None,
    mean_expr_per_ct: Optional[dict] = None,
    xenium_min_expr: float = DEFAULT_MIN_XENIUM_EXPRESSION,
    xenium_max_expr: float = DEFAULT_MAX_XENIUM_EXPRESSION,
    dimred_counts_input: str = "raw",
    require_nmf_model_cache: bool = False,
    nmf_objective: str = "auto",
) -> GeneListBuilder:
    """Select genes using per-celltype NMF loadings with factor-aware duplicate resolution.

    NMF is fitted independently within each cell type; there is no global (all-cells
    pooled) mode.

    Args:
        adata: Annotated data matrix
        probeset_size: Target number of genes
        celltype_column: Cell type column
        n_components: Number of NMF factors to use
        pool_size_per_celltype: Phase-1 pool size per cell type
            (distributed across that cell type's factors).
        min_cells_per_celltype: Minimum cells per cell type
        random_state: Random seed for reproducibility
        nmf_n_jobs: Workers for per-celltype NMF fitting when loadings are computed internally
            (1 = sequential, -1 = all cores)
        nmf_parallel_backend: Backend for per-celltype NMF parallelism ('process' or 'thread')
        nmf_loadings_per_celltype: Pre-computed per-celltype loadings {celltype: DataFrame}
        results_dir: Directory to save selection results
        nmf_model_cache_dir: Shared directory for NMF model pkl files (reused across
            different probeset sizes / strategies). Takes priority over results_dir
            for model caching. Leave None to use results_dir (old behaviour).
        mean_expr_per_ct: Per-celltype mean expression dict. When provided, the Phase-1
            pool is gated to genes with mean expression in
            [xenium_min_expr, xenium_max_expr]; when None the gate is skipped.
        xenium_min_expr: Lower bound for the Phase-1 pool expression gate. Defaults to
            DEFAULT_MIN_XENIUM_EXPRESSION; pass the CLI-resolved --xenium_min_expr to
            honour a custom bound on the dimred path.
        xenium_max_expr: Upper bound for the Phase-1 pool expression gate. Defaults to
            DEFAULT_MAX_XENIUM_EXPRESSION; pass the CLI-resolved --xenium_max_expr.
        dimred_counts_input: Matrix used for the internal NMF fit — ``"raw"`` (default,
            integer counts) or ``"lognorm"`` (``adata.X``). Ignored when loadings are
            supplied directly.
        require_nmf_model_cache: If ``True``, raise instead of recomputing when a
            required NMF model cache is missing.
        nmf_objective: NMF factorization objective — ``"auto"`` (default) derives the
            solver/beta_loss from ``dimred_counts_input``; ``"frobenius"``/``"kl"`` force
            that objective regardless of input. See ``_nmf_objective.py``.

    Returns:
        GeneListBuilder with factor assignments and provenance tracking

    Examples:
        >>> builder = select_genes_from_nmf(
        ...     adata,
        ...     probeset_size=500,
        ...     nmf_loadings_per_celltype=celltype_loadings
        ... )
    """
    # Resolve model cache directory: explicit nmf_model_cache_dir takes priority,
    # falling back to results_dir so existing behaviour is preserved.
    _nmf_cache_dir = nmf_model_cache_dir if nmf_model_cache_dir else results_dir

    # Compute per-celltype loadings if not provided
    if nmf_loadings_per_celltype is None:
        logger.info("Computing per-celltype NMF loadings...")
        if _nmf_cache_dir:
            logger.info(f"NMF model cache dir: {_nmf_cache_dir}")
        nmf_loadings_per_celltype = compute_nmf_per_celltype(
            adata=adata,
            celltype_column=celltype_column,
            n_components=n_components,
            random_state=random_state,
            min_cells=min_cells_per_celltype,
            n_jobs=nmf_n_jobs,
            parallel_backend=nmf_parallel_backend,
            cache_dir=_nmf_cache_dir,
            dimred_counts_input=dimred_counts_input,
            require_existing_cache=require_nmf_model_cache,
            nmf_objective=nmf_objective,
        )

    return _select_genes_from_dimred(
        adata=adata,
        probeset_size=probeset_size,
        reduction_type="nmf",
        n_components=n_components,
        pool_size_per_celltype=pool_size_per_celltype,
        dimred_loadings_per_celltype=nmf_loadings_per_celltype,
        results_dir=results_dir,
        mean_expr_per_ct=mean_expr_per_ct,
        xenium_min_expr=xenium_min_expr,
        xenium_max_expr=xenium_max_expr,
    )


def select_genes_from_pca(
    adata: AnnData,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    celltype_column: str = COL_CELLTYPE,
    n_components: int = DEFAULT_N_COMPONENTS_PCA,
    pool_size_per_celltype: int = 200,
    top_n_pcs: int = 5,
    min_cells_per_celltype: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
    random_state: int = DEFAULT_RANDOM_STATE,
    pca_n_jobs: int = 1,
    pca_parallel_backend: str = 'process',
    pca_loadings_per_celltype: Optional[dict[str, pd.DataFrame]] = None,
    results_dir: Optional[str] = None,
    nmf_model_cache_dir: Optional[str] = None,
    mean_expr_per_ct: Optional[dict] = None,
    xenium_min_expr: float = DEFAULT_MIN_XENIUM_EXPRESSION,
    xenium_max_expr: float = DEFAULT_MAX_XENIUM_EXPRESSION,
    dimred_counts_input: str = "raw",
) -> GeneListBuilder:
    """Select genes using per-celltype PCA loadings with factor-aware duplicate resolution.

    PCA is fitted independently within each cell type; there is no global (all-cells
    pooled) mode.

    Args:
        adata: Annotated data matrix
        probeset_size: Target number of genes
        celltype_column: Cell type column
        n_components: Total number of PCs computed
        pool_size_per_celltype: Phase-1 pool size per cell type
            (distributed across that cell type's PCs).
        top_n_pcs: Number of top PCs to use for gene selection
        min_cells_per_celltype: Minimum cells per cell type
        random_state: Random seed for reproducibility
        pca_n_jobs: Number of workers for per-celltype PCA
            (1 = sequential, -1 = all cores)
        pca_parallel_backend: Parallel backend for per-celltype PCA ('process' or 'thread')
        pca_loadings_per_celltype: Pre-computed per-celltype loadings {celltype: DataFrame}
        results_dir: Directory to save selection results
        nmf_model_cache_dir: Shared directory for PCA model pkl files (reused across
            different probeset sizes / strategies). Takes priority over results_dir
            for model caching. Leave None to use results_dir (old behaviour).
        mean_expr_per_ct: Per-celltype mean expression dict. When provided, the Phase-1
            pool is gated to genes with mean expression in
            [xenium_min_expr, xenium_max_expr]; when None the gate is skipped.
        xenium_min_expr: Lower bound for the Phase-1 pool expression gate. Defaults to
            DEFAULT_MIN_XENIUM_EXPRESSION; pass the CLI-resolved --xenium_min_expr to
            honour a custom bound on the dimred path.
        xenium_max_expr: Upper bound for the Phase-1 pool expression gate. Defaults to
            DEFAULT_MAX_XENIUM_EXPRESSION; pass the CLI-resolved --xenium_max_expr.
        dimred_counts_input: Matrix used for the internal PCA fit — ``"raw"`` (default)
            or ``"lognorm"`` (``adata.X``). Ignored when loadings are supplied directly.

    Returns:
        GeneListBuilder with factor assignments and provenance tracking

    Examples:
        >>> builder = select_genes_from_pca(
        ...     adata,
        ...     probeset_size=500,
        ...     top_n_pcs=5,
        ...     pca_loadings_per_celltype=celltype_loadings
        ... )
    """
    # Resolve model cache directory: explicit nmf_model_cache_dir takes priority,
    # falling back to results_dir so existing behaviour is preserved.
    _pca_cache_dir = nmf_model_cache_dir if nmf_model_cache_dir else results_dir

    # Compute per-celltype loadings if not provided
    if pca_loadings_per_celltype is None:
        logger.info("Computing per-celltype PCA loadings...")
        if _pca_cache_dir:
            logger.info(f"PCA model cache dir: {_pca_cache_dir}")
        pca_loadings_per_celltype = compute_pca_per_celltype(
            adata=adata,
            celltype_column=celltype_column,
            n_components=n_components,
            random_state=random_state,
            min_cells=min_cells_per_celltype,
            n_jobs=pca_n_jobs,
            parallel_backend=pca_parallel_backend,
            cache_dir=_pca_cache_dir,
            dimred_counts_input=dimred_counts_input,
        )

    # For PCA, limit to top N PCs
    n_components_to_use = min(top_n_pcs, n_components)
    logger.info(f"PCA: Using top {n_components_to_use} PCs (out of {n_components})")

    return _select_genes_from_dimred(
        adata=adata,
        probeset_size=probeset_size,
        reduction_type="pca",
        n_components=n_components_to_use,
        pool_size_per_celltype=pool_size_per_celltype,
        dimred_loadings_per_celltype=pca_loadings_per_celltype,
        results_dir=results_dir,
        mean_expr_per_ct=mean_expr_per_ct,
        xenium_min_expr=xenium_min_expr,
        xenium_max_expr=xenium_max_expr,
    )


def _select_genes_from_dimred(
    adata: AnnData,
    probeset_size: int,
    reduction_type: str,
    n_components: int,
    pool_size_per_celltype: int,
    dimred_loadings_per_celltype: Optional[dict[str, pd.DataFrame]],
    results_dir: Optional[str],
    mean_expr_per_ct: Optional[dict] = None,
    xenium_min_expr: float = DEFAULT_MIN_XENIUM_EXPRESSION,
    xenium_max_expr: float = DEFAULT_MAX_XENIUM_EXPRESSION,
) -> GeneListBuilder:
    """Core per-celltype dimension reduction gene selection with factor-aware resolution.

    Args:
        adata: Annotated data matrix
        probeset_size: Target number of genes
        reduction_type: 'nmf' or 'pca'
        n_components: Number of components to use
        pool_size_per_celltype: Phase-1 pool size per cell type.
        dimred_loadings_per_celltype: Per-celltype loadings {celltype: DataFrame}
        results_dir: Results directory
        mean_expr_per_ct: Per-celltype mean expression dict for the Phase-1 pool gate
            (skipped when None)
        xenium_min_expr / xenium_max_expr: Bounds for the Phase-1 pool expression gate;
            default to the DEFAULT_*_XENIUM_EXPRESSION constants

    Returns:
        GeneListBuilder with selected genes and factor assignments
    """
    logger.info(f"=== {reduction_type.upper()} Gene Selection (per-celltype) ===")
    logger.info(f"Target: {probeset_size} genes, Components: {n_components}")

    # Initialize GeneListBuilder
    strategy_name = f"dimred_only_{reduction_type}_per_celltype"
    builder = GeneListBuilder(
        strategy_name=strategy_name,
        analysis_type="per_celltype",
    )

    if dimred_loadings_per_celltype is None:
        raise ValueError(f"Per-celltype {reduction_type} loadings required but not provided")
    if not dimred_loadings_per_celltype:
        raise ValueError(
            f"Per-celltype {reduction_type} loadings dict is empty — NMF failed for all "
            f"cell types. Check for TypeError or insufficient cells in the log above."
        )

    _select_genes_per_celltype(
        builder=builder,
        celltype_loadings=dimred_loadings_per_celltype,
        n_components=n_components,
        pool_size_per_celltype=pool_size_per_celltype,
        probeset_size=probeset_size,
        results_dir=results_dir,
        mean_expr_per_ct=mean_expr_per_ct,
        xenium_min_expr=xenium_min_expr,
        xenium_max_expr=xenium_max_expr,
    )

    # Save results if directory provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        builder.to_csv(os.path.join(results_dir, panel_information_filename("dimred_only", reduction_type)))

    logger.info(f"{reduction_type.upper()} selection complete: {len(builder.get_selected_genes('initial'))} genes")
    return builder


# Display-only value mapping for the dimred `selection_strategy` column -- the
# internal strategy_name identifier (used elsewhere, e.g. cache validation) is left
# untouched; this only affects what {nmf,pca}_panel_information.csv shows.
_DIMRED_STRATEGY_DISPLAY = {
    "dimred_only_nmf_per_celltype": "nmf",
    "dimred_only_pca_per_celltype": "pca",
}


def format_dimred_ranked_df(full_df: pd.DataFrame) -> pd.DataFrame:
    """Trim/reorder a dimred (`dimred_only`, NMF or PCA)
    `{nmf,pca}_panel_information.csv` to its non-redundant column set
    (see `panel_information_filename()`).

    Expects `full_df` to already carry `in_panel` (derived from `final_selection`) and
    `informative_celltypes` (derived from `contributing_celltypes`) -- both computed
    once, generically, by run_single_selection.py for every strategy.

    Drops: rank (always blank -- dimred never assigns one), selected_initial /
    final_selection (identical to in_panel for this component), passed_xenium /
    xenium_failure_reason (structurally always blank -- Xenium filtering for dimred
    happens inside Phase-1 pool construction, before genes reach the builder),
    contributing_celltypes (redundant with informative_celltypes -- same celltype set,
    comma- vs. pipe-joined). Renames selection_score -> loading (the meaningful NMF/PCA
    term for the same value) and maps the internal selection_strategy identifier to a
    short display name ("nmf"/"pca").

    See docs/doc-pipeline/audit_3.md, "Selection-module CSV output reorganization".
    """
    df = full_df.rename(columns={"selection_score": "loading"})
    df = df.assign(
        selection_strategy=df["selection_strategy"].map(
            lambda s: _DIMRED_STRATEGY_DISPLAY.get(s, s)
        )
    )
    cols = [
        "gene", "in_panel", "selection_strategy", "analysis_type", "celltype",
        "informative_celltypes", "n_celltypes_selected", "component", "loading",
        "mean_expression",
    ]
    df = df[cols]
    df = df.sort_values(
        ["in_panel", "loading", "n_celltypes_selected"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return df


def _select_genes_per_celltype(
    builder: GeneListBuilder,
    celltype_loadings: dict[str, pd.DataFrame],
    n_components: int,
    pool_size_per_celltype: int,
    probeset_size: int,
    results_dir: Optional[str] = None,
    mean_expr_per_ct: Optional[dict] = None,
    xenium_min_expr: float = DEFAULT_MIN_XENIUM_EXPRESSION,
    xenium_max_expr: float = DEFAULT_MAX_XENIUM_EXPRESSION,
) -> None:
    """Select genes from per-celltype dimension reduction with 3-phase pool-based architecture.

    Architecture:
    - Phase 1: Create large oversampled pool per celltype-factor
    - Phase 2: Resolve within-celltype duplicates: assign each gene to the factor with
      highest abs(loading); pull next-best candidates for factors that lose a gene.
      Cross-celltype shared genes are tracked (not removed).
    - Phase 3:
        Step 1 — Strict per-CT-factor selection: top genes/combo by abs(loading).
                 Cross-celltype shared genes count once, so unique count may be < target.
        Step 2 — Fill gap (if < probeset_size): remaining pool sorted by
                 n_celltypes descending only (abs(loading) is NOT a tiebreak —
                 per-celltype loadings come from independent fits and aren't
                 comparable), preferring multi-CT genes.
        Step 3 — Trim (if > probeset_size from rounding): remove lowest-loading genes.

    Args:
        builder: GeneListBuilder to populate
        celltype_loadings: Dict mapping celltype -> loadings DataFrame
        n_components: Number of components per celltype
        pool_size_per_celltype: Genes per celltype in Phase 1 pool (e.g., 200)
        probeset_size: Target final size (e.g., 100)
        results_dir: Results directory for pool caching
        xenium_min_expr / xenium_max_expr: Expression bounds for the Phase-1 pool gate
            (default to DEFAULT_*_XENIUM_EXPRESSION).
    """
    logger.info(f"=" * 80)
    logger.info(f"POOL-BASED GENE SELECTION: Per-Celltype")
    logger.info(f"=" * 80)
    logger.info(f"Cell types: {len(celltype_loadings)}")
    logger.info(f"Components per celltype: {n_components}")
    logger.info(f"Pool size per celltype (Phase 1): {pool_size_per_celltype} genes")
    logger.info(f"Final probeset size (Phase 3): {probeset_size} genes")

    # ============================================================================
    # PHASE 1: POOL CREATION (Large pool without size constraints)
    # ============================================================================
    logger.info("")
    logger.info("─" * 80)
    logger.info("PHASE 1: Creating large gene pool")
    logger.info("─" * 80)

    # Calculate genes per celltype-factor combo for pool
    # pool_size_per_celltype = TOTAL genes per celltype distributed across factors
    # Example: 200 genes/celltype ÷ 5 factors = 40 genes per celltype-factor combo
    pool_genes_per_comp = max(1, pool_size_per_celltype // n_components)
    logger.info(f"Pool allocation: {pool_genes_per_comp} genes per celltype-factor combination")

    # Collect genes per celltype and factor (POOL, not final selection)
    celltype_genes_per_factor = {}
    pool_stats = {"total_selections": 0, "celltypes": {}}

    for celltype, loadings_df in celltype_loadings.items():
        component_cols = loadings_df.columns[:n_components]
        genes_per_factor = {}

        for comp_name in component_cols:
            comp_loadings = loadings_df[comp_name].abs()

            top_genes = comp_loadings.nlargest(pool_genes_per_comp)
            selected_genes = top_genes.index.tolist()

            genes_per_factor[comp_name] = selected_genes

        # Store pool data
        celltype_genes_per_factor[celltype] = {
            "genes_per_factor": genes_per_factor,
            "loadings_df": loadings_df,
            "component_cols": component_cols.tolist(),
        }

        ct_total = sum(len(g) for g in genes_per_factor.values())
        pool_stats["total_selections"] += ct_total
        pool_stats["celltypes"][celltype] = ct_total

        logger.info(f"  {celltype}: {ct_total} genes in pool")

    logger.info(f"✓ Phase 1 complete: {pool_stats['total_selections']} genes in raw pool (with duplicates)")

    # ──────────────────────────────────────────────────────────────────────────
    # XENIUM FILTER: Remove genes outside expression range from Phase 1 pool
    # ──────────────────────────────────────────────────────────────────────────
    if mean_expr_per_ct is not None:
        n_before = pool_stats['total_selections']
        n_removed = 0
        for celltype in list(celltype_genes_per_factor.keys()):
            if celltype not in mean_expr_per_ct:
                logger.warning(
                    f"No mean expression data for '{celltype}', "
                    f"skipping Xenium filter for this celltype"
                )
                continue
            ct_mean_expr = mean_expr_per_ct[celltype]
            for factor in list(celltype_genes_per_factor[celltype]['genes_per_factor'].keys()):
                before = len(celltype_genes_per_factor[celltype]['genes_per_factor'][factor])
                celltype_genes_per_factor[celltype]['genes_per_factor'][factor] = [
                    g for g in celltype_genes_per_factor[celltype]['genes_per_factor'][factor]
                    if xenium_min_expr
                    <= ct_mean_expr.get(g, 0.0)
                    <= xenium_max_expr
                ]
                after = len(celltype_genes_per_factor[celltype]['genes_per_factor'][factor])
                n_removed += before - after

        # Recalculate pool_stats totals after filtering
        pool_stats['total_selections'] = sum(
            len(g)
            for ct_data in celltype_genes_per_factor.values()
            for g in ct_data['genes_per_factor'].values()
        )
        logger.info(
            f"✓ Xenium filter: {n_before} → {pool_stats['total_selections']} pool genes "
            f"({n_removed} removed, expression range "
            f"[{xenium_min_expr}, {xenium_max_expr}])"
        )
    else:
        logger.info("Xenium filter skipped (no mean expression data provided)")

    # Cache pool to disk if results_dir provided
    if results_dir:
        import pickle
        from pathlib import Path
        
        pool_cache_dir = Path(results_dir) / "nmf_pools"
        pool_cache_dir.mkdir(parents=True, exist_ok=True)
        
        pool_cache_file = pool_cache_dir / "per_celltype_pool.pkl"
        with open(pool_cache_file, "wb") as f:
            pickle.dump(celltype_genes_per_factor, f)
        logger.info(f"✓ Pool cached to: {pool_cache_file}")

    # ============================================================================
    # PHASE 2: DUPLICATE RESOLUTION (within-celltype dedup + cross-CT tracking)
    # ============================================================================
    logger.info("")
    logger.info("─" * 80)
    logger.info("PHASE 2: Resolving within-celltype duplicates; tracking cross-CT sharing")
    logger.info("─" * 80)

    # Resolve within-celltype duplicates; track which celltypes selected each gene
    resolved = resolve_duplicates_factor_aware_per_celltype(
        celltype_genes_per_factor=celltype_genes_per_factor,
    )

    pool_genes = resolved["selected_genes"]  # Unique-gene pool (cross-CT sharing preserved)
    pool_factor_assignments = resolved["factor_assignments"]
    pool_celltype_assignments = resolved["celltype_assignments"]
    gene_celltype_mapping = resolved["gene_celltype_mapping"]
    n_celltypes_per_gene = resolved["n_celltypes_per_gene"]

    logger.info(
        f"✓ Phase 2 complete: {len(pool_genes)} unique genes "
        f"({resolved['duplicates_resolved']} within-celltype duplicates resolved, "
        f"{sum(1 for n in n_celltypes_per_gene.values() if n > 1)} genes shared across celltypes)"
    )

    # Cache pool to disk
    if results_dir:
        import pandas as pd
        from pathlib import Path

        pool_cache_dir = Path(results_dir) / "nmf_pools"

        pool_df = pd.DataFrame({
            "gene": pool_genes,
            "celltype": [pool_celltype_assignments.get(g) for g in pool_genes],
            "factor": [pool_factor_assignments.get(g) for g in pool_genes],
            "n_celltypes_selected": [n_celltypes_per_gene.get(g, 1) for g in pool_genes],
            "contributing_celltypes": [
                ', '.join(sorted(gene_celltype_mapping.get(g, [pool_celltype_assignments.get(g, '')])))
                for g in pool_genes
            ],
        })
        resolved_pool_file = pool_cache_dir / "resolved_pool.csv"
        pool_df.to_csv(resolved_pool_file, index=False)
        logger.info(f"✓ Resolved pool saved to: {resolved_pool_file}")

        import json
        stats_file = pool_cache_dir / "resolution_stats.json"
        with open(stats_file, "w") as f:
            json.dump({
                "pool_size_per_celltype": pool_size_per_celltype,
                "raw_pool_size": pool_stats["total_selections"],
                "resolved_pool_size": len(pool_genes),
                "duplicates_resolved": resolved["duplicates_resolved"],
                "celltypes": list(celltype_loadings.keys()),
                "n_components": n_components,
            }, f, indent=2)
        logger.info(f"✓ Resolution stats saved to: {stats_file}")

    # ============================================================================
    # PHASE 3: FINAL SELECTION
    # Step 1: Strict per-CT-factor selection (top N per combo by abs(loading))
    # Step 2: If below target (cross-CT sharing reduces unique count):
    #         fill from remaining pool sorted by n_celltypes desc ONLY
    #         (abs(loading) not comparable across independent per-CT fits)
    # Step 3: If above target (rounding up created over-count): trim lowest loading
    # ============================================================================
    logger.info("")
    logger.info("─" * 80)
    logger.info("PHASE 3: Strict CT-factor selection; fill by n_celltypes (desc)")
    logger.info("─" * 80)

    abs_loadings_per_celltype = {
        celltype: loadings_df.abs()
        for celltype, loadings_df in celltype_loadings.items()
    }

    # ---- Step 1: Strict per-CT-factor selection (no R²) ----------------------
    genes_per_celltype_final = probeset_size / len(celltype_loadings)
    genes_per_combo_final = genes_per_celltype_final / n_components
    genes_per_combo_final_rounded = math.ceil(genes_per_combo_final)

    logger.info(
        f"Strict allocation: {genes_per_combo_final_rounded} genes/combo "
        f"(target {genes_per_celltype_final:.2f}/CT, {genes_per_combo_final:.2f}/combo)"
    )

    final_selected_genes: list[str] = []
    final_factor_assignments: dict[str, str] = {}
    final_celltype_assignments: dict[str, str] = {}

    for celltype in celltype_loadings.keys():
        abs_loadings_df = abs_loadings_per_celltype[celltype]
        component_cols = celltype_loadings[celltype].columns[:n_components]

        for factor in component_cols:
            combo_pool_genes = [
                g for g in pool_genes
                if pool_celltype_assignments.get(g) == celltype
                and pool_factor_assignments.get(g) == factor
            ]

            if not combo_pool_genes:
                logger.warning(f"No pool genes for {celltype}/{factor} after Phase 2")
                continue

            sorted_combo = sorted(
                combo_pool_genes,
                key=lambda g: float(abs_loadings_df.at[g, factor])
                if g in abs_loadings_df.index else 0.0,
                reverse=True,
            )
            for gene in sorted_combo[:genes_per_combo_final_rounded]:
                if gene not in final_selected_genes:
                    final_selected_genes.append(gene)
                final_factor_assignments[gene] = factor
                final_celltype_assignments[gene] = celltype

    logger.info(
        f"Strict selection: {len(final_selected_genes)} unique genes "
        f"(target: {probeset_size})"
    )

    # ---- Step 2: Fill gap if below target ------------------------------------
    selected_set = set(final_selected_genes)
    if len(final_selected_genes) < probeset_size:
        gap = probeset_size - len(final_selected_genes)
        logger.info(
            f"Below target by {gap} genes (cross-celltype sharing reduces unique count). "
            f"Filling from remaining pool sorted by n_celltypes desc "
            f"(loadings not used — not comparable across independent per-CT NMF runs)."
        )
        fill_candidates = []
        for gene in pool_genes:
            if gene in selected_set:
                continue
            best_ct = pool_celltype_assignments[gene]
            best_factor = pool_factor_assignments[gene]
            fill_candidates.append((n_celltypes_per_gene.get(gene, 1), gene, best_ct, best_factor))

        # Sort by n_celltypes only — abs(loading) is NOT used here because loadings
        # come from independent per-CT NMF runs and are not comparable across celltypes.
        fill_candidates.sort(key=lambda x: x[0], reverse=True)

        for n_cts, gene, best_ct, best_factor in fill_candidates:
            if len(final_selected_genes) >= probeset_size:
                break
            final_selected_genes.append(gene)
            final_factor_assignments[gene] = best_factor
            final_celltype_assignments[gene] = best_ct
            selected_set.add(gene)

        logger.info(f"After fill: {len(final_selected_genes)} genes")

    # ---- Step 3: Trim if above target ----------------------------------------
    elif len(final_selected_genes) > probeset_size:
        n_to_remove = len(final_selected_genes) - probeset_size
        logger.info(f"Above target by {n_to_remove} genes. Trimming lowest-loading genes.")

        gene_scores = []
        for gene in final_selected_genes:
            ct = final_celltype_assignments.get(gene)
            fac = final_factor_assignments.get(gene)
            abs_df = abs_loadings_per_celltype.get(ct)
            score = float(abs_df.at[gene, fac]) if abs_df is not None and gene in abs_df.index else 0.0
            gene_scores.append((score, gene))

        gene_scores.sort(key=lambda x: x[0])  # ascending: lowest score first
        to_remove = {g for _, g in gene_scores[:n_to_remove]}
        final_selected_genes = [g for g in final_selected_genes if g not in to_remove]
        for gene in to_remove:
            del final_factor_assignments[gene]
            del final_celltype_assignments[gene]

        logger.info(f"✓ Trimmed to {len(final_selected_genes)} genes")

    logger.info(f"✓ Phase 3 complete: {len(final_selected_genes)} genes selected")

    # Build gene_details for ALL pool genes in (n_celltypes, loading) order.
    # Insertion order into the builder determines gap-fill priority.
    gene_details = []
    for gene in pool_genes:
        best_ct = pool_celltype_assignments[gene]
        best_factor = pool_factor_assignments[gene]
        abs_df = abs_loadings_per_celltype.get(best_ct)
        best_loading = (
            float(abs_df.at[gene, best_factor])
            if abs_df is not None and gene in abs_df.index else 0.0
        )
        n_cts = n_celltypes_per_gene.get(gene, 1)
        contributing = gene_celltype_mapping.get(gene, [best_ct])

        gene_details.append({
            'gene': gene,
            'n_celltypes': n_cts,
            'contributing_celltypes': ', '.join(sorted(contributing)),
            'best_celltype': best_ct,
            'best_factor': best_factor,
            'best_loading': best_loading,
        })

    # Sort by n_celltypes only — loadings not used for cross-CT ordering.
    gene_details.sort(key=lambda x: x['n_celltypes'], reverse=True)

    # ============================================================================
    # ADD GENES TO BUILDER
    # Add ALL pool genes in n_celltypes-descending order.
    # Only Phase 3 final genes are marked as selected_initial.
    # Non-selected pool genes become replacement candidates; since get_all_genes()
    # preserves insertion order for tied ranks, gap-filling picks multi-CT genes first.
    # ============================================================================
    logger.info("")
    logger.info(
        f"Adding Phase 2 pool to builder: {len(pool_genes)} genes "
        f"({len(final_selected_genes)} selected + "
        f"{len(pool_genes) - len(final_selected_genes)} replacement candidates)"
    )

    selected_set = set(final_selected_genes)

    # gene_details is already sorted by n_celltypes desc (only) from Phase 3
    for d in gene_details:
        gene = d['gene']
        celltype = d['best_celltype']
        factor = d['best_factor']
        loading = d['best_loading']
        n_cts = d['n_celltypes']
        contributing = d['contributing_celltypes']

        builder.add_gene(
            gene=gene,
            # abs(loading); no R² weighting. This IS the gene's loading value -- no
            # separate "loading" metadata key is kept (it would just duplicate this).
            # format_dimred_ranked_df() renames this column to "loading" for display,
            # since that's the meaningful term in NMF/PCA factor-loading terms.
            selection_score=loading,
            rank=None,
            celltype=celltype,
            component=str(factor),
            metadata={
                "n_celltypes_selected": n_cts,
                "contributing_celltypes": contributing,
            },
        )

    # Mark only Phase 3 final genes as initially selected
    for gene in final_selected_genes:
        builder.mark_selected(gene)

    logger.info(
        f"✓ Added {len(gene_details)} pool genes; "
        f"{len(final_selected_genes)} marked selected, "
        f"{len(gene_details) - len(selected_set)} available as replacement candidates"
    )
    logger.info(f"=" * 80)
