"""Random Forest gene selection.

This module provides classifier-based gene selection using Random Forest to rank
genes by feature importance. Uses cross-validation with F1-weighted scoring.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Import the canonical raw-count check from Utility-module. Hard import (like every
# other intra-cut import here): there is exactly one _validation.py in the cut and it
# only needs numpy/scipy/anndata, so a failure here means a broken checkout and should
# surface loudly rather than fall back to a divergent guess.
SCRIPT_DIR = Path(__file__).parent.absolute()
UTILITY_DIR = SCRIPT_DIR.parent / "Utility-module"
sys.path.insert(0, str(UTILITY_DIR))
from _validation import is_anndata_raw

# Use absolute imports (for script execution)
# --- load THIS directory's _constants.py by path (sibling dirs share the name) ---
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _cpath
_cspec = _ilu.spec_from_file_location("_constants", _cpath(__file__).resolve().parent / "_constants.py")
_sys.modules["_constants"] = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_sys.modules["_constants"])


from _constants import (
    COL_CELLTYPE,
    COL_RF_CELLTYPE,
    COL_RF_CELLTYPE_SCORES,
    COL_RF_CONTRIBUTING_CELLTYPES,
    DEFAULT_DEG_MAX_PVAL,
    DEFAULT_MIN_CELLS_PER_CELLTYPE,
    DEFAULT_PROBESET_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RF_MAX_DEPTH,
    DEFAULT_RF_N_ESTIMATORS,
    DEFAULT_RF_N_FOLDS,
    DEFAULT_RF_RANDOM_SEEDS,
    RF_CONTRIBUTING_CELLTYPE_MIN_SHARE,
    RF_CONTRIBUTING_CELLTYPE_SEP,
)
from _deg_selection import filter_celltypes_by_min_cells
from _gene_list_builder import GeneListBuilder, panel_information_filename

logger = logging.getLogger(__name__)


class RFCacheTooShortError(ValueError):
    """Raised when an explicit RF cache is valid but too short for the target."""

    def __init__(self, message: str, n_ranked: int, cache_file: str):
        super().__init__(message)
        self.n_ranked = n_ranked
        self.cache_file = cache_file


# =============================================================================
# Per-cell-type attribution of RF feature importance
# =============================================================================


def _forest_per_class_gini_importance(
    model: RandomForestClassifier, n_classes: int
) -> np.ndarray:
    """Decompose a fitted forest's Gini feature importance by target class.

    Gini impurity ``G = 1 - Σ_c p_c² = Σ_c p_c(1 - p_c)`` is already a sum of one
    term per class, so each split's total impurity decrease ``ΔG`` decomposes
    exactly as ``Σ_c Δg_c`` with ``g_c = p_c(1 - p_c)``. Summing ``Δg_c`` over
    every split on a feature and over every tree (each tree normalised by its own
    total impurity decrease, matching scikit-learn's MDI normalisation) yields a
    per-(feature, class) matrix whose row sums equal the tree-normalised
    ``feature_importances_``. No retraining — this only reads ``tree_`` internals.

    Args:
        model: A fitted ``RandomForestClassifier``.
        n_classes: Number of classes in the global label encoding. Columns of the
            returned matrix are indexed by that encoding; a class absent from an
            individual tree's bootstrap contributes zero from that tree.

    Returns:
        ``np.ndarray`` of shape ``(model.n_features_in_, n_classes)``, averaged
        over the forest's trees. The whole matrix sums to ~1.0 (0.0 if no tree
        ever split).
    """
    n_features = int(model.n_features_in_)
    accum = np.zeros((n_features, n_classes), dtype=np.float64)
    estimators = list(model.estimators_)

    for est in estimators:
        tree = est.tree_
        # Per-node class proportions. ``tree_.value`` holds proportions in
        # sklearn >= 1.4 and weighted counts in older versions; normalising each
        # node's vector to sum 1 handles both.
        val = tree.value.reshape(tree.node_count, -1).astype(np.float64)
        row_sums = val.sum(axis=1, keepdims=True)
        np.divide(val, row_sums, out=val, where=row_sums > 0)
        g = val * (1.0 - val)  # (n_nodes, n_tree_classes)

        w = tree.weighted_n_node_samples
        left, right = tree.children_left, tree.children_right
        feat, imp = tree.feature, tree.impurity
        tree_cls = np.asarray(est.classes_, dtype=int)  # global column indices

        tree_mat = np.zeros((n_features, n_classes), dtype=np.float64)
        total_decrease = 0.0
        for node in np.where(feat >= 0)[0]:  # internal nodes only
            lft, rgt = left[node], right[node]
            dg = w[node] * g[node] - w[lft] * g[lft] - w[rgt] * g[rgt]
            tree_mat[feat[node], tree_cls] += dg
            total_decrease += (
                w[node] * imp[node] - w[lft] * imp[lft] - w[rgt] * imp[rgt]
            )

        if total_decrease > 0:
            accum += tree_mat / total_decrease

    if estimators:
        accum /= len(estimators)
    return accum


def _derive_rf_celltype_fields(
    shares: np.ndarray,
    class_names: list[str],
    min_share: float = RF_CONTRIBUTING_CELLTYPE_MIN_SHARE,
    sep: str = RF_CONTRIBUTING_CELLTYPE_SEP,
) -> tuple[list[str], list[str], list[str]]:
    """Turn a per-gene class-share matrix into the three ``rf_*`` column values.

    Args:
        shares: ``(n_genes, n_classes)`` array; each row sums to 1.0, or to 0.0
            for a gene the forest never split on.
        class_names: Class (cell-type) names aligned to the columns of ``shares``.
        min_share: A class is "contributing" when its share of the gene's own
            importance is at least this. If none clear the bar the argmax class is
            used, so the field is never empty for an attributed gene.
        sep: Separator for the joined ``rf_contributing_celltypes`` string.

    Returns:
        ``(rf_celltype, rf_contributing_celltypes, rf_celltype_scores)`` lists,
        each of length ``n_genes``. An unattributed gene yields ``('', '', '{}')``.
    """
    rf_celltype: list[str] = []
    rf_contributing: list[str] = []
    rf_scores: list[str] = []

    for row in np.asarray(shares, dtype=np.float64):
        if float(row.sum()) <= 0.0:
            rf_celltype.append("")
            rf_contributing.append("")
            rf_scores.append("{}")
            continue

        argmax_idx = int(np.argmax(row))
        rf_celltype.append(class_names[argmax_idx])

        contrib_idx = [i for i, v in enumerate(row) if v >= min_share] or [argmax_idx]
        contrib_idx.sort(key=lambda i: float(row[i]), reverse=True)
        rf_contributing.append(sep.join(class_names[i] for i in contrib_idx))

        rf_scores.append(
            json.dumps(
                {
                    class_names[i]: round(float(v), 4)
                    for i, v in enumerate(row)
                    if v > 0
                },
                sort_keys=True,
            )
        )

    return rf_celltype, rf_contributing, rf_scores


def _rf_celltype_columns_from_summary(gene_summary: pd.DataFrame) -> dict[str, list]:
    """Extract the three ``rf_*`` column value lists from a ``gene_summary``.

    Fills defaults for summaries produced before per-class attribution existed
    (e.g. an old ``rf_gene_scores.pkl``), so ``add_genes`` always receives the
    columns and old caches still load.
    """
    defaults = {
        COL_RF_CELLTYPE: "",
        COL_RF_CONTRIBUTING_CELLTYPES: "",
        COL_RF_CELLTYPE_SCORES: "{}",
    }
    if not any(col in gene_summary.columns for col in defaults):
        logger.warning(
            "RF gene_summary has no per-cell-type attribution columns "
            "(pre-upgrade cache); rf_celltype/rf_contributing_celltypes will be empty"
        )
    out: dict[str, list] = {}
    for col, default in defaults.items():
        if col in gene_summary.columns:
            out[col] = gene_summary[col].fillna(default).tolist()
        else:
            out[col] = [default] * len(gene_summary)
    return out


def _attach_rf_celltype_fields(
    gene_summary: pd.DataFrame,
    class_accum: np.ndarray,
    candidate_genes: list[str],
    class_names: list[str],
) -> pd.DataFrame:
    """Normalise the F1-weighted per-class accumulation into per-gene shares and
    write ``rf_celltype`` / ``rf_contributing_celltypes`` / ``rf_celltype_scores``
    onto ``gene_summary`` (matched by gene name — ``class_accum`` rows are in
    ``candidate_genes`` order, ``gene_summary`` is sorted by score).

    Returns a genes × cell-types ``DataFrame`` of shares for the pkl cache.
    """
    row_tot = class_accum.sum(axis=1, keepdims=True)
    shares = np.divide(
        class_accum, row_tot, out=np.zeros_like(class_accum), where=row_tot > 0
    )
    rf_ct, rf_contrib, rf_scores = _derive_rf_celltype_fields(shares, class_names)
    gene_summary[COL_RF_CELLTYPE] = gene_summary["gene"].map(
        dict(zip(candidate_genes, rf_ct))
    )
    gene_summary[COL_RF_CONTRIBUTING_CELLTYPES] = gene_summary["gene"].map(
        dict(zip(candidate_genes, rf_contrib))
    )
    gene_summary[COL_RF_CELLTYPE_SCORES] = gene_summary["gene"].map(
        dict(zip(candidate_genes, rf_scores))
    )
    return pd.DataFrame(shares, index=candidate_genes, columns=class_names)


def _derive_rf_random_seeds(
    random_state: int, n_seeds: int = len(DEFAULT_RF_RANDOM_SEEDS)
) -> list[int]:
    """Derive the RF cross-validation seed list from a single ``random_state``.

    Special-cased for ``random_state == DEFAULT_RANDOM_STATE`` (42) to exactly
    reproduce the historical hardcoded ``DEFAULT_RF_RANDOM_SEEDS`` list, so the
    pipeline's default behavior is unchanged. Any other ``random_state`` draws
    ``n_seeds`` pseudo-random integers from a seeded RNG rather than a simple
    ``[random_state + i for i in range(n_seeds)]`` offset, so that two runs
    with adjacent (but different) ``random_state`` values don't end up
    training RF on 4-of-5 identical CV seeds.
    """
    if random_state == DEFAULT_RANDOM_STATE:
        return list(DEFAULT_RF_RANDOM_SEEDS)
    rng = np.random.default_rng(random_state)
    return rng.integers(0, 100_000, size=n_seeds).tolist()


def select_genes_with_rf(
    adata: AnnData,
    probeset_size: int = DEFAULT_PROBESET_SIZE,
    celltype_column: str = COL_CELLTYPE,
    max_depth: int = DEFAULT_RF_MAX_DEPTH,
    n_estimators: int = DEFAULT_RF_N_ESTIMATORS,
    n_folds: int = DEFAULT_RF_N_FOLDS,
    random_seeds: Optional[list[int]] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    min_cells_per_celltype: int = DEFAULT_MIN_CELLS_PER_CELLTYPE,
    use_deg_prefilter: bool = False,
    n_deg_per_group: Optional[int] = None,
    deg_results: Optional[GeneListBuilder] = None,
    rf_score_cache_file: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> GeneListBuilder:
    """Select genes using Random Forest classifier with cross-validation.

    Trains a Random Forest classifier to predict cell types from gene expression
    and selects genes with highest feature importance across folds. Optionally
    pre-filters to DEG genes before classification.

    Args:
        adata: Processed AnnData object (log-normalized)
        probeset_size: Target number of genes to select
        celltype_column: Column in adata.obs for classification labels
        max_depth: Maximum depth of each tree in forest
        n_estimators: Number of trees in Random Forest
        n_folds: Number of cross-validation folds
        random_seeds: List of random seeds for multiple CV runs. When ``None``
            (the default), derived from ``random_state`` via
            ``_derive_rf_random_seeds()`` — pass this explicitly to override
            that derivation entirely.
        random_state: Base seed used to derive ``random_seeds`` when the latter
            is not given explicitly. Also unifies RF's CV seeding with the
            NMF/PCA fit's own ``random_state`` for a combination run, so a
            single ``--random_state`` reproduces both. At the default value
            (42) this reproduces the historical hardcoded
            ``DEFAULT_RF_RANDOM_SEEDS = [42, 43, 44, 45, 46]`` exactly; any
            other value draws ``n_seeds`` pseudo-random seeds from a
            ``random_state``-seeded RNG instead of a simple offset, so two
            runs with adjacent ``random_state`` values don't end up training
            RF on mostly-identical CV seeds.
        min_cells_per_celltype: Minimum cells per cell type for inclusion
        use_deg_prefilter: If True, pre-filter to DEG genes before classification
        n_deg_per_group: Number of DEG genes per group to use (if use_deg_prefilter=True)
        deg_results: Pre-computed DEG results (optional, for caching)
        rf_score_cache_file: Explicit path to a cached ``rf_gene_scores.pkl``.
            When provided, the cached ranking is used instead of training RF.
        results_dir: Directory to save results (optional)

    Returns:
        GeneListBuilder with:
            - selected_genes: Top genes by F1-weighted importance
            - selection_score: F1-weighted feature importance
            - rank: Rank by weighted importance
            - metadata: Average importance, max importance, average F1

    Raises:
        ValueError: If not enough valid cell types after filtering
        RFCacheTooShortError: If ``rf_score_cache_file`` is valid but ranks fewer
            genes than ``probeset_size`` (rf_simple only; rf_deg falls back to
            bounded recomputation).

    Examples:
        >>> # Simple RF selection
        >>> builder = select_genes_with_rf(adata, probeset_size=100)
        
        >>> # RF with DEG pre-filtering
        >>> builder = select_genes_with_rf(
        ...     adata,
        ...     probeset_size=100,
        ...     use_deg_prefilter=True,
        ...     n_deg_per_group=200
        ... )
        
        >>> # RF with cached DEG results
        >>> deg_builder = select_degs(adata, ...)
        >>> rf_builder = select_genes_with_rf(
        ...     adata,
        ...     use_deg_prefilter=True,
        ...     deg_results=deg_builder
        ... )
    """
    logger.info("=== Random Forest Gene Selection ===")
    logger.info(f"Target probeset size: {probeset_size}")
    logger.info(f"Use DEG pre-filter: {use_deg_prefilter}")

    # Initialize GeneListBuilder
    strategy_name = "rf_deg" if use_deg_prefilter else "rf_simple"
    builder = GeneListBuilder(
        strategy_name=strategy_name,
        analysis_type="global",
    )

    # Default random seeds if not provided -- derived from random_state so a single
    # --random_state controls both this CV and the NMF/PCA fit (see _derive_rf_random_seeds).
    if random_seeds is None:
        random_seeds = _derive_rf_random_seeds(random_state)

    # Validate celltype column
    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_column}' not found in adata.obs")

    # Filter cell types by minimum cell count
    if min_cells_per_celltype > 0:
        valid_celltypes, excluded_celltypes, celltype_counts = filter_celltypes_by_min_cells(
            adata, celltype_column, min_cells_per_celltype
        )

        if len(excluded_celltypes) > 0:
            logger.info(f"Filtering to {len(valid_celltypes)} valid cell types")
            adata = adata[adata.obs[celltype_column].isin(valid_celltypes)].copy()
            adata.obs[celltype_column] = adata.obs[celltype_column].cat.remove_unused_categories()

    if adata.shape[0] == 0:
        raise ValueError("No cells remaining after filtering")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(adata.obs[celltype_column].values)
    class_names = [str(c) for c in label_encoder.classes_]
    n_classes = len(class_names)
    logger.info(f"Classes: {n_classes} ({', '.join(class_names[:5])}...)")

    rf_metadata = {
        "requested_panel_size": probeset_size,
        "rf_deg_recompute_attempts": 0,
        "rf_deg_candidate_pool_sizes": [],
        "rf_deg_cache_used": False,
        "rf_deg_cache_rejected_reason": None,
        "rf_short_panel_allowed": False,
        "short_panel_reason": None,
    }

    # Explicit cache reuse path. This is intentionally checked before DEG
    # prefiltering so benchmark baselines can reuse previous RF scores without
    # recomputing DEGs or retraining the forest. For rf_deg, a short cache falls
    # back to bounded recomputation; for rf_simple it remains a hard error.
    if rf_score_cache_file:
        try:
            cache_builder = _load_rf_scores_from_cache(
                cache_file=rf_score_cache_file,
                adata=adata,
                probeset_size=probeset_size,
                use_deg_prefilter=use_deg_prefilter,
                builder=builder,
                results_dir=results_dir,
            )
            _annotate_rf_panel_metadata(
                builder=cache_builder,
                probeset_size=probeset_size,
                selected_count=len(cache_builder.get_selected_genes("initial")),
                metadata={
                    **rf_metadata,
                    "rf_deg_cache_used": True,
                    "rf_deg_cache_rejected_reason": None,
                },
            )
            cache_builder.add_metadata("rf_deg_cache_used", True)
            cache_builder.add_metadata("rf_deg_cache_rejected_reason", None)
            return cache_builder
        except RFCacheTooShortError as e:
            if not use_deg_prefilter:
                raise
            rf_metadata["rf_deg_cache_rejected_reason"] = str(e)
            logger.warning(
                "RF-DEG cache is too short for target size; recomputing with "
                "larger DEG candidate pools: %s", e
            )

    if use_deg_prefilter and deg_results is None:
        return _select_genes_with_rf_deg_retries(
            adata=adata,
            probeset_size=probeset_size,
            celltype_column=celltype_column,
            max_depth=max_depth,
            n_estimators=n_estimators,
            n_folds=n_folds,
            random_seeds=random_seeds,
            n_deg_per_group=n_deg_per_group,
            results_dir=results_dir,
            metadata=rf_metadata,
        )

    # Optional DEG pre-filtering
    candidate_genes = adata.var_names.tolist()
    if use_deg_prefilter:
        logger.info("Applying DEG pre-filter...")
        
        # Try to use cached DEG results first
        if deg_results is not None:
            logger.info("Using cached DEG results")
            candidate_genes = deg_results.get_selected_genes()
        else:
            # Fallback: compute DEGs
            logger.info("No cached DEG results - computing new DEGs")
            candidate_genes = _get_deg_genes_for_classification(
                adata,
                celltype_column,
                n_deg_per_group,
                probeset_size,
            )
        
        logger.info(f"DEG pre-filter: {len(candidate_genes)} candidate genes")

    # Ensure candidate genes are present and aligned with the feature matrix.
    candidate_genes = [g for g in candidate_genes if g in adata.var_names]
    if not candidate_genes:
        raise ValueError("No candidate genes available after filtering to adata.var_names")

    # Use expression data for RF classification on the candidate gene subset.
    # This avoids training on unused columns and keeps feature_importances_ aligned.
    X_source = adata[:, candidate_genes].X
    if issparse(X_source):
        X = X_source.toarray()
    else:
        X = np.asarray(X_source)
    
    # Verify data is normalized (RF works better with log-normalized data)
    if is_anndata_raw(adata):
        logger.warning(
            "Data appears to contain raw counts (integer values). "
            "Random Forest typically works better with log-normalized data. "
            "Consider using log1p-transformed data in adata.X for better performance."
        )
    
    # Check for cached RF results
    if results_dir:
        cache_file = os.path.join(results_dir, 'rf_models', 'rf_gene_scores.pkl')
        if os.path.exists(cache_file):
            logger.info(f"Loading cached RF results from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                
                # Verify cache is compatible
                if (cached_results['n_genes'] == len(candidate_genes) and
                    cached_results['use_deg_prefilter'] == use_deg_prefilter):
                    
                    logger.info("✓ Using cached RF gene scores")
                    gene_summary = cached_results['gene_summary']

                    # Rebuild GeneListBuilder from cached results in bulk
                    builder.add_genes(
                        genes=gene_summary["gene"].tolist(),
                        ranks=list(range(1, len(gene_summary) + 1)),
                        scores=gene_summary["final_score"].tolist(),
                        **_rf_celltype_columns_from_summary(gene_summary),
                    )
                    
                    # Mark top N as selected
                    selected_genes = gene_summary.head(probeset_size)["gene"].tolist()
                    for gene in selected_genes:
                        builder.mark_selected(gene)
                    
                    _annotate_rf_panel_metadata(
                        builder=builder,
                        probeset_size=probeset_size,
                        selected_count=len(selected_genes),
                        metadata=rf_metadata,
                    )
                    logger.info(f"Selected {len(selected_genes)} genes from cache")
                    return builder
                else:
                    logger.warning("Cached results incompatible (different gene set). Recomputing...")
            except Exception as e:
                logger.warning(f"Failed to load cached RF results: {e}. Recomputing...")

    # Cross-validation with multiple seeds (not cached)
    all_gene_scores = []
    # F1-weighted accumulation of the per-(gene, class) Gini importance
    # decomposition across all fold-runs. Rows are in ``candidate_genes`` order.
    class_accum = np.zeros((len(candidate_genes), n_classes), dtype=np.float64)

    for seed_idx, seed in enumerate(random_seeds):
        logger.info(f"Running CV with seed {seed} ({seed_idx + 1}/{len(random_seeds)})")

        kfold = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y_encoded)):
            logger.info(f"  Fold {fold_idx + 1}/{n_folds}")

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # Train Random Forest. Class imbalance is handled once, via
            # `class_weight="balanced"` — do NOT also pass a balanced
            # `sample_weight`, which would apply the correction twice.
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=seed + fold_idx,  # Ensure different random state per fold,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="macro")
            logger.info(f"    F1 score: {f1:.4f}")

            # Record gene scores
            for gene, importance in zip(candidate_genes, model.feature_importances_):
                all_gene_scores.append(
                    {
                        "gene": gene,
                        "fold": f"seed{seed}_fold{fold_idx}",
                        "importance": importance,
                        "f1_score": f1,
                    }
                )

            # Per-class attribution of this fold's importances, F1-weighted so
            # "genes important in good models" dominate (same philosophy as
            # ``weighted_importance``). Does not affect the ranking.
            class_accum += f1 * _forest_per_class_gini_importance(model, n_classes)

    # Aggregate scores across folds
    gene_scoring_df = pd.DataFrame(all_gene_scores)

    # F1-weighted importance (genes important in good models ranked higher)
    # Each gene appears in multiple folds (n_folds × n_random_seeds)
    # Weight each appearance by model performance (F1), then average for consensus
    gene_scoring_df["weighted_importance"] = (
        gene_scoring_df["importance"] * gene_scoring_df["f1_score"]
    )

    # Aggregate across all fold appearances to get consensus gene ranking
    # - weighted_importance: Mean of (importance × f1), prioritizes genes important in good models
    gene_summary = gene_scoring_df.groupby("gene").agg(
        weighted_importance=("weighted_importance", "mean"),
    ).reset_index()

    gene_summary["final_score"] = gene_summary["weighted_importance"]
    gene_summary = gene_summary.sort_values("final_score", ascending=False)

    # Per-cell-type attribution (does not change rank / final_score / selection).
    gene_class_importance = _attach_rf_celltype_fields(
        gene_summary, class_accum, candidate_genes, class_names
    )

    # Add all genes to builder in bulk
    builder.add_genes(
        genes=gene_summary["gene"].tolist(),
        ranks=list(range(1, len(gene_summary) + 1)),
        scores=gene_summary["final_score"].tolist(),
        **_rf_celltype_columns_from_summary(gene_summary),
    )

    # Mark top N as selected
    selected_genes = gene_summary.head(probeset_size)["gene"].tolist()
    for gene in selected_genes:
        builder.mark_selected(gene)

    logger.info(f"Selected {len(selected_genes)} genes by Random Forest importance")

    # Save results if directory provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

        builder.to_csv(os.path.join(results_dir, panel_information_filename(strategy_name)))

        # Cache RF results for reuse by combination strategies
        model_dir = os.path.join(results_dir, 'rf_models')
        os.makedirs(model_dir, exist_ok=True)
        cache_file = os.path.join(model_dir, 'rf_gene_scores.pkl')
        try:
            cache_data = {
                'gene_summary': gene_summary,
                'all_gene_scores': gene_scoring_df,
                'gene_class_importance': gene_class_importance,
                'candidate_genes': candidate_genes,
                'n_genes': len(candidate_genes),
                'use_deg_prefilter': use_deg_prefilter,
                'n_folds': n_folds,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'random_seeds': random_seeds,
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"✓ Saved RF results to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save RF cache: {e}")

    _annotate_rf_panel_metadata(
        builder=builder,
        probeset_size=probeset_size,
        selected_count=len(selected_genes),
        metadata=rf_metadata,
    )

    return builder


def format_rf_ranked_df(full_df: pd.DataFrame) -> pd.DataFrame:
    """Trim/reorder an RF (`rf_deg`/`rf_simple`) `{strategy}_panel_information.csv` to
    its non-redundant column set (see `panel_information_filename()`).

    Expects `full_df` to already carry `in_panel` (derived from `final_selection`),
    computed once, generically, by run_single_selection.py for every strategy.

    Drops: selected_initial/final_selection (redundant with in_panel), analysis_type/
    selection_strategy (always "global"/the strategy name -- uninformative once you know
    which file this is), xenium_failure_reason/passed_xenium, selection_score (rank
    already encodes the ranking), celltype (always the literal "global" -- RF is one
    global multiclass model, carries zero signal), rf_contributing_celltypes/
    informative_celltypes (redundant with the single-label rf_celltype).

    See docs/doc-pipeline/audit_3.md, "Selection-module CSV output reorganization".
    """
    cols = ["gene", "in_panel", "rank", "rf_celltype", "rf_celltype_scores", "mean_expression"]
    df = full_df[cols]
    df = df.sort_values("rank", ascending=True).reset_index(drop=True)
    return df


def _select_genes_with_rf_deg_retries(
    adata: AnnData,
    probeset_size: int,
    celltype_column: str,
    max_depth: int,
    n_estimators: int,
    n_folds: int,
    random_seeds: list[int],
    n_deg_per_group: Optional[int],
    results_dir: Optional[str],
    metadata: dict,
    max_pval: float = DEFAULT_DEG_MAX_PVAL,
) -> GeneListBuilder:
    """Run RF-DEG with up to three larger DEG candidate pools.

    The first two attempts keep only genes with a Benjamini-Hochberg adjusted
    p-value <= ``max_pval`` per cell type; the final attempt drops that filter
    and draws on all genes so RF-DEG never hard-fails on a degenerate dataset.
    """
    groups = (
        adata.obs[celltype_column].cat.categories.tolist()
        if pd.api.types.is_categorical_dtype(adata.obs[celltype_column])
        else sorted(adata.obs[celltype_column].astype(str).unique().tolist())
    )
    n_groups = max(1, len(groups))
    default_per_group = max(50, probeset_size // n_groups * 3)
    base_per_group = n_deg_per_group or default_per_group
    attempt_sizes = [
        base_per_group,
        max(base_per_group * 2, probeset_size),
        len(adata.var_names),
    ]

    best_builder: Optional[GeneListBuilder] = None
    best_selected_count = -1

    for attempt_idx, attempt_size in enumerate(attempt_sizes, start=1):
        metadata["rf_deg_recompute_attempts"] = attempt_idx
        metadata["rf_deg_candidate_pool_sizes"].append(int(attempt_size))
        # Last attempt is the unfiltered "all genes" fallback (no significance gate).
        attempt_max_pval = max_pval if attempt_idx < len(attempt_sizes) else 1.0
        logger.info(
            "RF-DEG recompute attempt %d/%d with %s DEG genes per group (adj. p <= %s)",
            attempt_idx,
            len(attempt_sizes),
            attempt_size,
            attempt_max_pval,
        )

        candidate_genes = _get_deg_genes_for_classification(
            adata,
            celltype_column,
            attempt_size,
            probeset_size,
            max_pval=attempt_max_pval,
        )
        candidate_genes = [g for g in candidate_genes if g in adata.var_names]

        if not candidate_genes:
            logger.warning("RF-DEG attempt %d produced no DEG candidate genes", attempt_idx)
            continue

        attempt_results_dir = (
            os.path.join(results_dir, f"rf_deg_attempt_{attempt_idx}")
            if results_dir else None
        )
        attempt_builder = _run_rf_on_candidate_genes(
            adata=adata,
            candidate_genes=candidate_genes,
            y_labels=adata.obs[celltype_column].values,
            probeset_size=probeset_size,
            max_depth=max_depth,
            n_estimators=n_estimators,
            n_folds=n_folds,
            random_seeds=random_seeds,
            use_deg_prefilter=True,
            results_dir=attempt_results_dir,
        )
        selected_count = len(attempt_builder.get_selected_genes("initial"))
        logger.info(
            "RF-DEG attempt %d selected %d/%d genes from %d candidates",
            attempt_idx,
            selected_count,
            probeset_size,
            len(candidate_genes),
        )

        if selected_count > best_selected_count:
            best_builder = attempt_builder
            best_selected_count = selected_count

        if selected_count >= probeset_size:
            _annotate_rf_panel_metadata(
                builder=attempt_builder,
                probeset_size=probeset_size,
                selected_count=selected_count,
                metadata=metadata,
            )
            _save_rf_outputs(
                builder=attempt_builder,
                results_dir=results_dir,
                gene_summary=attempt_builder.metadata.get("_rf_gene_summary"),
                gene_scoring_df=attempt_builder.metadata.get("_rf_gene_scoring_df"),
                candidate_genes=candidate_genes,
                use_deg_prefilter=True,
                n_folds=n_folds,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_seeds=random_seeds,
            )
            return attempt_builder

    if best_builder is None:
        raise ValueError("RF-DEG failed: no DEG candidate genes available after 3 attempts")

    logger.warning(
        "RF-DEG remains short after 3 recompute attempts: selected %d/%d. "
        "Keeping available RF-DEG genes and marking panel as short.",
        best_selected_count,
        probeset_size,
    )
    metadata["rf_short_panel_allowed"] = True
    metadata["short_panel_reason"] = "rf_deg_candidate_pool_exhausted_after_retries"
    _annotate_rf_panel_metadata(
        builder=best_builder,
        probeset_size=probeset_size,
        selected_count=best_selected_count,
        metadata=metadata,
    )
    _save_rf_outputs(
        builder=best_builder,
        results_dir=results_dir,
        gene_summary=best_builder.metadata.get("_rf_gene_summary"),
        gene_scoring_df=best_builder.metadata.get("_rf_gene_scoring_df"),
        candidate_genes=best_builder.metadata.get("_rf_candidate_genes", []),
        use_deg_prefilter=True,
        n_folds=n_folds,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_seeds=random_seeds,
    )
    return best_builder


def _run_rf_on_candidate_genes(
    adata: AnnData,
    candidate_genes: list[str],
    y_labels,
    probeset_size: int,
    max_depth: int,
    n_estimators: int,
    n_folds: int,
    random_seeds: list[int],
    use_deg_prefilter: bool,
    results_dir: Optional[str] = None,
) -> GeneListBuilder:
    """Train RF on a fixed candidate gene set and return a ranked builder."""
    strategy_name = "rf_deg" if use_deg_prefilter else "rf_simple"
    builder = GeneListBuilder(strategy_name=strategy_name, analysis_type="global")

    X_source = adata[:, candidate_genes].X
    if issparse(X_source):
        X = X_source.toarray()
    else:
        X = np.asarray(X_source)

    if is_anndata_raw(adata):
        logger.warning(
            "Data appears to contain raw counts (integer values). "
            "Random Forest typically works better with log-normalized data."
        )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    class_names = [str(c) for c in label_encoder.classes_]
    n_classes = len(class_names)

    all_gene_scores = []
    # F1-weighted per-(gene, class) Gini importance decomposition; rows in
    # ``candidate_genes`` order.
    class_accum = np.zeros((len(candidate_genes), n_classes), dtype=np.float64)
    for seed_idx, seed in enumerate(random_seeds):
        logger.info(f"Running CV with seed {seed} ({seed_idx + 1}/{len(random_seeds)})")
        kfold = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y_encoded)):
            logger.info(f"  Fold {fold_idx + 1}/{n_folds}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # Class imbalance is handled once, via `class_weight="balanced"` —
            # do NOT also pass a balanced `sample_weight` (double correction).
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=seed + fold_idx,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="macro")
            logger.info(f"    F1 score: {f1:.4f}")

            for gene, importance in zip(candidate_genes, model.feature_importances_):
                all_gene_scores.append(
                    {
                        "gene": gene,
                        "fold": f"seed{seed}_fold{fold_idx}",
                        "importance": importance,
                        "f1_score": f1,
                    }
                )

            class_accum += f1 * _forest_per_class_gini_importance(model, n_classes)

    gene_scoring_df = pd.DataFrame(all_gene_scores)
    gene_scoring_df["weighted_importance"] = (
        gene_scoring_df["importance"] * gene_scoring_df["f1_score"]
    )
    gene_summary = gene_scoring_df.groupby("gene").agg(
        weighted_importance=("weighted_importance", "mean"),
    ).reset_index()
    gene_summary["final_score"] = gene_summary["weighted_importance"]
    gene_summary = gene_summary.sort_values("final_score", ascending=False)

    # Per-cell-type attribution (does not change rank / final_score / selection).
    gene_class_importance = _attach_rf_celltype_fields(
        gene_summary, class_accum, candidate_genes, class_names
    )

    builder.add_genes(
        genes=gene_summary["gene"].tolist(),
        ranks=list(range(1, len(gene_summary) + 1)),
        scores=gene_summary["final_score"].tolist(),
        **_rf_celltype_columns_from_summary(gene_summary),
    )

    selected_genes = gene_summary.head(probeset_size)["gene"].tolist()
    for gene in selected_genes:
        builder.mark_selected(gene)

    builder.add_metadata("_rf_gene_summary", gene_summary)
    builder.add_metadata("_rf_gene_scoring_df", gene_scoring_df)
    builder.add_metadata("_rf_gene_class_importance", gene_class_importance)
    builder.add_metadata("_rf_candidate_genes", candidate_genes)

    if results_dir:
        _save_rf_outputs(
            builder=builder,
            results_dir=results_dir,
            gene_summary=gene_summary,
            gene_scoring_df=gene_scoring_df,
            candidate_genes=candidate_genes,
            use_deg_prefilter=use_deg_prefilter,
            n_folds=n_folds,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_seeds=random_seeds,
        )

    return builder


def _annotate_rf_panel_metadata(
    builder: GeneListBuilder,
    probeset_size: int,
    selected_count: int,
    metadata: dict,
) -> None:
    """Add panel-size and RF retry metadata to a builder."""
    panel_status = "complete" if selected_count >= probeset_size else "short"
    builder.add_metadata("requested_panel_size", probeset_size)
    builder.add_metadata("final_panel_size", selected_count)
    builder.add_metadata("panel_size_status", panel_status)
    builder.add_metadata("short_panel_reason", metadata.get("short_panel_reason"))
    builder.add_metadata("rf_deg_recompute_attempts", metadata.get("rf_deg_recompute_attempts", 0))
    builder.add_metadata("rf_deg_candidate_pool_sizes", metadata.get("rf_deg_candidate_pool_sizes", []))
    builder.add_metadata("rf_deg_cache_used", metadata.get("rf_deg_cache_used", False))
    builder.add_metadata("rf_deg_cache_rejected_reason", metadata.get("rf_deg_cache_rejected_reason"))
    builder.add_metadata("rf_short_panel_allowed", metadata.get("rf_short_panel_allowed", False))


def _save_rf_outputs(
    builder: GeneListBuilder,
    results_dir: Optional[str],
    gene_summary: Optional[pd.DataFrame],
    gene_scoring_df: Optional[pd.DataFrame],
    candidate_genes: list[str],
    use_deg_prefilter: bool,
    n_folds: int,
    n_estimators: int,
    max_depth: int,
    random_seeds: list[int],
) -> None:
    """Save RF diagnostic outputs and cache."""
    if not results_dir:
        return

    os.makedirs(results_dir, exist_ok=True)
    _strategy_name = "rf_deg" if use_deg_prefilter else "rf_simple"
    builder.to_csv(os.path.join(results_dir, panel_information_filename(_strategy_name)))

    if gene_summary is None or gene_scoring_df is None:
        return

    model_dir = os.path.join(results_dir, "rf_models")
    os.makedirs(model_dir, exist_ok=True)
    cache_file = os.path.join(model_dir, "rf_gene_scores.pkl")
    try:
        cache_data = {
            "gene_summary": gene_summary,
            "all_gene_scores": gene_scoring_df,
            "gene_class_importance": builder.metadata.get("_rf_gene_class_importance"),
            "candidate_genes": candidate_genes,
            "n_genes": len(candidate_genes),
            "use_deg_prefilter": use_deg_prefilter,
            "n_folds": n_folds,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_seeds": random_seeds,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"✓ Saved RF results to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save RF cache: {e}")


def _load_rf_scores_from_cache(
    cache_file: str,
    adata: AnnData,
    probeset_size: int,
    use_deg_prefilter: bool,
    builder: GeneListBuilder,
    results_dir: Optional[str] = None,
) -> GeneListBuilder:
    """Rebuild a RF gene list from a previous ``rf_gene_scores.pkl`` cache."""
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"RF score cache file not found: {cache_file}")

    logger.info(f"Loading explicit RF score cache: {cache_file}")
    with open(cache_file, "rb") as f:
        cached_results = pickle.load(f)

    cached_prefilter = bool(cached_results.get("use_deg_prefilter", False))
    if cached_prefilter != use_deg_prefilter:
        raise ValueError(
            "RF score cache is incompatible with requested strategy: "
            f"cache use_deg_prefilter={cached_prefilter}, "
            f"requested use_deg_prefilter={use_deg_prefilter}"
        )

    if "gene_summary" not in cached_results:
        raise ValueError(f"RF score cache does not contain 'gene_summary': {cache_file}")

    gene_summary = cached_results["gene_summary"].copy()
    required_cols = {"gene", "final_score"}
    missing_cols = required_cols - set(gene_summary.columns)
    if missing_cols:
        raise ValueError(
            f"RF score cache missing required column(s): {sorted(missing_cols)} "
            f"in {cache_file}"
        )

    adata_genes = set(map(str, adata.var_names))
    n_before = len(gene_summary)
    gene_summary = gene_summary[gene_summary["gene"].astype(str).isin(adata_genes)].copy()
    gene_summary = gene_summary.sort_values("final_score", ascending=False).reset_index(drop=True)
    n_after = len(gene_summary)
    logger.info(
        f"RF cache genes compatible with input: {n_after}/{n_before} ranked genes present"
    )

    if n_after < probeset_size:
        raise RFCacheTooShortError(
            f"RF score cache has only {n_after} ranked genes present in input, "
            f"but target panel size is {probeset_size}: {cache_file}",
            n_ranked=n_after,
            cache_file=cache_file,
        )

    builder.add_genes(
        genes=gene_summary["gene"].tolist(),
        ranks=list(range(1, len(gene_summary) + 1)),
        scores=gene_summary["final_score"].tolist(),
        rf_score_cache_file=[cache_file] * len(gene_summary),
        **_rf_celltype_columns_from_summary(gene_summary),
    )

    selected_genes = gene_summary.head(probeset_size)["gene"].tolist()
    for gene in selected_genes:
        builder.mark_selected(gene)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        _strategy_name = "rf_deg" if use_deg_prefilter else "rf_simple"
        builder.to_csv(os.path.join(results_dir, panel_information_filename(_strategy_name)))

    logger.info(f"Selected {len(selected_genes)} genes from explicit RF cache")
    return builder


def _get_deg_genes_for_classification(
    adata: AnnData,
    celltype_column: str,
    n_deg_per_group: Optional[int],
    probeset_size: int,
    max_pval: float = DEFAULT_DEG_MAX_PVAL,
) -> list[str]:
    """Get DEG genes for pre-filtering before classification.

    Ranks genes by Scanpy DEG scores (Z-scores for Wilcoxon), keeps only genes
    whose Benjamini-Hochberg adjusted p-value is ``<= max_pval`` (per cell type),
    then takes the top ``n_deg_per_group`` per cell type. No fold-change cut-off
    is applied. Genes significant in several cell types are deduplicated, keeping
    the highest score.

    Args:
        adata: AnnData object
        celltype_column: Grouping column
        n_deg_per_group: Genes per group (if None, computed from probeset_size)
        probeset_size: Target size for final panel
        max_pval: Maximum adjusted p-value for inclusion. Pass ``1.0`` to disable
            the significance filter (used for the RF-DEG "all genes" fallback).

    Returns:
        List of DEG gene names ranked by Scanpy scores (descending)
    """
    # Ensure categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[celltype_column]):
        adata.obs[celltype_column] = adata.obs[celltype_column].astype("category")

    groups = adata.obs[celltype_column].cat.categories.tolist()

    # Calculate genes per group if not specified
    if n_deg_per_group is None:
        n_deg_per_group = max(50, probeset_size // len(groups) * 3)  # 3x oversampling

    logger.info(f"Running DEG analysis: {n_deg_per_group} genes per group")

    # Run DEG analysis
    sc.tl.rank_genes_groups(
        adata,
        groupby=celltype_column,
        method="wilcoxon",
        use_raw=False,
        n_genes=None,  # Get all genes to extract scores
        key_added="rank_genes_groups",
    )

    # Extract genes with scores per group
    apply_pval_filter = max_pval < 1.0
    deg_records = []
    for group in groups:
        try:
            # Use scanpy's get_rank_genes_groups_df to get proper scores
            group_df = sc.get.rank_genes_groups_df(adata, group=group, key='rank_genes_groups')
            # Keep only significant genes (adjusted p-value), then the top N per group
            if apply_pval_filter and 'pvals_adj' in group_df.columns:
                group_df = group_df[group_df['pvals_adj'] <= max_pval]
            group_df = group_df.head(n_deg_per_group)
            group_df['group'] = group
            deg_records.append(group_df)
        except (KeyError, ValueError) as e:
            logger.warning(f"Could not extract genes for group {group}: {e}")
            continue

    # Combine all groups
    if not deg_records:
        logger.warning("No DEG results extracted - returning empty list")
        return []
    
    deg_df = pd.concat(deg_records, ignore_index=True)
    
    # Remove duplicates, keeping highest score
    deg_df_dedup = deg_df.sort_values('scores', ascending=False).drop_duplicates(
        subset='names', keep='first'
    )
    
    # Return genes ranked by score
    deg_genes = deg_df_dedup['names'].tolist()

    _filt = f"adj. p <= {max_pval}" if apply_pval_filter else "no p-value filter"
    logger.info(
        f"Extracted {len(deg_genes)} unique DEG genes ranked by Scanpy scores ({_filt})"
    )

    return deg_genes
