"""
Consensus Non-negative Matrix Factorization (cNMF) for robust gene discovery.

This module implements the consensus NMF algorithm which runs NMF multiple
times and identifies stable, reproducible gene expression programs through
clustering and consensus building. Based on Kotliar et al. 2019 (eLife).

Reference: https://elifesciences.org/articles/43803
GitHub: https://github.com/dylkot/cNMF
"""

from __future__ import annotations

import logging
import gc
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Use absolute import instead of relative to avoid issues when imported from other modules
try:
    from ._constants import DEFAULT_DENSITY_THRESHOLD, DEFAULT_LOCAL_NEIGHBORHOOD_SIZE, DEFAULT_KMEANS_N_INIT
except ImportError:
    # Fallback for when module is imported directly (not as part of package)
    from _constants import DEFAULT_DENSITY_THRESHOLD, DEFAULT_LOCAL_NEIGHBORHOOD_SIZE, DEFAULT_KMEANS_N_INIT

__all__ = [
    'ConsensusNmf',
    'run_consensus_nmf_per_celltype',
    'run_consensus_nmf_global',
    'select_optimal_k',
]

logger = logging.getLogger(__name__)


##############################################################################
# CONSENSUS NMF CLASS
##############################################################################


class ConsensusNmf:
    """
    Consensus Non-negative Matrix Factorization (cNMF).

    This class implements the consensus NMF algorithm which runs NMF multiple
    times and identifies stable, reproducible gene expression programs through
    clustering and consensus building.

    Attributes:
        k_values: List of K values (number of components) to test.
        n_iter: Number of NMF iterations per K value.
        random_state: Base random seed for reproducibility.
        nmf_kwargs: Additional parameters for sklearn's NMF.
        results: Storage for all results by K value.
    """

    def __init__(self,
                 k_values: Union[int, List[int]] = 5,
                 n_iter: int = 100,
                 random_state: int = 42,
                 nmf_init: str = "nndsvda",
                 beta_loss: str = "kullback-leibler",
                 solver: str = "mu",
                 max_iter: int = 1000,
                 alpha_W: float = 0.0,
                 alpha_H: float = 0.0,
                 l1_ratio: float = 0.0):
        """
        Initialize ConsensusNmf.

        Args:
            k_values: K value(s) for number of components to test.
            n_iter: Number of NMF iterations per K (default: 100).
            random_state: Base random seed (default: 42).
            nmf_init: NMF initialization method (default: "nndsvda").
            beta_loss: Beta loss function (default: "kullback-leibler").
            solver: NMF solver (default: "mu" for multiplicative update).
            max_iter: Maximum iterations per NMF run (default: 1000).
            alpha_W: L2 regularization for W (usage) matrix (default: 0.0).
            alpha_H: L2 regularization for H (spectra) matrix (default: 0.0).
            l1_ratio: L1/L2 regularization ratio (default: 0.0).
        """
        # Store K values as list
        if isinstance(k_values, int):
            self.k_values = [k_values]
        else:
            self.k_values = sorted(k_values)

        self.n_iter = n_iter
        self.random_state = random_state

        # NMF parameters
        self.nmf_kwargs = {
            'init': nmf_init,
            'beta_loss': beta_loss,
            'solver': solver,
            'max_iter': max_iter,
            'alpha_W': alpha_W,
            'alpha_H': alpha_H,
            'l1_ratio': l1_ratio
        }

        # Results storage
        self.results = {}

        logger.info(f"Initialized ConsensusNmf with K={self.k_values}, n_iter={n_iter}")

    def factorize(self,
                  X: np.ndarray,
                  gene_names: Optional[np.ndarray] = None) -> Dict:
        """
        Run multiple NMF iterations for all K values.

        Args:
            X: Input data matrix (cells × genes) - should be raw counts.
            gene_names: Gene names corresponding to columns of X.

        Returns:
            Results for all K values containing merged spectra and usage matrices.
        """
        logger.info(f"Starting factorization for K values: {self.k_values}")

        # Handle sparse matrices
        if scipy.sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        # Filter genes with zero standard deviation
        std = np.std(X_dense, axis=0)
        valid_genes_mask = std > 0
        X_filtered = X_dense[:, valid_genes_mask]

        if gene_names is not None:
            filtered_gene_names = gene_names[valid_genes_mask]
        else:
            filtered_gene_names = np.arange(X_filtered.shape[1])

        n_cells, n_genes = X_filtered.shape
        logger.info(f"Data shape after filtering: {n_cells} cells × {n_genes} genes")

        # Run factorization for each K
        for k in self.k_values:
            logger.info(f"=== Running factorization for K={k} ===")

            if n_genes < k:
                logger.warning(f"Number of genes ({n_genes}) < K ({k}), skipping this K")
                continue

            all_H = []
            all_W = []
            all_errors = []

            # Run multiple NMF iterations
            for iteration in range(self.n_iter):
                # Use different random seed for each iteration
                iter_seed = self.random_state + iteration

                try:
                    nmf_model = NMF(
                        n_components=k,
                        random_state=iter_seed,
                        **self.nmf_kwargs
                    )

                    W = nmf_model.fit_transform(X_filtered)
                    H = nmf_model.components_

                    # Calculate reconstruction error
                    reconstruction = W @ H
                    error = np.linalg.norm(X_filtered - reconstruction, 'fro')

                    all_H.append(H)
                    all_W.append(W)
                    all_errors.append(error)

                    if (iteration + 1) % 20 == 0:
                        logger.info(f"  Completed {iteration + 1}/{self.n_iter} iterations for K={k}")

                except Exception as e:
                    logger.warning(f"  Iteration {iteration} failed for K={k}: {e}")
                    continue

            if len(all_H) == 0:
                logger.error(f"All iterations failed for K={k}")
                continue

            logger.info(f"Successfully completed {len(all_H)}/{self.n_iter} iterations for K={k}")

            # Store results
            self.results[k] = {
                'all_H': all_H,
                'all_W': all_W,
                'errors': all_errors,
                'n_iter_successful': len(all_H),
                'gene_names': filtered_gene_names,
                'mean_error': np.mean(all_errors),
                'std_error': np.std(all_errors)
            }

            logger.info(f"K={k}: Mean reconstruction error = {np.mean(all_errors):.4f} ± {np.std(all_errors):.4f}")

            # Clean up memory
            gc.collect()

        return self.results

    def consensus(self,
                  k: int,
                  density_threshold: float = DEFAULT_DENSITY_THRESHOLD,
                  local_neighborhood_size: float = DEFAULT_LOCAL_NEIGHBORHOOD_SIZE,
                  refit_usage: bool = True) -> Dict:
        """
        Compute consensus for a specific K value.

        Args:
            k: K value to compute consensus for.
            density_threshold: Threshold for filtering outlier spectra (default: 0.5).
                Lower values = more strict filtering.
            local_neighborhood_size: Fraction of spectra to use for local density
                calculation (default: 0.30).
            refit_usage: Whether to refit usage matrix with consensus spectra
                (default: True).

        Returns:
            Consensus results including consensus_H, consensus_W, cluster labels, etc.

        Raises:
            ValueError: If K value not found in results.
        """
        if k not in self.results:
            raise ValueError(f"K={k} not found in results. Run factorize() first.")

        logger.info(f"=== Computing consensus for K={k} ===")

        results_k = self.results[k]
        all_H = results_k['all_H']
        n_iter = len(all_H)

        # Merge all H matrices into one array
        merged_H = np.vstack(all_H)
        logger.info(f"Merged spectra shape: {merged_H.shape} ({n_iter} iterations × {k} components)")

        # Step 1: L2-normalize spectra to unit length
        H_norms = np.linalg.norm(merged_H, axis=1, keepdims=True)
        H_norms[H_norms == 0] = 1
        H_normalized = merged_H / H_norms

        # Step 2: Compute pairwise Euclidean distances
        logger.info("Computing pairwise distances...")
        distances = squareform(pdist(H_normalized, metric='euclidean'))

        # Step 3: Calculate local density
        logger.info("Calculating local density...")
        n_neighbors = max(1, int(local_neighborhood_size * n_iter))
        logger.info(f"Using {n_neighbors} nearest neighbors for density calculation")

        local_densities = []
        for i in range(len(H_normalized)):
            dists = distances[i, :]
            k_nearest = np.partition(dists, min(n_neighbors, len(dists)-1))[:n_neighbors]
            local_density = np.mean(k_nearest)
            local_densities.append(local_density)

        local_densities = np.array(local_densities)

        # Step 4: Filter by density threshold
        density_normalized = (local_densities - local_densities.min()) / (local_densities.max() - local_densities.min() + 1e-10)
        mask_pass = density_normalized < density_threshold
        n_filtered = mask_pass.sum()
        n_removed = len(mask_pass) - n_filtered

        logger.info(f"Density filtering: kept {n_filtered}/{len(mask_pass)} spectra (removed {n_removed})")

        if n_filtered < k:
            logger.warning(f"Too few spectra passed filtering ({n_filtered} < {k}). Using all spectra.")
            mask_pass = np.ones(len(mask_pass), dtype=bool)
            n_filtered = len(mask_pass)

        H_filtered = H_normalized[mask_pass]

        # Step 5: K-means clustering
        logger.info(f"Running K-means clustering with K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=DEFAULT_KMEANS_N_INIT)
        cluster_labels = kmeans.fit_predict(H_filtered)

        # Calculate silhouette score
        if n_filtered > k:
            silhouette = silhouette_score(H_filtered, cluster_labels)
            logger.info(f"Silhouette score: {silhouette:.4f}")
        else:
            silhouette = np.nan
            logger.warning("Not enough samples for silhouette score")

        # Step 6: Compute consensus spectra (median per cluster)
        logger.info("Computing consensus spectra...")
        consensus_H = np.zeros((k, merged_H.shape[1]))
        cluster_sizes = []

        for cluster_id in range(k):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = cluster_mask.sum()
            cluster_sizes.append(cluster_size)

            if cluster_size == 0:
                logger.warning(f"Cluster {cluster_id} is empty!")
                continue

            cluster_spectra = merged_H[mask_pass][cluster_mask]
            consensus_spectrum = np.median(cluster_spectra, axis=0)
            consensus_spectrum = consensus_spectrum / (consensus_spectrum.sum() + 1e-10)
            consensus_H[cluster_id] = consensus_spectrum

            logger.info(f"  Cluster {cluster_id}: {cluster_size} spectra")

        # Step 7: Refit usage matrix (optional)
        if refit_usage and 'all_W' in results_k and len(results_k['all_W']) > 0:
            logger.info("Refitting usage matrix with consensus spectra...")
            mean_W = np.mean(results_k['all_W'], axis=0)
            consensus_W = mean_W
        else:
            consensus_W = None

        # Create results dictionary
        consensus_results = {
            'consensus_H': consensus_H,
            'consensus_W': consensus_W,
            'cluster_labels': cluster_labels,
            'cluster_sizes': cluster_sizes,
            'silhouette_score': silhouette,
            'n_filtered': n_filtered,
            'n_removed': n_removed,
            'density_threshold': density_threshold,
            'local_densities': local_densities,
            'gene_names': results_k['gene_names']
        }

        # Store consensus results
        self.results[k]['consensus'] = consensus_results

        logger.info(f"=== Consensus computation complete for K={k} ===")

        return consensus_results

    def k_selection_metrics(self) -> pd.DataFrame:
        """
        Compute K-selection metrics for all K values.

        Returns:
            Metrics for each K value including reconstruction error and stability.
        """
        logger.info("Computing K-selection metrics...")

        metrics = []

        for k in self.k_values:
            if k not in self.results:
                continue

            results_k = self.results[k]

            metric_dict = {
                'k': k,
                'mean_error': results_k.get('mean_error', np.nan),
                'std_error': results_k.get('std_error', np.nan),
                'n_iter_successful': results_k.get('n_iter_successful', 0)
            }

            # Add consensus metrics if available
            if 'consensus' in results_k:
                consensus = results_k['consensus']
                metric_dict['silhouette_score'] = consensus.get('silhouette_score', np.nan)
                metric_dict['n_filtered'] = consensus.get('n_filtered', 0)
                metric_dict['n_removed'] = consensus.get('n_removed', 0)
            else:
                metric_dict['silhouette_score'] = np.nan
                metric_dict['n_filtered'] = 0
                metric_dict['n_removed'] = 0

            metrics.append(metric_dict)

        metrics_df = pd.DataFrame(metrics)

        logger.info("K-selection metrics computed:")
        logger.info(f"\n{metrics_df.to_string()}")

        return metrics_df

    def plot_k_selection(self,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 5)):
        """
        Create K-selection plots showing reconstruction error and stability.

        Args:
            save_path: Path to save the figure.
            figsize: Figure size (width, height).
        """
        metrics_df = self.k_selection_metrics()

        if len(metrics_df) == 0:
            logger.warning("No metrics available for plotting")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Reconstruction error
        ax = axes[0]
        ax.errorbar(metrics_df['k'], metrics_df['mean_error'],
                   yerr=metrics_df['std_error'], marker='o', capsize=5)
        ax.set_xlabel('Number of Components (K)')
        ax.set_ylabel('Reconstruction Error')
        ax.set_title('Reconstruction Error vs K')
        ax.grid(True, alpha=0.3)

        # Plot 2: Silhouette score (stability)
        ax = axes[1]
        valid_silhouette = metrics_df.dropna(subset=['silhouette_score'])
        if len(valid_silhouette) > 0:
            ax.plot(valid_silhouette['k'], valid_silhouette['silhouette_score'],
                   marker='o', color='green')
            ax.set_xlabel('Number of Components (K)')
            ax.set_ylabel('Silhouette Score')
            ax.set_title('Clustering Stability vs K')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No silhouette scores available\n(run consensus() first)',
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"K-selection plot saved to {save_path}")

        return fig

    def get_top_genes(self,
                      k: int,
                      n_genes: int = 50,
                      use_consensus: bool = True) -> Dict[int, pd.DataFrame]:
        """
        Get top genes for each component/factor.

        Args:
            k: K value to extract genes from.
            n_genes: Number of top genes per component (default: 50).
            use_consensus: Use consensus spectra if available (default: True).

        Returns:
            Dictionary mapping component index to DataFrame of top genes.

        Raises:
            ValueError: If K value not found or no spectra available.
        """
        if k not in self.results:
            raise ValueError(f"K={k} not found in results")

        results_k = self.results[k]

        # Decide which H matrix to use
        if use_consensus and 'consensus' in results_k:
            H = results_k['consensus']['consensus_H']
            source = "consensus"
        elif 'all_H' in results_k and len(results_k['all_H']) > 0:
            H = np.mean(results_k['all_H'], axis=0)
            source = "mean"
        else:
            raise ValueError(f"No spectra available for K={k}")

        gene_names = results_k['gene_names']

        logger.info(f"Extracting top {n_genes} genes per component from {source} spectra (K={k})")

        top_genes_by_component = {}

        for component_idx in range(k):
            component_weights = H[component_idx, :]
            top_indices = np.argsort(component_weights)[::-1][:n_genes]
            top_gene_names = gene_names[top_indices]
            top_weights = component_weights[top_indices]

            df = pd.DataFrame({
                'gene': top_gene_names,
                'weight': top_weights,
                'rank': np.arange(1, len(top_gene_names) + 1)
            })

            top_genes_by_component[component_idx] = df

            logger.info(f"  Component {component_idx}: Top gene = {top_gene_names[0]} (weight={top_weights[0]:.4f})")

        return top_genes_by_component


##############################################################################
# CONVENIENCE FUNCTIONS FOR INTEGRATION WITH EXISTING PIPELINE
##############################################################################


def run_consensus_nmf_per_celltype(
        adata,
        groupby: str = 'original',
        k_values: Union[int, List[int]] = [3, 5, 7, 9],
        n_iter: int = 100,
        random_state: int = 42,
        density_threshold: float = DEFAULT_DENSITY_THRESHOLD,
        results_dir: Optional[str] = None,
        **nmf_kwargs) -> Dict:
    """
    Run consensus NMF for each cell type separately.

    This function integrates cNMF into the existing per-celltype analysis workflow.

    Args:
        adata: Annotated data matrix.
        groupby: Column in adata.obs for grouping by cell type (default: 'original').
        k_values: K value(s) to test (default: [3, 5, 7, 9]).
        n_iter: Number of NMF iterations per K (default: 100).
        random_state: Random seed for reproducibility (default: 42).
        density_threshold: Threshold for consensus filtering (default: 0.5).
        results_dir: Directory to save results.
        **nmf_kwargs: Additional NMF parameters.

    Returns:
        Results for each cell type containing consensus spectra and metrics.
    """
    logger.info(f"=== Running Consensus NMF Per-Celltype ===")
    logger.info(f"K values to test: {k_values}")
    logger.info(f"Iterations per K: {n_iter}")

    # Validate groupby column
    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    # Ensure categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        adata.obs[groupby] = adata.obs[groupby].astype('category')

    # Determine valid cell types
    valid_celltypes = []
    for celltype in adata.obs[groupby].cat.categories:
        subset = adata[adata.obs[groupby] == celltype]
        min_k = max(k_values) if isinstance(k_values, list) else k_values
        if subset.shape[0] >= 10 and subset.shape[0] >= min_k * 2:
            valid_celltypes.append(celltype)
        else:
            logger.warning(f"Cell type '{celltype}' skipped: insufficient cells (has {subset.shape[0]})")

    if len(valid_celltypes) == 0:
        logger.error("No cell types have enough cells for analysis!")
        return {}

    logger.info(f"Valid cell types: {len(valid_celltypes)}/{len(adata.obs[groupby].cat.categories)}")

    # Process each cell type
    celltype_results = {}

    for celltype in valid_celltypes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing cell type: {celltype}")
        logger.info(f"{'='*60}")

        # Subset data
        subset = adata[adata.obs[groupby] == celltype].copy()
        n_cells = subset.shape[0]

        # Check for counts layer (need raw counts for NMF)
        if 'counts' not in subset.layers:
            logger.error(f"No 'counts' layer found for {celltype}! Skipping.")
            continue

        # Extract raw counts
        raw_counts = subset.layers['counts']
        gene_names = subset.var_names.to_numpy()

        # Initialize ConsensusNmf
        cnmf = ConsensusNmf(
            k_values=k_values,
            n_iter=n_iter,
            random_state=random_state,
            **nmf_kwargs
        )

        # Run factorization
        try:
            cnmf.factorize(raw_counts, gene_names=gene_names)
        except Exception as e:
            logger.error(f"Factorization failed for {celltype}: {e}")
            continue

        # Run consensus for each K
        consensus_results = {}
        for k in cnmf.k_values:
            if k in cnmf.results:
                try:
                    consensus = cnmf.consensus(k, density_threshold=density_threshold)
                    consensus_results[k] = consensus
                except Exception as e:
                    logger.error(f"Consensus computation failed for {celltype}, K={k}: {e}")

        # Store results
        celltype_results[celltype] = {
            'cnmf_object': cnmf,
            'consensus_by_k': consensus_results,
            'n_cells': n_cells,
            'n_genes': raw_counts.shape[1]
        }

        # Save results if directory provided
        if results_dir:
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            safe_celltype = celltype.replace(' ', '_').replace('/', '_')

            # Save K-selection plot
            plot_path = Path(results_dir) / f"k_selection_{safe_celltype}.png"
            cnmf.plot_k_selection(save_path=str(plot_path))

            # Save metrics
            metrics_df = cnmf.k_selection_metrics()
            metrics_path = Path(results_dir) / f"k_metrics_{safe_celltype}.csv"
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Saved metrics to {metrics_path}")

        # Clean up
        gc.collect()

    logger.info(f"\n{'='*60}")
    logger.info(f"Consensus NMF per-celltype complete: {len(celltype_results)} cell types processed")
    logger.info(f"{'='*60}\n")

    return celltype_results


def run_consensus_nmf_global(
        adata,
        k_values: Union[int, List[int]] = [5, 10, 15, 20],
        n_iter: int = 100,
        random_state: int = 42,
        density_threshold: float = DEFAULT_DENSITY_THRESHOLD,
        results_dir: Optional[str] = None,
        **nmf_kwargs) -> Dict:
    """
    Run consensus NMF on the whole dataset (global analysis).

    Args:
        adata: Annotated data matrix.
        k_values: K value(s) to test (default: [5, 10, 15, 20]).
        n_iter: Number of NMF iterations per K (default: 100).
        random_state: Random seed for reproducibility (default: 42).
        density_threshold: Threshold for consensus filtering (default: 0.5).
        results_dir: Directory to save results.
        **nmf_kwargs: Additional NMF parameters.

    Returns:
        Results containing consensus spectra and metrics for all K values.
    """
    logger.info(f"=== Running Global Consensus NMF ===")
    logger.info(f"K values to test: {k_values}")
    logger.info(f"Iterations per K: {n_iter}")
    logger.info(f"Dataset shape: {adata.shape}")

    # Check for counts layer
    if 'counts' not in adata.layers:
        logger.error("No 'counts' layer found! Cannot run NMF.")
        return {}

    # Extract raw counts
    raw_counts = adata.layers['counts']
    gene_names = adata.var_names.to_numpy()

    # Initialize ConsensusNmf
    cnmf = ConsensusNmf(
        k_values=k_values,
        n_iter=n_iter,
        random_state=random_state,
        **nmf_kwargs
    )

    # Run factorization
    try:
        cnmf.factorize(raw_counts, gene_names=gene_names)
    except Exception as e:
        logger.error(f"Factorization failed: {e}")
        return {}

    # Run consensus for each K
    consensus_results = {}
    for k in cnmf.k_values:
        if k in cnmf.results:
            try:
                consensus = cnmf.consensus(k, density_threshold=density_threshold)
                consensus_results[k] = consensus
            except Exception as e:
                logger.error(f"Consensus computation failed for K={k}: {e}")

    # Store results
    results = {
        'cnmf_object': cnmf,
        'consensus_by_k': consensus_results,
        'n_cells': adata.shape[0],
        'n_genes': adata.shape[1]
    }

    # Save results if directory provided
    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Save K-selection plot
        plot_path = Path(results_dir) / "k_selection_global.png"
        cnmf.plot_k_selection(save_path=str(plot_path))

        # Save metrics
        metrics_df = cnmf.k_selection_metrics()
        metrics_path = Path(results_dir) / "k_metrics_global.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics to {metrics_path}")

    logger.info("=== Global Consensus NMF complete ===")

    return results


def select_optimal_k(cnmf_object: ConsensusNmf,
                     method: str = 'elbow') -> int:
    """
    Select optimal K value from consensus NMF results.

    Args:
        cnmf_object: Fitted ConsensusNmf object.
        method: Selection method: 'elbow', 'silhouette', or 'manual'
            (default: 'elbow').

    Returns:
        Optimal K value.
    """
    metrics_df = cnmf_object.k_selection_metrics()

    if len(metrics_df) == 0:
        logger.warning("No metrics available for K selection")
        return cnmf_object.k_values[0] if cnmf_object.k_values else 5

    if method == 'elbow':
        # Find elbow point in reconstruction error curve
        errors = metrics_df['mean_error'].values
        k_vals = metrics_df['k'].values

        if len(errors) < 2:
            return k_vals[0]

        # Normalize to [0, 1]
        errors_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-10)
        k_norm = (k_vals - k_vals.min()) / (k_vals.max() - k_vals.min() + 1e-10)

        # Calculate distance to line from first to last point
        distances = []
        for i in range(len(k_vals)):
            dist = abs((errors_norm[-1] - errors_norm[0]) * k_norm[i] -
                      (k_norm[-1] - k_norm[0]) * errors_norm[i] +
                      k_norm[-1] * errors_norm[0] - errors_norm[-1] * k_norm[0])
            distances.append(dist)

        optimal_idx = np.argmax(distances)
        optimal_k = k_vals[optimal_idx]

        logger.info(f"Optimal K selected (elbow method): {optimal_k}")

    elif method == 'silhouette':
        # Select K with highest silhouette score
        valid_metrics = metrics_df.dropna(subset=['silhouette_score'])
        if len(valid_metrics) == 0:
            logger.warning("No silhouette scores available, using first K")
            return metrics_df['k'].iloc[0]

        optimal_idx = valid_metrics['silhouette_score'].idxmax()
        optimal_k = valid_metrics.loc[optimal_idx, 'k']

        logger.info(f"Optimal K selected (silhouette method): {optimal_k}")

    else:
        logger.warning(f"Unknown method '{method}', using first K value")
        optimal_k = metrics_df['k'].iloc[0]

    return int(optimal_k)
