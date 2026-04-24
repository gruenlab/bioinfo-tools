from typing import Callable

import numpy as np
from anndata.typing import AnnData
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from nico2_lib.typing import NumericArray


def find_components(
    adata: AnnData,
    embedding_function: Callable[[NumericArray], NumericArray] | None = None,
    min_clusters: int = 1,
    max_clusters: int = 10,
    default_number_of_factors: int = 3,
    seed: int = 42,
) -> tuple[int, np.ndarray]:
    """
    Optimizes cluster count using the elbow method and ensures minimum cluster size.
    """
    embedding_function = (
        PCA(n_components=25).fit_transform
        if embedding_function is None
        else embedding_function
    )
    embedding = embedding_function(adata.X)

    n_samples = embedding.shape[0]
    max_clusters = min(max_clusters, n_samples)
    cluster_range = range(min_clusters, max_clusters + 1)

    models = {}
    inertias = []

    for k in cluster_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(embedding)
        models[k] = km
        inertias.append(km.inertia_)

    elbow = KneeLocator(
        list(cluster_range), inertias, curve="convex", direction="decreasing"
    ).elbow

    number_of_factors = elbow if elbow is not None else default_number_of_factors
    number_of_factors = max(1, min(number_of_factors, n_samples))

    while number_of_factors > 1:
        labels = (
            models.get(number_of_factors).labels_
            if number_of_factors in models
            else KMeans(
                n_clusters=number_of_factors, random_state=seed, n_init="auto"
            ).fit_predict(embedding)
        )

        _, counts = np.unique(labels, return_counts=True)

        if counts.min() >= 5:
            break
        number_of_factors -= 1
    else:
        labels = np.zeros(n_samples, dtype=int)

    return int(number_of_factors), labels.astype(str)
