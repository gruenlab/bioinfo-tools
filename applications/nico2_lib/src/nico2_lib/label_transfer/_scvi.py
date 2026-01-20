import numpy as np
from anndata.typing import AnnData
from numpy.typing import NDArray


def scvi_transfer(
    adata: AnnData,
    reference: AnnData,
    annotation_key: str,
    device: str | None = None,
) -> NDArray[np.str_]:
    """Wrapper for scvis label transfer, returns array of labels"""
    try:
        import anndata as ad
        import scvi
    except ImportError as exc:
        raise ImportError(
            "scvi-tools and anndata are required for scvi label transfer."
        ) from exc

    if annotation_key not in reference.obs:
        raise KeyError(f"Missing '{annotation_key}' in reference.obs.")

    shared_genes = np.intersect1d(adata.var_names, reference.var_names)
    if shared_genes.size == 0:
        raise ValueError("No shared genes between adata and reference.")

    adata_sub = adata[:, shared_genes].copy()
    reference_sub = reference[:, shared_genes].copy()

    combined = ad.concat(
        {"query": adata_sub, "reference": reference_sub},
        label="dataset",
        index_unique=None,
        merge="same",
    )

    label_key = "scanvi_labels"
    combined.obs[label_key] = "Unknown"
    ref_mask = combined.obs["dataset"] == "reference"
    combined.obs.loc[ref_mask, label_key] = (
        reference_sub.obs[annotation_key].astype(str).values
    )

    layer = "counts" if "counts" in combined.layers else None
    scvi.model.SCVI.setup_anndata(
        combined,
        layer=layer,
        batch_key="dataset",
        labels_key=label_key,
    )
    train_kwargs: dict[str, object] = {}
    if device is not None:
        accelerator = "gpu" if device == "cuda" else device
        train_kwargs["accelerator"] = accelerator
        train_kwargs["devices"] = 1

    scvi_model = scvi.model.SCVI(combined)
    scvi_model.train(**train_kwargs)

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        adata=combined,
        labels_key=label_key,
        unlabeled_category="Unknown",
    )
    scanvi_model.train(max_epochs=20, **train_kwargs)

    predictions = scanvi_model.predict(combined)
    query_mask = combined.obs["dataset"] == "query"
    labels = np.array(predictions[query_mask], dtype=np.str_)

    return labels
