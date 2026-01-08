import tacco as tc
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from anndata.typing import AnnData  # type: ignore


def annotate_best_labels(
    adata: AnnData,
    reference: AnnData,
    annotation_key: str,
) -> NDArray[np.str_]:
    """
    Minimal wrapper for tc.tl.annotate that returns numpy array of best-fit labels.
    Uses all tacco defaults.
    """
    annotation_df: pd.DataFrame = tc.tl.annotate(  # type: ignore
        adata,
        reference,
        annotation_key,
    )

    best_idx = annotation_df.values.argmax(axis=1)  # type: ignore
    labels = np.array(annotation_df.columns[best_idx], dtype=np.str_)  # type: ignore

    return labels
