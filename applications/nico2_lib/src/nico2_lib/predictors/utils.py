from typing import Callable, Protocol
from numpy.typing import NDArray
from numpy import number
from anndata.typing import AnnData
import numpy as np

predictor_func = Callable[[AnnData, AnnData], AnnData]

class Predictor(Protocol):
    def fit(self, X: NDArray[number], y: NDArray[number]) -> None: ...
    def predict(self, X: NDArray[number]) -> NDArray[number]: ...


def placeholder(predictor: Predictor) -> Callable[[AnnData, AnnData], AnnData]:
    def func(adata_query: AnnData, adata_reference: AnnData) -> AnnData:
        shared_genes = np.intersect1d(adata_query.var_names, adata_reference.var_names)
        pass

    return func