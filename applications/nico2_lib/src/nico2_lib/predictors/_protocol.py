from typing import Protocol
from numpy import number
from numpy.typing import NDArray


class PredictorProtocol(Protocol):
    def fit(
        self,
        X: NDArray[number],
        y: NDArray[number],
    ) -> "PredictorProtocol": ...

    def predict(
        self,
        X: NDArray[number],
    ) -> NDArray[number]: ...
