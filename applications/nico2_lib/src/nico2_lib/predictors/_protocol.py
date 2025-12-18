from typing import Protocol
from numpy import number
from numpy.typing import NDArray


class Predictor(Protocol):
    def fit(
        self,
        X: NDArray[number],
        y: NDArray[number],
    ) -> "Predictor": ...

    def predict(
        self,
        X: NDArray[number],
    ) -> NDArray[number]: ...
