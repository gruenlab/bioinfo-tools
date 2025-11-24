
from typing import Protocol
from numpy.typing import NDArray
from numpy import number


class Predictor(Protocol):
    def fit(self, X: NDArray[number], y: NDArray[number]) -> None: ...
    def predict(self, X: NDArray[number]) -> NDArray[number]: ...
