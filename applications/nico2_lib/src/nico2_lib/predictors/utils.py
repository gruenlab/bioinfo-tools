
from typing import Protocol


class Predictor(Protocol):
    def fit(self): ...
    def predict(self): ...