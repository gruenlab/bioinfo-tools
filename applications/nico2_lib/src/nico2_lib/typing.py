from __future__ import annotations

from typing import TypeAlias

from numpy import intp, number
from numpy.typing import NDArray

NumericArray: TypeAlias = NDArray[number]
IndexArray: TypeAlias = NDArray[intp]
