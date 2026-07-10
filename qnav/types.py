"""Public numerical type aliases used across qnav signatures.

qnav functions accept anything NumPy can coerce to a float array
(``ArrayLike``) and return float64 arrays (``FloatArray``). Scalar-or-array
parameters (latitude, altitude, gains, ...) are ``ScalarOrArray``.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import numpy.typing as npt

__all__ = ["ArrayLike", "FloatArray", "ScalarOrArray"]

#: Anything coercible to a float ndarray (lists, tuples, scalars, ndarrays).
ArrayLike = npt.ArrayLike

#: A float64 NumPy array of unspecified shape.
FloatArray = npt.NDArray[np.float64]

#: A scalar or a float array; broadcasting follows NumPy rules.
ScalarOrArray = Union[float, FloatArray]
