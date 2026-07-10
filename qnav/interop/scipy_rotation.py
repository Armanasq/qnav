"""Bridge between qnav quaternions and ``scipy.spatial.transform.Rotation``.

qnav is Hamilton scalar-first ``[w, x, y, z]``; SciPy stores scalar-last
``[x, y, z, w]``. Both use the same passive/active algebra for
``Rotation.apply``: ``R.apply(v)`` equals :func:`qnav.attitude.quaternion.
rotate_vector` for the same physical rotation — equivalence is enforced by
tests, not assumed.

SciPy is imported lazily; calling any function without SciPy installed
raises ``ImportError`` naming the dependency.
"""

from __future__ import annotations

import numpy as np

from qnav._validate import ensure_unit_quaternion
from qnav.attitude import quaternion as quat

__all__ = ["from_scipy", "to_scipy"]


def _scipy_rotation():
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "qnav.interop.scipy_rotation requires SciPy: pip install 'qnav[interop]'"
        ) from exc
    return Rotation


def to_scipy(q: np.ndarray):
    """qnav quaternion(s) ``(..., 4)`` -> ``scipy...Rotation`` (batch-capable)."""
    Rotation = _scipy_rotation()
    a = ensure_unit_quaternion(q, "q")
    return Rotation.from_quat(quat.to_scalar_last(a))


def from_scipy(rotation) -> np.ndarray:
    """``scipy...Rotation`` -> qnav scalar-first unit quaternion(s).

    The result is canonicalized (non-negative scalar part) so round trips
    are deterministic under the quaternion double cover.
    """
    Rotation = _scipy_rotation()
    if not isinstance(rotation, Rotation):
        raise TypeError(f"expected scipy Rotation, got {type(rotation).__name__}")
    return quat.canonical(quat.from_scalar_last(np.asarray(rotation.as_quat(), dtype=float)))
