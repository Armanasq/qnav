"""Centralized input validation for qnav public entry points.

Every check raises an explicit, typed exception — nothing is silently
normalized, clamped, repaired, or discarded. Functions that deliberately
normalize (e.g. :func:`qnav.attitude.quaternion.normalize`) document that
behavior and emit :class:`qnav.errors.NormalizationWarning`.

All helpers return the validated array as ``float64 ndarray`` so callers can
use them as their single ``asarray`` conversion point.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ensure_array",
    "ensure_finite",
    "ensure_shape",
    "ensure_vector3",
    "ensure_unit_quaternion",
    "ensure_rotation_matrix",
    "ensure_positive_dt",
    "ensure_positive",
    "ensure_nonnegative",
    "ensure_monotonic",
    "ensure_covariance",
]

_UNIT_TOL = 1e-6
_ORTHO_TOL = 1e-6
_SYM_TOL = 1e-9


def ensure_array(x: object, name: str) -> np.ndarray:
    """Convert to float64 ndarray; reject non-numeric input."""
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric array-like, got {type(x).__name__}") from exc
    return a


def ensure_finite(x: object, name: str) -> np.ndarray:
    """Reject NaN/Inf anywhere in the array."""
    a = ensure_array(x, name)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
    return a


def ensure_shape(x: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Require an exact trailing shape; ``-1`` entries match any size."""
    a = ensure_array(x, name)
    if a.ndim < len(shape):
        raise ValueError(f"{name} must have at least {len(shape)} dims with trailing shape "
                         f"{shape}, got shape {a.shape}")
    trailing = a.shape[a.ndim - len(shape):]
    for want, got in zip(shape, trailing):
        if want != -1 and want != got:
            raise ValueError(f"{name} must have trailing shape {shape}, got {a.shape}")
    return a


def ensure_vector3(x: object, name: str) -> np.ndarray:
    """A finite vector (or batch of vectors) with trailing dimension 3."""
    return ensure_finite(ensure_shape(x, (3,), name), name)


def ensure_unit_quaternion(q: object, name: str = "q", tol: float = _UNIT_TOL) -> np.ndarray:
    """A finite, unit-norm quaternion (scalar-first, trailing dim 4)."""
    a = ensure_finite(ensure_shape(q, (4,), name), name)
    n = np.linalg.norm(a, axis=-1)
    if np.any(np.abs(n - 1.0) > tol):
        worst = float(np.max(np.abs(n - 1.0)))
        raise ValueError(f"{name} must be unit-norm within {tol:g} (worst deviation {worst:.3g}); "
                         "normalize explicitly with qnav.attitude.quaternion.normalize")
    return a


def ensure_rotation_matrix(R: object, name: str = "R", tol: float = _ORTHO_TOL) -> np.ndarray:
    """A finite, proper orthogonal 3x3 matrix (batch-capable)."""
    a = ensure_finite(ensure_shape(R, (3, 3), name), name)
    err = np.abs(np.swapaxes(a, -1, -2) @ a - np.eye(3)).max()
    if err > tol:
        raise ValueError(f"{name} is not orthogonal within {tol:g} (max |R^T R - I| = {err:.3g})")
    if np.any(np.linalg.det(a) < 0):
        raise ValueError(f"{name} has determinant -1 (reflection, not a rotation)")
    return a


def ensure_positive_dt(dt: float, name: str = "dt") -> float:
    """A strictly positive, finite scalar time step."""
    d = float(dt)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar, got {dt!r}")
    return d


def ensure_positive(x: float, name: str) -> float:
    """A strictly positive, finite scalar."""
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar, got {x!r}")
    return v


def ensure_nonnegative(x: float, name: str) -> float:
    """A finite scalar >= 0."""
    v = float(x)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar, got {x!r}")
    return v


def ensure_monotonic(t: object, name: str = "t", strict: bool = True) -> np.ndarray:
    """A finite 1-D timestamp array, (strictly) increasing."""
    a = ensure_finite(t, name)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    d = np.diff(a)
    if strict and np.any(d <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    if not strict and np.any(d < 0):
        raise ValueError(f"{name} must be non-decreasing")
    return a


def ensure_covariance(P: object, dim: int, name: str = "P",
                      sym_tol: float = _SYM_TOL, psd_tol: float = 1e-10) -> np.ndarray:
    """A finite, symmetric, positive-semidefinite ``dim x dim`` matrix.

    Symmetry is checked to ``sym_tol`` (relative to the largest element);
    PSD via smallest eigenvalue of the symmetrized matrix >= ``-psd_tol``.
    Returns the array unchanged (not symmetrized).
    """
    a = ensure_finite(P, name)
    if a.shape != (dim, dim):
        raise ValueError(f"{name} must have shape ({dim}, {dim}), got {a.shape}")
    scale = max(float(np.abs(a).max()), 1.0)
    if float(np.abs(a - a.T).max()) > sym_tol * scale:
        raise ValueError(f"{name} is not symmetric within {sym_tol:g} (relative)")
    w = np.linalg.eigvalsh(0.5 * (a + a.T))
    if float(w.min()) < -psd_tol * scale:
        raise ValueError(f"{name} is not positive semidefinite "
                         f"(min eigenvalue {float(w.min()):.3g})")
    return a
