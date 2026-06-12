"""Unit-quaternion algebra (Hamilton convention, scalar-first ``[w, x, y, z]``).

All functions are vectorized over leading batch dimensions: quaternions have
shape ``(..., 4)``, vectors ``(..., 3)``. Functions never mutate inputs.

Semantics
---------
A quaternion ``q_AB`` transforms coordinates from frame B to frame A:
``v_A = R(q_AB) v_B`` with ``R`` from :func:`qnav.attitude.dcm.from_quaternion`.
Composition chains adjacent indices: ``q_AC = mul(q_AB, q_BC)``.

References
----------
Solà, "Quaternion kinematics for the error-state Kalman filter"
(``__data/Quaternion kinematics .../Quaternion.tex``); ``__data/math.md``.
See ``docs/source_index.md`` and ``docs/math/formula_catalog.md``.
"""

from __future__ import annotations

import warnings

import numpy as np

from qnav.errors import NormalizationWarning

__all__ = [
    "identity", "mul", "conjugate", "inverse", "norm", "normalize", "canonical",
    "rotate_vector", "rotate_frame", "exp", "log", "power", "angle", "axis",
    "relative", "angular_distance", "from_scalar_last", "to_scalar_last",
    "from_jpl", "to_jpl", "mean", "left_matrix", "right_matrix", "is_unit",
    "random",
]

#: Below this rotation angle (rad) series expansions replace sin/θ-type ratios.
SMALL_ANGLE = 1e-8


def identity(shape: tuple = ()) -> np.ndarray:
    """Identity quaternion ``[1, 0, 0, 0]``, optionally batched to ``shape + (4,)``."""
    q = np.zeros(shape + (4,))
    q[..., 0] = 1.0
    return q


def mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product ``q1 ⊗ q2`` (applies ``q2`` first, then ``q1``).

    Solà eq. (12): for ``p = [pw, pv]``, ``q = [qw, qv]``::

        p ⊗ q = [pw·qw − pv·qv,  pw·qv + qw·pv + pv × qv]
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    w1, v1 = q1[..., :1], q1[..., 1:]
    w2, v2 = q2[..., :1], q2[..., 1:]
    w = w1 * w2 - np.sum(v1 * v2, axis=-1, keepdims=True)
    v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    return np.concatenate([w, v], axis=-1)


def conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate ``q* = [w, −x, −y, −z]``; equals the inverse for unit quaternions."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def inverse(q: np.ndarray) -> np.ndarray:
    """General inverse ``q* / ‖q‖²`` (valid for non-unit quaternions)."""
    q = np.asarray(q, dtype=float)
    n2 = np.sum(q * q, axis=-1, keepdims=True)
    return conjugate(q) / n2


def norm(q: np.ndarray) -> np.ndarray:
    """Euclidean norm over the last axis."""
    return np.linalg.norm(np.asarray(q, dtype=float), axis=-1)


def normalize(q: np.ndarray, *, warn_tol: float | None = None) -> np.ndarray:
    """Return ``q / ‖q‖``.

    Raises ``ValueError`` on (near-)zero norm. If ``warn_tol`` is given, a
    :class:`NormalizationWarning` is issued when ``|‖q‖ − 1| > warn_tol``.
    """
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("cannot normalize a (near-)zero quaternion")
    if warn_tol is not None and np.any(np.abs(n - 1.0) > warn_tol):
        warnings.warn(
            f"quaternion norm deviates from 1 by more than {warn_tol}; renormalizing",
            NormalizationWarning, stacklevel=2,
        )
    return q / n


def is_unit(q: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Elementwise check ``|‖q‖ − 1| ≤ tol`` (returns a boolean array)."""
    return np.abs(norm(q) - 1.0) <= tol


def canonical(q: np.ndarray) -> np.ndarray:
    """Resolve the ±q double cover: flip sign so that ``w ≥ 0``.

    For ``w == 0`` exactly, the sign of the first nonzero vector component is
    made positive, giving a deterministic representative.
    """
    q = np.asarray(q, dtype=float).copy()
    w = q[..., 0]
    flip = w < 0
    # deterministic tie-break on the w == 0 great circle
    zero_w = w == 0
    if np.any(zero_w):
        v = q[..., 1:]
        first_nz = np.argmax(np.abs(v) > 0, axis=-1)
        sign = np.take_along_axis(v, first_nz[..., None], axis=-1)[..., 0]
        flip = flip | (zero_w & (sign < 0))
    q[flip] = -q[flip]
    return q


def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` by ``q``: returns the vector part of ``q ⊗ [0,v] ⊗ q*``.

    Frame reading: if ``q = q_AB`` then this maps coordinates B → A
    (``v_A = rotate_vector(q_AB, v_B)``). Active reading: rotates ``v`` within
    one frame by angle/axis of ``q``.

    Uses the expanded Rodrigues-style form (no quaternion products):
    ``v' = v + 2 w (u × v) + 2 u × (u × v)`` with ``u`` the vector part.
    """
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    w, u = q[..., :1], q[..., 1:]
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def rotate_frame(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Coordinate transform in the opposite direction: ``v_B = rotate_frame(q_AB, v_A)``.

    Equivalent to ``rotate_vector(conjugate(q), v)``.
    """
    return rotate_vector(conjugate(q), v)


def exp(phi: np.ndarray) -> np.ndarray:
    """Quaternion exponential of a pure quaternion given as rotation vector ``φ ∈ ℝ³``.

    ``Exp(φ) = [cos(θ/2), sin(θ/2)·u]`` with ``θ = ‖φ‖``, ``u = φ/θ``
    (Solà eq. (101)). Uses a 2nd-order series of ``sin(θ/2)/θ`` below
    :data:`SMALL_ANGLE` for stability at θ → 0.
    """
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi, axis=-1, keepdims=True)
    half = 0.5 * theta
    small = theta < SMALL_ANGLE
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(small, 0.5 - theta**2 / 48.0, np.sin(half) / np.where(small, 1.0, theta))
    w = np.cos(half)
    return np.concatenate([w, k * phi], axis=-1)


def log(q: np.ndarray) -> np.ndarray:
    """Rotation-vector logarithm of a unit quaternion (inverse of :func:`exp`).

    Returns ``φ = θ·u ∈ ℝ³`` with ``θ ∈ [0, π]`` after canonicalizing the sign
    (so the geodesic, not the long way, is returned). Uses
    ``θ = 2·atan2(‖v‖, w)`` which is stable for all angles.
    """
    q = canonical(np.asarray(q, dtype=float))
    w = q[..., :1]
    v = q[..., 1:]
    nv = np.linalg.norm(v, axis=-1, keepdims=True)
    theta = 2.0 * np.arctan2(nv, w)
    small = nv < SMALL_ANGLE
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(small, 2.0 / np.where(np.abs(w) < 1e-12, 1.0, w),
                     theta / np.where(small, 1.0, nv))
    return k * v


def power(q: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Quaternion power ``q^t = Exp(t · Log(q))`` (geodesic scaling)."""
    t = np.asarray(t, dtype=float)[..., None]
    return exp(t * log(q))


def angle(q: np.ndarray) -> np.ndarray:
    """Rotation angle ``θ ∈ [0, π]`` encoded by the unit quaternion."""
    q = np.asarray(q, dtype=float)
    return 2.0 * np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), np.abs(q[..., 0]))


def axis(q: np.ndarray) -> np.ndarray:
    """Unit rotation axis; returns ``[1, 0, 0]`` for the identity (θ = 0) by convention."""
    q = canonical(np.asarray(q, dtype=float))
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    fallback = np.zeros_like(v)
    fallback[..., 0] = 1.0
    safe = n > 1e-12
    return np.where(safe, v / np.where(safe, n, 1.0), fallback)


def relative(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Relative rotation ``q1* ⊗ q2`` (the rotation taking attitude 1 to attitude 2)."""
    return mul(conjugate(q1), q2)


def angular_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Geodesic angle on SO(3) between two attitudes, in ``[0, π]``.

    Sign-invariant (q ≡ −q): ``θ = 2·arccos(|⟨q1, q2⟩|)`` computed via the
    stable atan2 form of :func:`angle` on the relative quaternion.
    """
    return angle(relative(q1, q2))


def left_matrix(q: np.ndarray) -> np.ndarray:
    """Matrix ``[q]_L`` with ``mul(q, p) = [q]_L @ p`` (Solà eq. (17)). Shape ``(...,4,4)``."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    M = np.empty(q.shape[:-1] + (4, 4))
    M[..., 0, 0], M[..., 0, 1], M[..., 0, 2], M[..., 0, 3] = w, -x, -y, -z
    M[..., 1, 0], M[..., 1, 1], M[..., 1, 2], M[..., 1, 3] = x, w, -z, y
    M[..., 2, 0], M[..., 2, 1], M[..., 2, 2], M[..., 2, 3] = y, z, w, -x
    M[..., 3, 0], M[..., 3, 1], M[..., 3, 2], M[..., 3, 3] = z, -y, x, w
    return M


def right_matrix(q: np.ndarray) -> np.ndarray:
    """Matrix ``[q]_R`` with ``mul(p, q) = [q]_R @ p`` (Solà eq. (18)). Shape ``(...,4,4)``."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    M = np.empty(q.shape[:-1] + (4, 4))
    M[..., 0, 0], M[..., 0, 1], M[..., 0, 2], M[..., 0, 3] = w, -x, -y, -z
    M[..., 1, 0], M[..., 1, 1], M[..., 1, 2], M[..., 1, 3] = x, w, z, -y
    M[..., 2, 0], M[..., 2, 1], M[..., 2, 2], M[..., 2, 3] = y, -z, w, x
    M[..., 3, 0], M[..., 3, 1], M[..., 3, 2], M[..., 3, 3] = z, y, -x, w
    return M


def mean(qs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted chordal-mean attitude of quaternions ``qs`` with shape ``(N, 4)``.

    Markley et al. quaternion averaging: the mean is the eigenvector of
    ``M = Σ wᵢ qᵢ qᵢᵀ`` with the largest eigenvalue. Sign-invariant and
    optimal in the sense of minimizing weighted chordal distance.
    """
    qs = np.asarray(qs, dtype=float)
    if qs.ndim != 2 or qs.shape[1] != 4:
        raise ValueError("mean expects an (N, 4) array of quaternions")
    if weights is None:
        weights = np.ones(qs.shape[0])
    w = np.asarray(weights, dtype=float)
    M = (qs.T * w) @ qs
    eigval, eigvec = np.linalg.eigh(M)
    return canonical(eigvec[:, -1])


def from_scalar_last(q: np.ndarray) -> np.ndarray:
    """Convert ``[x, y, z, w]`` (SciPy/ROS layout) to qnav's ``[w, x, y, z]``."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def to_scalar_last(q: np.ndarray) -> np.ndarray:
    """Convert qnav's ``[w, x, y, z]`` to scalar-last ``[x, y, z, w]``."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def from_jpl(q_jpl: np.ndarray) -> np.ndarray:
    """Convert a JPL-convention quaternion (scalar-last, ``ij = −k``) to Hamilton.

    The same physical rotation is represented; the JPL quaternion equals the
    Hamilton conjugate with scalar moved first (Solà §1.2, Table 1, treating
    both as the same frame-mapping ``q_AB``).
    """
    return conjugate(from_scalar_last(q_jpl))


def to_jpl(q: np.ndarray) -> np.ndarray:
    """Inverse of :func:`from_jpl`."""
    return to_scalar_last(conjugate(q))


def random(shape: tuple = (), rng: np.random.Generator | None = None) -> np.ndarray:
    """Uniformly distributed random unit quaternion(s) (Haar measure on SO(3))."""
    rng = rng or np.random.default_rng()
    q = rng.standard_normal(shape + (4,))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)
