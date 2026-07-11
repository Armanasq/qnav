"""Direction cosine matrices (rotation matrices in SO(3)).

A DCM ``R_AB`` transforms coordinates from frame B to frame A:
``v_A = R_AB v_B`` (see ``docs/conventions.md`` §3). Shapes ``(..., 3, 3)``.

References: ``__data/math.md`` §1.4 (quaternion → R); Shepperd (1978) branch
method for R → quaternion as presented in standard attitude references
(``__data/attitude.pdf``). See ``docs/math/quaternions.md``.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import so3

__all__ = [
    "identity", "from_quaternion", "to_quaternion", "to_quaternion_robust",
    "from_axis_angle",
    "rot_x", "rot_y", "rot_z", "is_orthogonal", "orthonormalize",
    "orthogonality_error",
]


def identity(shape: tuple = ()) -> np.ndarray:
    """Identity DCM(s) of shape ``shape + (3, 3)``."""
    return np.broadcast_to(np.eye(3), shape + (3, 3)).copy()


def from_quaternion(q: np.ndarray) -> np.ndarray:
    """DCM of a unit quaternion ``q = [w, x, y, z]`` (homogeneous form).

    ``R = (w² − ‖u‖²) I + 2 u uᵀ + 2 w [u]ₓ`` — exactly the matrix in
    ``__data/math.md`` §1.4 / Solà eq. (117). Input is assumed unit-norm
    (caller contract); the homogeneous form degrades gracefully (scales by
    ‖q‖²) rather than skewing if the norm drifts.
    """
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = w * w + x * x - y * y - z * z
    R[..., 0, 1] = 2.0 * (x * y - w * z)
    R[..., 0, 2] = 2.0 * (x * z + w * y)
    R[..., 1, 0] = 2.0 * (x * y + w * z)
    R[..., 1, 1] = w * w - x * x + y * y - z * z
    R[..., 1, 2] = 2.0 * (y * z - w * x)
    R[..., 2, 0] = 2.0 * (x * z - w * y)
    R[..., 2, 1] = 2.0 * (y * z + w * x)
    R[..., 2, 2] = w * w - x * x - y * y + z * z
    return R


def to_quaternion(R: np.ndarray) -> np.ndarray:
    """Quaternion ``[w, x, y, z]`` of a DCM via **Shepperd's method**.

    Branches on the largest of ``{tr R, R₀₀, R₁₁, R₂₂}`` so the divisor is
    always ≥ 1, avoiding the catastrophic cancellation of the naive
    trace-only formula near θ = π. Output is canonicalized (``w ≥ 0``).
    """
    R = np.asarray(R, dtype=float)
    batch: tuple[int, ...] = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    n = int(Rf.shape[0])
    q = np.empty((n, 4))
    for i in range(n):
        m = Rf[i]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        choices = np.array([tr, m[0, 0], m[1, 1], m[2, 2]])
        k = int(np.argmax(choices))
        if k == 0:
            s = np.sqrt(1.0 + tr) * 2.0  # s = 4w
            q[i] = [0.25 * s,
                    (m[2, 1] - m[1, 2]) / s,
                    (m[0, 2] - m[2, 0]) / s,
                    (m[1, 0] - m[0, 1]) / s]
        elif k == 1:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0  # s = 4x
            q[i] = [(m[2, 1] - m[1, 2]) / s,
                    0.25 * s,
                    (m[0, 1] + m[1, 0]) / s,
                    (m[0, 2] + m[2, 0]) / s]
        elif k == 2:
            s = np.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2]) * 2.0  # s = 4y
            q[i] = [(m[0, 2] - m[2, 0]) / s,
                    (m[0, 1] + m[1, 0]) / s,
                    0.25 * s,
                    (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2]) * 2.0  # s = 4z
            q[i] = [(m[1, 0] - m[0, 1]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s,
                    0.25 * s]
        if q[i, 0] < 0:
            q[i] = -q[i]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.reshape(batch + (4,))


def to_quaternion_robust(R: np.ndarray) -> np.ndarray:
    """Quaternion via **Bar-Itzhack's eigenvector method** — the optimal
    extraction for *noisy* (non-orthogonal) matrices.

    Builds the Davenport K-matrix of the matrix elements and takes its
    dominant eigenvector. For an exact rotation this equals Shepperd's
    result; for a matrix with orthogonality error ε, Shepperd-style methods
    return the quaternion of *some nearby rotation* depending on the branch
    taken, whereas this returns the quaternion of the **closest rotation in
    the chordal sense** — equivalent to projecting onto SO(3) first, at
    roughly half the cost of an SVD.

    ~10× slower than :func:`to_quaternion` for exact inputs; use it when the
    matrix comes from numerical integration, filtering, or interpolation.

    Reference: Bar-Itzhack, "New method for extracting the quaternion from a
    rotation matrix", JGCD 23(6), 2000.
    """
    R = np.asarray(R, dtype=float)
    batch: tuple[int, ...] = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    n = int(Rf.shape[0])
    q = np.empty((n, 4))
    for i in range(n):
        m = Rf[i]
        K = np.array([
            [m[0, 0] + m[1, 1] + m[2, 2], m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]],
            [m[2, 1] - m[1, 2], m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0]],
            [m[0, 2] - m[2, 0], m[0, 1] + m[1, 0], m[1, 1] - m[0, 0] - m[2, 2], m[1, 2] + m[2, 1]],
            [m[1, 0] - m[0, 1], m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], m[2, 2] - m[0, 0] - m[1, 1]],
        ]) / 3.0
        _, vec = np.linalg.eigh(K)
        v = vec[:, -1]
        q[i] = v if v[0] >= 0 else -v
    return q.reshape(batch + (4,))


def from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues formula for a (unit) axis and angle: ``Exp(angle · axis)``."""
    axis = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)[..., None]
    return so3.exp(angle * axis)


def _principal(c: np.ndarray, s: np.ndarray, k: int) -> np.ndarray:
    R = np.zeros(np.shape(c) + (3, 3))
    i, j = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[k]
    R[..., k, k] = 1.0
    R[..., i, i] = c
    R[..., j, j] = c
    R[..., i, j] = -s
    R[..., j, i] = s
    return R


def rot_x(angle: np.ndarray) -> np.ndarray:
    """Principal rotation about x. As ``R_AB``: frame B is frame A rotated by
    ``angle`` about x, and ``v_A = rot_x(angle) v_B``."""
    a = np.asarray(angle, dtype=float)
    return _principal(np.cos(a), np.sin(a), 0)


def rot_y(angle: np.ndarray) -> np.ndarray:
    """Principal rotation about y (see :func:`rot_x` for semantics)."""
    a = np.asarray(angle, dtype=float)
    return _principal(np.cos(a), np.sin(a), 1)


def rot_z(angle: np.ndarray) -> np.ndarray:
    """Principal rotation about z (see :func:`rot_x` for semantics)."""
    a = np.asarray(angle, dtype=float)
    return _principal(np.cos(a), np.sin(a), 2)


def orthogonality_error(R: np.ndarray) -> np.ndarray:
    """Frobenius norm of ``RᵀR − I`` (0 for a perfect rotation)."""
    R = np.asarray(R, dtype=float)
    E = np.swapaxes(R, -1, -2) @ R - np.eye(3)
    return np.linalg.norm(E, axis=(-2, -1))


def is_orthogonal(R: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Alias of :func:`qnav.attitude.so3.is_rotation` (orthogonal **and** det +1)."""
    return so3.is_rotation(R, tol=tol)


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Closest rotation in Frobenius norm (SVD polar projection).

    Explicit repair step — qnav never re-orthonormalizes behind your back.
    """
    return so3.project(R)
