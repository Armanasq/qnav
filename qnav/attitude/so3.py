"""SO(3) as a Lie group: hat/vee, exponential/logarithm maps, Jacobians, ⊞/⊟.

Rotation matrices ``R ∈ SO(3)`` have shape ``(..., 3, 3)``; tangent vectors
(rotation vectors) ``θ ∈ ℝ³`` shape ``(..., 3)``. All functions are batched.

Conventions (see ``docs/conventions.md`` §5):

- ``Exp(θ + δθ) ≈ Exp(θ)·Exp(Jr(θ)·δθ)`` defines the **right** Jacobian;
  ``Jl(θ) = Jr(−θ) = Jr(θ)ᵀ``.
- ``boxplus(R, δ) = R·Exp(δ)`` is the **right/local** perturbation.

References: Solà (Quaternion kinematics, §SO(3)); Hashim, "Special Orthogonal
Group SO(3) ... Overview, Mapping and Challenges" (``__data/Special Orthogonal
Group .../Paper_Hashim_SUBMIT.tex``). See ``docs/math/so3.md``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "hat", "vee", "exp", "log", "left_jacobian", "right_jacobian",
    "left_jacobian_inverse", "right_jacobian_inverse", "boxplus", "boxminus",
    "geodesic_distance", "is_rotation", "project",
]

#: Angle threshold (rad) below which Taylor series replace trigonometric ratios.
SMALL_ANGLE = 1e-4
#: Margin from π below which the generic log branch is replaced by the stable one.
NEAR_PI = 1e-6


def hat(omega: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix ``[ω]× ∈ so(3)`` with ``hat(ω) @ v = ω × v``."""
    omega = np.asarray(omega, dtype=float)
    x, y, z = omega[..., 0], omega[..., 1], omega[..., 2]
    S = np.zeros(omega.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -z, y
    S[..., 1, 0], S[..., 1, 2] = z, -x
    S[..., 2, 0], S[..., 2, 1] = -y, x
    return S


def vee(Omega: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hat`: extract ``ω`` from a skew-symmetric matrix."""
    Omega = np.asarray(Omega, dtype=float)
    return np.stack(
        [Omega[..., 2, 1], Omega[..., 0, 2], Omega[..., 1, 0]], axis=-1
    )


def _sin_cos_coeffs(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable ``A = sin θ / θ`` and ``B = (1 − cos θ)/θ²`` with series fallback."""
    small = theta < SMALL_ANGLE
    safe = np.where(small, 1.0, theta)
    A = np.where(small, 1.0 - theta**2 / 6.0, np.sin(safe) / safe)
    B = np.where(small, 0.5 - theta**2 / 24.0, (1.0 - np.cos(safe)) / safe**2)
    return A, B


def exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map (Rodrigues): ``Exp(φ) = I + A·[φ]× + B·[φ]ײ``.

    ``A = sin θ/θ``, ``B = (1 − cos θ)/θ²``, ``θ = ‖φ‖``. Exact for all θ;
    series below :data:`SMALL_ANGLE`.
    """
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi, axis=-1)
    A, B = _sin_cos_coeffs(theta)
    K = hat(phi)
    K2 = K @ K
    eye3 = np.broadcast_to(np.eye(3), K.shape)
    return eye3 + A[..., None, None] * K + B[..., None, None] * K2


def log(R: np.ndarray) -> np.ndarray:
    """Logarithm map: rotation vector ``φ = θ·u`` with ``θ ∈ [0, π]``.

    Three branches for numerical stability:

    - generic: ``θ = arccos((tr R − 1)/2)``, ``φ = θ/(2 sin θ) · vee(R − Rᵀ)``
    - small θ: series ``φ ≈ ½(1 + θ²/6)·vee(R − Rᵀ)``
    - θ near π: ``u`` from the largest diagonal entry of ``½(R + I) = u uᵀ + O(π−θ)``
      with sign fixed by the off-diagonal skew part.
    """
    R = np.asarray(R, dtype=float)
    batch: tuple[int, ...] = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    n = int(Rf.shape[0])
    out = np.empty((n, 3))
    tr = np.trace(Rf, axis1=-2, axis2=-1)
    cos_t = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    w = vee(Rf - np.swapaxes(Rf, -1, -2))  # = 2 sinθ · u
    for i in range(n):
        t = theta[i]
        if t < SMALL_ANGLE:
            out[i] = 0.5 * (1.0 + t * t / 6.0) * w[i]
        elif np.pi - t < NEAR_PI:
            # R ≈ 2uuᵀ − I  ⇒  uuᵀ = (R + I)/2 ; take column with largest diagonal
            S = 0.5 * (Rf[i] + np.eye(3))
            k = int(np.argmax(np.diag(S)))
            u = S[:, k] / np.sqrt(max(S[k, k], 1e-300))
            # resolve the ±u ambiguity using the (possibly tiny) skew part
            if u @ w[i] < 0:
                u = -u
            out[i] = t * u
        else:
            out[i] = (t / (2.0 * np.sin(t))) * w[i]
    return out.reshape(batch + (3,))


def left_jacobian(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian ``Jl(φ) = I + B·[φ]× + C·[φ]ײ`` (Solà eq. (145)).

    ``B = (1 − cos θ)/θ²``, ``C = (θ − sin θ)/θ³``; series below
    :data:`SMALL_ANGLE`. Satisfies ``Exp(φ + δ) ≈ Exp(Jl δ)·Exp(φ)``.
    """
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi, axis=-1)
    small = theta < SMALL_ANGLE
    safe = np.where(small, 1.0, theta)
    B = np.where(small, 0.5 - theta**2 / 24.0, (1.0 - np.cos(safe)) / safe**2)
    C = np.where(small, 1.0 / 6.0 - theta**2 / 120.0, (safe - np.sin(safe)) / safe**3)
    K = hat(phi)
    eye3 = np.broadcast_to(np.eye(3), K.shape)
    return eye3 + B[..., None, None] * K + C[..., None, None] * (K @ K)


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian ``Jr(φ) = Jl(−φ) = Jl(φ)ᵀ`` (Solà eq. (143))."""
    return left_jacobian(-np.asarray(phi, dtype=float))


def left_jacobian_inverse(phi: np.ndarray) -> np.ndarray:
    """Closed-form ``Jl⁻¹(φ) = I − ½[φ]× + D·[φ]ײ``.

    ``D = (1/θ²)(1 − (θ/2)·cot(θ/2))`` → ``1/12 + θ²/720 + …`` for small θ.
    Diverges at θ = 2π (outside the principal domain θ ≤ π returned by
    :func:`log`).
    """
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi, axis=-1)
    small = theta < SMALL_ANGLE
    safe = np.where(small, 1.0, theta)
    half = 0.5 * safe
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(
            small,
            1.0 / 12.0 + theta**2 / 720.0,
            (1.0 - half * np.cos(half) / np.sin(half)) / safe**2,
        )
    K = hat(phi)
    eye3 = np.broadcast_to(np.eye(3), K.shape)
    return eye3 - 0.5 * K + D[..., None, None] * (K @ K)


def right_jacobian_inverse(phi: np.ndarray) -> np.ndarray:
    """``Jr⁻¹(φ) = Jl⁻¹(−φ)``."""
    return left_jacobian_inverse(-np.asarray(phi, dtype=float))


def boxplus(R: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Right/local retraction ``R ⊞ δ = R · Exp(δ)``."""
    return np.asarray(R, dtype=float) @ exp(delta)


def boxminus(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Local difference ``R1 ⊟ R2 = Log(R2ᵀ R1)`` so that ``R2 ⊞ (R1 ⊟ R2) = R1``."""
    R2 = np.asarray(R2, dtype=float)
    return log(np.swapaxes(R2, -1, -2) @ np.asarray(R1, dtype=float))


def geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Riemannian (geodesic) distance ``‖Log(R1ᵀ R2)‖ ∈ [0, π]``."""
    return np.linalg.norm(boxminus(R2, R1), axis=-1)


def is_rotation(R: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Check ``‖RᵀR − I‖∞ ≤ tol`` and ``det R > 0`` elementwise over the batch."""
    R = np.asarray(R, dtype=float)
    E = np.swapaxes(R, -1, -2) @ R - np.eye(3)
    ortho = np.max(np.abs(E), axis=(-2, -1)) <= tol
    return ortho & (np.linalg.det(R) > 0)


def project(M: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3) (orthogonal Procrustes).

    ``R = U·diag(1, 1, det(UVᵀ))·Vᵀ`` from the SVD ``M = UΣVᵀ`` — the closest
    rotation in Frobenius norm. Never applied silently by other qnav functions.
    """
    M = np.asarray(M, dtype=float)
    U, _, Vt = np.linalg.svd(M)
    S = np.ones(M.shape[:-2] + (3,))
    S[..., 2] = np.where(np.linalg.det(U @ Vt) < 0, -1.0, 1.0)
    return (U * S[..., None, :]) @ Vt
