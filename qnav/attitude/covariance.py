"""Attitude covariance: semantics, propagation, and frame changes.

An attitude covariance ``P`` in qnav is a 3×3 matrix over the **right/local**
tangent error ``δθ`` defined by ``q_true = q̂ ⊗ Exp(δθ)`` (body-side error),
unless a function explicitly says "left/global". This matches the local
error-state formulation in Solà (``ErrorState.tex``).

References: Solà §"The error-state Kalman filter"; Kok–Hol–Schön tutorial.
See ``docs/math/filtering.md``.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import dcm as _dcm
from qnav.attitude import so3

__all__ = [
    "propagate_gyro", "local_to_global", "global_to_local",
    "transform_vector_covariance", "compose_covariance", "is_psd",
    "sample_attitudes",
]


def propagate_gyro(
    P: np.ndarray, omega_body: np.ndarray, dt: float, gyro_psd: np.ndarray | float
) -> np.ndarray:
    """One gyro-propagation step of a local attitude covariance.

    Discrete error dynamics (Solà, local error state):
    ``δθ_{k+1} = Exp(−ω dt)·δθ_k + Jr(ω dt)·dt·n_g``, so

    ``P ← F P Fᵀ + G Q Gᵀ`` with ``F = Exp(ω dt)ᵀ``, ``G = Jr(ω dt)·dt`` and
    ``Q = gyro_psd / dt`` the white-noise covariance of the rate noise
    (``gyro_psd`` in (rad/s)²/Hz, scalar or 3×3).
    """
    P = np.asarray(P, dtype=float)
    phi = np.asarray(omega_body, dtype=float) * dt
    F = np.swapaxes(so3.exp(phi), -1, -2)
    G = so3.right_jacobian(phi) * dt
    Q = np.asarray(gyro_psd, dtype=float)
    if Q.ndim == 0:
        Q = Q * np.eye(3)
    Qd = Q / dt
    return F @ P @ np.swapaxes(F, -1, -2) + G @ Qd @ np.swapaxes(G, -1, -2)


def local_to_global(P_local: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Convert a right/local covariance to the left/global error definition
    (``q_true = Exp(δθ_g) ⊗ q̂``): ``P_g = R P_l Rᵀ`` with ``R = R(q̂)``."""
    R = _dcm.from_quaternion(q)
    return R @ np.asarray(P_local, dtype=float) @ np.swapaxes(R, -1, -2)


def global_to_local(P_global: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Inverse of :func:`local_to_global`: ``P_l = Rᵀ P_g R``."""
    R = _dcm.from_quaternion(q)
    return np.swapaxes(R, -1, -2) @ np.asarray(P_global, dtype=float) @ R


def transform_vector_covariance(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Covariance of ``R v`` given ``cov(v) = P`` (deterministic R): ``R P Rᵀ``."""
    R = np.asarray(R, dtype=float)
    return R @ np.asarray(P, dtype=float) @ np.swapaxes(R, -1, -2)


def compose_covariance(
    P_ab: np.ndarray, P_bc: np.ndarray, R_bc: np.ndarray
) -> np.ndarray:
    """Local covariance of the composition ``R_ac = R_ab R_bc`` from independent
    local covariances of the factors.

    First order: ``δθ_ac = R_bcᵀ δθ_ab + δθ_bc`` ⇒
    ``P_ac = R_bcᵀ P_ab R_bc + P_bc``.
    """
    R_bc = np.asarray(R_bc, dtype=float)
    Rt = np.swapaxes(R_bc, -1, -2)
    return Rt @ np.asarray(P_ab, dtype=float) @ R_bc + np.asarray(P_bc, dtype=float)


def is_psd(P: np.ndarray, tol: float = 1e-10) -> bool:
    """Check symmetry and positive semidefiniteness (eigenvalues ≥ −tol)."""
    P = np.asarray(P, dtype=float)
    if not np.allclose(P, np.swapaxes(P, -1, -2), atol=1e-9):
        return False
    return bool(np.all(np.linalg.eigvalsh(P) >= -tol))


def sample_attitudes(
    q_mean: np.ndarray, P: np.ndarray, n: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Draw ``n`` attitude samples ``q̂ ⊗ Exp(δθ)``, ``δθ ~ N(0, P)`` (local).

    Useful for Monte-Carlo validation of covariance propagation.
    """
    from qnav.attitude import quaternion as quat

    rng = rng or np.random.default_rng()
    L = np.linalg.cholesky(np.asarray(P, dtype=float) + 1e-15 * np.eye(3))
    deltas = rng.standard_normal((n, 3)) @ L.T
    return quat.mul(np.asarray(q_mean, dtype=float), quat.exp(deltas))
