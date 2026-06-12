"""Analytic Jacobians for attitude operations.

All attitude perturbations are **right/local** tangent perturbations
(``R ⊞ δθ = R·Exp(δθ)``, ``q ⊞ δθ = q ⊗ Exp_q(δθ)``) unless stated otherwise,
matching ``docs/conventions.md`` §5. Every Jacobian here is verified against
central finite differences in ``tests/test_jacobians.py``.

References: Solà §"Perturbations, derivatives and integrals" (Jacobians of
rotation w.r.t. quaternion/vector), ``ErrorState.tex``.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import dcm as _dcm
from qnav.attitude import so3

__all__ = [
    "drotate_dtheta", "drotate_dv", "drotate_dq",
    "dexp_dphi", "dlog_dR_local", "dcomposition_left", "dcomposition_right",
    "dinverse_dtheta",
]


def drotate_dv(q: np.ndarray) -> np.ndarray:
    """∂(R(q)·v)/∂v = R(q). Shape ``(..., 3, 3)``."""
    return _dcm.from_quaternion(q)


def drotate_dtheta(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """∂(R(q ⊞ δθ)·v)/∂δθ at δθ = 0, i.e. ``−R(q)·[v]×`` (Solà eq. (174a)
    adapted to the right perturbation)."""
    R = _dcm.from_quaternion(q)
    return -R @ so3.hat(v)


def drotate_dq(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """∂(R(q)·v)/∂q — raw 3×4 Jacobian w.r.t. the quaternion components.

    Solà eq. (174b): with ``q = [w, u]``,
    ``∂(q⊗v⊗q*)/∂q = 2·[ w·v + u×v | uᵀv·I + u·vᵀ − v·uᵀ − w·[v]× ]``.
    This is the Jacobian of the *sandwich product* without the unit-norm
    constraint (off-manifold variations are included). Useful for
    full-quaternion EKFs; for minimal-state filters prefer
    :func:`drotate_dtheta`.
    """
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    w = q[..., 0]
    u = q[..., 1:]
    col0 = w[..., None] * v + np.cross(u, v)
    uv = np.sum(u * v, axis=-1)
    I = np.broadcast_to(np.eye(3), col0.shape[:-1] + (3, 3))
    block = (
        uv[..., None, None] * I
        + u[..., :, None] * v[..., None, :]
        - v[..., :, None] * u[..., None, :]
        - w[..., None, None] * so3.hat(v)
    )
    return 2.0 * np.concatenate([col0[..., :, None], block], axis=-1)


def dexp_dphi(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian of the exponential map: ``Exp(φ + δ) ≈ Exp(φ)·Exp(Jr δ)``."""
    return so3.right_jacobian(phi)


def dlog_dR_local(R: np.ndarray) -> np.ndarray:
    """Jacobian of ``Log`` w.r.t. a right perturbation of R:
    ``Log(R·Exp(δ)) ≈ Log(R) + Jr⁻¹(Log R)·δ``."""
    return so3.right_jacobian_inverse(so3.log(R))


def dcomposition_left(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """∂Log((Ra ⊞ δ)·Rb ⊟ (Ra·Rb))/∂δ = ``Rbᵀ`` (adjoint of the right factor)."""
    R_b = np.asarray(R_b, dtype=float)
    return np.swapaxes(R_b, -1, -2)


def dcomposition_right(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """∂Log(Ra·(Rb ⊞ δ) ⊟ (Ra·Rb))/∂δ = I."""
    R_a = np.asarray(R_a, dtype=float)
    return np.broadcast_to(np.eye(3), R_a.shape).copy()


def dinverse_dtheta(R: np.ndarray) -> np.ndarray:
    """∂((R ⊞ δ)⁻¹ ⊟ R⁻¹)/∂δ = ``−R`` (right-perturbation Jacobian of inversion)."""
    return -np.asarray(R, dtype=float)
