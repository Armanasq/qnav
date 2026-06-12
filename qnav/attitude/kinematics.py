"""Quaternion kinematics and angular-velocity integration.

State: ``q_WB`` (world-from-body attitude). Angular velocity ``ω`` is the
**body-frame** angular rate of the body w.r.t. the world, in rad/s, unless a
function explicitly says otherwise.

Continuous-time kinematics (Hamilton, Solà eq. (107)):

    q̇_WB = ½ · q_WB ⊗ [0, ω_B]

Integrators (all return a renormalized quaternion; normalization is part of
each method's documented contract):

- :func:`integrate_first_order` — Euler step on q̇ (cheapest, O(dt²) error)
- :func:`integrate_exponential` — exact for constant ω over the step (zeroth-
  order integrator, Solà eq. (225))
- :func:`integrate_midpoint` — exponential step with averaged rate (first-order
  integrator, Solà eq. (228))
- :func:`integrate_rk4` — classical RK4 on the quaternion ODE

References: Solà §"Time derivatives" and §"Quaternion integration"
(``Quaternion.tex``, ``RungeKutta.tex``). See ``docs/math/quaternions.md``.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat

__all__ = [
    "qdot", "omega_matrix", "integrate_first_order", "integrate_exponential",
    "integrate_midpoint", "integrate_rk4", "integrate",
    "angular_velocity_from_quaternions",
]


def omega_matrix(omega: np.ndarray) -> np.ndarray:
    """4×4 matrix ``Ω(ω)`` with ``q̇ = ½ Ω(ω) q`` for **body-frame** ω.

    ``Ω(ω) = [[0, −ωᵀ], [ω, −[ω]×]]`` (Solà eq. (108), right-multiplication
    matrix of the pure quaternion ω).
    """
    omega = np.asarray(omega, dtype=float)
    x, y, z = omega[..., 0], omega[..., 1], omega[..., 2]
    W = np.zeros(omega.shape[:-1] + (4, 4))
    W[..., 0, 1], W[..., 0, 2], W[..., 0, 3] = -x, -y, -z
    W[..., 1, 0], W[..., 1, 2], W[..., 1, 3] = x, z, -y
    W[..., 2, 0], W[..., 2, 1], W[..., 2, 3] = y, -z, x
    W[..., 3, 0], W[..., 3, 1], W[..., 3, 2] = z, y, -x
    return W


def qdot(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Time derivative ``q̇ = ½ q ⊗ [0, ω_B]`` (body-frame rate)."""
    q = np.asarray(q, dtype=float)
    w = np.asarray(omega_body, dtype=float)
    zero = np.zeros(w.shape[:-1] + (1,))
    return 0.5 * quat.mul(q, np.concatenate([zero, w], axis=-1))


def integrate_first_order(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    """Forward-Euler step ``q ← normalize(q + q̇·dt)``.

    Local error O(dt²); use only for very small ω·dt. Normalizes the result.
    """
    return quat.normalize(np.asarray(q, dtype=float) + qdot(q, omega_body) * dt)


def integrate_exponential(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    """Exponential-map step ``q ← q ⊗ Exp(ω·dt)`` — exact for constant ω.

    Result is unit-norm by construction (renormalized once to absorb
    round-off).
    """
    dq = quat.exp(np.asarray(omega_body, dtype=float) * dt)
    return quat.normalize(quat.mul(q, dq))


def integrate_midpoint(
    q: np.ndarray, omega_start: np.ndarray, omega_end: np.ndarray, dt: float
) -> np.ndarray:
    """First-order integrator with rate averaging (Solà eq. (228), truncated).

    ``q ← q ⊗ [ Exp(ω̄·dt) + (dt²/24)·[0, ω₀×ω₁] ]`` with ``ω̄ = (ω₀+ω₁)/2``.
    The cross-product term captures the non-commutativity (coning) to first
    order. Result is renormalized.
    """
    w0 = np.asarray(omega_start, dtype=float)
    w1 = np.asarray(omega_end, dtype=float)
    wbar = 0.5 * (w0 + w1)
    dq = quat.exp(wbar * dt)
    coning = np.concatenate(
        [np.zeros(w0.shape[:-1] + (1,)), (dt * dt / 24.0) * np.cross(w0, w1)], axis=-1
    )
    return quat.normalize(quat.mul(q, dq + coning))


def integrate_rk4(
    q: np.ndarray, omega_start: np.ndarray, omega_end: np.ndarray, dt: float
) -> np.ndarray:
    """Classical RK4 on ``q̇ = ½ q ⊗ [0, ω(t)]`` with ω linearly interpolated.

    Renormalizes the result (RK4 does not preserve the unit norm exactly).

    .. note::
       With gyro *samples* (linear interpolation of ω between t₀ and t₁) the
       interpolation error limits the global order to 2 for time-varying
       rates — RK4's 4th order applies to the interpolated ODE, not the true
       one (cf. Solà, ``RungeKutta.tex``). Its error constant is still
       markedly smaller than :func:`integrate_midpoint`'s; for constant ω it
       is near-exact. Tests verify both the order and this ranking.
    """
    w0 = np.asarray(omega_start, dtype=float)
    w1 = np.asarray(omega_end, dtype=float)
    wm = 0.5 * (w0 + w1)
    q = np.asarray(q, dtype=float)
    k1 = qdot(q, w0)
    k2 = qdot(q + 0.5 * dt * k1, wm)
    k3 = qdot(q + 0.5 * dt * k2, wm)
    k4 = qdot(q + dt * k3, w1)
    return quat.normalize(q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def integrate(
    q: np.ndarray,
    omega_body: np.ndarray,
    dt: float,
    method: str = "exponential",
    omega_end: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch to an integrator by name.

    ``method``: ``"first_order"``, ``"exponential"``, ``"midpoint"``, ``"rk4"``.
    ``omega_end`` is required for ``midpoint``/``rk4`` (rate at step end).
    """
    if method == "first_order":
        return integrate_first_order(q, omega_body, dt)
    if method == "exponential":
        return integrate_exponential(q, omega_body, dt)
    if method in ("midpoint", "rk4"):
        if omega_end is None:
            raise ValueError(f"method {method!r} requires omega_end")
        f = integrate_midpoint if method == "midpoint" else integrate_rk4
        return f(q, omega_body, omega_end, dt)
    raise ValueError(f"unknown integration method {method!r}")


def angular_velocity_from_quaternions(
    q0: np.ndarray, q1: np.ndarray, dt: float
) -> np.ndarray:
    """Mean body-frame angular velocity taking ``q0`` to ``q1`` over ``dt``:
    ``ω = Log(q0* ⊗ q1) / dt`` (inverse of :func:`integrate_exponential`)."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    return quat.log(quat.relative(q0, q1)) / dt
