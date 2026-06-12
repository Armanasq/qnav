"""Rigid-body rotational dynamics (Euler's equations) and 6-DoF propagation.

Euler's rotational equations in the body frame (``__data/Satdyn_mb_2010f.pdf``
satellite-dynamics notes):

    J ω̇ + ω × (J ω) = τ

with inertia matrix J [kg·m²] and applied torque τ [N·m], both in body axes.
Propagation uses RK4 on the coupled (q, ω) system with the quaternion
kinematics ``q̇ = ½ q ⊗ [0, ω]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from qnav.attitude import kinematics as kin
from qnav.attitude import quaternion as quat

__all__ = ["RigidBody", "euler_rotational_dynamics"]


def euler_rotational_dynamics(
    omega_body: np.ndarray, J: np.ndarray, torque_body: np.ndarray
) -> np.ndarray:
    """``ω̇ = J⁻¹ (τ − ω × Jω)``."""
    w = np.asarray(omega_body, dtype=float)
    J = np.asarray(J, dtype=float)
    tau = np.asarray(torque_body, dtype=float)
    return np.linalg.solve(J, tau - np.cross(w, J @ w))


@dataclass
class RigidBody:
    """Free rigid body with attitude ``q_nav_body`` and body rate ω.

    ``torque_fn(t, q, ω) -> τ_body`` allows control/disturbance injection;
    default is torque-free motion.
    """

    inertia: np.ndarray
    q: np.ndarray = field(default_factory=quat.identity)
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque_fn: Optional[Callable] = None

    def __post_init__(self) -> None:
        self.inertia = np.asarray(self.inertia, dtype=float)
        if self.inertia.shape != (3, 3):
            raise ValueError("inertia must be a 3×3 matrix")
        self.q = quat.normalize(np.asarray(self.q, dtype=float))
        self.omega = np.asarray(self.omega, dtype=float).copy()

    def _torque(self, t: float, q: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.zeros(3) if self.torque_fn is None else np.asarray(
            self.torque_fn(t, q, w), dtype=float
        )

    def step(self, dt: float, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """RK4 step of the coupled (q, ω) dynamics; returns the new (q, ω)."""

        def f(t_, q_, w_):
            return kin.qdot(q_, w_), euler_rotational_dynamics(
                w_, self.inertia, self._torque(t_, q_, w_)
            )

        q0, w0 = self.q, self.omega
        k1q, k1w = f(t, q0, w0)
        k2q, k2w = f(t + dt / 2, q0 + dt / 2 * k1q, w0 + dt / 2 * k1w)
        k3q, k3w = f(t + dt / 2, q0 + dt / 2 * k2q, w0 + dt / 2 * k2w)
        k4q, k4w = f(t + dt, q0 + dt * k3q, w0 + dt * k3w)
        self.q = quat.normalize(q0 + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q))
        self.omega = w0 + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
        return self.q, self.omega

    def kinetic_energy(self) -> float:
        """Rotational kinetic energy ``½ ωᵀ J ω`` (conserved torque-free)."""
        return float(0.5 * self.omega @ self.inertia @ self.omega)

    def angular_momentum_nav(self) -> np.ndarray:
        """Angular momentum in the navigation frame, ``R(q) J ω`` (conserved
        torque-free) — used as a simulation invariant in tests."""
        return quat.rotate_vector(self.q, self.inertia @ self.omega)
