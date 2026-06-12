"""Gradient-descent attitude filter (Madgwick-style, original implementation).

State: ``q_nav_body``. The measurement objective for each reference
direction ``v_nav`` with unit body measurement ``v_body`` is

    e(q) = R(q)ᵀ v_nav − v_body          (predicted minus measured, body frame)
    F(q) = ½ Σᵢ wᵢ ‖eᵢ(q)‖²

The filter fuses the gyro propagation with a normalized gradient step:

    q̇ = ½ q ⊗ [0, ω] − β ∇F/‖∇F‖

The gradient is computed **analytically** from the quaternion Jacobian of the
rotation (``qnav.attitude.jacobians.drotate_dq`` applied to the conjugate),
not from hand-expanded per-sensor matrices — one code path for any number of
reference directions, verified against finite differences.

β has units rad/s and equals the expected gyro error magnitude it can absorb.

Reference: gradient-descent filter exposition in the attitude survey
(``__data/Efficient Attitude Estimators .../attitudesurvey.tex``).
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import jacobians as jac
from qnav.attitude import kinematics as kin
from qnav.attitude import quaternion as quat
from qnav.filters.base import AttitudeFilter

__all__ = ["MadgwickStyleFilter"]

#: d(conjugate)/dq — diagonal sign matrix.
_CONJ = np.diag([1.0, -1.0, -1.0, -1.0])


class MadgwickStyleFilter(AttitudeFilter):
    """Gradient-descent complementary filter.

    Parameters
    ----------
    beta:
        Gradient-step gain [rad/s] (≈ expected gyro error magnitude).
    q0, nav_frame:
        See :class:`~qnav.filters.base.AttitudeFilter`.
    """

    def __init__(self, beta: float = 0.05, q0=None, nav_frame: str = "NED") -> None:
        super().__init__(q0=q0, nav_frame=nav_frame)
        if beta < 0:
            raise ValueError("beta must be nonnegative")
        self.beta = float(beta)

    def objective_gradient(
        self, v_nav: np.ndarray, v_body: np.ndarray, weights: np.ndarray | None = None
    ) -> np.ndarray:
        """∇_q F(q) (4-vector) for unitized direction pairs ``(N, 3)``.

        ``R(q)ᵀ v_nav = rotate_vector(q*, v_nav)``; chain rule through the
        conjugate gives ``∂e/∂q = drotate_dq(q*, v_nav) · diag(1,−1,−1,−1)``.
        """
        vn = np.atleast_2d(np.asarray(v_nav, dtype=float))
        vb = np.atleast_2d(np.asarray(v_body, dtype=float))
        vn = vn / np.linalg.norm(vn, axis=-1, keepdims=True)
        vb = vb / np.linalg.norm(vb, axis=-1, keepdims=True)
        w = np.ones(vn.shape[0]) if weights is None else np.asarray(weights, dtype=float)
        qc = quat.conjugate(self.q)
        grad = np.zeros(4)
        for i in range(vn.shape[0]):
            e = quat.rotate_vector(qc, vn[i]) - vb[i]
            J = jac.drotate_dq(qc, vn[i]) @ _CONJ  # 3×4
            grad += w[i] * (J.T @ e)
        return grad

    def step(
        self, omega_meas: np.ndarray, dt: float,
        v_nav: np.ndarray | None = None, v_body: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """One fused step; with no directions this is pure gyro propagation."""
        qd = kin.qdot(self.q, np.asarray(omega_meas, dtype=float))
        if v_nav is not None and v_body is not None:
            g = self.objective_gradient(v_nav, v_body, weights)
            n = np.linalg.norm(g)
            if n > 1e-12:
                qd = qd - self.beta * g / n
        self.q = quat.normalize(self.q + qd * dt)
        return self.q

    def predict(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        return self.step(omega_body, dt)
