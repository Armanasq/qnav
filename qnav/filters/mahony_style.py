"""Nonlinear complementary filter on SO(3) with gyro-bias estimation
(Mahony–Hamel–Pflimlin 2008 structure, original implementation).

State: ``q_nav_body`` and gyro bias ``b`` (rad/s, body frame).

For each reference direction ``vᵢ_nav`` with body measurement ``vᵢ_body``:

    v̂ᵢ_body = R(q)ᵀ vᵢ_nav                      (predicted body direction)
    ω_mes   = Σᵢ kᵢ (vᵢ_body × v̂ᵢ_body)          (correction, body frame)

Filter dynamics (discretized with the exponential integrator):

    q̇ = ½ q ⊗ [0, ω_meas − b + k_P ω_mes]
    ḃ = −k_I ω_mes

``ω_mes`` is zero exactly when measured and predicted directions agree; the
cross-product form makes the correction torque-like and globally smooth.

Reference: Mahony filter exposition in the attitude survey
(``__data/Efficient Attitude Estimators .../attitudesurvey.tex``) and
``__data/IROS2001.pdf`` (complementary filtering on SO(3)).
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import kinematics as kin
from qnav.attitude import quaternion as quat
from qnav.filters.base import AttitudeFilter

__all__ = ["MahonyFilter", "NonlinearComplementaryFilter"]


class MahonyFilter(AttitudeFilter):
    """Nonlinear complementary filter with bias estimation.

    Parameters
    ----------
    kp, ki:
        Proportional and integral (bias) gains [1/s]. ``ki = 0`` disables
        bias estimation.
    q0, nav_frame:
        See :class:`~qnav.filters.base.AttitudeFilter`.

    Use :meth:`step` with reference/measured direction pairs each sample.
    """

    def __init__(
        self, kp: float = 1.0, ki: float = 0.1, q0=None, nav_frame: str = "NED"
    ) -> None:
        super().__init__(q0=q0, nav_frame=nav_frame)
        if kp < 0 or ki < 0:
            raise ValueError("gains must be nonnegative")
        self.kp = float(kp)
        self.ki = float(ki)
        self.bias = np.zeros(3)

    def correction(
        self, v_nav: np.ndarray, v_body: np.ndarray, weights: np.ndarray | None = None
    ) -> np.ndarray:
        """``ω_mes = Σ kᵢ (vᵢ_body × v̂ᵢ_body)`` for unitized direction pairs.

        ``v_nav``/``v_body``: ``(N, 3)`` (or single vectors).
        """
        vn = np.atleast_2d(np.asarray(v_nav, dtype=float))
        vb = np.atleast_2d(np.asarray(v_body, dtype=float))
        vn = vn / np.linalg.norm(vn, axis=-1, keepdims=True)
        nb = np.linalg.norm(vb, axis=-1, keepdims=True)
        ok = (nb > 1e-12)[..., 0]
        vb = np.where(nb > 1e-12, vb / np.where(nb > 1e-12, nb, 1.0), vb)
        w = np.ones(vn.shape[0]) if weights is None else np.asarray(weights, dtype=float)
        vhat = quat.rotate_frame(self.q, vn)  # R(q)ᵀ v_nav
        omega_mes = np.sum(
            (w[:, None] * np.cross(vb, vhat))[ok], axis=0
        )
        return omega_mes

    def step(
        self, omega_meas: np.ndarray, dt: float,
        v_nav: np.ndarray | None = None, v_body: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """One filter step: correction (if directions given) + propagation."""
        omega_mes = np.zeros(3)
        if v_nav is not None and v_body is not None:
            omega_mes = self.correction(v_nav, v_body, weights)
            self.bias = self.bias - self.ki * omega_mes * dt
        omega = np.asarray(omega_meas, dtype=float) - self.bias + self.kp * omega_mes
        self.q = kin.integrate_exponential(self.q, omega, dt)
        return self.q

    def predict(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        """Gyro-only step (bias-corrected, no direction update)."""
        return self.step(omega_body, dt)


#: The Mahony filter *is* the nonlinear complementary filter; explicit alias.
NonlinearComplementaryFilter = MahonyFilter
