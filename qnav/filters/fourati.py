"""Fourati-style nonlinear attitude observer (Fourati et al. 2010).

A gyro-integration observer whose correction is one damped **Gauss-Newton
(Levenberg-Marquardt) step** on the combined gravity/magnetic alignment
residual. Unlike the Mahony cross-product correction (pure gradient
direction) or the Madgwick normalized-gradient step (fixed step length), the
LM step solves a local least-squares problem each sample — it scales the
correction by the observability of each error axis, giving faster transients
when one reference vector is nearly parallel to the error axis.

Per sample (state ``q_NB``):

1. predict ``q̄ ← normalize(q + ½ q ⊗ [0, ω] · dt)``
2. residual ``δ = y − ŷ`` with ``y = [−f̂; m̂]`` (measured gravity-down and
   field directions, body axes) and ``ŷ = [R(q̄)ᵀ ĝ_N; R(q̄)ᵀ m̂_N]``
3. ``ε = (XᵀX + λI₃)⁻¹ Xᵀ δ`` with the analytic alignment Jacobian
   ``X = 2 [ [ŷ_g]ₓ ; [ŷ_m]ₓ ] ∈ ℝ⁶ˣ³`` (right/body perturbation
   ``q ← q ⊗ [1, ε]`` gives ``∂ŷ/∂ε = 2[ŷ]ₓ``)
4. ``q ← normalize(q̄ ⊗ [1, k·dt·ε])`` — gain ``k`` [1/s] sets the
   correction bandwidth, exactly like Madgwick's β.

.. note::
   The widely-circulated transcription folds the correction into ``q̇`` as
   ``q̇ ⊗ [1, η]``, which makes the correction **vanish whenever ω ≈ 0**
   (a stationary platform never corrects). qnav applies the LM step to the
   state instead, preserving the published observer's intent at all rates;
   the deviation is intentional and tested.

Reference: Fourati, Manamanni, Afilal, Handrich, "A nonlinear filtering
approach for the attitude and dynamic body acceleration estimation based on
inertial and magnetic sensors", IEEE Sensors Journal 11(1), 2010.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.attitude import so3
from qnav.errors import ConventionError
from qnav.filters.base import AttitudeFilter

__all__ = ["FouratiFilter"]


class FouratiFilter(AttitudeFilter):
    """Nonlinear observer with a Levenberg-Marquardt alignment correction.

    Parameters
    ----------
    gain:
        Correction bandwidth k [1/s] (0 disables corrections). Values in
        1–10 1/s behave like Madgwick β of similar magnitude.
    m_ref:
        Magnetic field direction in the nav frame, shape (3,), any norm.
    lm_damping:
        LM damping λ on the 3×3 normal matrix (default 1e-5).
    """

    def __init__(
        self,
        gain: float = 5.0,
        m_ref: np.ndarray | None = None,
        lm_damping: float = 1e-5,
        q0=None,
        nav_frame: str = "NED",
    ) -> None:
        if nav_frame != "NED":
            raise ConventionError("FouratiFilter currently supports nav_frame='NED' only")
        super().__init__(q0=q0, nav_frame=nav_frame)
        if gain < 0:
            raise ValueError("gain must be nonnegative")
        self.gain = float(gain)
        self.lm_damping = float(lm_damping)
        m = np.array([1.0, 0.0, 0.0]) if m_ref is None else np.asarray(m_ref, dtype=float)
        n = np.linalg.norm(m)
        if n < 1e-12:
            raise ValueError("m_ref must be non-zero")
        self.m_ref = m / n
        self.g_ref = np.array([0.0, 0.0, 1.0])   # gravity-down, NED

    def _predict(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        """Gyro-only step (first-order, matching the observer's integrator)."""
        w = np.asarray(omega_body, dtype=float)
        qd = 0.5 * quat.mul(self.q, np.concatenate([[0.0], w]))
        self.q = quat.normalize(self.q + qd * dt)
        return self.q

    def step(
        self,
        omega_body: np.ndarray,
        dt: float,
        f_body: np.ndarray,
        m_body: np.ndarray,
    ) -> np.ndarray:
        """One observer step with gyro + accelerometer + magnetometer."""
        self.predict(omega_body, dt)
        if self.gain == 0:
            return self.q

        f = np.asarray(f_body, dtype=float)
        m = np.asarray(m_body, dtype=float)
        fn = np.linalg.norm(f)
        mn = np.linalg.norm(m)
        if fn < 1e-12 or mn < 1e-12:
            raise ValueError("zero-norm accelerometer or magnetometer sample")

        y = np.concatenate([-f / fn, m / mn])
        g_hat = quat.rotate_frame(self.q, self.g_ref)
        m_hat = quat.rotate_frame(self.q, self.m_ref)
        y_hat = np.concatenate([g_hat, m_hat])

        X = 2.0 * np.vstack([so3.hat(g_hat), so3.hat(m_hat)])    # ∂ŷ/∂ε, 6×3
        N = X.T @ X + self.lm_damping * np.eye(3)
        eps = np.linalg.solve(N, X.T @ (y - y_hat))
        step = min(self.gain * dt, 1.0) * eps      # clamp to a full GN step
        self.q = quat.normalize(quat.mul(self.q, np.concatenate([[1.0], step])))
        return self.q
