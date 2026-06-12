"""FKF: Fast Kalman Filter for attitude from MARG (Guo, Wu et al. 2017).

A linear Kalman filter whose **state is the quaternion itself** — possible
because both the process and measurement models are exactly linear in q:

- Process: first-order transition ``Φ = I₄ + ½ Ω(ω) dt`` with process noise
  mapped from gyro noise through ``Ξ(q) = ∂(½ q ⊗ [0, ω])/∂ω`` (4×3) —
  exactly the right-multiplication columns ``[q]_L[:, 1:]``.
- Measurement: a **full closed-form attitude observation** ``q_am`` from one
  accelerometer/magnetometer pair (qnav uses its SAAM solver), with the
  measurement covariance mapped from the 6D sensor noise through the
  numerically-evaluated Jacobian ``J = ∂q_am/∂[f, m]``.

Implementation note: the original derivation carries a page-long symbolic
expansion of the measurement quaternion and its Jacobian; qnav computes the
same quantities through the SAAM closed form plus central finite differences
(6 columns, ~1e-7 step). The covariance is identical to first order; the
code is auditable; the double-cover sign is resolved against the prediction
(``q_am ← −q_am`` if ``⟨q_am, q̄⟩ < 0``), which the symbolic path needs too.

Conventions: state ``q_NB``; ``f_body`` is specific force; noise parameters
are per-axis standard deviations of the *unit-normalized* sensor directions.

Reference: Guo, Wu, Qian et al., "Novel MARG-sensor orientation estimation
algorithm using fast Kalman filter", Journal of Sensors (2017).
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.determination.saam import saam
from qnav.errors import ConventionError
from qnav.filters.base import AttitudeFilter

__all__ = ["FastKalmanFilter"]


def _measurement_jacobian(f: np.ndarray, m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference ``∂q_saam/∂[f, m]`` (4×6), sign-stabilized."""
    q0 = saam(f, m)
    J = np.empty((4, 6))
    for i in range(6):
        df = np.zeros(3)
        dm = np.zeros(3)
        (df if i < 3 else dm)[i % 3] = eps
        qp = saam(f + df, m + dm)
        qm = saam(f - df, m - dm)
        # resolve double cover against the unperturbed solution
        if qp @ q0 < 0:
            qp = -qp
        if qm @ q0 < 0:
            qm = -qm
        J[:, i] = (qp - qm) / (2.0 * eps)
    return J


class FastKalmanFilter(AttitudeFilter):
    """Linear 4-state quaternion Kalman filter with algebraic measurements.

    Parameters
    ----------
    gyro_noise:
        σ_g [rad/s] per-axis gyro white noise (discrete, per sample).
    accel_noise, mag_noise:
        Per-axis direction-noise std of the normalized accel/mag readings.
    P0:
        Initial 4×4 covariance (default 0.01·I).
    """

    def __init__(
        self,
        gyro_noise: float = 0.01,
        accel_noise: float = 0.01,
        mag_noise: float = 0.01,
        P0: np.ndarray | None = None,
        q0=None,
        nav_frame: str = "NED",
    ) -> None:
        if nav_frame != "NED":
            raise ConventionError("FastKalmanFilter currently supports nav_frame='NED' only")
        super().__init__(q0=q0, nav_frame=nav_frame)
        self.sigma_g = float(gyro_noise)
        self.sigma_a = float(accel_noise)
        self.sigma_m = float(mag_noise)
        self.P = 0.01 * np.eye(4) if P0 is None else np.asarray(P0, dtype=float).copy()
        if self.P.shape != (4, 4):
            raise ValueError("P0 must be 4×4")

    def predict(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        """First-order transition + process noise via the rate Jacobian Ξ(q)."""
        w = np.asarray(omega_body, dtype=float)
        Phi = np.eye(4) + 0.5 * dt * self._omega4(w)
        # Ξ = ∂(½ q ⊗ [0, ω])/∂ω = ½ [q]_L[:, 1:]  (4×3)
        Xi = 0.5 * quat.left_matrix(self.q)[:, 1:]
        Qd = (dt**2) * (self.sigma_g**2) * (Xi @ Xi.T)
        self.q = quat.normalize(Phi @ self.q)
        self.P = Phi @ self.P @ Phi.T + Qd
        return self.q

    def update(self, f_body: np.ndarray, m_body: np.ndarray) -> np.ndarray:
        """Fuse one closed-form attitude observation (identity H, Joseph form)."""
        f = np.asarray(f_body, dtype=float)
        m = np.asarray(m_body, dtype=float)
        q_am = saam(f, m)
        if q_am @ self.q < 0:                     # double-cover continuity
            q_am = -q_am

        J = _measurement_jacobian(f, m)
        Sigma_am = np.diag([self.sigma_a**2] * 3 + [self.sigma_m**2] * 3)
        Rv = J @ Sigma_am @ J.T + 1e-12 * np.eye(4)   # PSD guard

        S = self.P + Rv                            # H = I₄
        K = self.P @ np.linalg.solve(S.T, np.eye(4)).T
        IK = np.eye(4) - K
        self.q = quat.normalize(self.q + K @ (q_am - self.q))
        self.P = IK @ self.P @ IK.T + K @ Rv @ K.T
        return self.q

    def step(
        self, omega_body: np.ndarray, dt: float,
        f_body: np.ndarray, m_body: np.ndarray,
    ) -> np.ndarray:
        """Predict + update in one call."""
        self.predict(omega_body, dt)
        return self.update(f_body, m_body)

    @staticmethod
    def _omega4(w: np.ndarray) -> np.ndarray:
        """``Ω(ω)`` with ``q̇ = ½ Ω q`` (Hamilton, body rate)."""
        x, y, z = w
        return np.array([
            [0.0, -x, -y, -z],
            [x, 0.0, z, -y],
            [y, -z, 0.0, x],
            [z, y, -x, 0.0],
        ])
