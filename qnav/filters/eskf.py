"""Error-state Kalman filter (ESKF) for attitude + gyro bias.

Formulation: Solà, "Quaternion kinematics for the error-state Kalman filter"
(``__data/Quaternion kinematics .../ErrorState.tex``), **local** error
definition.

Nominal state: ``q`` (= q_nav_body, Hamilton scalar-first), gyro bias ``b``
[rad/s, body]. Error state ``δx = [δθ, δb] ∈ ℝ⁶`` with the **right/local**
attitude error ``q_true = q̂ ⊗ Exp(δθ)``; ``P`` is its 6×6 covariance.

Process model (gyro input ``ω_meas = ω + b + n_g``, ``ḃ = n_b``):

    q̂ ← q̂ ⊗ Exp((ω_meas − b̂) dt)
    F  = [[Exp(ω̂ dt)ᵀ, −Jr(ω̂ dt)·dt], [0, I]]
    Qd = diag(σ_g²·dt·I, σ_bw²·dt·I)   (mapped through G = diag(Jr dt, I dt)/dt …
         qnav uses the standard first-order discrete equivalents shown)

Measurement model (known direction ``v_nav``, unit body measurement):

    v_body = Exp(−δθ)·R̂ᵀ v_nav + n   ⇒   H = [ [v̂_body]× , 0 ]

Update is Joseph-form; injection ``q̂ ← q̂ ⊗ Exp(δθ)``, ``b̂ += δb``; the
post-injection covariance reset uses ``G_θ = I − ½[δθ]×`` (Solà eq. (287)).

Consistency of P is validated by Monte-Carlo NEES in
``tests/test_filters.py``.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.attitude import so3
from qnav.filters.base import AttitudeFilter

__all__ = ["Eskf"]


class Eskf(AttitudeFilter):
    """Attitude + gyro-bias error-state Kalman filter.

    Parameters
    ----------
    gyro_noise_density:
        σ_g [rad/s/√Hz] — gyro white-noise density.
    gyro_bias_walk:
        σ_bw [rad/s²/√Hz] — bias random-walk density.
    P0:
        Initial 6×6 error covariance (default: diag(0.1² rad², 0.01² (rad/s)²)).
    q0, b0, nav_frame:
        Initial nominal state.
    """

    def __init__(
        self,
        gyro_noise_density: float,
        gyro_bias_walk: float = 0.0,
        P0: np.ndarray | None = None,
        q0=None,
        b0: np.ndarray | None = None,
        nav_frame: str = "NED",
    ) -> None:
        super().__init__(q0=q0, nav_frame=nav_frame)
        self.bias = np.zeros(3) if b0 is None else np.asarray(b0, dtype=float).copy()
        self.sigma_g = float(gyro_noise_density)
        self.sigma_bw = float(gyro_bias_walk)
        if P0 is None:
            P0 = np.diag([0.1**2] * 3 + [0.01**2] * 3)
        self.P = np.asarray(P0, dtype=float).copy()
        if self.P.shape != (6, 6):
            raise ValueError("P0 must be 6×6 over [δθ, δb]")

    # -- prediction --------------------------------------------------------
    def predict(self, omega_meas: np.ndarray, dt: float) -> np.ndarray:
        """Propagate nominal state and error covariance with one gyro sample."""
        w_hat = np.asarray(omega_meas, dtype=float) - self.bias
        phi = w_hat * dt
        self.q = quat.normalize(quat.mul(self.q, quat.exp(phi)))

        F = np.zeros((6, 6))
        F[:3, :3] = so3.exp(phi).T
        F[:3, 3:] = -so3.right_jacobian(phi) * dt
        F[3:, 3:] = np.eye(3)

        Qd = np.zeros((6, 6))
        Qd[:3, :3] = (self.sigma_g**2 * dt) * np.eye(3)
        Qd[3:, 3:] = (self.sigma_bw**2 * dt) * np.eye(3)

        self.P = F @ self.P @ F.T + Qd
        return self.q

    # -- updates -----------------------------------------------------------
    def update_direction(
        self, v_nav: np.ndarray, v_body_meas: np.ndarray, sigma: float
    ) -> np.ndarray:
        """Fuse a unit-direction measurement (e.g. gravity or magnetic field).

        ``v_nav``: known direction in the navigation frame (any norm —
        unitized); ``v_body_meas``: measured direction in the body frame
        (unitized); ``sigma``: per-axis noise std of the unit measurement.

        Returns the innovation (3-vector) for monitoring.
        """
        vn = np.asarray(v_nav, dtype=float)
        vn = vn / np.linalg.norm(vn)
        vb = np.asarray(v_body_meas, dtype=float)
        nb = np.linalg.norm(vb)
        if nb < 1e-12:
            raise ValueError("zero-norm body measurement")
        vb = vb / nb

        v_hat = quat.rotate_frame(self.q, vn)  # R̂ᵀ v_nav
        H = np.zeros((3, 6))
        H[:, :3] = so3.hat(v_hat)
        R = (sigma**2) * np.eye(3)

        innov = vb - v_hat
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))
        dx = K @ innov

        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        self._inject(dx)
        return innov

    def update_gravity(self, f_body: np.ndarray, sigma: float) -> np.ndarray:
        """Accelerometer update assuming low dynamics: the measured specific
        force direction equals the body-frame "up" direction
        (``−g_nav`` normalized: ``[0,0,−1]`` NED, ``[0,0,+1]`` ENU)."""
        up = np.array([0.0, 0.0, -1.0]) if self.nav_frame == "NED" else np.array([0.0, 0.0, 1.0])
        return self.update_direction(up, f_body, sigma)

    def update_magnetometer(self, m_nav: np.ndarray, m_body: np.ndarray, sigma: float) -> np.ndarray:
        """Magnetometer update against the known local field direction
        ``m_nav`` (gate disturbances upstream — see ``qnav.heading.disturbance``)."""
        return self.update_direction(m_nav, m_body, sigma)

    # -- internals ----------------------------------------------------------
    def _inject(self, dx: np.ndarray) -> None:
        dtheta, dbias = dx[:3], dx[3:]
        self.q = quat.normalize(quat.mul(self.q, quat.exp(dtheta)))
        self.bias = self.bias + dbias
        G = np.eye(6)
        G[:3, :3] = np.eye(3) - 0.5 * so3.hat(dtheta)
        self.P = G @ self.P @ G.T

    @property
    def attitude_std(self) -> np.ndarray:
        """Per-axis attitude error std [rad] (sqrt of P diagonal, local frame)."""
        return np.sqrt(np.diag(self.P)[:3])
