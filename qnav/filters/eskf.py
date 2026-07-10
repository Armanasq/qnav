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

from qnav._validate import ensure_covariance, ensure_nonnegative, ensure_positive, ensure_vector3
from qnav.attitude import quaternion as quat
from qnav.attitude import so3
from qnav.filters.base import AttitudeFilter
from qnav.filters.contracts import UpdateResult
from qnav.types import ArrayLike

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
        self.bias = np.zeros(3) if b0 is None else ensure_vector3(b0, "b0").copy()
        self.sigma_g = ensure_nonnegative(gyro_noise_density, "gyro_noise_density")
        self.sigma_bw = ensure_nonnegative(gyro_bias_walk, "gyro_bias_walk")
        if P0 is None:
            P0 = np.diag([0.1**2] * 3 + [0.01**2] * 3)
        self.P = ensure_covariance(P0, 6, "P0").copy()

    # -- prediction --------------------------------------------------------
    def _predict(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        """Propagate nominal state and error covariance with one gyro sample."""
        w_hat = omega_body - self.bias
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
        self, v_nav: ArrayLike, v_body_meas: ArrayLike, sigma: float,
        *, timestamp: float | None = None, sensor_id: str = "direction",
    ) -> np.ndarray:
        """Fuse a unit-direction measurement (e.g. gravity or magnetic field).

        ``v_nav``: known direction in the navigation frame (any norm —
        unitized); ``v_body_meas``: measured direction in the body frame
        (unitized); ``sigma``: per-axis noise std of the unit measurement.

        Returns the innovation (3-vector) for backward compatibility; the
        full :class:`~qnav.filters.contracts.UpdateResult` (NIS, innovation
        covariance, state correction) is stored in ``self.last_update`` and
        aggregated per ``sensor_id`` in ``self.innovation_stats``.
        """
        sigma = ensure_positive(sigma, "sigma")
        vn = ensure_vector3(v_nav, "v_nav")
        nn = np.linalg.norm(vn)
        if nn < 1e-12:
            raise ValueError("v_nav must have non-zero norm")
        vn = vn / nn
        vb = ensure_vector3(v_body_meas, "v_body_meas")
        nb = np.linalg.norm(vb)
        if nb < 1e-12:
            raise ValueError("v_body_meas must have non-zero norm")
        vb = vb / nb

        v_hat = quat.rotate_frame(self.q, vn)  # R̂ᵀ v_nav
        H = np.zeros((3, 6))
        H[:, :3] = so3.hat(v_hat)
        R = (sigma**2) * np.eye(3)

        innov = vb - v_hat
        S = H @ self.P @ H.T + R
        S_inv_innov = np.linalg.solve(S, innov)
        nis = float(innov @ S_inv_innov)
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))
        dx = K @ innov

        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        self._inject(dx)
        self._record_update(UpdateResult(
            accepted=True, innovation=innov, innovation_covariance=S, nis=nis,
            state_correction=dx, timestamp=timestamp, sensor_id=sensor_id,
        ))
        return innov

    def update_gravity(
        self, f_body: ArrayLike, sigma: float,
        *, timestamp: float | None = None, sensor_id: str = "accel",
    ) -> np.ndarray:
        """Accelerometer update assuming low dynamics: the measured specific
        force direction equals the body-frame "up" direction
        (``−g_nav`` normalized: ``[0,0,−1]`` NED, ``[0,0,+1]`` ENU)."""
        up = np.array([0.0, 0.0, -1.0]) if self.nav_frame == "NED" else np.array([0.0, 0.0, 1.0])
        return self.update_direction(up, f_body, sigma, timestamp=timestamp, sensor_id=sensor_id)

    def update_magnetometer(
        self, m_nav: ArrayLike, m_body: ArrayLike, sigma: float,
        *, timestamp: float | None = None, sensor_id: str = "mag",
    ) -> np.ndarray:
        """Magnetometer update against the known local field direction
        ``m_nav`` (gate disturbances upstream — see ``qnav.heading.disturbance``)."""
        return self.update_direction(m_nav, m_body, sigma, timestamp=timestamp, sensor_id=sensor_id)

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
