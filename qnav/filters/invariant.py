"""Left-invariant attitude EKF (attitude + gyro bias).

Differences from the reference :class:`~qnav.filters.Eskf` (right/local
error): the attitude error is defined in the **navigation frame** (left-
invariant), ``q_true = Exp(δθ_n) ⊗ q̂``. For gyro propagation the attitude
error dynamics become state-independent (``δθ̇_n = −R δb_g``, no ``[ω]×``
term), which is the invariant-EKF property that makes the linearization
valid far from the estimate — the practical benefit is convergence from
large initial attitude errors.

The gyro-bias state breaks exact invariance (this is the standard
"imperfect IEKF"); the bias coupling uses the estimated attitude.

Error state: ``δx = [δθ_n, δb_g] ∈ ℝ⁶``; covariance over that ordering.
Updates share the gated Joseph-form kernel with every other qnav ESKF —
gating, robust losses, quarantine, and UpdateResult reporting behave
identically, so estimator comparisons isolate the error definition.

Reference: Barrau & Bonnabel, "The Invariant Extended Kalman Filter as a
Stable Observer", IEEE TAC 2017.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from qnav._validate import ensure_covariance, ensure_nonnegative, ensure_positive, ensure_vector3
from qnav.attitude import dcm as dcm_mod
from qnav.attitude import quaternion as quat
from qnav.attitude import so3
from qnav.filters._kalman import gated_joseph_update
from qnav.filters.base import AttitudeFilter
from qnav.filters.robust import GatePolicy, SensorMonitor
from qnav.types import ArrayLike

__all__ = ["LeftInvariantEskf"]


class LeftInvariantEskf(AttitudeFilter):
    """Attitude + gyro-bias filter with left-invariant (nav-frame) error.

    Constructor arguments mirror :class:`~qnav.filters.Eskf` exactly so the
    two can be compared under identical configurations.
    """

    def __init__(
        self,
        gyro_noise_density: float,
        gyro_bias_walk: float = 0.0,
        P0: np.ndarray | None = None,
        q0=None,
        b0: np.ndarray | None = None,
        nav_frame: str = "NED",
        gate: Optional[GatePolicy] = None,
    ) -> None:
        super().__init__(q0=q0, nav_frame=nav_frame)
        self.bias = np.zeros(3) if b0 is None else ensure_vector3(b0, "b0").copy()
        self.sigma_g = ensure_nonnegative(gyro_noise_density, "gyro_noise_density")
        self.sigma_bw = ensure_nonnegative(gyro_bias_walk, "gyro_bias_walk")
        if P0 is None:
            P0 = np.diag([0.1**2] * 3 + [0.01**2] * 3)
        self.P = ensure_covariance(P0, 6, "P0").copy()
        self.gate = gate
        self.monitors: Dict[str, SensorMonitor] = {}

    def _predict(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        w_hat = omega_body - self.bias
        R = dcm_mod.from_quaternion(self.q)
        self.q = quat.normalize(quat.mul(self.q, quat.exp(w_hat * dt)))

        # left-invariant error dynamics: dtheta_n' = dtheta_n - R dbg dt
        F = np.eye(6)
        F[:3, 3:] = -R * dt

        Qd = np.zeros((6, 6))
        # gyro white noise enters through the (rotated) body rate; the
        # nav-frame covariance increment is isotropic: R (s^2 dt I) R^T = s^2 dt I
        Qd[:3, :3] = (self.sigma_g**2 * dt) * np.eye(3)
        Qd[3:, 3:] = (self.sigma_bw**2 * dt) * np.eye(3)

        self.P = F @ self.P @ F.T + Qd
        return self.q

    def update_direction(
        self, v_nav: ArrayLike, v_body_meas: ArrayLike, sigma: float,
        *, timestamp: float | None = None, sensor_id: str = "direction",
    ) -> np.ndarray:
        """Fuse a unit-direction pair — same call contract as ``Eskf``.

        The innovation is formed in the navigation frame (left-invariant
        form): ``z = R v_body_meas − v_nav`` with ``H = [[v_nav]×, 0]`` —
        the Jacobian does not depend on the attitude estimate.
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

        # nav-frame innovation: R̂ v_b = Exp(−δθ) v_nav ⇒ z ≈ +[v_nav]× δθ
        innov = quat.rotate_vector(self.q, vb) - vn
        H = np.zeros((3, 6))
        H[:, :3] = so3.hat(vn)                        # state-independent
        gated_joseph_update(self, H, (sigma**2) * np.eye(3), innov,
                            inject=self._inject, sensor_id=sensor_id,
                            timestamp=timestamp)
        return innov

    def update_gravity(self, f_body: ArrayLike, sigma: float, *,
                       timestamp: float | None = None,
                       sensor_id: str = "accel") -> np.ndarray:
        up = np.array([0.0, 0.0, -1.0]) if self.nav_frame == "NED" else np.array([0.0, 0.0, 1.0])
        return self.update_direction(up, f_body, sigma,
                                     timestamp=timestamp, sensor_id=sensor_id)

    def update_magnetometer(self, m_nav: ArrayLike, m_body: ArrayLike, sigma: float, *,
                            timestamp: float | None = None,
                            sensor_id: str = "mag") -> np.ndarray:
        return self.update_direction(m_nav, m_body, sigma,
                                     timestamp=timestamp, sensor_id=sensor_id)

    def set_monitor(self, sensor_id: str, monitor: SensorMonitor) -> None:
        self.monitors[sensor_id] = monitor

    def _inject(self, dx: np.ndarray) -> None:
        dtheta_n, dbias = dx[:3], dx[3:]
        # left error: q <- Exp(dtheta_n) (x) q
        self.q = quat.normalize(quat.mul(quat.exp(dtheta_n), self.q))
        self.bias = self.bias + dbias
        G = np.eye(6)
        G[:3, :3] = np.eye(3) + 0.5 * so3.hat(dtheta_n)
        self.P = G @ self.P @ G.T

    @property
    def attitude_std(self) -> np.ndarray:
        """Per-axis attitude error std [rad] (nav-frame/left tangent)."""
        return np.sqrt(np.diag(self.P)[:3])
