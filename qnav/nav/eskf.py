"""15-state navigation error-state Kalman filter.

Error state (ordering fixed, all right/local perturbations):

    δx = [δθ (0:3), δv (3:6), δp (6:9), δbg (9:12), δba (12:15)]

- ``δθ`` [rad]: body-frame attitude error, ``q_true = q̂ (x) Exp(δθ)``
- ``δv`` [m/s]: velocity error in the navigation frame (NED or ECEF)
- ``δp`` [m]: position error in the navigation frame (for NED states the
  nominal position is geodetic; δp is converted through the meridian /
  transverse radii on injection)
- ``δbg`` [rad/s], ``δba`` [m/s²]: additive IMU bias errors (random walks)

Nominal propagation is the shared strapdown kernel
(:func:`qnav.nav.mechanization.propagate_state`) — full Earth rate,
transport rate, Coriolis, and Somigliana gravity. The **error** dynamics use
the standard first-order discrete blocks:

    F_θθ = Exp(ŵ dt)ᵀ          F_θbg = −Jr(ŵ dt)·dt
    F_vθ = −R [f̂]× dt          F_vba = −R dt
    F_pv = I dt                 (bias blocks: identity)

Approximation (documented): Earth-rate and transport-rate terms are kept in
the *nominal* mechanization but omitted from the *error* Jacobian — they
contribute O(Ω·dt) ≈ 7e-7 per 10 ms step to F, far below MEMS/tactical gyro
noise. This is the standard small-Ω simplification; do not use this filter
to align a navigation-grade INS by gyrocompassing.

Updates share the gated Joseph-form kernel with the attitude ESKF
(:func:`qnav.filters._kalman.gated_joseph_update`): same NIS gating, robust
losses, quarantine, and UpdateResult reporting.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from qnav._validate import (
    ensure_covariance,
    ensure_nonnegative,
    ensure_positive,
    ensure_positive_dt,
    ensure_vector3,
)
from qnav.attitude import quaternion as quat
from qnav.attitude import so3
from qnav.filters._kalman import gated_joseph_update
from qnav.filters.base import EstimatorLifecycle
from qnav.filters.contracts import EstimatorHealth
from qnav.filters.robust import GatePolicy, SensorMonitor
from qnav.frames.earth import meridian_radius, transverse_radius
from qnav.nav.mechanization import propagate_state
from qnav.nav.state import NavState
from qnav.types import ArrayLike

__all__ = ["NavEskf"]

_DEFAULT_P0 = np.diag(
    [0.05**2] * 3      # attitude [rad²]
    + [0.5**2] * 3     # velocity [(m/s)²]
    + [5.0**2] * 3     # position [m²]
    + [0.01**2] * 3    # gyro bias [(rad/s)²]
    + [0.05**2] * 3    # accel bias [(m/s²)²]
)


class NavEskf(EstimatorLifecycle):
    """15-state inertial navigation ESKF (NED or ECEF, set by the state).

    Parameters
    ----------
    state0:
        Initial :class:`~qnav.nav.state.NavState` (frame fixes the
        mechanization).
    gyro_noise_density, accel_noise_density:
        White-noise densities [rad/s/√Hz], [m/s²/√Hz].
    gyro_bias_walk, accel_bias_walk:
        Bias random-walk densities [rad/s²/√Hz], [m/s³/√Hz].
    P0:
        Initial 15x15 error covariance (default: modest tactical-grade
        uncertainty; see ``_DEFAULT_P0`` ordering above).
    gate:
        Optional innovation gating / robust-loss policy for all updates.
    """

    def __init__(
        self,
        state0: NavState,
        gyro_noise_density: float,
        accel_noise_density: float,
        gyro_bias_walk: float = 0.0,
        accel_bias_walk: float = 0.0,
        P0: np.ndarray | None = None,
        gate: Optional[GatePolicy] = None,
    ) -> None:
        super().__init__()
        if not isinstance(state0, NavState):
            raise TypeError("state0 must be a NavState")
        self.state = state0
        self.sigma_g = ensure_nonnegative(gyro_noise_density, "gyro_noise_density")
        self.sigma_a = ensure_nonnegative(accel_noise_density, "accel_noise_density")
        self.sigma_bg = ensure_nonnegative(gyro_bias_walk, "gyro_bias_walk")
        self.sigma_ba = ensure_nonnegative(accel_bias_walk, "accel_bias_walk")
        self.P = ensure_covariance(P0 if P0 is not None else _DEFAULT_P0, 15, "P0").copy()
        self.gate = gate
        self.monitors: Dict[str, SensorMonitor] = {}
        self._fused_directions: Deque[np.ndarray] = deque(maxlen=50)

    # EstimatorLifecycle expects the attitude under .q
    @property
    def q(self) -> np.ndarray:  # type: ignore[override]
        """Nominal attitude quaternion (frame per ``self.state.frame``)."""
        return self.state.q

    def set_monitor(self, sensor_id: str, monitor: SensorMonitor) -> None:
        """Attach a quarantine/timeout monitor to one measurement stream."""
        self.monitors[sensor_id] = monitor

    # -- prediction -----------------------------------------------------------
    def predict(self, omega_ib_b: ArrayLike, f_b: ArrayLike, dt: float) -> NavState:
        """Propagate nominal state and error covariance with one IMU sample."""
        w_meas = ensure_vector3(omega_ib_b, "omega_ib_b")
        f_meas = ensure_vector3(f_b, "f_b")
        dt = ensure_positive_dt(dt)

        w = w_meas - self.state.bg
        f = f_meas - self.state.ba
        R = quat_to_dcm(self.state.q)  # attitude at step start (F linearization point)

        self.state = propagate_state(self.state, w, f, dt)

        phi = w * dt
        F = np.eye(15)
        F[0:3, 0:3] = so3.exp(phi).T
        F[0:3, 9:12] = -so3.right_jacobian(phi) * dt
        F[3:6, 0:3] = -R @ so3.hat(f) * dt
        F[3:6, 12:15] = -R * dt
        F[6:9, 3:6] = np.eye(3) * dt

        Qd = np.zeros((15, 15))
        Qd[0:3, 0:3] = (self.sigma_g**2 * dt) * np.eye(3)
        Qd[3:6, 3:6] = (self.sigma_a**2 * dt) * np.eye(3)
        Qd[9:12, 9:12] = (self.sigma_bg**2 * dt) * np.eye(3)
        Qd[12:15, 12:15] = (self.sigma_ba**2 * dt) * np.eye(3)

        self.P = F @ self.P @ F.T + Qd
        return self.state

    # -- updates ---------------------------------------------------------------
    def update_position(
        self, p_meas: ArrayLike, sigma: ArrayLike,
        *, timestamp: float | None = None, sensor_id: str = "position",
    ) -> np.ndarray:
        """Fuse a position measurement in the state's own convention.

        NED states: ``p_meas = [lat, lon, h]`` [rad, rad, m]; the innovation
        is formed in meters (N/E/D) through the local radii. ECEF states:
        ``p_meas = r_ecef`` [m]. ``sigma``: scalar or per-axis std [m].
        Returns the innovation [m].
        """
        p = ensure_vector3(p_meas, "p_meas")
        R_meas = _sigma_to_cov(sigma)
        if self.state.frame == "NED":
            lat, lon, h = (float(x) for x in self.state.p)
            M = float(meridian_radius(lat))
            N = float(transverse_radius(lat))
            innov = np.array([
                (p[0] - lat) * (M + h),
                (p[1] - lon) * (N + h) * np.cos(lat),
                -(p[2] - h),
            ])
        else:
            innov = p - self.state.p
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        gated_joseph_update(self, H, R_meas, innov, inject=self._inject,
                            sensor_id=sensor_id, timestamp=timestamp)
        return innov

    def update_velocity(
        self, v_meas: ArrayLike, sigma: ArrayLike,
        *, timestamp: float | None = None, sensor_id: str = "velocity",
    ) -> np.ndarray:
        """Fuse a navigation-frame velocity measurement [m/s]."""
        v = ensure_vector3(v_meas, "v_meas")
        innov = v - self.state.v
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        gated_joseph_update(self, H, _sigma_to_cov(sigma), innov,
                            inject=self._inject, sensor_id=sensor_id,
                            timestamp=timestamp)
        return innov

    def update_direction(
        self, v_nav: ArrayLike, v_body_meas: ArrayLike, sigma: float,
        *, timestamp: float | None = None, sensor_id: str = "direction",
    ) -> np.ndarray:
        """Fuse a unit-direction pair (gravity, magnetic field) — same
        semantics as :meth:`qnav.filters.Eskf.update_direction`."""
        sigma = ensure_positive(sigma, "sigma")
        vn = ensure_vector3(v_nav, "v_nav")
        vn = vn / np.linalg.norm(vn)
        vb = ensure_vector3(v_body_meas, "v_body_meas")
        nb = np.linalg.norm(vb)
        if nb < 1e-12:
            raise ValueError("v_body_meas must have non-zero norm")
        vb = vb / nb
        v_hat = quat.rotate_frame(self.state.q, vn)
        H = np.zeros((3, 15))
        H[:, 0:3] = so3.hat(v_hat)
        innov = vb - v_hat
        result = gated_joseph_update(self, H, (sigma**2) * np.eye(3), innov,
                                     inject=self._inject, sensor_id=sensor_id,
                                     timestamp=timestamp)
        if result.accepted:
            self._fused_directions.append(vn)
        return innov

    def update_measurement(
        self, model: object, value: object, sigma: ArrayLike,
        *, timestamp: float | None = None, sensor_id: str | None = None,
        **aux: object,
    ) -> np.ndarray:
        """Fuse any :mod:`qnav.nav.measurements` model through the shared
        gated kernel.

        ``model.residual(state, value, **aux)`` supplies the innovation and
        Jacobian; ``sigma`` is the scalar or per-component measurement std in
        the model's units. Returns the innovation. All gating, robust
        weighting, quarantine, and reporting behave exactly as for the
        built-in updates.
        """
        innov, H = model.residual(self.state, value, **aux)  # type: ignore[attr-defined]
        innov = np.atleast_1d(np.asarray(innov, dtype=float))
        m = innov.shape[0]
        s = np.asarray(sigma, dtype=float)
        if s.ndim == 0:
            s = np.full(m, float(s))
        if s.shape != (m,) or np.any(~np.isfinite(s)) or np.any(s <= 0):
            raise ValueError(f"sigma must be a positive scalar or length-{m} vector")
        sid = sensor_id if sensor_id is not None else type(model).__name__
        gated_joseph_update(self, np.asarray(H, dtype=float), np.diag(s**2), innov,
                            inject=self._inject, sensor_id=sid, timestamp=timestamp)
        return innov

    # -- internals ---------------------------------------------------------------
    def _inject(self, dx: np.ndarray) -> None:
        dtheta, dv, dp, dbg, dba = (dx[0:3], dx[3:6], dx[6:9], dx[9:12], dx[12:15])
        q_new = quat.normalize(quat.mul(self.state.q, quat.exp(dtheta)))
        if self.state.frame == "NED":
            lat, lon, h = (float(x) for x in self.state.p)
            M = float(meridian_radius(lat))
            N = float(transverse_radius(lat))
            p_new = np.array([
                lat + dp[0] / (M + h),
                lon + dp[1] / ((N + h) * np.cos(lat)),
                h - dp[2],
            ])
        else:
            p_new = self.state.p + dp
        self.state = self.state.evolve(
            q=q_new, v=self.state.v + dv, p=p_new,
            bg=self.state.bg + dbg, ba=self.state.ba + dba,
        )
        # post-injection covariance reset (attitude block only; Solà eq. 287)
        G = np.eye(15)
        G[0:3, 0:3] = np.eye(3) - 0.5 * so3.hat(dtheta)
        self.P = G @ self.P @ G.T

    # -- diagnostics ---------------------------------------------------------------
    @property
    def position_std(self) -> np.ndarray:
        """Per-axis position error std [m]."""
        return np.sqrt(np.diag(self.P)[6:9])

    @property
    def velocity_std(self) -> np.ndarray:
        """Per-axis velocity error std [m/s]."""
        return np.sqrt(np.diag(self.P)[3:6])

    @property
    def attitude_std(self) -> np.ndarray:
        """Per-axis attitude error std [rad]."""
        return np.sqrt(np.diag(self.P)[0:3])

    @property
    def health(self) -> EstimatorHealth:
        base = EstimatorLifecycle.health.fget(self)  # type: ignore[attr-defined]
        if base is not EstimatorHealth.INVALID and not (
            np.all(np.isfinite(self.state.v)) and np.all(np.isfinite(self.state.p))
        ):
            return EstimatorHealth.INVALID
        return base


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Body-to-nav DCM from ``q_nav_body`` (thin wrapper for clarity)."""
    from qnav.attitude import dcm
    return dcm.from_quaternion(q)


def _sigma_to_cov(sigma: ArrayLike) -> np.ndarray:
    """Scalar or per-axis std -> 3x3 diagonal covariance."""
    s = np.asarray(sigma, dtype=float)
    if s.ndim == 0:
        s = np.full(3, float(s))
    if s.shape != (3,):
        raise ValueError(f"sigma must be scalar or length-3, got shape {s.shape}")
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("sigma must be finite and positive")
    return np.diag(s**2)
