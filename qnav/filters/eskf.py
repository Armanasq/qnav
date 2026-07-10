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

from collections import deque
from typing import Deque, Dict, Optional

from qnav._validate import ensure_covariance, ensure_nonnegative, ensure_positive, ensure_vector3
from qnav.attitude import quaternion as quat
from qnav.attitude import so3
from qnav.filters.base import AttitudeFilter
from qnav.filters.contracts import EstimatorHealth, UpdateResult
from qnav.filters.robust import GatePolicy, SensorMonitor
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
        gate: Optional[GatePolicy] = None,
    ) -> None:
        super().__init__(q0=q0, nav_frame=nav_frame)
        self.bias = np.zeros(3) if b0 is None else ensure_vector3(b0, "b0").copy()
        self.sigma_g = ensure_nonnegative(gyro_noise_density, "gyro_noise_density")
        self.sigma_bw = ensure_nonnegative(gyro_bias_walk, "gyro_bias_walk")
        if P0 is None:
            P0 = np.diag([0.1**2] * 3 + [0.01**2] * 3)
        self.P = ensure_covariance(P0, 6, "P0").copy()
        #: innovation gating / robust-loss policy; None = plain Kalman updates.
        self.gate = gate
        #: optional per-sensor quarantine monitors (see :meth:`set_monitor`).
        self.monitors: Dict[str, SensorMonitor] = {}
        # unit nav-frame directions of recently *accepted* updates, for
        # observability assessment (yaw about a single fused direction is
        # unobservable).
        self._fused_directions: Deque[np.ndarray] = deque(maxlen=50)

    def set_monitor(self, sensor_id: str, monitor: SensorMonitor) -> None:
        """Attach a quarantine/timeout monitor to one measurement stream."""
        self.monitors[sensor_id] = monitor

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

        When a :class:`~qnav.filters.robust.GatePolicy` is configured, the
        NIS is tested against its chi-square threshold: hard rejection leaves
        the state untouched (``last_update.accepted`` is False); soft
        inflation scales the measurement noise by ``nis/threshold``. Robust
        losses de-weight accepted measurements by inflating the noise by
        ``1/w``. A quarantined sensor (see :meth:`set_monitor`) is evaluated
        but never fused. ``nis`` in the recorded result is always the
        *pre-inflation* value tested against the gate.
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
        nis = float(innov @ np.linalg.solve(S, innov))

        threshold: float | None = None
        weight = 1.0
        rejection: str | None = None
        if self.gate is not None:
            threshold = self.gate.threshold(3)
            if nis > threshold:
                if self.gate.on_gate == "reject":
                    rejection = "nis_gate"
                else:  # soft inflation: keep the update, reduce its trust
                    weight *= threshold / nis
            if rejection is None:
                weight *= self.gate.robust_weight(nis, 3)

        monitor = self.monitors.get(sensor_id)
        if monitor is not None:
            allowed = monitor.note_measurement(rejection is None, timestamp)
            if rejection is None and not allowed:
                rejection = "quarantine"

        if rejection is not None:
            self._record_update(UpdateResult(
                accepted=False, innovation=innov, innovation_covariance=S,
                nis=nis, gate_threshold=threshold, robust_weight=0.0,
                rejection_reason=rejection, timestamp=timestamp,
                sensor_id=sensor_id,
            ))
            return innov

        if weight != 1.0:
            R = R / weight
            S = H @ self.P @ H.T + R

        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))
        dx = K @ innov

        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        self._inject(dx)
        self._fused_directions.append(vn)
        self._record_update(UpdateResult(
            accepted=True, innovation=innov, innovation_covariance=S, nis=nis,
            gate_threshold=threshold, robust_weight=weight,
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

    # -- recovery actions ----------------------------------------------------
    def inflate_covariance(self, factor: float, *, attitude_only: bool = False) -> None:
        """Multiply the error covariance by ``factor`` (> 1 admits more
        correction from subsequent measurements — the standard soft recovery
        from suspected divergence). ``attitude_only`` leaves the bias block
        untouched."""
        f = ensure_positive(factor, "factor")
        if attitude_only:
            self.P[:3, :3] *= f
            self.P[:3, 3:] *= np.sqrt(f)
            self.P[3:, :3] *= np.sqrt(f)
        else:
            self.P *= f

    def reinitialize_from_vectors(
        self,
        f_body: ArrayLike,
        m_body: ArrayLike | None = None,
        m_ref: ArrayLike | None = None,
        *,
        keep_bias: bool = True,
        attitude_std0: float = 0.35,
    ) -> np.ndarray:
        """Reset the attitude from a deterministic closed-form solve (FQA).

        Uses the accelerometer (tilt) and optionally magnetometer + reference
        field (yaw). The attitude covariance block is reset to
        ``attitude_std0²·I`` and its cross-correlations cleared; the bias
        estimate and bias covariance are preserved when ``keep_bias`` (reset
        to zero / constructor default variance otherwise). Update history is
        cleared. Returns the new quaternion.
        """
        from qnav.determination.fqa import fqa

        f = ensure_vector3(f_body, "f_body")
        m = None if m_body is None else ensure_vector3(m_body, "m_body")
        mr = None if m_ref is None else ensure_vector3(m_ref, "m_ref")
        q_new = fqa(f, m, mr)
        if self.nav_frame == "ENU":
            raise NotImplementedError(
                "reinitialize_from_vectors currently supports nav_frame='NED' "
                "(FQA is NED-referenced); convert or reinitialize manually"
            )
        self.q = quat.normalize(q_new)
        std0 = ensure_positive(attitude_std0, "attitude_std0")
        self.P[:3, :3] = std0**2 * np.eye(3)
        self.P[:3, 3:] = 0.0
        self.P[3:, :3] = 0.0
        if not keep_bias:
            self.bias = np.zeros(3)
            self.P[3:, 3:] = 0.01**2 * np.eye(3)
        self._fused_directions.clear()
        self.last_update = None
        self.innovation_stats.clear()
        return self.q

    # -- health --------------------------------------------------------------
    @property
    def health(self) -> EstimatorHealth:
        """Extends the base checks with attitude observability: when every
        recently fused nav-frame direction is (near-)collinear, the rotation
        about that direction is unconstrained and the estimator reports
        ``UNOBSERVABLE`` instead of ``HEALTHY``."""
        base = AttitudeFilter.health.fget(self)  # type: ignore[attr-defined]
        if base is not EstimatorHealth.HEALTHY:
            return base
        if len(self._fused_directions) >= 10:
            d = np.stack(tuple(self._fused_directions))
            # all directions within ~5° of the first (or its antipode)
            if np.all(np.abs(d @ d[0]) > np.cos(np.deg2rad(5.0))):
                return EstimatorHealth.UNOBSERVABLE
        return base
