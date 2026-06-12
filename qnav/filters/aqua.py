"""AQUA: Algebraic Quaternion Algorithm (Valenti, Dryanovski, Xiao 2015).

A complementary filter with a fully algebraic correction step. Its defining
features, preserved in this implementation:

- **Decoupled corrections.** The accelerometer correction is a zero-yaw
  (tilt-only) delta quaternion; the magnetometer correction is a yaw-only
  delta. A magnetic disturbance therefore *cannot* corrupt roll/pitch —
  structurally, not just statistically.
- **Algebraic deltas.** Both corrections are closed-form shortest-arc
  quaternions (square roots only, no trig), computed in the navigation frame
  from the predicted attitude.
- **Threshold-gated interpolation.** Each delta is scaled toward identity
  with gain α (accel) / β (mag): LERP when the delta is small (cheap,
  first-order exact), SLERP when large — the original's two-regime scheme.
- **Adaptive gain.** Optionally scales α by how far the measured specific
  force magnitude is from gravity — a piecewise-linear trust gate that
  suspends accelerometer corrections during dynamic maneuvers.

qnav conventions: state ``q_NB`` (nav-from-body, default NED); gyro in rad/s
(body); accelerometer is **specific force**; magnetometer any norm.

Reference: Valenti, Dryanovski, Xiao, "Keeping a good attitude: A
quaternion-based orientation filter for IMUs and MARGs", Sensors 15(8), 2015.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import kinematics as kin
from qnav.attitude import quaternion as quat
from qnav.errors import ConventionError
from qnav.filters.base import AttitudeFilter

__all__ = ["AquaFilter"]


def _tilt_delta(g_nav: np.ndarray) -> np.ndarray:
    """Zero-yaw shortest-arc quaternion taking ``g_nav`` to ``[0,0,1]``.

    ``g_nav`` is the measured gravity-down direction expressed in nav axes
    using the predicted attitude (unit norm). Antipodal-safe via the
    two-branch half-angle form.
    """
    gx, gy, gz = g_nav
    if gz >= 0.0:
        s = np.sqrt(2.0 * (1.0 + gz))
        return np.array([0.5 * s, gy / s, -gx / s, 0.0])
    # near-antipodal branch: rotate via the horizontal axis representation
    s = np.sqrt(2.0 * (1.0 - gz))
    return np.array([gy / s, 0.5 * s, 0.0, gx / s])


def _yaw_delta(m_nav: np.ndarray) -> np.ndarray:
    """Yaw-only quaternion aligning the horizontal field with magnetic north.

    Operates on the horizontal projection only — the vertical (dip) component
    is discarded, so dip-angle errors never feed the correction.
    """
    lx, ly = m_nav[0], m_nav[1]
    gamma = lx * lx + ly * ly
    if gamma < 1e-24:
        return np.array([1.0, 0.0, 0.0, 0.0])
    sg = np.sqrt(gamma)
    if lx >= 0.0:
        d = np.sqrt(2.0 * (gamma + lx * sg))
        return np.array([np.sqrt(gamma + lx * sg) / np.sqrt(2.0 * gamma), 0.0, 0.0, -ly / d])
    d = np.sqrt(2.0 * (gamma - lx * sg))
    return np.array([ly / d, 0.0, 0.0, -np.sqrt(gamma - lx * sg) / np.sqrt(2.0 * gamma)])


def _scale_toward_identity(dq: np.ndarray, gain: float, lerp_threshold: float) -> np.ndarray:
    """Interpolate a delta quaternion toward identity by ``gain`` ∈ [0, 1].

    LERP when ``dq_w > lerp_threshold`` (small rotation — first-order exact
    and division-free), SLERP otherwise. Always returns unit norm.
    """
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    if dq[0] > lerp_threshold:
        out = (1.0 - gain) * q_id + gain * dq
    else:
        ang = np.arccos(np.clip(dq[0], -1.0, 1.0))
        s = np.sin(ang)
        out = q_id * np.sin((1.0 - gain) * ang) / s + dq * np.sin(gain * ang) / s
    return out / np.linalg.norm(out)


class AquaFilter(AttitudeFilter):
    """Algebraic quaternion complementary filter (tilt/yaw decoupled).

    Parameters
    ----------
    alpha:
        Accelerometer correction gain ∈ (0, 1].
    beta:
        Magnetometer correction gain ∈ (0, 1].
    lerp_threshold:
        Delta-w above which LERP replaces SLERP (default 0.9 ≈ 52° delta).
    adaptive:
        Scale ``alpha`` by the specific-force magnitude gate (suspends accel
        trust during maneuvers).
    gravity:
        Local gravity magnitude [m/s²] for the adaptive gate.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        beta: float = 0.01,
        lerp_threshold: float = 0.9,
        adaptive: bool = False,
        gravity: float = 9.80665,
        q0=None,
        nav_frame: str = "NED",
    ) -> None:
        if nav_frame != "NED":
            raise ConventionError("AquaFilter currently supports nav_frame='NED' only")
        super().__init__(q0=q0, nav_frame=nav_frame)
        if not 0.0 < alpha <= 1.0 or not 0.0 < beta <= 1.0:
            raise ValueError("alpha and beta must be in (0, 1]")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lerp_threshold = float(lerp_threshold)
        self.adaptive = bool(adaptive)
        self.gravity = float(gravity)

    # -- initialization ------------------------------------------------------
    def initialize(self, f_body: np.ndarray, m_body: np.ndarray | None = None) -> np.ndarray:
        """Algebraic attitude from one accel (+ optional mag) sample.

        Tilt from the shortest-arc construction, yaw from the de-tilted
        horizontal field — the determination half of the original algorithm.
        """
        f = np.asarray(f_body, dtype=float)
        fn = np.linalg.norm(f)
        if fn < 1e-12:
            raise ValueError("zero-norm accelerometer sample")
        q_t = _tilt_delta(-f / fn)          # body-frame down-direction = −f̂
        if m_body is not None:
            m = np.asarray(m_body, dtype=float)
            mn = np.linalg.norm(m)
            if mn > 1e-12:
                m_nav = quat.rotate_vector(q_t, m / mn)
                q_t = quat.mul(_yaw_delta(m_nav), q_t)
        self.q = quat.canonical(quat.normalize(q_t))
        return self.q

    # -- gyro-only propagation -----------------------------------------------
    def predict(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        """Gyro-only exponential propagation (no corrections)."""
        self.q = kin.integrate_exponential(self.q, omega_body, dt)
        return self.q

    # -- per-sample update ---------------------------------------------------
    def step(
        self,
        omega_body: np.ndarray,
        dt: float,
        f_body: np.ndarray | None = None,
        m_body: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict with the gyro, then apply tilt and yaw corrections."""
        q = kin.integrate_exponential(self.q, omega_body, dt)

        if f_body is not None:
            f = np.asarray(f_body, dtype=float)
            fn = np.linalg.norm(f)
            if fn > 1e-12:
                g_nav = quat.rotate_vector(q, -f / fn)
                dq = _tilt_delta(g_nav)
                gain = self._effective_alpha(fn)
                dq = _scale_toward_identity(dq, gain, self.lerp_threshold)
                q = quat.normalize(quat.mul(dq, q))

        if m_body is not None:
            m = np.asarray(m_body, dtype=float)
            mn = np.linalg.norm(m)
            if mn > 1e-12:
                m_nav = quat.rotate_vector(q, m / mn)
                dq = _scale_toward_identity(
                    _yaw_delta(m_nav), self.beta, self.lerp_threshold
                )
                q = quat.normalize(quat.mul(dq, q))

        self.q = q
        return self.q

    def _effective_alpha(self, f_norm: float, t1: float = 0.1, t2: float = 0.2) -> float:
        """Piecewise-linear magnitude gate (Valenti Fig. 5): full gain when the
        specific-force magnitude error is below ``t1·g``, zero above ``t2·g``."""
        if not self.adaptive:
            return self.alpha
        err = abs(f_norm - self.gravity) / self.gravity
        return self.alpha * float(np.clip((t2 - err) / t1, 0.0, 1.0))
