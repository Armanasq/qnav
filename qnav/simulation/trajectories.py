"""Analytic attitude/motion trajectory generators with exact derivatives.

Each generator returns a :class:`Trajectory`: sampled time, ground-truth
attitude ``q_nav_body``, **body-frame** angular velocity, navigation-frame
acceleration of the body origin, and navigation-frame velocity/position when
defined. Rates are analytic (not finite-differenced) wherever possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnav.attitude import quaternion as quat

__all__ = ["Trajectory", "static_pose", "constant_rate", "sinusoidal_euler", "coning"]


@dataclass(frozen=True)
class Trajectory:
    """Sampled ground-truth motion (navigation frame declared by producer)."""

    t: np.ndarray            # (N,)
    q: np.ndarray            # (N, 4)  q_nav_body
    omega_body: np.ndarray   # (N, 3)  body-frame angular velocity [rad/s]
    accel_nav: np.ndarray    # (N, 3)  inertial acceleration of body origin [m/s²]
    nav_frame: str = "NED"

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def n(self) -> int:
        return int(self.t.size)


def static_pose(q0: np.ndarray, duration: float, dt: float, nav_frame: str = "NED") -> Trajectory:
    """Motionless body at attitude ``q0``."""
    t = np.arange(0.0, duration, dt)
    n = t.size
    return Trajectory(
        t=t, q=np.tile(quat.normalize(np.asarray(q0, dtype=float)), (n, 1)),
        omega_body=np.zeros((n, 3)), accel_nav=np.zeros((n, 3)), nav_frame=nav_frame,
    )


def constant_rate(
    omega_body: np.ndarray, duration: float, dt: float,
    q0: np.ndarray | None = None, nav_frame: str = "NED",
) -> Trajectory:
    """Constant body-rate spin: ``q(t) = q0 ⊗ Exp(ω t)`` (exact)."""
    t = np.arange(0.0, duration, dt)
    w = np.asarray(omega_body, dtype=float)
    q0 = quat.identity() if q0 is None else quat.normalize(np.asarray(q0, dtype=float))
    qs = quat.mul(q0, quat.exp(t[:, None] * w))
    return Trajectory(
        t=t, q=qs, omega_body=np.tile(w, (t.size, 1)),
        accel_nav=np.zeros((t.size, 3)), nav_frame=nav_frame,
    )


def sinusoidal_euler(
    amp: np.ndarray, freq: np.ndarray, duration: float, dt: float,
    nav_frame: str = "NED",
) -> Trajectory:
    """Sinusoidal ZYX Euler motion: yaw/pitch/roll = amp·sin(2πf t).

    ``amp``/``freq``: per-angle (yaw, pitch, roll) amplitude [rad] and
    frequency [Hz]. ω is recovered exactly from consecutive quaternions via
    the log map at machine precision of the sampling (midpoint rate), which
    keeps q and ω kinematically consistent by construction.
    """
    from qnav.attitude import euler as _euler

    t = np.arange(0.0, duration, dt)
    amp = np.asarray(amp, dtype=float)
    freq = np.asarray(freq, dtype=float)
    ang = amp * np.sin(2.0 * np.pi * freq * t[:, None])
    qs = _euler.to_quaternion(ang, "ZYX")
    w = np.zeros((t.size, 3))
    # consistent discrete rate: q_{k+1} = q_k ⊗ Exp(ω_k dt)
    w[:-1] = quat.log(quat.relative(qs[:-1], qs[1:])) / dt
    w[-1] = w[-2]
    return Trajectory(t=t, q=qs, omega_body=w, accel_nav=np.zeros((t.size, 3)),
                      nav_frame=nav_frame)


def coning(
    half_angle: float, spin_rate: float, duration: float, dt: float,
    nav_frame: str = "NED",
) -> Trajectory:
    """Classic coning motion — the standard integrator stress test.

    The body symmetry axis sweeps a cone of half-angle β about nav-z at rate
    Ω: ``q(t) = Exp([0, 0, Ωt]) ⊗ Exp([β, 0, 0])``. Body rates follow from
    the exact relative-rotation log between samples.
    """
    t = np.arange(0.0, duration, dt)
    q_spin = quat.exp(np.stack(
        [np.zeros_like(t), np.zeros_like(t), spin_rate * t], axis=-1))
    q_tilt = quat.exp(np.array([half_angle, 0.0, 0.0]))
    qs = quat.mul(q_spin, q_tilt)
    w = np.zeros((t.size, 3))
    w[:-1] = quat.log(quat.relative(qs[:-1], qs[1:])) / dt
    w[-1] = w[-2]
    return Trajectory(t=t, q=qs, omega_body=w, accel_nav=np.zeros((t.size, 3)),
                      nav_frame=nav_frame)
