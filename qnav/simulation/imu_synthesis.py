"""Synthetic IMU/MARG measurement generation from ground-truth trajectories.

Pipeline: :class:`~qnav.simulation.trajectories.Trajectory` (truth) +
:class:`~qnav.sensors.imu.ImuModel` (errors) + gravity/magnetic environments
→ time-stamped gyro/accel/mag measurements, with optional bias trajectories,
dropout, and timestamp jitter (:mod:`qnav.simulation.noise_injection`).

Sign conventions: accelerometer output is specific force
``f_B = R_BN (a_N − g_N)`` (see ``qnav.sensors.accelerometer``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.sensors.imu import ImuModel
from qnav.sensors.noise import random_walk
from qnav.simulation.gravity import ConstantGravity, GravityModel
from qnav.simulation.magnetic_field import MagneticEnvironment
from qnav.simulation.trajectories import Trajectory

__all__ = ["ImuDataset", "synthesize"]


@dataclass(frozen=True)
class ImuDataset:
    """Synthetic measurement set aligned with its ground-truth trajectory."""

    truth: Trajectory
    gyro: np.ndarray                       # (N, 3) [rad/s]
    accel: np.ndarray                      # (N, 3) [m/s²] specific force
    mag: Optional[np.ndarray] = None       # (N, 3) field units
    gyro_bias_true: Optional[np.ndarray] = None  # (N, 3) if simulated


def synthesize(
    truth: Trajectory,
    imu: ImuModel,
    gravity: GravityModel | None = None,
    magnetic: MagneticEnvironment | None = None,
    rng: np.random.Generator | None = None,
    simulate_bias_walk: bool = False,
) -> ImuDataset:
    """Generate IMU/MARG measurements along ``truth``.

    ``rng=None`` produces noise-free (deterministic-error-only) outputs.
    ``simulate_bias_walk=True`` adds a gyro bias random walk on top of the
    model's constant bias (requires ``rng``); the realized bias trajectory is
    returned for estimator validation.
    """
    gravity = gravity or ConstantGravity()
    g_nav = gravity.vector(truth.nav_frame)
    dt = truth.dt

    # true specific force in body coordinates: f_B = R_BN (a_N − g_N)
    f_body = quat.rotate_frame(truth.q, truth.accel_nav - g_nav)

    bias_traj = None
    if simulate_bias_walk:
        if rng is None:
            raise ValueError("simulate_bias_walk requires an rng")
        bias_traj = imu.gyro.bias + random_walk(
            truth.n, dt, imu.gyro.noise.bias_walk, rng
        )

    gyro_meas = imu.measure_gyro(
        truth.omega_body, dt, rng=rng, bias_trajectory=bias_traj,
    )
    accel_meas = imu.measure_accel(f_body, dt, rng=rng, omega_body=truth.omega_body)

    mag_meas = None
    if magnetic is not None and imu.magnetometer is not None:
        m_nav = magnetic.field_nav(truth.t)
        m_body = quat.rotate_frame(truth.q, m_nav)
        mag_meas = imu.measure_mag(m_body, dt, rng=rng)

    return ImuDataset(
        truth=truth, gyro=gyro_meas, accel=accel_meas, mag=mag_meas,
        gyro_bias_true=bias_traj,
    )
