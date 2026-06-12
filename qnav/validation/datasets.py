"""Canonical synthetic validation datasets (fixed seeds, documented truth).

These are the shared scenarios used by tests, benchmarks, and documentation
examples, so every number in the docs is regenerable.
"""

from __future__ import annotations

import numpy as np

from qnav.sensors import GyroModel, ImuModel, MagnetometerModel, NoiseModel
from qnav.sensors.accelerometer import AccelerometerModel
from qnav.simulation import MagneticEnvironment, sinusoidal_euler, synthesize
from qnav.simulation.imu_synthesis import ImuDataset

__all__ = ["marg_dataset"]


def marg_dataset(
    duration: float = 60.0, dt: float = 0.01, seed: int = 42,
    gyro_bias: np.ndarray | None = None,
) -> ImuDataset:
    """Standard MARG scenario: sinusoidal Euler motion, consumer-grade noise.

    Gyro: 0.005 rad/s/√Hz noise, constant bias (default [0.02, −0.01, 0.015]
    rad/s); accel: 0.05 m/s²/√Hz; mag: 60° dip, 50 µT, 0.3 µT/√Hz.
    NED navigation frame, FRD body.
    """
    rng = np.random.default_rng(seed)
    truth = sinusoidal_euler(
        amp=np.deg2rad([30.0, 20.0, 25.0]), freq=[0.1, 0.17, 0.23],
        duration=duration, dt=dt, nav_frame="NED",
    )
    bias = np.array([0.02, -0.01, 0.015]) if gyro_bias is None else np.asarray(gyro_bias)
    imu = ImuModel(
        gyro=GyroModel(bias=bias, noise=NoiseModel(density=0.005)),
        accelerometer=AccelerometerModel(noise=NoiseModel(density=0.05)),
        magnetometer=MagnetometerModel(noise=NoiseModel(density=0.3e-6)),
    )
    return synthesize(
        truth, imu, magnetic=MagneticEnvironment(), rng=rng,
    )
