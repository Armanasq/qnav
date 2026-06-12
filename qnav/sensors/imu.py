"""Composite IMU/MARG sensor suite.

Bundles gyro, accelerometer, and (optionally) magnetometer models with their
mounting alignments into one object that turns *true* body-frame kinematics
into *measured* sensor outputs — the single entry point used by
``qnav.simulation.imu_synthesis``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qnav.sensors.accelerometer import AccelerometerModel
from qnav.sensors.alignment import SensorAlignment, lever_arm_acceleration
from qnav.sensors.gyro import GyroModel
from qnav.sensors.magnetometer import MagnetometerModel

__all__ = ["ImuModel"]


@dataclass(frozen=True)
class ImuModel:
    """An IMU (optionally MARG) with per-sensor error models and alignments."""

    gyro: GyroModel = field(default_factory=GyroModel)
    accelerometer: AccelerometerModel = field(default_factory=AccelerometerModel)
    magnetometer: Optional[MagnetometerModel] = None
    gyro_alignment: SensorAlignment = field(default_factory=SensorAlignment)
    accel_alignment: SensorAlignment = field(default_factory=SensorAlignment)
    mag_alignment: SensorAlignment = field(default_factory=SensorAlignment)

    def measure_gyro(
        self, omega_body: np.ndarray, dt: float,
        rng: np.random.Generator | None = None, **kw
    ) -> np.ndarray:
        """Gyro output in the **gyro sensor frame** for true body rates."""
        w_s = self.gyro_alignment.to_sensor(omega_body)
        return self.gyro.measure(w_s, dt, rng=rng, **kw)

    def measure_accel(
        self, f_body: np.ndarray, dt: float,
        rng: np.random.Generator | None = None,
        omega_body: np.ndarray | None = None,
        alpha_body: np.ndarray | None = None, **kw
    ) -> np.ndarray:
        """Accelerometer output in its sensor frame.

        ``f_body`` is the specific force at the **body origin**; if rates and
        angular acceleration are given, the lever-arm contribution at the
        mounting position is added before the error model.
        """
        f = np.asarray(f_body, dtype=float)
        if omega_body is not None:
            a0 = np.zeros_like(f) if alpha_body is None else np.asarray(alpha_body, dtype=float)
            f = f + lever_arm_acceleration(omega_body, a0, self.accel_alignment.lever_arm)
        f_s = self.accel_alignment.to_sensor(f)
        return self.accelerometer.measure(f_s, dt, rng=rng, **kw)

    def measure_mag(
        self, m_body: np.ndarray, dt: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Magnetometer output in its sensor frame (raises if no magnetometer)."""
        if self.magnetometer is None:
            raise ValueError("this ImuModel has no magnetometer")
        m_s = self.mag_alignment.to_sensor(m_body)
        return self.magnetometer.measure(m_s, dt, rng=rng)
