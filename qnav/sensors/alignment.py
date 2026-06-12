"""Sensor-to-body frame alignment.

Each sensor measures in its own frame S; the vehicle works in body frame B.
Alignment is the fixed transform ``q_BS`` (body-from-sensor):

    x_B = R(q_BS) · x_S

Angular rates and field/force vectors are free vectors → rotation only.
For accelerometers mounted away from the body origin, the lever arm adds
rigid-body terms (provided here explicitly).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qnav.attitude import quaternion as quat

__all__ = ["SensorAlignment", "lever_arm_acceleration"]


@dataclass(frozen=True)
class SensorAlignment:
    """Fixed mounting of a sensor on the body: rotation ``q_BS`` and lever arm
    ``r_B`` (sensor position in body coordinates, meters)."""

    q_body_sensor: np.ndarray = field(default_factory=quat.identity)
    lever_arm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_body(self, v_sensor: np.ndarray) -> np.ndarray:
        """Rotate free vector(s) sensor → body: ``v_B = R(q_BS) v_S``."""
        return quat.rotate_vector(self.q_body_sensor, v_sensor)

    def to_sensor(self, v_body: np.ndarray) -> np.ndarray:
        """Rotate free vector(s) body → sensor."""
        return quat.rotate_frame(self.q_body_sensor, v_body)


def lever_arm_acceleration(
    omega_body: np.ndarray, alpha_body: np.ndarray, lever_arm: np.ndarray
) -> np.ndarray:
    """Extra acceleration sensed at a lever arm ``r`` (body frame):

    ``a_lever = α × r + ω × (ω × r)``

    Subtract this (after rotating the sensor output into the body frame) to
    refer an off-origin accelerometer to the body origin.
    """
    w = np.asarray(omega_body, dtype=float)
    al = np.asarray(alpha_body, dtype=float)
    r = np.asarray(lever_arm, dtype=float)
    return np.cross(al, r) + np.cross(w, np.cross(w, r))
