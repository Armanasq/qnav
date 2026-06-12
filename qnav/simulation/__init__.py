"""Ground-truth trajectory generation and synthetic IMU/MARG measurements."""

from qnav.simulation import (  # noqa: F401
    gravity,
    imu_synthesis,
    magnetic_field,
    noise_injection,
    rigid_body,
    trajectories,
    vehicle_state,
)
from qnav.simulation.gravity import ConstantGravity, WGS84Gravity  # noqa: F401
from qnav.simulation.imu_synthesis import ImuDataset, synthesize  # noqa: F401
from qnav.simulation.magnetic_field import MagneticEnvironment  # noqa: F401
from qnav.simulation.rigid_body import RigidBody  # noqa: F401
from qnav.simulation.trajectories import Trajectory, coning, constant_rate, sinusoidal_euler, static_pose  # noqa: F401
from qnav.simulation.vehicle_state import VehicleState  # noqa: F401

__all__ = [
    "ConstantGravity", "ImuDataset", "MagneticEnvironment", "RigidBody",
    "Trajectory", "VehicleState", "WGS84Gravity",
    "coning", "constant_rate", "sinusoidal_euler", "static_pose", "synthesize",
    "gravity", "imu_synthesis", "magnetic_field", "noise_injection",
    "rigid_body", "trajectories", "vehicle_state",
]
