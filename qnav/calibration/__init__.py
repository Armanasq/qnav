"""Sensor calibration: gyro bias, accelerometer, magnetometer, alignment,
lever arms, time offsets, temperature models — with observability reporting."""

from qnav.calibration import (  # noqa: F401
    accel_calibration,
    frame_alignment,
    gyro_bias,
    lever_arm,
    mag_ellipsoid,
    observability,
    soft_hard_iron,
    temperature,
    time_offset,
)
from qnav.calibration.accel_calibration import calibrate_accelerometer  # noqa: F401
from qnav.calibration.frame_alignment import align_from_vector_pairs  # noqa: F401
from qnav.calibration.gyro_bias import detect_static_intervals, estimate_bias  # noqa: F401
from qnav.calibration.lever_arm import LeverArmEstimate, estimate_lever_arm  # noqa: F401
from qnav.calibration.mag_ellipsoid import MagCalibration, fit_ellipsoid  # noqa: F401
from qnav.calibration.observability import (  # noqa: F401
    Observability,
    ObservabilityReport,
    assess_least_squares,
)
from qnav.calibration.temperature import (  # noqa: F401
    TemperatureBiasModel,
    fit_temperature_bias,
)
from qnav.calibration.time_offset import TimeOffsetEstimate, estimate_time_offset  # noqa: F401

__all__ = [
    "LeverArmEstimate", "MagCalibration", "Observability",
    "ObservabilityReport", "TemperatureBiasModel", "TimeOffsetEstimate",
    "align_from_vector_pairs", "assess_least_squares",
    "calibrate_accelerometer", "detect_static_intervals", "estimate_bias",
    "estimate_lever_arm", "estimate_time_offset", "fit_ellipsoid",
    "fit_temperature_bias",
    "accel_calibration", "frame_alignment", "gyro_bias", "lever_arm",
    "mag_ellipsoid", "observability", "soft_hard_iron", "temperature",
    "time_offset",
]
