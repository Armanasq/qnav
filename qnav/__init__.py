"""qnav — convention-safe attitude, heading, frame-transform, and navigation math.

Subpackages
-----------
- :mod:`qnav.attitude` — representations, conversions, kinematics, covariance
- :mod:`qnav.frames` — typed frames, transform graph, Earth geodesy
- :mod:`qnav.heading` — tilt-compensated compass, declination, disturbance
- :mod:`qnav.sensors` — IMU/magnetometer models, noise, Allan variance
- :mod:`qnav.calibration` — gyro/accel/mag calibration
- :mod:`qnav.determination` — TRIAD, Davenport, QUEST, SVD (Wahba solvers)
- :mod:`qnav.filters` — complementary, Mahony-/Madgwick-style, EKF, ESKF
- :mod:`qnav.simulation` — rigid-body, trajectories, synthetic IMU/MARG
- :mod:`qnav.metrics` — attitude/heading error metrics, consistency (NEES)
- :mod:`qnav.validation` — invariants, reference cases, benchmark runner

Normative conventions: ``docs/conventions.md``.
"""

__version__ = "0.3.0"

from qnav import (  # noqa: F401
    attitude,
    calibration,
    determination,
    filters,
    frames,
    geomag,
    heading,
    highlevel,
    interop,
    metrics,
    nav,
    sensors,
    simulation,
    types,
    validation,
)
from qnav.highlevel import AttitudeEstimate, estimate_attitude  # noqa: F401

from qnav.errors import (  # noqa: F401
    CalibrationError,
    ConventionError,
    DegenerateGeometryWarning,
    FrameGraphError,
    FrameMismatchError,
    GimbalLockWarning,
    NormalizationWarning,
    QnavError,
    QnavWarning,
)

#: Supported public API: these subpackages and exceptions. Modules or symbols
#: prefixed with ``_`` are internal and may change without deprecation.
__all__ = [
    "__version__",
    "attitude", "calibration", "determination", "filters", "frames",
    "geomag", "heading", "highlevel", "interop", "metrics", "nav", "sensors",
    "simulation", "types", "validation",
    "AttitudeEstimate", "estimate_attitude",
    "QnavError", "QnavWarning", "CalibrationError", "ConventionError",
    "FrameGraphError", "FrameMismatchError", "DegenerateGeometryWarning",
    "GimbalLockWarning", "NormalizationWarning",
]
