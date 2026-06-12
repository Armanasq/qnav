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

__version__ = "0.1.0"

from qnav import attitude, frames  # noqa: F401
from qnav.errors import (  # noqa: F401
    ConventionError,
    DegenerateGeometryWarning,
    FrameGraphError,
    FrameMismatchError,
    GimbalLockWarning,
    NormalizationWarning,
    QnavError,
    QnavWarning,
)
