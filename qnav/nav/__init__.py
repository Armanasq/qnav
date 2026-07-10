"""Inertial navigation: canonical state, strapdown mechanization, 15-state ESKF.

- :class:`~qnav.nav.state.NavState` — attitude, velocity, position, IMU biases
  (NED geodetic or ECEF conventions, documented in :mod:`qnav.nav.state`)
- :mod:`qnav.nav.mechanization` — NED/ECEF strapdown kernels (Earth rate,
  transport rate, Coriolis, Somigliana gravity); the single source of
  propagation math for the navigation stack
- :mod:`qnav.nav.increments` — coning/sculling-corrected IMU increments
- :class:`~qnav.nav.eskf.NavEskf` — 15-state error-state Kalman filter
  ``[δθ, δv, δp, δbg, δba]`` sharing the gated update kernel and estimator
  lifecycle (snapshot/restore/reset/health) with :mod:`qnav.filters`
"""

from qnav.nav import eskf, increments, measurements, mechanization, state  # noqa: F401
from qnav.nav.eskf import NavEskf  # noqa: F401
from qnav.nav.increments import accumulate_increments  # noqa: F401
from qnav.nav.mechanization import (  # noqa: F401
    gravity_ecef,
    propagate_ecef,
    propagate_ned,
    propagate_state,
)
from qnav.nav.state import NavState  # noqa: F401

__all__ = [
    "NavEskf", "NavState", "accumulate_increments", "gravity_ecef",
    "propagate_ecef", "propagate_ned", "propagate_state",
    "eskf", "increments", "measurements", "mechanization", "state",
]
