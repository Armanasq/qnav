"""Heading subsystem: tilt-compensated compass, declination, disturbance gates.

Heading = clockwise-from-north angle in [0, 2π); see ``docs/conventions.md`` §7.
"""

from qnav.heading import compass, declination, disturbance, magnetic_model, tilt_compensation  # noqa: F401
from qnav.heading.compass import (  # noqa: F401
    heading_from_accel_mag,
    magnetic_heading,
    true_heading,
    wrap_heading,
)
from qnav.heading.tilt_compensation import roll_pitch_from_accel  # noqa: F401

__all__ = [
    "compass", "declination", "disturbance", "magnetic_model", "tilt_compensation",
    "heading_from_accel_mag", "magnetic_heading", "true_heading", "wrap_heading",
    "roll_pitch_from_accel",
]
