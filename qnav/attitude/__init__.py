"""Attitude representations, conversions, kinematics, and uncertainty.

Conventions (normative: ``docs/conventions.md``): Hamilton quaternions,
scalar-first ``[w, x, y, z]``; ``q_AB``/``R_AB`` maps coordinates B → A;
right/local tangent perturbations; radians everywhere.
"""

from qnav.attitude import (  # noqa: F401
    conversions,
    covariance,
    dcm,
    euler,
    interpolation,
    jacobians,
    kinematics,
    mrp,
    quaternion,
    rotvec,
    so3,
)

__all__ = [
    "conversions", "covariance", "dcm", "euler", "interpolation",
    "jacobians", "kinematics", "mrp", "quaternion", "rotvec", "so3",
]
