"""Rotation vector (axis–angle as ``φ = θ·u ∈ ℝ³``) and axis–angle utilities.

The rotation vector is the SO(3) tangent representation: ``R = Exp(φ)``,
``q = Exp_q(φ)``. Principal domain ``θ = ‖φ‖ ∈ [0, π]`` is returned by all
logarithms; inputs of any magnitude are accepted.

References: Solà §"Rotation vector"; Hashim SO(3) survey. See
``docs/math/so3.md``.
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.attitude import so3

__all__ = [
    "to_quaternion", "from_quaternion", "to_dcm", "from_dcm",
    "to_axis_angle", "from_axis_angle", "wrap",
]


def to_quaternion(phi: np.ndarray) -> np.ndarray:
    """``q = [cos(θ/2), sin(θ/2) u]`` — the quaternion exponential."""
    return quat.exp(phi)


def from_quaternion(q: np.ndarray) -> np.ndarray:
    """Principal rotation vector of a unit quaternion (θ ∈ [0, π])."""
    return quat.log(q)


def to_dcm(phi: np.ndarray) -> np.ndarray:
    """Rodrigues formula ``R = Exp(φ)``."""
    return so3.exp(phi)


def from_dcm(R: np.ndarray) -> np.ndarray:
    """Principal rotation vector of a DCM (θ ∈ [0, π])."""
    return so3.log(R)


def to_axis_angle(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``φ`` into unit axis and angle; identity maps to (``[1,0,0]``, 0)."""
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi, axis=-1)
    safe = theta > 1e-12
    axis = np.where(
        safe[..., None],
        phi / np.where(safe, theta, 1.0)[..., None],
        np.broadcast_to([1.0, 0.0, 0.0], phi.shape),
    )
    return axis, theta


def from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """``φ = angle · axis`` (axis is normalized defensively; zero axis with
    nonzero angle raises)."""
    axis = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)
    n = np.linalg.norm(axis, axis=-1)
    if np.any((n < 1e-12) & (np.abs(angle) > 0)):
        raise ValueError("zero axis with nonzero angle is undefined")
    n = np.where(n < 1e-12, 1.0, n)
    return (axis / n[..., None]) * angle[..., None]


def wrap(phi: np.ndarray) -> np.ndarray:
    """Wrap a rotation vector to the principal domain θ ∈ [0, π].

    ``θ' = θ mod 2π`` folded into [0, π] by flipping the axis when θ' > π.
    """
    axis, theta = to_axis_angle(phi)
    t = np.mod(theta, 2.0 * np.pi)
    flip = t > np.pi
    t = np.where(flip, 2.0 * np.pi - t, t)
    axis = np.where(flip[..., None], -axis, axis)
    return axis * t[..., None]
