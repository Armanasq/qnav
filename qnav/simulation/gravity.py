"""Gravity model abstraction for simulation.

Default: WGS-84 normal gravity at a declared latitude/height
(:func:`qnav.frames.earth.normal_gravity`); a simple constant-g model is
provided for unit tests and quick studies. Sign convention per
``docs/conventions.md`` §7: ``g_NED = [0, 0, +g]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnav.errors import ConventionError
from qnav.frames import earth

__all__ = ["GravityModel", "ConstantGravity", "WGS84Gravity", "STANDARD_G"]

STANDARD_G = 9.80665  # standard gravity [m/s²]


class GravityModel:
    """Interface: ``vector(frame) -> (3,) gravity vector`` in NED or ENU."""

    def vector(self, frame: str = "NED") -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantGravity(GravityModel):
    """Uniform gravity of magnitude ``g`` (default standard gravity)."""

    g: float = STANDARD_G

    def vector(self, frame: str = "NED") -> np.ndarray:
        if frame == "NED":
            return np.array([0.0, 0.0, self.g])
        if frame == "ENU":
            return np.array([0.0, 0.0, -self.g])
        raise ConventionError(f"frame must be 'NED' or 'ENU', got {frame!r}")


@dataclass(frozen=True)
class WGS84Gravity(GravityModel):
    """Somigliana normal gravity at geodetic latitude [rad] and height [m]."""

    latitude: float
    height: float = 0.0

    def vector(self, frame: str = "NED") -> np.ndarray:
        g = float(earth.normal_gravity(self.latitude, self.height))
        if frame == "NED":
            return np.array([0.0, 0.0, g])
        if frame == "ENU":
            return np.array([0.0, 0.0, -g])
        raise ConventionError(f"frame must be 'NED' or 'ENU', got {frame!r}")
