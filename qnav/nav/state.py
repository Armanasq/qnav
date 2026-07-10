"""Canonical inertial navigation state.

Frames
------
``frame="NED"``: attitude ``q`` is ``q_nb`` (NED <- body), velocity ``v`` is
NED [m/s], position ``p`` is geodetic ``[lat, lon, h]`` [rad, rad, m] on the
WGS-84 ellipsoid.

``frame="ECEF"``: attitude ``q`` is ``q_eb`` (ECEF <- body), velocity ``v``
is ECEF [m/s], position ``p`` is ``r_ecef`` [m].

Biases ``bg`` [rad/s] and ``ba`` [m/s²] are body-frame additive IMU errors:
``omega_meas = omega + bg + noise``, ``f_meas = f + ba + noise``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from qnav._validate import ensure_finite, ensure_unit_quaternion, ensure_vector3

__all__ = ["NavState"]

_FRAMES = ("NED", "ECEF")


@dataclass(frozen=True)
class NavState:
    """Immutable navigation state; use :meth:`evolve` to derive new states."""

    q: np.ndarray
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame: str = "NED"

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", ensure_unit_quaternion(self.q, "q").copy())
        for name in ("v", "p", "bg", "ba"):
            object.__setattr__(self, name, ensure_vector3(getattr(self, name), name).copy())
        ensure_finite(self.p, "p")
        if self.frame not in _FRAMES:
            raise ValueError(f"frame must be one of {_FRAMES}, got {self.frame!r}")
        if self.frame == "NED" and abs(float(self.p[0])) > np.pi / 2 + 1e-9:
            raise ValueError(f"latitude {float(self.p[0])} outside [-pi/2, pi/2] rad "
                             "(NED position is [lat, lon, h] in radians/meters)")

    def evolve(self, **changes: object) -> "NavState":
        """A copy with the given fields replaced (validation re-runs)."""
        return replace(self, **changes)  # type: ignore[arg-type]
