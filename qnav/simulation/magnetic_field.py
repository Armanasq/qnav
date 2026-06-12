"""Magnetic-field environment for simulation.

Wraps :mod:`qnav.heading.magnetic_model` with optional localized disturbance
injection, so estimator robustness to magnetic anomalies can be tested
deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from qnav.heading.magnetic_model import field_from_elements

__all__ = ["MagneticEnvironment"]


@dataclass(frozen=True)
class MagneticEnvironment:
    """Local field defined by (declination, inclination, intensity) plus an
    optional time-dependent disturbance ``disturbance_fn(t) -> (3,)`` added in
    the navigation frame."""

    declination: float = 0.0
    inclination: float = np.deg2rad(60.0)
    intensity: float = 50e-6  # [T]
    nav_frame: str = "NED"
    disturbance_fn: Optional[Callable[[float], np.ndarray]] = None

    def field_nav(self, t: np.ndarray | float = 0.0) -> np.ndarray:
        """Field vector(s) in the navigation frame at time(s) ``t``."""
        base = field_from_elements(
            self.declination, self.inclination, self.intensity, frame=self.nav_frame
        )
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        if self.disturbance_fn is None:
            out = np.broadcast_to(base, t_arr.shape + (3,)).copy()
        else:
            out = np.stack([base + np.asarray(self.disturbance_fn(tk), dtype=float)
                            for tk in t_arr])
        return out[0] if np.isscalar(t) or np.ndim(t) == 0 else out
