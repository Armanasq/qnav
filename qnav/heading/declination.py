"""Declination correction utilities.

Declination D is the angle from true north to magnetic north, **positive
east** (NOAA/WMM convention): ``ψ_true = ψ_mag + D``. qnav takes D as user
input (from a WMM/IGRF service or chart); see
:mod:`qnav.heading.magnetic_model` for why coefficient tables are not bundled.
"""

from __future__ import annotations

import numpy as np

from qnav.heading.compass import heading_difference, wrap_heading

__all__ = ["apply_declination", "remove_declination", "grid_convergence_correction"]


def apply_declination(psi_mag: np.ndarray, declination: np.ndarray) -> np.ndarray:
    """Magnetic → true heading: ``ψ_true = wrap(ψ_mag + D)`` (D positive east)."""
    return wrap_heading(np.asarray(psi_mag, dtype=float) + np.asarray(declination, dtype=float))


def remove_declination(psi_true: np.ndarray, declination: np.ndarray) -> np.ndarray:
    """True → magnetic heading: ``ψ_mag = wrap(ψ_true − D)``."""
    return wrap_heading(np.asarray(psi_true, dtype=float) - np.asarray(declination, dtype=float))


def grid_convergence_correction(psi_true: np.ndarray, convergence: np.ndarray) -> np.ndarray:
    """True → grid heading: ``ψ_grid = wrap(ψ_true − γ)`` with grid convergence γ
    (positive when grid north is east of true north)."""
    return wrap_heading(heading_difference(psi_true, convergence))
