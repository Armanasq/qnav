"""Geomagnetic reference models (WMM spherical-harmonic synthesis)."""

from qnav.geomag.wmm import MagneticFieldEstimate, wmm_elements, wmm_field

__all__ = ["MagneticFieldEstimate", "wmm_field", "wmm_elements"]
