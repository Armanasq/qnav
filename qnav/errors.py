"""Typed exceptions and warnings used across qnav.

Numerical degradation is reported with warnings (computation continues with a
documented, deterministic fallback); structural misuse (frame mismatches,
invalid sequences) raises exceptions.
"""

from __future__ import annotations


class QnavError(Exception):
    """Base class for all qnav exceptions."""


class FrameMismatchError(QnavError):
    """A transform was composed or applied with non-matching frames."""


class FrameGraphError(QnavError):
    """A frame-graph lookup failed (unknown frame or no path)."""


class ConventionError(QnavError):
    """An invalid convention token was supplied (Euler sequence, frame name, ...)."""


class CalibrationError(QnavError):
    """A calibration problem is infeasible or numerically degenerate."""


class QnavWarning(UserWarning):
    """Base class for all qnav warnings."""


class GimbalLockWarning(QnavWarning):
    """Euler extraction hit a gimbal-lock region; third angle was set to zero."""


class DegenerateGeometryWarning(QnavWarning):
    """Input geometry is (near-)degenerate; result follows the documented fallback."""


class NormalizationWarning(QnavWarning):
    """An input deviated from unit norm beyond tolerance and was renormalized."""
