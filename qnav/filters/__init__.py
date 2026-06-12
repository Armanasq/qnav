"""Attitude estimators (stepwise: ``predict``/``update`` per sample).

Selection guide
---------------
- :class:`~qnav.filters.complementary.ComplementaryFilter` — simplest; geodesic
  blend of gyro propagation with an accel/mag fix.
- :class:`~qnav.filters.mahony_style.MahonyFilter` — nonlinear complementary
  filter on SO(3); gyro-bias estimation; robust default without covariance.
- :class:`~qnav.filters.madgwick_style.MadgwickStyleFilter` — gradient-descent
  fusion; analytic general-direction gradient.
- :class:`~qnav.filters.eskf.Eskf` — error-state KF (attitude + gyro bias,
  6×6 covariance, local error). **Recommended** when uncertainty matters.
- :class:`~qnav.filters.ekf.QuaternionEkf` — total-state quaternion EKF, kept
  as a documented reference/baseline.

A UKF is deliberately not included in v0.1: for the attitude problem with
analytic Jacobians available throughout, the ESKF is the better-understood,
cheaper baseline; a UKF will be added only with a benchmark demonstrating a
concrete accuracy/consistency win (see ``docs/design/api_principles.md``).
"""

from qnav.filters import base, complementary, ekf, eskf, madgwick_style, mahony_style  # noqa: F401
from qnav.filters import mahony_style as nonlinear_complementary  # noqa: F401
from qnav.filters.base import AttitudeFilter  # noqa: F401
from qnav.filters.complementary import ComplementaryFilter  # noqa: F401
from qnav.filters.ekf import QuaternionEkf  # noqa: F401
from qnav.filters.eskf import Eskf  # noqa: F401
from qnav.filters.madgwick_style import MadgwickStyleFilter  # noqa: F401
from qnav.filters.mahony_style import MahonyFilter, NonlinearComplementaryFilter  # noqa: F401

__all__ = [
    "AttitudeFilter", "ComplementaryFilter", "Eskf", "MadgwickStyleFilter",
    "MahonyFilter", "NonlinearComplementaryFilter", "QuaternionEkf",
    "base", "complementary", "ekf", "eskf", "madgwick_style",
    "mahony_style", "nonlinear_complementary",
]
