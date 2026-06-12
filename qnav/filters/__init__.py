"""Attitude estimators (stepwise: ``predict``/``update`` per sample).

Selection guide
---------------
Covariance-bearing filters (statistical uncertainty output):

- :class:`~qnav.filters.eskf.Eskf` — error-state KF (attitude + gyro bias,
  6×6 covariance, right/local error). **Recommended default** when
  uncertainty matters; the only filter with NEES-verified consistency.
- :class:`~qnav.filters.ukf.UkfAttitude` — unscented filter on the SO(3)
  tangent (USQUE-style). No linearization in the update; prefer it over the
  ESKF when initial uncertainty exceeds ~20°.
- :class:`~qnav.filters.fkf.FastKalmanFilter` — linear 4-state quaternion KF
  with closed-form attitude measurements; cheapest covariance-bearing option.
- :class:`~qnav.filters.ekf.QuaternionEkf` — total-state quaternion EKF,
  kept as a documented reference/baseline.

Complementary/observer filters (no covariance, minimal tuning):

- :class:`~qnav.filters.complementary.ComplementaryFilter` — simplest;
  geodesic blend of gyro propagation with an accel/mag fix.
- :class:`~qnav.filters.mahony_style.MahonyFilter` — nonlinear complementary
  filter on SO(3); integral gyro-bias estimation; robust default.
- :class:`~qnav.filters.madgwick_style.MadgwickStyleFilter` —
  gradient-descent fusion; analytic general-direction gradient.
- :class:`~qnav.filters.aqua.AquaFilter` — algebraic corrections with
  structurally decoupled tilt/yaw; magnetic disturbances cannot touch
  roll/pitch.
- :class:`~qnav.filters.fourati.FouratiFilter` — Levenberg-Marquardt
  alignment observer; observability-scaled corrections.
- :class:`~qnav.filters.roleq.RoleqFilter` — recursive linear quaternion
  estimator (one OLEQ fixed-point iteration per sample).
"""

from qnav.filters import (  # noqa: F401
    aqua, base, complementary, ekf, eskf, fkf, fourati, madgwick_style,
    mahony_style, roleq, ukf,
)
from qnav.filters import mahony_style as nonlinear_complementary  # noqa: F401
from qnav.filters.aqua import AquaFilter  # noqa: F401
from qnav.filters.base import AttitudeFilter  # noqa: F401
from qnav.filters.complementary import ComplementaryFilter  # noqa: F401
from qnav.filters.ekf import QuaternionEkf  # noqa: F401
from qnav.filters.eskf import Eskf  # noqa: F401
from qnav.filters.fkf import FastKalmanFilter  # noqa: F401
from qnav.filters.fourati import FouratiFilter  # noqa: F401
from qnav.filters.madgwick_style import MadgwickStyleFilter  # noqa: F401
from qnav.filters.mahony_style import MahonyFilter, NonlinearComplementaryFilter  # noqa: F401
from qnav.filters.roleq import RoleqFilter  # noqa: F401
from qnav.filters.ukf import UkfAttitude  # noqa: F401

__all__ = [
    "AquaFilter", "AttitudeFilter", "ComplementaryFilter", "Eskf",
    "FastKalmanFilter", "FouratiFilter", "MadgwickStyleFilter",
    "MahonyFilter", "NonlinearComplementaryFilter", "QuaternionEkf",
    "RoleqFilter", "UkfAttitude",
    "aqua", "base", "complementary", "ekf", "eskf", "fkf", "fourati",
    "madgwick_style", "mahony_style", "nonlinear_complementary", "roleq",
    "ukf",
]
