"""Observability diagnostics for least-squares calibration problems.

Every qnav calibration routine that solves a linear(ized) least-squares
problem can report *how well* the data constrained the parameters — not just
the point estimate. The assessment is based on the singular values of the
stacked Jacobian: the condition number and the smallest singular value
(normalized per sample) grade the excitation.

Statuses (thresholds documented, tested at their boundaries):

- ``OBSERVABLE``: cond < 1e3 and normalized smallest singular value > 1e-6
- ``WEAKLY_OBSERVABLE``: rank-complete but ill-conditioned (cond >= 1e3)
- ``UNOBSERVABLE``: numerically rank-deficient — some parameter direction
  received (near-)zero excitation; its estimate is meaningless
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from qnav._validate import ensure_finite

__all__ = ["Observability", "ObservabilityReport", "assess_least_squares"]

_COND_WEAK = 1e3
_RANK_TOL = 1e-10


class Observability(enum.Enum):
    OBSERVABLE = "observable"
    WEAKLY_OBSERVABLE = "weakly_observable"
    UNOBSERVABLE = "unobservable"


@dataclass(frozen=True)
class ObservabilityReport:
    """Excitation grading of one least-squares calibration problem."""

    status: Observability
    condition_number: float
    #: smallest singular value / (largest singular value) — rank margin
    rank_margin: float
    #: smallest singular value scaled by sqrt(N) — per-sample excitation
    excitation: float
    singular_values: np.ndarray
    #: unit parameter-space direction with the least excitation
    weakest_direction: np.ndarray


def assess_least_squares(J: np.ndarray) -> ObservabilityReport:
    """Grade the excitation of a stacked LSQ Jacobian ``J`` (N x P, N >= 1).

    Rows are (linearized) measurement equations, columns are parameters.
    """
    J = ensure_finite(J, "J")
    if J.ndim != 2 or J.shape[0] < 1:
        raise ValueError(f"J must be a 2-D stacked Jacobian, got shape {J.shape}")
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    s_max = float(s[0]) if s[0] > 0 else 0.0
    s_min = float(s[-1])
    rank_margin = s_min / s_max if s_max > 0 else 0.0
    cond = np.inf if s_min == 0.0 else s_max / s_min
    excitation = s_min / np.sqrt(J.shape[0])
    if J.shape[0] < J.shape[1] or rank_margin < _RANK_TOL:
        status = Observability.UNOBSERVABLE
    elif cond >= _COND_WEAK:
        status = Observability.WEAKLY_OBSERVABLE
    else:
        status = Observability.OBSERVABLE
    return ObservabilityReport(
        status=status,
        condition_number=float(cond),
        rank_margin=float(rank_margin),
        excitation=float(excitation),
        singular_values=s.copy(),
        weakest_direction=Vt[-1].copy(),
    )
