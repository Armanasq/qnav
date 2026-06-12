"""Filter-consistency metrics: NEES and innovation tests.

A covariance-bearing estimator is *consistent* when its reported uncertainty
matches its realized error. The normalized estimation error squared

    NEES_k = e_kᵀ P_k⁻¹ e_k

is χ²(dim e)-distributed for a consistent Gaussian filter; the time-averaged
NEES over N Monte-Carlo runs falls in the corresponding χ² interval.

Reference: standard estimation practice (Bar-Shalom-style consistency
testing), applied to the local attitude error of ``docs/conventions.md`` §5.
"""

from __future__ import annotations

import numpy as np

__all__ = ["nees", "average_nees", "nees_bounds"]


def nees(errors: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """Per-sample NEES for errors ``(N, d)`` and covariances ``(N, d, d)``."""
    e = np.asarray(errors, dtype=float)
    P = np.asarray(covariances, dtype=float)
    sol = np.linalg.solve(P, e[..., None])[..., 0]
    return np.sum(e * sol, axis=-1)


def average_nees(errors: np.ndarray, covariances: np.ndarray) -> float:
    """Time-averaged NEES (should be ≈ d for a consistent filter)."""
    return float(np.mean(nees(errors, covariances)))


def nees_bounds(dim: int, n_samples: int, confidence: float = 0.95):
    """Two-sided χ² acceptance interval for the *average* NEES.

    Uses the Wilson–Hilferty cube approximation of χ² quantiles (keeps qnav
    SciPy-free); accurate to ~1% for n·d ≥ 10. Returns ``(lo, hi)`` for the
    average NEES of ``n_samples`` independent d-dimensional errors.
    """
    k = dim * n_samples
    # standard normal quantile via Acklam-style rational approximation
    z = _norm_ppf(0.5 + confidence / 2.0)
    h = 2.0 / (9.0 * k)
    hi = k * (1.0 - h + z * np.sqrt(h)) ** 3
    lo = k * (1.0 - h - z * np.sqrt(h)) ** 3
    return lo / n_samples, hi / n_samples


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, ~1e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        ql = np.sqrt(-2 * np.log(p))
        return (((((c[0] * ql + c[1]) * ql + c[2]) * ql + c[3]) * ql + c[4]) * ql + c[5]) / \
               ((((d[0] * ql + d[1]) * ql + d[2]) * ql + d[3]) * ql + 1)
    if p > phigh:
        ql = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * ql + c[1]) * ql + c[2]) * ql + c[3]) * ql + c[4]) * ql + c[5]) / \
                ((((d[0] * ql + d[1]) * ql + d[2]) * ql + d[3]) * ql + 1)
    qm = p - 0.5
    r = qm * qm
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * qm / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
