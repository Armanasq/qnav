"""Temperature-dependent IMU bias calibration.

Fits ``b(T) = c0 + c1 (T − T_ref) + ... + ck (T − T_ref)^k`` per axis from
(temperature, bias) observations — e.g. static bias estimates collected
during a thermal soak. Returns the coefficients, their covariance (from the
LSQ normal equations and the residual variance), and an observability
report: without sufficient temperature *range*, higher-order coefficients
are unobservable and the fit refuses to activate them silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qnav._validate import ensure_finite
from qnav.calibration.observability import (
    Observability,
    ObservabilityReport,
    assess_least_squares,
)
from qnav.errors import CalibrationError

__all__ = ["TemperatureBiasModel", "fit_temperature_bias"]


@dataclass(frozen=True)
class TemperatureBiasModel:
    """Per-axis polynomial bias model ``b(T) = Σ coeffs[k] (T − T_ref)^k``."""

    coeffs: np.ndarray            #: (order+1, n_axes)
    t_ref: float
    covariance: np.ndarray        #: (order+1, order+1), shared across axes
    rms_residual: np.ndarray      #: per-axis [bias units]
    observability: ObservabilityReport

    def predict(self, temperature: float | np.ndarray) -> np.ndarray:
        """Bias at ``temperature`` [same units as the fit], shape (..., n_axes)."""
        dT = np.asarray(temperature, dtype=float) - self.t_ref
        powers = np.stack([dT**k for k in range(self.coeffs.shape[0])], axis=-1)
        return powers @ self.coeffs


def fit_temperature_bias(
    temperatures: np.ndarray,
    biases: np.ndarray,
    order: int = 1,
    t_ref: float | None = None,
) -> TemperatureBiasModel:
    """Fit the polynomial bias-vs-temperature model.

    ``temperatures``: (N,) [degC or K, caller's choice, used consistently];
    ``biases``: (N, n_axes). Raises :class:`CalibrationError` when the
    temperature excitation cannot support the requested order (rank-deficient
    Vandermonde) instead of returning garbage coefficients.
    """
    T = ensure_finite(temperatures, "temperatures")
    B = ensure_finite(biases, "biases")
    if T.ndim != 1:
        raise ValueError(f"temperatures must be 1-D, got shape {T.shape}")
    B = np.atleast_2d(B)
    if B.shape[0] != T.size:
        raise ValueError("temperatures and biases must have the same length")
    if order < 0:
        raise ValueError("order must be >= 0")
    if T.size < order + 2:
        raise CalibrationError(
            f"need at least order+2 = {order + 2} samples for order {order}")

    t0 = float(np.mean(T)) if t_ref is None else float(t_ref)
    dT = T - t0
    V = np.vander(dT, N=order + 1, increasing=True)   # (N, order+1)

    report = assess_least_squares(V)
    if report.status is Observability.UNOBSERVABLE:
        raise CalibrationError(
            f"temperature excitation (range {dT.max() - dT.min():.3g}) cannot "
            f"support a degree-{order} model; reduce the order or widen the sweep"
        )

    coeffs, residuals, rank, _ = np.linalg.lstsq(V, B, rcond=None)
    resid = B - V @ coeffs
    dof = max(T.size - (order + 1), 1)
    sigma2 = float(np.sum(resid**2) / (resid.shape[1] * dof))
    cov = sigma2 * np.linalg.inv(V.T @ V)
    rms = np.sqrt(np.mean(resid**2, axis=0))

    return TemperatureBiasModel(
        coeffs=coeffs, t_ref=t0, covariance=cov, rms_residual=rms,
        observability=report,
    )
