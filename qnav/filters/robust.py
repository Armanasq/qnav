"""Robust measurement handling: chi-square gating, robust losses, sensor
quarantine.

All policies operate on the normalized innovation squared (NIS)
``νᵀ S⁻¹ ν`` and its square root, the Mahalanobis distance ``r``. For a
consistent filter, NIS is chi-square distributed with ``dim(ν)`` degrees of
freedom; a measurement whose NIS exceeds the chi-square quantile at the
configured confidence is an outlier candidate.

Robust weights follow the M-estimator convention: the effective measurement
noise is inflated by ``1/w`` so ``w = 1`` is a plain Kalman update and
``w → 0`` discards the measurement smoothly. Default tuning constants are
the standard 95%-efficiency values (Huber 1.345, Cauchy 2.385, Tukey 4.685)
for a unit-variance residual.

SciPy-free: chi-square quantiles use the Wilson–Hilferty cube approximation
(accurate to ~1% for dof >= 3 at the confidences used for gating), sharing
the normal quantile in :mod:`qnav.metrics.covariance_consistency`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from qnav._validate import ensure_positive
from qnav.metrics.covariance_consistency import _norm_ppf

__all__ = [
    "GatePolicy",
    "SensorMonitor",
    "cauchy_weight",
    "chi2_quantile",
    "detect_saturation",
    "huber_weight",
    "tukey_weight",
]


def chi2_quantile(dof: int, p: float) -> float:
    """Chi-square quantile via the Wilson–Hilferty approximation.

    Exact for ``dof`` 1 and 2 (closed forms); Wilson–Hilferty otherwise,
    accurate to ~1% for ``dof >= 3`` at gating confidences (0.9–0.999).
    Accuracy is pinned against reference values in ``tests/test_robust.py``.
    """
    if dof < 1:
        raise ValueError(f"dof must be >= 1, got {dof}")
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if dof == 1:  # exact: chi2(1) = Z², so Q(p) = z((1+p)/2)²
        return float(_norm_ppf(0.5 + p / 2.0) ** 2)
    if dof == 2:  # exact: chi2(2) is Exp(1/2), Q(p) = −2 ln(1−p)
        return float(-2.0 * np.log1p(-p))
    z = _norm_ppf(p)
    h = 2.0 / (9.0 * dof)
    return dof * (1.0 - h + z * np.sqrt(h)) ** 3


def huber_weight(r: float, k: float = 1.345) -> float:
    """Huber weight: 1 inside ``k``, ``k/r`` outside (linear tail influence)."""
    r = abs(float(r))
    return 1.0 if r <= k else k / r


def cauchy_weight(r: float, k: float = 2.385) -> float:
    """Cauchy weight ``1 / (1 + (r/k)²)`` (redescending influence, never 0)."""
    return 1.0 / (1.0 + (float(r) / k) ** 2)


def tukey_weight(r: float, k: float = 4.685) -> float:
    """Tukey biweight ``(1 − (r/k)²)²`` inside ``k``, exactly 0 outside."""
    r = abs(float(r))
    if r >= k:
        return 0.0
    u = 1.0 - (r / k) ** 2
    return u * u


_LOSSES = {"huber": huber_weight, "cauchy": cauchy_weight, "tukey": tukey_weight}

_DEFAULT_SCALES = {"huber": 1.345, "cauchy": 2.385, "tukey": 4.685}


@dataclass(frozen=True)
class GatePolicy:
    """How an estimator treats each measurement's innovation.

    Parameters
    ----------
    confidence:
        Chi-square gate confidence; NIS above ``chi2_quantile(dim, confidence)``
        triggers ``on_gate``.
    on_gate:
        ``"reject"`` — hard rejection, state untouched; ``"inflate"`` — soft:
        the measurement noise is scaled by ``nis/threshold`` so the update is
        applied with proportionally reduced trust.
    loss:
        Robust loss applied to *every* accepted measurement (including
        inflated ones), as a multiplicative noise inflation ``1/w`` with
        ``w = loss(mahalanobis_distance / sqrt(dim))`` normalized per axis.
    loss_scale:
        Tuning constant; ``None`` selects the standard 95%-efficiency value.
    """

    confidence: float = 0.997
    on_gate: Literal["reject", "inflate"] = "reject"
    loss: Literal["none", "huber", "cauchy", "tukey"] = "none"
    loss_scale: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")
        if self.on_gate not in ("reject", "inflate"):
            raise ValueError(f"on_gate must be 'reject' or 'inflate', got {self.on_gate!r}")
        if self.loss != "none" and self.loss not in _LOSSES:
            raise ValueError(f"loss must be 'none' or one of {tuple(_LOSSES)}, got {self.loss!r}")
        if self.loss_scale is not None:
            ensure_positive(self.loss_scale, "loss_scale")

    def threshold(self, dim: int) -> float:
        """Chi-square gate threshold for a ``dim``-dimensional innovation."""
        return chi2_quantile(dim, self.confidence)

    def robust_weight(self, nis: float, dim: int) -> float:
        """Robust weight in (0, 1] from the per-axis Mahalanobis distance."""
        if self.loss == "none":
            return 1.0
        k = self.loss_scale if self.loss_scale is not None else _DEFAULT_SCALES[self.loss]
        r = np.sqrt(max(float(nis), 0.0) / dim)
        return float(_LOSSES[self.loss](r, k))


class SensorMonitor:
    """Per-sensor quarantine with hysteresis recovery and timeout detection.

    A sensor is quarantined after ``quarantine_after`` consecutive gate
    rejections. While quarantined its measurements are still *evaluated*
    (innovation, NIS) but not fused; ``recover_after`` consecutive in-gate
    evaluations release it. ``timeout`` (seconds, optional) flags a sensor
    whose last sample is older than the given age — detection only; the
    caller decides the response.
    """

    def __init__(
        self,
        quarantine_after: int = 5,
        recover_after: int = 3,
        timeout: Optional[float] = None,
    ) -> None:
        if quarantine_after < 1 or recover_after < 1:
            raise ValueError("quarantine_after and recover_after must be >= 1")
        self.quarantine_after = int(quarantine_after)
        self.recover_after = int(recover_after)
        self.timeout = None if timeout is None else ensure_positive(timeout, "timeout")
        self.quarantined = False
        self._consecutive_rejected = 0
        self._consecutive_ok = 0
        self.last_seen: Optional[float] = None

    def note_measurement(self, in_gate: bool, timestamp: Optional[float] = None) -> bool:
        """Record one gate evaluation; returns True if fusion is allowed *now*.

        Call before fusing: a True return means the sensor is trusted for
        this measurement (and the measurement itself passed the gate).
        """
        if timestamp is not None:
            self.last_seen = float(timestamp)
        if self.quarantined:
            if in_gate:
                self._consecutive_ok += 1
                if self._consecutive_ok >= self.recover_after:
                    self.quarantined = False
                    self._consecutive_ok = 0
                    self._consecutive_rejected = 0
                    return True
            else:
                self._consecutive_ok = 0
            return False
        if in_gate:
            self._consecutive_rejected = 0
            return True
        self._consecutive_rejected += 1
        if self._consecutive_rejected >= self.quarantine_after:
            self.quarantined = True
            self._consecutive_ok = 0
        return False

    def timed_out(self, now: float) -> bool:
        """True when a timeout is configured and the sensor is silent too long."""
        if self.timeout is None or self.last_seen is None:
            return False
        return (float(now) - self.last_seen) > self.timeout


def detect_saturation(
    x: np.ndarray, full_scale: float, margin: float = 0.02
) -> np.ndarray:
    """Boolean mask of samples within ``margin`` (fraction) of ±``full_scale``.

    Saturated samples carry clipped, non-Gaussian errors; callers should
    reject or de-weight them before fusion.
    """
    fs = ensure_positive(full_scale, "full_scale")
    a = np.abs(np.asarray(x, dtype=float))
    return np.any(a >= fs * (1.0 - margin), axis=-1) if a.ndim > 0 else a >= fs * (1.0 - margin)
