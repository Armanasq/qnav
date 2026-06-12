"""Allan variance/deviation for inertial-sensor noise identification.

The overlapping Allan variance at averaging time ``τ = m·dt`` of a rate
signal Ω is (IEEE-STD-952; Kok–Hol–Schön tutorial):

    σ²(τ) = 1/(2(N−2m)) Σ_k (θ_{k+2m} − 2θ_{k+m} + θ_k)² / τ²

with ``θ`` the cumulative integral of Ω. Slope landmarks on the log-log
deviation plot: white noise (angle random walk) slope −1/2 with
``N = σ(1)`` read at τ = 1 s; bias instability is the flat minimum
(``B ≈ σ_min/0.664``); rate random walk slope +1/2.
"""

from __future__ import annotations

import numpy as np

__all__ = ["allan_variance", "allan_deviation", "identify_noise"]


def allan_variance(
    x: np.ndarray, dt: float, m_list: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan variance of a rate signal ``x`` ((N,) or (N, k)).

    Returns ``(taus, avar)``; ``avar`` has shape ``(len(taus),) + x.shape[1:]``.
    ``m_list`` are the averaging window lengths (default: ~60 log-spaced).
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N < 9:
        raise ValueError("need at least 9 samples for Allan variance")
    if m_list is None:
        m_list = np.unique(
            np.logspace(0, np.log10((N - 1) // 2), 60).astype(int)
        )
    m_list = np.asarray(m_list, dtype=int)
    if np.any(m_list < 1) or np.any(2 * m_list >= N):
        raise ValueError("window lengths must satisfy 1 <= m < N/2")
    theta = np.cumsum(x, axis=0) * dt
    taus = m_list * dt
    out = np.empty((m_list.size,) + x.shape[1:])
    for i, m in enumerate(m_list):
        d = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        out[i] = np.sum(d * d, axis=0) / (2.0 * (N - 2 * m) * (m * dt) ** 2)
    return taus, out


def allan_deviation(
    x: np.ndarray, dt: float, m_list: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(taus, sqrt(avar))`` — see :func:`allan_variance`."""
    taus, av = allan_variance(x, dt, m_list)
    return taus, np.sqrt(av)


def identify_noise(taus: np.ndarray, adev: np.ndarray) -> dict:
    """Estimate noise density (ARW) and bias instability from an Allan curve.

    - ``density``: least-squares fit of the slope −1/2 segment (τ ≤ τ_min/3)
      evaluated at τ = 1 s: ``N = σ(τ)·√τ`` averaged over the segment.
    - ``bias_instability``: ``min σ / 0.664`` (1st-order GM approximation).
    - ``tau_min``: τ at the deviation minimum.

    Heuristic identification for 1-D curves; inspect the plot for real work.
    """
    taus = np.asarray(taus, dtype=float)
    adev = np.asarray(adev, dtype=float)
    if adev.ndim != 1:
        raise ValueError("identify_noise expects a single-axis Allan curve")
    i_min = int(np.argmin(adev))
    tau_min = taus[i_min]
    seg = taus <= max(tau_min / 3.0, taus[0])
    density = float(np.mean(adev[seg] * np.sqrt(taus[seg])))
    return {
        "density": density,
        "bias_instability": float(adev[i_min] / 0.664),
        "tau_min": float(tau_min),
    }
