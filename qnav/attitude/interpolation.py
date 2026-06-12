"""Attitude interpolation: slerp, normalized lerp, and piecewise trajectories.

All interpolators are **sign-safe**: the shorter geodesic is taken by flipping
the second quaternion when ``⟨q0, q1⟩ < 0`` (q ≡ −q double cover).
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat

__all__ = ["slerp", "nlerp", "slerp_series"]

_DOT_PARALLEL = 1.0 - 1e-12


def _align(q0: np.ndarray, q1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    return q0, np.where(d < 0, -q1, q1)


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation ``q(t) = q0 ⊗ Exp(t·Log(q0* ⊗ q1))``.

    Constant angular velocity along the geodesic; ``t`` may be scalar or
    batched (broadcast against the quaternion batch). For nearly parallel
    inputs falls back to normalized lerp (the geodesic formula degenerates).
    """
    q0, q1 = _align(q0, q1)
    t = np.asarray(t, dtype=float)
    d = np.clip(np.sum(quat.normalize(q0) * quat.normalize(q1), axis=-1), -1.0, 1.0)
    out = quat.mul(q0, quat.power(quat.relative(q0, q1), t))
    near = d > _DOT_PARALLEL
    if np.any(near):
        lerped = nlerp(q0, q1, t)
        out = np.where(np.broadcast_to(near[..., None], out.shape), lerped, out)
    return out


def nlerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Normalized linear interpolation (chord, then renormalize).

    Cheaper than slerp, not constant-rate; error is O(θ³) in the arc angle.
    """
    q0, q1 = _align(q0, q1)
    t = np.asarray(t, dtype=float)[..., None]
    return quat.normalize((1.0 - t) * q0 + t * q1)


def slerp_series(times: np.ndarray, qs: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Piecewise slerp through keyframes ``(times[i], qs[i])``.

    ``times`` must be strictly increasing; queries outside the range raise
    (no silent extrapolation). Shapes: times ``(N,)``, qs ``(N, 4)``,
    t_query ``(M,)`` → ``(M, 4)``.
    """
    times = np.asarray(times, dtype=float)
    qs = np.asarray(qs, dtype=float)
    t_query = np.atleast_1d(np.asarray(t_query, dtype=float))
    if times.ndim != 1 or qs.shape != (times.size, 4):
        raise ValueError("expected times (N,) and qs (N, 4)")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    if np.any(t_query < times[0]) or np.any(t_query > times[-1]):
        raise ValueError("query time outside keyframe range; extrapolation is not supported")
    idx = np.clip(np.searchsorted(times, t_query, side="right") - 1, 0, times.size - 2)
    t0, t1 = times[idx], times[idx + 1]
    u = (t_query - t0) / (t1 - t0)
    return slerp(qs[idx], qs[idx + 1], u)
