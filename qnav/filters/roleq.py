"""ROLEQ: Recursive Optimal Linear Estimator of Quaternion (Zhou et al. 2018).

Turns the batch OLEQ solver into a per-sample filter: each vector-observation
pair contributes a symmetric involution ``Wᵢ`` whose +1-eigenspace contains
the true quaternion; one fixed-point iteration of the weighted average
``q ← normalize(½(I + Σ wᵢ Wᵢ) q)`` per sample, seeded by gyro propagation,
converges to the OLEQ optimum while tracking.

Construction of W (derivation, not transcription): the alignment constraint
``[0, v_ref] ⊗ q = q ⊗ [0, v_body]`` right-multiplied by ``[0, v_body]``
gives ``L([0,v_ref]) R([0,v_body]) q = −q`` for unit vectors, i.e.

    W = −L([0, v_ref]) · R([0, v_body])

is symmetric, orthogonal (an involution), and satisfies ``W q = q`` exactly
for consistent measurements. The fixed-point map ``½(I + W̄)`` is the
projector average onto the +1 eigenspaces.

Conventions: state ``q_NB``; ``f_body`` is specific force; references default
to NED gravity-down and the supplied magnetic field direction.

Reference: Zhou, Wu, Fourati et al., "Recursive linear continuous quaternion
attitude estimator from vector observations", IET Radar, Sonar & Navigation
(2018).
"""

from __future__ import annotations

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.errors import ConventionError
from qnav.filters.base import AttitudeFilter

__all__ = ["RoleqFilter"]


def _involution(v_ref: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """``W = −L([0, v_ref]) R([0, v_body])`` with ``W q = q`` for the true q."""
    pr = np.concatenate([[0.0], v_ref])
    pb = np.concatenate([[0.0], v_body])
    return -quat.left_matrix(pr) @ quat.right_matrix(pb)


class RoleqFilter(AttitudeFilter):
    """Recursive OLEQ: gyro propagation + one linear fixed-point correction.

    Parameters
    ----------
    m_ref:
        Magnetic field direction in the nav frame (any norm).
    weights:
        Relative weights (accelerometer, magnetometer); normalized to sum 1.
    """

    def __init__(
        self,
        m_ref: np.ndarray,
        weights: tuple[float, float] = (0.5, 0.5),
        q0=None,
        nav_frame: str = "NED",
    ) -> None:
        if nav_frame != "NED":
            raise ConventionError("RoleqFilter currently supports nav_frame='NED' only")
        super().__init__(q0=q0, nav_frame=nav_frame)
        m = np.asarray(m_ref, dtype=float)
        n = np.linalg.norm(m)
        if n < 1e-12:
            raise ValueError("m_ref must be non-zero")
        self.m_ref = m / n
        self.g_ref = np.array([0.0, 0.0, 1.0])    # gravity-down, NED
        w = np.asarray(weights, dtype=float)
        if w.shape != (2,) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be two nonnegative values with positive sum")
        self.weights = w / w.sum()

    def predict(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        """Gyro-only propagation (first-order transition, as in the paper)."""
        w = np.asarray(omega_body, dtype=float)
        qd = 0.5 * quat.mul(self.q, np.concatenate([[0.0], w]))
        self.q = quat.normalize(self.q + qd * dt)
        return self.q

    def step(
        self,
        omega_body: np.ndarray,
        dt: float,
        f_body: np.ndarray,
        m_body: np.ndarray,
    ) -> np.ndarray:
        """Propagate, then apply one OLEQ fixed-point iteration."""
        self.predict(omega_body, dt)

        f = np.asarray(f_body, dtype=float)
        m = np.asarray(m_body, dtype=float)
        fn = np.linalg.norm(f)
        mn = np.linalg.norm(m)
        if fn < 1e-12 or mn < 1e-12:
            return self.q                         # skip correction, keep gyro

        W = (
            self.weights[0] * _involution(self.g_ref, -f / fn)
            + self.weights[1] * _involution(self.m_ref, m / mn)
        )
        q = 0.5 * (np.eye(4) + W) @ self.q
        n = np.linalg.norm(q)
        if n < 1e-12:                              # antipodal degenerate seed
            return self.q
        self.q = q / n
        return self.q
