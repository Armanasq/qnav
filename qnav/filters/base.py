"""Filter foundations: explicit state, noise, and measurement declarations.

Every qnav estimator is a **stepwise** object — construct once, then call
``predict(gyro, dt)`` and ``update_*(...)`` per sample. Batch processing is a
thin loop the user controls (no hidden batch-in-constructor semantics).

Every estimator must declare:

- state definition (attributes & meaning)
- error definition (for covariance-bearing filters)
- process model and noise semantics (continuous densities vs discrete)
- per-measurement frame and noise semantics
- explicit assumptions (e.g. "accelerometer ≈ gravity only")
"""

from __future__ import annotations

import abc

import numpy as np

from qnav.attitude import quaternion as quat

__all__ = ["AttitudeFilter"]


class AttitudeFilter(abc.ABC):
    """Base class for attitude estimators producing ``q_nav_body``.

    Subclasses keep the current estimate in ``self.q`` (scalar-first
    Hamilton, ``q_nav_body`` for the declared navigation frame) and document
    their additional state.
    """

    #: navigation frame of the estimate ("NED" or "ENU"); set by subclass.
    nav_frame: str

    def __init__(self, q0: np.ndarray | None = None, nav_frame: str = "NED") -> None:
        if nav_frame not in ("NED", "ENU"):
            raise ValueError(f"nav_frame must be 'NED' or 'ENU', got {nav_frame!r}")
        self.nav_frame = nav_frame
        self.q = quat.identity() if q0 is None else quat.normalize(np.asarray(q0, dtype=float))

    @abc.abstractmethod
    def predict(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        """Propagate the attitude with a body-frame rate sample; returns q."""

    def run(self, omegas: np.ndarray, dt: float, update_fn=None) -> np.ndarray:
        """Convenience batch loop: predict each sample, optionally calling
        ``update_fn(self, k)`` after each predict. Returns ``(N, 4)``."""
        omegas = np.asarray(omegas, dtype=float)
        out = np.empty((omegas.shape[0], 4))
        for k in range(omegas.shape[0]):
            self.predict(omegas[k], dt)
            if update_fn is not None:
                update_fn(self, k)
            out[k] = self.q
        return out
