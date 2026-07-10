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
from typing import Callable, Optional

import numpy as np

from qnav._validate import ensure_positive_dt, ensure_vector3
from qnav.attitude import quaternion as quat
from qnav.types import ArrayLike

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

    def predict(self, omega_body: ArrayLike, dt: float) -> np.ndarray:
        """Propagate the attitude with a body-frame rate sample [rad/s]; returns q.

        Validates that ``omega_body`` is a finite 3-vector and ``dt`` a finite
        positive scalar, then dispatches to the subclass propagation.
        """
        omega = ensure_vector3(omega_body, "omega_body")
        if omega.ndim != 1:
            raise ValueError(f"omega_body must be a single 3-vector, got shape {omega.shape}")
        return self._predict(omega, ensure_positive_dt(dt))

    @abc.abstractmethod
    def _predict(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        """Subclass propagation; receives validated inputs."""

    def run(
        self,
        omegas: np.ndarray,
        dt: float,
        update_fn: Optional[Callable[["AttitudeFilter", int], object]] = None,
    ) -> np.ndarray:
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
