"""Accelerometer measurement model.

The accelerometer measures **specific force** in the sensor frame:

    f_true = R_SN (a_N − g_N)            [m/s², sensor frame]
    f_meas = (I + S_a) f_true + b_a + n_a

At rest in NED (``g_N = [0, 0, +g]``, sensor level): ``f_true = [0, 0, −g]``
— i.e. "+1 g up". This sign convention is used consistently across
``qnav.heading`` and ``qnav.simulation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qnav.sensors.noise import NoiseModel

__all__ = ["AccelerometerModel", "specific_force"]


def specific_force(a_nav: np.ndarray, g_nav: np.ndarray, R_sensor_nav: np.ndarray) -> np.ndarray:
    """True specific force in the sensor frame: ``f = R_SN (a_N − g_N)``."""
    diff = np.asarray(a_nav, dtype=float) - np.asarray(g_nav, dtype=float)
    return np.einsum("...ij,...j->...i", np.asarray(R_sensor_nav, dtype=float), diff)


@dataclass(frozen=True)
class AccelerometerModel:
    """Deterministic + stochastic accelerometer error parameters (sensor frame)."""

    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale_misalignment: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    noise: NoiseModel = field(default_factory=NoiseModel)

    def measure(
        self, f_true: np.ndarray, dt: float,
        rng: np.random.Generator | None = None,
        bias_trajectory: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the error model to true specific force ``(..., 3)``."""
        f = np.asarray(f_true, dtype=float)
        out = f + f @ np.asarray(self.scale_misalignment, dtype=float).T
        out = out + (bias_trajectory if bias_trajectory is not None else self.bias)
        if rng is not None and self.noise.density > 0:
            out = out + self.noise.discrete_noise_sigma(dt) * rng.standard_normal(f.shape)
        return out

    def correct(self, f_meas: np.ndarray, bias_estimate: np.ndarray | None = None) -> np.ndarray:
        """``f̂ = (I + S)⁻¹ (f_meas − b̂)``."""
        b = self.bias if bias_estimate is None else np.asarray(bias_estimate, dtype=float)
        A = np.eye(3) + np.asarray(self.scale_misalignment, dtype=float)
        return np.linalg.solve(A, (np.asarray(f_meas, dtype=float) - b)[..., None])[..., 0]
