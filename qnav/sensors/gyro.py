"""Gyroscope measurement model.

    ω_meas = (I + S_g) ω_true + b_g + n_g        [rad/s, sensor frame]

``S_g`` is the scale-factor/misalignment matrix (0 = ideal), ``b_g`` the bias,
``n_g`` white noise with density ``noise.density``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qnav.sensors.noise import NoiseModel

__all__ = ["GyroModel"]


@dataclass(frozen=True)
class GyroModel:
    """Deterministic + stochastic gyro error parameters (sensor frame)."""

    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale_misalignment: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    noise: NoiseModel = field(default_factory=NoiseModel)

    def measure(
        self, omega_true: np.ndarray, dt: float,
        rng: np.random.Generator | None = None,
        bias_trajectory: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the error model to true rates ``(..., 3)``.

        ``bias_trajectory`` (same shape) overrides the constant bias, for
        time-varying bias simulation; ``rng=None`` disables noise.
        """
        w = np.asarray(omega_true, dtype=float)
        out = w + w @ np.asarray(self.scale_misalignment, dtype=float).T
        out = out + (bias_trajectory if bias_trajectory is not None else self.bias)
        if rng is not None and self.noise.density > 0:
            out = out + self.noise.discrete_noise_sigma(dt) * rng.standard_normal(w.shape)
        return out

    def correct(self, omega_meas: np.ndarray, bias_estimate: np.ndarray | None = None) -> np.ndarray:
        """Invert the deterministic model:
        ``ω̂ = (I + S)⁻¹ (ω_meas − b̂)`` (uses the model bias if no estimate given)."""
        b = self.bias if bias_estimate is None else np.asarray(bias_estimate, dtype=float)
        A = np.eye(3) + np.asarray(self.scale_misalignment, dtype=float)
        return np.linalg.solve(A, (np.asarray(omega_meas, dtype=float) - b)[..., None])[..., 0]
