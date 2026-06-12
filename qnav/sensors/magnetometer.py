"""Magnetometer measurement model with hard/soft-iron effects.

    m_meas = A · m_true + b + n     [sensor frame]

- ``b`` — **hard iron** offset (permanent magnetization + electronics bias).
- ``A`` — **soft iron** + scale/misalignment matrix (I = ideal). Induced
  magnetization makes A symmetric positive definite in the classical model,
  but sensor misalignment adds a rotation; qnav stores the full 3×3.

Under this model a constant-magnitude true field maps measurements onto an
**ellipsoid** — the basis of ellipsoid calibration
(:mod:`qnav.calibration.mag_ellipsoid`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qnav.sensors.noise import NoiseModel

__all__ = ["MagnetometerModel"]


@dataclass(frozen=True)
class MagnetometerModel:
    """Hard/soft-iron + noise parameters (sensor frame)."""

    hard_iron: np.ndarray = field(default_factory=lambda: np.zeros(3))
    soft_iron: np.ndarray = field(default_factory=lambda: np.eye(3))
    noise: NoiseModel = field(default_factory=NoiseModel)

    def measure(
        self, m_true: np.ndarray, dt: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Apply ``m = A m_true + b (+ n)`` to true field(s) ``(..., 3)``."""
        m = np.asarray(m_true, dtype=float) @ np.asarray(self.soft_iron, dtype=float).T
        m = m + self.hard_iron
        if rng is not None and self.noise.density > 0:
            m = m + self.noise.discrete_noise_sigma(dt) * rng.standard_normal(m.shape)
        return m

    def correct(self, m_meas: np.ndarray) -> np.ndarray:
        """Invert the deterministic model: ``m̂ = A⁻¹ (m_meas − b)``."""
        d = np.asarray(m_meas, dtype=float) - self.hard_iron
        return np.linalg.solve(np.asarray(self.soft_iron, dtype=float), d[..., None])[..., 0]
