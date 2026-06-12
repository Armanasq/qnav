"""Stochastic error models for inertial sensors.

Standard continuous-time model (Kok–Hol–Schön tutorial; IEEE-STD-952 terms):

    y(t) = (I + S) x(t) + b(t) + n(t)
    ḃ(t) = n_b(t)

- ``n`` — white noise with PSD ``N²`` ((unit)²/Hz); over a sample interval dt
  the discrete noise has σ = N/√dt. N is the **noise density** (e.g.
  rad/s/√Hz), a.k.a. angle/velocity random walk coefficient.
- ``n_b`` — bias random walk with PSD ``B_rw²``; discrete bias increment
  σ_b = B_rw·√dt.
- First-order Gauss–Markov bias is available as an alternative
  (``ḃ = −b/τ + n_b``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NoiseModel", "white_noise", "random_walk", "gauss_markov"]


@dataclass(frozen=True)
class NoiseModel:
    """Per-axis stochastic parameters of a triaxial sensor.

    Attributes
    ----------
    density:
        White-noise density N [unit/√Hz] (scalar → isotropic).
    bias_walk:
        Bias random-walk density [unit·√Hz⁻¹/s... i.e. unit/s/√Hz]
        (drives ḃ; scalar → isotropic). 0 disables.
    bias_tau:
        Gauss–Markov time constant [s]; ``None``/``inf`` → pure random walk.
    """

    density: float = 0.0
    bias_walk: float = 0.0
    bias_tau: float | None = None

    def discrete_noise_sigma(self, dt: float) -> float:
        """σ of the discrete white measurement noise for sample interval dt."""
        return self.density / np.sqrt(dt)

    def discrete_bias_sigma(self, dt: float) -> float:
        """σ of the discrete bias increment over dt."""
        return self.bias_walk * np.sqrt(dt)


def white_noise(
    shape: tuple, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Zero-mean Gaussian samples with standard deviation ``sigma``."""
    return sigma * rng.standard_normal(shape)


def random_walk(
    n_steps: int, dt: float, walk_density: float, rng: np.random.Generator,
    b0: np.ndarray | None = None, n_axes: int = 3,
) -> np.ndarray:
    """Bias random-walk trajectory ``(n_steps, n_axes)``: ``b_{k+1} = b_k + w``,
    ``w ~ N(0, (walk_density·√dt)²)``."""
    steps = walk_density * np.sqrt(dt) * rng.standard_normal((n_steps, n_axes))
    b = np.cumsum(steps, axis=0)
    if b0 is not None:
        b += np.asarray(b0, dtype=float)
    return b


def gauss_markov(
    n_steps: int, dt: float, tau: float, walk_density: float,
    rng: np.random.Generator, b0: np.ndarray | None = None, n_axes: int = 3,
) -> np.ndarray:
    """First-order Gauss–Markov bias: ``b_{k+1} = e^{−dt/τ} b_k + w``."""
    phi = np.exp(-dt / tau)
    sig = walk_density * np.sqrt(dt)
    b = np.zeros((n_steps, n_axes))
    prev = np.zeros(n_axes) if b0 is None else np.asarray(b0, dtype=float)
    for k in range(n_steps):
        prev = phi * prev + sig * rng.standard_normal(n_axes)
        b[k] = prev
    return b
