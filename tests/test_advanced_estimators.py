"""Phase 10: left-invariant EKF vs the reference ESKF under identical data.

The comparison uses `qnav.validation.comparison` — one pre-generated
dataset, identical initialization, noise settings, and measurement
schedule for every estimator. Assertions state only what the data shows.
"""

import numpy as np
import pytest

from qnav.attitude import quaternion as quat
from qnav.filters import Eskf, UkfAttitude
from qnav.filters.invariant import LeftInvariantEskf
from qnav.validation.comparison import compare_attitude_estimators

DOWN = np.array([0.0, 0.0, -1.0])
MAG = np.array([0.55, 0.0, 0.835])
DT = 0.01


def _dataset(n=1500, seed=0, rate_scale=0.5):
    """Pre-generated truth + noisy sensor data (identical for all filters)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    omega_true = rate_scale * np.stack(
        [np.sin(1.3 * t), 0.7 * np.cos(2.1 * t), 0.4 * np.sin(0.9 * t + 1)], axis=1)
    q = quat.identity()
    q_true = np.empty((n, 4))
    gyro = np.empty((n, 3))
    acc = np.empty((n, 3))
    mag = np.empty((n, 3))
    bg_true = np.array([0.01, -0.005, 0.008])
    for k in range(n):
        q = quat.normalize(quat.mul(q, quat.exp(omega_true[k] * DT)))
        q_true[k] = q
        gyro[k] = omega_true[k] + bg_true + 0.005 * rng.standard_normal(3)
        acc[k] = quat.rotate_frame(q, DOWN) + 0.02 * rng.standard_normal(3)
        mag[k] = quat.rotate_frame(q, MAG) + 0.02 * rng.standard_normal(3)
    return gyro, acc, mag, q_true


def _update_fn(acc, mag):
    def fn(f, k):
        # update_direction is the common contract across all three filters
        f.update_direction(DOWN, acc[k], sigma=0.03)
        f.update_direction(MAG, mag[k], sigma=0.03)
    return fn


def _make(cls, q0=None):
    return cls(gyro_noise_density=0.005, gyro_bias_walk=1e-4, q0=q0,
               P0=np.diag([0.3**2] * 3 + [0.02**2] * 3))


class TestSmallError:
    def test_both_formulations_converge_and_agree(self):
        gyro, acc, mag, q_true = _dataset()
        results = compare_attitude_estimators(
            {"eskf": _make(Eskf), "liekf": _make(LeftInvariantEskf)},
            gyro, q_true, DT, update_fn=_update_fn(acc, mag),
        )
        by = {r.name: r for r in results}
        assert by["eskf"].converged and by["liekf"].converged
        assert by["eskf"].rmse_deg < 2.0 and by["liekf"].rmse_deg < 2.0
        # near the truth the two error definitions are equivalent to first
        # order: steady-state accuracy must be comparable (within 50%)
        ratio = by["liekf"].rmse_deg / by["eskf"].rmse_deg
        assert 0.5 < ratio < 2.0

    def test_mean_nis_near_dimension(self):
        gyro, acc, mag, q_true = _dataset()
        results = compare_attitude_estimators(
            {"eskf": _make(Eskf)}, gyro, q_true, DT,
            update_fn=_update_fn(acc, mag))
        nis = results[0].mean_nis
        assert 0.5 < nis["direction"] < 6.0  # chi2(3) mean is 3 (R here is conservative)


class TestLargeInitialError:
    @pytest.mark.parametrize("err_deg", [120.0, 160.0])
    def test_liekf_converges_from_large_error(self, err_deg):
        gyro, acc, mag, q_true = _dataset(n=2000)
        q0_bad = quat.exp(np.deg2rad(err_deg) * np.array([0.0, 0.0, 1.0]))
        P0 = np.diag([1.0**2] * 3 + [0.02**2] * 3)  # honest large uncertainty

        def make(cls):
            return cls(gyro_noise_density=0.005, gyro_bias_walk=1e-4,
                       q0=q0_bad, P0=P0)

        results = compare_attitude_estimators(
            {"eskf": make(Eskf), "liekf": make(LeftInvariantEskf),
             "ukf": UkfAttitude(gyro_noise_density=0.005, q0=q0_bad,
                                P0=1.0**2 * np.eye(3))},
            gyro, q_true, DT, update_fn=_update_fn(acc, mag),
        )
        by = {r.name: r for r in results}
        # the invariant filter must converge where its linearization claims
        # validity; the UKF is the existing large-error reference
        assert by["liekf"].converged, by["liekf"]
        assert by["ukf"].converged, by["ukf"]
        # record the reference ESKF outcome without asserting failure —
        # it may or may not recover; the claim under test is about LIEKF
        assert by["liekf"].final_error_deg < 5.0
