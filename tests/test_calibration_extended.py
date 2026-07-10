"""Extended calibration (Phase 9): observability grading, time offsets,
temperature bias models, lever-arm estimation."""

import numpy as np
import pytest

from qnav.calibration import (
    Observability,
    assess_least_squares,
    estimate_lever_arm,
    estimate_time_offset,
    fit_temperature_bias,
)
from qnav.errors import CalibrationError


class TestObservability:
    def test_well_conditioned(self):
        rng = np.random.default_rng(0)
        r = assess_least_squares(rng.standard_normal((100, 3)))
        assert r.status is Observability.OBSERVABLE
        assert r.condition_number < 10

    def test_rank_deficient(self):
        J = np.zeros((50, 3))
        J[:, 0] = 1.0  # only one parameter excited
        r = assess_least_squares(J)
        assert r.status is Observability.UNOBSERVABLE
        # weakest direction lies in the unexcited subspace
        assert abs(r.weakest_direction[0]) < 1e-12

    def test_weakly_observable(self):
        rng = np.random.default_rng(1)
        J = rng.standard_normal((100, 3))
        J[:, 2] = J[:, 0] + 1e-4 * rng.standard_normal(100)  # near-collinear
        r = assess_least_squares(J)
        assert r.status is Observability.WEAKLY_OBSERVABLE

    def test_underdetermined(self):
        r = assess_least_squares(np.ones((2, 3)))
        assert r.status is Observability.UNOBSERVABLE


class TestTimeOffset:
    def _streams(self, true_offset, n=2000, dt=0.005, noise=0.02, seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(n) * dt
        # band-limited excitation signal
        x = np.sin(2 * np.pi * 0.7 * t) + 0.5 * np.sin(2 * np.pi * 2.3 * t + 1.0)
        xa = x + noise * rng.standard_normal(n)
        xb = np.interp(t - true_offset, t, x) + noise * rng.standard_normal(n)
        return t, xa, t, xb

    @pytest.mark.parametrize("true_offset", [0.0, 0.033, -0.047])
    def test_recovers_offset_subsample(self, true_offset):
        # B's content lags A by true_offset, so B's stamps must be corrected
        # by -true_offset ("add offset to B's stamps to align with A")
        est = estimate_time_offset(*self._streams(true_offset), max_offset=0.2)
        assert est.reliable
        assert est.offset == pytest.approx(-true_offset, abs=2e-3)

    def test_unreliable_flag_on_uncorrelated_noise(self):
        rng = np.random.default_rng(2)
        t = np.arange(2000) * 0.005
        est = estimate_time_offset(t, rng.standard_normal(2000),
                                   t, rng.standard_normal(2000), max_offset=0.1)
        assert not est.reliable

    def test_rejects_zero_excitation(self):
        t = np.arange(100) * 0.01
        with pytest.raises(ValueError, match="excitation"):
            estimate_time_offset(t, np.ones(100), t, np.ones(100))

    def test_rejects_nonmonotonic_time(self):
        t = np.arange(100) * 0.01
        bad = t.copy()
        bad[50] = bad[49]
        with pytest.raises(ValueError):
            estimate_time_offset(bad, np.sin(t), t, np.sin(t))


class TestTemperatureBias:
    def test_recovers_linear_model(self):
        rng = np.random.default_rng(3)
        T = np.linspace(10.0, 50.0, 40)
        c0 = np.array([0.01, -0.02, 0.005])
        c1 = np.array([1e-4, 2e-4, -1e-4])
        B = c0 + np.outer(T - 30.0, c1) + 1e-5 * rng.standard_normal((40, 3))
        model = fit_temperature_bias(T, B, order=1, t_ref=30.0)
        np.testing.assert_allclose(model.coeffs[0], c0, atol=1e-5)
        np.testing.assert_allclose(model.coeffs[1], c1, atol=1e-6)
        assert model.observability.status is Observability.OBSERVABLE
        np.testing.assert_allclose(model.predict(30.0), c0, atol=1e-5)

    def test_insufficient_range_rejected(self):
        T = np.full(20, 25.0) + 1e-13 * np.arange(20)   # essentially isothermal
        B = np.zeros((20, 3))
        with pytest.raises(CalibrationError, match="excitation"):
            fit_temperature_bias(T, B, order=2)

    def test_too_few_samples_rejected(self):
        with pytest.raises(CalibrationError, match="samples"):
            fit_temperature_bias(np.array([20.0, 30.0]), np.zeros((2, 3)), order=1)

    def test_covariance_shrinks_with_more_data(self):
        rng = np.random.default_rng(4)

        def cov_trace(n):
            T = np.linspace(10, 50, n)
            B = 1e-4 * np.outer(T - 30, np.ones(3)) + 1e-5 * rng.standard_normal((n, 3))
            return np.trace(fit_temperature_bias(T, B, order=1).covariance)

        assert cov_trace(200) < cov_trace(20)


class TestLeverArm:
    def _motion(self, r_true, n=500, seed=5):
        rng = np.random.default_rng(seed)
        t = np.arange(n) * 0.01
        w = np.stack([1.5 * np.sin(3 * t), 1.0 * np.cos(2 * t), 0.8 * np.sin(5 * t)], axis=1)
        wd = np.stack([4.5 * np.cos(3 * t), -2.0 * np.sin(2 * t), 4.0 * np.cos(5 * t)], axis=1)
        from qnav.attitude import so3
        ad = np.stack([(so3.hat(wd[k]) + so3.hat(w[k]) @ so3.hat(w[k])) @ r_true
                       for k in range(n)])
        return w, wd, ad + 1e-3 * rng.standard_normal((n, 3))

    def test_recovers_lever_arm(self):
        r_true = np.array([0.15, -0.05, 0.30])
        est = estimate_lever_arm(*self._motion(r_true))
        np.testing.assert_allclose(est.lever_arm, r_true, atol=1e-3)
        assert est.observability.status is Observability.OBSERVABLE
        assert np.all(np.diag(est.covariance) > 0)

    def test_no_rotation_unobservable(self):
        n = 100
        with pytest.raises(CalibrationError, match="excite"):
            estimate_lever_arm(np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3)))
