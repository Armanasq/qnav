"""Robustness layer (Phase 3): gating, robust losses, quarantine, recovery."""

import numpy as np
import pytest

from qnav.attitude import quaternion as quat
from qnav.filters import (
    Eskf,
    EstimatorHealth,
    GatePolicy,
    SensorMonitor,
    UpdateResult,
    cauchy_weight,
    chi2_quantile,
    detect_saturation,
    huber_weight,
    tukey_weight,
)

DOWN = np.array([0.0, 0.0, -9.81])  # NED specific force at rest
MAG = np.array([0.3, 0.0, 0.5])


class TestChi2Quantile:
    # reference values (scipy.stats.chi2.ppf)
    @pytest.mark.parametrize("dof,p,expected", [
        (3, 0.95, 7.8147), (3, 0.997, 13.9313), (6, 0.95, 12.5916),
        (1, 0.95, 3.8415), (2, 0.99, 9.2103),
    ])
    def test_matches_reference_within_2pct(self, dof, p, expected):
        assert chi2_quantile(dof, p) == pytest.approx(expected, rel=0.02)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            chi2_quantile(0, 0.95)
        with pytest.raises(ValueError):
            chi2_quantile(3, 1.0)


class TestRobustWeights:
    def test_huber(self):
        assert huber_weight(0.5) == 1.0
        assert huber_weight(2.69) == pytest.approx(0.5, rel=1e-3)
        assert huber_weight(-2.69) == huber_weight(2.69)

    def test_cauchy(self):
        assert cauchy_weight(0.0) == 1.0
        assert cauchy_weight(2.385) == pytest.approx(0.5)
        assert 0.0 < cauchy_weight(100.0) < 0.01  # never exactly zero

    def test_tukey(self):
        assert tukey_weight(0.0) == 1.0
        assert tukey_weight(4.685) == 0.0
        assert tukey_weight(10.0) == 0.0
        assert 0.0 < tukey_weight(2.0) < 1.0

    def test_policy_validation(self):
        with pytest.raises(ValueError):
            GatePolicy(confidence=1.5)
        with pytest.raises(ValueError):
            GatePolicy(on_gate="explode")
        with pytest.raises(ValueError):
            GatePolicy(loss="l2")
        with pytest.raises(ValueError):
            GatePolicy(loss="huber", loss_scale=-1.0)


def _settled_eskf(**kwargs):
    """An ESKF that has converged on clean static measurements."""
    f = Eskf(gyro_noise_density=0.005, gyro_bias_walk=1e-6, **kwargs)
    for _ in range(50):
        f.predict(np.zeros(3), 0.01)
        f.update_gravity(DOWN, sigma=0.02)
        f.update_magnetometer(MAG, MAG, sigma=0.02)
    return f


class TestGating:
    def test_hard_rejection_leaves_state_untouched(self):
        f = _settled_eskf(gate=GatePolicy(confidence=0.997, on_gate="reject"))
        q_before, P_before = f.q.copy(), f.P.copy()
        # a wildly wrong magnetometer sample (magnetic disturbance)
        f.update_magnetometer(MAG, np.array([-0.3, 0.1, -0.5]), sigma=0.02)
        r = f.last_update
        assert not r.accepted and r.rejection_reason == "nis_gate"
        assert r.nis > r.gate_threshold
        np.testing.assert_array_equal(f.q, q_before)
        np.testing.assert_array_equal(f.P, P_before)

    def test_inlier_passes_gate(self):
        f = _settled_eskf(gate=GatePolicy())
        f.update_gravity(DOWN + 0.02 * np.ones(3), sigma=0.02)
        assert f.last_update.accepted and f.last_update.robust_weight == 1.0

    def test_soft_inflation_reduces_correction(self):
        outlier = np.array([-0.3, 0.1, -0.5])
        hard = _settled_eskf(gate=None)
        hard.update_magnetometer(MAG, outlier, sigma=0.02)
        dx_plain = np.linalg.norm(hard.last_update.state_correction)

        soft = _settled_eskf(gate=GatePolicy(on_gate="inflate"))
        soft.update_magnetometer(MAG, outlier, sigma=0.02)
        r = soft.last_update
        assert r.accepted and r.robust_weight < 1.0
        assert np.linalg.norm(r.state_correction) < dx_plain

    def test_robust_loss_deweights_large_residuals(self):
        f = _settled_eskf(gate=GatePolicy(loss="huber"))
        # ~1.5σ-per-axis residual: inside the 0.997 gate, outside Huber's k
        f.update_magnetometer(MAG, MAG + np.array([0.15, -0.15, 0.0]), sigma=0.1)
        r = f.last_update
        assert r.accepted and 0.0 < r.robust_weight < 1.0

    def test_gate_default_off_preserves_legacy(self):
        f = _settled_eskf()
        f.update_magnetometer(MAG, np.array([-0.3, 0.1, -0.5]), sigma=0.02)
        assert f.last_update.accepted  # no gate configured: always fused


class TestQuarantine:
    def test_quarantine_and_hysteresis_recovery(self):
        f = _settled_eskf(gate=GatePolicy())
        f.set_monitor("mag", SensorMonitor(quarantine_after=2, recover_after=3))
        bad = np.array([-0.3, 0.1, -0.5])

        for _ in range(2):
            f.update_magnetometer(MAG, bad, sigma=0.02)
        assert f.monitors["mag"].quarantined

        # good measurements are evaluated but not fused while quarantined
        for _ in range(2):
            f.update_magnetometer(MAG, MAG, sigma=0.02)
            assert f.last_update.rejection_reason == "quarantine"
        # third consecutive in-gate sample releases the sensor and fuses
        f.update_magnetometer(MAG, MAG, sigma=0.02)
        assert not f.monitors["mag"].quarantined
        assert f.last_update.accepted

    def test_relapse_resets_recovery_count(self):
        m = SensorMonitor(quarantine_after=1, recover_after=2)
        assert not m.note_measurement(False)   # quarantined
        assert not m.note_measurement(True)    # 1 of 2
        assert not m.note_measurement(False)   # relapse resets
        assert not m.note_measurement(True)    # 1 of 2 again
        assert m.note_measurement(True)        # released

    def test_timeout_detection(self):
        m = SensorMonitor(timeout=0.5)
        assert not m.timed_out(10.0)           # never seen: no timeout claim
        m.note_measurement(True, timestamp=10.0)
        assert not m.timed_out(10.4)
        assert m.timed_out(10.6)


class TestSaturation:
    def test_mask(self):
        x = np.array([[0.1, 0.2, 0.3], [15.9, 0.0, 0.0], [-16.0, 0.0, 0.0]])
        mask = detect_saturation(x, full_scale=16.0, margin=0.02)
        np.testing.assert_array_equal(mask, [False, True, True])


class TestRecovery:
    def test_inflate_covariance(self):
        f = _settled_eskf()
        tr = np.trace(f.P)
        f.inflate_covariance(10.0)
        assert np.trace(f.P) == pytest.approx(10.0 * tr)
        assert f.health is not EstimatorHealth.INVALID

    def test_inflate_attitude_only_preserves_bias_block(self):
        f = _settled_eskf()
        bias_block = f.P[3:, 3:].copy()
        f.inflate_covariance(100.0, attitude_only=True)
        np.testing.assert_array_equal(f.P[3:, 3:], bias_block)
        assert f.health is not EstimatorHealth.INVALID  # P stays PSD

    def test_reinitialize_from_vectors_recovers_large_error(self):
        f = _settled_eskf()
        bias = f.bias.copy()
        # corrupt the attitude by 120 degrees
        f.q = quat.mul(f.q, quat.exp(np.array([0.0, 0.0, np.deg2rad(120)])))
        f.reinitialize_from_vectors(DOWN, MAG, MAG, keep_bias=True)
        np.testing.assert_array_equal(f.bias, bias)
        assert f.health is EstimatorHealth.INITIALIZING  # history cleared
        # converges again on clean data
        for _ in range(50):
            f.predict(np.zeros(3), 0.01)
            f.update_gravity(DOWN, sigma=0.02)
            f.update_magnetometer(MAG, MAG, sigma=0.02)
        err = quat.angular_distance(f.q, quat.identity())
        assert err < np.deg2rad(3.0)


class TestHealthExtensions:
    def test_diverging_on_sustained_high_nis(self):
        f = _settled_eskf()
        stats = f.innovation_stats["accel"]
        for _ in range(15):
            stats.record(UpdateResult(
                accepted=True, innovation=np.ones(3),
                innovation_covariance=np.eye(3), nis=40.0,
            ))
        assert f.health is EstimatorHealth.DIVERGING

    def test_unobservable_with_single_direction(self):
        f = Eskf(gyro_noise_density=0.005)
        for _ in range(15):
            f.predict(np.zeros(3), 0.01)
            f.update_gravity(DOWN, sigma=0.02)   # gravity only: yaw unobservable
        assert f.health is EstimatorHealth.UNOBSERVABLE

    def test_second_direction_restores_observability(self):
        f = Eskf(gyro_noise_density=0.005)
        for _ in range(15):
            f.predict(np.zeros(3), 0.01)
            f.update_gravity(DOWN, sigma=0.02)
            f.update_magnetometer(MAG, MAG, sigma=0.02)
        assert f.health is EstimatorHealth.HEALTHY
