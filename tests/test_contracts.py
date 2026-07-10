"""Estimator contracts (Phase 2): snapshots, reset, health, update results.

Lifecycle tests are parametrized over every concrete filter so the contract
holds uniformly across the estimator family.
"""

import numpy as np
import pytest

from qnav.filters import (
    AquaFilter,
    ComplementaryFilter,
    Eskf,
    EstimatorHealth,
    FastKalmanFilter,
    FouratiFilter,
    InnovationStatistics,
    MadgwickStyleFilter,
    MahonyFilter,
    Measurement,
    QuaternionEkf,
    RoleqFilter,
    UkfAttitude,
    UpdateResult,
)

FILTER_FACTORIES = [
    pytest.param(lambda: AquaFilter(), id="Aqua"),
    pytest.param(lambda: ComplementaryFilter(), id="Complementary"),
    pytest.param(lambda: Eskf(gyro_noise_density=0.01, gyro_bias_walk=1e-5), id="Eskf"),
    pytest.param(lambda: FastKalmanFilter(), id="FastKF"),
    pytest.param(lambda: FouratiFilter(), id="Fourati"),
    pytest.param(lambda: MadgwickStyleFilter(), id="Madgwick"),
    pytest.param(lambda: MahonyFilter(), id="Mahony"),
    pytest.param(lambda: QuaternionEkf(gyro_noise_density=0.01), id="QuaternionEkf"),
    pytest.param(lambda: RoleqFilter(m_ref=np.array([0.3, 0.0, 0.5])), id="Roleq"),
    pytest.param(lambda: UkfAttitude(gyro_noise_density=0.01), id="Ukf"),
]


def _propagate(f, n=25, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        f.predict(0.3 * rng.standard_normal(3), 0.01)


class TestLifecycle:
    @pytest.mark.parametrize("make", FILTER_FACTORIES)
    def test_reset_restores_constructed_state(self, make):
        f = make()
        q0 = f.q.copy()
        _propagate(f)
        assert not np.allclose(f.q, q0)
        f.reset()
        np.testing.assert_array_equal(f.q, q0)

    @pytest.mark.parametrize("make", FILTER_FACTORIES)
    def test_snapshot_restore_roundtrip(self, make):
        f = make()
        _propagate(f, n=10)
        snap = f.snapshot(timestamp=0.1)
        q_at_snap = f.q.copy()
        _propagate(f, n=10, seed=1)
        assert not np.allclose(f.q, q_at_snap)
        f.restore(snap)
        np.testing.assert_array_equal(f.q, q_at_snap)
        # restored filter must continue functioning
        f.predict(np.array([0.1, 0.0, 0.0]), 0.01)

    @pytest.mark.parametrize("make", FILTER_FACTORIES)
    def test_snapshot_is_deep(self, make):
        f = make()
        snap = f.snapshot()
        q_saved = np.asarray(snap.state["q"]).copy()
        _propagate(f)
        np.testing.assert_array_equal(np.asarray(snap.state["q"]), q_saved)

    def test_restore_rejects_wrong_type(self):
        a, b = MahonyFilter(), RoleqFilter(m_ref=np.array([0.3, 0.0, 0.5]))
        with pytest.raises(ValueError, match="snapshot is for"):
            b.restore(a.snapshot())

    @pytest.mark.parametrize("make", FILTER_FACTORIES)
    def test_reset_after_snapshot_restore(self, make):
        f = make()
        q0 = f.q.copy()
        _propagate(f)
        f.restore(f.snapshot())
        f.reset()
        np.testing.assert_array_equal(f.q, q0)


class TestHealth:
    def test_initializing_then_healthy(self):
        f = Eskf(gyro_noise_density=0.01)
        assert f.health is EstimatorHealth.INITIALIZING
        f.predict(np.zeros(3), 0.01)
        assert f.health is EstimatorHealth.INITIALIZING  # prediction only
        f.update_gravity(np.array([0.0, 0.0, -9.81]), sigma=0.02)
        assert f.health is EstimatorHealth.HEALTHY

    def test_invalid_on_nonfinite_state(self):
        f = Eskf(gyro_noise_density=0.01)
        f.q = np.array([np.nan, 0.0, 0.0, 0.0])
        assert f.health is EstimatorHealth.INVALID

    def test_invalid_on_indefinite_covariance(self):
        f = Eskf(gyro_noise_density=0.01)
        f.update_gravity(np.array([0.0, 0.0, -9.81]), sigma=0.02)
        f.P = np.diag([1.0] * 5 + [-1.0])
        assert f.health is EstimatorHealth.INVALID

    def test_degraded_on_consecutive_rejections(self):
        f = Eskf(gyro_noise_density=0.01)
        f.update_gravity(np.array([0.0, 0.0, -9.81]), sigma=0.02)
        stats = f.innovation_stats["accel"]
        for _ in range(3):
            stats.record(UpdateResult(
                accepted=False, innovation=np.zeros(3),
                innovation_covariance=np.eye(3), nis=100.0,
                rejection_reason="gate",
            ))
        assert f.health is EstimatorHealth.DEGRADED


class TestUpdateResult:
    def test_eskf_populates_last_update(self):
        f = Eskf(gyro_noise_density=0.01)
        innov = f.update_gravity(np.array([0.0, 0.0, -9.81]), sigma=0.02, timestamp=1.5)
        r = f.last_update
        assert r is not None and r.accepted
        np.testing.assert_array_equal(r.innovation, innov)   # legacy return preserved
        assert r.innovation_covariance.shape == (3, 3)
        assert r.state_correction is not None and r.state_correction.shape == (6,)
        assert np.isfinite(r.nis) and r.nis >= 0.0
        assert r.timestamp == 1.5 and r.sensor_id == "accel"

    def test_nis_matches_definition(self):
        f = Eskf(gyro_noise_density=0.01)
        f.update_gravity(np.array([0.1, 0.2, -9.7]), sigma=0.05)
        r = f.last_update
        expected = float(r.innovation @ np.linalg.solve(r.innovation_covariance, r.innovation))
        assert r.nis == pytest.approx(expected)

    def test_sensor_streams_tracked_separately(self):
        f = Eskf(gyro_noise_density=0.01)
        f.update_gravity(np.array([0.0, 0.0, -9.81]), sigma=0.02)
        f.update_magnetometer(np.array([0.3, 0.0, 0.5]), np.array([0.3, 0.0, 0.5]), sigma=0.05)
        f.update_gravity(np.array([0.0, 0.0, -9.81]), sigma=0.02)
        assert f.innovation_stats["accel"].count == 2
        assert f.innovation_stats["mag"].count == 1

    def test_reset_clears_update_history(self):
        f = Eskf(gyro_noise_density=0.01)
        f.update_gravity(np.array([0.0, 0.0, -9.81]), sigma=0.02)
        f.reset()
        assert f.last_update is None and f.innovation_stats == {}
        assert f.health is EstimatorHealth.INITIALIZING


class TestMeasurement:
    def test_valid_construction(self):
        m = Measurement(
            value=np.array([0.1, 0.2, 9.7]), timestamp=2.0, frame="body",
            covariance=0.01 * np.eye(3), sensor_id="imu0", sequence_id=7,
            validity_interval=(1.99, 2.01),
        )
        assert m.value.shape == (3,) and m.timestamp == 2.0

    def test_rejects_nonfinite_value(self):
        with pytest.raises(ValueError, match="non-finite"):
            Measurement(value=np.array([np.nan, 0.0, 0.0]), timestamp=0.0)

    def test_rejects_bad_covariance_shape(self):
        with pytest.raises(ValueError, match="covariance"):
            Measurement(value=np.zeros(3), timestamp=0.0, covariance=np.eye(2))

    def test_rejects_inverted_validity_interval(self):
        with pytest.raises(ValueError, match="validity_interval"):
            Measurement(value=np.zeros(3), timestamp=0.0, validity_interval=(2.0, 1.0))


class TestInnovationStatistics:
    def test_mean_nis_converges(self):
        stats = InnovationStatistics()
        rng = np.random.default_rng(0)
        for _ in range(2000):
            nis = float(np.sum(rng.standard_normal(3) ** 2))  # true chi2(3)
            stats.record(UpdateResult(
                accepted=True, innovation=np.zeros(3),
                innovation_covariance=np.eye(3), nis=nis,
            ))
        assert stats.mean_nis == pytest.approx(3.0, rel=0.1)
        assert stats.var_nis == pytest.approx(6.0, rel=0.2)
        assert stats.accepted == 2000 and stats.consecutive_rejections == 0
