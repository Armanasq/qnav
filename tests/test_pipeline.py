"""Timestamped fusion pipeline (Phase 4): timing edge cases and replay."""

import numpy as np
import pytest

from qnav.attitude import quaternion as quat
from qnav.filters import (
    ClockDiscontinuityError,
    Eskf,
    FusionPipeline,
    Measurement,
)

DOWN = np.array([0.0, 0.0, -9.81])
MAG = np.array([0.3, 0.0, 0.5])


def _gravity_handler(est, m):
    est.update_gravity(m.value, sigma=0.02, timestamp=m.timestamp, sensor_id=m.sensor_id)


def _mag_handler(est, m):
    est.update_magnetometer(MAG, m.value, sigma=0.02,
                            timestamp=m.timestamp, sensor_id=m.sensor_id)


def _pipeline(max_lag=0.5):
    est = Eskf(gyro_noise_density=0.005, gyro_bias_walk=1e-6)
    p = FusionPipeline(est, max_lag=max_lag, max_gap=0.05)
    p.register_handler("accel", _gravity_handler)
    p.register_handler("mag", _mag_handler)
    return p


class TestImuPath:
    def test_variable_dt(self):
        p = _pipeline()
        assert p.process_imu(np.zeros(3), 0.0).action == "init"
        r1 = p.process_imu(np.zeros(3), 0.01)
        r2 = p.process_imu(np.zeros(3), 0.035)   # jittered sample
        assert r1.dt == pytest.approx(0.01) and r2.dt == pytest.approx(0.025)

    def test_duplicate_gyro_dropped(self):
        p = _pipeline()
        p.process_imu(np.zeros(3), 0.0)
        p.process_imu(np.zeros(3), 0.01)
        r = p.process_imu(np.zeros(3), 0.01)
        assert not r.applied and r.reason == "duplicate"

    def test_clock_discontinuity_raises(self):
        p = _pipeline()
        p.process_imu(np.zeros(3), 100.0)
        p.process_imu(np.zeros(3), 100.01)
        with pytest.raises(ClockDiscontinuityError):
            p.process_imu(np.zeros(3), 3.0)

    def test_gap_flagged_but_propagated(self):
        p = _pipeline()
        p.process_imu(np.zeros(3), 0.0)
        r = p.process_imu(np.array([0.1, 0.0, 0.0]), 0.2)  # 0.2 s gap
        assert r.applied and r.gap and r.dt == pytest.approx(0.2)


class TestMeasurementPath:
    def test_in_order_update(self):
        p = _pipeline()
        p.process_imu(np.zeros(3), 0.0)
        p.process_imu(np.zeros(3), 0.01)
        r = p.process_measurement(Measurement(value=DOWN, timestamp=0.01, sensor_id="accel"))
        assert r.applied and r.action == "update"
        assert p.estimator.last_update.accepted

    def test_duplicate_measurement_dropped(self):
        p = _pipeline()
        p.process_imu(np.zeros(3), 0.0)
        p.process_imu(np.zeros(3), 0.01)
        m = Measurement(value=DOWN, timestamp=0.01, sensor_id="accel", sequence_id=42)
        assert p.process_measurement(m).applied
        r = p.process_measurement(m)
        assert not r.applied and r.reason == "duplicate"

    def test_unknown_sensor_raises(self):
        p = _pipeline()
        p.process_imu(np.zeros(3), 0.0)
        with pytest.raises(KeyError, match="no handler"):
            p.process_measurement(Measurement(value=DOWN, timestamp=0.0, sensor_id="uwb"))

    def test_measurement_before_any_imu_dropped(self):
        p = _pipeline()
        r = p.process_measurement(Measurement(value=DOWN, timestamp=0.0, sensor_id="accel"))
        assert not r.applied and r.reason == "no_imu_yet"

    def test_too_old_rejected(self):
        p = _pipeline(max_lag=0.1)
        for k in range(30):
            p.process_imu(np.zeros(3), 0.01 * k)
        r = p.process_measurement(
            Measurement(value=DOWN, timestamp=0.05, sensor_id="accel"))
        assert not r.applied and r.reason == "too_old"

    def test_time_offset_applied(self):
        p = _pipeline(max_lag=0.1)
        for k in range(30):
            p.process_imu(np.zeros(3), 0.01 * k)
        # raw stamp too old, but a known +0.2 s offset brings it in-window
        p.set_time_offset("accel", 0.2)
        r = p.process_measurement(
            Measurement(value=DOWN, timestamp=0.05, sensor_id="accel", sequence_id=1))
        assert r.applied


class TestDelayedReplay:
    def _run(self, delay_mag: bool):
        """Static scenario; magnetometer sample at t=0.10 arrives either
        in order or 0.05 s late. Returns the final quaternion."""
        rng = np.random.default_rng(0)
        p = _pipeline()
        gyro = [0.002 * rng.standard_normal(3) for _ in range(21)]
        p.process_imu(gyro[0], 0.0)
        mag_meas = Measurement(value=MAG, timestamp=0.10, sensor_id="mag", sequence_id=9)
        for k in range(1, 21):
            t = 0.01 * k
            p.process_imu(gyro[k], t)
            if not delay_mag and abs(t - 0.10) < 1e-12:
                assert p.process_measurement(mag_meas).action == "update"
            if delay_mag and abs(t - 0.15) < 1e-12:
                r = p.process_measurement(mag_meas)
                assert r.action == "replay" and r.replayed_events > 0
            if abs(t - 0.05) < 1e-12 or abs(t - 0.20) < 1e-12:
                p.process_measurement(Measurement(
                    value=DOWN, timestamp=t, sensor_id="accel", sequence_id=k))
        return p.estimator.q.copy(), p.estimator.bias.copy()

    def test_replay_equals_in_order_processing(self):
        q_in_order, b_in_order = self._run(delay_mag=False)
        q_replayed, b_replayed = self._run(delay_mag=True)
        assert quat.angular_distance(q_in_order, q_replayed) < 1e-12
        np.testing.assert_allclose(b_replayed, b_in_order, atol=1e-15)

    def test_pipeline_continues_after_replay(self):
        p = _pipeline()
        p.process_imu(np.zeros(3), 0.0)
        for k in range(1, 11):
            p.process_imu(np.zeros(3), 0.01 * k)
        late = Measurement(value=DOWN, timestamp=0.05, sensor_id="accel", sequence_id=1)
        assert p.process_measurement(late).action == "replay"
        r = p.process_imu(np.zeros(3), 0.11)
        assert r.applied and r.dt == pytest.approx(0.01)


class TestInterpolation:
    def test_attitude_at_matches_snapshots_and_interpolates(self):
        p = _pipeline()
        omega = np.array([0.0, 0.0, 1.0])   # 1 rad/s yaw
        p.process_imu(omega, 0.0)
        for k in range(1, 11):
            p.process_imu(omega, 0.01 * k)
        q_meas = p.attitude_at(0.05)
        expected = quat.exp(np.array([0.0, 0.0, 0.05]))
        assert quat.angular_distance(q_meas, expected) < 1e-3
        # midpoint between snapshots
        q_mid = p.attitude_at(0.055)
        expected_mid = quat.exp(np.array([0.0, 0.0, 0.055]))
        assert quat.angular_distance(q_mid, expected_mid) < 1e-3

    def test_outside_window_raises(self):
        p = _pipeline()
        p.process_imu(np.zeros(3), 1.0)
        p.process_imu(np.zeros(3), 1.01)
        with pytest.raises(ValueError, match="outside history"):
            p.attitude_at(0.5)
