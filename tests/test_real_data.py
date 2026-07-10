"""Real-sensor-data validation (Phase 11).

The compact fixture (40 s of the public RepoIMU TStick trial at 100 Hz,
see tests/fixtures/README.md) always runs; the full benchmark collection
(EuRoC-MAV, TUM-VI, OxIOD, ...) runs only when present under qnav/data or
$QNAV_DATA_DIR. Dataset conventions (quaternion order, gravity direction)
are *verified* here against physics, not assumed by the loader.
"""

import pathlib
import tracemalloc

import numpy as np
import pytest

from qnav.attitude import quaternion as quat
from qnav.filters import Eskf, GatePolicy, LeftInvariantEskf, SensorMonitor
from qnav.validation.imu_datasets import (
    available_datasets,
    load_attitude_dataset,
)
from qnav.validation.replay_eval import heading_aligned_errors, replay_attitude

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "repoimu_tstick_04_1_40s.npz"
UP = np.array([0.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def ds():
    return load_attitude_dataset(FIXTURE)


def _q0(d):
    return d.q_ref[np.flatnonzero(d.valid)[0]]


def _m_ref(d, seconds=2.0):
    idx = np.flatnonzero(d.valid)[: int(seconds / d.dt)]
    return np.mean([quat.rotate_vector(d.q_ref[k], d.mag[k]) for k in idx], axis=0)


def _marg_update(m_ref):
    def fn(f, d, k):
        f.update_direction(UP, d.accel[k], sigma=0.05, sensor_id="accel")
        f.update_direction(m_ref, d.mag[k], sigma=0.1, sensor_id="mag")
    return fn


class TestConventions:
    """The loader's documented conventions hold on the real data."""

    def test_ground_truth_matches_gyro_integration(self, ds):
        # body rate reconstructed from consecutive reference quaternions must
        # track the measured gyro (scalar-first, q_ref_body convention)
        ks = np.arange(0, 3000, 10)
        w_est = np.stack([
            quat.log(quat.mul(quat.conjugate(ds.q_ref[k]), ds.q_ref[k + 1])) / ds.dt
            for k in ks])
        resid = np.linalg.norm(w_est - ds.gyro[ks], axis=1).mean()
        signal = np.abs(ds.gyro[ks]).mean()
        assert resid < 0.5 * max(signal, 0.05)

    def test_gravity_is_reference_z_up(self, ds):
        ks = np.arange(0, 3000, 50)
        pred = np.stack([quat.rotate_frame(ds.q_ref[k], 9.81 * UP) for k in ks])
        assert np.linalg.norm(pred - ds.accel[ks], axis=1).mean() < 1.0


class TestFixtureReplay:
    def test_tilt_accuracy_gravity_only(self, ds):
        r = replay_attitude(ds, lambda d: Eskf(
            gyro_noise_density=0.005, gyro_bias_walk=1e-5, q0=_q0(d), nav_frame="ENU"))
        assert r.tilt_rmse_deg < 1.0          # measured 0.26-0.3 deg
        assert r.realtime_factor > 1.0

    def test_marg_full_attitude(self, ds):
        r = replay_attitude(ds, lambda d: Eskf(
            gyro_noise_density=0.005, gyro_bias_walk=1e-5, q0=_q0(d), nav_frame="ENU"),
            _marg_update(_m_ref(ds)))
        assert r.rmse_deg < 3.0               # measured ~0.6 deg on the trial
        assert r.tilt_rmse_deg < 1.0
        assert r.rejection_rate == 0.0

    def test_invariant_filter_comparable_on_real_data(self, ds):
        m_ref = _m_ref(ds)
        reports = {}
        for name, cls in (("eskf", Eskf), ("liekf", LeftInvariantEskf)):
            reports[name] = replay_attitude(ds, lambda d, c=cls: c(
                gyro_noise_density=0.005, gyro_bias_walk=1e-5, q0=_q0(d),
                nav_frame="ENU"), _marg_update(m_ref))
        assert reports["liekf"].rmse_deg < 2 * reports["eskf"].rmse_deg + 0.5

    def test_unknown_initial_attitude_converges(self, ds):
        """Large-initial-error handling on real data: identity init."""
        r = replay_attitude(ds, lambda d: Eskf(
            gyro_noise_density=0.005, gyro_bias_walk=1e-5,
            P0=np.diag([1.0**2] * 3 + [0.01**2] * 3), nav_frame="ENU"),
            _marg_update(_m_ref(ds)), settle_s=10.0)
        assert r.rmse_deg < 5.0


class TestFaultInjection:
    def test_magnetic_disturbance_gated(self, ds):
        """A 10 s synthetic magnetic disturbance must be rejected by the gate
        and must not corrupt the attitude beyond the clean-run error."""
        m_ref = _m_ref(ds)
        disturbed = ds.mag.copy()
        k0, k1 = 1500, 2500
        disturbed[k0:k1] += np.array([0.8, -0.5, 0.3])   # strong iron disturbance

        def make(d):
            f = Eskf(gyro_noise_density=0.005, gyro_bias_walk=1e-5, q0=_q0(d),
                     nav_frame="ENU", gate=GatePolicy(confidence=0.999))
            f.set_monitor("mag", SensorMonitor(quarantine_after=10, recover_after=20))
            return f

        def update(f, d, k):
            f.update_direction(UP, d.accel[k], sigma=0.05, sensor_id="accel")
            f.update_direction(m_ref, disturbed[k], sigma=0.1, sensor_id="mag")

        r = replay_attitude(ds, make, update)
        assert r.rejection_rate > 0.05        # the disturbance was detected
        assert r.rmse_deg < 5.0               # and did not wreck the estimate

    def test_accel_dropout_survived(self, ds):
        """Losing the accelerometer for 5 s only drifts on gyro noise."""
        def update(f, d, k):
            if not 1000 <= k < 1500:
                f.update_direction(UP, d.accel[k], sigma=0.05, sensor_id="accel")

        r = replay_attitude(ds, lambda d: Eskf(
            gyro_noise_density=0.005, gyro_bias_walk=1e-5, q0=_q0(d),
            nav_frame="ENU"), update)
        assert r.tilt_rmse_deg < 2.0

    def test_accel_spikes_gated(self, ds):
        """Saturation-style accelerometer spikes are rejected, not fused."""
        rng = np.random.default_rng(0)
        spiked = ds.accel.copy()
        spike_idx = rng.choice(len(ds), size=len(ds) // 50, replace=False)
        spiked[spike_idx] += rng.choice([-1, 1], (spike_idx.size, 3)) * 30.0

        def make(d):
            return Eskf(gyro_noise_density=0.005, gyro_bias_walk=1e-5, q0=_q0(d),
                        nav_frame="ENU", gate=GatePolicy(confidence=0.999))

        def update(f, d, k):
            f.update_direction(UP, spiked[k], sigma=0.05, sensor_id="accel")

        r = replay_attitude(ds, make, update)
        assert r.rejection_rate > 0.01
        assert r.tilt_rmse_deg < 2.0


class TestLongDuration:
    def test_no_memory_growth_over_repeated_replay(self, ds):
        """Steady-state estimator memory must not grow with samples."""
        f = Eskf(gyro_noise_density=0.005, gyro_bias_walk=1e-5, q0=_q0(ds),
                 nav_frame="ENU")
        for k in range(500):                  # warm-up
            f.predict(ds.gyro[k % len(ds)], ds.dt)
            f.update_direction(UP, ds.accel[k % len(ds)], sigma=0.05)
        tracemalloc.start()
        base = tracemalloc.take_snapshot()
        for k in range(4000):
            f.predict(ds.gyro[k % len(ds)], ds.dt)
            f.update_direction(UP, ds.accel[k % len(ds)], sigma=0.05)
        growth = sum(s.size_diff for s in
                     tracemalloc.take_snapshot().compare_to(base, "filename"))
        tracemalloc.stop()
        assert growth < 512 * 1024, f"memory grew by {growth} bytes over 4000 steps"


needs_full_data = pytest.mark.skipif(
    not any(p.suffix in (".hdf5", ".h5") for p in available_datasets()),
    reason="full benchmark collection not present (set QNAV_DATA_DIR)",
)


@needs_full_data
class TestFullCollection:
    def test_euroc_trial_tilt_accuracy(self):
        paths = [p for p in available_datasets() if "EurocMAV" in p.name]
        if not paths:
            pytest.skip("EuRoC-MAV files not present")
        ds = load_attitude_dataset(paths[0])
        r = replay_attitude(ds, lambda d: Eskf(
            gyro_noise_density=0.005, gyro_bias_walk=1e-5, q0=_q0(d), nav_frame="ENU"))
        # regression tripwire, not a quality claim: aggressive MAV flight
        # with ungated fixed-sigma gravity aiding measures ~3.7 deg tilt
        assert r.tilt_rmse_deg < 5.0
        assert r.realtime_factor > 1.0


class TestErrorMetrics:
    def test_heading_alignment_recovers_known_offset(self):
        rng = np.random.default_rng(1)
        q_ref = quat.canonical(quat.random((300,), rng=rng))
        offset = quat.exp(np.array([0.0, 0.0, 0.4]))
        q_est = np.stack([quat.mul(quat.conjugate(offset), q) for q in q_ref])
        total, tilt, heading, psi = heading_aligned_errors(
            q_est, q_ref, np.ones(300, bool))
        # E = R_ref R_est^T = Rz(+0.4): the alignment reports the rotation
        # from the estimate frame to the reference frame
        assert psi == pytest.approx(0.4, abs=1e-6)
        assert np.max(total) < 1e-6 and np.max(tilt) < 1e-6

    def test_rejects_all_invalid(self):
        with pytest.raises(ValueError, match="valid"):
            heading_aligned_errors(np.tile([1.0, 0, 0, 0], (5, 1)),
                                   np.tile([1.0, 0, 0, 0], (5, 1)),
                                   np.zeros(5, bool))


class TestConventionReport:
    def test_fixture_report_machine_readable(self, ds):
        import json
        from qnav.validation.imu_datasets import verify_conventions
        r = verify_conventions(ds)
        json.dumps(r)                      # must be JSON-serializable
        assert r["ok"]
        assert r["gyro_consistency"]["verdict"] == "confirmed"
        assert r["gravity"]["confirmed"]

    def test_conjugated_ground_truth_is_contradicted(self, ds):
        """Feeding the wrong quaternion convention must be detected."""
        from qnav.validation.imu_datasets import verify_conventions
        wrong = ds.__class__(
            name=ds.name, dt=ds.dt, gyro=ds.gyro, accel=ds.accel,
            q_ref=quat.conjugate(ds.q_ref), valid=ds.valid, mag=ds.mag,
            movement=ds.movement)
        assert not verify_conventions(wrong)["ok"]


class TestGravityAidingDefault:
    def test_default_update_applies_accelerometer(self, ds):
        """Regression: trials without a magnetometer must still be
        gravity-aided (update_fn=None means the default accel update)."""
        r = replay_attitude(ds, lambda d: Eskf(
            gyro_noise_density=0.005, gyro_bias_walk=1e-5, q0=_q0(d),
            nav_frame="ENU"))
        assert "accel" in r.mean_nis          # accelerometer updates ran
        assert r.tilt_rmse_deg < 1.0          # and they aided the tilt
