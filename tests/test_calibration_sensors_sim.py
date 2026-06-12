"""Calibration, sensor models, Allan variance, and simulation invariants."""

import numpy as np
import pytest

from qnav.attitude import quaternion as quat
from qnav.calibration import (
    calibrate_accelerometer, detect_static_intervals, estimate_bias, fit_ellipsoid,
)
from qnav.calibration.frame_alignment import align_from_vector_pairs, align_gyro_to_body
from qnav.calibration.soft_hard_iron import calibrate, quality_report
from qnav.errors import CalibrationError
from qnav.sensors import GyroModel, ImuModel, MagnetometerModel, NoiseModel
from qnav.sensors.accelerometer import AccelerometerModel
from qnav.sensors.alignment import lever_arm_acceleration
from qnav.sensors.allan import allan_deviation, identify_noise
from qnav.simulation import RigidBody, constant_rate, coning, synthesize
from qnav.simulation.imu_synthesis import ImuDataset
from qnav.simulation.noise_injection import apply_dropout, dropout_mask, inject_outliers
from qnav.validation.datasets import marg_dataset


class TestMagCalibration:
    def make_raw(self, rng, n=500):
        truth = MagnetometerModel(
            hard_iron=np.array([12.0, -5.0, 8.0]),
            soft_iron=np.array([[1.1, 0.05, 0.0], [0.05, 0.9, 0.02], [0.0, 0.02, 1.05]]),
        )
        dirs = rng.standard_normal((n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        m_true = 50.0 * dirs
        return truth, truth.measure(m_true)

    def test_recovers_hard_iron(self, rng):
        truth, raw = self.make_raw(rng)
        cal = fit_ellipsoid(raw)
        assert np.allclose(cal.hard_iron, truth.hard_iron, atol=1e-6)
        assert cal.rms_residual / cal.radius < 1e-9

    def test_corrected_is_spherical(self, rng):
        truth, raw = self.make_raw(rng)
        cal, fn = calibrate(raw, field_intensity=50.0)
        r = np.linalg.norm(fn(raw), axis=1)
        assert np.allclose(r, 50.0, atol=1e-6)
        q = quality_report(cal)
        assert q["relative_residual"] < 1e-9

    def test_noisy_still_good(self, rng):
        truth, raw = self.make_raw(rng, n=2000)
        raw = raw + 0.3 * rng.standard_normal(raw.shape)
        cal = fit_ellipsoid(raw)
        assert np.allclose(cal.hard_iron, truth.hard_iron, atol=0.2)

    def test_planar_data_rejected(self, rng):
        ang = rng.uniform(0, 2 * np.pi, 200)
        m = np.column_stack([50 * np.cos(ang), 50 * np.sin(ang), np.full(200, 3.0)])
        with pytest.raises(CalibrationError):
            fit_ellipsoid(m)

    def test_too_few_points(self):
        with pytest.raises(CalibrationError):
            fit_ellipsoid(np.ones((5, 3)))


class TestAccelCalibration:
    def test_recovers_bias_scale(self, rng):
        truth = AccelerometerModel(
            bias=np.array([0.2, -0.1, 0.3]),
            scale_misalignment=np.diag([0.02, -0.01, 0.03]),
        )
        dirs = rng.standard_normal((300, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        f_static = truth.measure(9.80665 * dirs, dt=0.01)
        cal = calibrate_accelerometer(f_static)
        assert np.allclose(cal.bias, truth.bias, atol=1e-6)
        corrected = cal.correct(f_static)
        assert np.allclose(np.linalg.norm(corrected, axis=1), 9.80665, atol=1e-6)


class TestGyroBias:
    def test_static_detection_and_bias(self, rng):
        n, dt = 2000, 0.01
        bias = np.array([0.01, -0.02, 0.005])
        gyro = bias + 0.001 * rng.standard_normal((n, 3))
        accel = np.tile([0.0, 0, -9.81], (n, 1)) + 0.01 * rng.standard_normal((n, 3))
        # inject motion in the middle
        gyro[800:1200] += 0.5
        mask = detect_static_intervals(gyro, accel, dt)
        assert mask[:700].mean() > 0.9 and mask[850:1150].mean() < 0.1
        b, s = estimate_bias(gyro, mask)
        assert np.allclose(b, bias, atol=1e-3)

    def test_insufficient_samples(self):
        with pytest.raises(CalibrationError):
            estimate_bias(np.zeros((10, 3)))


class TestAlignment:
    def test_vector_pair_alignment(self, rng):
        q_bs = quat.random((), rng)
        v_s = rng.standard_normal((20, 3))
        v_b = quat.rotate_vector(q_bs, v_s)
        al = align_from_vector_pairs(v_b, v_s)
        assert quat.angular_distance(al.q_body_sensor, q_bs) < 1e-9

    def test_gyro_alignment_discards_static(self, rng):
        q_bs = quat.random((), rng)
        w_s = rng.standard_normal((100, 3))
        w_s[:50] *= 1e-4  # static-ish samples must be discarded
        w_b = quat.rotate_vector(q_bs, w_s)
        al = align_gyro_to_body(w_b, w_s)
        assert quat.angular_distance(al.q_body_sensor, q_bs) < 1e-9

    def test_unobservable_raises(self):
        v = np.tile([0.0, 0, 1.0], (10, 1))
        with pytest.raises(CalibrationError):
            align_from_vector_pairs(v, v)

    def test_lever_arm(self):
        # pure spin about z, arm along x: centripetal accel −ω² r x̂
        a = lever_arm_acceleration(np.array([0, 0, 2.0]), np.zeros(3), np.array([0.5, 0, 0]))
        assert np.allclose(a, [-2.0, 0, 0], atol=1e-12)


class TestSensorsModels:
    def test_gyro_correct_inverts_measure(self, rng):
        g = GyroModel(bias=np.array([0.01, 0, -0.02]),
                      scale_misalignment=0.01 * rng.standard_normal((3, 3)))
        w = rng.standard_normal((50, 3))
        meas = g.measure(w, dt=0.01)
        assert np.allclose(g.correct(meas), w, atol=1e-12)

    def test_mag_correct_inverts_measure(self, rng):
        m = MagnetometerModel(hard_iron=np.array([1.0, 2, 3]),
                              soft_iron=np.eye(3) + 0.05 * rng.standard_normal((3, 3)))
        x = rng.standard_normal((20, 3))
        assert np.allclose(m.correct(m.measure(x)), x, atol=1e-12)

    def test_noise_scaling(self):
        nm = NoiseModel(density=0.01)
        assert abs(nm.discrete_noise_sigma(0.01) - 0.1) < 1e-12

    def test_allan_identifies_white_noise(self, rng):
        dt, n = 0.01, 200000
        density = 0.02
        x = (density / np.sqrt(dt)) * rng.standard_normal(n)
        taus, adev = allan_deviation(x, dt)
        est = identify_noise(taus, adev)
        assert abs(est["density"] - density) / density < 0.1

    def test_allan_input_validation(self):
        with pytest.raises(ValueError):
            allan_deviation(np.zeros(5), 0.01)


class TestSimulation:
    def test_synthesis_static_accel_is_minus_g(self):
        from qnav.simulation import static_pose
        truth = static_pose(quat.identity(), 1.0, 0.01)
        ds = synthesize(truth, ImuModel())
        assert np.allclose(ds.accel, [0, 0, -9.80665], atol=1e-9)
        assert np.allclose(ds.gyro, 0.0, atol=1e-12)

    def test_constant_rate_consistency(self):
        tr = constant_rate(np.array([0.1, -0.2, 0.3]), 5.0, 0.01)
        # q_{k+1} == q_k ⊗ Exp(w dt)
        dq = quat.relative(tr.q[:-1], tr.q[1:])
        assert np.allclose(quat.log(dq), tr.omega_body[:-1] * tr.dt, atol=1e-10)

    def test_coning_rate_magnitude_constant(self):
        tr = coning(half_angle=0.2, spin_rate=2.0, duration=3.0, dt=0.005)
        rates = np.linalg.norm(tr.omega_body[:-1], axis=1)
        assert np.std(rates) / np.mean(rates) < 1e-6

    def test_rigid_body_conservation(self):
        body = RigidBody(inertia=np.diag([1.0, 2.0, 3.0]),
                         omega=np.array([0.3, 0.5, -0.2]))
        E0 = body.kinetic_energy()
        L0 = body.angular_momentum_nav()
        for _ in range(2000):
            body.step(0.005)
        assert abs(body.kinetic_energy() - E0) / E0 < 1e-9
        assert np.linalg.norm(body.angular_momentum_nav() - L0) / np.linalg.norm(L0) < 1e-7

    def test_bias_walk_returned(self):
        ds = marg_dataset(duration=2.0)
        assert isinstance(ds, ImuDataset)
        rng = np.random.default_rng(0)
        from qnav.simulation import sinusoidal_euler
        truth = sinusoidal_euler(np.deg2rad([10, 5, 5]), [0.1, 0.2, 0.3], 2.0, 0.01)
        imu = ImuModel(gyro=GyroModel(noise=NoiseModel(density=0.001, bias_walk=1e-4)))
        out = synthesize(truth, imu, rng=rng, simulate_bias_walk=True)
        assert out.gyro_bias_true is not None and out.gyro_bias_true.shape == (truth.n, 3)

    def test_dropout_and_outliers(self, rng):
        keep = dropout_mask(1000, 0.1, rng)
        assert 0.8 < keep.mean() < 0.98
        x = np.zeros((1000, 3))
        xn = apply_dropout(x, keep, fill="nan")
        assert np.isnan(xn[~keep]).all()
        xo = inject_outliers(x, rate=0.05, magnitude=10.0, rng=rng)
        assert (np.abs(xo) > 1.0).any()
