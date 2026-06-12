"""Quaternion kinematics and integrators: order of accuracy, exactness."""

import numpy as np
import pytest

from qnav.attitude import kinematics as kin, quaternion as quat
from tests.conftest import TOL_ALG, TOL_NUM


class TestQdot:
    def test_qdot_matches_omega_matrix(self, rng):
        q = quat.random((), rng)
        w = rng.standard_normal(3)
        assert np.allclose(kin.qdot(q, w), 0.5 * kin.omega_matrix(w) @ q, atol=TOL_ALG)

    def test_finite_difference(self, rng):
        q = quat.random((), rng)
        w = np.array([0.3, -0.2, 0.5])
        dt = 1e-8
        q1 = quat.mul(q, quat.exp(w * dt))
        assert np.allclose((q1 - q) / dt, kin.qdot(q, w), atol=1e-6)


class TestIntegrators:
    def test_exponential_exact_constant_rate(self, rng):
        w = np.array([0.1, -0.4, 0.2])
        q = quat.identity()
        for _ in range(1000):
            q = kin.integrate_exponential(q, w, 0.001)
        assert quat.angular_distance(q, quat.exp(w)) < 1e-12

    # rk4 with linearly interpolated rate samples is order-2 in the rate
    # interpolation error (documented in integrate_rk4); its constant is
    # smaller than midpoint's, checked separately below.
    @pytest.mark.parametrize("method,order", [("first_order", 1), ("midpoint", 2), ("rk4", 2)])
    def test_convergence_order(self, method, order):
        # time-varying rate; reference via fine exponential steps
        def w_of_t(t):
            return np.array([0.8 * np.sin(3 * t), 0.5 * np.cos(2 * t), 0.3 * t])

        T = 1.0

        def reference(n=200000):
            q = quat.identity()
            dt = T / n
            for k in range(n):
                q = kin.integrate_exponential(q, w_of_t((k + 0.5) * dt), dt)
            return q

        q_ref = reference()

        def run(n):
            q = quat.identity()
            dt = T / n
            for k in range(n):
                t0, t1 = k * dt, (k + 1) * dt
                if method == "first_order":
                    q = kin.integrate_first_order(q, w_of_t(t0), dt)
                else:
                    q = kin.integrate(q, w_of_t(t0), dt, method=method, omega_end=w_of_t(t1))
            return quat.angular_distance(q, q_ref)

        e1, e2 = run(100), run(200)
        rate = np.log2(e1 / e2)
        assert rate > order - 0.5, f"{method}: observed order {rate:.2f}"

    def test_all_methods_unit_norm(self, rng):
        q = quat.random((), rng)
        w0, w1 = rng.standard_normal((2, 3))
        for m in ("first_order", "exponential"):
            assert abs(np.linalg.norm(kin.integrate(q, w0, 0.1, method=m)) - 1) < TOL_ALG
        for m in ("midpoint", "rk4"):
            assert abs(np.linalg.norm(kin.integrate(q, w0, 0.1, method=m, omega_end=w1)) - 1) < TOL_ALG

    def test_midpoint_requires_omega_end(self):
        with pytest.raises(ValueError):
            kin.integrate(quat.identity(), np.zeros(3), 0.1, method="midpoint")

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            kin.integrate(quat.identity(), np.zeros(3), 0.1, method="euler5")


class TestInverseKinematics:
    def test_angular_velocity_recovery(self, rng):
        q0 = quat.random((), rng)
        w = np.array([0.2, 0.1, -0.3])
        dt = 0.05
        q1 = kin.integrate_exponential(q0, w, dt)
        assert np.allclose(kin.angular_velocity_from_quaternions(q0, q1, dt), w, atol=TOL_NUM)

    def test_bad_dt(self):
        with pytest.raises(ValueError):
            kin.angular_velocity_from_quaternions(quat.identity(), quat.identity(), 0.0)


class TestInterpolation:
    def test_slerp_endpoints(self, rng):
        from qnav.attitude import interpolation as interp
        q0, q1 = quat.random((), rng), quat.random((), rng)
        assert quat.angular_distance(interp.slerp(q0, q1, 0.0), q0) < TOL_NUM
        assert quat.angular_distance(interp.slerp(q0, q1, 1.0), q1) < TOL_NUM

    def test_slerp_constant_rate(self, rng):
        from qnav.attitude import interpolation as interp
        q0, q1 = quat.random((), rng), quat.random((), rng)
        ts = np.linspace(0, 1, 11)
        qs = np.stack([interp.slerp(q0, q1, t) for t in ts])
        steps = quat.angular_distance(qs[:-1], qs[1:])
        assert np.std(steps) < 1e-9

    def test_slerp_sign_safety(self, rng):
        from qnav.attitude import interpolation as interp
        q0 = quat.random((), rng)
        q1 = quat.mul(q0, quat.exp(np.array([0.2, 0, 0])))
        mid_a = interp.slerp(q0, q1, 0.5)
        mid_b = interp.slerp(q0, -q1, 0.5)  # flipped representation
        assert quat.angular_distance(mid_a, mid_b) < TOL_NUM

    def test_slerp_parallel(self, rng):
        from qnav.attitude import interpolation as interp
        q0 = quat.random((), rng)
        out = interp.slerp(q0, q0, 0.3)
        assert quat.angular_distance(out, q0) < TOL_NUM

    def test_series(self, rng):
        from qnav.attitude import interpolation as interp
        times = np.array([0.0, 1.0, 2.0])
        qs = quat.random((3,), rng)
        out = interp.slerp_series(times, qs, np.array([0.0, 0.5, 1.0, 2.0]))
        assert quat.angular_distance(out[0], qs[0]) < TOL_NUM
        assert quat.angular_distance(out[2], qs[1]) < TOL_NUM
        with pytest.raises(ValueError):
            interp.slerp_series(times, qs, np.array([2.5]))
