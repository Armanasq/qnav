#!/usr/bin/env python3
"""Reproducible qnav benchmark suite.

Run:  python benchmarks/run_benchmarks.py [output.json]

Covers the documented hot paths: quaternion/SO(3) batch operations,
attitude determination, attitude ESKF steps, 15-state navigation ESKF
propagation and updates, IMU preintegration, and real-time replay factors
at 100 Hz / 400 Hz / 1 kHz. Results carry the full environment record
(CPU, OS, Python, NumPy, BLAS, thread count, commit SHA) — numbers without
that context are not comparable and must not be quoted in documentation.
"""

from __future__ import annotations

import sys

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.attitude import so3
from qnav.determination import quest_q, saam_q
from qnav.filters import Eskf
from qnav.nav import ImuPreintegration, NavEskf, NavState
from qnav.validation.benchmark_runner import environment, run_benchmark, save_results

RNG = np.random.default_rng(0)


def bench_quaternion_ops(results):
    q1 = quat.random((100_000,), rng=RNG)
    q2 = quat.random((100_000,), rng=RNG)
    v = RNG.standard_normal((100_000, 3))
    results.append(run_benchmark("quat.mul batch 100k", lambda: quat.mul(q1, q2), 100_000))
    results.append(run_benchmark("quat.rotate_vector batch 100k",
                                 lambda: quat.rotate_vector(q1, v), 100_000))
    phi = 0.5 * RNG.standard_normal((100_000, 3))
    results.append(run_benchmark("quat.exp batch 100k", lambda: quat.exp(phi), 100_000))


def bench_so3_ops(results):
    phi = 0.5 * RNG.standard_normal((100_000, 3))
    R = so3.exp(phi)
    results.append(run_benchmark("so3.exp batch 100k", lambda: so3.exp(phi), 100_000))
    results.append(run_benchmark("so3.log batch 100k", lambda: so3.log(R), 100_000))
    results.append(run_benchmark("so3.right_jacobian batch 100k",
                                 lambda: so3.right_jacobian(phi), 100_000))


def bench_determination(results):
    v_nav = np.array([[0.0, 0.0, 1.0], [0.55, 0.0, 0.84]])
    q_true = quat.random((), rng=RNG)
    v_body = np.stack([quat.rotate_frame(q_true, v) for v in v_nav])

    def quest_1000():
        for _ in range(1000):
            quest_q(v_nav, v_body)

    def saam_1000():
        for _ in range(1000):
            saam_q(v_body[0], v_body[1])

    results.append(run_benchmark("quest 1000 solves", quest_1000, 1000))
    results.append(run_benchmark("saam 1000 solves", saam_1000, 1000))


def bench_attitude_eskf(results):
    gyro = 0.1 * RNG.standard_normal((1000, 3))

    def predict_1000():
        f = Eskf(gyro_noise_density=0.005, gyro_bias_walk=1e-6)
        for k in range(1000):
            f.predict(gyro[k], 0.01)

    def predict_update_1000():
        f = Eskf(gyro_noise_density=0.005, gyro_bias_walk=1e-6)
        down = np.array([0.0, 0.0, -1.0])
        for k in range(1000):
            f.predict(gyro[k], 0.01)
            f.update_gravity(down, sigma=0.02)

    results.append(run_benchmark("Eskf.predict x1000", predict_1000, 1000))
    results.append(run_benchmark("Eskf predict+gravity update x1000",
                                 predict_update_1000, 1000))


def bench_nav_eskf(results):
    lat, lon, h = np.deg2rad(45.0), 0.0, 100.0
    omega = np.array([0.01, -0.02, 0.005])
    f_b = np.array([0.0, 0.0, -9.806])
    gyro = omega + 0.002 * RNG.standard_normal((1000, 3))
    acc = f_b + 0.02 * RNG.standard_normal((1000, 3))

    def make():
        return NavEskf(NavState(q=quat.identity(), p=[lat, lon, h]),
                       gyro_noise_density=0.002, accel_noise_density=0.02,
                       gyro_bias_walk=1e-6, accel_bias_walk=1e-5)

    def propagate_1000():
        f = make()
        for k in range(1000):
            f.predict(gyro[k], acc[k], 0.01)

    def full_loop_1000():
        f = make()
        for k in range(1000):
            f.predict(gyro[k], acc[k], 0.01)
            if k % 10 == 9:
                f.update_position([lat, lon, h], sigma=2.0)
                f.update_velocity(np.zeros(3), sigma=0.1)

    results.append(run_benchmark("NavEskf.predict x1000", propagate_1000, 1000))
    results.append(run_benchmark("NavEskf predict + 10Hz pos/vel x1000",
                                 full_loop_1000, 1000))


def bench_preintegration(results):
    w = 0.1 * RNG.standard_normal((1000, 3))
    f = np.array([0.0, 0.0, -9.8]) + 0.1 * RNG.standard_normal((1000, 3))

    def integrate_1000():
        p = ImuPreintegration(0.002, 0.02)
        for k in range(1000):
            p.integrate(w[k], f[k], 0.005)

    results.append(run_benchmark("ImuPreintegration.integrate x1000",
                                 integrate_1000, 1000))


def bench_realtime_replay(results):
    """Real-time factor: wall time to process 10 s of IMU at each rate."""
    for hz in (100, 400, 1000):
        n = 10 * hz
        gyro = 0.05 * RNG.standard_normal((n, 3))
        acc = np.array([0.0, 0.0, -9.806]) + 0.02 * RNG.standard_normal((n, 3))
        dt = 1.0 / hz

        def replay(n=n, gyro=gyro, acc=acc, dt=dt):
            f = NavEskf(NavState(q=quat.identity(),
                                 p=[np.deg2rad(45.0), 0.0, 100.0]),
                        gyro_noise_density=0.002, accel_noise_density=0.02)
            for k in range(n):
                f.predict(gyro[k], acc[k], dt)

        r = run_benchmark(f"NavEskf 10s replay @ {hz} Hz", replay, n, repeats=5)
        results.append(r)
        rtf = 10.0 / r.median_s
        print(f"  -> real-time factor at {hz} Hz: {rtf:.1f}x")


def bench_long_covariance(results):
    """1-hour covariance propagation at 100 Hz (health of long missions)."""
    omega = np.array([0.001, 0.0, 0.0])
    f_b = np.array([0.0, 0.0, -9.806])

    def one_minute():   # scaled sample of the hour-long behavior
        f = NavEskf(NavState(q=quat.identity(), p=[np.deg2rad(45.0), 0.0, 100.0]),
                    gyro_noise_density=0.002, accel_noise_density=0.02,
                    gyro_bias_walk=1e-6, accel_bias_walk=1e-5)
        for _ in range(6000):
            f.predict(omega, f_b, 0.01)
        assert np.all(np.isfinite(f.P))

    results.append(run_benchmark("NavEskf 60s covariance propagation @100Hz",
                                 one_minute, 6000, repeats=3))


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results.json"
    results: list = []
    for section in (bench_quaternion_ops, bench_so3_ops, bench_determination,
                    bench_attitude_eskf, bench_nav_eskf, bench_preintegration,
                    bench_realtime_replay, bench_long_covariance):
        print(f"[{section.__name__}]")
        section(results)
    env = environment()
    print("\nEnvironment:", env)
    print(f"\n{'benchmark':45s} {'median':>10s} {'p95':>10s} {'per-item':>12s}")
    for r in results:
        print(f"{r.name:45s} {r.median_s*1e3:8.2f}ms {r.p95_s*1e3:8.2f}ms "
              f"{r.per_item_ns:10.0f}ns")
    save_results(results, out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
