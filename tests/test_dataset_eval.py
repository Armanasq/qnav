"""Tests for the real-dataset evaluation script (benchmarks/run_dataset_eval.py)."""

import argparse
import hashlib
import importlib.util
import pathlib

import numpy as np
import pytest

from qnav.attitude import quaternion as quat
from qnav.validation.imu_datasets import load_attitude_dataset

_SCRIPT = pathlib.Path(__file__).parent.parent / "benchmarks" / "run_dataset_eval.py"
_spec = importlib.util.spec_from_file_location("run_dataset_eval", _SCRIPT)
assert _spec is not None and _spec.loader is not None
eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_mod)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "repoimu_tstick_04_1_40s.npz"


@pytest.fixture(scope="module")
def ds():
    return load_attitude_dataset(FIXTURE)


class TestInitialAttitude:
    def test_accel_tilt_levels_gravity(self, ds):
        q0 = eval_mod._initial_attitude(ds, "accel_tilt")
        assert q0 is not None
        assert np.isclose(np.linalg.norm(q0), 1.0, atol=1e-9)
        f_glob = quat.rotate_vector(q0, ds.accel[0])
        cos_angle = f_glob[2] / np.linalg.norm(f_glob)
        assert np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1))) < 2.0

    def test_accel_mag_is_yaw_rotation_of_tilt(self, ds):
        q_tilt = eval_mod._initial_attitude(ds, "accel_tilt")
        q_am = eval_mod._initial_attitude(ds, "accel_mag")
        # the relative rotation q_am ∘ q_tilt⁻¹ must be a pure global-z yaw
        dq = quat.mul(q_am, quat.conjugate(q_tilt))
        rotvec = quat.log(dq)
        assert np.hypot(rotvec[0], rotvec[1]) < 1e-9
        # deterministic
        q_am2 = eval_mod._initial_attitude(ds, "accel_mag")
        np.testing.assert_array_equal(q_am, q_am2)

    def test_perturbed_seed_is_stable(self, ds):
        q1 = eval_mod._initial_attitude(ds, "perturbed_45")
        q2 = eval_mod._initial_attitude(ds, "perturbed_45")
        np.testing.assert_array_equal(q1, q2)
        # matches an independently recomputed sha256-based expectation
        seed = int.from_bytes(hashlib.sha256(ds.name.encode()).digest()[:8], "little")
        axis = np.random.default_rng(seed).standard_normal(3)
        axis /= np.linalg.norm(axis)
        first = int(np.flatnonzero(ds.valid)[0])
        expected = quat.normalize(
            quat.mul(ds.q_ref[first], quat.exp(np.deg2rad(45.0) * axis)))
        np.testing.assert_allclose(q1, expected, atol=1e-12)


class TestTrialReport:
    def test_heading_observability_fields(self):
        args = argparse.Namespace(init="accel_tilt", mag_reference="calibration",
                                  config="universal", calib_seconds=5.0)
        out, cfg = eval_mod.evaluate_trial(FIXTURE, args)
        assert out["heading_observable"] is True
        assert out["heading_reference"] == "mag"
        assert out["heading_metric_interpretation"] == "aligned-absolute"
        assert out["noise_values"] == cfg == eval_mod.UNIVERSAL

    def test_heading_unobservable_without_mag(self):
        args = argparse.Namespace(init="accel_tilt", mag_reference="none",
                                  config="universal", calib_seconds=5.0)
        out, _ = eval_mod.evaluate_trial(FIXTURE, args)
        assert out["heading_observable"] is False
        assert out["heading_reference"] == "none"
        assert out["heading_metric_interpretation"] == "drift-only"
