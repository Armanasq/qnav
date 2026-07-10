"""Tests for centralized input validation (qnav._validate) and its wiring
into the filter layer."""

import numpy as np
import pytest

from qnav import _validate as v
from qnav.filters import Eskf, MahonyFilter


class TestEnsureHelpers:
    def test_finite_rejects_nan_and_inf(self):
        with pytest.raises(ValueError, match="non-finite"):
            v.ensure_finite([1.0, np.nan, 0.0], "x")
        with pytest.raises(ValueError, match="non-finite"):
            v.ensure_finite([np.inf], "x")

    def test_array_rejects_non_numeric(self):
        with pytest.raises(TypeError):
            v.ensure_array("not a number", "x")

    def test_shape_trailing_and_batch(self):
        assert v.ensure_shape(np.zeros((5, 3)), (3,), "x").shape == (5, 3)
        with pytest.raises(ValueError, match="trailing shape"):
            v.ensure_shape(np.zeros(4), (3,), "x")

    def test_unit_quaternion(self):
        v.ensure_unit_quaternion([1.0, 0, 0, 0])
        with pytest.raises(ValueError, match="unit-norm"):
            v.ensure_unit_quaternion([2.0, 0, 0, 0])

    def test_rotation_matrix(self):
        v.ensure_rotation_matrix(np.eye(3))
        with pytest.raises(ValueError, match="orthogonal"):
            v.ensure_rotation_matrix(2 * np.eye(3))
        with pytest.raises(ValueError, match="reflection"):
            v.ensure_rotation_matrix(np.diag([1.0, 1.0, -1.0]))

    def test_positive_dt(self):
        assert v.ensure_positive_dt(0.01) == 0.01
        for bad in (0.0, -1.0, np.nan, np.inf):
            with pytest.raises(ValueError):
                v.ensure_positive_dt(bad)

    def test_monotonic(self):
        v.ensure_monotonic([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            v.ensure_monotonic([0.0, 1.0, 1.0])
        v.ensure_monotonic([0.0, 1.0, 1.0], strict=False)
        with pytest.raises(ValueError, match="non-decreasing"):
            v.ensure_monotonic([0.0, 2.0, 1.0], strict=False)

    def test_covariance(self):
        v.ensure_covariance(np.eye(3), 3)
        with pytest.raises(ValueError, match="shape"):
            v.ensure_covariance(np.eye(3), 6)
        asym = np.eye(3)
        asym[0, 1] = 0.5
        with pytest.raises(ValueError, match="symmetric"):
            v.ensure_covariance(asym, 3)
        with pytest.raises(ValueError, match="semidefinite"):
            v.ensure_covariance(np.diag([1.0, 1.0, -1.0]), 3)


class TestFilterBoundary:
    def test_predict_rejects_bad_dt(self):
        f = MahonyFilter()
        with pytest.raises(ValueError, match="dt"):
            f.predict(np.zeros(3), 0.0)
        with pytest.raises(ValueError, match="dt"):
            f.predict(np.zeros(3), -0.01)

    def test_predict_rejects_nan_rate(self):
        f = MahonyFilter()
        with pytest.raises(ValueError, match="non-finite"):
            f.predict([np.nan, 0.0, 0.0], 0.01)

    def test_predict_rejects_wrong_shape(self):
        f = MahonyFilter()
        with pytest.raises(ValueError):
            f.predict([0.0, 0.0], 0.01)

    def test_eskf_constructor_validation(self):
        with pytest.raises(ValueError, match="gyro_noise_density"):
            Eskf(gyro_noise_density=-1.0)
        with pytest.raises(ValueError, match="P0"):
            Eskf(gyro_noise_density=0.01, P0=np.eye(3))
        bad = np.diag([1.0] * 5 + [-1.0])
        with pytest.raises(ValueError, match="semidefinite"):
            Eskf(gyro_noise_density=0.01, P0=bad)

    def test_eskf_update_validation(self):
        f = Eskf(gyro_noise_density=0.01)
        with pytest.raises(ValueError, match="sigma"):
            f.update_gravity([0.0, 0.0, -1.0], sigma=0.0)
        with pytest.raises(ValueError, match="non-zero norm"):
            f.update_gravity([0.0, 0.0, 0.0], sigma=0.02)

    def test_valid_predict_unchanged(self):
        f = Eskf(gyro_noise_density=0.01)
        q = f.predict([0.01, -0.02, 0.03], 0.01)
        assert np.isfinite(q).all() and abs(np.linalg.norm(q) - 1) < 1e-12
