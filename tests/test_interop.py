"""Interop adapters (Phase 8): SciPy rotation bridge, CSV/pandas loaders."""

import numpy as np
import pytest

from qnav.attitude import quaternion as quat
from qnav.interop.datasets import ImuData, imu_from_dataframe, load_imu_csv

scipy_spatial = pytest.importorskip("scipy.spatial.transform")
pd = pytest.importorskip("pandas")

from qnav.interop.scipy_rotation import from_scipy, to_scipy  # noqa: E402


class TestScipyBridge:
    def test_roundtrip_batch(self):
        rng = np.random.default_rng(0)
        q = quat.canonical(quat.random((50,), rng=rng))
        back = from_scipy(to_scipy(q))
        np.testing.assert_allclose(back, q, atol=1e-14)

    def test_same_physical_rotation(self):
        rng = np.random.default_rng(1)
        q = quat.random((20,), rng=rng)
        v = rng.standard_normal((20, 3))
        v_qnav = quat.rotate_vector(q, v)
        v_scipy = to_scipy(q).apply(v)
        np.testing.assert_allclose(v_scipy, v_qnav, atol=1e-12)

    def test_from_scipy_matrix_source(self):
        R = scipy_spatial.Rotation.from_euler("zyx", [30, 10, 5], degrees=True)
        q = from_scipy(R)
        np.testing.assert_allclose(quat.norm(q), 1.0, atol=1e-15)
        np.testing.assert_allclose(to_scipy(q).as_matrix(), R.as_matrix(), atol=1e-14)

    def test_rejects_non_unit(self):
        with pytest.raises(ValueError, match="unit-norm"):
            to_scipy(np.array([2.0, 0.0, 0.0, 0.0]))

    def test_rejects_wrong_type(self):
        with pytest.raises(TypeError, match="Rotation"):
            from_scipy(np.eye(3))


def _write_csv(path, rows, header="t,gx,gy,gz,ax,ay,az,mx,my,mz"):
    lines = [header] + [",".join(str(x) for x in r) for r in rows]
    path.write_text("\n".join(lines))


class TestCsvLoader:
    def test_load_with_mag(self, tmp_path):
        p = tmp_path / "imu.csv"
        _write_csv(p, [[0.00, 0.1, 0.2, 0.3, 0.0, 0.0, -9.8, 20, 0, 43],
                       [0.01, 0.1, 0.2, 0.3, 0.0, 0.0, -9.8, 20, 0, 43]])
        d = load_imu_csv(p, mag_cols=(7, 8, 9))
        assert len(d) == 2 and d.gyro.shape == (2, 3) and d.mag.shape == (2, 3)
        assert d.dropped_rows == 0

    def test_nonfinite_rejected_by_default(self, tmp_path):
        p = tmp_path / "imu.csv"
        _write_csv(p, [[0.00, 0.1, 0.2, 0.3, 0, 0, -9.8, 0, 0, 0],
                       [0.01, np.nan, 0.2, 0.3, 0, 0, -9.8, 0, 0, 0]])
        with pytest.raises(ValueError, match="drop_bad_rows"):
            load_imu_csv(p)
        d = load_imu_csv(p, drop_bad_rows=True)
        assert len(d) == 1 and d.dropped_rows == 1

    def test_nonmonotonic_time_rejected(self, tmp_path):
        p = tmp_path / "imu.csv"
        _write_csv(p, [[0.01, 0, 0, 0, 0, 0, -9.8, 0, 0, 0],
                       [0.00, 0, 0, 0, 0, 0, -9.8, 0, 0, 0]])
        with pytest.raises(ValueError, match="increasing"):
            load_imu_csv(p)

    def test_empty_file_rejected(self, tmp_path):
        p = tmp_path / "imu.csv"
        p.write_text("t,gx,gy,gz,ax,ay,az\n")
        with pytest.raises(ValueError):
            load_imu_csv(p)


class TestDataFrameLoader:
    def test_roundtrip(self):
        df = pd.DataFrame({
            "t": [0.0, 0.01], "gx": [0.1, 0.1], "gy": [0.2, 0.2], "gz": [0.3, 0.3],
            "ax": [0.0, 0.0], "ay": [0.0, 0.0], "az": [-9.8, -9.8],
        })
        d = imu_from_dataframe(df)
        assert isinstance(d, ImuData) and len(d) == 2 and d.mag is None

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match="DataFrame"):
            imu_from_dataframe({"t": [0.0]})
