"""Public API contract: import smoke tests and stability of key symbols.

These tests define the supported surface. A failure here means a breaking
change that requires a deprecation cycle per the policy in README.md.
"""

import importlib

import pytest

import qnav

SUBPACKAGES = [
    "attitude", "calibration", "determination", "filters", "frames",
    "geomag", "heading", "interop", "metrics", "nav", "sensors", "simulation", "types",
    "validation",
]

MODULES = [
    "qnav.attitude.quaternion", "qnav.attitude.so3", "qnav.attitude.dcm",
    "qnav.attitude.euler", "qnav.attitude.rotvec", "qnav.attitude.mrp",
    "qnav.attitude.kinematics", "qnav.attitude.conversions",
    "qnav.attitude.covariance", "qnav.attitude.jacobians",
    "qnav.attitude.interpolation",
    "qnav.frames.core", "qnav.frames.transforms", "qnav.frames.graph",
    "qnav.frames.earth", "qnav.frames.conventions",
    "qnav.heading.compass", "qnav.heading.tilt_compensation",
    "qnav.heading.declination", "qnav.heading.disturbance",
    "qnav.geomag.wmm",
    "qnav.filters.base", "qnav.filters.contracts",
    "qnav.nav.state", "qnav.nav.mechanization", "qnav.nav.increments",
    "qnav.nav.eskf", "qnav.nav.measurements", "qnav.nav.preintegration",
    "qnav.interop.datasets",
    "qnav.sensors.imu", "qnav.sensors.allan", "qnav.sensors.noise",
    "qnav.calibration.gyro_bias", "qnav.calibration.mag_ellipsoid",
    "qnav.calibration.accel_calibration", "qnav.calibration.frame_alignment",
    "qnav.simulation.trajectories", "qnav.simulation.imu_synthesis",
    "qnav.metrics.attitude_error", "qnav.metrics.covariance_consistency",
]


def test_version_is_string():
    assert isinstance(qnav.__version__, str) and qnav.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_accessible(name):
    assert getattr(qnav, name) is not None


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod):
    importlib.import_module(mod)


def test_all_exports_resolve():
    for name in qnav.__all__:
        assert hasattr(qnav, name), name


def test_key_symbols():
    from qnav.determination import (  # noqa: F401
        davenport_q, famc_q, flae_q, fqa_q, oleq_q, quest_q, saam_q,
        svd_attitude, triad_dcm,
    )
    from qnav.filters import (  # noqa: F401
        AquaFilter, AttitudeFilter, ComplementaryFilter, Eskf,
        FastKalmanFilter, FouratiFilter, MadgwickStyleFilter, MahonyFilter,
        QuaternionEkf, RoleqFilter, UkfAttitude,
    )
    from qnav.frames import Frame, FrameGraph, FrameTransform  # noqa: F401


def test_py_typed_shipped():
    import pathlib

    assert (pathlib.Path(qnav.__file__).parent / "py.typed").exists()
