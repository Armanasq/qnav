# Changelog

All notable changes to qnav are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
the policy in README.md ("Public API, versioning, and deprecation").

## [Unreleased]

### Added
- Centralized input validation (`qnav._validate`, internal): finiteness,
  shape, unit-norm, rotation-matrix, positive-dt, monotonic-timestamp, and
  covariance (symmetry + PSD) checks with explicit `ValueError`/`TypeError`.
- All attitude filters now validate `predict(omega, dt)` inputs at the shared
  `AttitudeFilter.predict` boundary: non-finite rates, wrong shapes, and
  non-positive time steps raise `ValueError` instead of corrupting state.
- `Eskf` validates constructor inputs (noise densities >= 0, `P0` symmetric
  PSD 6x6) and measurement updates (`sigma` > 0, non-zero direction norms).
- Public type aliases: `qnav.types.ArrayLike`, `FloatArray`, `ScalarOrArray`.
- `py.typed` marker: qnav's inline annotations are now visible to mypy/pyright.
- Public API definition: `qnav.__all__` exports all subpackages and
  exceptions; import surface protected by `tests/test_public_api.py`.
- All README and getting-started examples are executed in CI
  (`tests/test_doc_examples.py`).
- CI: mypy job, NumPy 1.22 minimum-dependency job, NumPy 2.x latest job,
  Python 3.13, package build + metadata validation + clean-venv wheel
  installation smoke test, coverage floor (88%).

### Changed
- Scalar-accepting geodesy/heading/interpolation signatures retyped from
  `np.ndarray` to `ScalarOrArray`/`ArrayLike` (behavior unchanged).
- Version is now single-sourced from `qnav.__version__`
  (`pyproject.toml` reads it dynamically).

### Fixed
- README examples used non-existent APIs (`quest.solve`,
  `MagCalibration(ellipsoid)`, `FrameTransform(q=..., t=..., from_frame=...)`);
  all examples now run against the real API.
- `docs/getting_started.md` referenced non-existent
  `overlapping_allan_variance`, `check_all`, and a `magnitude=` keyword;
  corrected to `allan_deviation`, `qnav.validation.invariants` functions, and
  `intensity=`.
- Removed unverified performance and test-count claims from documentation.

## [0.1.0] - 2026-07-04

Initial release: attitude representations and SO(3) math, typed frames and
WGS-84 geodesy, heading, sensor models, calibration, 9 attitude-determination
solvers, 10 attitude filters, WMM2025, simulation, metrics, validation.
