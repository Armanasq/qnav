# Changelog

All notable changes to qnav are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
the policy in README.md ("Public API, versioning, and deprecation").

## [Unreleased]

### Added
- Observability-aware calibration extensions (`qnav.calibration`):
  `assess_least_squares` (SVD-based OBSERVABLE / WEAKLY_OBSERVABLE /
  UNOBSERVABLE grading with condition number, excitation metric, weakest
  direction), `estimate_time_offset` (normalized cross-correlation with
  sub-sample parabolic refinement and reliability flag),
  `fit_temperature_bias` (polynomial bias-vs-temperature with covariance;
  refuses orders the thermal sweep cannot support), and
  `estimate_lever_arm` (rigid-body LSQ with covariance; raises on
  unexciting motion). All return covariance and observability diagnostics.
- Optional interop adapters (`qnav.interop`, extras `qnav[interop]`):
  lossless SciPy `Rotation` bridge (`to_scipy`/`from_scipy`, scalar-first
  <-> scalar-last, physical-rotation equivalence tested) and dataset
  loaders (`load_imu_csv` NumPy-only, `imu_from_dataframe` for pandas)
  producing validated monotonic `ImuData`; bad rows are rejected unless
  dropping is requested explicitly. ROS 2 and GTSAM adapters are
  deliberately deferred until they can run against the real packages in CI.
- IMU preintegration (`qnav.nav.preintegration.ImuPreintegration`,
  Forster-style on-manifold): gravity-free body-frame delta rotation/
  velocity/position, first-order bias Jacobians with `corrected()`
  re-linearization, 9x9 preintegrated covariance, interval length.
  Cross-checked in tests against per-sample recursion (bit-exact), NED
  mechanization (Earth-term bounds), finite-difference bias Jacobians,
  and Monte-Carlo covariance consistency (NEES).
- Modular navigation measurement models (`qnav.nav.measurements`):
  GNSS position/velocity (lever-arm aware), barometric altitude,
  rangefinder height (tilt-compensated with attitude Jacobian), external
  attitude/pose/body-velocity, wheel speed, nonholonomic constraint, ZUPT,
  ZARU, UWB range, magnetometer yaw, dual-antenna heading. Each documents
  frame/unit contracts, observability, and failure modes; all Jacobians
  are finite-difference verified against the filter's own error
  definition. `NavEskf.update_measurement(model, value, sigma)` fuses any
  model through the shared gated kernel — no sensor branches inside the
  estimator.
- Inertial navigation stack (`qnav.nav`): immutable `NavState` (attitude,
  velocity, position, gyro/accel biases; NED-geodetic or ECEF), strapdown
  mechanization kernels with Earth rate, transport rate, Coriolis, and
  WGS-84 Somigliana gravity (`propagate_ned`/`propagate_ecef` — the single
  source of propagation math), coning/sculling-corrected IMU increments
  (`accumulate_increments`), and a 15-state error-state Kalman filter
  (`NavEskf`, error order `[dtheta, dv, dp, dbg, dba]`) with position,
  velocity, and direction updates.
- `NavEskf` shares the gated Joseph-form update kernel
  (`qnav.filters._kalman.gated_joseph_update`) and the estimator lifecycle
  (snapshot/restore/reset/health) with the attitude filters; the attitude
  `Eskf` was refactored onto the same kernel (behavior unchanged, tested).
- Numerical validation: static equilibrium (60 s), free-fall closed form,
  northward transport rate, ECEF static + NED/ECEF free-fall cross-check,
  coning drift versus the analytic `-a^2*Omega/2` rate, Monte-Carlo
  position NEES, long-run covariance symmetry/PSD.
- Documented approximation: Earth/transport-rate terms are kept in the
  nominal mechanization but omitted from the error-state Jacobian
  (O(Omega*dt) per step — below MEMS/tactical noise; not suitable for
  gyrocompass alignment).
- Timestamped fusion pipeline (`qnav.filters.FusionPipeline`): variable
  gyro dt, multi-rate sensors, duplicate gyro/measurement detection
  (sensor_id + sequence_id), gap flagging, clock-discontinuity detection
  (`ClockDiscontinuityError`), per-sensor time offsets (online-estimation
  hook), bounded delayed-measurement rollback-and-replay over a fixed-lag
  snapshot history, and SLERP attitude interpolation (`attitude_at`).
  Every call returns a `ProcessReport`; nothing is dropped silently.
  Replay is bit-exact: a delayed measurement yields the same state as
  in-order processing (tested).
- Robust measurement handling (`qnav.filters.robust`): SciPy-free
  `chi2_quantile` (exact for dof 1–2, Wilson–Hilferty otherwise),
  Huber/Cauchy/Tukey weights, immutable `GatePolicy` (chi-square NIS gate
  with hard rejection or soft covariance inflation plus optional robust
  loss), `SensorMonitor` (quarantine after N consecutive rejections,
  hysteresis recovery, timeout detection), `detect_saturation`.
- `Eskf` accepts `gate=GatePolicy(...)` and per-sensor monitors
  (`set_monitor`); rejected/quarantined measurements leave the state
  untouched and are reported via `last_update.rejection_reason`.
  Default behavior (no gate) is unchanged.
- `Eskf` recovery actions: `inflate_covariance(factor, attitude_only=...)`
  and `reinitialize_from_vectors(...)` (deterministic FQA reinit, bias
  preserved or reset, covariance reset, history cleared); rollback via the
  existing `snapshot()`/`restore()`.
- Health now detects `DIVERGING` (sustained windowed mean NIS far above its
  chi-square mean) and, for `Eskf`, `UNOBSERVABLE` (all recently fused
  directions collinear — yaw about that axis unconstrained).
- Estimator contracts (`qnav.filters.contracts`): `Measurement`,
  `UpdateResult`, `InnovationStatistics`, `EstimatorHealth`,
  `EstimatorSnapshot` — exported from `qnav.filters`.
- Every attitude filter now supports `reset()` (deterministic
  as-constructed state), `snapshot()`/`restore()` (deep, type-checked),
  and a `health` property (INITIALIZING / HEALTHY / DEGRADED / INVALID,
  covering non-finite state and asymmetric/indefinite covariance).
- `Eskf` updates populate `last_update` (`UpdateResult` with innovation,
  innovation covariance, NIS, state correction) and per-sensor
  `innovation_stats`; `update_*` methods accept optional `timestamp` and
  `sensor_id` keywords. Legacy innovation return values are unchanged.
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
