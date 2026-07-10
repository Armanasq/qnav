# qnav

**Convention-safe attitude, heading, frame-transform, and navigation math for robotics, aerospace, and sensor fusion.**

qnav is built around one premise: *convention errors are runtime errors*. Every rotation, every frame transform, and every filter state has an explicit, tested, documented convention. Nothing defaults silently. The library is pure NumPy, vectorized over arbitrary batch dimensions, and has no optional dependencies for its core math.

[![CI](https://github.com/armanasq/qnav/actions/workflows/ci.yml/badge.svg)](https://github.com/armanasq/qnav/actions/workflows/ci.yml)
[![Docs](https://github.com/armanasq/qnav/actions/workflows/docs.yml/badge.svg)](https://armanasq.github.io/qnav/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why qnav?

Attitude math has a notorious silent-bug problem. The same physical rotation can be represented as Hamilton or JPL quaternions, active or passive DCMs, ZYX or XYZ Euler angles — and **none of these choices cause a runtime error**. The error only surfaces as drift in a deployed system, often thousands of lines of code from the mixing point.

qnav addresses this at the architecture level:

- **Labeled frames everywhere.** Transforms carry source/target frame tags checked at composition. The graph of known transforms raises on ambiguous paths rather than silently picking one.
- **One quaternion convention, explicitly enforced.** Hamilton, scalar-first `[w,x,y,z]`. JPL and scalar-last forms are available only through explicitly-named bridge functions (`from_jpl`, `from_scalar_last`).
- **No implicit unit-norm assumptions.** Functions that require unit quaternions say so. Functions that normalize say so. Nothing renormalizes silently and discards your error signal.
- **Typed errors and warnings.** `FrameMismatchError`, `GimbalLockWarning`, `DegenerateGeometryWarning` — failure modes are named and testable, not silent `NaN` propagation.

---

## What's inside

```
qnav/
├── attitude/          SO(3) math: quaternion, DCM, Euler, rotation vector, MRP,
│                      SO(3) Lie group (exp/log/Jacobians), kinematics, SLERP, Jacobians, covariance
├── frames/            Typed coordinate frames: FrameTransform, FrameGraph, WGS-84 geodesy,
│                      NED/ENU/ECEF/ECI, FRD/FLU, ROS REP-103, aerospace stability/wind frames
├── heading/           Tilt-compensated compass, magnetic field model, declination, disturbance gates
├── sensors/           IMU noise model (white noise, random walk, Gauss-Markov), Allan variance,
│                      gyro/accel/magnetometer models, sensor alignment, lever-arm correction
├── calibration/       Gyro static bias estimation, magnetometer ellipsoid fit (hard/soft-iron),
│                      accelerometer 6-position calibration, frame alignment from vector pairs
├── determination/     Wahba problem, TRIAD, Davenport q-method, QUEST, SVD, OLEQ, FLAE,
│                      and closed-form acc+mag solvers: SAAM, FAMC, FQA
├── filters/           Complementary, Mahony, Madgwick-style, AQUA, Fourati (LM observer),
│                      ROLEQ, Fast KF, quaternion EKF, ESKF (6-state δθ+bias), UKF on SO(3)
├── geomag/            World Magnetic Model (WMM2025): spherical-harmonic field synthesis,
│                      declination/inclination/intensity + secular variation
├── simulation/        Rigid-body dynamics (RK4 Euler equations), IMU synthesis, trajectory
│                      generators, noise injection (dropout, outliers, jitter), vehicle state
├── metrics/           Attitude error (geodesic, RMSE), NEES/χ² consistency bounds (SciPy-free),
│                      heading error, covariance diagnostics
├── nav/               Inertial navigation: NavState, NED/ECEF strapdown mechanization
│                      (Earth rate, transport, Coriolis, Somigliana gravity), coning/sculling,
│                      15-state ESKF, modular measurement models (GNSS, baro, ZUPT, UWB, ...),
│                      Forster-style IMU preintegration
└── validation/        Mathematical invariants, reference cases against closed-form solutions,
                       benchmark runner, canonical MARG dataset
```

---

## Installation

```bash
pip install qnav                          # core (NumPy only)
pip install "qnav[test]"                  # + pytest, hypothesis
pip install "qnav[docs]"                  # + MkDocs Material
pip install "qnav[dev]"                   # everything + ruff, mypy
```

From source:

```bash
git clone https://github.com/armanasq/qnav
cd qnav
pip install -e ".[dev]"
pytest
```

---

## Core conventions (summary)

> Full normative specification: [`docs/conventions.md`](docs/conventions.md)

| Topic | Convention |
|---|---|
| Quaternion algebra | **Hamilton** `ij=k` |
| Memory layout | **Scalar-first** `[w, x, y, z]` |
| Rotation semantics | **Passive** coordinate transform: `v_A = R(q_AB) v_B` |
| Composition | `q_AC = q_AB ⊗ q_BC` (chain adjacent indices) |
| SO(3) perturbation | **Right/local**: `q_true = q̂ ⊗ Exp(δθ)` |
| Covariance | 3×3 over the **right/local** tangent |
| Euler angles | Explicit `seq` string; `"ZYX"` = intrinsic yaw-pitch-roll |
| Angle unit | **Radians** everywhere; degrees only at explicit user boundaries |
| Gravity NED | `[0, 0, +g]` (down is +z); accelerometer reads specific force = −g |
| Heading | Clockwise-from-north, `[0, 2π)`; distinct from yaw |

---

## Usage examples

### Quaternion algebra

```python
import numpy as np
from qnav.attitude import quaternion as quat

# Rotation: 90° about z
q_AB = quat.exp(np.array([0.0, 0.0, np.pi / 2]))

# Rotate a vector from frame B to frame A
v_B = np.array([1.0, 0.0, 0.0])
v_A = quat.rotate_vector(q_AB, v_B)   # → [0, 1, 0]

# Compose: q_AC = q_AB ⊗ q_BC (apply B→C first, then A→B)
q_BC = quat.exp(np.array([np.pi / 4, 0.0, 0.0]))
q_AC = quat.mul(q_AB, q_BC)

# Geodesic distance (sign-invariant, double-cover safe)
print(np.rad2deg(quat.angular_distance(q_AB, q_AC)))  # 45°
```

### SO(3) Lie group operations

```python
import numpy as np
from qnav.attitude import so3

R = so3.exp(np.array([0.1, -0.2, 0.3]))   # Rodrigues, exact

# Log: 3-branch (small-angle, generic, near-π), all numerically stable
phi = so3.log(R)                            # rotation vector

# Right Jacobian: linearizes Exp(θ + δ) ≈ Exp(θ)·Exp(Jr(θ) δ)
Jr = so3.right_jacobian(phi)

# Manifold retraction: R ⊞ δ = R·Exp(δ)
R_new = so3.boxplus(R, np.array([0.01, 0.0, 0.0]))
```

### Error-state Kalman filter

```python
import numpy as np
from qnav.filters import Eskf
from qnav.heading.magnetic_model import field_from_elements

rng = np.random.default_rng(0)
dt = 0.01
m_nav = field_from_elements(declination=0.0, inclination=np.deg2rad(60.0))

f = Eskf(
    gyro_noise_density=0.005,   # rad/s/√Hz  (datasheet: in-run noise)
    gyro_bias_walk=1e-5,        # rad/s²/√Hz (bias instability slope)
    nav_frame="NED",
)

for k in range(200):            # static body, noisy sensors
    gyro = 0.005 * rng.standard_normal(3)
    accel = np.array([0.0, 0.0, -9.81]) + 0.05 * rng.standard_normal(3)
    mag = m_nav + 0.02 * rng.standard_normal(3)
    f.predict(gyro, dt=dt)
    f.update_gravity(accel, sigma=0.02)
    f.update_magnetometer(m_nav, mag, sigma=0.02)

print(f"attitude std (x/y/z): {np.rad2deg(f.attitude_std)} deg")
print(f"gyro bias estimate: {np.rad2deg(f.bias) * 3600} deg/hr")
```

### Attitude determination (Wahba problem)

```python
import numpy as np
from qnav.attitude import quaternion as quat
from qnav.determination import quest_q

# Gravity (NED: [0,0,1] down) and magnetic field direction in nav frame
v_nav = np.array([[0.0, 0.0, 1.0], [0.3, 0.0, 0.95]])
v_nav /= np.linalg.norm(v_nav, axis=1, keepdims=True)

# The same vectors observed in the body frame (here: body rotated 30° in yaw)
q_true = quat.exp(np.array([0.0, 0.0, np.deg2rad(30)]))
v_body = np.stack([quat.rotate_frame(q_true, v) for v in v_nav])

q_nav_body = quest_q(v_nav, v_body, weights=np.array([1.0, 0.5]))
assert quat.angular_distance(q_nav_body, q_true) < 1e-9
```

### Typed frame transforms

```python
import numpy as np
from qnav.attitude import quaternion as quat
from qnav.frames import FrameTransform

q_body_imu = quat.exp(np.array([0.0, 0.0, np.deg2rad(5)]))   # 5° mounting yaw
t_body_imu = np.array([0.10, 0.0, -0.05])                     # lever arm [m]

T_body_imu = FrameTransform(target="BODY", source="IMU",
                            rotation=q_body_imu, translation=t_body_imu)

v_imu = np.array([1.0, 0.0, 0.0])
v_body = T_body_imu.apply_vector(v_imu)

# Composition checks frame labels; a non-matching chain raises FrameMismatchError
T_imu_body = T_body_imu.inverse()
identity = T_body_imu @ T_imu_body        # T_body_body
```

### WGS-84 geodesy

```python
import numpy as np
from qnav.frames.earth import geodetic_to_ecef, dcm_ecef_to_ned, normal_gravity

lat, lon, h = np.deg2rad(48.8566), np.deg2rad(2.3522), 35.0  # Paris
r_ecef = geodetic_to_ecef(lat, lon, h)
R_ned_ecef = dcm_ecef_to_ned(lat, lon)
g = normal_gravity(lat, h)   # Somigliana + free-air, m/s²
```

### Full inertial navigation (15-state ESKF)

```python
import numpy as np
from qnav.attitude import quaternion as quat
from qnav.nav import NavEskf, NavState
from qnav.nav.measurements import BaroAltitude, GnssPosition, ZuptVelocity

lat, lon, h = np.deg2rad(48.85), np.deg2rad(2.35), 35.0
f = NavEskf(
    NavState(q=quat.identity(), p=[lat, lon, h], frame="NED"),
    gyro_noise_density=0.002,    # rad/s/√Hz
    accel_noise_density=0.02,    # m/s²/√Hz
    gyro_bias_walk=1e-6, accel_bias_walk=1e-5,
)

rng = np.random.default_rng(0)
for k in range(500):             # 5 s static at 100 Hz
    gyro = 0.002 * rng.standard_normal(3)
    accel = np.array([0.0, 0.0, -9.806]) + 0.02 * rng.standard_normal(3)
    f.predict(gyro, accel, dt=0.01)
    if k % 100 == 99:            # 1 Hz GNSS + baro
        f.update_measurement(GnssPosition(), np.array([lat, lon, h]), sigma=2.0)
        f.update_measurement(BaroAltitude(), h, sigma=0.5)
    f.update_measurement(ZuptVelocity(), None, sigma=0.02)

print("position 1σ [m]:", f.position_std)
print("health:", f.health.name)
```

Measurement models (`qnav.nav.measurements`) are modular — GNSS position/velocity with lever arms, barometric altitude, rangefinder, external attitude/pose/velocity, wheel odometry, nonholonomic constraints, ZUPT/ZARU, UWB ranges, magnetic yaw, dual-antenna heading — each with a documented frame/unit contract and a finite-difference-verified Jacobian. All fuse through one gated Joseph-form kernel with chi-square NIS gating, Huber/Cauchy/Tukey robust losses, and per-sensor quarantine (`GatePolicy`, `SensorMonitor`).

### Magnetometer calibration

```python
import numpy as np
from qnav.attitude import quaternion as quat
from qnav.calibration.mag_ellipsoid import fit_ellipsoid

# Simulate raw readings: rotations of a fixed field, plus hard-iron offset
rng = np.random.default_rng(1)
field = np.array([0.2, 0.0, 0.45])
qs = quat.random((500,), rng=rng)
raw = np.stack([quat.rotate_frame(q, field) for q in qs]) + np.array([0.05, -0.02, 0.01])

cal = fit_ellipsoid(raw)          # SVD quadric fit → SPD validation
corrected = cal.correct(raw)      # maps to sphere: soft + hard iron compensated
assert np.std(np.linalg.norm(corrected, axis=1)) < 1e-6
```

---

## Attitude determination: algorithm selection guide

| Scenario | Recommended | Why |
|---|---|---|
| Exactly 2 reference vectors, no noise weighting needed | `triad` | O(1), deterministic, closes analytically |
| ≥ 2 vectors, weighted, real-time embedded | `quest` | Newton iteration on the characteristic polynomial |
| ≥ 2 vectors, fastest optimal solver | `flae` | Quartic characteristic polynomial; companion roots + Newton polish |
| ≥ 2 vectors, batch, numerical robustness paramount | `svd` | SVD-based; handles near-degenerate configs gracefully |
| Online alignment with N vectors and covariance output | `davenport` | Full 4×4 eigensystem; straightforward to extend |
| Overdetermined system, stacked observations | `oleq` | Left/right matrix accumulation; no eigendecomposition |
| Acc+mag pair, maximum throughput (batched) | `saam` | One square root; closed form; vectorized; no dip angle needed |
| Acc+mag pair, degeneracy diagnostics | `famc` | Analytic Davenport elimination; pivots expose collinearity |
| Acc+mag pair, magnetically hostile environment | `fqa` | Factored form: magnetic disturbance provably cannot affect tilt |

---

## Filters: what each one gives you

| Filter | State | Bias estimation | Covariance | Use when |
|---|---|---|---|---|
| `ComplementaryFilter` | q | ✗ | ✗ | Prototype; dead-reckoning not needed |
| `MahonyFilter` | q + integral bias | integral (not statistical) | ✗ | Low-resource embedded; PI tuning is intuitive |
| `MadgwickStyleFilter` | q | ✗ | ✗ | Single gain β; good default for slowly-moving platforms |
| `AquaFilter` | q | ✗ | ✗ | Magnetic disturbances must not touch roll/pitch (structural decoupling) |
| `FouratiFilter` | q | ✗ | ✗ | Fast transients; observability-scaled LM corrections |
| `RoleqFilter` | q | ✗ | ✗ | Zero tuning; linear fixed-point correction |
| `FastKalmanFilter` | q ∈ ℝ⁴ | ✗ | 4×4 P | Cheapest covariance-bearing option; algebraic measurements |
| `QuaternionEkf` | q (total state) | ✗ | Total-state P | When you need uncertainty without bias state |
| `Eskf` | q + gyro bias δθ∈ℝ⁶ | ✓ statistical | Error-state 6×6 P | Production; use when you need NEES-consistent uncertainty |
| `UkfAttitude` | q, 3×3 tangent P | ✗ | Error-state 3×3 P | Initial uncertainty > 20°; no linearization in updates |

`Eskf` is the only filter whose covariance has a documented, tested statistical meaning (right/local tangent, NEES bounds verified by Monte-Carlo in the test suite). `UkfAttitude` recovers from a 115° initial error — verified in the test suite — where ESKF linearization fails.

---

## Testing

```bash
pytest                             # full suite
pytest -k eskf                     # filter tests only
pytest tests/test_so3.py -v        # SO(3) math
pytest --hypothesis-seed=0         # property tests with fixed seed
```

The test suite covers:
- **Hypothesis property tests** on quaternion algebra (associativity, conjugate, double-cover, exp/log round-trip across random inputs)
- **Finite-difference Jacobian verification** for all analytical Jacobians in `attitude/jacobians.py` and `filters/madgwick_style.py`
- **Convergence order tests** for all integrators (measured empirically against fine-grid reference)
- **NEES consistency** for the ESKF (Monte-Carlo average NEES within χ²(3) bounds)
- **Reference cases** against closed-form solutions (Shepperd, QUEST, WGS-84 exact geodetic)
- **All 18 Euler sequences** (12 Tait-Bryan + 6 proper), including gimbal-lock recovery

---

## Documentation

```bash
mkdocs serve            # local preview at http://127.0.0.1:8000
mkdocs build            # static site to site/
```

Published at **https://armanasq.github.io/qnav/**

Math pages cover: quaternion algebra, SO(3) Lie group, Euler angles, coordinate frames, attitude determination, filtering, heading/tilt, and a complete formula catalog with source traceability to 7 primary references.

---

## Public API, versioning, and deprecation

The supported public API is: the subpackages and exceptions exported by `qnav.__all__`, plus every symbol in those subpackages' `__all__` lists. Anything prefixed with an underscore (modules like `qnav._validate`, functions, attributes) is internal and may change without notice.

qnav follows semantic versioning:

- **Patch** (0.1.x): bug fixes, documentation, numerical improvements within stated tolerances.
- **Minor** (0.x.0): new features; public behavior changes only through a deprecation cycle.
- **Deprecation cycle**: the old symbol keeps working and emits `DeprecationWarning` naming the replacement for at least one minor release before removal. Incorrect *results* (wrong math) are fixed immediately and noted in the changelog as behavior changes.

Pre-1.0 caveat: minor releases may include breaking changes, but always with a deprecation shim when technically possible. `tests/test_public_api.py` defines the import surface that these rules protect.

**Stability labels** (machine-readable in [`api_manifest.json`](api_manifest.json), enforced in CI): the attitude/frames/determination/metrics core is **stable**; the navigation stack (`qnav.nav`), estimator contracts/robustness/pipeline/invariant filter, interop, extended calibration, and validation tooling are **provisional** — functional and tested (including against real sensor datasets, see `benchmarks/results/`), but their APIs may change in a minor release while real-world validation matures.

All documentation examples are executed in CI (`tests/test_doc_examples.py`).

---

## Contributing

1. Fork and branch off `main`.
2. Run `pytest` and `ruff check qnav/ tests/`.
3. Every new formula must cite a source in `docs/conventions.md` or `docs/math/formula_catalog.md`.
4. Every public function must have a docstring stating its convention (frame, units, array shape contract).

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## References

The mathematical foundation traces to seven primary sources archived in `__data/`:

1. **Solà** — *Quaternion kinematics for the error-state Kalman filter* — primary reference for ESKF, kinematics, Lie group operations
2. **Hashim** — *Special Orthogonal Group SO(3): Overview, Mapping and Challenges* — SO(3) Lie theory, Rodrigues, all representation singularities
3. **Al-Jlailaty & Mansour** — *Efficient Attitude Estimators: A Tutorial and Survey* — Mahony, Madgwick, Allan variance, strapdown algorithms
4. **Kok, Hol & Schön** — *Using Inertial Sensors for Position and Orientation Estimation* — sensor models, TRIAD/QUEST/Davenport, EKF smoothing
5. **Parwana & Kothari** — *Quaternions and Attitude Representation* — kinematics, double-cover, MRP
6. **NIMA TR8350.2** — WGS-84 defining parameters
7. **Markley & Crassidis** — *Fundamentals of Spacecraft Attitude Determination and Control* — Wahba problem, QUEST, covariance
