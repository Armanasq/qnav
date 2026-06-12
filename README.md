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
pytest                                    # 248 tests, ~45 s
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
from qnav.filters import Eskf
import numpy as np

f = Eskf(
    gyro_noise_density=0.005,   # rad/s/√Hz  (datasheet: in-run noise)
    gyro_bias_walk=1e-5,        # rad/s²/√Hz (bias instability slope)
    nav_frame="NED",
)

for gyro, accel, mag in sensor_stream:
    f.predict(gyro, dt=0.01)
    f.update_gravity(accel, sigma=0.02)
    f.update_magnetometer(mag_nav_reference, mag, sigma=0.02)

print(f"attitude std (roll/pitch/yaw): {np.rad2deg(f.attitude_std)} deg")
print(f"gyro bias estimate: {np.rad2deg(f.bias * 3600)} deg/hr")
```

### Attitude determination (Wahba problem)

```python
from qnav.determination import quest

# Gravity (NED: [0,0,1] down) and magnetic field in nav frame
v_nav = np.array([[0, 0, 1.0], [0.3, 0.0, 0.95]])
v_nav /= np.linalg.norm(v_nav, axis=1, keepdims=True)

# Same vectors measured in body frame (noisy)
v_body = np.array([[...], [...]])

q_nav_body = quest.solve(v_nav, v_body, weights=[1.0, 0.5])
```

### Typed frame transforms

```python
from qnav.frames import Frame, FrameTransform

LIDAR = Frame("LIDAR")
IMU   = Frame("IMU")
BODY  = Frame("BODY")

T_body_imu  = FrameTransform(q=q_body_imu,  t=t_body_imu,  from_frame=IMU,  to_frame=BODY)
T_body_lidar = FrameTransform(q=q_body_lidar, t=t_body_lidar, from_frame=LIDAR, to_frame=BODY)

# apply_vector checks frame consistency at runtime
v_body = T_body_imu.apply_vector(v_imu)

# Compose: raises FrameMismatchError if the chain doesn't close
T_lidar_imu = T_body_imu.inverse().compose(T_body_lidar.inverse())
```

### WGS-84 geodesy

```python
from qnav.frames.earth import geodetic_to_ecef, dcm_ecef_to_ned, normal_gravity

lat, lon, h = np.deg2rad(48.8566), np.deg2rad(2.3522), 35.0  # Paris
r_ecef = geodetic_to_ecef(lat, lon, h)
R_ned_ecef = dcm_ecef_to_ned(lat, lon)
g = normal_gravity(lat, h)   # Somigliana + free-air, m/s²
```

### Magnetometer calibration

```python
from qnav.calibration.mag_ellipsoid import fit_ellipsoid, MagCalibration

# raw_data: (N, 3) array of magnetometer readings during arbitrary rotation
ellipsoid = fit_ellipsoid(raw_data)          # SVD quadric fit → SPD validation
cal = MagCalibration(ellipsoid)
corrected = cal.correct(raw_data)            # maps sphere, compensates soft+hard iron
```

---

## Attitude determination: algorithm selection guide

| Scenario | Recommended | Why |
|---|---|---|
| Exactly 2 reference vectors, no noise weighting needed | `triad` | O(1), deterministic, closes analytically |
| ≥ 2 vectors, weighted, real-time embedded | `quest` | Newton iteration on the characteristic polynomial; ~5 µs |
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
- **30+ Hypothesis property tests** on quaternion algebra (associativity, conjugate, double-cover, exp/log round-trip across random inputs)
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
