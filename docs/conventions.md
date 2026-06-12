# qnav Conventions

This document is the **normative contract** for every module in qnav. All public APIs,
documentation, and tests must conform to it. Any deviation must be explicit in the API
name or signature — never implicit.

---

## 1. Quaternions

| Property | qnav convention |
|---|---|
| Algebra | **Hamilton** ( `i·j = k` , right-handed composition ) |
| Memory layout | **Scalar-first**: `q = [w, x, y, z]` |
| Norm | Unit quaternions for attitude; norm policy per function (documented) |
| Double cover | `q` and `−q` encode the same rotation; canonicalization (`w ≥ 0`) is **opt-in**, never silent |

qnav never uses the JPL convention internally. `qnav.attitude.quaternion.from_jpl` /
`to_jpl` and `from_scalar_last` / `to_scalar_last` are the only sanctioned bridges to
other conventions.

Source basis: Solà, *Quaternion kinematics for the error-state Kalman filter*
(`__data/Quaternion kinematics...`); project baseline `__data/math.md`.

## 2. Rotation semantics: what does `q_AB` mean?

A rotation in qnav is always labelled by a **target frame A** and **source frame B**:

```
v_A = R(q_AB) · v_B
```

- `q_AB` (equivalently `R_AB`) maps the **coordinates** of a fixed vector from frame B
  to frame A — a *passive* coordinate transformation.
- The same matrix, read as an operator inside a single frame, *actively* rotates a
  vector by the rotation that carries frame A's axes onto frame B's axes.
- Composition: `q_AC = q_AB ⊗ q_BC` (chain adjacent indices).
- Inverse: `q_BA = q_AB*` (conjugate, for unit quaternions).

Convenience aliases used throughout docs and code:

- `q_WB` ("world from body") — body-to-world attitude. For a body-mounted sensor
  measuring `v_B`, the world-frame value is `v_W = R(q_WB) v_B`. This matches
  `__data/math.md` and Solà's Hamilton usage.

Functions that *rotate vectors* are named `rotate_vector` (active reading) and
functions that *change coordinates between declared frames* live in `qnav.frames`
where source/target are checked at runtime.

## 3. Rotation matrices (DCMs)

- `R ∈ SO(3)`: `Rᵀ R = I`, `det R = +1`, right-handed.
- `R(q)` for `q = [w, x, y, z]`:

```
R = [[1−2(y²+z²),  2(xy−wz),   2(xz+wy)],
     [2(xy+wz),    1−2(x²+z²), 2(yz−wx)],
     [2(xz−wy),    2(yz+wx),   1−2(x²+y²)]]
```

- DCM → quaternion uses **Shepperd's method** (branch on largest diagonal entry) for
  numerical stability; never the naive trace-only formula.
- Orthogonality repair (`dcm.orthonormalize`) uses the SVD polar projection
  `R ← U Vᵀ` (with determinant correction) and is **never applied silently**.

## 4. Euler angles

- Default sequence: **intrinsic Z-Y′-X″** (yaw ψ, pitch θ, roll φ) — aerospace
  standard. `R_AB = Rz(ψ) Ry(θ) Rx(φ)` maps B→A coordinates.
- Every Euler API takes/returns an explicit `seq` string (e.g. `"ZYX"`, intrinsic;
  lowercase `"zyx"` = extrinsic) and operates in **radians**.
- Gimbal lock: |pitch| within `gimbal_tol` of π/2 (proper sequences: angle 0/π) raises
  a `GimbalLockWarning` and resolves the lost degree of freedom by setting the third
  rotation to zero — documented, deterministic behavior.
- An intrinsic sequence `"ABC"` equals the extrinsic sequence `"cba"` reversed; tests
  enforce this identity.

## 5. SO(3) tangent space

- `hat(ω)` maps `ℝ³ → so(3)` skew matrices; `vee` is its inverse.
- `Exp(θ) = exp(hat(θ))`, `Log(R) = vee(log(R))` with rotation vector `θ = θ·u`.
- Small-angle thresholds: series expansions are used for `θ < 1e-4` rad (documented
  per function); `Log` near `θ = π` uses the stable largest-axis branch.
- Right (`Jr`) and left (`Jl`) Jacobians follow Solà/Chirikjian:
  `Exp(θ + δ) ≈ Exp(θ) Exp(Jr(θ) δ)`, `Jl(θ) = Jr(−θ) = Jr(θ)ᵀ`.
- `boxplus(R, δ) = R · Exp(δ)` (**right/local** perturbation) and
  `boxminus(R₁, R₂) = Log(R₂⁻¹ R₁)`; left variants are separate, explicitly named
  functions. Attitude covariances are 3×3 over the **right/local** tangent unless a
  function says otherwise.

## 6. Coordinate frames

| Frame | Definition | Handedness |
|---|---|---|
| **ECI** | Earth-centered inertial (J2000-style; qnav treats it kinematically) | RH |
| **ECEF** | Earth-centered, Earth-fixed; x → 0°N 0°E, z → north pole | RH |
| **NED** | Local tangent: x North, y East, z Down | RH |
| **ENU** | Local tangent: x East, y North, z Up | RH |
| **FRD** | Aircraft/marine body: x Forward, y Right, z Down (pairs with NED) | RH |
| **FLU** | Robotics body (ROS REP-103): x Forward, y Left, z Up (pairs with ENU) | RH |

- qnav supports **both NED and ENU as first-class**; nothing defaults silently.
  Frame-aware APIs require a frame token (e.g. `frame="NED"`).
- `R_ENU_NED = [[0,1,0],[1,0,0],[0,0,−1]]` (its own inverse), `R_FLU_FRD = diag(1,−1,−1)`.
- Geodetic ↔ ECEF uses the WGS-84 ellipsoid; NED/ENU rotations from ECEF are functions
  of geodetic latitude/longitude.
- Left-handed frames are **not representable**; constructing one raises.

## 7. Gravity and magnetic field

- Gravity vector in NED: `g_NED = [0, 0, +g]` (down is +z); in ENU: `[0, 0, −g]`.
  An accelerometer at rest measures **specific force** `f = −g` (i.e. `+g` "up" in
  sensor terms); every sensor model states this sign explicitly.
- Magnetic field: declination D (positive east), inclination/dip I (positive down,
  northern hemisphere typical). NED components
  `m_NED = B·[cos I cos D, cos I sin D, sin I]`.
- Headings are clockwise-from-north (compass convention) in `qnav.heading`,
  wrapped to `[0, 2π)`; yaw in `qnav.attitude.euler` is the mathematical ZYX angle.
  Conversion helpers are provided and tested.

## 8. Units, shapes, and dtypes

- **Radians**, **meters**, **seconds**, **Tesla** (magnetics may be unit-free where only
  direction matters — documented per function). Degrees only via explicit
  `np.deg2rad`/`np.rad2deg` at user boundaries.
- All array functions are **vectorized over leading batch dimensions**: a quaternion
  argument has shape `(..., 4)`, a DCM `(..., 3, 3)`, a vector `(..., 3)`.
- Computations are float64. Functions never mutate inputs.

## 9. Numerical policy

- Every function that requires unit-norm input states its policy: `assume_normalized`
  (caller contract), or it normalizes defensively and documents the tolerance.
- `arccos`/`arcsin` arguments are always clamped to `[−1, 1]`.
- Division guards use documented epsilons; behavior at the singular point is defined
  (limit value), not NaN.
- Failure behavior: invalid frames or non-conformable transforms raise typed
  exceptions (`FrameMismatchError`); degraded numerical conditions issue warnings
  (`GimbalLockWarning`, `DegenerateGeometryWarning`) — never silent.

## 10. Estimators

Every filter declares: state definition, error definition (for ESKF: right/local
tangent error), process and measurement noise semantics, frame of each measurement,
and covariance meaning. Filters are **stepwise objects** (`predict`/`update`), not
batch constructors; batch helpers wrap them.

## 11. Traceability

Every implemented formula cites a source from `docs/source_index.md`, a standard
identity, or an in-repo derivation in `docs/math/`. See `docs/math/formula_catalog.md`.
