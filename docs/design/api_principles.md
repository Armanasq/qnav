# API Principles

These principles govern every public interface in qnav. They are not style preferences — each one prevents a specific class of bugs.

## 1. Conventions are data, not settings

No global state. No `set_convention()`. No `USE_JPL=True` environment variable.

Every function that depends on a convention takes it as an explicit argument. `euler.to_dcm(angles, seq="ZYX")` — the sequence is the third argument, not a module-level default. `Eskf(nav_frame="NED")` — the nav frame is a constructor argument.

This means convention ambiguity becomes a type error or a missing argument, not a wrong answer.

## 2. Labeled frames

Frame labels are not strings for documentation — they are checked at runtime.

`FrameTransform.compose(other)` verifies that `self.from_frame == other.to_frame`. `FrameTransform.apply_vector(v)` knows what frame `v` should be in. Mismatches raise `FrameMismatchError` with a message that names both frames and the operation that failed.

This turns the category of "I accidentally used a body-frame measurement in a world-frame function" from a silent wrong answer into an immediate, specific exception.

## 3. Fail loudly, fail specifically

Silent degradation is a defect. qnav has five exception types and three warning types, all in `qnav.errors`:

| Type | When |
|---|---|
| `FrameMismatchError` | Frame-incompatible transform composition or vector application |
| `CalibrationError` | Not enough data for a calibration (e.g., too few static samples) |
| `ConventionError` | Invalid Euler sequence, unrecognized frame name, etc. |
| `NumericalError` | Algorithm failure (e.g., SVD did not converge) |
| `GimbalLockWarning` | Middle Euler angle within `gimbal_tol` of the singular value |
| `DegenerateGeometryWarning` | Free-fall in tilt estimation, collinear attitude observations |
| `NormalizationWarning` | Quaternion norm deviates from 1 by more than a specified tolerance |

Every failure mode is documented per function. The distinction between error (unrecoverable) and warning (handled, but caller should know) is explicit.

## 4. No silent normalization

Functions that require unit-norm inputs document their contract as one of:

- **Assume normalized** (caller's responsibility; no check at runtime — for inner loops)
- **Normalize defensively** (normalizes and documents the tolerance)
- **Raise on non-unit** (checks norm and raises `ValueError` if too far from 1)

The `normalize(q, warn_tol=tol)` function is the sanctioned way to normalize: it raises on near-zero norm (indicating a degenerate input) and optionally warns when the norm is suspiciously far from 1 (indicating accumulated drift that should be investigated, not silently corrected).

## 5. Batch dimensions are first-class

Every array operation supports arbitrary leading batch dimensions. A `(..., 4)` quaternion array works identically to a `(4,)` scalar quaternion. This is not a convenience feature — it is the contract. Functions that break on batched inputs are defects.

The implementation pattern: operations on the last axis (via `[..., :]` indexing, `np.linalg.norm(..., axis=-1)`, `np.stack(..., axis=-1)`) rather than a Python loop over the leading dimensions.

## 6. Explicit unit bridges

Unit ambiguity is handled the same way as convention ambiguity: through explicitly named functions. `from_jpl`, `to_jpl`, `from_scalar_last`, `to_scalar_last` are the only interfaces to other quaternion conventions. There is no implicit conversion.

Similarly: all function signatures specify their unit in the docstring (radians, meters, Tesla, rad/s). Degree inputs go through `np.deg2rad` at the call site — they never enter the library.

## 7. Immutability

Functions never mutate their inputs. Every function returns new arrays. The one exception: `canonical(q)` returns a modified copy when the input has `w < 0`, which is documented.

## 8. Formula traceability

Every non-trivial formula cites a source in its docstring, using the abbreviation from `docs/math/formula_catalog.md`. The citation format is `[SOLA eq.(101)]`, `[KOK eq.models-quatMult]`, etc.

Discrepancies between sources are documented in `formula_catalog.md` with a resolution. The code implements the resolved version — never silently choosing one source over another.

## 9. Tests are the spec

Where the docstring is ambiguous, the test is authoritative. Tests verify:

- Mathematical identities (round-trips, group axioms, Jacobian correctness via finite differences)
- Convergence orders of integrators and filters (measured, not just claimed)
- NEES statistical consistency of the ESKF (Monte-Carlo)
- Reference cases against closed-form solutions

Adding a feature means adding a test. Removing a test requires justification.

## 10. Minimal surface area

qnav does not try to be a navigation suite. It is a math library. It does not:

- Read sensor data from hardware
- Implement INS mechanization (integrated navigation system with position/velocity states)
- Wrap the World Magnetic Model
- Provide a Python interface to GNSS data

These are all important, but they are downstream of correct, convention-safe math primitives. Build on qnav; do not try to replace the things it doesn't do.
