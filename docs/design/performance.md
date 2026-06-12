# Performance

qnav is not a numerics performance library. It is a correctness library. But correctness should not require sacrifice in performance for typical workloads, and the design supports efficient use.

## Batched operations

The single most important performance feature is batch vectorization. Processing 1000 quaternions should take the same number of Python calls as processing 1:

```python
# Do this — one call, one NumPy kernel
qs_rotated = quat.rotate_vector(q_batch, v_batch)   # q_batch: (1000, 4), v_batch: (1000, 3)

# Not this — 1000 Python function calls
for q, v in zip(q_batch, v_batch):
    v_rot = quat.rotate_vector(q, v)
```

All qnav functions support arbitrary leading batch dimensions through NumPy broadcasting. The key patterns internally are `[..., :]` indexing, `np.linalg.norm(x, axis=-1)`, and `np.cross(a, b)` (which broadcasts natively).

## Memory layout

qnav uses **float64** throughout. This matches NumPy's default and avoids silent precision loss from mixed-precision arithmetic. For embedded targets where float32 is required, cast at the boundary and be aware that some operations (quaternion log near θ = 0, SO(3) Jacobians near singular points) will accumulate error faster in float32.

Arrays are row-major (C order), which is NumPy's default. For matrix chains `A @ B @ C`, the cost is dominated by the matrix multiplications, not the layout.

## Avoid repeated conversions

Converting between representations has cost. Profile before optimizing, but common anti-patterns:

```python
# Inefficient: converts to DCM on every update
for omega in gyro_stream:
    R = dcm.from_quaternion(f.q)
    # ... use R
    f.predict(omega, dt)

# Better: convert once per display or output step
# The filter state is a quaternion; use rotate_vector/rotate_frame directly
v_body = quat.rotate_frame(f.q, v_nav)   # no DCM needed
```

## Filter loop

The ESKF predict step is the hot path in a real-time loop. At 1 kHz with `gyro_bias_walk=0` (no bias drift), the predict step is:

1. One quaternion multiplication (`quat.mul`)
2. One SO(3) exponential (`so3.exp`) — 3×3 Rodrigues
3. One SO(3) right Jacobian (`so3.right_jacobian`) — 3×3 closed form
4. One 6×6 matrix multiplication (`F @ P @ F.T + Qd`)

All of these are O(1) operations on small fixed-size arrays. At 1 kHz, the predict step takes ~10 µs on a modern laptop. The update step (3×3 system solve) adds ~5 µs. These numbers are well within real-time budget for most platforms.

## The O(N²) bottleneck in SO(3) log

The `so3.log` function has a Python loop over the batch dimension. This is a known limitation and a deliberate trade-off: the three-branch logic (small angle, generic, near-π) is difficult to vectorize cleanly in pure NumPy without introducing masked operations that are harder to read and test.

For batch sizes up to ~1000, the Python loop is fast enough (< 1 ms). For larger batches in offline processing, consider:

```python
# Vectorized alternative for the generic branch only (no near-pi handling):
tr = np.trace(R_batch, axis1=-2, axis2=-1)
theta = np.arccos(np.clip((tr - 1) / 2, -1, 1))
skew = R_batch - R_batch.swapaxes(-1, -2)
phi = (theta / (2 * np.sin(theta + 1e-300)))[..., None, None] * skew
result = np.stack([phi[..., 2, 1], phi[..., 0, 2], phi[..., 1, 0]], axis=-1)
```

This misses the near-π branch; use only when you can guarantee angles < 170°.

## Benchmark runner

```python
from qnav.validation.benchmark_runner import run_benchmarks
results = run_benchmarks()
```

This runs timing benchmarks for:

- Quaternion multiply, rotate, log, exp (batch sizes 1, 100, 10000)
- SO(3) exp, log, Jacobians
- ESKF predict + update cycle
- QUEST solver (2-vector and N-vector)
- Magnetometer ellipsoid fit

Results are printed as a table. There are no performance regression tests in CI — performance is hardware-dependent and not the primary contract.

## Pure NumPy means no JIT

qnav has no Numba, Cython, or C extensions. This keeps the installation trivial and the code auditable.

For applications where the filter runs faster than Python can call NumPy, the recommended path is to use qnav's **tested reference implementations** to validate a Cython or C++ reimplementation, rather than running qnav in the hot loop. The test suite is designed to support this: every algorithm has reference cases that can be used as acceptance criteria.
