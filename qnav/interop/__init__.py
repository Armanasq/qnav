"""Optional interoperability adapters.

The qnav core stays NumPy-only; everything here degrades to a clear
``ImportError`` naming the missing dependency. Install extras with
``pip install "qnav[interop]"``.

Available adapters:

- :mod:`qnav.interop.scipy_rotation` — lossless bridges to/from
  ``scipy.spatial.transform.Rotation`` (scalar-first Hamilton <-> SciPy's
  scalar-last layout, batch-capable)
- :mod:`qnav.interop.datasets` — CSV (NumPy-only) and pandas DataFrame
  loaders producing validated, monotonic IMU arrays ready for the filters
  and :class:`~qnav.filters.pipeline.FusionPipeline`

Deferred (not shipped until they can be tested against the real packages):
ROS 2 message/tf adapters and GTSAM rotation/pose/preintegration bridges.
qnav will not ship adapter code that its CI cannot execute.
"""

from qnav.interop import datasets, scipy_rotation  # noqa: F401

__all__ = ["datasets", "scipy_rotation"]
