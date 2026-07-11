"""One-call batch attitude estimation from logged sensor arrays.

:func:`estimate_attitude` is the front door for the most common task in
practice: *"I have recorded gyro/accel(/mag) arrays — give me orientation
over time."* It wires up any qnav attitude filter, initializes the attitude
from the first usable accelerometer(+magnetometer) sample instead of
identity, runs the predict/update loop (uniform or per-sample timestamps),
tolerates sensor dropouts (NaN rows are skipped, never fused), and returns
an :class:`AttitudeEstimate` with quaternions, Euler/heading/DCM converters,
per-sample attitude uncertainty (covariance-bearing filters), and the final
gyro-bias estimate.

Conventions (as everywhere in qnav — see ``docs/conventions.md``):

- quaternions are scalar-first Hamilton ``q_nav_body``,
- ``nav_frame="NED"`` pairs with FRD body axes, ``"ENU"`` with FLU,
- gyro in rad/s, accelerometer is *specific force* in m/s² (at rest it
  reads ``≈ +g`` pointing away from gravity), magnetometer in any
  consistent unit,
- sample ``k`` propagates the state over ``(t[k-1], t[k]]`` with
  ``gyro[k]``; row 0 of the output is the initial attitude.

The stepwise filter API (:mod:`qnav.filters`) remains the primary interface
for online/embedded use; this module is a batch convenience layered on top
of it, not a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from qnav.attitude import dcm as _dcm
from qnav.attitude import euler as _euler
from qnav.attitude import quaternion as quat
from qnav.determination.fqa import fqa
from qnav.filters import (
    AquaFilter,
    ComplementaryFilter,
    Eskf,
    FastKalmanFilter,
    FouratiFilter,
    LeftInvariantEskf,
    MadgwickStyleFilter,
    MahonyFilter,
    QuaternionEkf,
    RoleqFilter,
    UkfAttitude,
)
from qnav.filters.base import AttitudeFilter
from qnav.filters.contracts import EstimatorHealth
from qnav.frames.conventions import attitude_ned_frd_to_enu_flu, enu_to_ned, flu_to_frd
from qnav.heading.compass import wrap_heading

__all__ = ["AttitudeEstimate", "estimate_attitude"]

#: methods accepted by :func:`estimate_attitude`, and what they map to.
_METHOD_CLASSES = {
    "eskf": Eskf,
    "invariant": LeftInvariantEskf,
    "ukf": UkfAttitude,
    "ekf": QuaternionEkf,
    "mahony": MahonyFilter,
    "madgwick": MadgwickStyleFilter,
    "complementary": ComplementaryFilter,
    "aqua": AquaFilter,
    "fourati": FouratiFilter,
    "roleq": RoleqFilter,
    "fkf": FastKalmanFilter,
}

#: methods whose per-sample correction requires BOTH accel and mag.
_NEEDS_ACC_AND_MAG = frozenset({"fourati", "roleq", "fkf"})

#: methods with a 3×3 (or 6×6) tangent-space covariance → attitude_std output.
_TANGENT_COVARIANCE = frozenset({"eskf", "invariant", "ukf"})


@dataclass(frozen=True)
class AttitudeEstimate:
    """Batch attitude-estimation result (output of :func:`estimate_attitude`).

    Attributes
    ----------
    t:
        ``(N,)`` sample times [s] (the input timestamps, or ``k·dt``).
    q:
        ``(N, 4)`` scalar-first Hamilton ``q_nav_body`` per sample; row 0 is
        the initial attitude.
    method, nav_frame:
        The estimator and navigation frame that produced the estimate.
    attitude_std:
        ``(N, 3)`` per-axis 1σ attitude error [rad] over the filter's
        tangent space, or ``None`` for filters without a tangent-space
        covariance (see the selection guide in :mod:`qnav.filters`).
    gyro_bias:
        Final gyro-bias estimate [rad/s] for bias-estimating filters
        (``eskf``, ``invariant``, ``mahony``), else ``None``.
    filter:
        The underlying stepwise filter in its final state — continue it
        online, inspect ``innovation_stats``, or ``snapshot()`` it.
    n_updates_applied / n_updates_skipped:
        Vector corrections fused vs. skipped because a sensor row was NaN
        (dropout) — dropped data is counted, never silently fused.
    """

    t: np.ndarray
    q: np.ndarray
    method: str
    nav_frame: str
    filter: AttitudeFilter
    attitude_std: Optional[np.ndarray] = None
    gyro_bias: Optional[np.ndarray] = None
    n_updates_applied: int = 0
    n_updates_skipped: int = 0

    def __len__(self) -> int:
        return int(self.q.shape[0])

    def euler(self, seq: str = "ZYX", degrees: bool = False) -> np.ndarray:
        """``(N, 3)`` Euler angles in ``seq`` order (``"ZYX"`` → yaw, pitch,
        roll); radians unless ``degrees``."""
        e = _euler.from_quaternion(self.q, seq)
        return np.rad2deg(e) if degrees else e

    def dcm(self) -> np.ndarray:
        """``(N, 3, 3)`` direction cosine matrices ``R_nav_body``."""
        return _dcm.from_quaternion(self.q)

    def heading(self, degrees: bool = False) -> np.ndarray:
        """``(N,)`` heading (clockwise from north, ``[0, 2π)``).

        NED/FRD: heading = yaw. ENU/FLU: yaw is counterclockwise from east,
        so heading = π/2 − yaw. Magnetic vs. true depends on the magnetic
        reference used (see :func:`estimate_attitude`).
        """
        yaw = _euler.from_quaternion(self.q, "ZYX")[..., 0]
        psi = wrap_heading(yaw if self.nav_frame == "NED" else np.pi / 2.0 - yaw)
        return np.rad2deg(psi) if degrees else psi

    @property
    def health(self) -> EstimatorHealth:
        """Final estimator condition (see :class:`EstimatorHealth`)."""
        return self.filter.health

    def to_dataframe(self):
        """The estimate as a pandas DataFrame (requires ``qnav[interop]``).

        Columns: ``t, q_w, q_x, q_y, q_z, yaw, pitch, roll`` [rad] and,
        when available, ``att_std_x/y/z`` [rad].
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_dataframe requires pandas: pip install 'qnav[interop]'"
            ) from exc
        e = self.euler("ZYX")
        data = {
            "t": self.t,
            "q_w": self.q[:, 0], "q_x": self.q[:, 1],
            "q_y": self.q[:, 2], "q_z": self.q[:, 3],
            "yaw": e[:, 0], "pitch": e[:, 1], "roll": e[:, 2],
        }
        if self.attitude_std is not None:
            for i, ax in enumerate("xyz"):
                data[f"att_std_{ax}"] = self.attitude_std[:, i]
        return pd.DataFrame(data)


def _as_samples(name: str, x, n: Optional[int]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {a.shape}")
    if n is not None and a.shape[0] != n:
        raise ValueError(f"{name} has {a.shape[0]} samples, expected {n}")
    return a


def _initial_attitude(
    nav_frame: str,
    f0: Optional[np.ndarray],
    m0: Optional[np.ndarray],
    m_ref: Optional[np.ndarray],
) -> np.ndarray:
    """Closed-form init from one accel(+mag) sample; identity if no accel."""
    if f0 is None:
        return quat.identity()
    if nav_frame == "NED":
        return fqa(f0, m0, m_ref)
    # ENU/FLU matched pair: solve in NED/FRD, then convert the attitude.
    f = flu_to_frd(f0)
    m = None if m0 is None else flu_to_frd(m0)
    mr = None if m_ref is None else enu_to_ned(m_ref)
    return attitude_ned_frd_to_enu_flu(fqa(f, m, mr))


def estimate_attitude(
    gyro,
    accel=None,
    mag=None,
    *,
    dt: Optional[float] = None,
    t=None,
    method: str = "eskf",
    nav_frame: str = "NED",
    q0=None,
    mag_ref=None,
    gyro_noise_density: float = 0.005,
    gyro_bias_walk: float = 1e-5,
    accel_sigma: float = 0.05,
    mag_sigma: float = 0.1,
    filter_kwargs: Optional[Dict[str, object]] = None,
) -> AttitudeEstimate:
    """Estimate orientation over time from logged sensor arrays, in one call.

    Parameters
    ----------
    gyro:
        ``(N, 3)`` body angular rate [rad/s]. Must be finite (a NaN gyro
        sample cannot be propagated — clean or crop the log first).
    accel:
        ``(N, 3)`` specific force [m/s²], or None for gyro-only dead
        reckoning. NaN rows are treated as dropouts and skipped.
    mag:
        ``(N, 3)`` magnetic field (any consistent unit), or None. Without a
        magnetometer, heading is unobservable and will drift with the gyro.
        NaN rows are skipped.
    dt / t:
        Exactly one of: uniform sample period [s], or ``(N,)`` strictly
        increasing per-sample timestamps [s].
    method:
        One of ``"eskf"`` (default; recommended — NEES-consistent
        covariance and gyro-bias estimation), ``"invariant"``, ``"ukf"``,
        ``"ekf"``, ``"mahony"``, ``"madgwick"``, ``"complementary"``,
        ``"aqua"``, ``"fourati"``, ``"roleq"``, ``"fkf"``. See the
        selection guide in :mod:`qnav.filters`.
    nav_frame:
        ``"NED"`` (FRD body axes) or ``"ENU"`` (FLU body axes). Some
        methods are NED-only and will raise ``ConventionError``.
    q0:
        Initial ``q_nav_body``. Default: solved in closed form (FQA) from
        the first accel(+mag) row whose used sensors are all finite;
        identity when ``accel`` is None.
    mag_ref:
        Navigation-frame magnetic field direction used as the magnetometer
        reference. Default: the first usable magnetometer row rotated into
        the navigation frame with the initial attitude — self-consistent
        with the FQA initialization, and yields *magnetic* (not true)
        heading. Pass a WMM-derived field (:mod:`qnav.geomag`) for true
        heading.
    gyro_noise_density, gyro_bias_walk:
        Process-noise densities [rad/s/√Hz, rad/s²/√Hz] for the
        covariance-bearing methods (ignored by gain-based filters).
    accel_sigma, mag_sigma:
        Per-axis noise std of the *unit-direction* measurements for the
        Kalman-type methods (``fkf``: quaternion-measurement noise σ).
    filter_kwargs:
        Extra keyword arguments forwarded to the filter constructor
        (overrides the defaults above), e.g. ``{"kp": 2.0}`` for Mahony or
        ``{"gate": GatePolicy(...)}`` for the ESKF.

    Returns
    -------
    AttitudeEstimate
        Quaternions per sample plus converters, uncertainty, bias, health,
        and the live filter object.

    Examples
    --------
    >>> import numpy as np
    >>> from qnav import estimate_attitude
    >>> n = 100
    >>> gyro = np.zeros((n, 3))
    >>> accel = np.tile([0.0, 0.0, -9.81], (n, 1))     # static, NED/FRD
    >>> est = estimate_attitude(gyro, accel, dt=0.01)
    >>> est.q.shape, est.attitude_std.shape
    ((100, 4), (100, 3))
    """
    if method not in _METHOD_CLASSES:
        raise ValueError(
            f"unknown method {method!r}; choose from {sorted(_METHOD_CLASSES)}"
        )
    if nav_frame not in ("NED", "ENU"):
        raise ValueError(f"nav_frame must be 'NED' or 'ENU', got {nav_frame!r}")

    w = _as_samples("gyro", gyro, None)
    n = w.shape[0]
    if n == 0:
        raise ValueError("gyro must contain at least one sample")
    if not np.all(np.isfinite(w)):
        k = int(np.argmax(~np.all(np.isfinite(w), axis=1)))
        raise ValueError(
            f"gyro contains a non-finite sample at row {k}; propagation "
            "cannot bridge gyro gaps — clean or crop the log first"
        )

    f = None if accel is None else _as_samples("accel", accel, n)
    m = None if mag is None else _as_samples("mag", mag, n)
    f_ok = np.zeros(n, bool) if f is None else np.all(np.isfinite(f), axis=1)
    m_ok = np.zeros(n, bool) if m is None else np.all(np.isfinite(m), axis=1)
    # non-Optional views for indexing; f_ok/m_ok are all-False when the
    # sensor is absent, so the empty placeholders are never actually read
    f_data = f if f is not None else np.empty((0, 3))
    m_data = m if m is not None else np.empty((0, 3))

    if (dt is None) == (t is None):
        raise ValueError("provide exactly one of dt (uniform) or t (timestamps)")
    if t is not None:
        times = np.asarray(t, dtype=float)
        if times.shape != (n,):
            raise ValueError(f"t must have shape ({n},), got {times.shape}")
        dts = np.diff(times)
        if not np.all(np.isfinite(dts)) or np.any(dts <= 0):
            raise ValueError("t must be finite and strictly increasing")
    else:
        assert dt is not None  # narrowed by the exclusive check above
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"dt must be a positive finite scalar, got {dt}")
        times = np.arange(n) * dt
        dts = np.full(max(n - 1, 0), dt)

    if method in _NEEDS_ACC_AND_MAG and (f is None or m is None):
        raise ValueError(f"method {method!r} requires both accel and mag")

    # -- initialization ------------------------------------------------------
    m_nav = None
    if mag_ref is not None:
        m_nav = np.asarray(mag_ref, dtype=float)
        nm = np.linalg.norm(m_nav)
        if m_nav.shape != (3,) or nm < 1e-12:
            raise ValueError("mag_ref must be a non-zero 3-vector")
        m_nav = m_nav / nm

    if q0 is not None:
        q_init = quat.normalize(np.asarray(q0, dtype=float))
        if q_init.shape != (4,):
            raise ValueError(f"q0 must have shape (4,), got {q_init.shape}")
    else:
        # first row where every provided-and-used sensor is finite
        usable = f_ok & m_ok if m is not None else f_ok
        if usable.any():
            k0 = int(np.argmax(usable))
            q_init = _initial_attitude(
                nav_frame, f_data[k0], m_data[k0] if m is not None else None, m_nav
            )
        else:
            q_init = quat.identity()

    if m_nav is None and m is not None and m_ok.any():
        km = int(np.argmax(m_ok))
        m_nav = quat.rotate_vector(q_init, m_data[km])
        m_nav = m_nav / np.linalg.norm(m_nav)

    # -- construct the filter --------------------------------------------------
    common = {"q0": q_init, "nav_frame": nav_frame}
    defaults: Dict[str, object]
    if method in ("eskf", "invariant"):
        defaults = {"gyro_noise_density": gyro_noise_density,
                    "gyro_bias_walk": gyro_bias_walk, **common}
    elif method in ("ukf", "ekf"):
        defaults = {"gyro_noise_density": gyro_noise_density, **common}
    elif method in ("fourati", "roleq"):
        defaults = {"m_ref": m_nav, **common}
    elif method == "fkf":
        defaults = {"gyro_noise": gyro_noise_density, "accel_noise": accel_sigma,
                    "mag_noise": mag_sigma, **common}
    else:  # mahony, madgwick, complementary, aqua: gain-based class defaults
        defaults = dict(common)
    if filter_kwargs:
        defaults.update(filter_kwargs)
    flt = _METHOD_CLASSES[method](**defaults)

    up = np.array([0.0, 0.0, -1.0 if nav_frame == "NED" else 1.0])

    # -- run -------------------------------------------------------------------
    q_out = np.empty((n, 4))
    want_std = method in _TANGENT_COVARIANCE
    std_out = np.empty((n, 3)) if want_std else np.empty((0, 3))
    applied = 0
    skipped = 0
    if f is not None:
        skipped += int(np.count_nonzero(~f_ok[1:]))
    if m is not None:
        skipped += int(np.count_nonzero(~m_ok[1:]))

    def _read_std() -> np.ndarray:
        if method in ("eskf", "invariant"):
            return flt.attitude_std
        return np.sqrt(np.diag(flt.P))          # ukf: 3×3 tangent covariance

    def _correct(k: int, step_dt: float) -> int:
        """Apply sample k's vector corrections; returns #updates fused."""
        c = 0
        if method in ("eskf", "invariant"):
            if f_ok[k]:
                flt.update_gravity(f_data[k], sigma=accel_sigma, timestamp=times[k])
                c += 1
            if m_ok[k] and m_nav is not None:
                flt.update_magnetometer(m_nav, m_data[k], sigma=mag_sigma,
                                        timestamp=times[k])
                c += 1
        elif method in ("ukf", "ekf"):
            if f_ok[k]:
                flt.update_direction(up, f_data[k], sigma=accel_sigma)
                c += 1
            if m_ok[k] and m_nav is not None:
                flt.update_direction(m_nav, m_data[k], sigma=mag_sigma)
                c += 1
        elif method in ("mahony", "madgwick"):
            vn, vb = [], []
            if f_ok[k]:
                vn.append(up)
                vb.append(f_data[k])
            if m_ok[k] and m_nav is not None:
                vn.append(m_nav)
                vb.append(m_data[k])
            if vn:
                flt.step(w[k], step_dt, np.stack(vn), np.stack(vb))
                c += len(vn)
            else:
                flt.predict(w[k], step_dt)
        elif method == "complementary":
            flt.predict(w[k], step_dt)
            if f_ok[k]:
                flt.update(f_data[k], m_data[k] if m_ok[k] else None)
                c += 1 + int(m_ok[k])
        elif method == "aqua":
            fb = f_data[k] if f_ok[k] else None
            mb = m_data[k] if m_ok[k] else None
            flt.step(w[k], step_dt, f_body=fb, m_body=mb)
            c += int(fb is not None) + int(mb is not None)
        else:  # fourati, roleq, fkf — need the full accel+mag pair
            if f_ok[k] and m_ok[k]:
                flt.step(w[k], step_dt, f_data[k], m_data[k])
                c += 2
            else:
                flt.predict(w[k], step_dt)
        return c

    q_out[0] = flt.q
    if want_std:
        std_out[0] = _read_std()
    # step-style filters propagate inside _correct (their step() is
    # predict+update fused); the Kalman-type filters predict here.
    step_methods = ("mahony", "madgwick", "complementary", "aqua",
                    "fourati", "roleq", "fkf")
    for k in range(1, n):
        if method not in step_methods:
            flt.predict(w[k], dts[k - 1])
        applied += _correct(k, dts[k - 1])
        q_out[k] = flt.q
        if want_std:
            std_out[k] = _read_std()

    bias = np.asarray(flt.bias, dtype=float).copy() if hasattr(flt, "bias") else None
    return AttitudeEstimate(
        t=times,
        q=q_out,
        method=method,
        nav_frame=nav_frame,
        filter=flt,
        attitude_std=std_out if want_std else None,
        gyro_bias=bias,
        n_updates_applied=applied,
        n_updates_skipped=skipped,
    )
