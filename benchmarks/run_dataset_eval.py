#!/usr/bin/env python3
"""Real-dataset attitude evaluation across the benchmark collection.

Usage::

    python benchmarks/run_dataset_eval.py \
        --limit-per-collection 5 \
        --init ground_truth \
        --mag-reference oracle \
        --config universal \
        --max-failure-rate 0.05 \
        --output-dir benchmarks/results

Gravity aiding from the accelerometer is applied on **every** trial;
magnetometer aiding is added when the trial has one. Trials whose loader
convention verification fails are recorded as failures.

Experiment axes (reported separately — never mixed):

``--init``
    ``ground_truth``  oracle-initialized tracking evaluation (labeled as
                      such; measures steady-state tracking only),
    ``accel_mag``     deployable: FQA-style init from the first accel/mag
                      sample (tilt-only when no magnetometer),
    ``identity``      worst-case unknown initial attitude,
    ``perturbed_10 / _45 / _90 / _150``
                      ground truth rotated by a fixed angle (convergence
                      studies).

``--mag-reference``
    ``oracle``        field derived from ground-truth attitude over the
                      calibration segment (oracle information — labeled),
    ``calibration``   deployable: field derived from accel-tilt leveling
                      over the calibration segment (heading-free; the
                      aligned metric absorbs the yaw gauge),
    ``none``          magnetometer disabled everywhere.

The calibration segment (first ``--calib-seconds``) is always excluded from
the error metrics.

``--config``
    ``universal``     one fixed noise configuration for all trials (fair
                      cross-dataset ranking),
    ``per-dataset``   collection-specific values (documented approximations
                      from the sensor classes used by each benchmark).

Exit status: 0 only when the failure rate is within ``--max-failure-rate``
and every ``--require-collection`` has at least one successful trial.
Writes ``attitude-real-data.json`` (full report), ``attitude-real-data.md``
(summary), and ``environment.json`` into ``--output-dir``. No absolute
paths are recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.filters import Eskf, GatePolicy
from qnav.validation.benchmark_runner import environment
from qnav.validation.imu_datasets import (
    available_datasets,
    data_root,
    load_attitude_dataset,
    verify_conventions,
)
from qnav.validation.replay_eval import replay_attitude

UP = np.array([0.0, 0.0, 1.0])

#: universal configuration: consumer/industrial MEMS class, one setting for
#: every trial so cross-dataset rankings are comparable.
UNIVERSAL = {"gyro_nd": 0.005, "gyro_bw": 1e-5, "acc_sigma": 0.05, "mag_sigma": 0.1}

#: per-collection approximations from the sensor classes used by each
#: benchmark (ADIS16448 for EuRoC, BMI160-class for TUM-VI/OxIOD phones,
#: Xsens/APDM/Shimmer wearables for Caruso, mocap-grade MEMS for RepoIMU and
#: Myon). These are datasheet-informed engineering values, not per-unit
#: calibrations; provenance is this comment.
PER_DATASET = {
    "EuRoC-MAV": {"gyro_nd": 0.0002, "gyro_bw": 4e-6, "acc_sigma": 0.03, "mag_sigma": 0.1},
    "TUM-VI": {"gyro_nd": 0.0008, "gyro_bw": 2e-5, "acc_sigma": 0.05, "mag_sigma": 0.1},
    "OxIOD": {"gyro_nd": 0.005, "gyro_bw": 5e-5, "acc_sigma": 0.08, "mag_sigma": 0.15},
    "RepoIMU": {"gyro_nd": 0.003, "gyro_bw": 1e-5, "acc_sigma": 0.05, "mag_sigma": 0.1},
    "Caruso-Sassari": {"gyro_nd": 0.003, "gyro_bw": 1e-5, "acc_sigma": 0.05, "mag_sigma": 0.1},
    "Myon": {"gyro_nd": 0.003, "gyro_bw": 1e-5, "acc_sigma": 0.05, "mag_sigma": 0.1},
}

INIT_MODES = ("ground_truth", "accel_mag", "identity",
              "perturbed_10", "perturbed_45", "perturbed_90", "perturbed_150")
MAG_MODES = ("oracle", "calibration", "none")


def _tilt_quaternion(f_body: np.ndarray) -> np.ndarray:
    """Zero-yaw attitude aligning the measured specific force with global +z.

    Shortest-arc rotation taking the body-frame up direction (f̂, since the
    accelerometer reads ``R(q)ᵀ [0,0,+g]`` in the z-up reference) onto e3.
    Heading is a free gauge here; the heading-aligned metric absorbs it.
    """
    fn = np.linalg.norm(f_body)
    if fn < 1e-6:
        return quat.identity()
    fh = f_body / fn
    e3 = np.array([0.0, 0.0, 1.0])
    axis = np.cross(fh, e3)
    s = np.linalg.norm(axis)
    c = float(fh @ e3)
    if s < 1e-12:
        return quat.identity() if c > 0 else quat.exp(np.array([np.pi, 0.0, 0.0]))
    return quat.exp(axis / s * np.arctan2(s, c))


def _initial_attitude(ds, mode: str) -> np.ndarray | None:
    first = int(np.flatnonzero(ds.valid)[0])
    if mode == "ground_truth":
        return ds.q_ref[first]
    if mode == "identity":
        return None
    if mode == "accel_mag":
        # deployable tilt initialization from the first accelerometer sample;
        # heading stays at the yaw gauge (absorbed by the aligned metric)
        return _tilt_quaternion(ds.accel[0])
    if mode.startswith("perturbed_"):
        deg = float(mode.split("_")[1])
        rng = np.random.default_rng(abs(hash(ds.name)) % 2**32)
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        return quat.normalize(quat.mul(ds.q_ref[first], quat.exp(np.deg2rad(deg) * axis)))
    raise ValueError(f"unknown init mode {mode!r}")


def _mag_reference(ds, mode: str, calib_seconds: float) -> np.ndarray | None:
    if ds.mag is None or mode == "none":
        return None
    n_cal = max(int(calib_seconds / ds.dt), 10)
    if mode == "oracle":
        idx = np.flatnonzero(ds.valid[:n_cal])
        if idx.size < 5:
            return None
        return np.mean([quat.rotate_vector(ds.q_ref[k], ds.mag[k]) for k in idx], axis=0)
    if mode == "calibration":
        # deployable: level each calibration sample with accel tilt only
        # (yaw gauge fixed at 0; valid while yaw stays roughly constant over
        # the segment, which holds for the benchmarks' static starts). No
        # ground truth involved.
        refs = []
        for k in range(0, n_cal, max(1, n_cal // 100)):
            q_tilt = _tilt_quaternion(ds.accel[k])
            refs.append(quat.rotate_vector(q_tilt, ds.mag[k]))
        return np.mean(refs, axis=0) if refs else None
    raise ValueError(f"unknown mag-reference mode {mode!r}")


def evaluate_trial(path: Path, args) -> tuple[dict, dict]:
    ds = load_attitude_dataset(path)
    conventions = verify_conventions(ds)
    if not conventions["ok"]:
        raise ValueError(f"convention verification failed: {conventions}")

    cfg = dict(UNIVERSAL)
    if args.config == "per-dataset":
        cfg.update(PER_DATASET.get(path.parent.name, {}))

    q0 = _initial_attitude(ds, args.init)
    m_ref = _mag_reference(ds, args.mag_reference, args.calib_seconds)

    def make(d):
        p0 = np.diag([0.3**2] * 3 + [0.01**2] * 3) if args.init == "ground_truth" \
            else np.diag([1.5**2] * 3 + [0.01**2] * 3)
        return Eskf(gyro_noise_density=cfg["gyro_nd"], gyro_bias_walk=cfg["gyro_bw"],
                    q0=q0, P0=p0, nav_frame="ENU",
                    gate=GatePolicy(confidence=0.999))

    def update(f, d, k):
        # gravity aiding on every trial, magnetometer only when referenced
        f.update_direction(UP, d.accel[k], sigma=cfg["acc_sigma"], sensor_id="accel")
        if m_ref is not None:
            f.update_direction(m_ref, d.mag[k], sigma=cfg["mag_sigma"], sensor_id="mag")

    report = replay_attitude(ds, make, update, settle_s=args.calib_seconds)
    out = asdict(report)
    out["mag_aided"] = m_ref is not None
    out["conventions"] = conventions
    return out, cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit-per-collection", type=int, default=None,
                    help="evaluate at most N trials per collection")
    ap.add_argument("--init", choices=INIT_MODES, default="ground_truth")
    ap.add_argument("--mag-reference", choices=MAG_MODES, default="calibration")
    ap.add_argument("--config", choices=("universal", "per-dataset"), default="universal")
    ap.add_argument("--calib-seconds", type=float, default=5.0)
    ap.add_argument("--max-failure-rate", type=float, default=0.05)
    ap.add_argument("--require-collection", action="append", default=[],
                    help="fail unless this collection has >= 1 successful trial")
    ap.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = ap.parse_args(argv)

    root = data_root()
    paths = [p for p in available_datasets() if p.suffix in (".hdf5", ".h5")]
    if not paths:
        print("No datasets found (set QNAV_DATA_DIR or populate qnav/data).")
        return 2
    def collection_of(p: Path) -> str:
        try:
            return p.relative_to(root).parts[0]
        except ValueError:
            return p.parent.name

    if args.limit_per_collection:
        by_dir: dict = defaultdict(list)
        for p in paths:
            by_dir[collection_of(p)].append(p)
        paths = [p for group in by_dir.values() for p in group[: args.limit_per_collection]]

    reports, failures = [], []
    cfg_used: dict = {}
    for p in paths:
        rel = str(p.relative_to(root))
        try:
            out, cfg_used = evaluate_trial(p, args)
            out["path"] = rel
            out["collection"] = collection_of(p)
            reports.append(out)
            print(f"{out['collection']:15s} {out['dataset'][:36]:36s} "
                  f"rmse={out['rmse_deg']:6.2f}° tilt={out['tilt_rmse_deg']:5.2f}° "
                  f"mag={'y' if out['mag_aided'] else 'n'} "
                  f"rej={out['rejection_rate']:.3f} rtf={out['realtime_factor']:4.0f}x")
        except Exception as exc:  # noqa: BLE001 - survey run: record, judge at exit
            failures.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{p.parent.name:15s} {p.stem[:36]:36s} FAILED: {exc}")

    total = len(reports) + len(failures)
    failure_rate = len(failures) / total if total else 1.0

    agg = {}
    by_coll: dict = defaultdict(list)
    for r in reports:
        by_coll[r["collection"]].append(r)
    for coll, rs in sorted(by_coll.items()):
        agg[coll] = {
            "trials": len(rs),
            "median_rmse_deg": float(np.median([r["rmse_deg"] for r in rs])),
            "median_tilt_rmse_deg": float(np.median([r["tilt_rmse_deg"] for r in rs])),
            "median_heading_rmse_deg": float(np.median([r["heading_rmse_deg"] for r in rs])),
            "median_realtime_factor": float(np.median([r["realtime_factor"] for r in rs])),
            "mag_aided_trials": int(sum(r["mag_aided"] for r in rs)),
        }

    payload = {
        "environment": environment(),
        "configuration": {
            "estimator": "Eskf (gate 0.999, gravity always, mag when referenced)",
            "init_mode": args.init,
            "init_is_oracle": args.init in ("ground_truth",) or args.init.startswith("perturbed"),
            "mag_reference_mode": args.mag_reference,
            "mag_reference_is_oracle": args.mag_reference == "oracle",
            "noise_config": args.config,
            "noise_values": cfg_used or UNIVERSAL,
            "calibration_segment_s": args.calib_seconds,
            "aggregation": "median over trials per collection",
            "metrics_exclude": "calibration segment and ground-truth gaps",
            "dataset_licenses": "see qnav/data provenance; public research "
                                "benchmarks (BROAD-format exports)",
        },
        "summary": {
            "trials_succeeded": len(reports),
            "trials_failed": len(failures),
            "failure_rate": failure_rate,
        },
        "aggregate": agg,
        "reports": reports,
        "failures": failures,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "attitude-real-data.json").write_text(json.dumps(payload, indent=2))
    (args.output_dir / "environment.json").write_text(json.dumps(environment(), indent=2))
    _write_markdown(args.output_dir / "attitude-real-data.md", payload)

    print(f"\n{len(reports)} succeeded, {len(failures)} failed "
          f"(rate {failure_rate:.3f}); reports in {args.output_dir}")
    for coll, a in agg.items():
        print(f"  {coll:15s} n={a['trials']:3d} rmse={a['median_rmse_deg']:6.2f}° "
              f"tilt={a['median_tilt_rmse_deg']:5.2f}°")

    if failure_rate > args.max_failure_rate:
        print(f"FAIL: failure rate {failure_rate:.3f} > {args.max_failure_rate}")
        return 1
    for coll in args.require_collection:
        if agg.get(coll, {}).get("trials", 0) < 1:
            print(f"FAIL: required collection {coll!r} has no successful trials")
            return 1
    return 0


def _write_markdown(path: Path, payload: dict) -> None:
    cfg = payload["configuration"]
    env = payload["environment"]
    s = payload["summary"]
    lines = [
        "# Real-data attitude evaluation",
        "",
        f"- commit: `{env['commit']}` · {env['cpu']} · NumPy {env['numpy']} · "
        f"Python {env['python']}",
        f"- estimator: {cfg['estimator']}",
        f"- initialization: **{cfg['init_mode']}**"
        + (" *(oracle information)*" if cfg["init_is_oracle"] else " (deployable)"),
        f"- magnetic reference: **{cfg['mag_reference_mode']}**"
        + (" *(oracle information)*" if cfg["mag_reference_is_oracle"] else ""),
        f"- noise config: {cfg['noise_config']} {cfg['noise_values']}",
        f"- trials: {s['trials_succeeded']} succeeded, {s['trials_failed']} failed "
        f"(rate {s['failure_rate']:.3f})",
        "",
        "| collection | trials | median RMSE [°] | median tilt [°] | "
        "median heading [°] | mag-aided | median RTF |",
        "|---|---|---|---|---|---|---|",
    ]
    for coll, a in sorted(payload["aggregate"].items()):
        lines.append(
            f"| {coll} | {a['trials']} | {a['median_rmse_deg']:.2f} | "
            f"{a['median_tilt_rmse_deg']:.2f} | {a['median_heading_rmse_deg']:.2f} | "
            f"{a['mag_aided_trials']}/{a['trials']} | "
            f"{a['median_realtime_factor']:.0f}x |")
    if payload["failures"]:
        lines += ["", "## Failures", ""]
        lines += [f"- `{f['path']}`: {f['error']}" for f in payload["failures"]]
    lines += ["", f"_Aggregation: {cfg['aggregation']}; metrics exclude "
              f"{cfg['metrics_exclude']}._", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
