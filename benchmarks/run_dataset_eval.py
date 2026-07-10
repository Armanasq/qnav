#!/usr/bin/env python3
"""Real-dataset attitude evaluation across the benchmark collection.

Run:  python benchmarks/run_dataset_eval.py [--limit N] [output.json]

Replays every available benchmark trial (see
``qnav.validation.imu_datasets``) through the reference ESKF — gravity
aiding always, magnetometer aiding when the trial has one (reference field
derived from the first seconds of ground truth) — and reports
heading-aligned RMSE, tilt RMSE, heading RMSE, NIS, rejection rate, and
real-time factor per trial, plus per-collection aggregates. Results include
the full environment record; numbers without it are not comparable.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from qnav.attitude import quaternion as quat
from qnav.filters import Eskf, GatePolicy
from qnav.validation.benchmark_runner import environment
from qnav.validation.imu_datasets import available_datasets, load_attitude_dataset
from qnav.validation.replay_eval import replay_attitude


def reference_field(ds, seconds: float = 2.0) -> np.ndarray:
    """Local magnetic field in the reference frame from early ground truth."""
    idx = np.flatnonzero(ds.valid)[: max(int(seconds / ds.dt), 10)]
    return np.mean([quat.rotate_vector(ds.q_ref[k], ds.mag[k]) for k in idx], axis=0)


def evaluate_trial(path: Path):
    ds = load_attitude_dataset(path)
    q0 = ds.q_ref[np.flatnonzero(ds.valid)[0]]

    def make(d):
        return Eskf(gyro_noise_density=0.005, gyro_bias_walk=1e-5, q0=q0,
                    nav_frame="ENU", gate=GatePolicy(confidence=0.999))

    if ds.mag is not None:
        m_ref = reference_field(ds)

        def update(f, d, k):
            f.update_direction(np.array([0.0, 0.0, 1.0]), d.accel[k],
                               sigma=0.05, sensor_id="accel")
            f.update_direction(m_ref, d.mag[k], sigma=0.1, sensor_id="mag")
    else:
        update = None

    return replay_attitude(ds, make, update)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit"):
            limit = int(a.split("=", 1)[1]) if "=" in a else None
    out = args[0] if args else "benchmarks/dataset_results.json"

    paths = [p for p in available_datasets() if p.suffix in (".hdf5", ".h5")]
    if not paths:
        print("No datasets found (set QNAV_DATA_DIR or populate qnav/data).")
        sys.exit(1)
    if limit:
        # deterministic spread: first `limit` per collection
        by_dir = defaultdict(list)
        for p in paths:
            by_dir[p.parent.name].append(p)
        paths = [p for group in by_dir.values() for p in group[:limit]]

    reports, failures = [], []
    for p in paths:
        try:
            r = evaluate_trial(p)
            reports.append(r)
            print(f"{p.parent.name:16s} {r.dataset[:38]:38s} "
                  f"rmse={r.rmse_deg:6.2f}° tilt={r.tilt_rmse_deg:5.2f}° "
                  f"head={r.heading_rmse_deg:6.2f}° rej={r.rejection_rate:.3f} "
                  f"rtf={r.realtime_factor:5.0f}x")
        except Exception as exc:  # noqa: BLE001 - survey run: record and continue
            failures.append({"path": str(p), "error": f"{type(exc).__name__}: {exc}"})
            print(f"{p.parent.name:16s} {p.stem[:38]:38s} FAILED: {exc}")

    print("\nPer-collection aggregate (median over trials):")
    by_coll = defaultdict(list)
    for p, r in zip([p for p in paths if str(p) not in {f['path'] for f in failures}], reports):
        by_coll[p.parent.name].append(r)
    for coll, rs in sorted(by_coll.items()):
        med = np.median([r.rmse_deg for r in rs])
        med_t = np.median([r.tilt_rmse_deg for r in rs])
        print(f"  {coll:16s} n={len(rs):3d}  rmse={med:6.2f}°  tilt={med_t:5.2f}°")

    payload = {
        "environment": environment(),
        "estimator": "Eskf (gravity + mag-when-available, gate 0.999)",
        "reports": [asdict(r) for r in reports],
        "failures": failures,
    }
    Path(out).write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {len(reports)} reports ({len(failures)} failures) to {out}")


if __name__ == "__main__":
    main()
