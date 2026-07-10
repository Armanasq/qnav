"""Reproducible micro-benchmark runner (stdlib-only timing).

Used by ``benchmarks/run_benchmarks.py``; deterministic inputs (fixed seed),
median-of-repeats timing, and machine-readable results for regression
tracking. See ``docs/design/performance.md`` for the policy on when a C++
kernel is justified.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from typing import Callable

__all__ = ["BenchResult", "environment", "run_benchmark", "save_results"]


@dataclass
class BenchResult:
    name: str
    n_items: int
    median_s: float
    best_s: float
    p95_s: float
    per_item_ns: float
    repeats: int
    warmup: int


def run_benchmark(
    name: str, fn: Callable[[], object], n_items: int, repeats: int = 7,
    warmup: int = 2,
) -> BenchResult:
    """Median-of-``repeats`` wall time of ``fn()`` processing ``n_items``."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    return BenchResult(
        name=name, n_items=n_items, median_s=med, best_s=times[0], p95_s=p95,
        per_item_ns=1e9 * med / max(n_items, 1), repeats=repeats, warmup=warmup,
    )


def environment() -> dict:
    """Full environment record required for comparable benchmark numbers."""
    import os
    import subprocess

    import numpy as np

    blas = "unknown"
    try:  # numpy >= 1.26 exposes the build config
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")
    except TypeError:
        pass
    sha = "unknown"
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "architecture": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": blas,
        "threads": os.cpu_count(),
        "commit": sha,
    }


def save_results(results: list, path: str) -> None:
    """Write results + environment info as JSON for regression comparison."""
    payload = {"environment": environment(), "results": [asdict(r) for r in results]}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
