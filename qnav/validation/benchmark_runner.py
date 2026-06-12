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

__all__ = ["BenchResult", "run_benchmark", "save_results"]


@dataclass
class BenchResult:
    name: str
    n_items: int
    median_s: float
    best_s: float
    per_item_ns: float
    repeats: int


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
    return BenchResult(
        name=name, n_items=n_items, median_s=med, best_s=times[0],
        per_item_ns=1e9 * med / max(n_items, 1), repeats=repeats,
    )


def save_results(results: list, path: str) -> None:
    """Write results + environment info as JSON for regression comparison."""
    payload = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
