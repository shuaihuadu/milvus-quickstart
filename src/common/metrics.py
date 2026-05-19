"""
延迟 / QPS 统计与 CSV 输出工具。
"""
from __future__ import annotations

import csv
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np


@dataclass
class LatencyStats:
    label: str
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, label: str, samples_ms: Iterable[float]) -> "LatencyStats":
        arr = np.asarray(list(samples_ms), dtype=np.float64)
        if arr.size == 0:
            return cls(label, 0, 0, 0, 0, 0, 0, 0)
        return cls(
            label=label,
            n=int(arr.size),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(arr.mean()),
            min_ms=float(arr.min()),
            max_ms=float(arr.max()),
        )

    def to_row(self) -> dict:
        return asdict(self)


@contextmanager
def timer():
    """用法：
        with timer() as t:
            do_something()
        print(t.ms)
    """

    class _T:
        ms: float = 0.0

    t = _T()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.ms = (time.perf_counter() - start) * 1000.0


def append_csv(path: str | Path, row: dict, fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    file_exists = p.exists() and p.stat().st_size > 0
    fields = fieldnames or list(row.keys())
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def write_json(path: str | Path, data) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
