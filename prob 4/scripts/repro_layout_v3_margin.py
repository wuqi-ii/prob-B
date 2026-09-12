"""原始复现：候选布局列表 + step=1.0，崩溃时打印诊断。"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math  # noqa: E402

from cumcm_b4 import coverage  # noqa: E402
from cumcm_b4.config import RECEIVER_MIN_M, StrategyConfig  # noqa: E402

CANDS = [
    ("p22 8/990/13/1854", 8, 990.0, 13, 1854.0),
    ("p23 8/980/14/1848", 8, 980.0, 14, 1848.0),
    ("p23 9/980/13/1854", 9, 980.0, 13, 1854.0),
    ("p23 8/990/14/1848", 8, 990.0, 14, 1848.0),
    ("p22 8/990/13/1860", 8, 990.0, 13, 1860.0),
]

attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 3
print("python", sys.version.split()[0], flush=True)
for att in range(attempts):
    print(f"--- attempt {att} ---", flush=True)
    for name, n1, r1, n2, r2 in CANDS:
        cfg = StrategyConfig(
            scan_inner_count=n1, scan_inner_radius_m=r1,
            scan_outer_count=n2, scan_outer_radius_m=r2,
        )
        t = time.time()
        try:
            v = coverage.coverage_worst_required_radius(cfg, step_m=1.0)
        except Exception as exc:
            print(f"CRASH att={att} {name}: {type(exc).__name__}: {exc}", flush=True)
            print("  type(coverage.math) =", type(coverage.math), flush=True)
            print("  coverage.math is math?", coverage.math is math, flush=True)
            traceback.print_exc()
            raise SystemExit(1)
        print(f"{name:22s} step1={v:8.3f} margin={RECEIVER_MIN_M-v:7.2f} "
              f"{'OK' if v <= RECEIVER_MIN_M else 'FAIL'} ({time.time()-t:.0f}s)", flush=True)
print("no crash in", attempts, "attempts", flush=True)
