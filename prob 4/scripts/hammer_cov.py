"""猛打工程内 coverage 函数，带看门狗，在 math 被污染的瞬间抓现行。"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b4 import coverage  # noqa: E402
from cumcm_b4.config import StrategyConfig  # noqa: E402

print("python", sys.version.split()[0], flush=True)
cfg = StrategyConfig(
    scan_inner_count=9, scan_inner_radius_m=980.0,
    scan_outer_count=13, scan_outer_radius_m=1854.0,
)
pts = coverage.scan_points(cfg)
real_math = math

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
t0 = time.time()
for rep in range(reps):
    try:
        v = coverage.coverage_worst_required_radius(cfg, step_m=2.0)
    except Exception as exc:
        print(f"CRASH on rep {rep}: {type(exc).__name__}: {exc}", flush=True)
        print("  type(coverage.math) =", type(coverage.math), flush=True)
        print("  coverage.math is math?", coverage.math is real_math, flush=True)
        print("  type(math.pi) =", type(math.pi), flush=True)
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
    polluted = coverage.math is not real_math
    pi_bad = not isinstance(coverage.math.pi, float)
    print(f"rep {rep}: worst={v:.3f} polluted={polluted} pi_bad={pi_bad} ({time.time()-t0:.0f}s)",
          flush=True)
    if polluted or pi_bad:
        print("DETECTED POLLUTION", flush=True)
        break
print("done", flush=True)
