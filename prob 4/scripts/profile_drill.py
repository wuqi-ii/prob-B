"""解析真实演练日志，按行为类别与策略阶段分解虚拟时间。

与 scripts/profile_time.py（离线、按代码路径打点）互补：本脚本完全依据
http.jsonl 里服务端返回的 virtual_time_s 做差分，得到真机实测占比。

行为类别：
  move       纯移动（相邻动作位置差 / 5 m/s）
  measure    频谱检测（每次 5 s）
  switch     频道切换（每次 1 s，仅换频道时）
  optical    光学精确定位（clear 必含 3 s）
  laser      激光清除（仅 clear 成功时 2 s）

阶段类别：
  scan       位置落在包围网扫描点上
  locate     位置不在扫描点上（逼近/验证/二分/环形）
  clear      /clear 动作本身

用法：
    python scripts/profile_drill.py                 # 全部演练
    python scripts/profile_drill.py --tags drill12
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.config import (  # noqa: E402
    DOG_SPEED_MPS,
    LASER_COST_S,
    MEASURE_COST_S,
    OPTICAL_COST_S,
    SWITCH_COST_S,
    StrategyConfig,
)
from cumcm_b4.coverage import scan_points  # noqa: E402

BEHAVIORS = ("move", "measure", "switch", "optical", "laser")
PHASES = ("scan", "locate", "clear")


def analyze(path: Path, cfg: StrategyConfig, scan_set=None):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    reqs = [r for r in rows if r["kind"] == "request"]
    resp = {r["request_id"]: r for r in rows if r["kind"] == "response"}

    if scan_set is None:
        scan_set = {(round(x, 3), round(y, 3)) for x, y in scan_points(cfg)}

    behavior = {b: 0.0 for b in BEHAVIORS}
    phase = {p: 0.0 for p in PHASES}
    # 交叉表：阶段 -> 行为
    cross = {p: {b: 0.0 for b in BEHAVIORS} for p in PHASES}
    counts = {"measure": 0, "clear": 0, "clear_hit": 0, "switch": 0}
    dist_total = 0.0

    cur = (0.0, 0.0)
    cur_ch = 1
    prev_t = 0.0
    total = 0.0

    for r in reqs:
        ep = r.get("endpoint")
        if ep not in ("measure", "clear"):
            rr = resp.get(r["request_id"])
            if rr is not None:
                prev_t = rr["response"].get("virtual_time_s", prev_t)
            continue
        rr = resp.get(r["request_id"])
        if rr is None:
            continue
        rbody = rr["response"]
        t = rbody.get("virtual_time_s", prev_t)
        dt = t - prev_t
        prev_t = t
        total = max(total, t)

        p = r["payload"]["position"]
        pos = (p["x"], p["y"])
        ch = r["payload"]["channel"]

        dist = math.dist(cur, pos)
        cur = pos
        dist_total += dist
        move_s = dist / DOG_SPEED_MPS

        hit = rbody.get("clear_result") == "success"
        if ep == "clear":
            counts["clear"] += 1
            if hit:
                counts["clear_hit"] += 1
            opt_s = OPTICAL_COST_S
            laser_s = LASER_COST_S if hit else 0.0
            sw_s = 0.0
            meas_s = 0.0
            ph = "clear"
        else:
            counts["measure"] += 1
            sw_s = SWITCH_COST_S if ch != cur_ch else 0.0
            if sw_s:
                counts["switch"] += 1
            meas_s = MEASURE_COST_S
            opt_s = 0.0
            laser_s = 0.0
            ph = "scan" if (round(pos[0], 3), round(pos[1], 3)) in scan_set else "locate"
        cur_ch = ch

        # 残差：dt 里除上述记账外的部分（服务端额外开销/浮点）
        resid = dt - (move_s + sw_s + meas_s + opt_s + laser_s)
        if resid > 1e-6:
            move_s += resid  # 归到移动，保守

        for b, v in (("move", move_s), ("measure", meas_s), ("switch", sw_s),
                     ("optical", opt_s), ("laser", laser_s)):
            behavior[b] += v
            phase[ph] += v
            cross[ph][b] += v

    return {
        "total": total,
        "behavior": behavior,
        "phase": phase,
        "cross": cross,
        "counts": counts,
        "dist": dist_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    args = ap.parse_args()

    runs_dir = ROOT / "logs" / "runs"
    idx_path = ROOT / "logs" / "drill_index.jsonl"
    tags = args.tags
    if tags is None:
        tags = [d.name for d in sorted(runs_dir.iterdir()) if (d / "http.jsonl").exists()]

    cfg = StrategyConfig()
    per = []
    for tag in tags:
        p = runs_dir / tag / "http.jsonl"
        if not p.exists():
            continue
        # 用该局自己的生效配置重建扫描点集合，否则历史局（不同布局）会被误判
        scan_set = None
        ec = runs_dir / tag / "effective_config.json"
        if ec.exists():
            try:
                raw = json.loads(ec.read_text(encoding="utf-8"))
                known = {k: v for k, v in raw.items()
                         if k in StrategyConfig.__dataclass_fields__ and v is not None}
                own = StrategyConfig(**known)
                n_own = own.scan_inner_count + own.scan_outer_count + 1
                scan_set = {(round(x, 3), round(y, 3)) for x, y in scan_points(own)}
            except Exception:  # noqa: BLE001
                scan_set = None
        try:
            res = analyze(p, cfg, scan_set)
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}: 解析失败 {exc}")
            continue
        res["n_points"] = len(scan_set) if scan_set else None
        if res["counts"]["measure"] == 0:
            continue
        per.append((tag, res))

    if not per:
        print("没有可解析的演练日志")
        return

    print(f"共 {len(per)} 局（当前配置为 {len(scan_points(cfg))} 点包围网）\n")

    def line(name, res):
        t = res["total"]
        b, ph = res["behavior"], res["phase"]
        npts = f"{res.get('n_points')}点 " if res.get("n_points") else ""
        print(f"--- {name} {npts}总虚拟时间 {t:8.1f} s | 位移 {res['dist']:8.1f} m | "
              f"measure {res['counts']['measure']} 次 | clear {res['counts']['clear']} "
              f"(命中 {res['counts']['clear_hit']}) | 换频道 {res['counts']['switch']} 次")
        print("    [行为]  " + "  ".join(
            f"{k}={v:7.1f}s({v/t*100:5.2f}%)" for k, v in b.items() if v > 0))
        print("    [阶段]  " + "  ".join(
            f"{k}={v:7.1f}s({v/t*100:5.2f}%)" for k, v in ph.items() if v > 0))

    for name, res in per:
        line(name, res)
        print()

    # 平均（按局平均，而非按秒加权）
    if len(per) > 1:
        n = len(per)
        avg_b = {k: sum(r["behavior"][k] for _, r in per) / n for k in BEHAVIORS}
        avg_p = {k: sum(r["phase"][k] for _, r in per) / n for k in PHASES}
        avg_t = sum(r["total"] for _, r in per) / n
        print(f"=== 按局平均（n={n}）总时间 {avg_t:.1f} s ===")
        print("    [行为]  " + "  ".join(
            f"{k}={v:7.1f}s({v/avg_t*100:5.2f}%)" for k, v in avg_b.items() if v > 0))
        print("    [阶段]  " + "  ".join(
            f"{k}={v:7.1f}s({v/avg_t*100:5.2f}%)" for k, v in avg_p.items() if v > 0))


if __name__ == "__main__":
    main()
