"""量化「扇形剪枝」的收益。

思想：对某个频道 f，一旦在位置 A 测得示向度 θ（误差 ±BEARING_ERROR_DEG），
真源必落在以 A 为顶点、方向 θ±1°、半径 ≤ RECEIVER_MAX_M 的窄扇形内。
此后若在位置 B 要对 f 做测量，而扇形内**任意**可能源位置到 B 的距离都
> RECEIVER_MAX_M，则 B 处物理上必然 no_signal —— 该次测量可以安全跳过，
省下 MEASURE_COST_S + 可能的 SWITCH_COST_S。

本脚本在真机日志上按时间顺序重放，统计可跳过的测量次数与节省时间。
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
    BEARING_ERROR_DEG,
    DOG_SPEED_MPS,
    LASER_COST_S,
    MEASURE_COST_S,
    OPTICAL_COST_S,
    RECEIVER_MAX_M,
    SWITCH_COST_S,
)

SAMPLES = 64  # 扇形采样点数（含两条边与弧）


def fan_points(a: tuple[float, float], theta_deg: float, rmax: float) -> list[tuple[float, float]]:
    """返回扇形（顶点 a、方向 theta、半角误差、半径 rmax）的采样点集。"""
    pts = [a]
    for i in range(SAMPLES + 1):
        t = -BEARING_ERROR_DEG + 2 * BEARING_ERROR_DEG * i / SAMPLES
        rad = math.radians(theta_deg + t)
        pts.append((a[0] + rmax * math.cos(rad), a[1] + rmax * math.sin(rad)))
    for i in range(1, SAMPLES):  # 弧上采样
        rad = math.radians(theta_deg + 2 * BEARING_ERROR_DEG * i / SAMPLES)
        pts.append((a[0] + rmax * math.cos(rad), a[1] + rmax * math.sin(rad)))
        rad = math.radians(theta_deg - 2 * BEARING_ERROR_DEG * i / SAMPLES)
        pts.append((a[0] + rmax * math.cos(rad), a[1] + rmax * math.sin(rad)))
    return pts


def replay(path: Path, rmax: float, use_two_bearings: bool) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    reqs = [r for r in rows if r["kind"] == "request"]
    resp = {r["request_id"]: r["response"] for r in rows if r["kind"] == "response"}

    # 频道状态：已观测到的扇形采样点集合；已清除则记录清除点
    fan: dict[int, list[tuple[float, float]]] = {}
    cleared: set[int] = set()
    last_chan: int | None = None

    total = 0
    skipped = 0
    saved_measure = 0.0
    saved_switch = 0.0
    # 剪枝后仍需做的测量（用于验证正确性：跳过的一定是 no_signal）
    violated = 0

    for r in reqs:
        ep = r["endpoint"]
        if ep == "measure":
            ch = r["payload"]["channel"]
            p = r["payload"]["position"]
            b = (p["x"], p["y"])
            total += 1
            would_switch = (last_chan is not None and last_chan != ch)
            if ch in fan:
                dmin = min(math.dist(b, q) for q in fan[ch])
                if dmin > rmax:
                    skipped += 1
                    saved_measure += MEASURE_COST_S
                    if would_switch:
                        saved_switch += SWITCH_COST_S
                    # 正确性校验：被跳过的必须真的是 no_signal
                    if resp[r["request_id"]].get("measure_result") != "no_signal":
                        violated += 1
                    continue  # 跳过则 last_chan 不变
            last_chan = ch
            res = resp[r["request_id"]]
            if res.get("measure_result") == "direction":
                newpts = fan_points(b, res["svd_deg"], rmax)
                if use_two_bearings and ch in fan:
                    # 取交集：只保留同时满足两次观测约束的采样点
                    inter = [q for q in fan[ch] if math.dist(q, b) <= rmax + 1e-9]
                    ang_ok = []
                    for q in inter:
                        pred = math.degrees(math.atan2(q[1] - b[1], q[0] - b[0])) % 360
                        diff = abs((pred - res["svd_deg"] + 180) % 360 - 180)
                        if diff <= BEARING_ERROR_DEG + 1e-9:
                            ang_ok.append(q)
                    if ang_ok:
                        fan[ch] = ang_ok
                    else:
                        fan[ch] = newpts
                else:
                    fan[ch] = newpts
        elif ep == "clear":
            ch = r["payload"]["channel"]
            last_chan = ch
            if resp[r["request_id"]].get("clear_result") == "success":
                cleared.add(ch)
                fan.pop(ch, None)  # 已清除，后续不再测该频道

    return {
        "total": total,
        "skipped": skipped,
        "violated": violated,
        "saved": saved_measure + saved_switch,
        "saved_measure": saved_measure,
        "saved_switch": saved_switch,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--rmax", type=float, default=RECEIVER_MAX_M)
    ap.add_argument("--two", action="store_true", help="启用双示向度交集收缩")
    args = ap.parse_args()

    runs = ROOT / "logs" / "runs"
    tags = args.tags or sorted(p.name for p in runs.iterdir() if (p / "http.jsonl").exists())
    print(f"剪枝参数: rmax={args.rmax:.0f} m  双示向度交集={'开' if args.two else '关'}")
    print(f"{'局':<10}{'测量':>6}{'可省':>7}{'占比':>8}{'省测量s':>9}{'省切换s':>9}{'合计s':>8}{'误剪':>6}")
    tot_s = tot_n = 0
    sum_saved = 0.0
    for tag in tags:
        p = runs / tag / "http.jsonl"
        if not p.exists():
            continue
        try:
            r = replay(p, args.rmax, args.two)
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}: 失败 {exc}")
            continue
        tot_n += r["total"]
        tot_s += r["skipped"]
        sum_saved += r["saved"]
        print(
            f"{tag:<10}{r['total']:>6}{r['skipped']:>7}"
            f"{(r['skipped'] / r['total'] * 100) if r['total'] else 0:>7.1f}%"
            f"{r['saved_measure']:>9.0f}{r['saved_switch']:>9.0f}{r['saved']:>8.0f}"
            f"{r['violated']:>6}"
        )
    print(f"\n合计: {tot_n} 次测量中可省 {tot_s} 次 ({tot_s / tot_n * 100:.1f}%)，"
          f"省时 {sum_saved:.0f} s / 局均 {sum_saved / max(1, len(tags)):.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
