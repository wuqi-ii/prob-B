"""把一次真实演练的行走轨迹画成 HTML（内嵌 SVG）。

路线按时间顺序分三段着色（浅 → 深），可直观看出"先遍历包围网、后回头清理"的节奏。

用法：
    python scripts/plot_route.py --tag drill12
    python scripts/plot_route.py --tag drill12 --segments 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.config import ARENA_RADIUS_M, StrategyConfig  # noqa: E402
from cumcm_b4.coverage import scan_points  # noqa: E402

# 路线分段配色（同一色系由浅到深 = 时间由早到晚）
SEG_COLORS = ("#CECBF6", "#AFA9EC", "#7F77DD", "#534AB7", "#3C3489")

VIEW = 640          # SVG 边长
PAD = 26            # 图区外留白
HALF_M = 1900.0     # 视野半径（米），略大于外环 1848


def load(tag: str):
    d = ROOT / "logs" / "runs" / tag
    rows = [json.loads(l) for l in (d / "http.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    reqs = [r for r in rows if r["kind"] == "request"]
    resp = {r["request_id"]: r for r in rows if r["kind"] == "response"}

    path = [(0.0, 0.0)]
    hits: list[tuple[float, float]] = []
    n_meas = n_clear = 0
    for r in reqs:
        ep = r.get("endpoint")
        if ep not in ("measure", "clear"):
            continue
        p = r["payload"]["position"]
        pos = (p["x"], p["y"])
        if math.dist(path[-1], pos) > 1e-6:
            path.append(pos)
        if ep == "measure":
            n_meas += 1
        else:
            n_clear += 1
            if resp[r["request_id"]]["response"].get("clear_result") == "success":
                hits.append(pos)

    # 该局自己的扫描点布局
    cfg = StrategyConfig()
    ec = d / "effective_config.json"
    if ec.exists():
        try:
            raw = json.loads(ec.read_text(encoding="utf-8"))
            known = {k: v for k, v in raw.items()
                     if k in StrategyConfig.__dataclass_fields__ and v is not None}
            cfg = StrategyConfig(**known)
        except Exception:  # noqa: BLE001
            pass
    total_t = max((r["response"].get("virtual_time_s", 0.0) for r in rows if r["kind"] == "response"),
                  default=0.0)
    return path, scan_points(cfg), hits, cfg, n_meas, n_clear, total_t


def build_svg(path, pts, hits, n_seg: int) -> str:
    cx = cy = VIEW / 2
    k = (VIEW / 2 - PAD) / HALF_M

    def px(p):
        return (cx + p[0] * k, cy - p[1] * k)

    inner = [(i, math.dist(path[i], path[i + 1])) for i in range(len(path) - 1)]
    total = sum(d for _, d in inner) or 1.0
    # 按累计位移切段
    bounds, acc = [], 0.0
    for s in range(1, n_seg):
        bounds.append(total * s / n_seg)
    segs: list[list[tuple[float, float]]] = [[] for _ in range(n_seg)]
    si = 0
    segs[0].append(path[0])
    for i, d in inner:
        while si < len(bounds) and acc + d > bounds[si]:
            si += 1
            segs[si].append(path[i + 1])
        acc += d
        segs[si].append(path[i + 1])

    parts = [
        f'<svg viewBox="0 0 {VIEW} {VIEW}" width="100%" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">',
        f'<title>演练行走轨迹</title>',
        f'<desc>机器人从原点出发遍历包围网扫描点并清除干扰源的轨迹，按时间由浅到深分为 {n_seg} 段</desc>',
    ]
    # 目标区域
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{ARENA_RADIUS_M * k:.1f}" fill="var(--bg-arena)" '
                 f'stroke="var(--fg-line)" stroke-width="1" stroke-dasharray="5 4"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{500 * k:.1f}" fill="none" '
                 f'stroke="var(--fg-line)" stroke-width="0.5" stroke-dasharray="3 5" opacity="0.6"/>')
    # 路线分段
    for s, seg in enumerate(segs):
        if len(seg) < 2:
            continue
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in (px(p) for p in seg))
        parts.append(f'<path d="{d}" fill="none" stroke="{SEG_COLORS[s % len(SEG_COLORS)]}" '
                     f'stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>')
    # 扫描点
    for p in pts:
        x, y = px(p)
        parts.append(f'<rect x="{x - 3:.1f}" y="{y - 3:.1f}" width="6" height="6" rx="1.5" '
                     f'fill="var(--fg-point)" stroke="var(--bg-card)" stroke-width="0.8"/>')
    # 干扰源（清除命中位置）
    for p in hits:
        x, y = px(p)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="none" stroke="#E24B4A" stroke-width="2"/>')
        parts.append(f'<line x1="{x - 3.4:.1f}" y1="{y:.1f}" x2="{x + 3.4:.1f}" y2="{y:.1f}" stroke="#E24B4A" stroke-width="1.6"/>')
    # 起点
    x, y = px((0.0, 0.0))
    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#639922"/>')
    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="none" stroke="#639922" stroke-width="1.2" opacity="0.7"/>')
    # 比例尺 500 m
    bx, by = PAD + 6, VIEW - PAD - 4
    parts.append(f'<line x1="{bx}" y1="{by}" x2="{bx + 500 * k:.1f}" y2="{by}" stroke="var(--fg-text)" stroke-width="1.5"/>')
    parts.append(f'<text x="{bx}" y="{by - 7}" fill="var(--fg-text)" font-size="12" font-family="sans-serif">500 m</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="drill12")
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path, pts, hits, cfg, n_meas, n_clear, total_t = load(args.tag)
    dist = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
    svg = build_svg(path, pts, hits, args.segments)
    n_pts = cfg.scan_inner_count + cfg.scan_outer_count + 1

    names = ("轨迹前段", "轨迹中段", "轨迹后段")
    seg_legend = "\n".join(
        f'  <span><i class="sw" style="background:{SEG_COLORS[s % len(SEG_COLORS)]}"></i>'
        f'{names[s] if s < len(names) else f"第 {s + 1} 段"}</span>'
        for s in range(args.segments)
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{args.tag} 行走轨迹</title>
<style>
  :root {{
    --bg-card: #ffffff; --fg-text: #2C2C2A; --fg-line: #B4B2A9;
    --fg-point: #5F5E5A; --bg-arena: #F1EFE8; --bg-page: #F7F7F5; --bd: rgba(0,0,0,.12);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg-card: #1e1e1c; --fg-text: #D3D1C7; --fg-line: #5F5E5A;
      --fg-point: #B4B2A9; --bg-arena: #26251f; --bg-page: #161614; --bd: rgba(255,255,255,.14);
    }}
  }}
  body {{ margin:0; padding:24px; background:var(--bg-page); color:var(--fg-text);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:760px; margin:0 auto; }}
  h1 {{ font-size:16px; font-weight:500; margin:0 0 4px; }}
  .sub {{ font-size:13px; opacity:.7; margin:0 0 16px; }}
  .card {{ background:var(--bg-card); border:0.5px solid var(--bd); border-radius:12px; padding:8px; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:12px; margin:16px 0; }}
  .stat {{ background:var(--bg-card); border:0.5px solid var(--bd); border-radius:8px; padding:10px 12px; }}
  .stat .k {{ font-size:12px; opacity:.65; }}
  .stat .v {{ font-size:20px; font-weight:500; margin-top:2px; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:16px; margin-top:14px; font-size:12px; opacity:.8; }}
  .legend span {{ display:flex; align-items:center; gap:6px; }}
  .sw {{ width:14px; height:3px; border-radius:2px; display:inline-block; }}
  .dot {{ width:8px; height:8px; border-radius:2px; display:inline-block; }}
  .ring {{ width:11px; height:11px; border-radius:50%; border:2px solid #E24B4A; display:inline-block; }}
</style></head><body><div class="wrap">
<h1>{args.tag} 行走轨迹</h1>
<p class="sub">{n_pts} 点包围网 · 内环 {cfg.scan_inner_count}×{cfg.scan_inner_radius_m:.0f} m · 外环 {cfg.scan_outer_count}×{cfg.scan_outer_radius_m:.0f} m · 验证延后 {cfg.task_bias_verify_m:.0f} m</p>
<div class="card">{svg}</div>
<div class="stats">
  <div class="stat"><div class="k">总位移</div><div class="v">{dist:,.0f} m</div></div>
  <div class="stat"><div class="k">虚拟时间</div><div class="v">{total_t/3600:.2f} h</div></div>
  <div class="stat"><div class="k">频谱检测</div><div class="v">{n_meas} 次</div></div>
  <div class="stat"><div class="k">清除调用</div><div class="v">{n_clear} 次</div></div>
  <div class="stat"><div class="k">干扰源</div><div class="v">{len(hits)} 个</div></div>
</div>
<div class="legend">
{seg_legend}
  <span><i class="dot" style="background:#888780"></i>扫描点</span>
  <span><i class="ring"></i>干扰源</span>
  <span><i class="dot" style="background:#639922;border-radius:50%"></i>起点</span>
  <span><i class="dot" style="background:transparent;border:1px dashed #888780"></i>目标区域 1800 m</span>
</div>
</div></body></html>"""

    out = Path(args.out) if args.out else ROOT / "outputs" / f"route_{args.tag}.html"
    out.write_text(html, encoding="utf-8")
    print(f"已生成 {out}")
    print(f"位移 {dist:.1f} m | 顶点 {len(path)} | 扫描点 {len(pts)} | 源 {len(hits)} | "
          f"检测 {n_meas} | 清除 {n_clear} | 虚拟 {total_t:.1f} s")


if __name__ == "__main__":
    main()
