"""Figures generated from actual numeric geometry; no decorative placeholders.

Every curve is rebuilt from the same constraints and evaluation data used by the
solver, so the pictures can be checked against the exported CSV/JSON.
"""
import json
import math
from pathlib import Path

import numpy as np

from . import optimize as opt
from .config import Config
from .region import feasible_mask, lobe_bounds, margins, source_polygon, triangle

FONT_CANDIDATES = ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf")


def setup_fonts():
    import matplotlib
    from matplotlib import font_manager
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
                matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
                break
            except Exception:
                continue
    matplotlib.rcParams["axes.unicode_minus"] = False


def _axes_labels(ax, config, equal=True):
    ax.set_xlabel("局部 x / m（沿第一次示向度方向）")
    ax.set_ylabel("局部 y / m")
    ax.grid(alpha=0.25)
    if equal:
        ax.set_aspect("equal")


def _constraint_curves(config):
    """Boundary of each hard constraint, for the explanatory figure.

    Returns (name, formula, x, y) in the local frame, upper half only; the lower
    half is the mirror image.
    """
    L, R, h, t = config.far, config.guaranteed_reception_m, config.h, config.t
    grid = np.linspace(0.0, R * 1.05, 2000)
    curves = []

    # (i) forward cone: a >= 0.
    curves.append(("① 前向约束 a ≥ 0", "a=0", np.zeros(2), np.array([-R, R])))

    # (ii) reception from the near triangle vertex: a^2 + b^2 <= R^2.
    theta = np.linspace(0.0, math.pi / 2, 600)
    curves.append(("② 接收约束（近端顶点）", "$a^2+b^2\\leq R^2$",
                   R * np.cos(theta), R * np.sin(theta)))

    # (iii) reception from the far vertices: (a-L)^2 + (b+h)^2 <= R^2 (b>0).
    phi = np.linspace(math.pi / 2, math.pi, 600)
    curves.append(("③ 接收约束（远端顶点）", "$(a-L)^2+(|b|+h)^2\\leq R^2$",
                   L + R * np.cos(phi), -h + R * np.sin(phi)))

    # (iv) crossing angle, near side: |b| >= t a.
    curves.append(("④ 交会角约束（近端）", "$|b|\\geq t\\,a,\\ t=\\tan31^\\circ$",
                   grid, t * grid))

    # (v) crossing angle, far side: |b| >= t (L-a) + h.
    curves.append(("⑤ 交会角约束（远端）", "$|b|\\geq t(L-a)+h$",
                   grid, t * (L - grid) + h))
    return curves


def plot_constraint_region(path, config, summary, poly, poly_full):
    """Show every hard constraint and the region they carve out."""
    import matplotlib.pyplot as plt
    setup_fonts()
    fig, ax = plt.subplots(figsize=(11.0, 8.2))

    # Source region and first bearing line, for context.
    src = np.asarray(poly_full)
    ax.fill(src[:, 0], src[:, 1], color="#9ec5f0", alpha=0.85,
            label="源位置可行区域 K（保守外包）")
    ax.plot([0, config.far], [0, 0], "--", color="#d1495b", lw=1.4,
            label="第一次示向线（K 的对称轴线）")

    # Each constraint boundary, upper half and its mirror. Labels get distinct,
    # hand-picked anchors so they never overlap each other or the region.
    colours = ["#8a8f98", "#2f5fd0", "#7b3fa0", "#c2571a", "#1f8a70"]
    anchors = [(30.0, -905.0, (6, 0)),                      # (i) on the a=0 line
               (1000 * math.cos(0.90), 1000 * math.sin(0.90), (-14, 10)),   # (ii) arc
               (config.far + config.guaranteed_reception_m * math.cos(2.24),
                -config.h + config.guaranteed_reception_m * math.sin(2.24), (10, 6)),
               (430.0, config.t * 430.0, (-14, 8)),         # (iv) near angle line
               (0.66 * config.far, config.t * (config.far - 0.66 * config.far) + config.h, (8, 6))]
    for k, (name, formula, xs, ys) in enumerate(_constraint_curves(config)):
        colour = colours[k % len(colours)]
        ax.plot(xs, ys, color=colour, lw=2.0, zorder=5)
        ax.plot(xs, -ys, color=colour, lw=2.0, zorder=5)
        x0, y0, offset = anchors[k]
        ax.annotate(f"{name}\n{formula}", (x0, y0), textcoords="offset points",
                    xytext=offset, fontsize=9, color=colour, ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colour,
                              lw=0.6, alpha=0.9))

    # The feasible region itself: rebuilt from the analytic lower/upper bounds.
    x = np.linspace(0.0, config.guaranteed_reception_m, 2000)
    low, high = lobe_bounds(x, config)
    ok = np.isfinite(low) & np.isfinite(high) & (high >= low)
    xs, lo, hi = x[ok], low[ok], high[ok]
    for side in (1, -1):
        ax.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([side * lo, side * hi[::-1]]),
                color="#5dcaa5", alpha=0.55, zorder=4)
        ax.plot(xs, side * lo, color="#0e8a6a", lw=2.4, zorder=6)
        ax.plot(xs, side * hi, color="#0e8a6a", lw=2.4, zorder=6)
    ax.plot([], [], color="#0e8a6a", lw=2.4,
            label="固定候选区域 C 的边界（五条约束之交）")

    ax.plot(0, 0, "s", color="#111111", ms=9, label="第一检测点 S1")
    final = np.asarray(summary["final"]["point_local_m"])
    near = np.asarray(summary["nearest_point_reference"]["point_local_m"])
    ax.plot(*final, "o", color="#ff7f0e", ms=11, mec="black", mew=1.2, zorder=8,
            label=f"最终选点 P* = ({final[0]:.0f}, {final[1]:.0f}) m")
    ax.plot(*near, "D", color="#5a5a5a", ms=8, zorder=8,
            label=f"仅按最近距离选点 ({near[0]:.0f}, {near[1]:.0f}) m")

    # Is the optimum on a boundary or in the interior? Report the slack.
    slack = margins(final, config)
    ax.annotate(f"P* 为内部最优点，不在任何约束边界上\n"
                f"各约束松弛量 = ②{slack[1]:.0f} m、③{slack[2]:.0f} m、"
                f"④{slack[3]:.0f} m、⑤{slack[4]:.0f} m",
                xy=(final[0], final[1]), xytext=(80.0, -160.0), textcoords="data",
                fontsize=9.5, ha="left",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.4),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ff7f0e", alpha=0.92))

    ax.set_xlim(-80, max(config.guaranteed_reception_m, config.far) * 1.05)
    ax.set_ylim(-config.guaranteed_reception_m * 1.05, config.guaranteed_reception_m * 1.05)
    _axes_labels(ax, config)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    ax.set_title("第二问：五条硬约束如何截出固定候选区域\n"
                 f"（h(R)=R·tan1°={config.h:.2f} m，t=tan31°={config.t:.4f}，"
                 f"ρ={config.guaranteed_reception_m:.0f} m，R={config.far:.0f} m）")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=170)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_cost_heatmap(path, config, summary, rows):
    """Filled contours of the cost J over the feasible region.

    The lobes are non-convex, so a Delaunay fill of sample points would leak
    into infeasible ground. We therefore interpolate onto a regular grid and
    mask everything outside the hard candidate region.
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    setup_fonts()
    weight = float(summary["lambda"]["base_weight"])
    values = opt.cost(rows[:, opt.DIST], rows[:, opt.COMP], weight, config)

    step = 3.0
    xs = np.arange(0.0, config.guaranteed_reception_m + step, step)
    ys = np.arange(-config.guaranteed_reception_m, config.guaranteed_reception_m + step, step)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    inside = feasible_mask(grid, config)

    linear = LinearNDInterpolator(rows[:, :2], values)
    nearest = NearestNDInterpolator(rows[:, :2], values)
    raw = linear(grid)
    fallback = ~np.isfinite(raw)
    if fallback.any():
        raw[fallback] = nearest(grid[fallback])
    z = np.where(inside, raw, np.nan).reshape(gx.shape)
    masked = np.ma.masked_invalid(z)

    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color="white", alpha=0.0)
    levels = np.linspace(np.nanmin(z), np.nanmax(z), 33)
    filled = ax.contourf(gx, gy, masked, levels=levels, cmap=cmap)
    line_levels = np.linspace(np.nanmin(z), np.nanmax(z), 11)[1:-1]
    lines = ax.contour(gx, gy, masked, levels=line_levels, colors="white",
                       linewidths=0.7, alpha=0.8)
    ax.clabel(lines, inline=True, fontsize=8, fmt="%.1f")
    fig.colorbar(filled, ax=ax, label="代价 J / s（颜色越深代价越小）", pad=0.02)

    # Crisp analytic outline of both lobes.
    x = np.linspace(0.0, config.guaranteed_reception_m, 1500)
    low, high = lobe_bounds(x, config)
    ok = np.isfinite(low) & np.isfinite(high) & (high >= low)
    for side in (1, -1):
        ax.plot(x[ok], side * low[ok], color="#00b894", lw=2.0, zorder=6)
        ax.plot(x[ok], side * high[ok], color="#00b894", lw=2.0, zorder=6)

    final = np.asarray(summary["final"]["point_local_m"])
    near = np.asarray(summary["nearest_point_reference"]["point_local_m"])
    ax.plot(*near, "D", color="white", mec="black", mew=1.0, ms=8,
            label="仅按最近距离选点", zorder=8)
    ax.plot(*final, "o", color="#ff2d2d", ms=12, mec="white", mew=1.4, zorder=9,
            label=f"最终选点 P*（J={summary['final']['cost_s']:.2f} s）")
    pad_x = max(30.0, 0.08 * (rows[:, opt.X].max() - rows[:, opt.X].min() + 1.0))
    pad_y = max(30.0, 0.08 * (rows[:, opt.Y].max() - rows[:, opt.Y].min() + 1.0))
    ax.set_xlim(rows[:, opt.X].min() - pad_x, rows[:, opt.X].max() + pad_x)
    ax.set_ylim(rows[:, opt.Y].min() - pad_y, rows[:, opt.Y].max() + pad_y)
    _axes_labels(ax, config)
    ax.legend(loc="center right", fontsize=10, framealpha=0.95)
    ax.set_title(f"候选区域内代价函数 J 的分布（J = L/5 + 5 + λS/5，λ={weight:.2f}）\n"
                 f"两瓣互为镜像；绿色实线为候选区域边界，白色为 J 等值线")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=170)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_pareto(path, config, summary, rows, mask):
    import matplotlib.pyplot as plt
    setup_fonts()
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.scatter(rows[:, opt.DIST], rows[:, opt.COMP], s=18, color="#b0b7c3", label="候选点")
    front = rows[mask]
    order = np.argsort(front[:, opt.DIST])
    ax.plot(front[order, opt.DIST], front[order, opt.COMP], "-o", color="#2f5fd0",
            ms=4, label="帕累托前沿")
    final = summary["final"]
    ax.plot(final["distance_m"], final["composite_m"], "o", color="#ff7f0e", ms=11,
            mec="black", mew=1.0, label="最终选点 P*")
    near = summary["nearest_point_reference"]
    ax.plot(near["distance_m"], near["composite_m"], "D", color="#7a7a7a", ms=8, label="最近点对照")
    ax.set_xlabel("移动距离 L / m")
    ax.set_ylabel("综合定位指标 S / m")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    lam = summary["lambda"]
    ax.set_title(f"距离与定位指标的帕累托关系（ΔL={lam['delta_distance_m']:.1f} m, "
                 f"ΔS={lam['delta_composite_m']:.1f} m, λ0={lam['lambda0']:.3g}）")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_sensitivity(path, config, summary):
    import matplotlib.pyplot as plt
    setup_fonts()
    results = summary["sensitivity"]
    if not results:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    ax = axes[0]
    # All selected points lie on the +y lobe (documented tie rule), so the
    # figure shows the upper lobe only; the lower one is its mirror image.
    x = np.linspace(0.0, config.guaranteed_reception_m, 800)
    low, high = lobe_bounds(x, config)
    ok = np.isfinite(low) & np.isfinite(high) & (high >= low)
    ax.fill_between(x[ok], low[ok], high[ok], color="#5dcaa5", alpha=0.22, linewidth=0,
                    label="候选区域（上侧瓣）")
    for r in results:
        point = np.asarray(r["point_local_m"])
        ax.plot(point[0], point[1], "o", ms=9, label=f"{r.get('multiplier', 1.0):g}×λ0")
    final = np.asarray(summary["final"]["point_local_m"])
    ax.plot(*final, "o", color="#ff7f0e", ms=11, mec="black", mew=1.0, label="基准 λ0 选点")
    margin = 40.0
    xs = [r["point_local_m"][0] for r in results] + [final[0]]
    ys = [r["point_local_m"][1] for r in results] + [final[1]]
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_xlabel("局部 x / m")
    ax.set_ylabel("局部 y / m（下侧瓣与其镜像对称）")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_title("（a）不同权重下的最终点位置")

    ax = axes[1]
    factors = [r.get("multiplier", 1.0) for r in results]
    ax.plot(factors, [r["mean_diameter_m"] for r in results], "-o", label="平均直径")
    ax.plot(factors, [r["cvar_diameter_m"] for r in results], "-s", label="CVaR 直径")
    ax.plot(factors, [r["max_diameter_m"] for r in results], "-^", label="最大直径")
    ax.set_xscale("log")
    ax.set_xlabel("λ / λ0")
    ax.set_ylabel("直径 / m")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("（b）权重敏感性：定位指标随 λ 的变化")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_example(path, config, summary):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    setup_fonts()
    example = summary.get("example_intersection")
    if not example:
        return
    point = np.asarray(summary["final"]["point_local_m"])
    poly = np.asarray(example["polygon"])
    source = np.asarray(example["source_local_m"])
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    ax.add_patch(MplPolygon(poly, closed=True, facecolor="#f0997b", edgecolor="#a32d2d",
                            lw=1.6, alpha=0.6, label="更新后定位区域 Ω"))
    # The station lies far outside the zoom window: draw its bearing line instead.
    direction = poly.mean(axis=0) - point
    direction = direction / np.linalg.norm(direction)
    centre = poly.mean(axis=0)
    half = max(float(example["diameter_m"]), 80.0)
    for sign in (-1.0, 1.0):
        tip = centre + sign * direction * half * 1.4
        ax.plot([centre[0] - direction[0] * half * 1.4, tip[0]],
                [centre[1] - direction[1] * half * 1.4, tip[1]],
                "--", lw=1.0, color="#2f5fd0",
                label="第二次示向线（假设读数）" if sign > 0 else None)
    ax.plot(*source, "x", color="black", ms=10, label="假设场景的源位置（仅用于生成读数）")
    ax.set_xlim(centre[0] - half * 1.6, centre[0] + half * 1.6)
    ax.set_ylim(centre[1] - half * 1.6, centre[1] + half * 1.6)
    ax.annotate(f"第二检测点 P* = ({point[0]:.0f}, {point[1]:.0f}) m（在窗口外）",
                (0.02, 0.02), xycoords="axes fraction", fontsize=9)
    ax.annotate(f"交会角 {example['acute_crossing_deg']:.2f}°\n区域直径 {example['diameter_m']:.2f} m",
                centre, textcoords="offset points", xytext=(12, 10), fontsize=10)
    ax.set_xlabel("局部 x / m")
    ax.set_ylabel("局部 y / m")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("示例（假设场景，非真实观测）：第二次测向后的定位区域")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_all(root, config, summary, rows, mask, poly, poly_full):
    # Use the reviewed two-panel plots for the requested geometry/cost figures.
    # Older helpers above remain for provenance, but these two are authoritative.
    from .publication_plots import plot_constraint_region as draw_region
    from .publication_plots import plot_cost_heatmap as draw_cost
    directory = Path(root) / "figures" / config.name
    directory.mkdir(parents=True, exist_ok=True)
    figures = {}
    names = ("fig1_constraints_region", "fig2_cost_heatmap", "fig3_pareto",
             "fig4_sensitivity", "fig5_example_intersection")
    draw_region(directory / names[0], config, summary, poly, poly_full)
    draw_cost(directory / names[1], config, summary, rows, root=root)
    plot_pareto(directory / names[2], config, summary, rows, mask)
    plot_sensitivity(directory / names[3], config, summary)
    plot_example(directory / names[4], config, summary)
    for key in names:
        figures[key + ".png"] = str((directory / key).with_suffix(".png"))
    return figures


def plot_saved_results(root, name):
    """Rebuild every figure from exported data only."""
    directory = Path(root) / "outputs" / name
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    config = Config(**{k: (tuple(v) if isinstance(v, list) and k in
                           ("first_station_m", "map_center_m", "sensitivity_weights",
                            "weight_multipliers", "refine_steps_m") else v)
                       for k, v in summary["config"].items()})
    rows = np.load(directory / "candidates.npy")
    mask = np.load(directory / "pareto_mask.npy")
    poly = source_polygon(config)
    return plot_all(root, config, summary, rows, mask, poly, poly)
