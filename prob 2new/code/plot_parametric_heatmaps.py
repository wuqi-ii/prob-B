"""Put the cost surfaces for several R values in one comparable figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from bearing2 import optimize as opt
from bearing2.config import Config
from bearing2.region import feasible_mask


def load_run(path: Path):
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    config = Config(**summary["config"])
    rows = np.genfromtxt(path / "candidates.csv", delimiter=",", skip_header=1)
    weight = float(summary["lambda"]["base_weight"])
    values = opt.cost(rows[:, opt.DIST], rows[:, opt.COMP], weight, config)
    return config, summary, rows, values


def surface(config, rows, values, nx=250, ny=260):
    xpad = max(20.0, .04 * np.ptp(rows[:, opt.X]))
    ypad = max(20.0, .04 * np.ptp(rows[:, opt.Y]))
    xs = np.linspace(max(0.0, rows[:, opt.X].min() - xpad), rows[:, opt.X].max() + xpad, nx)
    ys = np.linspace(rows[:, opt.Y].min() - ypad, rows[:, opt.Y].max() + ypad, ny)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    inside = feasible_mask(points, config)
    linear = LinearNDInterpolator(rows[:, :2], values)
    z = linear(points)
    missing = inside & ~np.isfinite(z)
    if missing.any():
        z[missing] = NearestNDInterpolator(rows[:, :2], values)(points[missing])
    z[~inside] = np.nan
    return gx, gy, z.reshape(gx.shape)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "parametric_R"
    paths = sorted(root.glob("outputs/parametric_R*p00"))
    runs = [load_run(path) for path in paths]
    if not runs:
        raise FileNotFoundError("run code/run_parametric.py first")

    all_values = np.concatenate([run[3] for run in runs])
    lo, hi = float(all_values.min()), float(all_values.max())
    levels = np.linspace(lo, hi, 36)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), constrained_layout=True)
    heat_axes = []
    image = None
    for ax, (config, summary, rows, values) in zip(axes.flat, runs):
        gx, gy, z = surface(config, rows, values)
        image = ax.contourf(gx, gy, z, levels=levels, cmap="viridis", extend="both")
        local_levels = np.linspace(float(values.min()), float(values.max()), 6)[1:-1]
        lines = ax.contour(gx, gy, z, levels=local_levels, colors="white",
                           linewidths=.55, alpha=.75)
        ax.clabel(lines, fmt="%.0f", fontsize=6.5)
        point = np.asarray(summary["final"]["point_local_m"])
        ax.plot(point[0], point[1], "*", ms=12, color="#ff3b30", mec="white", mew=.8)
        ax.set_title(f"R = {config.far:.0f} m    P* = ({point[0]:.0f}, {point[1]:.0f})")
        ax.set_xlabel("相对坐标 a / m")
        ax.set_ylabel("相对坐标 b / m")
        ax.set_aspect("equal", adjustable="box")
        upper = rows[:, opt.Y] >= 0
        xlo, xhi = rows[upper, opt.X].min(), rows[upper, opt.X].max()
        ylo, yhi = rows[upper, opt.Y].min(), rows[upper, opt.Y].max()
        xpad, ypad = max(8.0, .06 * (xhi - xlo)), max(8.0, .08 * (yhi - ylo))
        ax.set_xlim(max(0.0, xlo - xpad), xhi + xpad)
        ax.set_ylim(ylo - ypad, yhi + ypad)
        ax.grid(alpha=.12)
        ax.text(.03, .04, f"Jmin={summary['final']['cost_s']:.1f} s\n"
                f"λ={summary['lambda']['base_weight']:.2f}", transform=ax.transAxes,
                fontsize=8, color="white", bbox=dict(fc="black", ec="none", alpha=.5, pad=2.5))
        heat_axes.append(ax)

    extra = axes.flat[len(runs)]
    extra.axis("off")
    extra.text(.05, .85, "统一色标：绝对综合代价 J / s", fontsize=12, weight="bold")
    extra.text(.05, .67, "仅显示 +b 候选瓣；-b 候选瓣与其镜像对称", fontsize=10)
    extra.text(.05, .57, "红色星号：该 R 下的最优相对坐标", fontsize=10)
    extra.text(.05, .44, "白色等值线：各子图自身的局部代价层次", fontsize=10)
    extra.text(.05, .24, "所有子图使用同一颜色范围，\n因此可以直接比较不同 R 的绝对代价。",
               fontsize=10, linespacing=1.5)
    assert image is not None
    cbar = fig.colorbar(image, ax=heat_axes, shrink=.92, pad=.015)
    cbar.set_label("综合代价 J / s")
    fig.savefig(output / "cost_heatmaps_by_R.png", dpi=190)
    fig.savefig(output / "cost_heatmaps_by_R.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
