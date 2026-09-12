"""Plot the numerical coordinate functions produced by run_parametric.py."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "parametric_R"
    with (out / "optimal_points.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    r = np.array([float(x["R_m"]) for x in rows])
    a = np.array([float(x["a_star_m"]) for x in rows])
    b = np.array([float(x["b_star_m"]) for x in rows])
    d = np.array([float(x["distance_m"]) for x in rows])

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    axes[0].plot(r, a, "o-", lw=1.8, label=r"$a^*(R)$")
    axes[0].plot(r, b, "s-", lw=1.8, label=r"$b^*(R)$")
    axes[0].plot(r, d, "^-", lw=1.8, label=r"$\|P^*(R)-S_1\|$")
    axes[0].set_xlabel("最远可能距离 R / m")
    axes[0].set_ylabel("距离 / m")
    axes[0].grid(alpha=.25)
    axes[0].legend()

    axes[1].plot(r, a / r, "o-", lw=1.8, label=r"$a^*(R)/R$")
    axes[1].plot(r, b / r, "s-", lw=1.8, label=r"$b^*(R)/R$")
    axes[1].set_xlabel("最远可能距离 R / m")
    axes[1].set_ylabel("无量纲比例")
    axes[1].grid(alpha=.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "parametric_curve.png", dpi=180)
    fig.savefig(out / "parametric_curve.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
