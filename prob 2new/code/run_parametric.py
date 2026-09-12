"""Solve question 2 for a grid of source-distance bounds R.

The physical receiver limits remain unchanged.  Only ``source_far_m`` varies,
so the output is a reproducible numerical representation of
P*(R)=(a*(R), +/-b*(R)).
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from bearing2.config import load_config
from bearing2.pipeline import run_pipeline


def parse_values(text: str) -> list[float]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if len(values) < 2:
        raise argparse.ArgumentTypeError("provide at least two comma-separated R values")
    return values


def label(value: float) -> str:
    return f"R{value:07.2f}".replace(".", "p")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="求解参数化第二检测点 P*(R)")
    parser.add_argument("--config", type=Path, default=root / "configs" / "default.json")
    parser.add_argument("--r-values", type=parse_values,
                        default=parse_values("500,750,1000,1250,1500"))
    parser.add_argument("--plots", action="store_true", help="为每个 R 生成完整图组")
    parser.add_argument("--sensitivity", action="store_true", help="为每个 R 运行权重敏感性")
    parser.add_argument("--keep-map-boundary", action="store_true",
                        help="保留配置中的场地裁剪；默认关闭，因为 R 已代表裁剪后的最远距离")
    args = parser.parse_args()

    path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    base = load_config(path)
    out = root / "outputs" / "parametric_R"
    rows: list[dict] = []

    for far in args.r_values:
        cfg = replace(base, name=f"parametric_{label(far)}", source_far_m=far,
                      use_map_boundary=base.use_map_boundary if args.keep_map_boundary else False)
        summary = run_pipeline(root, cfg, plots=args.plots, sensitivity=args.sensitivity)
        final = summary["final"]
        rows.append({
            "R_m": far,
            "h_R_m": cfg.h,
            "a_star_m": final["point_local_m"][0],
            "b_star_m": abs(final["point_local_m"][1]),
            "distance_m": final["distance_m"],
            "mean_diameter_m": final["mean_diameter_m"],
            "cvar_diameter_m": final["cvar_diameter_m"],
            "max_diameter_m": final["max_diameter_m"],
            "composite_m": final["composite_m"],
            "cost_s": final["cost_s"],
            "lambda": summary["lambda"]["base_weight"],
            "candidate_points": summary["candidate_region"]["evaluated_points"],
            "min_constraint_slack_m": summary["checks"]["min_slack_m"],
            "audit_upper_m": summary["checks"]["worst_case_audit"]["upper_m"],
            "run_output": str(root / "outputs" / cfg.name),
        })

    segments = []
    for left, right in zip(rows, rows[1:]):
        width = right["R_m"] - left["R_m"]
        segments.append({
            "R_lo_m": left["R_m"], "R_hi_m": right["R_m"],
            "a_lo_m": left["a_star_m"], "b_lo_m": left["b_star_m"],
            "da_dR": (right["a_star_m"] - left["a_star_m"]) / width,
            "db_dR": (right["b_star_m"] - left["b_star_m"]) / width,
            "formula": "a=a_lo+da_dR*(R-R_lo); b=b_lo+db_dR*(R-R_lo)",
        })

    fields = list(rows[0])
    write_csv(out / "optimal_points.csv", rows, fields)
    write_csv(out / "piecewise_linear.csv", segments, list(segments[0]))
    payload = {
        "model": "P*(R)=(a*(R), +/-b*(R))",
        "physical_max_reception_m": base.max_reception_m,
        "guaranteed_reception_m": base.guaranteed_reception_m,
        "map_boundary_used": bool(args.keep_map_boundary and base.use_map_boundary),
        "interpolation_note": "分段线性插值仅作快速初值；使用前必须重新执行候选区硬约束校验。",
        "points": rows,
        "segments": segments,
    }
    (out / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Parameter table: {out / 'optimal_points.csv'}")
    print(f"Piecewise function: {out / 'piecewise_linear.csv'}")


if __name__ == "__main__":
    main()
