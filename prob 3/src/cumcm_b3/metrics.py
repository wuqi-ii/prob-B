"""四项统计指标与结果导出。

赛题要求的统计量（问题 3 正文）：

    被清除干扰源个数的比例 = 被清除个数 / 干扰源总数
    平均定位清除时间      = 定位清除总时间 / 被清除个数

其中"定位清除总时间"包含机器狗移动时间、频道切换时间、检测时间、
光学精确定位时间与清除时间——也就是从头到尾的虚拟时间总量。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .strategy import RunStats


@dataclass
class CaseResult:
    """单次运行的四项统计 + 诊断信息。"""

    case_id: str
    seed: int
    total_sources: int
    cleared_count: int
    cleared_ratio: float
    total_time_s: float
    mean_clear_time_s: float
    virtual_time_s: float
    stop_reason: str
    requests: int
    clear_attempts: int
    clear_success: int
    clear_miss: int
    scan_points_visited: int
    real_elapsed_s: float = 0.0
    per_target_time_s: Dict[int, float] = None  # type: ignore[assignment]

    def as_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["per_target_time_s"] = self.per_target_time_s or {}
        return d


def build_result(case_id: str, seed: int, total_sources: int,
                 stats: RunStats, virtual_time_s: Optional[float] = None) -> CaseResult:
    cleared = stats.cleared_count
    total_time = virtual_time_s if virtual_time_s is not None else stats.virtual_time_s
    ratio = cleared / total_sources if total_sources else 0.0
    mean_time = total_time / cleared if cleared else float("nan")
    return CaseResult(
        case_id=case_id,
        seed=seed,
        total_sources=total_sources,
        cleared_count=cleared,
        cleared_ratio=ratio,
        total_time_s=total_time,
        mean_clear_time_s=mean_time,
        virtual_time_s=total_time,
        stop_reason=stats.stop_reason,
        requests=stats.requests,
        clear_attempts=stats.clear_attempts,
        clear_success=stats.clear_success,
        clear_miss=stats.clear_miss,
        scan_points_visited=stats.scan_points_visited,
        real_elapsed_s=getattr(stats, "real_elapsed_s", 0.0),
        per_target_time_s=dict(stats.per_target_time_s),
    )


def aggregate(results: List[CaseResult]) -> Dict[str, Any]:
    """多次演练的汇总统计。"""
    if not results:
        return {}
    n = len(results)
    cleared = [r.cleared_count for r in results]
    ratios = [r.cleared_ratio for r in results]
    times = [r.total_time_s for r in results]
    means = [r.mean_clear_time_s for r in results if math.isfinite(r.mean_clear_time_s)]
    requests = [r.requests for r in results]
    return {
        "cases": n,
        "total_sources": sum(r.total_sources for r in results),
        "total_cleared": sum(cleared),
        "cleared_mean": sum(cleared) / n,
        "cleared_min": min(cleared),
        "cleared_max": max(cleared),
        "ratio_mean": sum(ratios) / n,
        "ratio_min": min(ratios),
        "perfect_cases": sum(1 for r in ratios if r >= 1.0 - 1e-12),
        "total_time_mean_s": sum(times) / n,
        "total_time_min_s": min(times),
        "total_time_max_s": max(times),
        "mean_clear_time_s": sum(means) / len(means) if means else float("nan"),
        "requests_mean": sum(requests) / n,
        "requests_max": max(requests),
    }


def json_safe(payload: Any) -> Any:
    """把非有限浮点（inf / nan）替换为 ``None``，保证产物是**严格合法**的 JSON。

    Python 的 ``json.dumps`` 默认会写出 ``Infinity`` / ``NaN`` 字面量，
    这不是 RFC 8259 允许的写法，JS 的 ``JSON.parse`` 及其他语言的严格解析器会直接报错。
    本工程有多处参数取 ``inf``（如虚拟时间软预算），必须在这里统一收口。
    """
    if isinstance(payload, float):
        return payload if math.isfinite(payload) else None
    if isinstance(payload, dict):
        return {k: json_safe(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [json_safe(v) for v in payload]
    return payload


def dumps_strict(payload: Any, **kwargs: Any) -> str:
    """严格合法的 JSON 文本（不含 Infinity / NaN）。"""
    return json.dumps(json_safe(payload), allow_nan=False, **kwargs)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_strict(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, record: Any) -> None:
    """向 JSON Lines 追加一行，严格合法（供演练索引等使用）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(dumps_strict(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def format_table(results: List[CaseResult]) -> str:
    """生成论文可直接引用的 Markdown 表格。"""
    lines = [
        "| 案例 | 干扰源总数 | 清除个数 | 清除比例 | 总时间(s) | 平均定位清除时间(s) | 请求数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.case_id} | {r.total_sources} | {r.cleared_count} | "
            f"{r.cleared_ratio:.3f} | {r.total_time_s:.1f} | {r.mean_clear_time_s:.1f} | {r.requests} |"
        )
    agg = aggregate(results)
    if agg:
        lines.append(
            f"| **平均** | {agg['total_sources']/agg['cases']:.1f} | {agg['cleared_mean']:.2f} | "
            f"{agg['ratio_mean']:.3f} | {agg['total_time_mean_s']:.1f} | "
            f"{agg['mean_clear_time_s']:.1f} | {agg['requests_mean']:.1f} |"
        )
    return "\n".join(lines)
