"""问题 3 **演练测试** 入口。——只连接已经由人工启动的演练会话。

!!! 重要 !!!
本脚本**不会**也不会有办法去触发正式测试：模拟器只对本机 HTTP 端口提供
/enter、/measure、/clear、/exit 四个动作接口，测试模块的选择必须在
模拟器界面上由人点击。因此使用前请先在界面选择 **「问题 3 演练测试」**。

标准流程
--------
1. 模拟器保持联网、登录（队伍号 202623001124）；
2. 界面点击「问题 3 演练测试」→ 等待数据准备 → 5 秒倒计时结束、接口就绪；
3. 运行本脚本：

       python scripts/run_drill.py
       python scripts/run_drill.py --tag drill01 --verbose

4. 结束后在模拟器中导出该次日志，文件名不要修改，放入支撑材料。
5. 本局干扰源总数会**自动**从模拟器本地真值文件读取（`--sim-data-dir`）；
   读不到时清除比例记为未知，绝不猜测分母。必要时可用 `--true-sources N` 手工指定。
6. 每局产物落在 `logs/runs/<标签>/`（http.jsonl / summary.json / effective_config.json），
   核心指标追加到 `logs/drill_index.jsonl`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b3.config import StrategyConfig  # noqa: E402
from cumcm_b3.metrics import append_jsonl, build_result, write_json  # noqa: E402
from cumcm_b3.experimental_strategy import make_strategy as DogStrategy  # noqa: E402

# 模拟器把每局演练的真值写在自己的数据目录里（正式测试不写这个文件）。
# 有了它就能自动算出诚实的清除比例，不必人工去界面读数——也才可能发现"漏检"。
# 模拟器目录被移动过一次（math\CUMCM2026B -> CUMCM2026B），写死单个路径会让真值
# 静默读不到、清除比例变成未知。这里按候选顺序挑一个真实存在的，并在都不存在时
# 打印明确提示，而不是默默放弃。
SIM_DATA_DIR_CANDIDATES = (
    r"C:\Users\ASUS\Desktop\CUMCM2026B\Jammers-simulator-win64"
    r"\Jammers-simulator\JammersSimulatorData\behavior-logs",
    r"C:\Users\ASUS\Desktop\math\CUMCM2026B\Jammers-simulator-win64"
    r"\Jammers-simulator\JammersSimulatorData\behavior-logs",
)


def default_sim_data_dir() -> str:
    for path in SIM_DATA_DIR_CANDIDATES:
        if os.path.isdir(path):
            return path
    print("警告：未找到模拟器 behavior-logs 目录，本局将无法自动读取干扰源真值。"
          f"可用 --sim-data-dir 指定。已尝试：{SIM_DATA_DIR_CANDIDATES}")
    return SIM_DATA_DIR_CANDIDATES[0]


DEFAULT_SIM_DATA_DIR = default_sim_data_dir()


def read_practice_truth(data_dir: str, not_before_epoch: float, timeout_s: float = 15.0):
    """读取本局演练的真值（干扰源总数）。

    返回 (jammer_count, file_name, payload) 或 (None, None, None)。
    模拟器在测试结束后才写这个文件，所以要轮询等一会儿。
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        return None, None, None
    deadline = time.time() + timeout_s
    while True:
        candidates = []
        for path in directory.glob("practice-p3-*.result.json"):
            try:
                if path.stat().st_mtime >= not_before_epoch:
                    candidates.append(path)
            except OSError:
                continue
        if candidates:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            try:
                payload = json.loads(newest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if payload and isinstance(payload.get("jammer_count"), int):
                return payload["jammer_count"], newest.name, payload
        if time.time() >= deadline:
            return None, None, None
        time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description="问题 3 演练测试（仅限演练）")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--robot-id", default="202623001124")
    parser.add_argument("--tag", default=datetime.now().strftime("drill_%m%d_%H%M%S"))
    parser.add_argument("--config", default="")
    parser.add_argument("--second-a", type=float, default=None,
                        help="覆盖配置中的第二检测点局部纵向坐标 a（米）")
    parser.add_argument("--second-b", type=float, default=None,
                        help="覆盖配置中的第二检测点局部横向距离 |b|（米）")
    parser.add_argument("--run-group", default="",
                        help="把本局归入 logs/runs/<分组>/ 下的子目录，用于隔离不同参数组的结果")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-real-s", type=float, default=900.0)
    parser.add_argument("--true-sources", type=int, default=0,
                        help="本局干扰源总数；0 表示自动从模拟器本地真值文件读取")
    parser.add_argument("--sim-data-dir", default=DEFAULT_SIM_DATA_DIR,
                        help="模拟器 behavior-logs 目录，用于自动读取本局干扰源真值")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认")
    parser.add_argument("--force", action="store_true",
                        help="即使就绪探测未通过也继续（默认不继续）")
    args = parser.parse_args()

    # ---- 就绪探测：只发一个 GET /（不是四个动作），确认接口真的在服务 ----
    # 注意：模拟器启动时就保留 TCP 端口，"端口可连"不等于"接口已开放"。
    from urllib.parse import urlsplit
    from cumcm_b3.client import probe_interface

    parts = urlsplit(args.base_url)
    host = parts.hostname or "127.0.0.1"
    port = parts.port or 2026
    tcp_ok, ready, detail = probe_interface(host, port)
    print(f"接口探测：{detail}")
    if not ready:
        if not args.force:
            print(
                "\n机器狗接口当前不可用，脚本不做任何动作。请确认：\n"
                "  1) 已在模拟器界面点击「问题 3 演练测试」（不是正式测试）；\n"
                "  2) 数据准备完成、5 秒倒计时已结束、界面显示接口就绪；\n"
                f"  3) --base-url 与模拟器设置中的端口一致（当前 {args.base_url}）。\n"
                "端口被模拟器保留但接口未开放时，连接会得到空响应，属于正常现象。\n"
                "确认无误仍要继续，可加 --force 跳过本检查。"
            )
            return 2
        print("（--force：忽略就绪检查继续执行）")

    if not args.yes:
        print("即将连接本机模拟器执行【问题 3 演练测试】。")
        print(f"  robot_id = {args.robot_id}")
        print(f"  base_url = {args.base_url}")
        print("若当前并非演练会话，请立即终止（Ctrl+C）。")
        try:
            input("确认请输入 y 后回车：")
        except EOFError:
            print("无交互输入，已取消。")
            return 1

    cfg = StrategyConfig.from_file(args.config) if args.config else StrategyConfig()
    overrides = {}
    if args.second_a is not None:
        overrides["second_station_local_a_m"] = args.second_a
    if args.second_b is not None:
        overrides["second_station_local_b_m"] = args.second_b
    if overrides:
        cfg = replace(cfg, **overrides)
    # StrategyConfig 是 frozen dataclass，必须用 replace 派生新实例
    cfg = replace(cfg, base_url=args.base_url, robot_id=args.robot_id)
    # 参数校验失败要给人话，不要甩 traceback（此处尚未 /enter，不会有会话风险）
    try:
        cfg.validate()
    except ValueError as exc:
        print(f"\n策略参数不可用：{exc}")
        print("提示：第二检测点的局部坐标 (a, |b|) 必须在问题 2 给出的候选区域内，"
              "可用 configs/second_station_example.json 作模板试算。")
        return 4
    print(f"策略参数：调度器 {cfg.scheduler_mode}，"
          f"动态覆盖 {'开' if cfg.adaptive_search else '关'}，"
          f"联合行程 {'开' if cfg.joint_service else '关'}，"
          f"共享观测 {cfg.shared_observations}，路线2-opt {cfg.route_polish}，"
          f"多起点路线 {cfg.route_multistart}，自适应补测 {cfg.adaptive_verify}，"
          f"耗时评分 {cfg.route_time_score}，逐步重排 {cfg.step_replan}，安全清除点 {cfg.safe_clear_point}，"
          f"扫描平移 {cfg.scan_relocation}，接收距离界 {cfg.received_range_bound}，"
          f"失败重定位 {cfg.recovery_verify}，"
          f"边界动态R第二点 {cfg.dynamic_second_station}，"
          f"计划停靠点复用 {cfg.planned_stop_reuse}，"
          f"角度舍入容差 {cfg.bearing_rounding_slack_deg:g}°，"
          f"16源完成即停止 {cfg.stop_at_source_upper_bound}，"
          f"发现16源即停止搜索 {cfg.stop_search_when_all_sources_known}，"
          f"扫描点全测未清除频道 {cfg.scan_all_channels_at_scan_points}，"
          f"环半径 {cfg.scan_ring_radius_m:.0f} m × {cfg.scan_ring_count} 环点 "
          f"+ 原点，逼近步长系数 {cfg.approach_step_ratio}，"
          f"第二检测点 ({cfg.second_station_local_a_m:g}, ±{cfg.second_station_local_b_m:g}) m，"
          f"清除后复测 {'开' if cfg.post_clear_verify else '关'}")

    from cumcm_b3.backend_http import HttpBackend

    log_dir = ROOT / "logs"
    # --run-group 把不同参数组的结果隔离开，互不覆盖：
    #   logs/runs/<tag>/                    默认（基线组）
    #   logs/runs/<run-group>/<tag>/        指定分组时
    run_dir = log_dir / "runs"
    if args.run_group:
        run_dir = run_dir / args.run_group
    run_dir = run_dir / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "http.jsonl"
    config_path = run_dir / "effective_config.json"
    config_record = cfg.to_dict()
    config_record["robot_id"] = "***"
    write_json(config_path, config_record)
    backend = HttpBackend(cfg, log_path=str(log_path))
    backend.max_real_s = args.max_real_s

    strat = DogStrategy(backend, cfg, verbose=args.verbose)
    started = time.time()
    exited = False
    stats = None
    run_error = None

    try:
        stats = strat.run()          # run() 正常返回时已经调用过 /exit
        exited = True
    except KeyboardInterrupt:
        print("\n收到中断信号，正在尝试 /exit 收尾……")
    except Exception as exc:  # noqa: BLE001 —— 任何异常都必须收尾，不能把会话挂死
        run_error = f"{type(exc).__name__}: {exc}"
        print(f"\n运行中断：{type(exc).__name__}: {exc}")
    finally:
        if not exited:
            # 关键护栏：异常退出时也必须 /exit，否则会一直挂到 25 分钟窗口超时
            try:
                backend.exit()
                exited = True
                print("已调用 /exit 收尾。")
            except Exception as exc:  # noqa: BLE001
                print(f"/exit 收尾失败（接口可能已关闭）：{exc}")

    if stats is None:
        failure_record = {
            "tag": args.tag,
            "run_group": args.run_group or None,
            "mode": "问题3演练测试",
            "status": "failed",
            "error": run_error or "interrupted",
            "started_at": datetime.fromtimestamp(started).isoformat(),
            "finished_at": datetime.now().isoformat(),
            "http_log": str(log_path),
            "effective_config": str(config_path),
        }
        write_json(run_dir / "summary.json", failure_record)
        append_jsonl(log_dir / "drill_index.jsonl", failure_record)
        print(f"\n本局未正常结束。HTTP 逐条日志已写入：{log_path}")
        return 1

    known = sorted(stats.channel_records.keys())
    cleared = [ch for ch, rec in stats.channel_records.items() if rec.get("cleared")]

    # 干扰源真值：优先用命令行回填，否则从模拟器本地真值文件自动读取。
    truth_note = ""
    total_sources = args.true_sources
    if total_sources <= 0:
        truth, truth_file, _ = read_practice_truth(args.sim_data_dir, started - 60.0)
        if truth:
            total_sources = truth
            truth_note = f"（自动读取 {truth_file}）"

    if total_sources > 0:
        ratio = stats.cleared_count / total_sources
    else:
        # 未知分母时不能把“发现过目标”直接写成 100%。覆盖与内部状态
        # 仍可判断算法是否完成，但题目定义的清除比例必须记为未知。
        total_sources = 0
        ratio = None
        truth_note = "（真值不可得，清除比例未知）"

    result = build_result(args.tag, 0, total_sources, stats)
    result_row = result.as_row()
    result_row["total_sources"] = total_sources if total_sources > 0 else None
    result_row["cleared_ratio"] = ratio
    outdir = ROOT / "outputs" / "drill"
    outdir.mkdir(parents=True, exist_ok=True)
    run_record = {
        "tag": args.tag,
        "run_group": args.run_group or None,
        "mode": "问题3演练测试",
        "started_at": datetime.fromtimestamp(started).isoformat(),
        "finished_at": datetime.now().isoformat(),
        "true_sources": total_sources,
        "truth_note": truth_note,
        "result": result_row,
        "honest_ratio": ratio,
        "discovered_channels": known,
        "cleared_channels": cleared,
        "status": strat.status(),
        "trace": backend.trace,
        "events": stats.events,
        "http_log": str(log_path),
        "effective_config": str(config_path),
        "strategy_parameters": {
            "adaptive_search": cfg.adaptive_search,
            "joint_service": cfg.joint_service,
            "shared_observations": cfg.shared_observations,
            "route_polish": cfg.route_polish,
            "route_multistart": cfg.route_multistart,
            "adaptive_verify": cfg.adaptive_verify,
            "route_time_score": cfg.route_time_score,
            "step_replan": cfg.step_replan,
            "safe_clear_point": cfg.safe_clear_point,
            "scan_relocation": cfg.scan_relocation,
            "received_range_bound": cfg.received_range_bound,
            "recovery_verify": cfg.recovery_verify,
            "dynamic_second_station": cfg.dynamic_second_station,
            "planned_stop_reuse": cfg.planned_stop_reuse,
            "bearing_rounding_slack_deg": cfg.bearing_rounding_slack_deg,
            "stop_at_source_upper_bound": cfg.stop_at_source_upper_bound,
            "stop_search_when_all_sources_known": cfg.stop_search_when_all_sources_known,
            "scan_all_channels_at_scan_points": cfg.scan_all_channels_at_scan_points,
            "post_clear_verify": cfg.post_clear_verify,
            "scheduler_mode": cfg.scheduler_mode,
            "verify_near_fraction": cfg.verify_near_fraction,
            "second_station_local_a_m": cfg.second_station_local_a_m,
            "second_station_local_b_m": cfg.second_station_local_b_m,
        },
    }
    write_json(outdir / f"{args.tag}.json", run_record)
    write_json(run_dir / "summary.json", run_record)
    index_record = {
        "tag": args.tag,
        "run_group": args.run_group or None,
        "started_at": run_record["started_at"],
        "finished_at": run_record["finished_at"],
        "true_sources": total_sources if total_sources > 0 else None,
        "cleared_count": stats.cleared_count,
        "cleared_ratio": ratio,
        "total_time_s": stats.virtual_time_s,
        "mean_clear_time_s": (result.mean_clear_time_s if stats.cleared_count else None),
        "requests": stats.requests,
        "clear_attempts": stats.clear_attempts,
        "clear_success": stats.clear_success,
        "stop_reason": stats.stop_reason,
        "all_clear": strat.status()["all_clear"],
        "scheduler_mode": cfg.scheduler_mode,
        "adaptive_search": cfg.adaptive_search,
        "joint_service": cfg.joint_service,
        "shared_observations": cfg.shared_observations,
        "route_polish": cfg.route_polish,
        "route_multistart": cfg.route_multistart,
        "adaptive_verify": cfg.adaptive_verify,
        "route_time_score": cfg.route_time_score,
        "step_replan": cfg.step_replan,
        "safe_clear_point": cfg.safe_clear_point,
        "scan_relocation": cfg.scan_relocation,
        "received_range_bound": cfg.received_range_bound,
        "recovery_verify": cfg.recovery_verify,
        "dynamic_second_station": cfg.dynamic_second_station,
        "planned_stop_reuse": cfg.planned_stop_reuse,
        "bearing_rounding_slack_deg": cfg.bearing_rounding_slack_deg,
        "stop_at_source_upper_bound": cfg.stop_at_source_upper_bound,
        "stop_search_when_all_sources_known": cfg.stop_search_when_all_sources_known,
        "scan_all_channels_at_scan_points": cfg.scan_all_channels_at_scan_points,
        "post_clear_verify": cfg.post_clear_verify,
        "verify_near_fraction": cfg.verify_near_fraction,
        "second_station_local_a_m": cfg.second_station_local_a_m,
        "second_station_local_b_m": cfg.second_station_local_b_m,
        "run_dir": str(run_dir),
    }
    append_jsonl(log_dir / "drill_index.jsonl", index_record)

    print("\n=== 演练测试结束 ===")
    print(f"案例标签        : {args.tag}")
    print(f"发现频道数      : {len(known)}  {known if len(known) <= 20 else ''}")
    print(f"清除干扰源个数  : {stats.cleared_count}")
    total_display = str(total_sources) if total_sources > 0 else "未知"
    print(f"干扰源总数(真值): {total_display} {truth_note}")
    print(f"定位清除总时间  : {stats.virtual_time_s:.1f} s "
          f"({stats.virtual_time_s/60:.1f} min)")
    print(f"平均定位清除时间: {result.mean_clear_time_s:.1f} s")
    print(f"HTTP 请求数     : {stats.requests}")
    print(f"清除尝试/命中   : {stats.clear_attempts}/{stats.clear_success}")
    if cfg.adaptive_search:
        print(f"访问扫描点数    : {stats.scan_points_visited}（动态布局）")
        print(f"完成依据        : {strat.status().get('completion_basis')}")
    else:
        print(f"访问扫描点数    : {stats.scan_points_visited}/{len(strat.scan_points)}")
    print(f"停止原因        : {stats.stop_reason}")
    print(f"计时最大偏差    : {getattr(backend, 'max_drift', 0.0):.3f} s（本地记账 vs 模拟器回报）")
    print(f"墙钟耗时        : {time.time() - started:.1f} s")
    print(f"HTTP 逐条日志   : {log_path}")
    print(f"本局日志目录    : {run_dir}")
    print(f"明细已写入      : {run_dir / 'summary.json'}")

    if ratio is None:
        print("\n清除比例        : 未知（未取得本局干扰源总数）")
    else:
        print(f"\n清除比例        : {stats.cleared_count}/{total_sources} = {ratio:.4f}")
    if ratio is not None and ratio < 1.0:
        missed = [ch for ch in range(1, 21) if ch not in set(cleared)]
        print(f"！！ 有 {total_sources - stats.cleared_count} 个源未被清除。")
        print(f"   - 已发现但未清除的频道：{[ch for ch in known if ch not in set(cleared)]}")
        print(f"   - 全程从未出现的频道  ：{[ch for ch in missed if ch not in known]}（若真值大于发现数，说明有漏检）")
        print("   漏检说明扫描点没覆盖到，建议换 configs/conservative_1300.json（余量 63.5 m）重跑。")

    print("\n请在模拟器中导出本次日志，文件名保持不变，放入支撑材料。")
    if not strat.status()["all_clear"]:
        print("内部完成判据未满足，本局按失败返回。")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
