# CUMCM 2026 B 题 · 问题 3：机器狗干扰源自动搜索、定位与清除

本目录是问题 3 的完整工程：**策略引擎 + 离线自验模拟器 + 真实模拟器适配层 + 验证数据 + 文档**。

### 2026-09-12：边界动态 R 第二检测点实验（暂不替换主配置）

新增 `configs/optimized_dynamic_R.json`：首次示向扇形被场地边界截短时，计算安全径向上界 `R`，并用
问题2新参数函数 `P*(R)` 生成第二检测点；未截短时仍用 `(764,±526)`。200局与
`optimized_threefix` 同种子配对均为2633/2633全清，但动态方案平均慢6.23秒；加入原点共同参与路线评分后
仍平均慢3.29秒且差异区间跨0。说明孤立第二点更近不必然缩短完整服务路线。功能作为实验开关保留，主配置不改。
详见 `logs/experiments/dynamic_R_validation_20260912/README.md`。

### 2026-09-12：三项日志驱动优化

**真实演练 optimized_threefix01 已完成：11/11全清，3182.29秒（53.0分钟），171请求，11次清除全部命中，7/7扫描任务完成，正常all_done退出。** 本局实际触发1次经连续覆盖证明的扫描点平移；17次共享观测；没有触发失败重定位、give_up或清除失败。服务端与本地虚拟时间一致。日志位于 `logs/runs/optimized_threefix/optimized_threefix01/`。同为11源的历史drill10为3530.88秒，但源位置不同，不能将348.59秒差值全部解释为策略收益。

新增候选 `configs/optimized_threefix.json`：按真实0.01°方向输出增加0.005001°舍入容差；成功清除16源后依据题设上限立即停止；在连续1000米覆盖证明成立时，把下一扫描点向服务路线平移并按完整任务路线评分。固定七点保留为不可证明时的兜底。

新种子独立200局：基线3473.24秒，候选3223.45秒（-7.19%），两组各2615/2615全清。相对不平移的稳健候选再省14.33秒；全1000米50局再省15.24秒，各667/667全清。固定16源50局中，提前停止单项省45.34秒。98项测试通过，包括±1.005°极限误差与历史失败案例。详见[三项修改及证据](logs/experiments/threefix_validation_20260912/README.md)。本轮未连接真实模拟器，默认配置未替换。

> 后续扩展验证发现：原多起点候选在seed32181留下1个已发现目标（近乎平行的示向导致无效逼近后放弃）。此前200局全清结论仅适用于当时样本，不能当普遍保证。正在/已由后续恢复版验证替代，见 `logs/experiments/refine_validation_20260912/FAILURES.md`。旧结果保留，不能把本轮未全清均时用作提速证据。

### 2026-09-12：共享观测候选（默认未改）

**后续更优候选：`configs/multistart_shared_01.json`。** 在新的独立200局中，原基线3467.84秒，上轮候选3324.18秒，新多起点路线候选3216.20秒，分别比原基线省4.14%、7.26%；新候选比上轮再省3.25%。各2588/2588全清。50局全1000米压力测试省8.04%，各648/648全清。88项自动测试通过；本轮1330个离线运行全部完成，无异常续跑，但不能认为上轮异常根因已修复。详见 [本轮完整记录](logs/experiments/forward_validation_20260912/README.md)。

新配置保留七点覆盖，比较多条起步路线并2-opt优化，每次取得反馈后重排。`route_multistart=true`，`adaptive_verify=false`；后者已实现但效果不佳，不启用。演练日志同步记录两个开关。默认仍不变，待用户开启演练后可用 `python scripts/run_drill.py --config configs/multistart_shared_01.json --tag multistart01 --verbose`。本轮未连接真实模拟器。

独立200局：原策略3491.30秒，`configs/shared_near_01.json` 候选3346.84秒，平均省144.46秒（4.14%），两组各2594/2594全清；候选160局更快、40局更慢。50局全1000米接收半径测试平均省5.78%，各656/656全清。不是大幅提速或最优解证明。

保留七点1000米覆盖保证和标准第二检测点，先试点比例改为0.1，按可行域收缩收益顺路测其他频道。新增 `shared_observations=off/opportunistic/batch` 和 `route_polish`，演练摘要、索引与有效配置均记录。原基线不变。

离线试用：`python scripts/run_offline.py --config configs/shared_near_01.json --cases 50 --start-seed 28000 --outdir outputs/shared_candidate_28000`。
待用户开启演练后，可用 `python scripts/run_drill.py --config configs/shared_near_01.json --tag shared01 --verbose`，本轮未执行。

85项自动测试通过。批量进程曾异常退出，已保留失败记录并原代码续跑，根因未明。见 [实验结果及复现](logs/experiments/shared_validation_20260912/README.md) 和 [异常记录](logs/experiments/shared_validation_20260912/runtime_notes.md)。

> 策略原型由队员给定（七点覆盖 + 滚动最近邻 + 机会性复测），本目录负责把它工程化、
> 用离线模拟器做零风险闭环验证，最后只通过**演练测试**对接真实模拟器。
>
> **本工程不会、也无法触发正式测试。** 模拟器只对本机 HTTP 端口暴露
> `/enter`、`/measure`、`/clear`、`/exit` 四个动作接口，测试模块的选择只能在
> 模拟器界面上由人点击。所有脚本在动作前都先做端口探测，端口不通即静默退出。

---

## 1. 快速开始

```bash
# 0) 环境：Python >= 3.9，运行期零第三方依赖
python -V

# 1) 跑单元测试（含几何、覆盖、第二检测点、协议和端到端闭环）
python -m unittest discover -s tests -v

# 2) 离线批量验证（本地自建模拟器，随机案例，零风险）
python scripts/run_offline.py --cases 100 --start-seed 1000 --outdir outputs/baseline100

# 3) 参数敏感性扫描
python scripts/sweep.py --cases 30 --start-seed 300

# 4) 【仅演练测试】连真实模拟器
#    先在模拟器界面点击「问题 3 演练测试」，等倒计时结束、接口就绪，再执行：
python scripts/run_drill.py --tag drill01 --verbose
```

第二检测点的局部坐标可直接覆盖。`a` 沿第一次示向方向，程序自动在 `+b`、`-b`
两个镜像位置中选择路径较短的一侧：

```bash
# 真实演练：使用 (750, ±520) m，并在日志中记录这组参数
python scripts/run_drill.py --tag second_750_520 --second-a 750 --second-b 520 --verbose

# 先用同一组坐标做离线批量测试
python scripts/run_offline.py --cases 100 --second-a 750 --second-b 520 --outdir outputs/second_750_520
```

2026-09-12 时间下界诊断记录见 `logs/analysis/oracle_20260912/README.md`：同200局当前平均3486.4秒，已知真值时的最优清除时间均值被夹在1742.5–1842.8秒；此区间不包含未知目标搜索成本。此前将覆盖与目标访问两条下界相加、据此声称距最优不足3%的判断不成立。

调度器支持两种可复现模式：默认 `route_insert` 先精确规划剩余扫描点的最短开放路线，
再按最小绕路增量插入验证和清除任务；`nearest` 保留为旧式最近邻对照。两者都保持
七点布局和最坏覆盖距离 988.5 m 不变。最近邻对照配置为
`configs/scheduler_nearest.json`。

2026-09-12 新增两个**默认关闭**的实验开关：`adaptive_search`（逐频道覆盖证明与动态扫描位置）和
`joint_service`（验证后的预计清除行程计价）。独立200局四组全部全清；动态覆盖平均快1.12%，
联合行程平均慢0.54%，组合平均快0.57%但优势未得到统计确认，因此保留原默认策略。
候选配置为 `configs/experiment_adaptive.json`、`experiment_joint.json`、`experiment_combined.json`。
完整方法、命令和记录见 [实验记录](logs/experiments/search_validation_20260912/README.md)。

也可以复制 `configs/second_station_example.json`，修改两个字段后通过 `--config` 加载。
坐标不满足问题2候选区域的硬约束时，程序会在连接模拟器前拒绝运行。

`scripts/run_drill.py` 会先做**就绪探测**：向接口发一个 `GET /`（不是四个动作之一，
不消耗任何测试资源），只有收到符合协议的 JSON 响应才认为接口就绪；不可用就直接退出，
**不发送任何动作**（返回码 2）。可连时默认还要人工输入 `y` 二次确认（`--yes` 可跳过）。

> **为什么不能只看端口**：模拟器启动时就会**保留** TCP 端口，会话未开始或已结束时端口仍可连，
> 但 HTTP 层会直接关闭连接（空响应 / RST）。只看端口就 `/enter`，会把脚本指向一个并未就绪的会话。

---

## 2. 目录结构

```
prob 3/
├─ README.md                本文件：运行方法、目录说明、指标口径
├─ MODEL.md                 ★ 模型与算法：策略设计、覆盖性证明、伪代码、假设与限制
├─ RESULTS.md               ★ 结果：离线基准、敏感性分析、演练测试记录
├─ WORK_PROGRESS.md         进度看板与待办
├─ requirements.txt         运行依赖说明（运行期零第三方依赖）
├─ configs/                 策略参数配置（可直接用 --config 加载）
│   ├─ baseline.json            基准：环半径 1150 m
│   ├─ conservative_1300.json   保守：环半径 1300 m（余量 +63.5 m）
│   ├─ robust_1560.json         理论最优覆盖：环半径 1560 m
│   └─ second_station_example.json  第二检测点坐标试验模板
├─ src/cumcm_b3/            核心库
│   ├─ config.py                赛题物理常量 + 策略参数 + 覆盖最坏距离解析式
│   ├─ geometry.py              交会定位内核（半平面裁剪）、凸包、旋转卡壳直径
│   ├─ coverage.py              七点扫描布局与覆盖校核
│   ├─ second_station.py        第二检测点候选域（复用问题 2 结论）
│   ├─ tracker.py               20 个频道的观测账本与可行域推断
│   ├─ strategy.py              ★ 策略主引擎（三类任务 + 覆盖主路线 + 最小绕路插入）
│   ├─ shared_strategy.py       叠加：机会性共享观测、开放路线 2-opt、停靠复用
│   ├─ experimental_strategy.py 叠加：动态覆盖/联合行程实验 + 策略工厂 make_strategy
│   ├─ forward_strategy.py      叠加：多起点路线、恢复验证、源数上界停机
│   ├─ refined_strategy.py      叠加：稳健清除点、逐步重规划
│   ├─ relocated_strategy.py    叠加：扫描点平移（须通过连续覆盖证书）
│   ├─ coverage_certificate.py  连续覆盖证书（扫描点平移的准入判据）
│   ├─ backend_offline.py       离线自验模拟器（复现赛题全部规则）
│   ├─ backend_http.py          真实模拟器适配层（含墙钟看门狗）
│   ├─ client.py                模拟器 HTTP 客户端（复用自 web/simulator_client.py）
│   ├─ metrics.py               四项统计指标与结果导出
│   └─ __init__.py
├─ scripts/
│   ├─ run_offline.py           离线批量验证
│   ├─ run_drill.py             ★ 演练测试入口（唯一真实通路）
│   └─ sweep.py                 参数敏感性扫描
├─ tests/                   11 个测试模块，共 111 项（几何 / 覆盖 / 协议 / 调度 /
│                           各策略层 / 产物序列化 / 配置守卫）
├─ pyproject.toml           ruff 静态检查规则（当前零告警）
├─ ref/                     赛题资料（附件1 通信协议）
├─ logs/                    演练日志与汇总索引
│   ├─ drill_index.jsonl        每局一行核心指标，方便批量比较
│   └─ runs/<测试标签>/         每局独立保存 HTTP、摘要和实际配置
├─ outputs/                 结果数据（baseline100 / sweep / drill / offline*）
└─ docs/
    └─ STRATEGY.md          策略原型的形式化汇总（队员给定版本的完整表述）
```

---

## 3. 四项统计指标的口径

赛题要求报出的量必须在论文里写清口径，本工程统一如下：

| 指标 | 口径 |
|---|---|
| **被清除干扰源个数** | 收到 `clear_result = success` 的**不同频道**数（一个源占一个频道，故等于源数） |
| **干扰源总数** | 该案例实际生成的源数；演练时读取模拟器生成的本局真值文件，无法取得时标为未知 |
| **清除比例** | 被清除个数 / 干扰源总数 |
| **定位清除总时间** | 全程虚拟时间：移动 + 切换频道(1 s) + 检测(5 s) + 光学定位(3 s) + 清除(2 s) |
| **平均定位清除时间** | 定位清除总时间 / 被清除个数 |

本工程的“总时间”即赛题定义的**虚拟时间**；另有 `real_elapsed_s`（真实墙钟）仅作诊断，
不参与成绩计算，但受 20 分钟程序运行时限约束。

---

## 4. 关键物理常量（取自赛题附录，代码中不可调）

| 量 | 值 |
|---|---|
| 目标区域 | 半径 1800 m 圆域 |
| 干扰源个数 | 10 ~ 16（未知） |
| 频道数 | 20（一个源占一个频道） |
| 有效接收半径 | 1000 ~ 1500 m（未知，保障设计按最坏 1000 m） |
| 示向度误差 | ±1° |
| 移动速度 | 5 m/s |
| 切换频道 | 1 s |
| 单次检测 | 5 s |
| 光学精确定位 | 3 s |
| 激光清除 | 2 s |
| 信号过强盲区 | ≤ 5 m（无示向度） |
| 可精确定位距离 | ≤ 20 m |
| 虚拟时间上限 | 100 h |
| 程序运行上限 | 20 min |

---

## 5. 安全约定（务必遵守）

1. **只用演练测试。** `run_drill.py` 是唯一入口，且不含任何测试模块选择逻辑。
2. **跑前先确认界面。** 执行前请目视确认为「问题 3 演练测试」、倒计时已结束。
3. **可中断。** 默认要求人工 `y` 确认；运行中 `Ctrl+C` 会尝试 `/exit` 收尾。
4. **任何异常都会 `/exit`。** 使用 `finally` 护栏，协议错误也不会把会话挂到 25 分钟窗口超时。
5. **看门狗。** `backend_http` 内置墙钟看门狗（默认 900 s）与请求数上限，
   接近即主动 `/exit`，避免被判超时。
6. **凭据不入库。** 队伍号 `202623001124` 可写入配置；**密码只在模拟器登录界面使用**，
   不写入任何配置或日志文件。
7. **就绪探测先于一切动作。** 见下文第 1 节与第 7 节。

---

## 6. 演练日志记录

每次运行 `scripts/run_drill.py --tag <标签>` 都会自动创建：

```text
logs/runs/<标签>/http.jsonl
logs/runs/<标签>/summary.json
logs/runs/<标签>/effective_config.json
```

其中 `http.jsonl` 可逐条回放动作和响应，`summary.json` 包含清除比例、耗时、频道记录、
策略事件和轨迹，`effective_config.json` 保存该局实际参数。核心指标还会追加到
`logs/drill_index.jsonl`，后续可直接读取这个索引比较不同配置。

历史演练 `drill02`、`drill03` 保持原位置：HTTP 日志在 `logs/`，摘要在
`outputs/drill/`。新规则从下一次演练开始生效。

## 7. 已知边界与后续工作

- 第二轮真实演练已完成`round2_01`—`round2_05`共5局，累计71/71全清，所有清除全部一次命中，均完成7/7扫描且无恢复、失败或放弃。逐局日志和汇总位于`logs/runs/second_round/`，详见`ROUND_SUMMARY.md`。
- 修复后的 `drill02`—`drill12` 十一次真实演练累计清除 **147/147** 个目标，清除动作 147 次全部命中、
  零 HTTP 错误，均正常 `all_done` 并主动退出。三组口径不同、**不可直接同比**：drill02–drill10 为基线配置
  （`verify_near_fraction=0`），drill11–drill12 用的是**当时的默认配置**（`verify_near_fraction=0.4`），
  而当前推荐配置 `configs/optimized_threefix.json` 已把该比例调为 **0.1**。样本量仍有限，
  当前结论是连接与策略闭环已经跑通，不能据此宣称路线最优。
- 环半径 1150 m 的最坏点间距为 988.5 m，余量仅 11.5 m。若对边界源不放心，
  用 `--config configs/conservative_1300.json` 换 1300 m（余量 63.5 m，代价约 +8% 行程）。
- 详见 `WORK_PROGRESS.md` 的待办清单。

## 8. 排查手册（现场经验）

| 现象 | 原因 | 处理 |
|---|---|---|
| 就绪探测失败：`端口可连但被对端立即关闭` | 端口被模拟器保留，但会话未开始或已结束 | 回模拟器界面确认会话状态；这是正常现象 |
| `TCP 连接失败` | 模拟器没在运行，或端口改了 | 检查模拟器设置里的端口与 `--base-url` |
| HTTP 400，响应体只有 `accepted:false`，**没有任何 message** | 请求字段不合法 | 模拟器的响应体不携带原因；去界面「指令与反馈」看诊断。常见原因：字段名写错、少了必需字段（`/measure` 与 `/clear` 都需要 `position` + `channel`）、**多带了未知字段** |
| 提示"必须先 /enter" | 接口未开放或未先调用 `/enter` | 正常流程不会出现，属于脚本装配问题 |
| 会话挂住、界面一直显示测试进行中 | 策略异常退出而未发 `/exit` | 已由 `run_drill.py` 的 `finally` 护栏兜住；必要时用界面的「中止测试」 |
| `--true-sources` 读不到 | `--sim-data-dir` 指错了 | 指向 `JammersSimulatorData/behavior-logs`；该目录只在演练后出现新文件 |

> **字段白名单**：机器狗请求体只允许 `arena_id`、`robot_id`、`request_id`、`position`、`channel`
> 五个键，模拟器用严格白名单解析，**多一个字段也会被 400 拒绝**。
> 这条已由 `tests/test_http_protocol.py` 固化。

## 9. 当前最终策略与论文材料（2026-09-12）

当前推荐配置为 `configs/optimized_threefix.json`。最终口径是：原点加1150 m正六边形六点完备扫描，第二检测点局部坐标 $(764,\pm526)$ m，近端试探比例0.1，$1.005001^\circ$ 鲁棒楔形，多起点路线插入与开放2-opt，机会性共享观测，带连续1000 m覆盖证书的扫描点平移，清除后复测、异常恢复及清除16源后的题设上限停机。

论文与说明材料：

- `docs/STRATEGY.md`：当前最终策略规格；
- `MODEL.md`：与最终代码一致的数学模型及覆盖证明；
- `reports/ANALYSIS_MODELING_REPORT.md`：建模分析与敏感性预检；
- `reports/RESULTS_REPORT.md`：离线配对实验、五次演练和正式测试待填表；
- `docs/PAPER_Q3_WRITING_GUIDE.md`：第三问写作注意事项；
- `paper/sections/7_problem3.typ`：可并入总论文的第三问正文；
- `output/pdf/problem3_chapter.pdf`：独立排版预览。

历史章节中的旧配置与旧结论仅用于保留优化过程，不再代表当前策略。正式结果仍须在三次正式测试后填入，五次演练不得冒充正式测试。

## 10. 动态 R 后续优化结论（2026-09-12）

- 新增默认关闭的 `planned_stop_reuse`：仅当机器狗已经到达的停靠点严格落入某频道的问题2候选区域时，原地兼任第二检测点；检查包含1000米最坏接收、31度交会角与示向显示舍入容差。
- 30局中严格复用没有触发，结果与动态R混合策略逐局一致；说明它安全但机会很少。
- 尝试把验证软延期到未来扫描点，虽减少请求却平均增加23.53—49.16秒，原因是破坏全路线顺序；相关代码已删除，只保留实验日志。
- 尝试给动态点增加50—400米路线节省门槛也无收益，参数已删除，避免继续膨胀模型。
- 当前推荐配置仍是 `configs/optimized_threefix.json`。动态R和停靠复用只作为独立实验，详见 `logs/experiments/planned_stop_reuse_20260912/README.md`。

## 11. 精简动作候选（2026-09-13）

新增待真实演练的候选 `configs/optimized_lean_actions.json`：扫描点跳过已有两次方向的频道，发现16个频道后立即停止剩余搜索，正常 `/clear success` 后不再固定复测，并启用路线感知安全清除点与逐步重排。

新种子60000—60199同案例配对200局：原 `optimized_threefix` 平均3217.395秒，候选3068.079秒，平均节省149.316秒（4.64%），近似95%区间为节省138.160—160.471秒；196局更快、4局更慢，两组均2611/2611全清。平均请求数166.37降至139.44。

全1000米接收半径50局：3378.899秒降至3246.517秒，两组均671/671全清。16源案例候选平均约201.1秒/源，最好172.1秒/源，已覆盖“最好约180秒”的量级。当前主配置仍不覆盖，下一步应先做真实演练。完整记录见 `logs/experiments/lean_actions_20260913/README.md`。

首次真实演练 `lean01` 已完成：完成7/7覆盖扫描，发现并清除11个频道，11次清除全部命中，已知未清除集合为空，正常`all_done`；总虚拟时间3291.587655秒，请求145次。触发1次认证扫描平移、14次共享观测、10次安全清除点和25次逐步重排测量，无清除失败、恢复或放弃。模拟器真值文件本局不可得，因此不虚构真值清除比例。完整记录见 `logs/runs/lean_actions_real/README.md`。
