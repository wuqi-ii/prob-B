# B题第二问：第二检测点选择策略（robust second-station selection）

## 参数化 R 版本（prob 2new）

本目录新增 `source_far_m=R`，把“第一次示向后的源最远可能距离”与设备物理最大接收距离
`max_reception_m=1500` 分离。`h(R)=R tan(eps)`，候选区域、场景抽样和约束审计均随 `R`
重算。批量入口为 `python code/run_parametric.py --config configs/parametric_explore.json`，结果见
`outputs/parametric_R/`；详细说明见 `PARAMETRIC_R.md`。未设置 `source_far_m` 时严格回退到旧模型。

在保留三项硬保证（再次接收信号、交会角 ≥ 30°、定位区域直径 ≤ 约 169 m）的固定候选区域内，
用「移动时间 + 平均定位表现 + 尾部定位表现（CVaR）」的综合代价选择第二检测点。

方法推导见 `ques 2.md`；模型与数值边界见 `MODEL.md`；**结果汇总与亮点见 `res.md`**；进度与实测结果见 `WORK_PROGRESS.md`。

## 运行方法

```bash
# 依赖（本地 Python 3.11 已具备）
pip install -r requirements.txt

# 完整求解（含图与权重敏感性）
python code/run.py --config configs/default.json

# 靠近地图边界的演示配置
python code/run.py --config configs/boundary.json

# 快速求解（不出图、不做敏感性）
python code/run.py --config configs/default.json --skip-plots --skip-sensitivity

# 仅凭已保存数据重绘全部图
python code/plot_results.py --name default

# 加密重绘前两张主图（候选区域 / 代价深浅），不重新优化
python code/refine_figures.py --all

# 测试（49 项：几何、物理约束、审计夹逼、场景、CVaR、选点、绘图、修复回归）
python -m pytest tests -q
```

输出位置：`outputs/<name>/`（summary.json、candidates.csv、sensitivity.csv、
scenarios.npz、candidates.npy、pareto_mask.npy、audit.txt、visual_review/）与
`figures/<name>/`（5 张主图的 PNG+PDF，前两张另有 SVG 与 `refined/` 加密版）。

## 模块说明

```text
code/bearing2/
  config.py      参数校验与派生几何量（h、t、种子）；含 rho/alpha/场景数/权重倍数
  geometry.py    半平面裁剪、前向扇形、圆域外包含、旋转卡壳直径、退化回退、
                 游程简化 + 斜接外扩（保守包含）、坐标变换
  region.py      源可行域 K 外包含、候选区域 C 的裕量/回代检查、169 m 上界、
                 解析边界采样与精确尖端（可行性二分）
  scenarios.py   共享场景抽样：面积均匀 + 地图拒收 + 对偶镜像 + 多重度加权 + 误差分布
  metrics.py     逐场景直径（只用读数，不泄露真实源）、均值、加权 CVaR、最大值
  optimize.py    网格+边界候选评价、帕累托前沿、lambda0 归一化（ranking/basis 标注）、
                 有界模式搜索细化、权重敏感性
  risk.py        连续最坏读数的区间夹逼审计（上界=加宽扇形，下界=采样最大）
  pipeline.py    求解串联、硬约束回代、独立场景验证、分辨率检查、数据导出
  plotting.py    五张图入口（前两张转交 publication_plots）
  publication_plots.py  候选区域与代价深浅图的审查版实现（分侧插值、掩码）
code/refine_figures.py   只重绘主图的命令入口
q1_code/         第一题交付：直径求解内核与对照验证（第二问的几何内核与其同源）
```

## 结果解读（默认配置实测，2026-09-11 修复后重算）

- 最终点（局部坐标，x 沿第一示向度方向）：**(764, 526) m**，移动 **927.56 m / 185.51 s**（另加检测 5 s）；J = 343.25 s。
- 场景直径：平均 **65.67 m**，CVaR₀.₉ **117.50 m**，最大 **124.02 m**，连续最坏审计 **[124.263, 124.312] m**——全部低于 169 m 统一上界（实际用到约 73.5%）。
- 回代硬约束：接收 ✓、交会角 ✓（最小实测 33.18°）、强信号盲区排除 ✓、最小松弛 57.58 m。
- 对照：最近可行点为候选区尖端 (666.814, 526.811)，距离更近（849.81 m）但平均直径 **71.95 m**、CVaR **131.58 m**——"最近"不是免费午餐。
- 权重敏感性：λ 从 0.25λ0 到 4λ0，选点从最近尖端 (666.814, 526.811) 移到 (852.000, 523.542)（正好落在接收圆上，L = 1000.00 m）；λ 增大时平均直径下降、移动时间上升。
- 边界配置：最优解为**角点解** (666.814, 526.811)，交会角恰为设计下限 30.0000°，被"远端接收圆 + 远端交会角"两条约束同时卡住。

## 假设与限制（引用结论时必须一并说明）

1. 第一检测点与示向度是演示输入。
2. 源按面积均匀、误差按 ±1° 均匀是**声明的工作假设**；独立验证集只证明稳定性，不构成无偏终评。
3. 网格 + 局部细化不等于连续空间全局最优。
4. λ0 是归一化折中约定，不是已证明的真实后续耗时系数。
5. ρ=0.5、α=0.9 为可修改基准；目前实现的是 λ 扫描，尚未实现 ρ/α 二维敏感性，结论应连同参数一起引用。
6. 直径 ≤ 40 m 不保证存在 20 m 覆盖圆；判断"可一次清除"应使用最小包围圆（本问未展开）。

## 新版候选区域图与代价深浅图（2026-09-11）

在项目目录执行：

```powershell
python code/refine_figures.py --all
python -m pytest -q
```

只画一个案例可用 `python code/refine_figures.py --name default --mesh-step 5`。

- `code/bearing2/publication_plots.py`：解析候选边界、读取保存场景、加密评价代价、分侧掩码插值及排版。
- `code/refine_figures.py`：只绘图的命令入口，不运行优化、不替换保存点。
- `figures/default/refined/` 与 `figures/boundary/refined/`：每例两张新图，均有 PNG/PDF/SVG；同时同步到外层标准文件名 `fig1_constraints_region`、`fig2_cost_heatmap`，避免误开旧图。
- `01_candidate_region`：整体几何关系和所选侧局部放大，标明接收/角度约束边界。
- `02_cost_heatmap`：只展示上侧候选区域，颜色越深代价越低；下侧求值仍保留在审计缓存中。
- `outputs/<案例>/visual_review/`：新增的绘图样本与缓存；保留原始求解输出。

原网格最近点不是连续区域精确最近点；新图中的 P* 表示已保存选点，不表示已证明全局最优。每例 1777 个加密显示样本使用原 scenarios.npz 和原 λ 求值，只在可行区域内作分侧线性插值。

**引用结果前请阅读 [PLOT_REVIEW.md](PLOT_REVIEW.md)**（外部代码核查记录）与 [res.md](res.md) 的"外部核查与已修复问题"一节。核查提出的问题已逐条核实并修复：非对称地图下的镜像重复计权（改为多重度倒数加权）、`offset_outward` 未真正外扩（改为斜接偏移）、多边形简化的偏差不受控（改为游程合并）、λ 退化语义与备注不符（新增 `ranking`）、平局规则可能抬高代价（窗口收窄且瓣内取最小代价）、"最近点"表述（改用精确尖端）、最优解落在角点而粗网格够不到（候选集加入解析边界与精确尖端）。修复后两个案例已完整重算，49 项测试通过。
