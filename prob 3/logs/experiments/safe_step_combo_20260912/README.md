# safe_step 三开关叠加验证

## 背景（为什么重测）

`refine_validation_20260912` 当年测过 `safe_step` 系列：比 `forward_route` 更快
（200 局 3217.69 vs 3229.13 s），但 **perfect 只有 199/200**——seed32181 因"两次
方向接近平行（240.68°/242.00°）导致可行域很长、逼近连续 no_signal 后 give_up"而漏 1 个源。
该缺陷后来由 `recovery_verify` 修复并被采纳，但 **`safe_step` 的三个组件
（`safe_clear_point` / `step_replan` / `received_range_bound`）没有被重新组合测试**，
随 `optimized_threefix.json` 一起设为 `false`。

本轮在**修复后的代码**上重新组合这三开关，做配对验证。

## 做法

- 基线：`configs/optimized_threefix.json`
- 候选：`configs/optimized_safe_step.json` = 基线 + `safe_clear_point`、`step_replan`、`received_range_bound` 三个开关置 `true`。
- 只跑离线模拟器，未连接 2026 端口。
- 入口：`cumcm_b3.experimental_strategy.make_strategy`（RelocatedStrategy，继承链含 Refined→Forward→Shared→Experimental）。

## 结果（配对，同种子）

### 常规 200 局（seed 40000–40199）

| 方案 | 平均时间 | 未全清 | 配对差 |
|---|---:|---|---:|
| baseline | 3223.45 s | 无 | — |
| **safe_step** | **3207.80 s** | **无** | **−15.65 s，95%CI [−26.2, −5.1]**（不含 0） |

- 变化 **−0.49%**；200 局中 **140 局更快 / 60 局更慢**；2655/2655 源全清。
- 相对更早的原基线（3473.24 s）累计 **−7.64%**。

### 全 1000 m 接收半径 50 局（seed 41000–41049，最坏接收半径）

| 方案 | 平均时间 | 未全清 |
|---|---:|---|
| baseline | 3259.54 s | 无 |
| **safe_step** | **3243.26 s** | **无** |

- 变化 **−0.50%**，667/667 源全清。**不损害最坏情况鲁棒性。**

### 历史失败种子 32181

| 方案 | 清除 | 时间 |
|---|---|---|
| baseline | 12/12 | 3478 s |
| **safe_step** | **12/12** | **3230 s** |

`recovery_verify` 已修复该缺陷，三开关叠加后仍全清且更快。

## 单开关消融（80 局 seed 40000–40079）

| 开关 | 变化 |
|---|---:|
| `+safe_clear_point` | −0.21% |
| `+step_replan` | −0.09% |
| `+received_range_bound` | −0.06% |
| **三者全开** | **−0.60%** |

三者有叠加效应，故保留组合。

## 结论

统计显著（95% CI 不含 0）、鲁棒性不退化、历史失败种子已修复。
`configs/optimized_safe_step.json` 为**推荐候选**；本轮不自动替换
`configs/optimized_threefix.json`，也不作为默认，需真实演练确认后再定。

## 产物

- `comparison.json`：200 局 + 50 局汇总与配对统计
- `cases.jsonl`：逐局 baseline / safe_step 时间
- `baseline_config.json` / `safe_step_config.json`：两侧有效配置

复现：

```powershell
python - <<'PY'
# 见本目录生成脚本；核心为在 optimized_threefix 上 replace 三个开关为 True
PY
```
