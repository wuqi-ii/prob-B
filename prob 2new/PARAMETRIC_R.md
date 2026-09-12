# 第二检测点的参数化方案

将第一次示向后源的最远可能距离记为参数 `R=source_far_m`，设备物理最大接收距离仍为
`max_reception_m=1500 m`，保证接收距离仍为 `guaranteed_reception_m=1000 m`。

参数化源区域与候选区域使用

\[
h(R)=R\tan\varepsilon,\qquad
P^*(R)=(a^*(R),\pm b^*(R)).
\]

其中 `P*(R)` 由与旧方案相同的“移动时间 + 平均定位直径 + CVaR”目标数值求得。
单次求解仍保存完整的候选点、场景、约束审计和稳定性检查。

运行默认试验：

```powershell
python code/run_parametric.py
```

快速探索较宽的 `R` 范围（推荐先运行）：

```powershell
python code/run_parametric.py --config configs/parametric_explore.json
```

自定义参数网格：

```powershell
python code/run_parametric.py --r-values 400,600,800,1000,1200,1500
```

汇总结果保存在 `outputs/parametric_R/optimal_points.csv`；相邻节点构造的数值分段函数保存在
`outputs/parametric_R/piecewise_linear.csv`。插值点只是快速初值，交给第三问前必须用对应 `R`
重新校验五条硬约束。若场地截出的源区域高度不对称，精确策略仍应使用完整可行域，而不是只用
标量 `R`。

生成不同 `R` 的联合代价热力图：

```powershell
python code/plot_parametric_heatmaps.py
```
