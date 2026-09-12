# 演练日志目录（问题 4）

每次运行 `scripts/run_drill.py --tag <标签>` 会为每一局建立：

```text
logs/runs/<测试标签>/
├─ http.jsonl             每条 HTTP 请求、响应与错误
├─ summary.json           本局指标、频道、策略事件、后端轨迹与定向源真值（若可读）
└─ effective_config.json  本局实际使用的策略参数（队号脱敏）
```

`logs/drill_index.jsonl` 每局追加一行核心指标（含 `directional_sources` 定向源数量），
用于后续批量比较和参数优化。

## 说明

- `logs/runs/drill_offline_demo/` 是一次**离线自验演示**（用本地模拟器跑出，与真实演练
  完全相同的产物结构：http.jsonl / summary.json / effective_config.json），供参考日志格式。
  真实演练需在模拟器界面点「问题 4 演练测试」后运行 `scripts/run_drill.py` 才会生成。
- 产物 JSON 均为严格合法 JSON（非有限浮点统一写成 `null`）。
- 队号 `202623001124` 会写入配置但日志中脱敏为 `***`；密码只用于模拟器登录界面，
  不写入任何文件。
