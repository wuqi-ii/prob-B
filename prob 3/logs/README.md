# 演练日志目录

从下一次演练开始，`scripts/run_drill.py` 会为每一局建立：

```text
logs/runs/<测试标签>/
├─ http.jsonl             每条 HTTP 请求、响应与错误
├─ summary.json           本局指标、频道、策略事件和后端轨迹
└─ effective_config.json  本局实际使用的策略参数（队号脱敏）
```

`logs/drill_index.jsonl` 每局追加一行核心指标，用于后续批量比较和参数优化。

早期演练 `drill02`、`drill03` 的逐条日志仍保留在 `logs/drill02_http.jsonl` 与
`logs/drill03_http.jsonl`，对应摘要位于 `outputs/drill/`。不会移动或改写这些历史文件。

## 历史产物的一处已知瑕疵

`runs/drill04`—`runs/drill07/effective_config.json` 里的 `virtual_time_budget_s`
被写成了 `Infinity`——Python `json.dumps` 的默认行为，但**不是合法 JSON**，
严格解析器（如 JS `JSON.parse`）会报错。

自 2026-09-11 起 `metrics.write_json` / `append_jsonl` 已统一把非有限浮点收口为 `null`，
后续产物都是严格合法的。这 4 个历史文件为了保留"当时实际生效配置"的原貌**未做改写**；
若要交给严格解析的工具读取，请先把它替换成 `null`：

```bash
sed -i 's/: Infinity/: null/' logs/runs/drill0*/effective_config.json
```
