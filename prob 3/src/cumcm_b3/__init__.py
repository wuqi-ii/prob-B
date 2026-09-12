"""CUMCM 2026 B 题 · 问题 3：机器狗自动搜索定位与清除策略。

包结构
------
config            全局参数与策略参数（含覆盖性几何校验）
geometry          交割定位几何内核：凸多边形表示 + 半平面裁剪 + 旋转卡壳直径
coverage          七点扫描布局与"保证不漏"的覆盖性论证
second_station    沿用问题 2 结论的第二检测点候选生成
tracker           频道状态跟踪（未发现 / 已发现 / 已定位 / 已清除）
planner           三类路径点竞争 + 滚动最近邻决策
executor          到位后的具体动作：普查扫描、补测验证、逼近与清除
strategy          顶层编排：主循环
backend_offline   本地自验模拟器（严格复现附录规则）
client            真实模拟器 HTTP 客户端（复制自 ../web/simulator_client.py）
metrics           四项统计指标与结果导出
"""

__version__ = "0.1.0"
