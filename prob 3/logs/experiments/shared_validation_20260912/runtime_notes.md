# 运行异常记录

补充：validation 第一次续跑完成 batch_f02 的157局后退出码1，没有返回堆栈。第二次续跑从 seed26157 开始，补齐200局并退出码0。最终 audit.json 核验种子完整、没有跳过失败种子。不完整时审计曾按预期拒绝通过。

本轮全部为离线实验，Python 3.11.9。下列异常没有删除对应种子，没有修改策略绕过，也不计作成功的第一次运行。

- pilot 首次进程在 baseline 完成27局后中断：geometry.py clean_polygon 第33行调用 float 时，TypeError: 'tuple' object is not callable。新进程检查未发现 float 全局覆盖；源码未变化。使用 --resume 从 seed25027 重跑后完成。
- validation 首次进程在 batch_f02 完成69局后中断：geometry.py clip_halfplane 第113行调用 float 时，TypeError: 'float' object is not callable。其余三组各200局已完成。使用 --resume 从 seed26069 原代码原配置重跑。

根因未查明，不能宣称已修复或断言是 Python 自身问题。最终表格描述保留全部种子、续跑完成后的策略效果，不代表批量进程无异常。源代码哈希与已完成逐局日志由续跑机制检查；旧记录不覆盖。真实演练前应继续关注运行环境稳定性。
