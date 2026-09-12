# 未通过的验证与数值问题（保留原始结果）

本目录是修复前验证，不是全清提速证据。seed32181三组优化策略均只清除11/12；原基线12/12。forward_route/safe_step/relocated_safe_step合计各2585/2586，perfect_cases=199/200。不要引用其未全清均时作为性能优势。

频道4的头两次方向240.675919°、242.003349°接近平行，可行域仍很长。向估计位置逼近后连续no_signal，4次尝试后永久give_up；随后扫描再次获得有效方向也未恢复。新增recovery_verify：失败后最多两次去标准第二点重定位，获得真实观测后重排；不是把已发现目标直接标为清除。原seed及同批200场景后由recovery_regression_20260912复测全清。

扫描平移还发现坐标重合时以list.index找扫描点会标错逻辑任务。定向单测先失败，改为使用稳定S<index>编号后通过。旧随机验证未观察到这个碰撞，仍保留修复前快照。

极限±1°误差测试最初8/10：输出保留6位小数使真实方向误差出现-1.0000004644°，原严格±1°半平面求交为空。证据在../extreme_before_20260912，两侧误差各10个目标、观测、真值均保留。这与覆盖不足无关，需要对输出舍入设置微小保守容差，不能将失败场景删除。

运行记录：range_pilot任务恢复时进程句柄已不存在，approach070完成22局后未有最终comparison；原代码参数续跑剩余种子完成，没有跳过。recovery_regression最初完成25局后clean_polygon float调用出现TypeError: tuple object is not callable，原代码续跑后完成200局。recovery_validation先完成baseline200、recovered43后退出码1，无堆栈；续跑后recovered200、safe_step77时同类TypeError，已由改进的实验脚本保存failure_<seed>_<attempt>.json。根因未明，不称运行零故障，也不将其归咎于某个已知Python缺陷。
