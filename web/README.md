# CUMCM 2026 B 题模拟机连接客户端

本目录只负责连接官方模拟机及发送动作，不包含问题 1、2 的建模求解。

## 文件

- `simulator_client.py`：四个官方接口的零依赖 Python 客户端。
- `smoke_test.py`：真实模拟机的保守联调脚本，默认只探测端口。
- `web_app.py` 与 `static/`：本地网页控制台及同源代理。
- `tests/test_simulator_client.py`：本地假服务器测试，不连接官方模拟机。
- `logs/simulator.jsonl`：运行后生成的请求/响应日志（已被 `.gitignore` 排除）。

要求 Python 3.10 或更高版本，无需安装第三方包。

## 1. 下载并启动官方模拟机

官方地址：<https://pan.baidu.com/s/1P1yfVjY0RufU93XOdzhOLw?pwd=2026>

解压密码：`2026`

模拟机是绿色软件，数据保存在程序目录旁边，移动时需要整体移动。打开后用真实队号、队员 1 姓名和手机号登录。连接调试请选择“问题 3 演练测试”，不要选择正式测试。

启动演练后等待 5 秒倒计时结束，服务才会在以下地址开放：

```text
http://127.0.0.1:2026
```

测试未开始或已结束时，端口关闭属于正常现象。

## 2. 先运行完全离线的测试

在 PowerShell 中执行：

```powershell
Set-Location 'C:\Users\ASUS\Desktop\math\web'
python -m unittest discover -s tests -v
```

## 3. 无动作连接探测

这条命令只检查 TCP 端口，不调用 `/enter`，不会产生模拟动作：

```powershell
python simulator_client.py probe
```

返回 `"reachable": true` 表示端口已开放。

也可以启动可视化控制台：

```powershell
python web_app.py
```

然后打开 <http://127.0.0.1:8080>。页面每 5 秒进行一次无动作端口探测；只有填写队号并确认“演练测试”后，动作按钮才会真正调用模拟机。网页通过本地 Python 代理连接官方服务，避免浏览器直接访问 `2026` 端口产生跨域问题。

## 4. 演练联调

将下面的 `你的真实队号` 替换为模拟机当前登录的队号。脚本要求显式写出 `--confirm-exercise`，避免误在正式测试中运行：

```powershell
python smoke_test.py --robot-id '你的真实队号' --confirm-exercise
```

这会依次执行 `/enter` 和 `/exit`。若还要在原点、通道 1 做一次测量：

```powershell
python smoke_test.py --robot-id '你的真实队号' --confirm-exercise --measure
```

不要把演练确认参数用于正式测试。正式测试每题仅三次机会。

## 5. 在算法中调用

```python
from simulator_client import Position, SimulatorClient

client = SimulatorClient(robot_id="当前登录的真实队号")
session = client.enter()
print("剩余真实时间：", session["remaining_real_duration_s"])

observation = client.measure(Position(300, 400), channel=1)
print(observation)

clear_result = client.clear(Position(300, 400), channel=1)
print(clear_result)

client.exit()
```

必须逐个等待请求完成，不能并发。客户端内部已加串行锁；网络重试会复用完全相同的 JSON 和 `request_id`。新的动作会生成新的 ID。

## 注意事项

- `robot_id` 必须与模拟机当前登录队号完全相同。
- 通道只能是整数 1–20；坐标必须有限且绝对值不超过 2,000,000。
- 必须同时检查 HTTP 状态和响应中的 `accepted`；客户端已经完成这两层检查。
- `accepted=false` 时响应中的 `virtual_time_s=0`，它不是当前累计虚拟时间。
- `/clear` 不会切换接收通道。
- `/enter` 返回的 `remaining_real_duration_s` 才是本次剩余时间，不应写死为 1200 秒。
- 若真实模拟机行为与文档不一致，保留 `logs/simulator.jsonl` 并联系 `cumcm2026b@163.com`。
