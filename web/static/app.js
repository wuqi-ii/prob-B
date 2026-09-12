const $ = (id) => document.getElementById(id);
const output = $("output");
const buttons = [...document.querySelectorAll("[data-action]")];

function write(kind, value) {
  const stamp = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  output.textContent = `[${stamp}] ${kind}\n${text}\n\n` + (output.textContent === "等待操作…" ? "" : output.textContent);
}

async function refreshStatus() {
  const badge = $("status");
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const data = await response.json();
    badge.className = `status ${data.reachable ? "online" : "offline"}`;
    badge.innerHTML = `<span></span>${data.reachable ? "模拟机已连接" : "模拟机未开放"}`;
  } catch (error) {
    badge.className = "status offline";
    badge.innerHTML = "<span></span>本地代理异常";
  }
}

function payloadFor(action) {
  const payload = {
    action,
    robot_id: $("robotId").value.trim(),
    confirm_exercise: $("exercise").checked,
  };
  if (action === "measure") Object.assign(payload, { x: Number($("mx").value), y: Number($("my").value), channel: Number($("channel").value) });
  if (action === "clear") Object.assign(payload, { x: Number($("cx").value), y: Number($("cy").value), channel: Number($("clearChannel").value) });
  return payload;
}

async function sendAction(action) {
  const payload = payloadFor(action);
  if (!payload.robot_id) { write("未发送", "请填写当前登录队号。"); return; }
  if (!payload.confirm_exercise) { write("未发送", "请先确认当前是演练测试；网页不会在未确认时发送动作。"); return; }
  buttons.forEach((button) => { button.disabled = true; });
  write("请求", payload);
  try {
    const response = await fetch("/api/action", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const data = await response.json();
    write(response.ok ? "响应" : "错误", data);
  } catch (error) {
    write("网络错误", String(error));
  } finally {
    buttons.forEach((button) => { button.disabled = false; });
    refreshStatus();
  }
}

buttons.forEach((button) => button.addEventListener("click", () => sendAction(button.dataset.action)));
$("clearLog").addEventListener("click", () => { output.textContent = "等待操作…"; });
refreshStatus();
setInterval(refreshStatus, 5000);
