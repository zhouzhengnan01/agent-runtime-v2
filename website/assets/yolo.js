const qs = (sel) => document.querySelector(sel);

const baseOrigin = window.location.origin;

const elements = {
  wsStatus: qs("#cvWsStatus"),
  taskHint: qs("#cvTaskHint"),
  resultStat: qs("#cvResultStat"),
  resultJson: qs("#cvResultJson"),
  log: qs("#cvLog"),
  evidenceList: qs("#cvEvidenceList"),
  boxList: qs("#cvBoxList"),
  modelDir: qs("#cvModelDir"),
  extList: qs("#cvExtList"),
  modelFile: qs("#cvModelFile"),
  overwrite: qs("#cvOverwrite"),
  uploadBtn: qs("#cvUploadBtn"),
  uploadOut: qs("#cvUploadOut"),
  refreshModels: qs("#cvRefreshModels"),
  modelTable: qs("#cvModelTable"),
  modelSelect: qs("#cvModelSelect"),
  agentSelect: qs("#cvAgentSelect"),
  refreshAgents: qs("#cvRefreshAgents"),
  nameInput: qs("#cvNameInput"),
  nameDesc: qs("#cvNameDesc"),
  nameBuild: qs("#cvNameBuild"),
  autoName: qs("#cvAutoName"),
  streamUrl: qs("#cvStreamUrl"),
  sourceId: qs("#cvSourceId"),
  interval: qs("#cvInterval"),
  alwaysReturn: qs("#cvAlwaysReturn"),
  startBtn: qs("#cvStartBtn"),
  stopBtn: qs("#cvStopBtn"),
  clearAll: qs("#cvClearAll"),
};

const wsState = {
  socket: null,
  agentId: "",
  sessionId: "",
  initRequestId: "",
  initDone: false,
  initPromise: null,
  initResolve: null,
  initReject: null,
  activeRpcId: "",
  resultCount: 0,
  evidenceSeen: new Set(),
  boxSeen: new Set(),
  lastSource: null,
};

const BUILTIN_AGENT = {
  id: "video_patrol_builtin",
  name: "视频巡检智能体",
  type: "video_patrol",
  template: "video_patrol",
  builtin: true,
};

function setPill(el, text, kind) {
  if (!el) return;
  el.textContent = text;
  el.className = `pill${kind ? ` ${kind}` : ""}`;
}

function setTaskHint(text) {
  if (elements.taskHint) elements.taskHint.textContent = text;
}

function formatBytes(value) {
  const size = Number(value || 0);
  if (!Number.isFinite(size) || size <= 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let idx = 0;
  let num = size;
  while (num >= 1024 && idx < units.length - 1) {
    num /= 1024;
    idx += 1;
  }
  return `${num.toFixed(num >= 10 || idx === 0 ? 0 : 1)} ${units[idx]}`;
}

function formatTime(iso) {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

function resolveUrl(raw) {
  const url = String(raw || "").trim();
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) return `${baseOrigin}${url}`;
  return `${baseOrigin}/${url}`;
}

function escapeHtml(text) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };
  return String(text || "").replace(/[&<>"']/g, (m) => map[m]);
}

async function apiFetch(url, options = {}) {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const contentType = resp.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await resp.json() : await resp.text();
  if (!resp.ok) {
    const detail = typeof payload === "string" ? payload : payload?.detail || payload?.message;
    throw new Error(detail || `请求失败: ${resp.status}`);
  }
  return payload;
}

async function loadModels() {
  try {
    const data = await apiFetch("/api/v1/upload/cv-models", { method: "GET" });
    if (elements.modelDir) elements.modelDir.textContent = data.model_dir || "storage/models/cv";
    if (elements.extList && Array.isArray(data.allowed_extensions)) {
      elements.extList.textContent = data.allowed_extensions.join(", ");
    }
    const models = Array.isArray(data.models) ? data.models : [];
    updateModelSelect(models);
    updateModelTable(models);
    if (elements.uploadOut) elements.uploadOut.textContent = `已加载 ${models.length} 个模型`;
  } catch (err) {
    if (elements.uploadOut) elements.uploadOut.textContent = `加载失败: ${err.message}`;
  }
}

function updateModelSelect(models) {
  if (!elements.modelSelect) return;
  const prev = elements.modelSelect.value;
  elements.modelSelect.innerHTML = "";
  if (!models.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "暂无模型";
    elements.modelSelect.appendChild(opt);
    return;
  }
  models.forEach((model) => {
    const opt = document.createElement("option");
    opt.value = model.name;
    opt.textContent = model.name;
    elements.modelSelect.appendChild(opt);
  });
  if (prev) elements.modelSelect.value = prev;
  if (!elements.modelSelect.value && models[0]) elements.modelSelect.value = models[0].name;
  syncAutoName();
}

function updateModelTable(models) {
  if (!elements.modelTable) return;
  elements.modelTable.innerHTML = "";
  if (!models.length) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="4">暂无模型，请先上传</td>';
    elements.modelTable.appendChild(row);
    return;
  }
  models.forEach((model) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="mono">${escapeHtml(model.name)}</td>
      <td>${formatBytes(model.size)}</td>
      <td>${formatTime(model.modified_at)}</td>
      <td><button class="btn sm ghost" data-model="${escapeHtml(model.name)}">选用</button></td>
    `;
    elements.modelTable.appendChild(row);
  });
}

async function uploadModel() {
  if (!elements.modelFile?.files?.length) {
    if (elements.uploadOut) elements.uploadOut.textContent = "请选择模型文件";
    return;
  }
  const file = elements.modelFile.files[0];
  const form = new FormData();
  form.append("file", file);
  const overwrite = elements.overwrite?.checked ? "true" : "false";
  try {
    if (elements.uploadOut) elements.uploadOut.textContent = "上传中...";
    const resp = await fetch(`/api/v1/upload/cv-model?overwrite=${overwrite}`, {
      method: "POST",
      body: form,
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data?.detail || data?.message || "上传失败");
    }
    if (elements.uploadOut) elements.uploadOut.textContent = `上传完成: ${data.filename}`;
    await loadModels();
  } catch (err) {
    if (elements.uploadOut) elements.uploadOut.textContent = `上传失败: ${err.message}`;
  }
}

function buildName() {
  const model = (elements.modelSelect?.value || "").trim();
  const desc = (elements.nameDesc?.value || "").trim();
  if (!model) return desc || "";
  return `cv:${model}${desc ? ` ${desc}` : ""}`;
}

function syncAutoName() {
  if (!elements.autoName?.checked) return;
  if (!elements.nameInput) return;
  elements.nameInput.value = buildName();
}

function buildSource(useFallback = false) {
  const rawUrl = (elements.streamUrl?.value || "").trim();
  if (!rawUrl) {
    if (useFallback && wsState.lastSource) return wsState.lastSource;
    throw new Error("请填写视频流地址");
  }
  const source = { id: (elements.sourceId?.value || "stream-1").trim() || "stream-1" };
  if (rawUrl.startsWith("rtsp://") || rawUrl.startsWith("rtsps://")) {
    source.rtsp = rawUrl;
  } else if (rawUrl.startsWith("rtmp://") || rawUrl.startsWith("rtmps://")) {
    source.rtmp = rawUrl;
  } else {
    throw new Error("仅支持 rtsp/rtmp 流地址");
  }
  return [source];
}

function buildWsUrl(agentId) {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/api/v1/jaip/session/${encodeURIComponent(agentId)}`;
}

function resetWsState(reason) {
  if (wsState.socket) {
    try {
      wsState.socket.close(1000, reason || "reset");
    } catch {
      // ignore
    }
  }
  wsState.socket = null;
  wsState.agentId = "";
  wsState.sessionId = "";
  wsState.initRequestId = "";
  wsState.initDone = false;
  wsState.initPromise = null;
  wsState.initResolve = null;
  wsState.initReject = null;
}

function sendWs(payload) {
  if (!wsState.socket || wsState.socket.readyState !== WebSocket.OPEN) {
    throw new Error("WebSocket 未连接");
  }
  wsState.socket.send(JSON.stringify(payload));
}

function handleWsMessage(data) {
  if (!data || typeof data !== "object") return;
  if (data.type === "connected") {
    wsState.sessionId = data.session_id || "";
    return;
  }

  if (data.id && data.id === wsState.initRequestId) {
    wsState.initDone = !data.error;
    if (data.error) {
      setPill(elements.wsStatus, "初始化失败", "bad");
      const message = data.error?.message || "初始化失败";
      if (wsState.initReject) wsState.initReject(new Error(message));
    } else {
      wsState.sessionId = data.result?.sessionId || wsState.sessionId;
      setPill(elements.wsStatus, "在线", "ok");
      if (wsState.initResolve) wsState.initResolve(wsState.socket);
    }
    wsState.initResolve = null;
    wsState.initReject = null;
    wsState.initPromise = null;
    return;
  }

  if (data.id && data.id === wsState.activeRpcId) {
    if (data.error) {
      appendLog("error", data.error?.message || "任务失败");
      return;
    }
    if (data.result === true) {
      appendLog("info", "任务已停止");
      setTaskHint("已停止");
      return;
    }
    if (data.result) {
      handleCvResult(data.result);
    }
  }
}

async function connectWs(agentId) {
  if (!agentId) throw new Error("请选择智能体");
  if (wsState.socket && wsState.agentId === agentId && wsState.initDone) {
    return wsState.socket;
  }
  if (wsState.initPromise && wsState.agentId === agentId) {
    return wsState.initPromise;
  }
  resetWsState("reconnect");
  wsState.agentId = agentId;

  wsState.initPromise = new Promise((resolve, reject) => {
    wsState.initResolve = resolve;
    wsState.initReject = reject;
  });

  const socket = new WebSocket(buildWsUrl(agentId));
  wsState.socket = socket;
  setPill(elements.wsStatus, "连接中", "warn");

  socket.onopen = () => {
    wsState.initRequestId = `cv_init_${Date.now()}`;
    sendWs({
      jsonrpc: "2.0",
      id: wsState.initRequestId,
      method: "session.initialize",
      params: {
        protocolVersion: "2.0",
        userContext: { userId: "cv-console" },
      },
    });
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleWsMessage(data);
    } catch (err) {
      appendLog("warn", `收到无法解析的消息: ${err.message}`);
    }
  };

  socket.onerror = () => {
    setPill(elements.wsStatus, "连接失败", "bad");
    if (wsState.initReject) wsState.initReject(new Error("WebSocket 连接失败"));
    wsState.initPromise = null;
  };

  socket.onclose = () => {
    setPill(elements.wsStatus, "已断开", "warn");
    wsState.initPromise = null;
  };

  return wsState.initPromise;
}

function appendLog(level, message, meta = {}) {
  if (!elements.log) return;
  const entry = document.createElement("div");
  entry.className = `patrol-log-entry${level === "error" ? " error" : level === "warn" ? " warn" : " cmd"}`;
  const metaEl = document.createElement("div");
  metaEl.className = "patrol-log-meta";
  const time = new Date().toLocaleTimeString();
  metaEl.innerHTML = `<span>${escapeHtml(time)}</span><span>${escapeHtml(meta.detail || "")}</span>`;
  const body = document.createElement("div");
  body.className = "patrol-log-text";
  body.textContent = message;
  entry.append(metaEl, body);
  elements.log.prepend(entry);
}

function renderAgentOptions(list, note) {
  if (!elements.agentSelect) return;
  const prev = elements.agentSelect.value;
  const agents = Array.isArray(list) && list.length ? list : [BUILTIN_AGENT];
  elements.agentSelect.innerHTML = "";
  agents.forEach((agent) => {
    const opt = document.createElement("option");
    opt.value = agent.id;
    const label = agent.builtin
      ? `${agent.name || agent.id} (内置)`
      : `${agent.name || agent.id} (${agent.id})`;
    opt.textContent = label;
    elements.agentSelect.appendChild(opt);
  });
  if (prev && agents.some((agent) => agent.id === prev)) {
    elements.agentSelect.value = prev;
  }
  if (!elements.agentSelect.value && agents[0]) {
    elements.agentSelect.value = agents[0].id;
  }
  if (note) appendLog("warn", note);
}

function addMedia(listEl, url, label, seenSet) {
  if (!listEl) return;
  const resolved = resolveUrl(url);
  if (!resolved || seenSet.has(resolved)) return;
  seenSet.add(resolved);
  const card = document.createElement("div");
  card.className = "patrol-media-card";
  const meta = document.createElement("div");
  meta.className = "patrol-media-meta";
  meta.innerHTML = `<span>${escapeHtml(label)}</span><a href="${escapeHtml(resolved)}" target="_blank" rel="noreferrer">打开</a>`;
  const img = document.createElement("img");
  img.loading = "lazy";
  img.alt = label;
  img.src = resolved;
  card.append(meta, img);
  listEl.prepend(card);
}

function countObjects(fullText) {
  if (!Array.isArray(fullText)) return 0;
  let total = 0;
  fullText.forEach((item) => {
    if (!item) return;
    if (Array.isArray(item.objects)) total += item.objects.length;
    else if (item.label && Array.isArray(item.box)) total += 1;
  });
  return total;
}

function handleCvResult(result) {
  wsState.resultCount += 1;
  if (elements.resultStat) elements.resultStat.textContent = `${wsState.resultCount} 次结果`;

  const sourceId = result.sourceId || "-";
  const fullText = result.full_text || [];
  const evidence = Array.isArray(result.evidence_image_urls) ? result.evidence_image_urls : [];
  const boxes = Array.isArray(result.evidence_image_box_urls) ? result.evidence_image_box_urls : [];

  const objCount = countObjects(fullText);
  appendLog("info", `source=${sourceId} 检测目标 ${objCount} 个，证据帧 ${evidence.length} 张`, {
    detail: `#${wsState.resultCount}`,
  });

  if (elements.resultJson) {
    elements.resultJson.textContent = JSON.stringify(fullText, null, 2) || "[]";
  }

  evidence.forEach((url) => addMedia(elements.evidenceList, url, `source ${sourceId}`, wsState.evidenceSeen));
  boxes.forEach((url) => addMedia(elements.boxList, url, `source ${sourceId}`, wsState.boxSeen));
}

function clearResults() {
  wsState.resultCount = 0;
  wsState.evidenceSeen.clear();
  wsState.boxSeen.clear();
  if (elements.resultStat) elements.resultStat.textContent = "0 次结果";
  if (elements.resultJson) elements.resultJson.textContent = "等待结果...";
  if (elements.log) elements.log.innerHTML = "";
  if (elements.evidenceList) elements.evidenceList.innerHTML = "";
  if (elements.boxList) elements.boxList.innerHTML = "";
}

async function startTask() {
  try {
    const agentId = ((elements.agentSelect?.value || "") || BUILTIN_AGENT.id).trim();
    const name = (elements.nameInput?.value || "").trim();
    if (!name) throw new Error("请填写任务名称");
    const intervalRaw = parseInt(elements.interval?.value || "10", 10);
    const interval = Number.isFinite(intervalRaw) ? Math.max(intervalRaw, 10) : 10;
    if (elements.interval) elements.interval.value = String(interval);
    const sources = buildSource();
    await connectWs(agentId);

    clearResults();
    wsState.activeRpcId = `cv_${Date.now()}`;
    wsState.lastSource = sources;
    const params = {
      state: "start",
      interval,
      name,
      backend: "cv",
      alwaysReturn: Boolean(elements.alwaysReturn?.checked),
      source: sources,
    };
    sendWs({ jsonrpc: "2.0", id: wsState.activeRpcId, method: "ComputerVisionTask", params });
    setTaskHint(`已发送启动指令: ${wsState.activeRpcId}`);
  } catch (err) {
    setTaskHint(`启动失败: ${err.message}`);
    appendLog("error", err.message || "启动失败");
  }
}

async function stopTask() {
  try {
    if (!wsState.activeRpcId) throw new Error("当前没有运行中的任务");
    const sources = buildSource(true);
    const params = {
      state: "stop",
      source: sources,
    };
    sendWs({ jsonrpc: "2.0", id: wsState.activeRpcId, method: "ComputerVisionTask", params });
    setTaskHint("已发送停止指令");
  } catch (err) {
    setTaskHint(`停止失败: ${err.message}`);
    appendLog("error", err.message || "停止失败");
  }
}

async function loadAgents() {
  try {
    const data = await apiFetch("/api/v1/agent/list", {
      method: "POST",
      body: JSON.stringify({ pageIndex: 0, pageSize: 200 }),
    });
    const list = data?.result?.data || [];
    const patrolAgents = list.filter(
      (agent) => (agent?.type || "").includes("video_patrol") || (agent?.template || "") === "video_patrol"
    );
    if (!patrolAgents.length) {
      renderAgentOptions([], `未找到巡检智能体（总数 ${list.length}），已使用内置巡检智能体`);
      return;
    }
    renderAgentOptions(patrolAgents);
  } catch (err) {
    renderAgentOptions([], `加载智能体失败: ${err.message}，已使用内置巡检智能体`);
  }
}

function bindEvents() {
  elements.refreshModels?.addEventListener("click", loadModels);
  elements.uploadBtn?.addEventListener("click", uploadModel);
  elements.refreshAgents?.addEventListener("click", loadAgents);
  elements.modelTable?.addEventListener("click", (event) => {
    const btn = event.target.closest("button[data-model]");
    if (!btn) return;
    const model = btn.getAttribute("data-model");
    if (model && elements.modelSelect) {
      elements.modelSelect.value = model;
      syncAutoName();
    }
  });
  elements.nameBuild?.addEventListener("click", () => {
    if (elements.nameInput) elements.nameInput.value = buildName();
  });
  elements.modelSelect?.addEventListener("change", syncAutoName);
  elements.nameDesc?.addEventListener("input", syncAutoName);
  elements.autoName?.addEventListener("change", syncAutoName);
  elements.startBtn?.addEventListener("click", startTask);
  elements.stopBtn?.addEventListener("click", stopTask);
  elements.clearAll?.addEventListener("click", clearResults);
}

async function init() {
  setPill(elements.wsStatus, "未连接", "warn");
  if (elements.interval && !elements.interval.value) elements.interval.value = "10";
  bindEvents();
  await loadModels();
  await loadAgents();
  syncAutoName();
}

init();
