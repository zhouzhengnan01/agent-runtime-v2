const DEFAULT_BASE_URL = `${location.origin}`;

function qs(selector, el = document) {
  return el.querySelector(selector);
}

function qsa(selector, el = document) {
  return Array.from(el.querySelectorAll(selector));
}

function escapeHtml(input) {
  return String(input)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function nowTime() {
  return new Date().toLocaleString();
}

const WS_ERROR_MAP = {
  "error.jaip_request_fail": "JAIP 请求失败（请检查智能体/工具/鉴权/网络）",
};

function normalizeWsError(raw) {
  const msg = String(raw || "").trim();
  if (!msg) return "";
  return WS_ERROR_MAP[msg] || msg;
}

function readJsonSafely(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function normalizeBaseUrl(url) {
  const trimmed = (url || "").trim();
  if (!trimmed) return DEFAULT_BASE_URL;
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) return trimmed.replace(/\/+$/, "");
  return `${location.protocol}//${trimmed}`.replace(/\/+$/, "");
}

function resolveAssetUrl(path) {
  if (!path) return "";
  try {
    return new URL(path, state?.baseUrl || DEFAULT_BASE_URL).toString();
  } catch {
    return path;
  }
}

function buildWsBaseUrl(baseUrl) {
  try {
    const url = new URL(baseUrl || DEFAULT_BASE_URL, location.origin);
    const protocol = url.protocol === "https:" ? "wss:" : "ws:";
    const path = url.pathname.replace(/\/+$/, "");
    return `${protocol}//${url.host}${path}`;
  } catch {
    const trimmed = (baseUrl || "").trim().replace(/\/+$/, "");
    if (trimmed.startsWith("wss://") || trimmed.startsWith("ws://")) return trimmed;
    if (trimmed.startsWith("https://")) return `wss://${trimmed.slice(8)}`;
    if (trimmed.startsWith("http://")) return `ws://${trimmed.slice(7)}`;
    return `${location.protocol === "https:" ? "wss" : "ws"}://${trimmed}`;
  }
}

function buildChatWsUrl(agentId) {
  const base = buildWsBaseUrl(state.baseUrl);
  return `${base}/api/v1/jaip/session/${encodeURIComponent(agentId)}`;
}

function rpcId(prefix = "req") {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return `${prefix}_${crypto.randomUUID().slice(0, 8)}`;
  }
  return `${prefix}_${Math.random().toString(16).slice(2, 10)}`;
}

const chatWsState = {
  socket: null,
  agentId: "",
  sessionId: "",
  initRequestId: "",
  initDone: false,
  initError: "",
  connectPromise: null,
  streams: new Map(),
};

function resetChatWs(reason = "") {
  if (chatWsState.socket && chatWsState.socket.readyState !== WebSocket.CLOSED) {
    try {
      chatWsState.socket.close(1000, reason || "reset");
    } catch {
      // ignore
    }
  }
  chatWsState.socket = null;
  chatWsState.agentId = "";
  chatWsState.sessionId = "";
  chatWsState.initRequestId = "";
  chatWsState.initDone = false;
  chatWsState.initError = "";
  chatWsState.connectPromise = null;
  chatWsState.streams = new Map();
}

const patrolWsState = {
  socket: null,
  agentId: "",
  sessionId: "",
  initRequestId: "",
  initDone: false,
  initError: "",
  connectPromise: null,
  streams: new Map(),
};

function resetPatrolWs(reason = "") {
  if (patrolWsState.socket && patrolWsState.socket.readyState !== WebSocket.CLOSED) {
    try {
      patrolWsState.socket.close(1000, reason || "reset");
    } catch {
      // ignore
    }
  }
  patrolWsState.socket = null;
  patrolWsState.agentId = "";
  patrolWsState.sessionId = "";
  patrolWsState.initRequestId = "";
  patrolWsState.initDone = false;
  patrolWsState.initError = "";
  patrolWsState.connectPromise = null;
  patrolWsState.streams = new Map();
}

const store = {
  get(key, fallback) {
    const raw = localStorage.getItem(key);
    if (raw === null || raw === undefined || raw === "") return fallback;
    const parsed = readJsonSafely(raw);
    return parsed ?? raw;
  },
  set(key, value) {
    localStorage.setItem(key, typeof value === "string" ? value : JSON.stringify(value));
  },
};

function generateStreamId() {
  return `stream_${Date.now().toString(36)}_${Math.random().toString(16).slice(2, 6)}`;
}

function toOptionalNumber(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return null;
  return num;
}

function toPositiveInt(value, fallback) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return fallback;
  return Math.floor(num);
}

function toNonNegativeInt(value, fallback) {
  const num = Number(value);
  if (!Number.isFinite(num) || num < 0) return fallback;
  return Math.floor(num);
}

function normalizePatrolStreams(list) {
  if (!Array.isArray(list)) return [];
  const seen = new Set();
  const out = [];
  list.forEach((item) => {
    if (!item || typeof item !== "object") return;
    const url = String(item.url || item.stream_url || "").trim();
    if (!url) return;
    let id = String(item.id || "").trim();
    if (!id || seen.has(id)) id = generateStreamId();
    seen.add(id);
    const reachable = typeof item.reachable === "boolean" ? item.reachable : undefined;
    out.push({
      id,
      name: String(item.name || "未命名").trim() || "未命名",
      url,
      interval: toOptionalNumber(item.interval),
      stopMinutes: toOptionalNumber(item.stopMinutes),
      prompt: String(item.prompt || "").trim(),
      createdAt: String(item.createdAt || new Date().toISOString()),
      reachable,
    });
  });
  return out;
}

function getPatrolStreamUrl(item) {
  if (!item || typeof item !== "object") return "";
  const url = item.url || item.stream_url || item.streamUrl || "";
  return String(url || "").trim();
}

function mergePatrolStreams(primary, incoming) {
  const out = [];
  const indexByUrl = new Map();

  const addItem = (item) => {
    if (!item || typeof item !== "object") return;
    const url = getPatrolStreamUrl(item);
    if (!url) return;
    const key = url;
    const idx = indexByUrl.get(key);
    if (idx === undefined) {
      out.push({ ...item, url });
      indexByUrl.set(key, out.length - 1);
      return;
    }
    const base = out[idx];
    if (!base.id && item.id) base.id = item.id;
    if (!base.name || base.name === "未命名") base.name = item.name;
    if (!base.prompt && item.prompt) base.prompt = item.prompt;
    if (base.interval == null && item.interval != null) base.interval = item.interval;
    if (base.stopMinutes == null && item.stopMinutes != null) base.stopMinutes = item.stopMinutes;
    if (!base.createdAt && item.createdAt) base.createdAt = item.createdAt;
    if (typeof base.reachable !== "boolean" && typeof item.reachable === "boolean") {
      base.reachable = item.reachable;
    }
  };

  if (Array.isArray(primary)) primary.forEach(addItem);
  if (Array.isArray(incoming)) incoming.forEach(addItem);
  return out;
}

const bootstrapPatrolStreams = Array.isArray(window.__PATROL_STREAMS__) ? window.__PATROL_STREAMS__ : [];
const storedPatrolStreams = store.get("jl_patrolStreams", []);
const mergedPatrolStreams = mergePatrolStreams(storedPatrolStreams, bootstrapPatrolStreams);
const normalizedPatrolStreams = normalizePatrolStreams(mergedPatrolStreams);

if (bootstrapPatrolStreams.length) {
  const storedList = Array.isArray(storedPatrolStreams) ? storedPatrolStreams : [];
  if (normalizedPatrolStreams.length && normalizedPatrolStreams.length !== storedList.length) {
    store.set("jl_patrolStreams", normalizedPatrolStreams);
  }
}

function isLiveStream(url) {
  const normalized = (url || "").trim().toLowerCase();
  return normalized.startsWith("rtsp://")
    || normalized.startsWith("rtsps://")
    || normalized.startsWith("rtmp://")
    || normalized.startsWith("rtmps://");
}

function isHttpUrl(url) {
  const normalized = (url || "").trim().toLowerCase();
  return normalized.startsWith("http://") || normalized.startsWith("https://");
}

function isHlsManifest(url) {
  const normalized = (url || "").trim().toLowerCase();
  return normalized.includes(".m3u8");
}

function deriveStreamName(url) {
  const value = (url || "").trim();
  if (!value) return "未命名";
  try {
    const parsed = new URL(value);
    return parsed.hostname || parsed.pathname || value;
  } catch {
    return value.replace(/^.*?:\/\//, "").slice(0, 32) || "未命名";
  }
}

const state = {
  baseUrl: normalizeBaseUrl(store.get("jl_baseUrl", DEFAULT_BASE_URL)),
  authHeaderName: store.get("jl_authHeaderName", ""),
  authHeaderValue: store.get("jl_authHeaderValue", ""),
  agents: [],
  selectedAgentId: store.get("jl_agentId", ""),
  agentTemplates: [],
  agentTemplateFilter: store.get("jl_agentTemplateFilter", ""),
  agentListSearch: store.get("jl_agentListSearch", ""),
  agentListTemplate: store.get("jl_agentListTemplate", ""),
  agentListStatus: store.get("jl_agentListStatus", ""),
  agentMetaCache: {},
  patrolStreams: normalizedPatrolStreams,
  patrolSelectedStreamId: store.get("jl_patrolStreamId", ""),
  patrolSelectedAgentId: store.get("jl_patrolAgentId", ""),
  patrolReviewAgentId: store.get("jl_patrolReviewAgentId", ""),
  patrolSearch: store.get("jl_patrolSearch", ""),
  patrolPageIndex: toNonNegativeInt(store.get("jl_patrolPageIndex", 0), 0),
  patrolPageSize: toPositiveInt(store.get("jl_patrolPageSize", 8), 8),
  patrolMode: store.get("jl_patrolMode", "ws"),
  agentsLoaded: false,
  tools: [],
  toolsLoaded: false,
  templates: null,
  health: {
    status: "idle",
    latencyMs: null,
    checkedAt: "",
  },
};

const patrolRuntime = {
  runningStreamId: "",
  streamCheckInFlight: false,
  streamCheckAt: 0,
  streamCheckHidden: 0,
  mediaItems: [],
  mediaSeen: new Set(),
  mediaBuffer: "",
  stats: {
    total: 0,
    low: 0,
    mid: 0,
    high: 0,
    media: 0,
  },
  pendingAction: "",
  pendingStreamId: "",
  previewStreamId: "",
  previewHlsStreamId: "",
  previewSourceType: "",
  previewUrl: "",
  hls: null,
  previewRetryCount: 0,
  previewRetryTimer: null,
  reviewRunning: false,
  reviewInFlight: false,
  reviewStopRequested: false,
  reviewTimer: null,
  reviewAbortController: null,
  reviewLastRunAt: 0,
};

const PATROL_PAGE_SIZES = [6, 8, 12, 20];
const PATROL_STREAM_CHECK_TTL_MS = 60 * 1000;
const PATROL_STREAM_CHECK_TIMEOUT_MS = 800;
const REVIEW_DEFAULT_INTERVAL_SECONDS = 10;
const REVIEW_CAPTURE_SECONDS = 10;
const REVIEW_SAMPLE_FPS = 2;

function getAgentId(agent) {
  if (!agent || typeof agent !== "object") return "";
  return agent.id || agent.agent_id || "";
}

function getAgentTemplateId(agent) {
  if (!agent || typeof agent !== "object") return "";
  const direct = agent.template_id || agent.templateId || agent.type || agent.template || "";
  if (direct) return direct;
  const cfgRaw = agent.config;
  if (!cfgRaw) return "";
  if (typeof cfgRaw === "object") return cfgRaw.template || cfgRaw.type || "";
  if (typeof cfgRaw === "string") {
    const parsed = readJsonSafely(cfgRaw);
    if (parsed) return parsed.template || parsed.type || "";
  }
  return "";
}

function getAgentModel(agent) {
  if (!agent || typeof agent !== "object") return "";
  if (agent.model) return agent.model;
  const cfgRaw = agent.config;
  if (!cfgRaw) return "";
  if (typeof cfgRaw === "object") return cfgRaw.model || "";
  if (typeof cfgRaw === "string") {
    const parsed = readJsonSafely(cfgRaw);
    if (parsed) return parsed.model || "";
  }
  return "";
}

function getAgentStatusInfo(agent) {
  const raw = (agent?.status || "").toString().trim().toLowerCase();
  if (raw === "active") return { value: "active", label: "启用", badge: "ok" };
  if (raw === "inactive" || raw === "disabled") return { value: raw, label: "停用", badge: "bad" };
  if (raw === "draft" || raw === "archived") return { value: raw, label: "草稿", badge: "warn" };
  if (raw) return { value: raw, label: raw, badge: "" };
  return { value: "unknown", label: "未知", badge: "" };
}

function buildAgentOptionsFromList(list, selectedId, emptyLabel = "（请先加载智能体）") {
  if (!list || !list.length) return `<option value="">${escapeHtml(emptyLabel)}</option>`;
  return list
    .map((a) => {
      const id = getAgentId(a);
      if (!id) return "";
      const name = a.name || id;
      const selected = id === selectedId ? "selected" : "";
      return `<option value="${escapeHtml(id)}" ${selected}>${escapeHtml(name)} (${escapeHtml(id)})</option>`;
    })
    .join("");
}

function buildAgentOptions(selectedId, emptyLabel = "（请先加载智能体）") {
  return buildAgentOptionsFromList(state.agents || [], selectedId, emptyLabel);
}

function getTemplateId(template) {
  if (!template || typeof template !== "object") return "";
  return template.id || template.type || template.template_id || "";
}

function buildTemplateOptions(list, selectedId) {
  const options = (list || [])
    .map((t) => {
      const id = getTemplateId(t);
      if (!id) return "";
      const name = t.name || id;
      const selected = id === selectedId ? "selected" : "";
      return `<option value="${escapeHtml(id)}" ${selected}>${escapeHtml(name)} (${escapeHtml(id)})</option>`;
    })
    .join("");
  return `<option value="">全部模板</option>${options}`;
}

function buildCreateTemplateOptions(list, selectedId) {
  const options = (list || [])
    .map((t) => {
      const id = getTemplateId(t);
      if (!id) return "";
      const name = t.name || id;
      const selected = id === selectedId ? "selected" : "";
      return `<option value="${escapeHtml(id)}" ${selected}>${escapeHtml(name)} (${escapeHtml(id)})</option>`;
    })
    .join("");
  return `<option value="">请选择模板</option>${options}`;
}

function filterAgentsByTemplate(list, templateId) {
  if (!templateId) return list || [];
  return (list || []).filter((agent) => getAgentTemplateId(agent) === templateId);
}

function isPatrolTemplateId(value) {
  const id = String(value || "").toLowerCase();
  return id.includes("patrol");
}

function isReviewTemplateId(value) {
  const id = String(value || "").toLowerCase();
  if (!id) return false;
  if (id.includes("patrol")) return false;
  return id.includes("inspection") || id.includes("review");
}

function getPatrolAgents(list = state.agents) {
  return (list || []).filter((agent) => {
    const templateId = getAgentTemplateId(agent) || agent.type || "";
    return isPatrolTemplateId(templateId);
  });
}

function getReviewAgents(list = state.agents) {
  return (list || []).filter((agent) => {
    const templateId = getAgentTemplateId(agent) || agent.type || "";
    return isReviewTemplateId(templateId);
  });
}

function getCachedAgentMeta(agentId) {
  const id = (agentId || "").trim();
  if (!id) return null;
  return state.agentMetaCache?.[id] || null;
}

function extractAgentPromptFromConfig(config) {
  if (!config || typeof config !== "object") return "";
  return String(
    config.custom_prompt
      || config.user_prompt
      || config.system_prompt
      || config.prompt
      || config.description
      || ""
  ).trim();
}

function extractAgentPrompt(agent) {
  if (!agent || typeof agent !== "object") return "";
  const direct = String(
    agent.custom_prompt
      || agent.user_prompt
      || agent.system_prompt
      || agent.prompt
      || agent.description
      || ""
  ).trim();
  if (direct) return direct;
  const cfgRaw = agent.config;
  if (!cfgRaw) return "";
  const cfg = typeof cfgRaw === "string" ? readJsonSafely(cfgRaw) : cfgRaw;
  return extractAgentPromptFromConfig(cfg);
}

function summarizeJsonSchema(schema) {
  if (!schema || typeof schema !== "object") return { required: [], properties: [] };
  const props = schema.properties && typeof schema.properties === "object"
    ? Object.keys(schema.properties)
    : [];
  const required = Array.isArray(schema.required)
    ? schema.required.map((item) => String(item))
    : [];
  return { required, properties: props };
}

function formatSchemaHint(schema) {
  const summary = summarizeJsonSchema(schema);
  const limit = 12;
  const clipList = (list) => {
    if (!list || !list.length) return "";
    if (list.length <= limit) return list.join(", ");
    return `${list.slice(0, limit).join(", ")}...`;
  };
  if (summary.required.length) return `字段要求：${clipList(summary.required)}`;
  if (summary.properties.length) return `字段建议：${clipList(summary.properties)}`;
  return "字段要求：—";
}

async function ensureAgentMeta(agentId, options = {}) {
  const id = (agentId || "").trim();
  if (!id) return null;
  const cached = getCachedAgentMeta(id);
  if (cached && !options.force) return cached;

  let detail = null;
  let jsonInfo = null;
  try {
    detail = await apiFetchStandard(`/api/v1/agent/${encodeURIComponent(id)}`);
  } catch {
    detail = null;
  }
  try {
    jsonInfo = await apiFetch(`/api/v1/agents/${encodeURIComponent(id)}/json/info`);
  } catch {
    jsonInfo = null;
  }

  const prompt = extractAgentPrompt(detail);
  const jsonSchema = jsonInfo?.json_schema || detail?.json_schema || null;
  const outputFormat = jsonInfo?.output_format || detail?.output_format || "";
  const meta = {
    agentId: id,
    prompt,
    jsonSchema,
    outputFormat,
    detail,
    jsonInfo,
    fetchedAt: Date.now(),
  };
  state.agentMetaCache[id] = meta;
  return meta;
}

function resolvePromptInfo(stream, agentId) {
  if (!stream) return { text: "", source: "default" };
  const streamPrompt = String(stream?.prompt || "").trim();
  if (streamPrompt) return { text: streamPrompt, source: "stream" };
  const cached = getCachedAgentMeta(agentId);
  const agentPrompt = String(cached?.prompt || "").trim();
  if (agentPrompt) return { text: agentPrompt, source: "agent" };
  const name = String(stream?.name || "巡检流").trim();
  return { text: `请分析${name}当前画面，输出简要结论。`, source: "default" };
}

function buildPromptMetaText(source, schema) {
  const sourceLabel = source === "stream"
    ? "来源：流配置"
    : source === "agent"
      ? "来源：智能体默认"
      : "来源：默认";
  const schemaHint = formatSchemaHint(schema);
  return `${sourceLabel} · ${schemaHint}`;
}

function buildPatrolAgentOptions(selectedId) {
  const agents = getPatrolAgents();
  return buildAgentOptionsFromList(agents, selectedId, "（暂无巡检智能体）");
}

function normalizePatrolPageSize(value) {
  const fallback = PATROL_PAGE_SIZES[0];
  const size = toPositiveInt(value, fallback);
  return PATROL_PAGE_SIZES.includes(size) ? size : fallback;
}

function buildPatrolPageSizeOptions(selectedSize) {
  return PATROL_PAGE_SIZES.map((size) => {
    const selected = size === selectedSize ? " selected" : "";
    return `<option value="${size}"${selected}>每页 ${size}</option>`;
  }).join("");
}

function isStreamReachable(stream) {
  return !stream || stream.reachable !== false;
}

function getReachablePatrolStreams(streams) {
  return (streams || []).filter(isStreamReachable);
}

function buildPatrolStreamOptions(selectedId) {
  const streams = getReachablePatrolStreams(state.patrolStreams);
  if (!streams.length) return '<option value="">（暂无巡检流）</option>';
  return streams.map((stream) => {
    const id = escapeHtml(stream.id || "");
    const name = escapeHtml(stream.name || deriveStreamName(stream.url || ""));
    const selected = stream.id === selectedId ? " selected" : "";
    return `<option value="${id}"${selected}>${name}</option>`;
  }).join("");
}

function filterPatrolStreams(streams, search = "") {
  const reachable = getReachablePatrolStreams(streams);
  const keyword = (search || "").trim().toLowerCase();
  if (!keyword) return reachable;
  return reachable.filter((stream) => {
    const hay = `${stream.name || ""} ${stream.url || ""}`.toLowerCase();
    return hay.includes(keyword);
  });
}

function buildPatrolStreamCards(streams, selectedId, runningId, previewId, search = "") {
  const filtered = filterPatrolStreams(streams, search);
  if (!filtered.length) return "";

  return filtered.map((stream) => {
    const isSelected = stream.id === selectedId;
    const isRunning = stream.id === runningId;
    const isPreviewing = stream.id === previewId;
    const live = isLiveStream(stream.url);
    const name = escapeHtml(stream.name || "未命名");
    const url = escapeHtml(stream.url || "");
    const interval = stream.interval ? `${stream.interval}s` : "";
    const stop = stream.stopMinutes ? `${stream.stopMinutes}m` : "";
    const createdAt = escapeHtml(stream.createdAt || "");
    const statusTags = [];
    if (isRunning) statusTags.push('<span class="pill ok">巡检中</span>');
    if (isPreviewing) statusTags.push('<span class="pill warn">预览中</span>');
    if (!statusTags.length) statusTags.push('<span class="pill">空闲</span>');
    const statusTag = statusTags.join("");
    const typeTag = live ? '<span class="tag accent">LIVE</span>' : '<span class="tag ghost">OFFLINE</span>';
    const intervalTag = interval ? `<span class="tag">间隔 ${interval}</span>` : "";
    const stopTag = stop ? `<span class="tag ghost">停止 ${stop}</span>` : "";
    const promptTag = stream.prompt ? '<span class="tag primary">有提示词</span>' : "";

    return `
      <article class="patrol-stream-card${isSelected ? " selected" : ""}${isRunning ? " running" : ""}" data-stream-id="${escapeHtml(stream.id)}">
        <div class="patrol-stream-main">
          <div class="patrol-stream-title-row">
            <div class="patrol-stream-title">
              <span class="patrol-stream-dot" aria-hidden="true"></span>
              <span class="patrol-stream-name">${name}</span>
            </div>
            <div class="patrol-stream-status">${statusTag}</div>
          </div>
          <div class="patrol-stream-url mono">${url || "-"}</div>
        </div>
        <div class="patrol-stream-meta">
          <div class="patrol-stream-badges">
            ${typeTag}
            ${intervalTag}
            ${stopTag}
            ${promptTag}
          </div>
          ${createdAt ? `<div class="patrol-stream-time mono">${createdAt}</div>` : ""}
        </div>
        <div class="patrol-stream-actions">
          <button class="btn sm ghost" data-action="console" data-stream-id="${escapeHtml(stream.id)}">控制台</button>
          <button class="btn sm" data-action="preview" data-stream-id="${escapeHtml(stream.id)}">预览</button>
          <button class="btn sm primary" data-action="start" data-stream-id="${escapeHtml(stream.id)}">启动</button>
          <button class="btn sm" data-action="stop" data-stream-id="${escapeHtml(stream.id)}">停止</button>
          <button class="btn sm danger" data-action="remove" data-stream-id="${escapeHtml(stream.id)}">删除</button>
        </div>
      </article>`;
  }).join("");
}

function buildTemplatesFromAgents(list) {
  const map = new Map();
  (list || []).forEach((agent) => {
    const templateId = getAgentTemplateId(agent);
    if (!templateId || map.has(templateId)) return;
    map.set(templateId, { id: templateId, type: templateId, name: templateId });
  });
  return Array.from(map.values());
}

function mergeTemplateLists(primary, fallback) {
  const map = new Map();
  (primary || []).forEach((tpl) => {
    const id = getTemplateId(tpl);
    if (!id || map.has(id)) return;
    map.set(id, tpl);
  });
  (fallback || []).forEach((tpl) => {
    const id = getTemplateId(tpl) || tpl.id || tpl.type || tpl.template_id || "";
    if (!id || map.has(id)) return;
    map.set(id, { id, type: id, name: tpl.name || id });
  });
  return Array.from(map.values());
}

function buildAgentsPreview(list) {
  if (!list || !list.length) {
    return `
      <div class="empty-state">
        <div>暂无智能体数据</div>
        <button class="btn sm" data-action="load-agents" type="button">加载智能体</button>
      </div>
    `;
  }
  return list.slice(0, 6).map((a) => {
    const id = escapeHtml(a.id || a.agent_id || "");
    const name = escapeHtml(a.name || "(未命名)");
    const type = escapeHtml(a.type || a.template_id || a.category || "-");
    const active = id && state.selectedAgentId === id;
    return `
      <div class="dash-item">
        <div>
          <div class="dash-title">${name}${active ? ' <span class="pill ok">当前</span>' : ""}</div>
          <div class="dash-meta mono">${id}</div>
          <div class="dash-meta">${type}</div>
        </div>
        <div class="row">
          <button class="btn sm" data-dash-select-agent="${id}" type="button">设为默认</button>
          <button class="btn sm primary" data-dash-chat-agent="${id}" type="button">对话</button>
        </div>
      </div>
    `;
  }).join("");
}

function toast(message, detail = "") {
  const el = qs("#toast");
  qs("#toastMsg").textContent = message;
  qs("#toastDetail").textContent = detail;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 3500);
}

async function apiFetch(path, options = {}) {
  const url = `${state.baseUrl}${path}`;
  const headers = new Headers(options.headers || {});
  if (!headers.has("Content-Type") && options.body) headers.set("Content-Type", "application/json");
  if (state.authHeaderName && state.authHeaderValue && !headers.has(state.authHeaderName)) {
    headers.set(state.authHeaderName, state.authHeaderValue);
  }

  const resp = await fetch(url, {
    ...options,
    headers,
  });

  const contentType = resp.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const payload = isJson ? await resp.json().catch(() => null) : await resp.text().catch(() => "");

  if (!resp.ok) {
    const detail = typeof payload === "string" ? payload : JSON.stringify(payload);
    throw new Error(`HTTP ${resp.status} ${resp.statusText}: ${detail}`);
  }
  return payload;
}

async function apiFetchStandard(path, options = {}) {
  const resp = await apiFetch(path, options);
  if (resp && typeof resp === "object" && "status" in resp && "result" in resp) return resp.result;
  return resp;
}

async function requestHlsStart(url, streamId, options = {}) {
  return apiFetch("/api/v1/hls/start", {
    method: "POST",
    body: JSON.stringify({
      url,
      stream_id: streamId || undefined,
      ...options,
    }),
  });
}

async function requestHlsStop(streamId, url, cleanup = false) {
  return apiFetch("/api/v1/hls/stop", {
    method: "POST",
    body: JSON.stringify({
      stream_id: streamId || undefined,
      url: url || undefined,
      cleanup,
    }),
  });
}

async function requestPatrolStreamCheck(urls, options = {}) {
  return apiFetchStandard("/api/v1/patrol-streams/check", {
    method: "POST",
    body: JSON.stringify({
      urls,
      timeout_ms: options.timeoutMs,
    }),
  });
}

async function requestPatrolClip(url, options = {}) {
  return apiFetchStandard("/api/v1/clip", {
    method: "POST",
    body: JSON.stringify({
      url,
      clip_seconds: options.clipSeconds,
      stream_id: options.streamId,
      prefer_hls: options.preferHls !== false,
      force_hls: options.forceHls === true,
    }),
  });
}

function destroyPatrolHls() {
  if (patrolRuntime.hls) {
    patrolRuntime.hls.destroy();
    patrolRuntime.hls = null;
  }
}

async function cleanupPatrolPreview(stopRemote = true) {
  const streamId = patrolRuntime.previewHlsStreamId || patrolRuntime.previewStreamId;
  const sourceType = patrolRuntime.previewSourceType;
  destroyPatrolHls();
  if (patrolRuntime.previewRetryTimer) {
    clearTimeout(patrolRuntime.previewRetryTimer);
    patrolRuntime.previewRetryTimer = null;
  }
  patrolRuntime.previewRetryCount = 0;
  patrolRuntime.previewStreamId = "";
  patrolRuntime.previewHlsStreamId = "";
  patrolRuntime.previewSourceType = "";
  patrolRuntime.previewUrl = "";
  if (stopRemote && sourceType === "proxy" && streamId) {
    try {
      await requestHlsStop(streamId);
    } catch {
      // ignore
    }
  }
}

function cleanupPatrolReview() {
  if (patrolRuntime.reviewTimer) {
    clearTimeout(patrolRuntime.reviewTimer);
    patrolRuntime.reviewTimer = null;
  }
  if (patrolRuntime.reviewAbortController) {
    try {
      patrolRuntime.reviewAbortController.abort();
    } catch {
      // ignore abort errors
    }
    patrolRuntime.reviewAbortController = null;
  }
  patrolRuntime.reviewRunning = false;
  patrolRuntime.reviewInFlight = false;
  patrolRuntime.reviewStopRequested = false;
  patrolRuntime.reviewLastRunAt = 0;
  patrolRuntime.runningStreamId = "";
}

function setActiveNav(route) {
  qsa("[data-nav]").forEach((a) => {
    a.classList.toggle("active", a.getAttribute("data-nav") === route);
  });
}

function route() {
  const hash = location.hash.replace(/^#/, "");
  return hash || "dashboard";
}

function setRoute(next) {
  location.hash = `#${next}`;
}

let lastRoute = "";

function render() {
  const r = route();
  if (lastRoute === "patrol" && r !== "patrol") {
    cleanupPatrolPreview();
  }
  if (lastRoute === "patrol-review" && r !== "patrol-review") {
    cleanupPatrolPreview();
    cleanupPatrolReview();
  }
  lastRoute = r;
  setActiveNav(r);
  const main = qs("#main");
  main.innerHTML = view(r);
  bind(r);
}

function view(r) {
  if (r === "agents") return viewAgents();
  if (r === "chat") return viewChat();
  if (r === "patrol") return viewPatrol();
  if (r === "patrol-review") return viewPatrol({ mode: "review" });
  if (r === "patrol-streams") return viewPatrolStreams();
  if (r === "tools") return viewTools();
  if (r === "prompts") return viewPrompts();
  if (r === "settings") return viewSettings();
  return viewDashboard();
}

function viewDashboard() {
  const base = escapeHtml(state.baseUrl);
  const selected = escapeHtml(state.selectedAgentId || "未选择");
  const agentCount = state.agentsLoaded ? String(state.agents.length) : "—";
  const toolCount = state.toolsLoaded ? String(state.tools.length) : "—";
  const timeNow = escapeHtml(nowTime());
  const healthStatus = state.health?.status || "idle";
  const healthText = healthStatus === "ok" ? "正常" : healthStatus === "fail" ? "异常" : "未检查";
  const healthClass = healthStatus === "ok" ? "pill ok" : healthStatus === "fail" ? "pill bad" : "pill";
  const healthLatency = state.health?.latencyMs ? `${state.health.latencyMs} ms` : "--";
  const healthCheckedAt = state.health?.checkedAt ? escapeHtml(state.health.checkedAt) : "—";
  const agentOptions = buildAgentOptions(state.selectedAgentId);
  const agentPreview = buildAgentsPreview(state.agents);
  return `
  <section class="page page--dashboard landing">
    <header class="landing-hero">
      <div class="hero-copy">
        <div class="hero-kicker">JetLinks Agent</div>
        <div class="hero-title">智能体产品化控制台</div>
        <div class="hero-sub">
          面向智能巡检、复判与多模态分析的统一工作台，从编排到上线，一套完成。
        </div>
        <div class="hero-actions">
          <button class="btn primary" data-route="chat" type="button">开始体验</button>
          <button class="btn" data-action="open-review" type="button">复判中心</button>
          <button class="btn ghost" data-action="docs" type="button">API Docs</button>
        </div>
        <div class="hero-meta">
          <span class="pill">默认 Agent：<span id="dashSelectedAgent">${selected}</span></span>
          <span class="pill">Base URL：<span class="mono" id="dashBaseUrlValue">${base}</span></span>
          <span class="pill">本地时间：<span class="mono">${timeNow}</span></span>
        </div>
      </div>
      <div class="hero-panels">
        <div class="hero-panel">
          <div class="panel-title">系统健康</div>
          <div class="panel-value"><span id="dashHealthBadge" class="${healthClass}">${healthText}</span></div>
          <div class="panel-meta">延迟 <span id="dashHealthLatency">${escapeHtml(healthLatency)}</span></div>
          <div class="panel-actions">
            <button class="btn sm" data-action="health" type="button">健康检查</button>
            <button class="btn sm ghost" data-route="settings" type="button">连接设置</button>
          </div>
        </div>
        <div class="hero-panel">
          <div class="panel-title">资源规模</div>
          <div class="panel-value">
            <div class="panel-stat"><span id="dashAgentCount">${agentCount}</span> 智能体</div>
            <div class="panel-stat"><span id="dashToolCount">${toolCount}</span> 工具</div>
          </div>
          <div class="panel-actions">
            <button class="btn sm" data-action="load-agents" type="button">加载智能体</button>
            <button class="btn sm" data-action="load-tools" type="button">加载工具</button>
          </div>
        </div>
        <div class="hero-panel wide">
          <div class="panel-title">联调入口</div>
          <div class="panel-meta">最近检查：<span class="dash-health-checked">${healthCheckedAt}</span></div>
          <div class="panel-actions">
            <button class="btn sm" data-route="chat" type="button">立即对话</button>
            <button class="btn sm" data-route="agents" type="button">智能体</button>
            <button class="btn sm" data-route="tools" type="button">工具</button>
            <button class="btn sm ghost" data-action="open-review" type="button">复判中心</button>
          </div>
        </div>
      </div>
    </header>

    <section class="landing-section">
      <div class="section-head">
        <div>
          <div class="section-title">产品级能力</div>
          <div class="section-desc">从接入、推理、复判到报告，一条链路打通。</div>
        </div>
        <span class="pill">Capabilities</span>
      </div>
      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-icon">01</div>
          <div class="feature-title">智能体编排</div>
          <div class="feature-desc">模板化能力、工具注册、策略切换，快速复用你的业务知识。</div>
        </div>
        <div class="feature-card">
          <div class="feature-icon">02</div>
          <div class="feature-title">多模态巡检</div>
          <div class="feature-desc">视频/图像/音频统一接入，实时推理并输出结构化证据。</div>
        </div>
        <div class="feature-card">
          <div class="feature-icon">03</div>
          <div class="feature-title">复判闭环</div>
          <div class="feature-desc">异常回放、证据展示、报告归档，支持快速复核与复用。</div>
        </div>
        <div class="feature-card">
          <div class="feature-icon">04</div>
          <div class="feature-title">可观测性</div>
          <div class="feature-desc">健康检查、延迟指标、执行日志，保障上线稳定性。</div>
        </div>
      </div>
    </section>

    <section class="landing-section">
      <div class="section-head">
        <div>
          <div class="section-title">智能体产品流</div>
          <div class="section-desc">一条明确的链路，保证从接入到复判都可落地。</div>
        </div>
        <span class="pill">Workflow</span>
      </div>
      <div class="flow-strip">
        <div class="flow-step">
          <div class="flow-index">01</div>
          <div class="flow-title">接入流</div>
          <div class="flow-desc">RTSP/RTMP/NVR</div>
        </div>
        <div class="flow-step">
          <div class="flow-index">02</div>
          <div class="flow-title">模型推理</div>
          <div class="flow-desc">VLM + 规则</div>
        </div>
        <div class="flow-step">
          <div class="flow-index">03</div>
          <div class="flow-title">结构化输出</div>
          <div class="flow-desc">证据 + JSON</div>
        </div>
        <div class="flow-step">
          <div class="flow-index">04</div>
          <div class="flow-title">复判与报告</div>
          <div class="flow-desc">复核 + 归档</div>
        </div>
      </div>
    </section>

    <section class="landing-section landing-section--control">
      <div class="section-head">
        <div>
          <div class="section-title">产品体验区</div>
          <div class="section-desc">可直接联调 HTTP /api/v1/agent/chat。</div>
        </div>
        <span class="pill">Live</span>
      </div>
      <div class="control-grid">
        <div class="control-card">
          <div class="control-head">
            <div>
              <div class="control-title">快速对话</div>
              <div class="control-desc">支持切换智能体与上下文。</div>
            </div>
            <span class="pill">非流式</span>
          </div>
          <div class="control-body">
            <div class="field">
              <div class="label">选择智能体</div>
              <select id="dashChatAgent">${agentOptions}</select>
            </div>
            <div class="field">
              <div class="label">上下文（JSON，可选）</div>
              <input id="dashChatContext" class="mono" placeholder='{"foo":"bar"}'>
            </div>
            <div class="field">
              <div class="label">发送消息</div>
              <textarea id="dashChatInput" placeholder="输入指令或问题…"></textarea>
            </div>
            <div class="row">
              <button class="btn primary" id="dashChatSend" type="button">发送</button>
              <button class="btn" id="dashChatClear" type="button">清空</button>
            </div>
            <div class="mini-log" id="dashChatLog"></div>
            <div id="dashChatOut" class="help mono"></div>
          </div>
        </div>

        <div class="control-card">
          <div class="control-stack">
            <div class="control-panel">
              <div class="control-head">
                <div>
                  <div class="control-title">连接配置</div>
                  <div class="control-desc">服务地址与鉴权信息即时生效。</div>
                </div>
                <span class="pill">即时生效</span>
              </div>
              <div class="control-body">
                <div class="split">
                  <div class="field">
                    <div class="label">Base URL</div>
                    <input id="dashBaseUrlInput" class="mono" placeholder="http://127.0.0.1:8005" value="${base}">
                  </div>
                  <div class="field">
                    <div class="label">默认 Agent</div>
                    <select id="dashDefaultAgent">${agentOptions}</select>
                  </div>
                </div>
                <div class="split">
                  <div class="field">
                    <div class="label">Header 名称（可选）</div>
                    <input id="dashAuthHeaderName" class="mono" placeholder="Authorization" value="${escapeHtml(state.authHeaderName || "")}">
                  </div>
                  <div class="field">
                    <div class="label">Header 值（可选）</div>
                    <input id="dashAuthHeaderValue" class="mono" placeholder="Bearer xxx" value="${escapeHtml(state.authHeaderValue || "")}">
                  </div>
                </div>
                <div class="row">
                  <button class="btn primary" id="dashSaveConfig" type="button">保存配置</button>
                  <button class="btn" data-route="settings" type="button">更多设置</button>
                </div>
                <div id="dashConfigOut" class="help mono"></div>
              </div>
            </div>

            <div class="control-panel">
              <div class="control-head">
                <div>
                  <div class="control-title">操作输出</div>
                  <div class="control-desc">最近检查：<span class="dash-health-checked">${healthCheckedAt}</span></div>
                </div>
                <span class="pill">API</span>
              </div>
              <div class="control-body">
                <div id="dashOut" class="console mono">等待操作…</div>
                <div class="help" id="dashOutHint" style="margin-top:8px">建议先做健康检查，确认服务可用。</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="landing-section">
      <div class="section-head">
        <div>
          <div class="section-title">智能体样板间</div>
          <div class="section-desc">最近 6 个智能体，快速切换与复用。</div>
        </div>
        <div class="row">
          <button class="btn sm" data-route="agents" type="button">全部</button>
          <button class="btn sm" data-action="load-agents" type="button">刷新</button>
        </div>
      </div>
      <div class="gallery-card">
        <div class="dash-list" id="dashAgentsList">${agentPreview}</div>
        <div id="dashAgentsOut" class="help mono"></div>
      </div>
    </section>

    <section class="landing-section">
      <div class="section-head">
        <div>
          <div class="section-title">快速入口</div>
          <div class="section-desc">常用功能一键访问。</div>
        </div>
        <span class="pill">Quick</span>
      </div>
      <div class="quick-grid">
        <button class="quick-card" data-route="agents" type="button">
          <div class="quick-icon">A</div>
          <div class="quick-title">智能体</div>
          <div class="quick-desc">管理与创建智能体，切换默认实例。</div>
        </button>
        <button class="quick-card" data-route="chat" type="button">
          <div class="quick-icon">C</div>
          <div class="quick-title">对话</div>
          <div class="quick-desc">快速发送消息进行联调。</div>
        </button>
        <button class="quick-card" data-route="patrol" type="button">
          <div class="quick-icon">V</div>
          <div class="quick-title">实时巡检</div>
          <div class="quick-desc">管理视频流并查看流式输出。</div>
        </button>
        <button class="quick-card" data-route="tools" type="button">
          <div class="quick-icon">T</div>
          <div class="quick-title">工具</div>
          <div class="quick-desc">查看可调用工具与说明文档。</div>
        </button>
        <button class="quick-card" data-route="prompts" type="button">
          <div class="quick-icon">P</div>
          <div class="quick-title">提示词</div>
          <div class="quick-desc">模板管理与快速填充调试。</div>
        </button>
        <button class="quick-card" data-route="settings" type="button">
          <div class="quick-icon">S</div>
          <div class="quick-title">设置</div>
          <div class="quick-desc">配置 Base URL 与鉴权信息。</div>
        </button>
        <a class="quick-card" href="/review.html" target="_blank" rel="noreferrer">
          <div class="quick-icon">R</div>
          <div class="quick-title">复判中心</div>
          <div class="quick-desc">查看调用记录与视频片段。</div>
        </a>
      </div>
      <div class="footer-links">
        <a href="/apps.html" target="_blank" rel="noreferrer">旧页面</a>
      </div>
    </section>
  </section>`;
}

function viewSettings() {
  const base = escapeHtml(state.baseUrl);
  const authHeaderName = escapeHtml(state.authHeaderName || "");
  const authHeaderValue = escapeHtml(state.authHeaderValue || "");
  return `
  <div class="container">
    <div class="card">
      <div class="card-header">
        <div class="card-title">设置</div>
        <span class="pill">连接后端</span>
      </div>
      <div class="card-body">
        <div class="field">
          <div class="label">Base URL</div>
          <input id="baseUrl" class="mono" placeholder="http://127.0.0.1:8005" value="${base}">
          <div class="help">默认使用同源：<span class="mono">${escapeHtml(DEFAULT_BASE_URL)}</span>。修改后会影响所有 API 请求。</div>
        </div>

        <div class="field">
          <div class="label">默认 Agent ID</div>
          <input id="defaultAgent" class="mono" placeholder="agent_id" value="${escapeHtml(state.selectedAgentId || "")}">
        </div>

        <div class="card" style="box-shadow:none; margin: 14px 0">
          <div class="card-header">
            <div class="card-title">请求头（可选）</div>
            <span class="pill">鉴权</span>
          </div>
          <div class="card-body">
            <div class="split">
              <div class="field" style="margin:0">
                <div class="label">Header 名称</div>
                <input id="authHeaderName" class="mono" placeholder="Authorization" value="${authHeaderName}">
              </div>
              <div class="field" style="margin:0">
                <div class="label">Header 值</div>
                <input id="authHeaderValue" class="mono" placeholder="Bearer xxx" value="${authHeaderValue}">
              </div>
            </div>
            <div class="help">当前后端默认未启用鉴权；如果你在网关/反代层加了认证，可以在这里配置。</div>
          </div>
        </div>

        <div class="row">
          <button class="btn primary" id="saveSettings">保存</button>
          <button class="btn" id="testConn">测试连接</button>
          <button class="btn danger" id="resetSettings">重置</button>
        </div>

        <div style="height:12px"></div>
        <div id="settingsOut" class="help mono"></div>
      </div>
    </div>
  </div>`;
}

function viewAgents() {
  const list = state.agents || [];
  const total = list.length;
  const activeCount = list.filter((a) => getAgentStatusInfo(a).value === "active").length;
  const templateSource = state.agentTemplates && state.agentTemplates.length
    ? state.agentTemplates
    : buildTemplatesFromAgents(list);
  const templateCount = new Set(templateSource.map((t) => getTemplateId(t)).filter(Boolean)).size;

  const templateOptions = buildTemplateOptions(templateSource, state.agentListTemplate || "");
  const createTemplateOptions = buildCreateTemplateOptions(templateSource, "");

  const cards = list.map((a) => {
    const rawId = a.id || a.agent_id || "";
    const id = escapeHtml(rawId);
    const rawName = a.name || "(未命名)";
    const rawTemplate = getAgentTemplateId(a) || a.type || a.template_id || "-";
    const rawModel = getAgentModel(a) || "";
    const name = escapeHtml(rawName);
    const templateText = escapeHtml(rawTemplate || "-");
    const model = escapeHtml(rawModel);
    const createdAt = escapeHtml(a.create_time || a.created_at || "");
    const active = rawId && state.selectedAgentId === rawId;
    const statusInfo = getAgentStatusInfo(a);
    const statusClass = statusInfo.badge ? `pill ${statusInfo.badge}` : "pill";
    const search = escapeHtml([rawId, rawName, rawTemplate, rawModel].filter(Boolean).join(" ").toLowerCase());
    return `
      <article class="agent-card" data-agent-id="${id}" data-template="${templateText}" data-status="${escapeHtml(statusInfo.value)}" data-search="${search}">
        <div class="agent-main">
          <div class="agent-title-row">
            <div class="agent-title">${name}</div>
            ${active ? '<span class="pill ok">当前</span>' : ""}
            <span class="${statusClass}">${escapeHtml(statusInfo.label)}</span>
          </div>
          <div class="agent-meta mono">${id}</div>
          <div class="agent-tags">
            <span class="tag primary">模板 ${templateText}</span>
            ${model ? `<span class="tag accent">模型 ${model}</span>` : ""}
            ${createdAt ? `<span class="tag ghost mono">${createdAt}</span>` : ""}
          </div>
        </div>
        <div class="agent-actions">
          <button class="btn sm" data-select-agent="${id}">设为默认</button>
          <button class="btn sm primary" data-chat-agent="${id}">对话</button>
        </div>
      </article>`;
  }).join("");

  return `
  <section class="page page--agents">
    <header class="page-head">
      <div>
        <div class="breadcrumb">JetLinks Agent / 管理</div>
        <div class="page-title">智能体控制台</div>
        <div class="page-sub">集中查看、筛选与调度智能体，快速进入对话或设置默认值。</div>
      </div>
      <div class="page-actions">
        <button class="btn" id="reloadAgents">刷新</button>
        <button class="btn primary" id="openCreateAgent">创建</button>
      </div>
    </header>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">智能体总量</div>
        <div class="stat-value">${total || "—"}</div>
        <div class="stat-meta">
          <span class="pill ok">启用 ${activeCount}</span>
          <span class="pill">模板 ${templateCount || "—"}</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-label">默认智能体</div>
        <div class="stat-value mono">${escapeHtml(state.selectedAgentId || "未选择")}</div>
        <div class="stat-meta">
          <span class="pill">用于快速对话入口</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-label">快速动作</div>
        <div class="stat-value">批量排查</div>
        <div class="stat-meta">
          <span class="pill warn">筛选 + 对话联动</span>
        </div>
      </div>
    </div>

    <div class="agents-grid">
      <div class="card agents-list-panel">
        <div class="card-header">
          <div>
            <div class="card-title">智能体列表</div>
            <div class="help">支持搜索名称 / ID / 模型，并可按模板与状态过滤。</div>
          </div>
        </div>
        <div class="card-body">
          <div class="agents-filters">
            <div class="field" style="margin:0">
              <div class="label">搜索</div>
              <input id="agentSearch" placeholder="输入名称、ID 或模型" value="${escapeHtml(state.agentListSearch || "")}">
            </div>
            <div class="field" style="margin:0">
              <div class="label">模板</div>
              <select id="agentTemplateFilter">${templateOptions}</select>
            </div>
            <div class="field" style="margin:0">
              <div class="label">状态</div>
              <select id="agentStatusFilter">
                <option value="">全部状态</option>
                <option value="active" ${state.agentListStatus === "active" ? "selected" : ""}>启用</option>
                <option value="inactive" ${state.agentListStatus === "inactive" ? "selected" : ""}>停用</option>
                <option value="disabled" ${state.agentListStatus === "disabled" ? "selected" : ""}>禁用</option>
              </select>
            </div>
            <div class="agents-count">
              <span class="pill">显示 <span id="agentVisibleCount">${total}</span> / <span id="agentTotalCount">${total}</span></span>
            </div>
          </div>

          <div class="agents-list" id="agentList">
            ${cards}
          </div>
          <div class="agents-empty" id="agentsEmpty" style="${total ? "display:none" : ""}">
            暂无智能体数据，点击「刷新」从后端加载。
          </div>
          <div style="height:12px"></div>
          <div id="agentsOut" class="help mono"></div>
        </div>
      </div>

      <div class="card agents-create-panel">
        <div class="card-header">
          <div class="card-title">创建智能体</div>
          <span class="pill">模板</span>
        </div>
        <div class="card-body">
          <div class="help">创建会调用 <span class="mono">POST /api/v1/agent/create</span>，并可从模板列表选择。</div>
          <div style="height:10px"></div>
          <div class="field">
            <div class="label">模板</div>
            <select id="templateSelect">${createTemplateOptions}</select>
          </div>
          <div class="field">
            <div class="label">名称</div>
            <input id="agentName" placeholder="比如：视频巡检助手">
          </div>
          <div class="field">
            <div class="label">Agent ID（可选）</div>
            <input id="agentId" class="mono" placeholder="不填则自动生成">
          </div>
          <div class="field">
            <div class="label">开场白（可选）</div>
            <textarea id="opening" placeholder="你好，我可以帮你……"></textarea>
          </div>
          <div class="row">
            <button class="btn" id="loadTemplates">刷新模板</button>
            <button class="btn primary" id="createAgent">创建/更新</button>
          </div>
          <div style="height:12px"></div>
          <div id="createOut" class="help mono"></div>
        </div>
      </div>
    </div>
  </section>`;
}

function viewChat() {
  const filteredAgents = filterAgentsByTemplate(state.agents || [], state.agentTemplateFilter);
  const agentOptions = buildAgentOptionsFromList(
    filteredAgents,
    state.selectedAgentId,
    "（该模板下暂无智能体）"
  );
  const templateOptions = buildTemplateOptions(state.agentTemplates || [], state.agentTemplateFilter);

  return `
  <section class="page page--chat">
    <main class="chat-shell">
      <div class="status-pill" id="chatStatus">WS 模式 · /api/v1/jaip/session/{agentId}</div>

      <div class="chat-control">
        <div class="field" style="margin:0">
          <div class="label">模板筛选</div>
          <select id="chatTemplateFilter">${templateOptions}</select>
        </div>
        <div class="field" style="margin:0">
          <div class="label">选择智能体</div>
          <select id="chatAgent">${agentOptions}</select>
        </div>
        <div class="field" style="margin:0">
          <div class="label">上下文（JSON，可选）</div>
          <input id="chatContext" class="mono" placeholder='{"foo":"bar"}'>
        </div>
      </div>

      <div class="chat-log" id="chatLog">
        <div class="message assistant">
          <div class="bubble">
            <div class="assistant-title">你好，我是智能体助手</div>
            <div class="assistant-text">直接输入你的问题或指令，我会按当前选择的智能体进行回复。</div>
          </div>
        </div>
      </div>

      <div class="composer">
        <div class="composer-inner">
          <div class="composer-main">
            <input id="chatInput" type="text" placeholder="输入指令或问题…" autocomplete="off">
            <div class="composer-tools">
              <button class="icon-btn" id="loadAgentsForChat" title="刷新智能体">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M21 12a9 9 0 1 1-2.64-6.36"></path>
                  <polyline points="21 3 21 9 15 9"></polyline>
                </svg>
              </button>
              <button class="icon-btn" id="copyChat" title="复制记录">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
              </button>
              <button class="icon-btn" id="clearChat" title="清空对话">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <path d="M3 6h18"></path>
                  <path d="M8 6V4h8v2"></path>
                  <path d="M10 11v6"></path>
                  <path d="M14 11v6"></path>
                  <path d="M5 6l1 14a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-14"></path>
                </svg>
              </button>
            </div>
          </div>
          <button class="btn primary" id="sendChat">发送</button>
        </div>
        <div class="composer-hint">
          <span class="tag">模式：WS</span>
          <span class="tag mono">/api/v1/jaip/session/{agentId}</span>
          <span class="tag">Enter 发送</span>
        </div>
        <div id="chatOut" class="help mono"></div>
      </div>
    </main>
  </section>`;
}

function viewPatrol(options = {}) {
  const mode = options?.mode || "ws";
  const isReviewMode = mode === "review";
  const pageTitle = isReviewMode ? "实时巡检（复判）" : "实时巡检控制台";
  const pageSub = isReviewMode
    ? "使用复判接口轮询分析巡检流，输出结构化结果与证据。"
    : "选择巡检流与智能体，查看实时输出与证据。";
  const agentLabel = isReviewMode ? "复判智能体" : "巡检智能体";
  const statusText = isReviewMode ? "复判模式" : "WS 未连接";
  const streamOptions = buildPatrolStreamOptions(state.patrolSelectedStreamId);
  return `
  <section class="page page--patrol">
    <header class="page-head patrol-head">
      <div>
        <div class="breadcrumb">JetLinks Agent / 实时巡检</div>
        <div class="page-title">${pageTitle}</div>
        <div class="page-sub">${pageSub}</div>
      </div>
    </header>

    <div class="card patrol-toolbar">
      <div class="card-body">
        <div class="patrol-toolbar-row">
          <div class="patrol-toolbar-fields">
            <div class="field" style="margin:0">
              <div class="label">巡检流</div>
              <select id="patrolStreamPicker">
                ${streamOptions}
              </select>
            </div>
            <div class="field" style="margin:0">
              <div class="label">${agentLabel}</div>
              <select id="patrolAgentSelect">
                <option value="">（加载中）</option>
              </select>
            </div>
          </div>
          <div class="patrol-toolbar-actions">
            <a class="btn sm ghost" href="#patrol-streams">巡检流列表</a>
            <button class="btn sm ghost" id="patrolRefreshAgents">刷新智能体</button>
            <button class="btn sm" id="patrolStart">启动</button>
            <button class="btn sm ghost" id="patrolStop">停止</button>
            <button class="btn sm" id="patrolClear">清空</button>
          </div>
        </div>
        <div class="patrol-toolbar-meta">
          <div id="patrolStreamMeta" class="patrol-selector-tags"></div>
          <div class="patrol-toolbar-status" id="patrolWsStatus">${statusText}</div>
        </div>
      </div>
    </div>

    <div class="patrol-dashboard">
      <div class="patrol-left">
        <div class="card patrol-stage">
          <div class="card-header">
            <div>
              <div class="card-title" id="patrolConsoleTitle">未选择</div>
              <div class="help mono" id="patrolConsoleUrl">—</div>
              <div class="help" id="patrolConsoleMeta">选择一个流开始查看输出。</div>
            </div>
            <div class="patrol-stage-actions">
              <button class="btn sm" id="patrolPreviewPlay">播放</button>
              <button class="btn sm ghost" id="patrolPreviewStop">停止</button>
              <a class="btn sm ghost" id="patrolPreviewLink" target="_blank" rel="noreferrer" style="display:none">打开流</a>
            </div>
          </div>
          <div class="patrol-preview-player">
            <video id="patrolPreviewVideo" class="patrol-preview-video" controls playsinline muted></video>
            <div class="patrol-preview-placeholder" id="patrolPreviewPlaceholder">选择流后播放预览</div>
            <div class="patrol-stage-overlay">
              <span class="pill">Live</span>
            </div>
          </div>
          <div class="patrol-stage-footer">
            <div class="patrol-stage-status">
              <span class="tag" id="patrolPreviewType">HLS</span>
              <span class="mono" id="patrolPreviewStatus">待命</span>
            </div>
            <div class="help" id="patrolPreviewMeta">选择巡检流后点击播放。</div>
          </div>
        </div>

        <div class="card patrol-strip">
          <div class="card-header">
            <div class="card-title">风险时间轴</div>
            <span class="pill">Timeline</span>
          </div>
          <div class="card-body">
            <div class="patrol-strip-bars">
              <span class="bar low" style="--bar:30%"></span>
              <span class="bar mid" style="--bar:46%"></span>
              <span class="bar low" style="--bar:28%"></span>
              <span class="bar high" style="--bar:72%"></span>
              <span class="bar low" style="--bar:36%"></span>
              <span class="bar mid" style="--bar:54%"></span>
              <span class="bar low" style="--bar:22%"></span>
              <span class="bar low" style="--bar:34%"></span>
              <span class="bar mid" style="--bar:52%"></span>
              <span class="bar high" style="--bar:68%"></span>
              <span class="bar low" style="--bar:26%"></span>
              <span class="bar mid" style="--bar:48%"></span>
              <span class="bar low" style="--bar:30%"></span>
              <span class="bar mid" style="--bar:56%"></span>
              <span class="bar low" style="--bar:24%"></span>
              <span class="bar high" style="--bar:70%"></span>
            </div>
            <div class="patrol-strip-legend">
              <span class="legend-item low">低风险</span>
              <span class="legend-item mid">中风险</span>
              <span class="legend-item high">高风险</span>
            </div>
          </div>
        </div>

        <div class="card patrol-summary">
          <div class="card-header">
            <div class="card-title">实时巡检总结</div>
            <span class="pill">Summary</span>
          </div>
          <div class="card-body">
            <div id="patrolConsoleOut" class="patrol-summary-text">等待巡检输出...</div>
          </div>
        </div>
      </div>

      <div class="patrol-right">
        <div class="card patrol-analytics">
          <div class="card-header">
            <div class="card-title">实时巡检分析</div>
            <span class="pill">Stats</span>
          </div>
          <div class="card-body">
            <div class="patrol-metrics">
              <div class="patrol-metric total">
                <div class="metric-label">风险事件总数</div>
                <div class="metric-value" id="patrolStatTotal">0</div>
              </div>
              <div class="patrol-metric low">
                <div class="metric-label">低风险</div>
                <div class="metric-value" id="patrolStatLow">0</div>
              </div>
              <div class="patrol-metric mid">
                <div class="metric-label">中风险</div>
                <div class="metric-value" id="patrolStatMid">0</div>
              </div>
              <div class="patrol-metric high">
                <div class="metric-label">高风险</div>
                <div class="metric-value" id="patrolStatHigh">0</div>
              </div>
            </div>
            <div class="patrol-prompt">
              <div class="label">巡检提示词</div>
              <div id="patrolPromptText" class="patrol-prompt-text">未配置</div>
              <div id="patrolPromptMeta" class="patrol-prompt-meta">字段要求：—</div>
              <a class="btn sm ghost" href="#patrol-streams">去编辑</a>
            </div>
          </div>
        </div>
      </div>

      <div class="card patrol-events patrol-events-wide">
        <div class="card-header">
          <div class="card-title">巡检事件（含证据帧/视频片段）</div>
          <span class="pill">Log + Media</span>
        </div>
        <div class="card-body">
          <div id="patrolLog" class="patrol-log patrol-events-list"></div>
        </div>
      </div>
    </div>
  </section>`;
}

function viewPatrolStreams() {
  const pageSizeOptions = buildPatrolPageSizeOptions(normalizePatrolPageSize(state.patrolPageSize));
  const backRoute = state.patrolMode === "review" ? "patrol-review" : "patrol";
  const backLabel = state.patrolMode === "review" ? "实时巡检（复判）" : "实时巡检";
  const agentLabel = state.patrolMode === "review" ? "复判智能体" : "巡检智能体";
  return `
  <section class="page page--patrol page--patrol-streams">
    <header class="page-head patrol-head">
      <div>
        <div class="breadcrumb">JetLinks Agent / 巡检流列表</div>
        <div class="page-title">巡检流列表</div>
        <div class="page-sub">管理视频流地址与巡检配置，快速进入实时巡检控制台。</div>
      </div>
      <div class="page-actions">
        <a class="btn" href="#${backRoute}">${backLabel}</a>
        <button class="btn" id="patrolRefreshAgents">刷新智能体</button>
        <button class="btn" id="patrolCheckStreams">检测可达</button>
        <button class="btn primary" id="patrolAddStream">新增流</button>
      </div>
    </header>

    <div class="patrol-layout">
      <div class="card patrol-form">
        <div class="card-header">
          <div>
            <div class="card-title">新增巡检流</div>
            <div class="help">支持 RTSP/RTMP/HTTP，本地保存配置，不改后端。</div>
          </div>
          <span class="pill">Stream</span>
        </div>
        <div class="card-body">
          <div class="split">
            <div class="field">
              <div class="label">流名称</div>
              <input id="patrolStreamName" placeholder="例如：厂区东门摄像头">
            </div>
            <div class="field">
              <div class="label">轮询间隔（秒）</div>
              <input id="patrolStreamInterval" type="number" min="0" placeholder="30">
            </div>
          </div>
          <div class="field">
            <div class="label">视频流地址</div>
            <input id="patrolStreamUrl" class="mono" placeholder="rtsp://... 或 http(s)://">
          </div>
          <div class="split">
            <div class="field">
              <div class="label">自动停止（分钟）</div>
              <input id="patrolStreamStop" type="number" min="0" placeholder="5">
            </div>
            <div class="field">
              <div class="label">${agentLabel}</div>
              <select id="patrolAgentSelect">
                <option value="">（加载中）</option>
              </select>
            </div>
          </div>
          <div class="field">
            <div class="label">巡检提示词（可选）</div>
            <textarea id="patrolStreamPrompt" placeholder="描述重点关注点，如：是否有人员闯入/安全帽识别"></textarea>
          </div>
          <div class="row">
            <button class="btn sm" id="patrolSyncAgentPrompt">同步智能体提示词</button>
            <button class="btn sm ghost" id="patrolApplyStream">应用到当前流</button>
          </div>
          <div id="patrolAgentSchemaHint" class="help mono"></div>
          <div class="row">
            <button class="btn primary" id="patrolSaveStream">保存到列表</button>
            <button class="btn ghost" id="patrolResetForm">清空</button>
          </div>
          <div style="height:12px"></div>
          <div id="patrolFormOut" class="help mono"></div>
        </div>
      </div>

      <div class="card patrol-list">
        <div class="card-header">
          <div>
            <div class="card-title">巡检流列表</div>
            <div class="help">点击流可设为当前选择，支持一键进入控制台。</div>
          </div>
          <span class="pill">Manage</span>
        </div>
        <div class="card-body">
          <div class="agents-filters">
            <div class="field" style="margin:0">
              <div class="label">搜索</div>
              <input id="patrolSearch" placeholder="名称 / 地址" value="${escapeHtml(state.patrolSearch || "")}">
            </div>
            <div class="agents-count">
              <span class="pill">可达/总数 <span id="patrolTotalCount">—</span></span>
            </div>
          </div>
          <div id="patrolStreamList" class="patrol-streams"></div>
          <div id="patrolEmpty" class="agents-empty">暂无巡检流配置，先新增一个。</div>
          <div class="patrol-list-footer">
            <div class="patrol-page-info" id="patrolPageInfo">—</div>
            <div class="patrol-page-controls">
              <button class="btn sm ghost" id="patrolPagePrev">上一页</button>
              <button class="btn sm ghost" id="patrolPageNext">下一页</button>
              <select id="patrolPageSize">
                ${pageSizeOptions}
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>`;
}

function viewTools() {
  const rows = (state.tools || []).map((t) => {
    const id = escapeHtml(t.id || "");
    const name = escapeHtml(t.name || "");
    const desc = escapeHtml(t.description || "");
    const asyncFlag = t.async ? `<span class="pill warn">async</span>` : `<span class="pill">sync</span>`;
    return `
      <tr>
        <td class="mono">${id}</td>
        <td>${name}</td>
        <td>${desc}</td>
        <td>${asyncFlag}</td>
      </tr>`;
  }).join("");

  return `
  <div class="container">
    <div class="card">
      <div class="card-header">
        <div class="card-title">工具列表</div>
        <div class="row">
          <button class="btn primary" id="loadTools">加载</button>
          <a class="btn" href="/api/v1/tools/list?include_document=true" target="_blank" rel="noreferrer">原始 JSON</a>
        </div>
      </div>
      <div class="card-body">
        <table class="table">
          <thead>
            <tr>
              <th>Tool ID</th>
              <th>名称</th>
              <th>描述</th>
              <th>模式</th>
            </tr>
          </thead>
          <tbody>${rows || `<tr><td colspan="4" class="help">暂无数据，点击「加载」。</td></tr>`}</tbody>
        </table>
        <div style="height:12px"></div>
        <div id="toolsOut" class="help mono"></div>
      </div>
    </div>
  </div>`;
}

function viewPrompts() {
  const templates = state.templates;
  const cards = templates
    ? Object.entries(templates)
        .map(([key, t]) => {
          return `
          <div class="card" style="box-shadow:none">
            <div class="card-header">
              <div>
                <div class="card-title">${escapeHtml(t.name || key)}</div>
                <div class="help mono">${escapeHtml(key)}</div>
              </div>
              <button class="btn" data-use-template="${escapeHtml(key)}">使用</button>
            </div>
            <div class="card-body">
              <div class="help">变量：<span class="mono">${escapeHtml((t.variables || []).join(", "))}</span></div>
              <div style="height:10px"></div>
              <div class="msg mono">${escapeHtml(t.template || "")}</div>
            </div>
          </div>`;
        })
        .join("")
    : `<div class="help">尚未加载模板。</div>`;

  return `
  <div class="container">
    <div class="grid">
      <div class="card">
        <div class="card-header">
          <div class="card-title">提示词模板</div>
          <div class="row">
            <button class="btn primary" id="loadPromptTemplates">加载</button>
            <a class="btn" href="/api/v1/prompt/templates" target="_blank" rel="noreferrer">原始 JSON</a>
          </div>
        </div>
        <div class="card-body" style="display:grid; gap:12px">
          ${cards}
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">快速填充</div>
          <span class="pill">生成提示词</span>
        </div>
        <div class="card-body">
          <div class="field">
            <div class="label">模板 Key</div>
            <input id="tmplKey" class="mono" placeholder="例如：code_generation">
          </div>
          <div class="field">
            <div class="label">变量（JSON）</div>
            <textarea id="tmplVars" class="mono" placeholder='{"language":"Python","requirements":"...","constraints":"..."}'></textarea>
            <div class="help">可调用 <span class="mono">POST /api/v1/prompt/template/apply</span> 进行渲染（如果后端开启）。</div>
          </div>
          <div class="row">
            <button class="btn" id="applyTemplate">渲染</button>
          </div>
          <div style="height:12px"></div>
          <div id="promptOut" class="help mono"></div>
        </div>
      </div>
    </div>
  </div>`;
}

function bind(r) {
  if (r === "dashboard") return bindDashboard();
  if (r === "settings") return bindSettings();
  if (r === "agents") return bindAgents();
  if (r === "chat") return bindChat();
  if (r === "patrol") return bindPatrol();
  if (r === "patrol-review") return bindPatrol();
  if (r === "patrol-streams") return bindPatrol();
  if (r === "tools") return bindTools();
  if (r === "prompts") return bindPrompts();
}

function refreshDashboardAgents() {
  const count = qs("#dashAgentCount");
  if (count) count.textContent = state.agentsLoaded ? String(state.agents.length) : "—";
  const selected = qs("#dashSelectedAgent");
  if (selected) selected.textContent = state.selectedAgentId || "未选择";

  const options = buildAgentOptions(state.selectedAgentId);
  const chatSelect = qs("#dashChatAgent");
  if (chatSelect) chatSelect.innerHTML = options;
  const defaultSelect = qs("#dashDefaultAgent");
  if (defaultSelect) defaultSelect.innerHTML = options;

  if (state.selectedAgentId) {
    if (chatSelect) chatSelect.value = state.selectedAgentId;
    if (defaultSelect) defaultSelect.value = state.selectedAgentId;
  }

  const list = qs("#dashAgentsList");
  if (list) list.innerHTML = buildAgentsPreview(state.agents);
}

function refreshDashboardTools() {
  const count = qs("#dashToolCount");
  if (count) count.textContent = state.toolsLoaded ? String(state.tools.length) : "—";
}

function refreshDashboardHealth() {
  const badge = qs("#dashHealthBadge");
  const latency = qs("#dashHealthLatency");
  const checkedList = qsa(".dash-health-checked");
  const status = state.health?.status || "idle";
  const text = status === "ok" ? "正常" : status === "fail" ? "异常" : "未检查";

  if (badge) {
    badge.textContent = text;
    badge.className = status === "ok" ? "pill ok" : status === "fail" ? "pill bad" : "pill";
  }
  if (latency) latency.textContent = state.health?.latencyMs ? `${state.health.latencyMs} ms` : "--";
  checkedList.forEach((el) => {
    el.textContent = state.health?.checkedAt || "—";
  });
}

function refreshDashboardConfig() {
  const baseValue = qs("#dashBaseUrlValue");
  if (baseValue) baseValue.textContent = state.baseUrl;
  const baseInput = qs("#dashBaseUrlInput");
  if (baseInput) baseInput.value = state.baseUrl;
  const authName = qs("#dashAuthHeaderName");
  if (authName) authName.value = state.authHeaderName || "";
  const authValue = qs("#dashAuthHeaderValue");
  if (authValue) authValue.value = state.authHeaderValue || "";

  const badge = qs("#baseUrlBadge");
  if (badge) badge.textContent = state.baseUrl;
}

function appendQuickMessage(role, content) {
  const log = qs("#dashChatLog");
  if (!log) return;
  const div = document.createElement("div");
  div.className = `msg ${role === "user" ? "user" : "assistant"}`;
  div.innerHTML = `
    <div class="msg-meta">
      <span>${role === "user" ? "你" : "智能体"}</span>
      <span class="mono">${escapeHtml(new Date().toLocaleTimeString())}</span>
    </div>
    <div>${escapeHtml(content)}</div>
  `;
  log.appendChild(div);
  while (log.children.length > 6) log.removeChild(log.firstChild);
  log.scrollTop = log.scrollHeight;
}

function bindDashboard() {
  qsa("[data-route]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-route");
      if (target) setRoute(target);
    });
  });

  qsa("[data-scroll]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const selector = btn.getAttribute("data-scroll");
      if (!selector) return;
      const target = qs(selector);
      if (!target) return;
      const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      target.scrollIntoView({ behavior: prefersReduced ? "auto" : "smooth", block: "start" });
    });
  });

  qsa('[data-action="docs"]').forEach((btn) => {
    btn.addEventListener("click", () => window.open(`${state.baseUrl}/docs`, "_blank"));
  });

  qsa('[data-action="open-review"]').forEach((btn) => {
    btn.addEventListener("click", () => window.open("/review.html", "_blank"));
  });

  qsa('[data-action="health"]').forEach((btn) => {
    btn.addEventListener("click", async () => {
      const out = qs("#dashOut");
      if (!out) return;
      out.textContent = "Checking /health ...";
      const start = performance.now();
      try {
        const res = await apiFetch("/health");
        const latencyMs = Math.max(1, Math.round(performance.now() - start));
        state.health = { status: "ok", latencyMs, checkedAt: nowTime() };
        refreshDashboardHealth();
        out.textContent = JSON.stringify(res, null, 2);
        toast("健康检查成功", `延迟 ${latencyMs} ms`);
      } catch (e) {
        state.health = { status: "fail", latencyMs: null, checkedAt: nowTime() };
        refreshDashboardHealth();
        out.textContent = String(e);
        toast("健康检查失败", String(e));
      }
    });
  });

  qsa('[data-action="load-agents"]').forEach((btn) => {
    btn.addEventListener("click", async () => {
      const out = qs("#dashOut");
      if (!out) return;
      out.textContent = "Loading agents ...";
      try {
        await loadAgents();
        refreshDashboardAgents();
        out.textContent = `Loaded agents: ${state.agents.length}`;
        toast("智能体加载成功", `${state.agents.length} 个`);
      } catch (e) {
        out.textContent = String(e);
        toast("智能体加载失败", String(e));
      }
    });
  });

  qsa('[data-action="load-tools"]').forEach((btn) => {
    btn.addEventListener("click", async () => {
      const out = qs("#dashOut");
      if (!out) return;
      out.textContent = "Loading tools ...";
      try {
        await loadTools(false);
        refreshDashboardTools();
        out.textContent = `Loaded tools: ${state.tools.length}`;
        toast("工具加载成功", `${state.tools.length} 个`);
      } catch (e) {
        out.textContent = String(e);
        toast("工具加载失败", String(e));
      }
    });
  });

  const dashChatAgent = qs("#dashChatAgent");
  if (dashChatAgent) {
    dashChatAgent.addEventListener("change", () => {
      const id = dashChatAgent.value;
      if (!id) return;
      state.selectedAgentId = id;
      store.set("jl_agentId", id);
      refreshDashboardAgents();
    });
  }

  const dashChatSend = qs("#dashChatSend");
  const dashChatClear = qs("#dashChatClear");
  if (dashChatClear) {
    dashChatClear.addEventListener("click", () => {
      const log = qs("#dashChatLog");
      const out = qs("#dashChatOut");
      if (log) log.innerHTML = "";
      if (out) out.textContent = "";
    });
  }
  if (dashChatSend) {
    const sendQuickChat = async () => {
      const out = qs("#dashChatOut");
      const agentId = (qs("#dashChatAgent")?.value || state.selectedAgentId || "").trim();
      const message = (qs("#dashChatInput")?.value || "").trim();
      const contextStr = (qs("#dashChatContext")?.value || "").trim();
      const context = contextStr ? readJsonSafely(contextStr) : {};

      if (!agentId) {
        if (out) out.textContent = "请先加载并选择智能体";
        return;
      }
      if (!message) return;
      if (contextStr && context === null) {
        if (out) out.textContent = "上下文 JSON 解析失败";
        return;
      }

      state.selectedAgentId = agentId;
      store.set("jl_agentId", agentId);

      appendQuickMessage("user", message);
      if (out) out.textContent = "Sending ...";
      const inputEl = qs("#dashChatInput");
      if (inputEl) inputEl.value = "";

      try {
        const result = await apiFetchStandard("/api/v1/agent/chat", {
          method: "POST",
          body: JSON.stringify({
            agent_id: agentId,
            message,
            context: context || {},
          }),
        });
        const answer =
          typeof result === "string"
            ? result
            : result?.message || result?.result || JSON.stringify(result, null, 2);
        appendQuickMessage("assistant", answer);
        if (out) out.textContent = "";
      } catch (e) {
        appendQuickMessage("assistant", `请求失败：${String(e)}`);
        if (out) out.textContent = String(e);
        toast("对话失败", String(e));
      }
    };

    dashChatSend.addEventListener("click", sendQuickChat);
    const chatInput = qs("#dashChatInput");
    if (chatInput) {
      chatInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") sendQuickChat();
      });
    }
  }

  const dashSaveConfig = qs("#dashSaveConfig");
  if (dashSaveConfig) {
    dashSaveConfig.addEventListener("click", () => {
      const base = normalizeBaseUrl(qs("#dashBaseUrlInput")?.value || "");
      const agentId = (qs("#dashDefaultAgent")?.value || "").trim();
      const authHeaderName = (qs("#dashAuthHeaderName")?.value || "").trim();
      const authHeaderValue = (qs("#dashAuthHeaderValue")?.value || "").trim();
      const out = qs("#dashConfigOut");

      state.baseUrl = base;
      state.selectedAgentId = agentId;
      state.authHeaderName = authHeaderName;
      state.authHeaderValue = authHeaderValue;
      store.set("jl_baseUrl", base);
      store.set("jl_agentId", agentId);
      store.set("jl_authHeaderName", authHeaderName);
      store.set("jl_authHeaderValue", authHeaderValue);

      refreshDashboardConfig();
      refreshDashboardAgents();

      if (out) {
        out.textContent = `Saved. baseUrl=${base} agentId=${agentId || "(none)"} auth=${authHeaderName ? `${authHeaderName}: *****` : "(none)"}`;
      }
      toast("已保存", base);
    });
  }

  const agentList = qs("#dashAgentsList");
  if (agentList) {
    agentList.addEventListener("click", (event) => {
      const loadTarget = event.target.closest('[data-action="load-agents"]');
      if (loadTarget) {
        const out = qs("#dashOut");
        if (!out) return;
        out.textContent = "Loading agents ...";
        loadAgents()
          .then(() => {
            refreshDashboardAgents();
            out.textContent = `Loaded agents: ${state.agents.length}`;
            toast("智能体加载成功", `${state.agents.length} 个`);
          })
          .catch((e) => {
            out.textContent = String(e);
            toast("智能体加载失败", String(e));
          });
        return;
      }

      const target = event.target.closest("[data-dash-select-agent],[data-dash-chat-agent]");
      if (!target) return;
      const selectId = target.getAttribute("data-dash-select-agent");
      const chatId = target.getAttribute("data-dash-chat-agent");
      const agentId = (selectId || chatId || "").trim();
      if (!agentId) return;

      state.selectedAgentId = agentId;
      store.set("jl_agentId", agentId);
      refreshDashboardAgents();

      if (chatId) {
        setRoute("chat");
        return;
      }
      toast("已选择智能体", agentId);
    });
  }

  refreshDashboardAgents();
  refreshDashboardTools();
  refreshDashboardHealth();
  refreshDashboardConfig();
}

function bindSettings() {
  const out = qs("#settingsOut");

  qs("#saveSettings").addEventListener("click", () => {
    const base = normalizeBaseUrl(qs("#baseUrl").value);
    const agentId = (qs("#defaultAgent").value || "").trim();
    const authHeaderName = (qs("#authHeaderName")?.value || "").trim();
    const authHeaderValue = (qs("#authHeaderValue")?.value || "").trim();
    state.baseUrl = base;
    state.selectedAgentId = agentId;
    state.authHeaderName = authHeaderName;
    state.authHeaderValue = authHeaderValue;
    store.set("jl_baseUrl", base);
    store.set("jl_agentId", agentId);
    store.set("jl_authHeaderName", authHeaderName);
    store.set("jl_authHeaderValue", authHeaderValue);
    out.textContent = `Saved. baseUrl=${base} agentId=${agentId || "(none)"} auth=${authHeaderName ? `${authHeaderName}: *****` : "(none)"}`;
    toast("已保存", base);
  });

  qs("#resetSettings").addEventListener("click", () => {
    state.baseUrl = DEFAULT_BASE_URL;
    state.selectedAgentId = "";
    state.authHeaderName = "";
    state.authHeaderValue = "";
    store.set("jl_baseUrl", DEFAULT_BASE_URL);
    store.set("jl_agentId", "");
    store.set("jl_authHeaderName", "");
    store.set("jl_authHeaderValue", "");
    render();
    toast("已重置", DEFAULT_BASE_URL);
  });

  qs("#testConn").addEventListener("click", async () => {
    out.textContent = "Testing /api/v1 ...";
    try {
      const res = await apiFetch("/api/v1/");
      out.textContent = JSON.stringify(res, null, 2);
      toast("连接正常", "API v1 OK");
    } catch (e) {
      out.textContent = String(e);
      toast("连接失败", String(e));
    }
  });
}

async function loadAgents() {
  const result = await apiFetchStandard("/api/v1/agent/list", {
    method: "POST",
    body: JSON.stringify({
      pageIndex: 0,
      pageSize: 50,
      sorts: [],
      terms: [],
    }),
  });

  // result can be {data,total,...} or list - tolerate both
  const list = Array.isArray(result) ? result : result?.data || result?.result?.data || [];
  state.agents = list;
  state.agentsLoaded = true;
  if (!state.selectedAgentId && list.length) {
    const firstId = list[0]?.id || list[0]?.agent_id;
    if (firstId) state.selectedAgentId = firstId;
  }
  if (list.length) {
    const derivedTemplates = buildTemplatesFromAgents(list);
    if (derivedTemplates.length) {
      state.agentTemplates = mergeTemplateLists(state.agentTemplates || [], derivedTemplates);
    }
  }
  return state.agents;
}

async function loadTools(includeDocument = true) {
  const list = await apiFetch(`/api/v1/tools/list?include_document=${includeDocument ? "true" : "false"}`);
  state.tools = Array.isArray(list) ? list : [];
  state.toolsLoaded = true;
  return state.tools;
}

function isExternalTool(tool) {
  if (!tool || typeof tool !== "object") return false;
  const source = String(tool.source || "").trim().toLowerCase();
  if (source) return source === "external";
  const id = String(tool.id || tool.name || "");
  return id.includes(":") || id.includes("#");
}

async function getAvailableExternalTools() {
  if (!state.toolsLoaded) {
    try {
      await loadTools(false);
    } catch (e) {
      console.warn("Load tools failed:", e);
      return [];
    }
  }
  return (state.tools || []).filter(isExternalTool);
}

async function loadTemplates() {
  const endpoints = [
    "/api/v1/agent/templates/list",
    "/api/v1/agent/agent-templates/list",
  ];
  let lastError = null;
  for (const path of endpoints) {
    try {
      const res = await apiFetch(path, {
        method: "POST",
        body: JSON.stringify({}),
      });
      const templates = res?.result || res?.data || res || [];
      return Array.isArray(templates) ? templates : templates?.data || [];
    } catch (e) {
      lastError = e;
    }
  }
  throw lastError || new Error("模板接口不可用");
}

async function loadAgentTemplates() {
  const list = await loadTemplates();
  const derived = buildTemplatesFromAgents(state.agents || []);
  const merged = mergeTemplateLists(list, derived);
  state.agentTemplates = merged;
  return merged;
}

function fillTemplateSelect(list) {
  const select = qs("#templateSelect");
  if (!select) return;
  select.innerHTML = (list || []).map((t) => {
    const id = escapeHtml(t.id || t.type || "");
    const name = escapeHtml(t.name || id);
    const desc = escapeHtml(t.description || "");
    return `<option value="${id}">${name}${desc ? ` - ${desc}` : ""}</option>`;
  }).join("");
}

function bindAgents() {
  const out = qs("#agentsOut");
  const createOut = qs("#createOut");
  const searchInput = qs("#agentSearch");
  const templateFilter = qs("#agentTemplateFilter");
  const statusFilter = qs("#agentStatusFilter");
  const listEl = qs("#agentList");
  const emptyEl = qs("#agentsEmpty");
  const visibleCount = qs("#agentVisibleCount");
  const totalCount = qs("#agentTotalCount");
  const templateSelect = qs("#templateSelect");

  qs("#reloadAgents").addEventListener("click", async () => {
    out.textContent = "Loading ...";
    try {
      await loadAgents();
      render();
      toast("已刷新智能体", `${state.agents.length} 个`);
    } catch (e) {
      out.textContent = String(e);
      toast("刷新失败", String(e));
    }
  });

  qs("#openCreateAgent")?.addEventListener("click", () => {
    qs(".agents-create-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  const refreshTemplateSelects = () => {
    const source = (state.agentTemplates && state.agentTemplates.length)
      ? state.agentTemplates
      : buildTemplatesFromAgents(state.agents || []);
    if (templateFilter) {
      templateFilter.innerHTML = buildTemplateOptions(source, state.agentListTemplate || "");
      templateFilter.value = state.agentListTemplate || "";
    }
    if (templateSelect) {
      const current = templateSelect.value || "";
      templateSelect.innerHTML = buildCreateTemplateOptions(source, current);
      templateSelect.value = current;
    }
  };

  qs("#loadTemplates").addEventListener("click", async () => {
    createOut.textContent = "Loading templates ...";
    try {
      const list = await loadAgentTemplates();
      refreshTemplateSelects();
      createOut.textContent = `Loaded templates: ${list.length}`;
      toast("模板加载成功", `${list.length} 个`);
    } catch (e) {
      createOut.textContent = String(e);
      toast("模板加载失败", String(e));
    }
  });

  templateSelect?.addEventListener("focus", () => {
    if (templateSelect.options.length > 1) return;
    loadAgentTemplates()
      .then(() => {
        refreshTemplateSelects();
      })
      .catch((e) => {
        if (createOut) createOut.textContent = `模板加载失败：${String(e)}`;
      });
  });

  const applyAgentFilters = () => {
    if (!listEl) return;
    const search = (searchInput?.value || "").trim().toLowerCase();
    const template = (templateFilter?.value || "").trim();
    const status = (statusFilter?.value || "").trim().toLowerCase();
    const cards = qsa(".agent-card", listEl);
    let visible = 0;
    cards.forEach((card) => {
      const hay = (card.getAttribute("data-search") || "").toLowerCase();
      const tpl = card.getAttribute("data-template") || "";
      const st = (card.getAttribute("data-status") || "").toLowerCase();
      const matches = (!search || hay.includes(search))
        && (!template || tpl === template)
        && (!status || st === status);
      card.hidden = !matches;
      if (matches) visible += 1;
    });
    if (visibleCount) visibleCount.textContent = String(visible);
    if (totalCount && !totalCount.textContent) totalCount.textContent = String(cards.length);
    if (emptyEl) emptyEl.style.display = visible ? "none" : "block";
  };

  searchInput?.addEventListener("input", () => {
    state.agentListSearch = searchInput.value;
    store.set("jl_agentListSearch", state.agentListSearch);
    applyAgentFilters();
  });

  templateFilter?.addEventListener("change", () => {
    state.agentListTemplate = templateFilter.value;
    store.set("jl_agentListTemplate", state.agentListTemplate);
    applyAgentFilters();
  });

  statusFilter?.addEventListener("change", () => {
    state.agentListStatus = statusFilter.value;
    store.set("jl_agentListStatus", state.agentListStatus);
    applyAgentFilters();
  });

  qs("#createAgent").addEventListener("click", async () => {
    createOut.textContent = "Creating ...";
    const templateId = qs("#templateSelect").value || null;
    const name = qs("#agentName").value.trim();
    const id = qs("#agentId").value.trim() || null;
    const opening = qs("#opening").value.trim() || null;
    if (!name) {
      createOut.textContent = "请输入名称";
      return;
    }

    try {
      const result = await apiFetchStandard("/api/v1/agent/create", {
        method: "POST",
        body: JSON.stringify({
          id,
          template_id: templateId,
          name,
          opening_statement: opening,
          suggested_questions: true,
          output_format: "markdown",
          config: templateId ? { type: templateId } : {},
        }),
      });
      createOut.textContent = JSON.stringify(result, null, 2);
      const newId = result?.id || id;
      if (newId) {
        state.selectedAgentId = newId;
        store.set("jl_agentId", newId);
      }
      toast("创建成功", newId || "");
      await loadAgents();
      render();
    } catch (e) {
      createOut.textContent = String(e);
      toast("创建失败", String(e));
    }
  });

  qsa("[data-select-agent]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-select-agent");
      if (!id) return;
      state.selectedAgentId = id;
      store.set("jl_agentId", id);
      toast("已选择智能体", id);
      render();
    });
  });

  qsa("[data-chat-agent]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-chat-agent");
      if (!id) return;
      state.selectedAgentId = id;
      store.set("jl_agentId", id);
      setRoute("chat");
    });
  });

  // best-effort auto load
  if (!state.agents.length) {
    qs("#reloadAgents").click();
  } else {
    refreshTemplateSelects();
    applyAgentFilters();
  }

  if (!state.agentTemplates.length) {
    loadAgentTemplates()
      .then(() => {
        refreshTemplateSelects();
        applyAgentFilters();
      })
      .catch((e) => {
        if (createOut) createOut.textContent = `模板加载失败：${String(e)}`;
      });
  }
}

function appendMessage(role, content) {
  const log = qs("#chatLog");
  if (!log) return;
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role === "user" ? "user" : "assistant"}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    const title = document.createElement("div");
    title.className = "assistant-title";
    title.textContent = "智能体";
    const text = document.createElement("div");
    text.className = "assistant-text";
    text.textContent = String(content || "");
    bubble.appendChild(title);
    bubble.appendChild(text);
  } else {
    const text = document.createElement("div");
    text.className = "user-text";
    text.textContent = String(content || "");
    bubble.appendChild(text);
  }
  wrapper.appendChild(bubble);
  log.appendChild(wrapper);
  log.scrollTop = log.scrollHeight;
}

function appendAssistantStream() {
  const log = qs("#chatLog");
  if (!log) return null;
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const title = document.createElement("div");
  title.className = "assistant-title";
  title.textContent = "智能体";
  const text = document.createElement("div");
  text.className = "assistant-text";
  text.textContent = "";
  bubble.appendChild(title);
  bubble.appendChild(text);
  wrapper.appendChild(bubble);
  log.appendChild(wrapper);
  log.scrollTop = log.scrollHeight;
  return text;
}

function bindChat() {
  const out = qs("#chatOut");
  const status = qs("#chatStatus");
  const templateSelect = qs("#chatTemplateFilter");
  const agentSelect = qs("#chatAgent");

  const updateStatus = () => {
    if (!status) return;
    const agentId = (qs("#chatAgent")?.value || state.selectedAgentId || "未选择").trim();
    let wsState = "未连接";
    if (chatWsState.socket) {
      const ready = chatWsState.socket.readyState;
      if (ready === WebSocket.OPEN) wsState = "在线";
      else if (ready === WebSocket.CONNECTING) wsState = "连接中";
      else wsState = "断开";
    }
    status.textContent = `WS 模式 · Agent ${agentId || "未选择"} · ${wsState}`;
  };

  const refreshTemplateOptions = () => {
    if (!templateSelect) return;
    templateSelect.innerHTML = buildTemplateOptions(state.agentTemplates || [], state.agentTemplateFilter);
    templateSelect.value = state.agentTemplateFilter || "";
  };

  const refreshChatAgentOptions = () => {
    if (!agentSelect) return;
    const templateId = (templateSelect?.value || "").trim();
    state.agentTemplateFilter = templateId;
    store.set("jl_agentTemplateFilter", templateId);
    const filtered = filterAgentsByTemplate(state.agents || [], templateId);
    const prevSelected = state.selectedAgentId;
    const nextSelected = filtered.some((a) => getAgentId(a) === prevSelected)
      ? prevSelected
      : (getAgentId(filtered[0]) || "");
    agentSelect.innerHTML = buildAgentOptionsFromList(
      filtered,
      nextSelected,
      "（该模板下暂无智能体）"
    );
    agentSelect.value = nextSelected || "";
    state.selectedAgentId = nextSelected || "";
    store.set("jl_agentId", state.selectedAgentId);
    updateStatus();
  };

  qsa("[data-fill]").forEach((btn) => {
    btn.addEventListener("click", () => {
      qs("#chatInput").value = btn.getAttribute("data-fill") || "";
      qs("#chatInput").focus();
    });
  });

  qs("#loadAgentsForChat").addEventListener("click", async () => {
    out.textContent = "Loading agents ...";
    try {
      await Promise.all([loadAgents(), loadAgentTemplates()]);
      render();
      toast("智能体已刷新", `${state.agents.length} 个`);
    } catch (e) {
      out.textContent = String(e);
      toast("加载失败", String(e));
    }
  });

  qs("#clearChat").addEventListener("click", () => {
    qs("#chatLog").innerHTML = "";
    out.textContent = "";
  });

  qs("#copyChat").addEventListener("click", async () => {
    const text = qs("#chatLog").innerText;
    await navigator.clipboard.writeText(text);
    toast("已复制", "对话记录已复制到剪贴板");
  });

  if (templateSelect) {
    templateSelect.addEventListener("change", () => {
      resetChatWs("template_change");
      refreshChatAgentOptions();
    });
  }

  function sendWs(payload) {
    const socket = chatWsState.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket 未连接");
    }
    socket.send(JSON.stringify(payload));
  }

  async function requestInit() {
    if (!chatWsState.socket || chatWsState.initRequestId) return;
    chatWsState.initError = "";
    chatWsState.initRequestId = rpcId("init");
    const availableTools = await getAvailableExternalTools();
    try {
      sendWs({
        jsonrpc: "2.0",
        id: chatWsState.initRequestId,
        method: "session.initialize",
        params: {
          protocolVersion: "2.0",
          userContext: { userId: "web" },
          availableTools,
        },
      });
    } catch (e) {
      chatWsState.initRequestId = "";
      chatWsState.initError = String(e);
      out.textContent = chatWsState.initError;
    }
  }

  function rejectExternalTool(executionId, toolName = "") {
    if (!executionId) return;
    try {
      sendWs({
        jsonrpc: "2.0",
        id: executionId,
        error: {
          code: -32000,
          message: `客户端未实现外部工具执行${toolName ? `: ${toolName}` : ""}`,
        },
      });
    } catch {
      // ignore
    }
  }

  function handleWsMessage(data) {
    if (!data || typeof data !== "object") return;

    if (data.type === "connected") {
      chatWsState.sessionId = data.session_id || "";
      updateStatus();
      return;
    }

    if (data.id && data.id === chatWsState.initRequestId) {
      chatWsState.initDone = !data.error;
      if (data.error) {
        chatWsState.initError = normalizeWsError(data.error?.message) || "初始化失败";
        out.textContent = chatWsState.initError;
        appendMessage("assistant", `初始化失败：${chatWsState.initError}`);
      } else if (data.result?.sessionId) {
        chatWsState.sessionId = data.result.sessionId;
      }
      updateStatus();
      return;
    }

    if (data.method === "session.response_start") {
      const responseId = data.params?.responseId || rpcId("resp");
      const textEl = appendAssistantStream();
      if (textEl) {
        chatWsState.streams.set(responseId, { textEl, buffer: "" });
      }
      out.textContent = "";
      return;
    }

    if (data.method === "session.response_chunk") {
      const responseId = data.params?.responseId || "";
      const chunk = data.params?.chunk?.content || "";
      if (!chunk) return;
      let entry = chatWsState.streams.get(responseId);
      if (!entry) {
        const textEl = appendAssistantStream();
        if (!textEl) return;
        entry = { textEl, buffer: "" };
        chatWsState.streams.set(responseId, entry);
      }
      entry.buffer += chunk;
      entry.textEl.textContent = entry.buffer;
      return;
    }

    if (data.method === "session.response_end") {
      const responseId = data.params?.responseId || "";
      const status = data.params?.status || "";
      const error = normalizeWsError(data.params?.metadata?.error) || "";
      if (status === "error") {
        appendMessage("assistant", `请求失败：${error || "未知错误"}`);
      }
      chatWsState.streams.delete(responseId);
      return;
    }

    if (data.method === "tools.confirm") {
      const toolType = data.params?.toolType || "external";
      const toolName = data.params?.call?.toolName || "";
      if (toolType === "external") {
        appendMessage("assistant", `需要外部工具：${toolName || "未知工具"}（前端未实现，已拒绝）`);
        rejectExternalTool(data.id, toolName);
      }
      return;
    }

    if (data.method === "tools.execute") {
      const toolName = data.params?.toolName || "";
      appendMessage("assistant", `需要执行工具：${toolName || "未知工具"}（前端未实现，已拒绝）`);
      rejectExternalTool(data.id, toolName);
      return;
    }

    if (data.error) {
      const message = normalizeWsError(data.error?.message);
      appendMessage("assistant", `请求失败：${message || "未知错误"}`);
      return;
    }
  }

  function waitForInit(timeoutMs = 8000) {
    if (chatWsState.initDone) return Promise.resolve(true);
    if (chatWsState.initError) return Promise.reject(new Error(chatWsState.initError));

    return new Promise((resolve, reject) => {
      const start = Date.now();
      const timer = setInterval(() => {
        if (chatWsState.initDone) {
          clearInterval(timer);
          resolve(true);
          return;
        }
        if (chatWsState.initError) {
          clearInterval(timer);
          reject(new Error(chatWsState.initError));
          return;
        }
        if (Date.now() - start > timeoutMs) {
          clearInterval(timer);
          reject(new Error("初始化超时"));
        }
      }, 120);
    });
  }

  async function ensureChatWs(agentId) {
    if (chatWsState.socket && chatWsState.agentId === agentId) {
      if (chatWsState.socket.readyState === WebSocket.OPEN) return chatWsState.socket;
      if (chatWsState.connectPromise) return chatWsState.connectPromise;
    }

    resetChatWs("agent_switch");
    chatWsState.agentId = agentId;
    const wsUrl = buildChatWsUrl(agentId);
    const socket = new WebSocket(wsUrl);
    chatWsState.socket = socket;

    socket.addEventListener("message", (event) => {
      try {
        const payload = typeof event.data === "string" ? JSON.parse(event.data) : event.data;
        handleWsMessage(payload);
      } catch (e) {
        out.textContent = `消息解析失败: ${String(e)}`;
      }
    });

    socket.addEventListener("close", () => {
      if (chatWsState.socket === socket) {
        chatWsState.socket = null;
        chatWsState.initRequestId = "";
        chatWsState.initDone = false;
        chatWsState.connectPromise = null;
      }
      updateStatus();
    });

    socket.addEventListener("error", () => {
      out.textContent = "WebSocket 连接失败";
      updateStatus();
    });

    chatWsState.connectPromise = new Promise((resolve, reject) => {
      socket.addEventListener("open", () => {
        updateStatus();
        void requestInit();
        resolve(socket);
      }, { once: true });

      socket.addEventListener("error", () => {
        reject(new Error("WebSocket 连接失败"));
      }, { once: true });
    });

    return chatWsState.connectPromise;
  }

  async function send() {
    const agentId = qs("#chatAgent").value || state.selectedAgentId;
    const message = qs("#chatInput").value.trim();
    const contextStr = qs("#chatContext").value.trim();
    const context = contextStr ? readJsonSafely(contextStr) : {};

    if (!agentId) {
      out.textContent = "请先选择智能体";
      return;
    }
    if (!message) return;
    if (contextStr && context === null) {
      out.textContent = "上下文 JSON 解析失败";
      return;
    }

    state.selectedAgentId = agentId;
    store.set("jl_agentId", agentId);

    appendMessage("user", message);
    qs("#chatInput").value = "";
    out.textContent = "WS 发送中...";

    try {
      await ensureChatWs(agentId);
      if (!chatWsState.initDone) {
        await requestInit();
        await waitForInit();
      }
      sendWs({
        jsonrpc: "2.0",
        id: rpcId("msg"),
        method: "session.message",
        params: {
          sessionId: chatWsState.sessionId || undefined,
          messageType: "text",
          content: message,
          context: context || {},
        },
      });
      out.textContent = "";
    } catch (e) {
      appendMessage("assistant", `请求失败：${String(e)}`);
      out.textContent = String(e);
      toast("对话失败", String(e));
    }
  }

  qs("#sendChat").addEventListener("click", send);
  qs("#chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.isComposing) {
      e.preventDefault();
      send();
    }
  });

  if (!state.agents.length) {
    qs("#loadAgentsForChat").click();
  } else {
    qs("#chatAgent").addEventListener("change", () => {
      const id = qs("#chatAgent").value;
      if (!id) return;
      state.selectedAgentId = id;
      store.set("jl_agentId", id);
      resetChatWs("agent_change");
      updateStatus();
    });
  }

  if (state.agentTemplates.length) {
    refreshTemplateOptions();
  } else {
    loadAgentTemplates()
      .then(() => {
        refreshTemplateOptions();
        refreshChatAgentOptions();
      })
      .catch((e) => {
        out.textContent = `模板加载失败：${String(e)}`;
      });
  }

  refreshChatAgentOptions();
  updateStatus();
}

function bindPatrol() {
  const formOut = qs("#patrolFormOut");
  const consoleOut = qs("#patrolConsoleOut");
  const listEl = qs("#patrolStreamList");
  const emptyEl = qs("#patrolEmpty");
  const totalEl = qs("#patrolTotalCount");
  const searchInput = qs("#patrolSearch");
  const pageInfoEl = qs("#patrolPageInfo");
  const pagePrevBtn = qs("#patrolPagePrev");
  const pageNextBtn = qs("#patrolPageNext");
  const pageSizeSelect = qs("#patrolPageSize");
  const agentSelect = qs("#patrolAgentSelect");
  const streamPicker = qs("#patrolStreamPicker");
  const streamMeta = qs("#patrolStreamMeta");
  const consoleTitle = qs("#patrolConsoleTitle");
  const consoleUrl = qs("#patrolConsoleUrl");
  const consoleMeta = qs("#patrolConsoleMeta");
  const wsStatus = qs("#patrolWsStatus");
  const logEl = qs("#patrolLog");
  const mediaList = qs("#patrolMediaList");
  const previewVideo = qs("#patrolPreviewVideo");
  const previewPlaceholder = qs("#patrolPreviewPlaceholder");
  const previewStatus = qs("#patrolPreviewStatus");
  const previewType = qs("#patrolPreviewType");
  const previewMeta = qs("#patrolPreviewMeta");
  const previewLink = qs("#patrolPreviewLink");
  const previewPlayBtn = qs("#patrolPreviewPlay");
  const previewStopBtn = qs("#patrolPreviewStop");
  const promptText = qs("#patrolPromptText");
  const promptMeta = qs("#patrolPromptMeta");
  const statTotal = qs("#patrolStatTotal");
  const statLow = qs("#patrolStatLow");
  const statMid = qs("#patrolStatMid");
  const statHigh = qs("#patrolStatHigh");

  const nameInput = qs("#patrolStreamName");
  const urlInput = qs("#patrolStreamUrl");
  const intervalInput = qs("#patrolStreamInterval");
  const stopInput = qs("#patrolStreamStop");
  const promptInput = qs("#patrolStreamPrompt");
  const agentSchemaHint = qs("#patrolAgentSchemaHint");

  const streamButton = qs("#patrolAddStream");
  const saveButton = qs("#patrolSaveStream");
  const applyButton = qs("#patrolApplyStream");
  const resetButton = qs("#patrolResetForm");
  const startButton = qs("#patrolStart");
  const stopButton = qs("#patrolStop");
  const clearButton = qs("#patrolClear");
  const syncPromptButton = qs("#patrolSyncAgentPrompt");

  const currentRoute = route();
  const isReviewMode = currentRoute === "patrol-review"
    ? true
    : currentRoute === "patrol"
      ? false
      : state.patrolMode === "review";
  if (currentRoute === "patrol-review" && state.patrolMode !== "review") {
    state.patrolMode = "review";
    store.set("jl_patrolMode", "review");
  } else if (currentRoute === "patrol" && state.patrolMode !== "ws") {
    state.patrolMode = "ws";
    store.set("jl_patrolMode", "ws");
  }
  const targetRoute = isReviewMode ? "patrol-review" : "patrol";

  function storePatrolStreams() {
    store.set("jl_patrolStreams", state.patrolStreams);
  }

  function updateWsStatus() {
    if (!wsStatus) return;
    const agentId = (isReviewMode
      ? (agentSelect?.value || state.patrolReviewAgentId || "未选择")
      : (agentSelect?.value || state.patrolSelectedAgentId || "未选择")
    ).trim();
    if (isReviewMode) {
      wsStatus.textContent = `复判模式 · Agent ${agentId || "未选择"}`;
      return;
    }
    let wsState = "未连接";
    if (patrolWsState.socket) {
      const ready = patrolWsState.socket.readyState;
      if (ready === WebSocket.OPEN) wsState = "在线";
      else if (ready === WebSocket.CONNECTING) wsState = "连接中";
      else wsState = "断开";
    }
    wsStatus.textContent = `WS ${wsState} · Agent ${agentId || "未选择"}`;
  }

  function fillFormFromStream(stream) {
    if (!stream) return;
    if (nameInput) nameInput.value = stream.name || "";
    if (urlInput) urlInput.value = stream.url || "";
    if (intervalInput) intervalInput.value = stream.interval ?? "";
    if (stopInput) stopInput.value = stream.stopMinutes ?? "";
    if (promptInput) promptInput.value = stream.prompt || "";
  }

  function updateAgentMetaView(agentId, stream) {
    if (promptText) {
      const info = resolvePromptInfo(stream, agentId);
      promptText.textContent = info.text || "未配置";
      if (promptMeta) {
        const cached = getCachedAgentMeta(agentId);
        promptMeta.textContent = buildPromptMetaText(info.source, cached?.jsonSchema);
      }
    }
    if (agentSchemaHint) {
      const cached = getCachedAgentMeta(agentId);
      agentSchemaHint.textContent = formatSchemaHint(cached?.jsonSchema);
    }
    if (agentId && !getCachedAgentMeta(agentId)) {
      ensureAgentMeta(agentId)
        .then(() => {
          const info = resolvePromptInfo(stream, agentId);
          if (promptText) promptText.textContent = info.text || "未配置";
          if (promptMeta) {
            const cached = getCachedAgentMeta(agentId);
            promptMeta.textContent = buildPromptMetaText(info.source, cached?.jsonSchema);
          }
          if (agentSchemaHint) {
            const cached = getCachedAgentMeta(agentId);
            agentSchemaHint.textContent = formatSchemaHint(cached?.jsonSchema);
          }
        })
        .catch(() => {
          // ignore
        });
    }
  }

  function normalizePromptText(input) {
    return String(input || "")
      .replace(/\r\n/g, "\n")
      .replace(/[ \t]+$/gm, "")
      .trim();
  }

  async function autoSyncPromptForAgentChange(oldAgentId, newAgentId, stream) {
    if (!stream || !oldAgentId || !newAgentId) return;
    if (oldAgentId === newAgentId) return;
    try {
      const newMeta = await ensureAgentMeta(newAgentId);
      const newPrompt = normalizePromptText(newMeta?.prompt);
      if (!newPrompt) return;
      if (agentSelect && agentSelect.value !== newAgentId) return;
      stream.prompt = newPrompt;
      stream.updatedAt = new Date().toISOString();
      storePatrolStreams();
      if (promptInput) promptInput.value = newPrompt;
      updateAgentMetaView(newAgentId, stream);
    } catch {
      // ignore
    }
  }

  function updateConsoleInfo(stream) {
    if (consoleTitle && consoleUrl && consoleMeta) {
      if (!stream) {
        consoleTitle.textContent = "未选择";
        consoleUrl.textContent = "—";
        consoleMeta.textContent = "选择一个巡检流后即可查看输出。";
      } else {
        consoleTitle.textContent = stream.name || "未命名";
        consoleUrl.textContent = stream.url || "—";
        const agentId = isReviewMode
          ? (agentSelect?.value || state.patrolReviewAgentId || "未选择")
          : (agentSelect?.value || state.patrolSelectedAgentId || "未选择");
        const modeLabel = isReviewMode ? "复判轮询" : (isLiveStream(stream.url) ? "实时流" : "离线分析");
        consoleMeta.textContent = `Agent ${agentId || "未选择"} · ${modeLabel}`;
      }
    }

    if (streamPicker) {
      streamPicker.value = stream?.id || "";
    }

    if (streamMeta) {
      if (!stream) {
        streamMeta.innerHTML = '<span class="tag ghost">未选择</span>';
      } else {
        const tags = [];
        tags.push(isLiveStream(stream.url) ? '<span class="tag accent">LIVE</span>' : '<span class="tag ghost">OFFLINE</span>');
        if (stream.interval) tags.push(`<span class="tag">间隔 ${stream.interval}s</span>`);
        if (isReviewMode && !stream.interval) {
          tags.push(`<span class="tag">间隔 ${getReviewIntervalSeconds(stream)}s</span>`);
        }
        if (stream.stopMinutes) tags.push(`<span class="tag ghost">停止 ${stream.stopMinutes}m</span>`);
        if (stream.prompt) tags.push('<span class="tag primary">有提示词</span>');
        streamMeta.innerHTML = tags.join("") || '<span class="tag ghost">无配置</span>';
      }
    }

    const promptAgentId = isReviewMode
      ? (agentSelect?.value || state.patrolReviewAgentId || "")
      : (agentSelect?.value || state.patrolSelectedAgentId || "");
    updateAgentMetaView(promptAgentId, stream);

    updatePreviewMeta(stream);
  }

  function normalizeMediaUrl(input) {
    const raw = String(input || "").trim();
    if (!raw) return "";
    if (raw.startsWith("/")) return resolveAssetUrl(raw);
    try {
      const parsed = new URL(raw);
      const host = parsed.hostname || "";
      const isPrivate = /^(127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[0-1])\.)/.test(host);
      if (isPrivate && state.baseUrl) {
        const base = new URL(state.baseUrl, location.origin);
        parsed.protocol = base.protocol;
        parsed.host = base.host;
        return parsed.toString();
      }
    } catch {
      // ignore
    }
    return raw;
  }

  function isVideoMediaUrl(url) {
    const normalized = String(url || "").trim().toLowerCase();
    if (!normalized) return false;
    if (/\.(mp4|webm|ogg)(\?|$)/.test(normalized)) return true;
    if (isHlsManifest(normalized)) return false;
    return false;
  }

  function replaceMediaBlocks(raw) {
    return String(raw || "").replace(/:::(image|video)\s*```json\s*([\s\S]*?)\s*```\s*:::/g, (match, kind, url) => {
      const cleaned = String(url || "").trim();
      if (!cleaned) return "";
      if (kind === "image") return `![证据帧](${cleaned})`;
      return `[视频片段](${cleaned})`;
    });
  }

  function isMarkdownTableDivider(line) {
    const s = String(line || "").trim();
    if (!s) return false;
    return /^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$/.test(s);
  }

  function splitMarkdownTableRow(line) {
    let s = String(line || "").trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|")) s = s.slice(0, -1);
    return s.split("|").map((cell) => cell.trim());
  }

  function renderInlineMarkdown(text) {
    const raw = String(text || "");
    const regex = /(!)?\[([^\]]*)\]\(([^)]+)\)/g;
    let out = "";
    let last = 0;
    let match = null;
    while ((match = regex.exec(raw)) !== null) {
      out += escapeHtml(raw.slice(last, match.index));
      const isImage = Boolean(match[1]);
      const label = (match[2] || "").trim();
      const url = normalizeMediaUrl(match[3]);
      if (url) {
        if (isImage) {
          out += `<img class="patrol-md-image" src="${escapeHtml(url)}" alt="${escapeHtml(label || "image")}">`;
        } else if (isVideoMediaUrl(url)) {
          out += `<video class="patrol-md-video" src="${escapeHtml(url)}" controls preload="metadata"></video>`;
        } else {
          const textLabel = label || url;
          out += `<a class="patrol-md-link" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(textLabel)}</a>`;
        }
      }
      last = regex.lastIndex;
    }
    out += escapeHtml(raw.slice(last));
    return out;
  }

  function renderPatrolMarkdown(raw) {
    const cleaned = replaceMediaBlocks(raw || "");
    if (!cleaned) return "";
    const lines = cleaned.split(/\r?\n/);
    const parts = [];
    let buffer = [];
    const flushText = () => {
      if (!buffer.length) return;
      const htmlLines = buffer.map((line) => renderInlineMarkdown(line));
      parts.push(`<div class="patrol-md-text">${htmlLines.join("<br>")}</div>`);
      buffer = [];
    };

    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      if (line && line.includes("|") && i + 1 < lines.length && isMarkdownTableDivider(lines[i + 1])) {
        const header = splitMarkdownTableRow(line);
        const rows = [];
        i += 2;
        while (i < lines.length) {
          const rowLine = lines[i];
          if (!rowLine || !rowLine.includes("|")) break;
          if (!isMarkdownTableDivider(rowLine)) {
            rows.push(splitMarkdownTableRow(rowLine));
          }
          i += 1;
        }
        i -= 1;
        flushText();
        const colCount = Math.max(header.length, ...rows.map((r) => r.length), 1);
        const padRow = (row) => row.concat(Array(colCount - row.length).fill(""));
        const headHtml = padRow(header)
          .map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`)
          .join("");
        const bodyHtml = rows
          .map((row) => `<tr>${padRow(row).map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`)
          .join("");
        parts.push(`<div class="patrol-md-table"><table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`);
        continue;
      }
      buffer.push(line);
    }
    flushText();
    return parts.join("");
  }

  function appendPatrolEntry(label, content = "", className = "", options = {}) {
    if (!logEl) return null;
    const skipStats = options?.skipStats === true;
    const entry = document.createElement("div");
    entry.className = `patrol-log-entry ${className}`.trim();
    const meta = document.createElement("div");
    meta.className = "patrol-log-meta";
    const metaLabel = document.createElement("span");
    metaLabel.textContent = label;
    const metaTime = document.createElement("span");
    metaTime.className = "mono";
    metaTime.textContent = new Date().toLocaleTimeString();
    meta.appendChild(metaLabel);
    meta.appendChild(metaTime);
    const body = document.createElement("div");
    body.className = "patrol-log-text";
    body.innerHTML = renderPatrolMarkdown(String(content || ""));
    entry.appendChild(meta);
    entry.appendChild(body);
    logEl.appendChild(entry);
    logEl.scrollTop = logEl.scrollHeight;
    if (!skipStats) {
      patrolRuntime.stats.total += 1;
      if (className.includes("error")) {
        patrolRuntime.stats.high += 1;
      } else if (className.includes("warn")) {
        patrolRuntime.stats.mid += 1;
      } else {
        patrolRuntime.stats.low += 1;
      }
      updatePatrolStats();
    }
    return body;
  }

  function resetPatrolMedia() {
    patrolRuntime.mediaItems = [];
    patrolRuntime.mediaSeen = new Set();
    patrolRuntime.mediaBuffer = "";
    patrolRuntime.stats.media = 0;
    updatePatrolStats();
    if (mediaList) mediaList.innerHTML = "";
  }

  function addMediaItem(type, url, options = {}) {
    const resolvedUrl = normalizeMediaUrl(url);
    if (!resolvedUrl || patrolRuntime.mediaSeen.has(resolvedUrl)) return;
    patrolRuntime.mediaSeen.add(resolvedUrl);
    patrolRuntime.mediaItems.push({ type, url: resolvedUrl, timestamp: Date.now() });
    patrolRuntime.stats.media += 1;
    updatePatrolStats();

    if (!options?.skipLog) {
      const label = type === "video" ? "视频片段" : "证据帧";
      const text = type === "video"
        ? `[视频片段](${resolvedUrl})`
        : `![证据帧](${resolvedUrl})`;
      appendPatrolEntry(label, text, "media", { skipStats: true });
    }

    if (!mediaList || options?.skipPanel) return;
    const card = document.createElement("div");
    card.className = `patrol-media-card ${type}`;
    const meta = document.createElement("div");
    meta.className = "patrol-media-meta";
    const label = document.createElement("span");
    label.textContent = type === "video" ? "视频片段" : "证据帧";
    const time = document.createElement("span");
    time.className = "mono";
    time.textContent = new Date().toLocaleTimeString();
    const link = document.createElement("a");
    link.href = resolvedUrl;
    link.textContent = "打开";
    link.target = "_blank";
    link.rel = "noreferrer";
    meta.appendChild(label);
    meta.appendChild(time);
    meta.appendChild(link);

    const media = type === "video" ? document.createElement("video") : document.createElement("img");
    if (type === "video") {
      media.controls = true;
      media.src = resolvedUrl;
    } else {
      media.src = resolvedUrl;
      media.loading = "lazy";
    }
    card.appendChild(meta);
    card.appendChild(media);
    mediaList.appendChild(card);
  }

  function setPreviewStatus(text) {
    if (!previewStatus) return;
    previewStatus.textContent = text || "";
  }

  function setPreviewType(text) {
    if (!previewType) return;
    previewType.textContent = text || "HLS";
  }

  function togglePreviewPlaceholder(show) {
    if (!previewPlaceholder) return;
    previewPlaceholder.style.display = show ? "flex" : "none";
  }

  function setPreviewLink(url) {
    if (!previewLink) return;
    if (url) {
      previewLink.href = url;
      previewLink.style.display = "inline-flex";
    } else {
      previewLink.removeAttribute("href");
      previewLink.style.display = "none";
    }
  }

  function resetPreviewRetry() {
    if (patrolRuntime.previewRetryTimer) {
      clearTimeout(patrolRuntime.previewRetryTimer);
      patrolRuntime.previewRetryTimer = null;
    }
    patrolRuntime.previewRetryCount = 0;
  }

  function schedulePreviewRetry(reason) {
    if (!patrolRuntime.previewUrl) return;
    if (patrolRuntime.previewRetryCount >= 5) {
      setPreviewStatus("播放失败");
      appendPatrolEntry("预览", `播放失败：${reason}，已停止重试。`, "error");
      return;
    }
    patrolRuntime.previewRetryCount += 1;
    const delayMs = Math.min(8000, 1500 * patrolRuntime.previewRetryCount);
    setPreviewStatus(`重连中(${patrolRuntime.previewRetryCount})`);
    if (patrolRuntime.previewRetryTimer) clearTimeout(patrolRuntime.previewRetryTimer);
    patrolRuntime.previewRetryTimer = setTimeout(() => {
      attachPreviewSource(patrolRuntime.previewUrl, patrolRuntime.previewSourceType)
        .catch((e) => {
          appendPatrolEntry("预览", `重连失败：${String(e)}`, "warn");
          schedulePreviewRetry("重连失败");
        });
    }, delayMs);
    appendPatrolEntry("预览", `播放中断，${Math.round(delayMs / 1000)}s 后重连（${reason}）`, "warn");
  }

  function updatePreviewMeta(stream) {
    if (!previewMeta) return;
    if (!stream) {
      previewMeta.textContent = "选择巡检流后点击播放。";
      return;
    }
    const name = stream.name || "未命名";
    if (patrolRuntime.previewStreamId === stream.id) {
      previewMeta.textContent = `${name} · 预览中`;
      return;
    }
    previewMeta.textContent = `${name} · 待播放`;
  }

  function updatePatrolStats() {
    if (statTotal) statTotal.textContent = String(patrolRuntime.stats.total || 0);
    if (statLow) statLow.textContent = String(patrolRuntime.stats.low || 0);
    if (statMid) statMid.textContent = String(patrolRuntime.stats.mid || 0);
    if (statHigh) statHigh.textContent = String(patrolRuntime.stats.high || 0);
  }

  function resetPatrolStats() {
    patrolRuntime.stats = {
      total: 0,
      low: 0,
      mid: 0,
      high: 0,
      media: 0,
    };
    updatePatrolStats();
  }

  function resetPreviewVideo() {
    if (!previewVideo) return;
    try {
      previewVideo.pause();
    } catch {
      // ignore
    }
    previewVideo.removeAttribute("src");
    previewVideo.load();
  }

  async function attachPreviewSource(url, sourceType) {
    if (!previewVideo) return;
    destroyPatrolHls();
    resetPreviewVideo();
    if (!url) return;
    const useHls = sourceType === "proxy" || sourceType === "hls";
    if (useHls) {
      if (previewVideo.canPlayType("application/vnd.apple.mpegurl")) {
        previewVideo.src = url;
      } else if (window.Hls && window.Hls.isSupported()) {
        const hls = new window.Hls({ lowLatencyMode: true });
        hls.on(window.Hls.Events.ERROR, (_event, data) => {
          if (!data || !data.fatal) return;
          if (data.type === window.Hls.ErrorTypes.NETWORK_ERROR) {
            hls.startLoad();
            schedulePreviewRetry("网络错误");
            return;
          }
          if (data.type === window.Hls.ErrorTypes.MEDIA_ERROR) {
            hls.recoverMediaError();
            schedulePreviewRetry("媒体错误");
            return;
          }
          hls.destroy();
          schedulePreviewRetry("播放异常");
        });
        hls.loadSource(url);
        hls.attachMedia(previewVideo);
        patrolRuntime.hls = hls;
      } else {
        throw new Error("当前浏览器不支持 HLS 播放");
      }
    } else {
      previewVideo.src = url;
    }
    try {
      await previewVideo.play();
    } catch {
      // ignore autoplay failures
    }
  }

  async function startPreview(stream) {
    if (!stream) {
      setPreviewStatus("请先选择巡检流");
      toast("未选择巡检流", "请先选择一个巡检流");
      return;
    }
    const rawUrl = (stream.url || "").trim();
    if (!rawUrl) {
      setPreviewStatus("流地址为空");
      toast("流地址为空", "请先设置流地址");
      return;
    }

    if (patrolRuntime.previewStreamId && patrolRuntime.previewStreamId !== stream.id) {
      await stopPreview(true);
    }

    resetPreviewRetry();
    setPreviewStatus("准备中…");
    togglePreviewPlaceholder(false);
    let sourceType = "direct";
    let sourceUrl = rawUrl;
    try {
      if (isLiveStream(rawUrl)) {
        const payload = await requestHlsStart(rawUrl, stream.id);
        const data = payload?.data || payload?.result || payload || {};
        if (payload?.success === false) {
          throw new Error(payload.message || "HLS 启动失败");
        }
        sourceUrl = data.playlist_url || data.m3u8_url || data.url || "";
        if (!sourceUrl) {
          throw new Error("HLS 地址为空");
        }
        patrolRuntime.previewHlsStreamId = data.stream_id || stream.id;
        sourceType = "proxy";
      } else if (isHlsManifest(rawUrl)) {
        sourceType = "hls";
      } else if (isHttpUrl(rawUrl)) {
        sourceType = "direct";
      } else {
        throw new Error("暂不支持该流地址");
      }

      const resolvedUrl = resolveAssetUrl(sourceUrl);
      setPreviewType(sourceType === "direct" ? "Direct" : "HLS");
      await attachPreviewSource(resolvedUrl, sourceType);
      patrolRuntime.previewStreamId = stream.id;
      patrolRuntime.previewSourceType = sourceType;
      patrolRuntime.previewUrl = resolvedUrl;
      setPreviewLink(resolvedUrl);
      updatePreviewMeta(stream);
      setPreviewStatus("播放中");
      togglePreviewPlaceholder(false);
      renderStreamList();
    } catch (e) {
      togglePreviewPlaceholder(true);
      setPreviewLink("");
      setPreviewStatus("播放失败");
      updatePreviewMeta(stream);
      toast("预览失败", String(e));
    }
  }

  async function stopPreview(stopRemote = true) {
    resetPreviewVideo();
    await cleanupPatrolPreview(stopRemote);
    togglePreviewPlaceholder(true);
    setPreviewLink("");
    setPreviewStatus("已停止");
    setPreviewType("HLS");
    updatePreviewMeta(
      state.patrolStreams.find((s) => s.id === state.patrolSelectedStreamId) || null
    );
    renderStreamList();
  }

  function captureMediaChunks(chunk) {
    patrolRuntime.mediaBuffer += chunk;
    const pattern = /:::(image|video)\\s*```json\\s*([\\s\\S]*?)\\s*```\\s*:::/g;
    let match = null;
    let lastIndex = 0;
    while ((match = pattern.exec(patrolRuntime.mediaBuffer)) !== null) {
      const type = match[1];
      const url = (match[2] || "").trim();
      if (url) addMediaItem(type, url, { skipLog: true });
      lastIndex = pattern.lastIndex;
    }
    if (lastIndex > 0) {
      patrolRuntime.mediaBuffer = patrolRuntime.mediaBuffer.slice(lastIndex);
    } else if (patrolRuntime.mediaBuffer.length > 8000) {
      patrolRuntime.mediaBuffer = patrolRuntime.mediaBuffer.slice(-4000);
    }
  }

  function renderStreamPicker() {
    if (!streamPicker) return;
    streamPicker.innerHTML = buildPatrolStreamOptions(state.patrolSelectedStreamId);
    if (state.patrolSelectedStreamId) {
      streamPicker.value = state.patrolSelectedStreamId;
    }
  }

  function renderStreamList() {
    if (!listEl) {
      renderStreamPicker();
      return;
    }
    const selectedId = state.patrolSelectedStreamId;
    const filtered = filterPatrolStreams(state.patrolStreams, state.patrolSearch);
    const total = filtered.length;
    const rawTotal = (state.patrolStreams || []).length;
    const pageSize = normalizePatrolPageSize(state.patrolPageSize);
    const totalPages = total ? Math.ceil(total / pageSize) : 1;
    let pageIndex = toNonNegativeInt(state.patrolPageIndex, 0);
    if (pageIndex > totalPages - 1) pageIndex = totalPages - 1;
    if (pageIndex < 0) pageIndex = 0;
    state.patrolPageIndex = pageIndex;
    store.set("jl_patrolPageIndex", pageIndex);
    state.patrolPageSize = pageSize;
    store.set("jl_patrolPageSize", pageSize);

    const start = pageIndex * pageSize;
    const end = Math.min(start + pageSize, total);
    const pageItems = filtered.slice(start, end);
    listEl.innerHTML = buildPatrolStreamCards(
      pageItems,
      selectedId,
      patrolRuntime.runningStreamId,
      patrolRuntime.previewStreamId,
      ""
    );

    if (totalEl) {
      totalEl.textContent = rawTotal && rawTotal !== total ? `${total}/${rawTotal}` : String(total);
    }
    if (emptyEl) {
      if (!total) {
        emptyEl.textContent = rawTotal
          ? "暂无可达巡检流（已过滤不可达）"
          : "暂无巡检流配置，先新增一个。";
        emptyEl.style.display = "block";
      } else {
        emptyEl.style.display = "none";
      }
    }
    if (pageInfoEl) {
      pageInfoEl.textContent = total ? `${start + 1}-${end} / ${total}` : "0 / 0";
    }
    if (pagePrevBtn) pagePrevBtn.disabled = pageIndex <= 0;
    if (pageNextBtn) pageNextBtn.disabled = pageIndex >= totalPages - 1;
    if (pageSizeSelect) pageSizeSelect.value = String(pageSize);
    renderStreamPicker();
  }

  async function probePatrolStreams(force = false, targets = null) {
    if (patrolRuntime.streamCheckInFlight) return;
    const now = Date.now();
    if (!force && patrolRuntime.streamCheckAt && now - patrolRuntime.streamCheckAt < PATROL_STREAM_CHECK_TTL_MS) {
      return;
    }
    const candidates = Array.isArray(targets) ? targets : state.patrolStreams;
    const urls = (candidates || [])
      .map((stream) => (stream?.url || "").trim())
      .filter(Boolean);
    if (!urls.length) return;

    patrolRuntime.streamCheckInFlight = true;
    try {
      const result = await requestPatrolStreamCheck(urls, { timeoutMs: PATROL_STREAM_CHECK_TIMEOUT_MS });
      const items = Array.isArray(result?.items) ? result.items : (Array.isArray(result) ? result : []);
      const map = new Map(items.map((item) => [item.url, item.reachable]));
      state.patrolStreams.forEach((stream) => {
        const url = (stream?.url || "").trim();
        if (!url || !map.has(url)) return;
        stream.reachable = map.get(url) === true;
      });
      let hidden = 0;
      state.patrolStreams.forEach((stream) => {
        if (stream?.reachable === false) hidden += 1;
      });
      patrolRuntime.streamCheckHidden = hidden;
      patrolRuntime.streamCheckAt = Date.now();
      if (force) {
        if (hidden) toast("已隐藏不可达巡检流", `${hidden} 个`);
        else toast("巡检流可达检测完成", "全部可达");
      }
      renderStreamList();
      syncSelectedStream();
    } catch (e) {
      if (force) toast("巡检流检测失败", String(e));
    } finally {
      patrolRuntime.streamCheckInFlight = false;
    }
  }

  function setSelectedStream(streamId) {
    const next = state.patrolStreams.find((stream) => stream.id === streamId);
    if (!next) return;
    state.patrolSelectedStreamId = streamId;
    store.set("jl_patrolStreamId", streamId);
    if (route() === "patrol-streams") {
      fillFormFromStream(next);
    }
    updateConsoleInfo(next);
    renderStreamList();
  }

  function syncSelectedStream() {
    const available = getReachablePatrolStreams(state.patrolStreams);
    if (!available.length) {
      state.patrolSelectedStreamId = "";
      updateConsoleInfo(null);
      renderStreamList();
      return;
    }
    const current = available.find((stream) => stream.id === state.patrolSelectedStreamId);
    const fallback = current ? current.id : available[0].id;
    setSelectedStream(fallback);
  }

  function refreshAgentOptions() {
    if (!agentSelect) return;
    const agents = isReviewMode ? getReviewAgents() : getPatrolAgents();
    const selectedId = isReviewMode ? state.patrolReviewAgentId : state.patrolSelectedAgentId;
    const emptyLabel = isReviewMode ? "（暂无复判智能体）" : "（暂无巡检智能体）";
    agentSelect.innerHTML = buildAgentOptionsFromList(agents, selectedId, emptyLabel);
    const hasSelected = agents.some((a) => getAgentId(a) === selectedId);
    if (!hasSelected) {
      const nextId = getAgentId(agents[0]) || "";
      if (isReviewMode) {
        state.patrolReviewAgentId = nextId;
        store.set("jl_patrolReviewAgentId", nextId);
      } else {
        state.patrolSelectedAgentId = nextId;
        store.set("jl_patrolAgentId", nextId);
      }
    }
    agentSelect.value = isReviewMode ? state.patrolReviewAgentId || "" : state.patrolSelectedAgentId || "";
    updateWsStatus();
    updateConsoleInfo(
      state.patrolStreams.find((stream) => stream.id === state.patrolSelectedStreamId) || null
    );
  }

  function resolveReviewAgentId() {
    const agents = getReviewAgents();
    const current = (agentSelect?.value || state.patrolReviewAgentId || "").trim();
    if (current && agents.some((agent) => getAgentId(agent) === current)) return current;
    const fallback = getAgentId(agents[0]) || "";
    if (fallback && current !== fallback) {
      state.patrolReviewAgentId = fallback;
      store.set("jl_patrolReviewAgentId", fallback);
      if (agentSelect) agentSelect.value = fallback;
    }
    return fallback;
  }

  function runPendingAction() {
    if (route() !== targetRoute) return;
    const action = patrolRuntime.pendingAction;
    const streamId = patrolRuntime.pendingStreamId;
    if (!action || !streamId) return;
    patrolRuntime.pendingAction = "";
    patrolRuntime.pendingStreamId = "";
    setSelectedStream(streamId);
    const stream = state.patrolStreams.find((s) => s.id === streamId) || null;
    if (action === "start") {
      if (isReviewMode) startReviewStream(stream);
      else startStream(stream);
    }
    if (action === "stop") {
      if (isReviewMode) stopReviewStream();
      else stopStream();
    }
    if (action === "preview") startPreview(stream);
  }

  function buildStartCommand(stream) {
    const interval = stream.interval ? ` 轮询间隔=${stream.interval}秒` : "";
    const stop = stream.stopMinutes ? ` 停止时间=${stream.stopMinutes}分钟` : "";
    return `启动巡检 ${stream.url}${interval}${stop}`;
  }

function buildReviewCommand(stream) {
    const intervalSeconds = getReviewIntervalSeconds(stream);
    const interval = intervalSeconds ? ` 轮询间隔=${intervalSeconds}秒` : "";
    return `复判巡检 ${stream.url}${interval}`;
  }

  function guessReviewMediaType(url) {
    const normalized = (url || "").trim().toLowerCase();
    if (/\.(png|jpe?g|webp|gif)(\?|$)/.test(normalized)) return "image";
    if (/\.(mp4|mov|mkv|avi)(\?|$)/.test(normalized)) return "video";
    if (isHlsManifest(normalized) || isLiveStream(normalized)) return "video";
    return "video";
  }

  function buildReviewMessage(stream, agentMeta) {
    const prompt = (stream?.prompt || "").trim();
    if (prompt) return prompt;
    const agentPrompt = String(agentMeta?.prompt || "").trim();
    if (agentPrompt) return agentPrompt;
    const name = (stream?.name || "巡检流").trim();
    return `请分析${name}当前画面，输出简要结论。`;
  }

  function getReviewIntervalMs(stream) {
    const seconds = getReviewIntervalSeconds(stream);
    return Math.max(5, seconds) * 1000;
  }

  function getReviewIntervalSeconds(stream) {
    const raw = Number(stream?.interval);
    if (Number.isFinite(raw) && raw > 0) return raw;
    return REVIEW_DEFAULT_INTERVAL_SECONDS;
  }

  function getReviewCaptureSettings() {
    const captureSeconds = Math.max(4, REVIEW_CAPTURE_SECONDS);
    const sampleFps = Math.max(0.5, REVIEW_SAMPLE_FPS);
    const maxFrames = Math.max(4, Math.round(sampleFps * Math.max(1, captureSeconds - 2)));
    return { captureSeconds, sampleFps, maxFrames };
  }

  function summarizeReviewResult(result) {
    if (!result) return "无响应";
    if (typeof result.summary_sentence === "string" && result.summary_sentence.trim()) {
      return result.summary_sentence.trim();
    }
    if (typeof result.raw_response === "string" && result.raw_response.trim()) {
      return result.raw_response.trim();
    }
    if (result.data && typeof result.data === "object") {
      try {
        return JSON.stringify(result.data, null, 2);
      } catch {
        return String(result.data);
      }
    }
    if (result.result && typeof result.result === "object") {
      try {
        return JSON.stringify(result.result, null, 2);
      } catch {
        return String(result.result);
      }
    }
    if (typeof result === "string") return result;
    try {
      return JSON.stringify(result, null, 2);
    } catch {
      return String(result);
    }
  }

  function isReviewHit(result) {
    if (!result || typeof result !== "object") return false;
    if (Number(result.hit) === 1) return true;
    const illegal = result.data?.illegal;
    if (illegal === true || illegal === "true") return true;
    const hit = result.data?.hit ?? result.result?.hit;
    return Number(hit) === 1;
  }

  function scheduleReviewNext(stream, agentId, delayMs) {
    if (!patrolRuntime.reviewRunning || patrolRuntime.reviewStopRequested) return;
    if (patrolRuntime.reviewTimer) clearTimeout(patrolRuntime.reviewTimer);
    patrolRuntime.reviewTimer = setTimeout(() => {
      runReviewOnce(stream, agentId);
    }, delayMs);
  }

  async function fetchReviewEvidence(detailPath) {
    if (!detailPath) return null;
    try {
      const record = await apiFetchStandard(detailPath);
      if (!record || typeof record !== "object") return null;
      const imageUrl = record?.image?.static_url || record?.image?.url || "";
      const thumbUrl = record?.video?.thumb_static_url || "";
      const videoUrl = record?.video?.static_url || "";
      return {
        imageUrl: String(imageUrl || ""),
        thumbUrl: String(thumbUrl || ""),
        videoUrl: String(videoUrl || ""),
      };
    } catch (e) {
      return null;
    }
  }

  async function runReviewOnce(stream, agentId) {
    if (!patrolRuntime.reviewRunning || patrolRuntime.reviewStopRequested) return;
    if (patrolRuntime.reviewInFlight) return;
    patrolRuntime.reviewInFlight = true;
    const controller = new AbortController();
    patrolRuntime.reviewAbortController = controller;
    const startedAt = Date.now();
    if (consoleOut) consoleOut.textContent = "复判中...";
    try {
      const url = (stream?.url || "").trim();
      const capture = getReviewCaptureSettings();
      let reviewUrl = url;
      let clipUrl = "";
      if ((isLiveStream(url) || isHlsManifest(url)) && url) {
        try {
          if (consoleOut) consoleOut.textContent = "抓取视频片段中...";
          const clip = await requestPatrolClip(url, {
            clipSeconds: capture.captureSeconds,
            streamId: stream?.id || "",
          });
          const clipPath = (clip && clip.file_path) ? String(clip.file_path) : "";
          clipUrl = (clip && clip.url) ? String(clip.url) : "";
          if (clipPath) {
            reviewUrl = clipPath;
          } else if (clipUrl) {
            reviewUrl = resolveAssetUrl(clipUrl);
          }
        } catch (e) {
          appendPatrolEntry("警告", `抓取片段失败，改用原始流：${String(e)}`, "warn");
          if (consoleOut) consoleOut.textContent = "复判中...";
        }
      }
      const agentMeta = await ensureAgentMeta(agentId).catch(() => null);
      const payload = {
        message: buildReviewMessage(stream, agentMeta),
        context: {
          files: [{ url: reviewUrl, media_type: guessReviewMediaType(reviewUrl) }],
          output_format: "json",
          video_max_frames: capture.maxFrames,
          video_sample_fps: capture.sampleFps,
          json_schema: agentMeta?.jsonSchema || undefined,
        },
      };
      const result = await apiFetchStandard(
        `/api/v1/agents/${encodeURIComponent(agentId)}/chat/json`,
        {
          method: "POST",
          body: JSON.stringify(payload),
          signal: controller.signal,
        }
      );
      const recordDetailPath = result?.meta?.review_record_detail
        || (result?.meta?.review_record_id ? `/api/v1/review-records/${result.meta.review_record_id}` : "");
      const evidenceEntry = appendPatrolEntry("证据帧", "证据生成中...", "media", { skipStats: true });
      const updateEvidenceEntry = (text) => {
        if (!evidenceEntry) return;
        evidenceEntry.innerHTML = renderPatrolMarkdown(String(text || ""));
      };
      const pickEvidenceText = (evidence) => {
        const imageUrl = (evidence?.imageUrl || evidence?.thumbUrl || "").trim();
        const videoUrl = (evidence?.videoUrl || clipUrl || "").trim();
        const parts = [];
        if (imageUrl) parts.push(`![证据帧](${imageUrl})`);
        if (videoUrl) parts.push(`[视频片段](${videoUrl})`);
        return parts.join("\n");
      };

      let evidence = await fetchReviewEvidence(recordDetailPath);
      let evidenceText = pickEvidenceText(evidence);
      if (!evidenceText) {
        evidenceText = pickEvidenceText({ videoUrl: clipUrl });
      }
      if (evidenceText) {
        updateEvidenceEntry(evidenceText);
      } else if (recordDetailPath) {
        setTimeout(async () => {
          const later = await fetchReviewEvidence(recordDetailPath);
          const laterText = pickEvidenceText(later);
          if (laterText) updateEvidenceEntry(laterText);
        }, 1500);
      } else {
        updateEvidenceEntry("暂无证据帧");
      }
      const summary = summarizeReviewResult(result);
      const className = result?.success === false ? "error" : (isReviewHit(result) ? "warn" : "");
      appendPatrolEntry("复判", summary, className);
      if (consoleOut) consoleOut.textContent = summary;
      patrolRuntime.reviewLastRunAt = Date.now();
    } catch (e) {
      if (e?.name !== "AbortError") {
        appendPatrolEntry("错误", `复判失败：${String(e)}`, "error");
        if (consoleOut) consoleOut.textContent = String(e);
      }
    } finally {
      patrolRuntime.reviewInFlight = false;
      patrolRuntime.reviewAbortController = null;
      if (patrolRuntime.reviewRunning && !patrolRuntime.reviewStopRequested) {
        const elapsed = Date.now() - startedAt;
        const delayMs = Math.max(1000, getReviewIntervalMs(stream) - elapsed);
        scheduleReviewNext(stream, agentId, delayMs);
      }
    }
  }

  function startReviewStream(stream) {
    if (!stream) {
      if (consoleOut) consoleOut.textContent = "请先选择一个巡检流";
      toast("未选择巡检流", "请先选择一个巡检流");
      return;
    }
    const agentId = resolveReviewAgentId();
    if (!agentId) {
      if (consoleOut) consoleOut.textContent = "未选择复判智能体";
      toast("未选择复判智能体", "请先选择复判智能体");
      return;
    }
    const rawUrl = (stream.url || "").trim();
    if (!rawUrl) {
      if (consoleOut) consoleOut.textContent = "流地址为空";
      toast("流地址为空", "请先设置流地址");
      return;
    }

    if (patrolRuntime.reviewRunning && patrolRuntime.runningStreamId && patrolRuntime.runningStreamId !== stream.id) {
      stopReviewStream(true);
      appendPatrolEntry("提示", "检测到已有运行中的复判巡检流，将切换到当前流。", "warn");
    } else if (patrolRuntime.reviewRunning && patrolRuntime.runningStreamId === stream.id) {
      appendPatrolEntry("提示", "当前巡检流已在复判中。", "warn");
      return;
    }

    state.patrolSelectedStreamId = stream.id;
    store.set("jl_patrolStreamId", stream.id);

    patrolRuntime.reviewRunning = true;
    patrolRuntime.reviewStopRequested = false;
    patrolRuntime.runningStreamId = stream.id;
    renderStreamList();
    updateConsoleInfo(stream);
    updateWsStatus();
    appendPatrolEntry("指令", buildReviewCommand(stream), "cmd");

    runReviewOnce(stream, agentId);
  }

  function stopReviewStream(silent = false) {
    if (!patrolRuntime.reviewRunning && !patrolRuntime.reviewInFlight) return;
    patrolRuntime.reviewRunning = false;
    patrolRuntime.reviewStopRequested = true;
    if (patrolRuntime.reviewTimer) {
      clearTimeout(patrolRuntime.reviewTimer);
      patrolRuntime.reviewTimer = null;
    }
    if (patrolRuntime.reviewAbortController) {
      try {
        patrolRuntime.reviewAbortController.abort();
      } catch {
        // ignore
      }
      patrolRuntime.reviewAbortController = null;
    }
    patrolRuntime.reviewInFlight = false;
    patrolRuntime.runningStreamId = "";
    if (!silent) {
      appendPatrolEntry("指令", "停止复判巡检", "cmd");
    }
    renderStreamList();
    updateWsStatus();
  }

  function sendPatrolWs(payload) {
    const socket = patrolWsState.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket 未连接");
    }
    socket.send(JSON.stringify(payload));
  }

  async function requestPatrolInit() {
    if (!patrolWsState.socket || patrolWsState.initRequestId) return;
    patrolWsState.initError = "";
    patrolWsState.initRequestId = rpcId("patrol_init");
    const availableTools = await getAvailableExternalTools();
    try {
      sendPatrolWs({
        jsonrpc: "2.0",
        id: patrolWsState.initRequestId,
        method: "session.initialize",
        params: {
          protocolVersion: "2.0",
          userContext: { userId: "patrol" },
          availableTools,
        },
      });
    } catch (e) {
      patrolWsState.initRequestId = "";
      patrolWsState.initError = String(e);
      if (consoleOut) consoleOut.textContent = patrolWsState.initError;
    }
  }

  function rejectExternalTool(executionId, toolName = "") {
    if (!executionId) return;
    try {
      sendPatrolWs({
        jsonrpc: "2.0",
        id: executionId,
        error: {
          code: -32000,
          message: `巡检控制台未实现外部工具执行${toolName ? `: ${toolName}` : ""}`,
        },
      });
    } catch {
      // ignore
    }
  }

  function handlePatrolWsMessage(data) {
    if (!data || typeof data !== "object") return;

    if (data.type === "connected") {
      patrolWsState.sessionId = data.session_id || "";
      updateWsStatus();
      return;
    }

    if (data.id && data.id === patrolWsState.initRequestId) {
      patrolWsState.initDone = !data.error;
      if (data.error) {
        patrolWsState.initError = normalizeWsError(data.error?.message) || "初始化失败";
        if (consoleOut) consoleOut.textContent = patrolWsState.initError;
        appendPatrolEntry("错误", `初始化失败：${patrolWsState.initError}`, "error");
      } else if (data.result?.sessionId) {
        patrolWsState.sessionId = data.result.sessionId;
      }
      updateWsStatus();
      return;
    }

    if (data.method === "session.response_start") {
      const responseId = data.params?.responseId || rpcId("resp");
      const textEl = appendPatrolEntry("输出", "", "stream");
      if (textEl) {
        patrolWsState.streams.set(responseId, { textEl, buffer: "" });
      }
      if (consoleOut) consoleOut.textContent = "";
      return;
    }

    if (data.method === "session.response_chunk") {
      const responseId = data.params?.responseId || "";
      const chunk = data.params?.chunk?.content || "";
      if (!chunk) return;
      let entry = patrolWsState.streams.get(responseId);
      if (!entry) {
        const textEl = appendPatrolEntry("输出", "", "stream");
        if (!textEl) return;
        entry = { textEl, buffer: "" };
        patrolWsState.streams.set(responseId, entry);
      }
      entry.buffer += chunk;
      entry.textEl.innerHTML = renderPatrolMarkdown(entry.buffer);
      captureMediaChunks(chunk);
      return;
    }

    if (data.method === "session.response_end") {
      const responseId = data.params?.responseId || "";
      const status = data.params?.status || "";
      const error = normalizeWsError(data.params?.metadata?.error) || "";
      if (status === "error") {
        appendPatrolEntry("错误", `请求失败：${error || "未知错误"}`, "error");
      }
      patrolWsState.streams.delete(responseId);
      return;
    }

    if (data.method === "tools.confirm") {
      const toolType = data.params?.toolType || "external";
      const toolName = data.params?.call?.toolName || "";
      if (toolType === "external") {
        appendPatrolEntry("工具", `需要外部工具：${toolName || "未知工具"}（已拒绝）`, "warn");
        rejectExternalTool(data.id, toolName);
      }
      return;
    }

    if (data.method === "tools.execute") {
      const toolName = data.params?.toolName || "";
      appendPatrolEntry("工具", `需要执行工具：${toolName || "未知工具"}（已拒绝）`, "warn");
      rejectExternalTool(data.id, toolName);
      return;
    }

    if (data.error) {
      const message = normalizeWsError(data.error?.message);
      appendPatrolEntry("错误", `请求失败：${message || "未知错误"}`, "error");
      return;
    }
  }

  function waitForPatrolInit(timeoutMs = 8000) {
    if (patrolWsState.initDone) return Promise.resolve(true);
    if (patrolWsState.initError) return Promise.reject(new Error(patrolWsState.initError));

    return new Promise((resolve, reject) => {
      const start = Date.now();
      const timer = setInterval(() => {
        if (patrolWsState.initDone) {
          clearInterval(timer);
          resolve(true);
          return;
        }
        if (patrolWsState.initError) {
          clearInterval(timer);
          reject(new Error(patrolWsState.initError));
          return;
        }
        if (Date.now() - start > timeoutMs) {
          clearInterval(timer);
          reject(new Error("初始化超时"));
        }
      }, 120);
    });
  }

  async function ensurePatrolWs(agentId) {
    if (patrolWsState.socket && patrolWsState.agentId === agentId) {
      if (patrolWsState.socket.readyState === WebSocket.OPEN) return patrolWsState.socket;
      if (patrolWsState.connectPromise) return patrolWsState.connectPromise;
    }

    resetPatrolWs("agent_switch");
    patrolWsState.agentId = agentId;
    const wsUrl = buildChatWsUrl(agentId);
    const socket = new WebSocket(wsUrl);
    patrolWsState.socket = socket;

    socket.addEventListener("message", (event) => {
      try {
        const payload = typeof event.data === "string" ? JSON.parse(event.data) : event.data;
        handlePatrolWsMessage(payload);
      } catch (e) {
        if (consoleOut) consoleOut.textContent = `消息解析失败: ${String(e)}`;
      }
    });

    socket.addEventListener("close", () => {
      if (patrolWsState.socket === socket) {
        patrolWsState.socket = null;
        patrolWsState.initRequestId = "";
        patrolWsState.initDone = false;
        patrolWsState.connectPromise = null;
      }
      updateWsStatus();
    });

    socket.addEventListener("error", () => {
      if (consoleOut) consoleOut.textContent = "WebSocket 连接失败";
      updateWsStatus();
    });

    patrolWsState.connectPromise = new Promise((resolve, reject) => {
      socket.addEventListener("open", () => {
        updateWsStatus();
        void requestPatrolInit();
        resolve(socket);
      }, { once: true });

      socket.addEventListener("error", () => {
        reject(new Error("WebSocket 连接失败"));
      }, { once: true });
    });

    return patrolWsState.connectPromise;
  }

  async function startStream(stream) {
    if (!stream) {
      if (consoleOut) consoleOut.textContent = "请先选择一个巡检流";
      toast("未选择巡检流", "请先选择一个巡检流");
      return;
    }
    const agentId = agentSelect?.value || state.patrolSelectedAgentId;
    if (!agentId) {
      if (consoleOut) consoleOut.textContent = "未选择巡检智能体";
      toast("未选择巡检智能体", "请先选择巡检智能体");
      return;
    }

    if (patrolRuntime.runningStreamId && patrolRuntime.runningStreamId !== stream.id) {
      appendPatrolEntry("提示", "检测到已有运行中的巡检流，将切换到当前流。", "warn");
    }

    state.patrolSelectedStreamId = stream.id;
    store.set("jl_patrolStreamId", stream.id);

    const command = buildStartCommand(stream);
    appendPatrolEntry("指令", command, "cmd");

    try {
      await ensurePatrolWs(agentId);
      if (!patrolWsState.initDone) {
        await requestPatrolInit();
        await waitForPatrolInit();
      }
      sendPatrolWs({
        jsonrpc: "2.0",
        id: rpcId("patrol_msg"),
        method: "session.message",
        params: {
          sessionId: patrolWsState.sessionId || undefined,
          messageType: "text",
          content: command,
          context: stream.prompt ? { system_prompt: stream.prompt } : {},
        },
      });
      patrolRuntime.runningStreamId = stream.id;
      renderStreamList();
      updateConsoleInfo(stream);
      updateWsStatus();
      if (consoleOut) consoleOut.textContent = "";
    } catch (e) {
      appendPatrolEntry("错误", `启动失败：${String(e)}`, "error");
      if (consoleOut) consoleOut.textContent = String(e);
      toast("巡检启动失败", String(e));
    }
  }

  async function stopStream() {
    const stream = state.patrolStreams.find((s) => s.id === state.patrolSelectedStreamId);
    if (!stream) {
      if (consoleOut) consoleOut.textContent = "请先选择一个巡检流";
      toast("未选择巡检流", "请先选择一个巡检流");
      return;
    }
    const agentId = agentSelect?.value || state.patrolSelectedAgentId;
    if (!agentId) {
      if (consoleOut) consoleOut.textContent = "未选择巡检智能体";
      toast("未选择巡检智能体", "请先选择巡检智能体");
      return;
    }
    const wasRunning = patrolRuntime.runningStreamId === stream.id;
    appendPatrolEntry("指令", "停止巡检", "cmd");
    try {
      await ensurePatrolWs(agentId);
      if (!patrolWsState.initDone) {
        await requestPatrolInit();
        await waitForPatrolInit();
      }
      sendPatrolWs({
        jsonrpc: "2.0",
        id: rpcId("patrol_stop"),
        method: "session.message",
        params: {
          sessionId: patrolWsState.sessionId || undefined,
          messageType: "text",
          content: "停止巡检",
          context: {},
        },
      });
      if (patrolRuntime.runningStreamId === stream.id) {
        patrolRuntime.runningStreamId = "";
        renderStreamList();
      }
      if (patrolRuntime.previewStreamId === stream.id) {
        await stopPreview(true);
      }
      updateWsStatus();
      if (consoleOut) consoleOut.textContent = "";
      if (wasRunning) {
        setTimeout(() => {
          if (patrolWsState.socket && patrolWsState.socket.readyState === WebSocket.OPEN) {
            resetPatrolWs("patrol_stop");
            updateWsStatus();
          }
        }, 800);
      }
    } catch (e) {
      appendPatrolEntry("错误", `停止失败：${String(e)}`, "error");
      if (consoleOut) consoleOut.textContent = String(e);
    }
  }

  streamButton?.addEventListener("click", () => {
    nameInput?.focus();
  });

  qs("#patrolRefreshAgents")?.addEventListener("click", async () => {
    if (consoleOut) consoleOut.textContent = "Loading agents ...";
    try {
      await loadAgents();
      refreshAgentOptions();
      if (consoleOut) consoleOut.textContent = "";
      const count = isReviewMode ? getReviewAgents().length : getPatrolAgents().length;
      const label = isReviewMode ? "复判智能体" : "巡检智能体";
      toast(`${label}已刷新`, `${count} 个`);
    } catch (e) {
      if (consoleOut) consoleOut.textContent = String(e);
      toast("刷新失败", String(e));
    }
  });

  qs("#patrolCheckStreams")?.addEventListener("click", () => {
    void probePatrolStreams(true);
  });

  syncPromptButton?.addEventListener("click", async () => {
    const agentId = isReviewMode
      ? (agentSelect?.value || state.patrolReviewAgentId || "")
      : (agentSelect?.value || state.patrolSelectedAgentId || "");
    if (!agentId) {
      if (formOut) formOut.textContent = "请先选择智能体";
      return;
    }
    if (formOut) formOut.textContent = "正在同步智能体提示词...";
    try {
      const meta = await ensureAgentMeta(agentId, { force: true });
      const prompt = String(meta?.prompt || "").trim();
      if (!prompt) {
        if (formOut) formOut.textContent = "智能体提示词为空";
        return;
      }
      if (promptInput) promptInput.value = prompt;
      if (agentSchemaHint) agentSchemaHint.textContent = formatSchemaHint(meta?.jsonSchema);
      if (formOut) formOut.textContent = `已同步智能体提示词：${agentId}`;
    } catch (e) {
      if (formOut) formOut.textContent = `同步失败：${String(e)}`;
    }
  });

  applyButton?.addEventListener("click", () => {
    const streamId = state.patrolSelectedStreamId;
    const stream = state.patrolStreams.find((item) => item.id === streamId);
    if (!stream) {
      if (formOut) formOut.textContent = "请先选择一个巡检流";
      return;
    }
    const url = (urlInput?.value || "").trim();
    if (!url) {
      if (formOut) formOut.textContent = "请输入视频流地址";
      return;
    }
    const name = (nameInput?.value || "").trim();
    const interval = toOptionalNumber(intervalInput?.value);
    const stopMinutes = toOptionalNumber(stopInput?.value);
    const prompt = (promptInput?.value || "").trim();
    const prevUrl = stream.url;
    stream.name = name || stream.name || deriveStreamName(url);
    stream.url = url;
    stream.interval = interval;
    stream.stopMinutes = stopMinutes;
    stream.prompt = prompt;
    if (prevUrl !== url) {
      stream.reachable = undefined;
    }
    stream.updatedAt = new Date().toISOString();
    storePatrolStreams();
    renderStreamList();
    updateConsoleInfo(stream);
    if (formOut) formOut.textContent = `已更新：${stream.name}`;
  });

  saveButton?.addEventListener("click", () => {
    const name = (nameInput?.value || "").trim();
    const url = (urlInput?.value || "").trim();
    const interval = toOptionalNumber(intervalInput?.value);
    const stopMinutes = toOptionalNumber(stopInput?.value);
    const prompt = (promptInput?.value || "").trim();
    if (!url) {
      if (formOut) formOut.textContent = "请输入视频流地址";
      return;
    }
    const stream = {
      id: generateStreamId(),
      name: name || deriveStreamName(url),
      url,
      interval,
      stopMinutes,
      prompt,
      createdAt: new Date().toISOString(),
    };
    state.patrolStreams.unshift(stream);
    storePatrolStreams();
    if (formOut) formOut.textContent = `已保存：${stream.name}`;
    state.patrolSearch = "";
    state.patrolPageIndex = 0;
    store.set("jl_patrolPageIndex", 0);
    if (searchInput) searchInput.value = "";
    renderStreamList();
    setSelectedStream(stream.id);
    void probePatrolStreams(true, [stream]);
  });

  resetButton?.addEventListener("click", () => {
    if (nameInput) nameInput.value = "";
    if (urlInput) urlInput.value = "";
    if (intervalInput) intervalInput.value = "";
    if (stopInput) stopInput.value = "";
    if (promptInput) promptInput.value = "";
    if (formOut) formOut.textContent = "";
  });

  searchInput?.addEventListener("input", () => {
    state.patrolSearch = searchInput.value;
    store.set("jl_patrolSearch", state.patrolSearch);
    state.patrolPageIndex = 0;
    store.set("jl_patrolPageIndex", 0);
    renderStreamList();
  });

  pagePrevBtn?.addEventListener("click", () => {
    state.patrolPageIndex = Math.max(0, state.patrolPageIndex - 1);
    store.set("jl_patrolPageIndex", state.patrolPageIndex);
    renderStreamList();
  });

  pageNextBtn?.addEventListener("click", () => {
    state.patrolPageIndex = state.patrolPageIndex + 1;
    store.set("jl_patrolPageIndex", state.patrolPageIndex);
    renderStreamList();
  });

  pageSizeSelect?.addEventListener("change", () => {
    state.patrolPageSize = normalizePatrolPageSize(pageSizeSelect.value);
    store.set("jl_patrolPageSize", state.patrolPageSize);
    state.patrolPageIndex = 0;
    store.set("jl_patrolPageIndex", 0);
    renderStreamList();
  });

  streamPicker?.addEventListener("change", () => {
    const nextId = streamPicker.value;
    if (!nextId) {
      state.patrolSelectedStreamId = "";
      store.set("jl_patrolStreamId", "");
      updateConsoleInfo(null);
      return;
    }
    setSelectedStream(nextId);
  });

  agentSelect?.addEventListener("change", () => {
    const prevAgentId = isReviewMode ? state.patrolReviewAgentId : state.patrolSelectedAgentId;
    if (isReviewMode) {
      state.patrolReviewAgentId = agentSelect.value;
      store.set("jl_patrolReviewAgentId", state.patrolReviewAgentId);
    } else {
      state.patrolSelectedAgentId = agentSelect.value;
      store.set("jl_patrolAgentId", state.patrolSelectedAgentId);
    }
    updateWsStatus();
    const currentStream = state.patrolStreams.find((stream) => stream.id === state.patrolSelectedStreamId) || null;
    updateConsoleInfo(currentStream);
    const agentId = isReviewMode ? state.patrolReviewAgentId : state.patrolSelectedAgentId;
    updateAgentMetaView(agentId, currentStream);
    void autoSyncPromptForAgentChange(prevAgentId, agentId, currentStream);
  });

  listEl?.addEventListener("click", (event) => {
    const btn = event.target.closest("[data-action]");
    const card = event.target.closest(".patrol-stream-card");
    const streamId = btn?.getAttribute("data-stream-id") || card?.getAttribute("data-stream-id");
    if (!streamId) return;
    const stream = state.patrolStreams.find((item) => item.id === streamId);
    if (!stream) return;

    const action = btn?.getAttribute("data-action") || "select";
    if (action === "select") {
      setSelectedStream(stream.id);
      return;
    }
    if (action === "console") {
      setSelectedStream(stream.id);
      setRoute(targetRoute);
      return;
    }
    if (action === "preview") {
      setSelectedStream(stream.id);
      if (route() === "patrol-streams") {
        patrolRuntime.pendingAction = "preview";
        patrolRuntime.pendingStreamId = stream.id;
        setRoute(targetRoute);
        return;
      }
      startPreview(stream);
      return;
    }
    if (action === "start") {
      setSelectedStream(stream.id);
      if (route() === "patrol-streams") {
        patrolRuntime.pendingAction = "start";
        patrolRuntime.pendingStreamId = stream.id;
        setRoute(targetRoute);
        return;
      }
      if (isReviewMode) startReviewStream(stream);
      else startStream(stream);
      return;
    }
    if (action === "stop") {
      setSelectedStream(stream.id);
      if (route() === "patrol-streams") {
        patrolRuntime.pendingAction = "stop";
        patrolRuntime.pendingStreamId = stream.id;
        setRoute(targetRoute);
        return;
      }
      if (isReviewMode) stopReviewStream();
      else stopStream();
      return;
    }
    if (action === "remove") {
      state.patrolStreams = state.patrolStreams.filter((item) => item.id !== stream.id);
      if (state.patrolSelectedStreamId === stream.id) {
        state.patrolSelectedStreamId = "";
      }
      if (patrolRuntime.runningStreamId === stream.id) {
        if (patrolRuntime.reviewRunning) {
          stopReviewStream(true);
        }
        patrolRuntime.runningStreamId = "";
      }
      if (patrolRuntime.previewStreamId === stream.id) {
        void stopPreview(true);
      }
      storePatrolStreams();
      renderStreamList();
      syncSelectedStream();
      if (formOut) formOut.textContent = `已删除：${stream.name}`;
    }
  });

  startButton?.addEventListener("click", () => {
    const stream = state.patrolStreams.find((item) => item.id === state.patrolSelectedStreamId);
    if (isReviewMode) startReviewStream(stream || null);
    else startStream(stream || null);
  });

  stopButton?.addEventListener("click", () => {
    if (isReviewMode) stopReviewStream();
    else stopStream();
  });

  clearButton?.addEventListener("click", () => {
    if (logEl) logEl.innerHTML = "";
    if (consoleOut) consoleOut.textContent = "等待巡检输出...";
    resetPatrolMedia();
    resetPatrolStats();
  });

  previewPlayBtn?.addEventListener("click", () => {
    const stream = state.patrolStreams.find((item) => item.id === state.patrolSelectedStreamId);
    startPreview(stream || null);
  });

  previewStopBtn?.addEventListener("click", () => {
    stopPreview(true);
  });

  previewVideo?.addEventListener("playing", () => {
    setPreviewStatus("播放中");
    togglePreviewPlaceholder(false);
    resetPreviewRetry();
  });

  previewVideo?.addEventListener("pause", () => {
    if (patrolRuntime.previewStreamId) setPreviewStatus("已暂停");
  });

  previewVideo?.addEventListener("error", () => {
    setPreviewStatus("播放失败");
    togglePreviewPlaceholder(true);
    schedulePreviewRetry("播放器错误");
  });

  if (!state.agents.length) {
    loadAgents()
      .then(() => {
        refreshAgentOptions();
        runPendingAction();
      })
      .catch((e) => {
        if (consoleOut) consoleOut.textContent = `智能体加载失败：${String(e)}`;
      });
  } else {
    refreshAgentOptions();
    runPendingAction();
  }

  if (searchInput && state.patrolSearch) searchInput.value = state.patrolSearch;
  if (previewVideo) {
    togglePreviewPlaceholder(!patrolRuntime.previewUrl);
    setPreviewStatus(patrolRuntime.previewUrl ? "播放中" : "待命");
    setPreviewLink(patrolRuntime.previewUrl || "");
    setPreviewType(patrolRuntime.previewSourceType === "direct" ? "Direct" : "HLS");
  }
  syncSelectedStream();
  void probePatrolStreams(false);
  resetPatrolStats();
  updateWsStatus();
}

function bindTools() {
  const out = qs("#toolsOut");
  qs("#loadTools").addEventListener("click", async () => {
    out.textContent = "Loading ...";
    try {
      await loadTools(true);
      render();
      toast("工具加载成功", `${state.tools.length} 个`);
    } catch (e) {
      out.textContent = String(e);
      toast("工具加载失败", String(e));
    }
  });

  if (!state.tools.length) {
    qs("#loadTools").click();
  }
}

function bindPrompts() {
  const out = qs("#promptOut");

  qs("#loadPromptTemplates").addEventListener("click", async () => {
    out.textContent = "Loading ...";
    try {
      const templates = await apiFetch("/api/v1/prompt/templates");
      state.templates = templates || null;
      render();
      toast("模板已加载", `${Object.keys(state.templates || {}).length} 个`);
    } catch (e) {
      out.textContent = String(e);
      toast("加载失败", String(e));
    }
  });

  qsa("[data-use-template]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-use-template");
      if (!key) return;
      qs("#tmplKey").value = key;
      qs("#tmplVars").focus();
    });
  });

  qs("#applyTemplate").addEventListener("click", async () => {
    out.textContent = "Applying ...";
    const key = qs("#tmplKey").value.trim();
    const varsStr = qs("#tmplVars").value.trim();
    if (!key) return (out.textContent = "请输入模板 Key");
    const vars = varsStr ? readJsonSafely(varsStr) : {};
    if (varsStr && vars === null) return (out.textContent = "变量 JSON 解析失败");

    try {
      const res = await apiFetch("/api/v1/prompt/template/apply", {
        method: "POST",
        body: JSON.stringify({ task_type: key, variables: vars || {} }),
      });
      out.textContent = typeof res === "string" ? res : JSON.stringify(res, null, 2);
      toast("渲染完成", key);
    } catch (e) {
      out.textContent = String(e);
      toast("渲染失败", String(e));
    }
  });
}

window.addEventListener("hashchange", render);

async function bootstrap() {
  // Topbar status
  qs("#baseUrlBadge").textContent = state.baseUrl;

  render();

  // best-effort background preload
  loadAgents()
    .then(() => {
      if (route() === "dashboard") refreshDashboardAgents();
    })
    .catch(() => null);
}

bootstrap().catch((e) => {
  console.error(e);
  toast("启动失败", String(e));
});
