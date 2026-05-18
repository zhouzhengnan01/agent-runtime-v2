import { defineStore } from "pinia";

import { listAgents, startSseRun } from "@/api/agents";
import { AcpClient, runAcpPrompt } from "@/api/acp";
import { listAppTemplates } from "@/api/apps";
import { listArtifacts } from "@/api/artifacts";
import { loadAdminToken, saveAdminToken } from "@/api/http";
import { getSandboxStatus, type SandboxStatus } from "@/api/sandbox";
import { readEventStream } from "@/api/sse";
import { uploadThreadFile } from "@/api/uploads";
import type {
  AgentRunResult,
  AgentSummary,
  AppTemplate,
  Artifact,
  AttachmentRef,
  ChatMessage,
  PendingAttachmentRequirement,
  RequiredInput,
  RuntimeEvent,
  TimelineItem,
  UploadBatchSummary,
  UploadItem,
  VerificationResult,
  WorkbenchMessage
} from "@/api/types";

const THREAD_STORAGE_KEY = "jetlinks.runtime.threadId";
const DEFAULT_APP_TEMPLATE_NAME = "algorithm-cpu-training-sandbox";
const MAX_TIMELINE_ITEMS = 120;
const MAX_UPLOAD_FILES = 200;
const UPLOAD_CONCURRENCY = 3;

type InspectorTab = "artifacts" | "abilities" | "verification" | "spec" | "timeline" | "acp";
type Transport = "acp-ws" | "sse";
type ViewKey = "chat" | "apps" | "skills" | "tools" | "workflows" | "cron";

export const useWorkbenchStore = defineStore("workbench", {
  state: () => ({
    adminToken: loadAdminToken(),
    agents: [] as AgentSummary[],
    selectedAgent: "default",
    transport: "acp-ws" as Transport,
    threadId: localStorage.getItem(THREAD_STORAGE_KEY) || `local-${Date.now().toString(36)}`,
    view: "chat" as ViewKey,
    inspectorTab: "artifacts" as InspectorTab,
    messages: [] as WorkbenchMessage[],
    timeline: [] as TimelineItem[],
    artifacts: [] as Artifact[],
    verification: null as VerificationResult | null,
    spec: null as unknown,
    selectedSkills: [] as string[],
    selectedMcpTools: [] as string[],
    selectedWorkflow: null as string | null,
    deepExecution: false,
    yoloExecution: false,
    appTemplates: [] as AppTemplate[],
    appFilter: "all",
    selectedAppTemplateName: DEFAULT_APP_TEMPLATE_NAME,
    pendingAttachmentRequirement: null as PendingAttachmentRequirement | null,
    loadingApps: false,
    attachments: [] as UploadItem[],
    requiredInputs: [] as RequiredInput[],
    requiredInputAccept: "*/*",
    requiredInputLabel: "文件",
    sandboxStatus: null as SandboxStatus | null,
    acpClient: null as AcpClient | null,
    acpConnected: false,
    acpSessionId: "",
    acpToolStatus: "",
    acpToolError: "",
    acpLastResult: null as Record<string, unknown> | null,
    fsPath: "/mnt/user-data/workspace/notes.txt",
    fsContent: "",
    terminalId: "",
    terminalCommand: "python",
    terminalArgs: "-c \"print('terminal-ok')\"",
    terminalOutput: "",
    rawRpcMethod: "session/list",
    rawRpcParams: "{}",
    input: "",
    loadingAgents: false,
    loadingArtifacts: false,
    running: false,
    error: "",
    finalResultApplied: false,
    messageSeq: 0,
    timelineSeq: 0
  }),
  getters: {
    activeAgent(state): AgentSummary | undefined {
      return state.agents.find((agent) => agent.name === state.selectedAgent);
    },
    runtimeOptions(state): Record<string, unknown> {
      const template = state.appTemplates.find((item) => item.name === state.selectedAppTemplateName);
      const templateOptions =
        template?.runtime_options && typeof template.runtime_options === "object" ? template.runtime_options : {};
      const options: Record<string, unknown> = {
        ...templateOptions,
        thread_id: state.threadId,
        selected_skills: [...state.selectedSkills],
        selected_mcp_tools: [...state.selectedMcpTools],
        ...(state.selectedAppTemplateName ? { app_template_name: state.selectedAppTemplateName, model_type: "chat" } : {}),
        ...(state.selectedWorkflow ? { workflow: state.selectedWorkflow } : {})
      };
      if (state.yoloExecution) {
        options.mode = "yolo";
        options.config_options = { max_tool_rounds: 16 };
      } else if (state.deepExecution) {
        options.mode = "autonomous";
        options.config_options = { max_tool_rounds: 12 };
      }
      return options;
    },
    acpRuntimeOptions(state): Record<string, unknown> {
      const template = state.appTemplates.find((item) => item.name === state.selectedAppTemplateName);
      const templateOptions =
        template?.runtime_options && typeof template.runtime_options === "object" ? template.runtime_options : {};
      const options: Record<string, unknown> = {
        ...templateOptions,
        threadId: state.threadId,
        selectedSkills: [...state.selectedSkills],
        selectedMcpTools: [...state.selectedMcpTools],
        ...(state.selectedAppTemplateName ? { appTemplateName: state.selectedAppTemplateName, modelType: "chat" } : {}),
        ...(state.selectedWorkflow ? { workflow: state.selectedWorkflow } : {})
      };
      if (state.yoloExecution) {
        options.mode = "yolo";
        options.configOptions = { max_tool_rounds: 16 };
      } else if (state.deepExecution) {
        options.mode = "autonomous";
        options.configOptions = { max_tool_rounds: 12 };
      }
      return options;
    },
    uploadBatchSummary(state): UploadBatchSummary | null {
      if (!state.attachments.length) return null;
      const failed = state.attachments.filter((item) => item.error).length;
      const uploading = state.attachments.filter((item) => item.uploading).length;
      return {
        total: state.attachments.length,
        completed: state.attachments.length - failed - uploading,
        failed,
        uploading
      };
    },
    visibleAttachments(state): UploadItem[] {
      return state.attachments.slice(0, 40);
    },
    hiddenAttachmentCount(state): number {
      return Math.max(0, state.attachments.length - 40);
    }
  },
  actions: {
    setAdminToken(token: string) {
      this.adminToken = token.trim();
      saveAdminToken(this.adminToken);
    },
    setThreadId(threadId: string) {
      this.threadId = threadId.trim() || `local-${Date.now().toString(36)}`;
      localStorage.setItem(THREAD_STORAGE_KEY, this.threadId);
    },
    newThread() {
      this.setThreadId(`local-${Date.now().toString(36)}`);
      this.messages = [];
      this.timeline = [];
      this.artifacts = [];
      this.verification = null;
      this.spec = null;
      this.error = "";
      this.finalResultApplied = false;
      this.attachments = [];
      this.requiredInputs = [];
      this.pendingAttachmentRequirement = null;
    },
    setView(view: ViewKey) {
      this.view = view;
      if (view !== "chat") this.inspectorTab = view === "cron" ? "timeline" : "abilities";
    },
    async loadAgents() {
      this.loadingAgents = true;
      try {
        this.agents = await listAgents();
        if (this.agents.some((agent) => agent.name === "default")) {
          this.selectedAgent = "default";
        } else if (!this.agents.some((agent) => agent.name === this.selectedAgent)) {
          this.selectedAgent = this.agents[0]?.name || "default";
        }
      } finally {
        this.loadingAgents = false;
      }
    },
    async loadSandboxStatus() {
      try {
        this.sandboxStatus = await getSandboxStatus();
      } catch (error) {
        this.sandboxStatus = { error: error instanceof Error ? error.message : String(error) };
      }
    },
    async loadAppTemplates() {
      this.loadingApps = true;
      try {
        this.appTemplates = await listAppTemplates();
        const selectedTemplate = this.appTemplates.find((template) => template.name === this.selectedAppTemplateName);
        const fallbackTemplate =
          selectedTemplate ||
          this.appTemplates.find((template) => template.name === DEFAULT_APP_TEMPLATE_NAME) ||
          this.appTemplates[0];
        this.selectedAppTemplateName = fallbackTemplate?.name || "";
      } finally {
        this.loadingApps = false;
      }
    },
    setDeepExecution(enabled: boolean) {
      this.deepExecution = enabled;
      if (enabled) this.yoloExecution = false;
    },
    setYoloExecution(enabled: boolean) {
      this.yoloExecution = enabled;
      if (enabled) this.deepExecution = false;
    },
    applyAppTemplate(name: string, options: { fillPrompt?: boolean; run?: boolean } = {}) {
      const template = this.appTemplates.find((item) => item.name === name);
      if (!template) return;
      this.applyAppTemplateCapabilities(template);
      if (options.fillPrompt !== false && template.prompt_examples?.length) {
        this.input = template.prompt_examples[0] || "";
      }
      if (options.run && templateRequiresAttachment(template) && !this.attachments.some((item) => item.path)) {
        this.pendingAttachmentRequirement = {
          templateName: template.name,
          label: "图片",
          accept: "image/*",
          reason: "该应用需要先上传图片后再运行。"
        };
        this.addTimeline("等待应用输入", `${template.title || template.name} · image/*`);
        this.view = "chat";
        return;
      }
      this.view = "chat";
      if (options.run && this.input.trim()) void this.sendCurrentMessage();
    },
    applyAppTemplateCapabilities(template: AppTemplate) {
      this.selectedAppTemplateName = template.name;
      this.selectedAgent = template.agent_name || "default";
      this.selectedSkills = [...(template.selected_skills || [])];
      this.selectedMcpTools = [...(template.selected_mcp_tools || [])];
      this.selectedWorkflow = template.workflow && !["agent_loop", "default"].includes(template.workflow)
        ? template.workflow
        : null;
      this.deepExecution = false;
      this.yoloExecution = false;
    },
    async refreshArtifacts() {
      this.loadingArtifacts = true;
      try {
        this.artifacts = await listArtifacts(this.threadId);
      } finally {
        this.loadingArtifacts = false;
      }
    },
    async ensureAcpSession() {
      if (this.transport !== "acp-ws") {
        this.acpToolError = "ACP 工具需要使用 ACP WS transport";
        throw new Error(this.acpToolError);
      }
      if (!this.acpClient) {
        this.acpClient = new AcpClient({
          token: this.adminToken,
          onEvent: (event) => this.handleRuntimeEvent(event as RuntimeEvent),
          onPermissionRequest: async ({ request }) => {
            const options = Array.isArray(request?.options) ? request.options : [];
            const autoApproved = this.yoloExecution;
            const approved =
              autoApproved ||
              window.confirm(`${request?.title || "Permission request"}${request?.description ? `\n\n${request.description}` : ""}`);
            const selectedOptionId = approved ? permissionOptionId(options, autoApproved) : null;
            this.addTimeline(
              "权限确认",
              approved ? (selectedOptionId ? `已选择 ${selectedOptionId}` : "已批准") : "已拒绝"
            );
            return { approved, selectedOptionId };
          }
        });
      }
      if (!this.acpConnected) {
        await this.acpClient.initialize();
        this.acpConnected = true;
      }
      if (!this.acpSessionId) {
        const session = await this.acpClient.newSession({
          cwd: "/",
          agentName: this.selectedAgent,
          threadId: this.threadId
        });
        this.acpSessionId = String(session.sessionId || "");
      }
      return this.acpSessionId;
    },
    async acpRpc(method: string, params: Record<string, unknown> = {}) {
      this.acpToolError = "";
      this.acpToolStatus = "执行中";
      try {
        await this.ensureAcpSession();
        if (!this.acpClient) throw new Error("ACP WS 未连接");
        const result = await this.acpClient.rpc(method, params);
        this.acpLastResult = result;
        this.acpToolStatus = "完成";
        return result;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        this.acpToolError = message;
        this.acpToolStatus = "失败";
        throw error;
      }
    },
    async connectAcpTools() {
      await this.ensureAcpSession();
      this.acpToolStatus = `已连接 ${this.acpSessionId}`;
    },
    closeAcpTools() {
      this.acpClient?.close();
      this.acpClient = null;
      this.acpConnected = false;
      this.acpSessionId = "";
      this.terminalId = "";
      this.acpToolStatus = "已关闭";
    },
    async listAcpSessions() {
      await this.acpRpc("session/list");
    },
    async cancelAcpSession() {
      if (!this.acpSessionId) await this.ensureAcpSession();
      await this.acpRpc("session/cancel", { sessionId: this.acpSessionId });
    },
    async closeAcpSession() {
      if (!this.acpSessionId) return;
      await this.acpRpc("session/close", { sessionId: this.acpSessionId });
      this.acpSessionId = "";
      this.terminalId = "";
    },
    async readAcpFile() {
      const result = await this.acpRpc("fs/read_text_file", {
        sessionId: this.acpSessionId,
        path: this.fsPath,
        limit: 400
      });
      this.fsContent = typeof result.content === "string" ? result.content : JSON.stringify(result, null, 2);
    },
    async writeAcpFile() {
      await this.acpRpc("fs/write_text_file", {
        sessionId: this.acpSessionId,
        path: this.fsPath,
        content: this.fsContent
      });
    },
    async createAcpTerminal() {
      const result = await this.acpRpc("terminal/create", {
        sessionId: this.acpSessionId,
        command: this.terminalCommand,
        args: splitArgs(this.terminalArgs)
      });
      this.terminalId = String(result.terminalId || result.terminal_id || "");
      await this.readAcpTerminalOutput();
    },
    async readAcpTerminalOutput() {
      if (!this.terminalId) return;
      const result = await this.acpRpc("terminal/output", {
        sessionId: this.acpSessionId,
        terminalId: this.terminalId
      });
      this.terminalOutput = typeof result.output === "string" ? result.output : JSON.stringify(result, null, 2);
    },
    async waitAcpTerminal() {
      if (!this.terminalId) return;
      await this.acpRpc("terminal/wait_for_exit", {
        sessionId: this.acpSessionId,
        terminalId: this.terminalId
      });
      await this.readAcpTerminalOutput();
    },
    async killAcpTerminal() {
      if (!this.terminalId) return;
      await this.acpRpc("terminal/kill", {
        sessionId: this.acpSessionId,
        terminalId: this.terminalId
      });
      await this.readAcpTerminalOutput();
    },
    async releaseAcpTerminal() {
      if (!this.terminalId) return;
      await this.acpRpc("terminal/release", {
        sessionId: this.acpSessionId,
        terminalId: this.terminalId
      });
      this.terminalId = "";
      this.terminalOutput = "";
    },
    async runRawAcpRpc() {
      await this.acpRpc(this.rawRpcMethod.trim(), parseJsonObject(this.rawRpcParams));
    },
    addMessage(role: "user" | "assistant", content: string, options: Partial<WorkbenchMessage> = {}) {
      this.messages.push({
        id: `msg-${++this.messageSeq}`,
        role,
        content,
        streaming: options.streaming,
        progress: options.progress
      });
      return this.messages.length - 1;
    },
    beginAssistantMessage() {
      const last = this.messages[this.messages.length - 1];
      if (last?.role === "assistant" && last.streaming) return last;
      this.addMessage("assistant", "", { streaming: true });
      return this.messages[this.messages.length - 1];
    },
    appendAssistantDelta(text: string) {
      if (!text) return;
      const message = this.beginAssistantMessage();
      if (message.progress) {
        message.content = "";
        message.progress = false;
      }
      message.content += text;
    },
    setAssistantProgress(text: string) {
      if (!text) return;
      const message = this.beginAssistantMessage();
      if (message.content && !message.progress) return;
      message.content = text;
      message.progress = true;
    },
    finishAssistantMessage(content?: string) {
      const last = this.messages[this.messages.length - 1];
      if (last?.role === "assistant" && last.streaming) {
        if (content) {
          last.content = content;
          last.progress = false;
        } else if (!last.content) {
          last.content = "已完成，但没有返回文本内容。";
        }
        last.streaming = false;
        return;
      }
      if (content) this.addMessage("assistant", content);
    },
    addTimeline(name: string, detail = "") {
      this.timeline.push({ id: `tl-${++this.timelineSeq}`, name, detail });
      if (this.timeline.length > MAX_TIMELINE_ITEMS) {
        this.timeline.splice(0, this.timeline.length - MAX_TIMELINE_ITEMS);
      }
    },
    upsertArtifact(artifact: unknown) {
      if (!artifact || typeof artifact !== "object") return;
      const next = artifact as Artifact;
      if (!next.path) return;
      const index = this.artifacts.findIndex((item) => item.path === next.path);
      if (index >= 0) this.artifacts[index] = next;
      else this.artifacts.push(next);
    },
    async uploadFiles(files: FileList | File[]) {
      this.error = "";
      const allFiles = Array.from(files || []).filter((file) => file.size >= 0);
      if (allFiles.length > MAX_UPLOAD_FILES) {
        this.error = `单次最多上传 ${MAX_UPLOAD_FILES} 个文件，当前选择了 ${allFiles.length} 个。请缩小文件夹范围或打包后上传。`;
        this.addTimeline("上传已拦截", this.error);
        return;
      }
      const selected = allFiles;
      if (!selected.length) return;
      this.setThreadId(this.threadId);
      const pending: UploadItem[] = selected.map((file) => {
        const relativePath = getRelativeUploadPath(file);
        return {
          id: `upload-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
          name: file.name,
          displayName: relativePath || file.name,
          relativePath,
          size: file.size,
          mime_type: file.type || null,
          uploading: true,
          progress: 0
        };
      });
      this.attachments.push(...pending);
      this.addTimeline("上传附件", summarizeUploadNames(pending));

      await runWithConcurrency(selected, UPLOAD_CONCURRENCY, async (file, index) => {
        const item = pending[index];
        try {
          const uploaded = await uploadThreadFile(this.threadId, this.adminToken, file, {
            timeoutMs: 120000,
            relativePath: item.relativePath,
            onProgress: (progress) => {
              const current = this.attachments.find((entry) => entry.id === item.id);
              if (current) {
                current.progress = progress;
                if (progress >= 100) current.uploading = false;
              }
            }
          });
          item.uploading = false;
          item.progress = 100;
          item.path = uploaded.path;
          item.mime_type = uploaded.mime_type || item.mime_type;
          const current = this.attachments.find((entry) => entry.id === item.id);
          if (current) Object.assign(current, item);
        } catch (error) {
          item.uploading = false;
          item.progress = 0;
          item.error = error instanceof Error ? error.message : String(error);
          const current = this.attachments.find((entry) => entry.id === item.id);
          if (current) Object.assign(current, item);
        }
      });
    },
    removeAttachment(id: string) {
      this.attachments = this.attachments.filter((item) => item.id !== id);
    },
    clearRequiredInputs() {
      this.requiredInputs = [];
      this.requiredInputAccept = "*/*";
      this.requiredInputLabel = "文件";
    },
    clearPendingAttachmentRequirement() {
      this.pendingAttachmentRequirement = null;
    },
    async handlePendingAttachmentFiles(files: FileList | File[]) {
      const selected = Array.from(files || []);
      if (!selected.length) return;
      this.pendingAttachmentRequirement = null;
      await this.uploadFiles(selected);
      if (this.attachments.some((item) => item.error || item.uploading)) return;
      if (!this.input.trim() && this.selectedAppTemplateName) {
        const template = this.appTemplates.find((item) => item.name === this.selectedAppTemplateName);
        this.input = template?.prompt_examples?.[0] || "继续处理刚上传的文件。";
      }
      await this.sendCurrentMessage();
    },
    async handleRequiredInputFiles(files: FileList | File[]) {
      const selected = Array.from(files || []);
      if (!selected.length) return;
      this.clearRequiredInputs();
      await this.uploadFiles(selected);
      if (this.attachments.some((item) => item.error || item.uploading)) return;
      this.input = "继续处理刚上传的文件。";
      await this.sendCurrentMessage();
    },
    handleRuntimeEvent(event: RuntimeEvent) {
      const data = event.data || {};
      switch (event.type) {
        case "run.started":
          this.setAssistantProgress("任务已启动，正在准备上下文...");
          this.addTimeline("启动", `${stringValue(data.workflow, "workflow")} · ${stringValue(data.thread_id)}`);
          break;
        case "spec.started":
          this.setAssistantProgress("正在理解需求并选择可用技能...");
          this.addTimeline("理解需求", `${arrayLength(data.allowed_skills)} 个可用技能`);
          break;
        case "spec.planner.started":
          this.setAssistantProgress("正在调用大模型规划图结构...");
          this.addTimeline("大模型规划", stringValue(data.skill_name));
          break;
        case "spec.planner.completed":
          this.setAssistantProgress("规划完成，正在整理结构化 Spec...");
          this.addTimeline("规划完成", stringValue(data.model, "LLM"));
          break;
        case "spec.completed":
          this.spec = data.spec || null;
          this.addTimeline("生成 Spec", stringValue(data.skill_name));
          break;
        case "skill.selected":
          this.addTimeline("选择技能", stringValue((data.skill as Record<string, unknown> | undefined)?.name));
          break;
        case "sandbox.started":
          this.addTimeline("沙盒启动", `${stringValue(data.profile_name)} · ${stringValue(data.image)}`);
          break;
        case "sandbox.completed":
          this.addTimeline("沙盒完成", `${numberValue(data.output_count)} 个输出`);
          break;
        case "sandbox.failed":
          this.addTimeline("沙盒失败", stringValue(data.error, "failed"));
          break;
        case "skill.started":
          this.setAssistantProgress("正在执行技能...");
          this.addTimeline("执行技能", `${stringValue(data.skill_name)} · attempt ${numberValue(data.attempt)}`);
          break;
        case "skill.completed":
          this.addTimeline("技能完成", `${numberValue(data.output_count)} 个输出`);
          break;
        case "artifact.created":
          this.upsertArtifact(data.artifact);
          this.addTimeline("生成文件", stringValue((data.artifact as Record<string, unknown> | undefined)?.name));
          break;
        case "preview.ready":
          this.upsertArtifact(data.artifact);
          break;
        case "verifier.started":
          this.setAssistantProgress("正在校验生成文件...");
          this.addTimeline("开始校验", `retry ${numberValue(data.retry_count)}`);
          break;
        case "verifier.completed":
          this.verification = (data.verification || null) as VerificationResult | null;
          this.addTimeline("校验完成", this.verification?.passed ? "通过" : "未通过");
          break;
        case "agent.message.delta":
          this.appendAssistantDelta(eventText(data));
          break;
        case "agent.message":
          this.finishAssistantMessage(eventText(data));
          break;
        case "run.completed":
          this.addTimeline("完成", "completed");
          this.applyFinalResult((data.result || null) as AgentRunResult | null);
          break;
        case "run.failed":
          this.addTimeline("失败", stringValue(data.error, "failed"));
          if (data.result) this.applyFinalResult(data.result as AgentRunResult);
          else if (data.error) this.finishAssistantMessage(`执行失败：${stringValue(data.error)}`);
          break;
      }
    },
    applyFinalResult(result: AgentRunResult | null) {
      if (!result || this.finalResultApplied) return;
      this.finalResultApplied = true;
      if (result.reply) {
        const last = this.messages[this.messages.length - 1];
        if (!(last?.role === "assistant" && last.content === result.reply)) {
          this.finishAssistantMessage(result.reply);
        }
      }
      if (result.artifacts?.length) this.artifacts = result.artifacts;
      this.verification = result.verification || this.verification;
      this.spec = result.spec ?? this.spec;
      this.maybePromptForRequiredInputs(result);
    },
    maybePromptForRequiredInputs(result: AgentRunResult) {
      const requiredInputs = result.metadata?.required_inputs || [];
      if (!result.metadata?.requires_input || !requiredInputs.length) return;
      const fileInputs = requiredInputs.filter((item) => requiredInputUsesFile(item));
      if (!fileInputs.length) return;
      this.requiredInputs = fileInputs;
      this.requiredInputAccept =
        [...new Set(fileInputs.map((item) => item.accept || acceptForRequiredInputType(item.type)).filter(Boolean))].join(
          ","
        ) || "*/*";
      this.requiredInputLabel = fileInputs.map((item) => requiredInputLabel(item.type)).join("、") || "文件";
      const reasons = fileInputs
        .map((item) => item.reason || item.description)
        .filter(Boolean)
        .slice(0, 2)
        .join("；");
      this.addTimeline("等待补充输入", `${this.requiredInputLabel} · ${this.requiredInputAccept}`);
      this.finishAssistantMessage(`需要上传${this.requiredInputLabel}${reasons ? `：${reasons}` : "。"}。`);
    },
    async sendCurrentMessage() {
      const content = this.input.trim();
      if (!content || this.running) return;
      if (this.attachments.some((item) => item.uploading)) {
        this.error = "附件仍在上传中";
        return;
      }
      if (this.attachments.some((item) => item.error)) {
        this.error = "存在上传失败的附件，请移除后重试";
        return;
      }
      this.error = "";
      this.running = true;
      this.finalResultApplied = false;
      this.requiredInputs = [];
      this.pendingAttachmentRequirement = null;
      this.setThreadId(this.threadId);
      this.addMessage("user", content);
      this.input = "";
      this.setAssistantProgress("正在发送请求...");

      const messages: ChatMessage[] = [{ role: "user", content }];
      const attachments: AttachmentRef[] = this.attachments
        .filter((item) => item.path)
        .map((item) => ({
          name: item.name,
          path: item.path,
          mime_type: item.mime_type,
          size: item.size
        }));

      try {
        if (this.transport === "acp-ws") {
          try {
            const result = this.acpClient && this.acpSessionId
              ? await this.acpClient.prompt({
                  agentName: this.selectedAgent,
                  content,
                  threadId: this.threadId,
                  messages,
                  attachments,
                  runtimeOptions: this.acpRuntimeOptions,
                  sessionId: this.acpSessionId
                })
              : await runAcpPrompt({
                  agentName: this.selectedAgent,
                  content,
                  threadId: this.threadId,
                  messages,
                  attachments,
                  runtimeOptions: this.acpRuntimeOptions,
                  token: this.adminToken,
                  onEvent: (event) => this.handleRuntimeEvent(event as RuntimeEvent),
                  onPermissionRequest: async ({ request }) => {
                    const options = Array.isArray(request?.options) ? request.options : [];
                    const autoApproved = this.yoloExecution;
                    const approved =
                      autoApproved ||
                      window.confirm(
                        `${request?.title || "Permission request"}${request?.description ? `\n\n${request.description}` : ""}`
                      );
                    const selectedOptionId = approved ? permissionOptionId(options, autoApproved) : null;
                    this.addTimeline(
                      "权限确认",
                      approved ? (selectedOptionId ? `已选择 ${selectedOptionId}` : "已批准") : "已拒绝"
                    );
                    return { approved, selectedOptionId };
                  }
                });
            this.applyFinalResult(result);
          } catch (error) {
            if (!shouldFallbackToSse(error)) throw error;
            this.addTimeline("连接降级", "ACP WS 已断开，切换 SSE");
            this.setAssistantProgress("WebSocket 断开，正在切换 SSE 通道...");
            await this.sendViaSse(messages, attachments);
          }
        } else {
          await this.sendViaSse(messages, attachments);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        this.error = message;
        this.finishAssistantMessage(`请求失败：${message}`);
      } finally {
        this.finishAssistantMessage();
        this.running = false;
        await this.refreshArtifacts().catch((error) => {
          const message = error instanceof Error ? error.message : String(error);
          this.addTimeline("刷新文件失败", message);
        });
        if (!this.requiredInputs.length) this.attachments = [];
      }
    },
    async sendViaSse(messages: ChatMessage[], attachments: AttachmentRef[]) {
      const body = await startSseRun(this.selectedAgent, this.adminToken, {
        messages,
        attachments,
        runtime_options: this.runtimeOptions
      });
      await readEventStream(body, (event) => this.handleRuntimeEvent(event));
    }
  }
});

function eventText(data: Record<string, unknown>): string {
  return stringValue(data.text || data.message);
}

function stringValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function numberValue(value: unknown): number {
  return typeof value === "number" ? value : 0;
}

function arrayLength(value: unknown): number {
  return Array.isArray(value) ? value.length : 0;
}

function shouldFallbackToSse(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return message.includes("code 1012") || message.includes("ACP WS 连接已关闭") || message.includes("ACP WS 连接异常");
}

function permissionOptionId(options: Array<{ id?: string; optionId?: string; name?: string }>, preferAllow: boolean): string | null {
  if (!options.length) return null;
  const selected = preferAllow
    ? options.find((item) => {
        const label = `${item.name || ""}`.toLowerCase();
        const kind = `${item.optionId || item.id || ""}`.toLowerCase();
        return label.includes("allow") || label.includes("approve") || kind.includes("allow");
      }) || options[0]
    : options[0];
  return selected.id || selected.optionId || selected.name || null;
}

function parseJsonObject(value: string): Record<string, unknown> {
  const parsed = JSON.parse(value || "{}");
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("params 必须是 JSON object");
  }
  return parsed as Record<string, unknown>;
}

function splitArgs(value: string): string[] {
  const args: string[] = [];
  const pattern = /"([^"]*)"|'([^']*)'|[^\s]+/g;
  for (const match of value.matchAll(pattern)) {
    args.push(match[1] ?? match[2] ?? match[0]);
  }
  return args;
}

function requiredInputUsesFile(item: RequiredInput): boolean {
  return ["file", "image", "video", "audio", "dataset", "model", "model_config"].includes(String(item.type || "file"));
}

function acceptForRequiredInputType(type: unknown): string {
  if (type === "image") return "image/*";
  if (type === "video") return "video/*";
  if (type === "audio") return "audio/*";
  if (type === "dataset") return ".zip,.tar,.tar.gz,.csv,.json,.jsonl,.parquet,.yaml,.yml";
  if (type === "model") return ".onnx,.pt,.pth,.bin,.safetensors,.gguf,.pkl,.joblib";
  if (type === "model_config") return ".json,.yaml,.yml,.toml";
  return "*/*";
}

function requiredInputLabel(type: unknown): string {
  if (type === "image") return "图片";
  if (type === "video") return "视频";
  if (type === "audio") return "音频";
  if (type === "dataset") return "数据集";
  if (type === "model") return "模型文件";
  if (type === "model_config") return "模型配置";
  return "文件";
}

function templateRequiresAttachment(template: AppTemplate): boolean {
  const selectedSkills = new Set(template.selected_skills || []);
  if (selectedSkills.has("data-auto-annotation")) return true;
  if (selectedSkills.has("behavior-detection") || selectedSkills.has("behavior-review")) return false;
  const text = [
    template.name,
    template.title,
    template.description,
    ...(template.selected_skills || []),
    ...(template.tags || []),
    ...(template.prompt_examples || [])
  ]
    .join(" ")
    .toLowerCase();
  return ["必须上传", "需要上传", "upload required", "image required", "annotation", "标注"].some((keyword) => text.includes(keyword));
}

function getRelativeUploadPath(file: File): string {
  return ((file as File & { webkitRelativePath?: string }).webkitRelativePath || "").trim();
}

async function runWithConcurrency<T>(
  items: T[],
  concurrency: number,
  worker: (item: T, index: number) => Promise<void>
): Promise<void> {
  let nextIndex = 0;
  const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      await worker(items[index], index);
    }
  });
  await Promise.all(workers);
}

function summarizeUploadNames(items: UploadItem[]): string {
  const names = items.slice(0, 8).map((item) => item.displayName || item.name);
  const suffix = items.length > names.length ? ` 等 ${items.length} 个文件` : "";
  return `${names.join("、")}${suffix}`;
}
