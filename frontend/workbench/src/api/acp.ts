import { tokenQuery } from "./http";
import type { AgentRunResult, AttachmentRef, ChatMessage, RuntimeOptions } from "./types";

type JsonRpcId = string | number;
type RpcResolver = {
  resolve: (value: Record<string, unknown>) => void;
  reject: (error: Error) => void;
};

type PermissionRequest = {
  sessionId: string;
  request?: {
    id?: string;
    title?: string;
    description?: string;
    options?: Array<{ id?: string; optionId?: string; name?: string }>;
  };
};

type PermissionDecision = {
  approved: boolean;
  selectedOptionId?: string | null;
};

export type AcpClientOptions = {
  token: string;
  onEvent: (event: unknown) => void;
  onPermissionRequest?: (request: PermissionRequest) => Promise<PermissionDecision>;
};

export type AcpPromptOptions = {
  agentName: string;
  content: string;
  threadId: string;
  messages: ChatMessage[];
  attachments: AttachmentRef[];
  runtimeOptions: RuntimeOptions;
};

export class AcpClient {
  private socket: WebSocket | null = null;
  private pending = new Map<JsonRpcId, RpcResolver>();
  private rpcId = 0;
  private connecting: Promise<void> | null = null;

  constructor(private readonly options: AcpClientOptions) {}

  get connected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  async connect(): Promise<void> {
    if (this.connected) return;
    if (this.connecting) return this.connecting;
    this.connecting = this.open().finally(() => {
      this.connecting = null;
    });
    return this.connecting;
  }

  async initialize(): Promise<Record<string, unknown>> {
    await this.connect();
    return this.rpc("initialize", {
      protocolVersion: 1,
      clientInfo: { name: "jetlinks-workbench-vue", version: "0.1.0" }
    });
  }

  async newSession(params: Record<string, unknown>): Promise<Record<string, unknown>> {
    await this.connect();
    return this.rpc("new_session", params);
  }

  async prompt(options: AcpPromptOptions & { sessionId: string }): Promise<AgentRunResult | null> {
    const result = await this.rpc("prompt", {
      sessionId: options.sessionId,
      agentName: options.agentName,
      threadId: options.threadId,
      prompt: [{ type: "text", text: options.content }],
      messages: options.messages,
      attachments: options.attachments,
      runtimeOptions: options.runtimeOptions
    });
    return (result.result || result) as AgentRunResult;
  }

  async rpc(method: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    await this.connect();
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error("ACP WS 未连接");
    }
    const id = ++this.rpcId;
    return new Promise<Record<string, unknown>>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      try {
        socket.send(JSON.stringify({ jsonrpc: "2.0", id, method, params }));
      } catch (error) {
        this.pending.delete(id);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  close(): void {
    const socket = this.socket;
    this.socket = null;
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
      socket.close();
    }
    this.rejectPending(new Error("ACP WS 连接已关闭"));
  }

  private open(): Promise<void> {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/api/acp/ws${tokenQuery(this.options.token)}`;
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(url, "acp.v1");
      let settled = false;
      const fail = (message: string) => {
        if (settled) return;
        settled = true;
        reject(new Error(message));
      };
      socket.onopen = () => {
        settled = true;
        this.socket = socket;
        resolve();
      };
      socket.onerror = () => fail("ACP WS 连接失败");
      socket.onclose = (event) => {
        if (!settled) {
          fail(`ACP WS 连接关闭：${event.reason || `code ${event.code}`}`);
          return;
        }
        if (this.socket === socket) this.socket = null;
        this.rejectPending(new Error(`ACP WS 连接已关闭：${event.reason || `code ${event.code}`}`));
      };
      socket.onmessage = (message) => this.handleMessage(message);
    });
  }

  private handleMessage(message: MessageEvent): void {
    let packet: Record<string, unknown>;
    try {
      packet = JSON.parse(String(message.data)) as Record<string, unknown>;
    } catch {
      this.rejectPending(new Error("ACP WS 返回了无法解析的消息"));
      this.socket?.close();
      return;
    }

    if (packet.method === "session/update") {
      const params = packet.params as
        | {
            sessionId?: string;
            update?: {
              sessionUpdate?: string;
              permissionRequest?: PermissionRequest["request"];
              _meta?: { jetlinksRuntimeEvent?: unknown };
            };
          }
        | undefined;
      if (params?.update?.sessionUpdate === "permission_request" && this.options.onPermissionRequest) {
        void this.options
          .onPermissionRequest({ sessionId: params.sessionId || "", request: params.update.permissionRequest })
          .then((decision) =>
            this.rpc("session/request_permission", {
              sessionId: params.sessionId || "",
              permissionRequestId: params.update?.permissionRequest?.id,
              approved: decision.approved,
              ...(decision.selectedOptionId ? { selectedOptionId: decision.selectedOptionId } : {})
            })
          )
          .catch((error) => {
            this.options.onEvent({
              type: "permission.failed",
              data: { error: error instanceof Error ? error.message : String(error) }
            });
          });
      }
      const event = params?.update?._meta?.jetlinksRuntimeEvent;
      if (event) this.options.onEvent(event);
      return;
    }

    const id = packet.id as JsonRpcId | undefined;
    if (id == null || !this.pending.has(id)) return;
    const request = this.pending.get(id);
    this.pending.delete(id);
    const error = packet.error as { message?: string } | undefined;
    if (error) request?.reject(new Error(error.message || "ACP WS error"));
    else request?.resolve((packet.result || {}) as Record<string, unknown>);
  }

  private rejectPending(error: Error): void {
    this.pending.forEach((request) => request.reject(error));
    this.pending.clear();
  }
}

export async function runAcpPrompt(
  options: AcpPromptOptions & AcpClientOptions
): Promise<AgentRunResult | null> {
  const client = new AcpClient(options);
  try {
    await client.initialize();
    const session = await client.newSession({
      cwd: "/",
      sessionId: options.threadId,
      agentName: options.agentName,
      threadId: options.threadId
    });
    return await client.prompt({ ...options, sessionId: String(session.sessionId || "") });
  } finally {
    client.close();
  }
}
