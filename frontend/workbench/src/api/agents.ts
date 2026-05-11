import { authHeaders, readJson } from "./http";
import type { AgentSummary, RunRequest } from "./types";

export async function listAgents(): Promise<AgentSummary[]> {
  const payload = await readJson<{ agents?: AgentSummary[] }>(await fetch("/api/agents", { cache: "no-store" }));
  return payload.agents || [];
}

export async function startSseRun(
  agentName: string,
  token: string,
  request: RunRequest
): Promise<ReadableStream<Uint8Array>> {
  const response = await fetch(`/api/agents/${encodeURIComponent(agentName)}/runs/stream`, {
    method: "POST",
    headers: authHeaders(token, { "Content-Type": "application/json" }),
    body: JSON.stringify(request)
  });
  if (!response.ok || !response.body) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(typeof payload?.detail === "string" ? payload.detail : response.statusText || `HTTP ${response.status}`);
  }
  return response.body;
}
