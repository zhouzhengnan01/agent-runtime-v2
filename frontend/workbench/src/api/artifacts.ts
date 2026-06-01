import { readJson } from "./http";
import type { Artifact } from "./types";

export async function listArtifacts(threadId: string, prefix?: string): Promise<Artifact[]> {
  const params = new URLSearchParams();
  if (prefix) params.set("prefix", prefix);
  const query = params.toString();
  const payload = await readJson<{ artifacts?: Artifact[] }>(
    await fetch(`/api/artifacts/${encodeURIComponent(threadId)}${query ? `?${query}` : ""}`, { cache: "no-store" })
  );
  return payload.artifacts || [];
}
