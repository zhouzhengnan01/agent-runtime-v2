import { readJson } from "./http";
import type { Artifact } from "./types";

export async function listArtifacts(threadId: string): Promise<Artifact[]> {
  const payload = await readJson<{ artifacts?: Artifact[] }>(
    await fetch(`/api/artifacts/${encodeURIComponent(threadId)}`, { cache: "no-store" })
  );
  return payload.artifacts || [];
}
