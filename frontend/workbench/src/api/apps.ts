import { readJson } from "./http";
import type { AppTemplate } from "./types";

export async function listAppTemplates(): Promise<AppTemplate[]> {
  const payload = await readJson<{ templates?: AppTemplate[] }>(
    await fetch("/api/apps/templates", { cache: "no-store" })
  );
  return payload.templates || [];
}
