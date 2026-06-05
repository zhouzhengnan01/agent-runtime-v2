import { authHeaders, readJson } from "./http";
import type { AttachmentRef } from "./types";

type UploadOptions = {
  timeoutMs?: number;
  relativePath?: string;
  onProgress?: (progress: number) => void;
};

export async function uploadThreadFile(
  threadId: string,
  token: string,
  file: File,
  options: UploadOptions = {}
): Promise<AttachmentRef> {
  options.onProgress?.(10);
  const form = new FormData();
  form.append("file", file);
  if (options.relativePath) form.append("relative_path", options.relativePath);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), options.timeoutMs || 120000);
  const response = await fetch(`/api/uploads/${encodeURIComponent(threadId)}`, {
    method: "POST",
    headers: authHeaders(token),
    body: form,
    signal: controller.signal
  }).finally(() => window.clearTimeout(timeout));
  const payload = await readJson<AttachmentRef>(response);
  options.onProgress?.(100);
  return payload;
}
