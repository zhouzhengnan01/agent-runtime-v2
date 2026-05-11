const ADMIN_TOKEN_KEY = "jetlinks.runtime.adminToken";

export function loadAdminToken(): string {
  return localStorage.getItem(ADMIN_TOKEN_KEY) || "";
}

export function saveAdminToken(token: string): void {
  if (token) {
    localStorage.setItem(ADMIN_TOKEN_KEY, token);
  } else {
    localStorage.removeItem(ADMIN_TOKEN_KEY);
  }
}

export function authHeaders(token: string, headers: HeadersInit = {}): Headers {
  const next = new Headers(headers);
  if (token && !next.has("Authorization")) {
    next.set("Authorization", `Bearer ${token}`);
  }
  return next;
}

export async function readJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof payload?.detail === "string" ? payload.detail : response.statusText || `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return payload as T;
}

export function tokenQuery(token: string): string {
  return token ? `?token=${encodeURIComponent(token)}` : "";
}
