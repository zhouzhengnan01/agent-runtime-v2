import { readJson } from "./http";

export type SandboxStatus = {
  available?: boolean;
  enabled?: boolean;
  profiles?: unknown[];
  error?: string;
};

export async function getSandboxStatus(): Promise<SandboxStatus> {
  return readJson<SandboxStatus>(await fetch("/api/sandbox/status", { cache: "no-store" }));
}
