export type Json = Record<string, any>;

async function req<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: {} };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    (init.headers as Record<string, string>)["content-type"] = "application/json";
  }
  const r = await fetch(url, init);
  const text = await r.text();
  const data = text ? JSON.parse(text) : null;
  if (!r.ok) {
    const d = data?.detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x: any) => `${x.loc?.slice(-1)[0]}: ${x.msg}`).join("; ") : r.statusText;
    throw new Error(msg);
  }
  return data as T;
}

export const api = {
  get: <T = any>(u: string) => req<T>("GET", u),
  post: <T = any>(u: string, b?: unknown) => req<T>("POST", u, b ?? {}),
  del: <T = any>(u: string) => req<T>("DELETE", u),
};

export type Model = {
  id: string; name: string; params: number; arch: string; kind: string; langs: string; context: number;
  license: string; support: string; family: string; notes: string; trust_remote_code?: boolean;
};
export type Preset = { id: string; name: string; badge: string; description: string; config: Json };
export type Fit = { id: string; finetune: string; memory_gb: number; fit: "fits" | "tight" | "no" | "slow" | "cpu" };
export type System = {
  info: Json;
  recommendations: { tier: string; presets: Preset[]; models: Fit[]; tips: string[]; memory_budget_gb: number };
  version: string;
};
export type Explainer = Record<string, { group: string; label: string; short: string; detail: string; typical: string; up: string; down: string; tip: string }>;

export type RunStatus = {
  state?: "queued" | "running" | "completed" | "failed" | "stopped";
  stage?: string; step?: number; total_steps?: number; progress?: number; eta_s?: number; elapsed_s?: number;
  error?: string | null; created?: number; started?: number; ended?: number; best?: Json; summary?: Json; model_info?: Json;
};
export type Run = {
  id: string; status: RunStatus; config: Json; has_model: boolean; created: number;
  model_meta?: Json; eval?: Json; pushed?: Json;
};
export type DatasetMeta = {
  name: string; created: number; source: Json; counts: { train: number; validation: number; test: number };
  question_types: Record<string, number>; label_counts: Record<string, number>; num_labels: number;
  state_chars_p50: number; state_chars_p95: number; example_question: Json | null; size_mb?: number;
};
export type Job = { id: string; state: "running" | "completed" | "failed"; progress: number; message: string; result: any; error: string | null };

export async function waitJob(id: string, onUpdate?: (j: Job) => void): Promise<Job> {
  for (;;) {
    const j = await api.get<Job>(`/api/jobs/${id}`);
    onUpdate?.(j);
    if (j.state !== "running") {
      if (j.state === "failed") throw new Error(j.error || "Job failed");
      return j;
    }
    await new Promise((r) => setTimeout(r, 800));
  }
}

export const fmt = {
  num: (v: any, d = 4) => (typeof v === "number" ? v.toFixed(d) : "—"),
  pct: (v: any, d = 1) => (typeof v === "number" ? `${(v * 100).toFixed(d)}%` : "—"),
  bytes: (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} KB`),
  params: (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : `${(n / 1e6).toFixed(1)}M`),
  dur: (s?: number) => {
    if (s === undefined || s === null || !isFinite(s)) return "—";
    if (s < 60) return `${Math.round(s)}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
    return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
  },
  ago: (t?: number) => {
    if (!t) return "—";
    const s = Date.now() / 1000 - t;
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    return new Date(t * 1000).toLocaleDateString();
  },
};
