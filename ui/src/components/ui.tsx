import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { Check, Copy, Info, Loader2, X } from "lucide-react";

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

type BtnProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
  loading?: boolean;
  icon?: ReactNode;
};
export function Button({ variant = "secondary", size = "md", loading, icon, className, children, disabled, ...p }: BtnProps) {
  const v = {
    primary: "bg-accent text-accent-fg hover:brightness-110 border border-transparent",
    secondary: "bg-panel-2 text-fg border border-line hover:border-line-strong hover:bg-hover",
    ghost: "text-muted hover:text-fg hover:bg-hover border border-transparent",
    danger: "bg-danger-soft text-danger border border-transparent hover:border-danger/40",
  }[variant];
  const s = size === "sm" ? "h-7 px-2.5 text-[12.5px] gap-1.5" : "h-8.5 px-3.5 text-[13px] gap-2";
  return (
    <button
      {...p}
      disabled={disabled || loading}
      className={cx("inline-flex items-center justify-center rounded-md font-medium transition-colors whitespace-nowrap disabled:opacity-50 disabled:pointer-events-none cursor-pointer", v, s, className)}
    >
      {loading ? <Loader2 className="size-3.5 animate-spin" /> : icon}
      {children}
    </button>
  );
}

export function Card({ className, children, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div {...p} className={cx("rounded-lg border border-line bg-panel", className)}>
      {children}
    </div>
  );
}

export function CardHeader({ title, description, right }: { title: ReactNode; description?: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-line px-4 py-3">
      <div className="min-w-0">
        <div className="text-[13px] font-medium">{title}</div>
        {description && <div className="mt-0.5 text-[12.5px] text-muted">{description}</div>}
      </div>
      {right}
    </div>
  );
}

const tone = {
  neutral: "bg-panel-2 text-muted border-line",
  accent: "bg-accent-soft text-accent border-transparent",
  danger: "bg-danger-soft text-danger border-transparent",
  warn: "bg-warn-soft text-warn border-transparent",
  info: "bg-info-soft text-info border-transparent",
};
export function Badge({ children, t = "neutral", className }: { children: ReactNode; t?: keyof typeof tone; className?: string }) {
  return <span className={cx("inline-flex items-center gap-1 rounded border px-1.5 py-px text-[11.5px] font-medium whitespace-nowrap", tone[t], className)}>{children}</span>;
}

export function StatusBadge({ state }: { state?: string }) {
  const map: Record<string, keyof typeof tone> = { completed: "accent", running: "info", queued: "neutral", failed: "danger", stopped: "warn" };
  return (
    <Badge t={map[state || ""] || "neutral"}>
      <span className={cx("size-1.5 rounded-full bg-current", state === "running" && "pulse-dot")} />
      {state || "unknown"}
    </Badge>
  );
}

export function Input(p: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...p} className={cx("h-8.5 w-full rounded-md border border-line bg-bg px-2.5 text-[13px] text-fg placeholder:text-faint outline-none transition focus:border-line-strong focus:ring-2 focus:ring-accent/20", p.className)} />;
}
export function Textarea(p: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...p} className={cx("w-full rounded-md border border-line bg-bg px-2.5 py-2 text-[13px] text-fg placeholder:text-faint outline-none transition focus:border-line-strong focus:ring-2 focus:ring-accent/20", p.className)} />;
}
export function Select({ options, ...p }: React.SelectHTMLAttributes<HTMLSelectElement> & { options: (string | { value: string; label: string })[] }) {
  return (
    <select {...p} className={cx("h-8.5 w-full rounded-md border border-line bg-bg px-2 text-[13px] text-fg outline-none focus:border-line-strong", p.className)}>
      {options.map((o) => (typeof o === "string" ? <option key={o} value={o}>{o}</option> : <option key={o.value} value={o.value}>{o.label}</option>))}
    </select>
  );
}

export function Field({ label, hint, children, onInfo, active }: { label: ReactNode; hint?: ReactNode; children: ReactNode; onInfo?: () => void; active?: boolean }) {
  return (
    <label className={cx("block rounded-md", active && "")} onFocus={onInfo}>
      <div className="mb-1.5 flex items-center gap-1.5 text-[12.5px] text-muted">
        <span className={cx(active && "text-fg")}>{label}</span>
        {onInfo && (
          <button type="button" onClick={onInfo} className="text-faint hover:text-accent" tabIndex={-1}>
            <Info className="size-3.5" />
          </button>
        )}
      </div>
      {children}
      {hint && <div className="mt-1 text-[12px] text-faint">{hint}</div>}
    </label>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label?: ReactNode }) {
  return (
    <button type="button" onClick={() => onChange(!checked)} className="inline-flex items-center gap-2 text-[13px] cursor-pointer">
      <span className={cx("relative h-[18px] w-8 rounded-full border transition-colors", checked ? "bg-accent border-accent" : "bg-panel-2 border-line-strong")}>
        <span className={cx("absolute top-[1px] size-3.5 rounded-full transition-all", checked ? "left-[15px] bg-accent-fg" : "left-[1px] bg-muted")} />
      </span>
      {label}
    </button>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange, className }: { tabs: { id: T; label: ReactNode; icon?: ReactNode }[]; value: T; onChange: (v: T) => void; className?: string }) {
  return (
    <div className={cx("flex items-center gap-1 border-b border-line overflow-x-auto", className)}>
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={cx(
            "relative -mb-px flex items-center gap-1.5 px-3 py-2 text-[13px] transition-colors whitespace-nowrap cursor-pointer border-b",
            value === t.id ? "border-fg text-fg" : "border-transparent text-muted hover:text-fg",
          )}
        >
          {t.icon}
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function Segmented<T extends string>({ options, value, onChange }: { options: { id: T; label: ReactNode }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex rounded-md border border-line bg-bg p-0.5">
      {options.map((o) => (
        <button key={o.id} onClick={() => onChange(o.id)} className={cx("rounded px-2.5 py-1 text-[12.5px] cursor-pointer", value === o.id ? "bg-panel-2 text-fg shadow-sm border border-line" : "text-muted hover:text-fg border border-transparent")}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Stat({ label, value, sub, accent }: { label: ReactNode; value: ReactNode; sub?: ReactNode; accent?: boolean }) {
  return (
    <Card className="px-4 py-3.5">
      <div className="text-[12px] text-muted">{label}</div>
      <div className={cx("mt-1 text-[22px] font-medium tracking-tight tabular", accent && "text-accent")}>{value}</div>
      {sub && <div className="mt-0.5 text-[12px] text-faint">{sub}</div>}
    </Card>
  );
}

export function Progress({ value, className }: { value: number; className?: string }) {
  return (
    <div className={cx("h-1.5 w-full overflow-hidden rounded-full bg-panel-2", className)}>
      <div className="h-full rounded-full bg-accent transition-all duration-500" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

export function Empty({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-line px-6 py-14 text-center">
      {icon && <div className="mb-3 text-faint">{icon}</div>}
      <div className="text-[14px] font-medium">{title}</div>
      {children && <div className="mt-1 max-w-md text-[13px] text-muted">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cx("size-4 animate-spin text-muted", className)} />;
}

export function Code({ children, className }: { children: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className={cx("group relative rounded-md border border-line bg-bg", className)}>
      <pre className="overflow-x-auto p-3 font-mono text-[12px] leading-relaxed text-muted">{children}</pre>
      <button
        onClick={() => {
          navigator.clipboard?.writeText(children);
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        }}
        className="absolute right-2 top-2 rounded border border-line bg-panel p-1 text-faint opacity-0 transition group-hover:opacity-100 hover:text-fg cursor-pointer"
      >
        {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
      </button>
    </div>
  );
}

export function PageHeader({ title, description, actions }: { title: ReactNode; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-[20px] font-medium tracking-tight">{title}</h1>
        {description && <p className="mt-1 text-[13px] text-muted">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Drawer({ open, onClose, title, children, width = 640 }: { open: boolean; onClose: () => void; title: ReactNode; children: ReactNode; width?: number }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative flex h-full w-full flex-col border-l border-line bg-panel shadow-2xl" style={{ maxWidth: width }}>
        <div className="flex items-center justify-between border-b border-line px-5 py-3.5">
          <div className="text-[14px] font-medium">{title}</div>
          <button onClick={onClose} className="text-muted hover:text-fg cursor-pointer"><X className="size-4" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-5">{children}</div>
      </div>
    </div>
  );
}

export function Alert({ t = "info", children }: { t?: "info" | "warn" | "danger" | "accent"; children: ReactNode }) {
  const c = { info: "bg-info-soft text-info", warn: "bg-warn-soft text-warn", danger: "bg-danger-soft text-danger", accent: "bg-accent-soft text-accent" }[t];
  return <div className={cx("rounded-md px-3 py-2 text-[12.5px] leading-relaxed", c)}>{children}</div>;
}

// --- toasts --------------------------------------------------------------
type Toast = { id: number; msg: string; t: "ok" | "err" };
const ToastCtx = createContext<(msg: string, t?: "ok" | "err") => void>(() => {});
export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const push = (msg: string, t: "ok" | "err" = "ok") => {
    const id = Date.now() + Math.random();
    setItems((x) => [...x, { id, msg, t }]);
    setTimeout(() => setItems((x) => x.filter((i) => i.id !== id)), 4500);
  };
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="fixed bottom-4 right-4 z-[60] flex flex-col gap-2">
        {items.map((i) => (
          <div key={i.id} className={cx("max-w-sm rounded-md border bg-panel px-3.5 py-2.5 text-[13px] shadow-xl", i.t === "err" ? "border-danger/40 text-danger" : "border-line text-fg")}>
            {i.msg}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}
export const useToast = () => useContext(ToastCtx);

// --- data hook -------------------------------------------------------------
export function useFetch<T>(url: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    if (!url) return;
    let alive = true;
    setLoading(true);
    fetch(url)
      .then(async (r) => {
        const j = await r.json();
        if (!r.ok) throw new Error(j?.detail || r.statusText);
        return j;
      })
      .then((j) => alive && (setData(j), setError(null)))
      .catch((e) => alive && setError(String(e.message || e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, nonce, ...deps]);
  return { data, error, loading, reload: () => setNonce((n) => n + 1), setData };
}
