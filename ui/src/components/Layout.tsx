import { useEffect, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { Activity, BookOpen, Cpu, Database, FlaskConical, LayoutDashboard, Menu, Moon, Plus, Settings, Sun, Trash2, X } from "lucide-react";
import { cx, useFetch } from "./ui";
import type { System } from "../api";

export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect width="32" height="32" rx="7" fill="var(--panel-2)" stroke="var(--border-strong)" />
      <path d="M9 21.5 15 10l3.2 6.1L21 11.5l2 3.7" fill="none" stroke="var(--accent)" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="23" cy="21.5" r="2" fill="var(--accent)" />
    </svg>
  );
}

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/new", label: "New run", icon: Plus },
  { to: "/runs", label: "Runs", icon: Activity },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/playground", label: "Playground", icon: FlaskConical },
  { section: "Workspace" },
  { to: "/compute", label: "Compute", icon: Cpu },
  { to: "/cleanup", label: "Cleanup", icon: Trash2 },
  { to: "/settings", label: "Settings", icon: Settings },
] as const;

function useTheme() {
  const [theme, setTheme] = useState<string>(() => {
    try {
      return localStorage.getItem("snapcall-theme") || "dark";
    } catch {
      return "dark";
    }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("snapcall-theme", theme);
    } catch {
      /* ignore */
    }
  }, [theme]);
  return [theme, setTheme] as const;
}

export default function Layout({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useTheme();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  const { data: sys } = useFetch<System>("/api/system");
  useEffect(() => setOpen(false), [loc.pathname]);
  const info = sys?.info;

  const sidebar = (
    <aside className="flex h-full w-[232px] shrink-0 flex-col border-r border-line bg-panel">
      <div className="flex h-14 items-center gap-2.5 px-4">
        <Logo className="size-7" />
        <div className="leading-tight">
          <div className="text-[14px] font-semibold tracking-tight">Snapcall</div>
          <div className="text-[11px] text-faint">Build better decision models</div>
        </div>
      </div>
      <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 py-2">
        {NAV.map((n, i) =>
          "section" in n ? (
            <div key={i} className="px-2.5 pb-1.5 pt-5 text-[11px] font-medium uppercase tracking-wider text-faint">{n.section}</div>
          ) : (
            <NavLink
              key={n.to}
              to={n.to}
              end={"end" in n}
              className={({ isActive }) =>
                cx("flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] transition-colors", isActive ? "bg-hover text-fg" : "text-muted hover:bg-hover/60 hover:text-fg")
              }
            >
              <n.icon className="size-4 shrink-0" strokeWidth={1.75} />
              {n.label}
            </NavLink>
          ),
        )}
      </nav>
      <div className="border-t border-line p-3">
        <div className="flex items-center justify-between">
          <a href="https://www.youtube.com/@AIAnytime" target="_blank" rel="noreferrer" className="text-[12px] text-faint hover:text-fg">
            Built by <span className="text-muted">AI Anytime</span>
          </a>
          <button onClick={() => setTheme(theme === "dark" ? "light" : "dark")} className="rounded p-1 text-faint hover:bg-hover hover:text-fg cursor-pointer" title="Toggle theme">
            {theme === "dark" ? <Sun className="size-3.5" /> : <Moon className="size-3.5" />}
          </button>
        </div>
        <div className="mt-1 text-[11px] text-faint">v{sys?.version ?? "…"}</div>
      </div>
    </aside>
  );

  return (
    <div className="flex h-full">
      <div className="hidden md:block">{sidebar}</div>
      {open && (
        <div className="fixed inset-0 z-40 flex md:hidden">
          <div className="absolute inset-0 bg-black/50" onClick={() => setOpen(false)} />
          <div className="relative">{sidebar}</div>
          <button className="absolute right-4 top-4 text-fg" onClick={() => setOpen(false)}><X /></button>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-line px-4 md:px-8">
          <button className="md:hidden text-muted" onClick={() => setOpen(true)}><Menu className="size-5" /></button>
          <div className="hidden md:block" />
          <div className="flex items-center gap-2">
            {info && (
              <NavLink to="/compute" className="flex items-center gap-2 rounded-md border border-line bg-panel px-2.5 py-1 text-[12px] text-muted hover:text-fg">
                <span className={cx("size-1.5 rounded-full", info.device === "cpu" ? "bg-warn" : "bg-accent")} />
                <span className="font-mono">{info.device.toUpperCase()}</span>
                <span className="hidden sm:inline text-faint">·</span>
                <span className="hidden sm:inline truncate max-w-[180px]">{info.accelerator}</span>
                {info.vram_gb > 0 && <span className="text-faint">{info.vram_gb} GB</span>}
              </NavLink>
            )}
            <a href="https://github.com/AIAnytime/snapcall#readme" target="_blank" rel="noreferrer" className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[12px] text-muted hover:text-fg">
              <BookOpen className="size-3.5" /> Docs
            </a>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[1200px] px-4 py-7 md:px-8">{children}</div>
        </main>
      </div>
    </div>
  );
}
