import { RefreshCw } from "lucide-react";
import { useState } from "react";
import { api, type Model, type System } from "../api";
import { Alert, Badge, Button, Card, CardHeader, PageHeader, Stat, useFetch } from "../components/ui";

const tone = { fits: "accent", tight: "warn", slow: "warn", no: "danger", cpu: "warn" } as const;

export default function Compute() {
  const s = useFetch<System>("/api/system");
  const { data: cat } = useFetch<{ models: Model[] }>("/api/catalog");
  const [busy, setBusy] = useState(false);
  const sys = s.data;
  if (!sys) return <div className="text-muted">Detecting hardware…</div>;
  const i = sys.info;
  const refresh = async () => {
    setBusy(true);
    s.setData(await api.get("/api/system?refresh=true"));
    setBusy(false);
  };
  return (
    <div className="space-y-6">
      <PageHeader title="Compute" description="What this machine can train, and how." actions={<Button size="sm" icon={<RefreshCw className="size-3.5" />} loading={busy} onClick={refresh}>Re-detect</Button>} />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Accelerator" value={<span className="text-[16px]">{i.accelerator}</span>} sub={`device: ${i.device}`} />
        <Stat label="Training memory" value={`${sys.recommendations.memory_budget_gb} GB`} sub={i.device === "mps" ? "≈65% of unified memory" : i.device === "cuda" ? "GPU VRAM" : "50% of RAM"} />
        <Stat label="System RAM" value={`${i.ram_gb} GB`} sub={`${i.ram_free_gb} GB free · ${i.cpu_cores} cores`} />
        <Stat label="Disk free" value={i.disk_free_gb ? `${i.disk_free_gb} GB` : "—"} sub="in SNAPCALL_HOME" />
      </div>
      <div className="flex flex-wrap gap-2 text-[12px]">
        <Badge>torch {i.torch}</Badge><Badge>transformers {i.transformers}</Badge><Badge>python {i.python}</Badge>
        <Badge t={i.bf16 ? "accent" : "neutral"}>bf16 {i.bf16 ? "yes" : "no"}</Badge><Badge t={i.qlora ? "accent" : "neutral"}>QLoRA {i.qlora ? "available" : "unavailable"}</Badge>
        <Badge t="info">tier: {sys.recommendations.tier}</Badge>
      </div>
      {sys.recommendations.tips.map((t) => <Alert key={t}>{t}</Alert>)}
      <Card>
        <CardHeader title="Model fit on this machine" description="Best fine-tuning mode per model and estimated peak memory at batch 8 × 512 tokens." />
        <table className="w-full text-[13px]">
          <thead className="text-left text-[12px] text-faint"><tr className="border-b border-line"><th className="px-4 py-2 font-normal">Model</th><th className="px-2 font-normal">Params</th><th className="px-2 font-normal">Mode</th><th className="px-2 font-normal">Memory</th><th className="px-4 font-normal">Fit</th></tr></thead>
          <tbody>
            {sys.recommendations.models.map((f) => {
              const m = cat?.models.find((x) => x.id === f.id);
              return (
                <tr key={f.id} className="border-b border-line last:border-0">
                  <td className="px-4 py-2"><div>{m?.name}</div><div className="font-mono text-[11.5px] text-faint">{f.id}</div></td>
                  <td className="px-2 font-mono tabular text-muted">{m ? (m.params < 1 ? `${Math.round(m.params * 1000)}M` : `${m.params.toFixed(1)}B`) : ""}</td>
                  <td className="px-2"><Badge>{f.finetune}</Badge></td>
                  <td className="px-2 font-mono tabular">
                    <div className="flex items-center gap-2">~{f.memory_gb} GB<div className="h-1 w-20 rounded-full bg-panel-2"><div className="h-1 rounded-full" style={{ width: `${Math.min(100, (f.memory_gb / Math.max(0.1, sys.recommendations.memory_budget_gb)) * 100)}%`, background: f.fit === "no" ? "var(--danger)" : "var(--accent)" }} /></div></div>
                  </td>
                  <td className="px-4"><Badge t={tone[f.fit]}>{f.fit}</Badge></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
