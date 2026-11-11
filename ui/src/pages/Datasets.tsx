import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Database, Play, Plus, Trash2 } from "lucide-react";
import { api, fmt, type DatasetMeta, type Json } from "../api";
import { Badge, Button, Card, CardHeader, Code, Drawer, Empty, PageHeader, Segmented, useFetch, useToast } from "../components/ui";
import DatasetCreator from "../components/DatasetCreator";

export default function Datasets() {
  const nav = useNavigate();
  const toast = useToast();
  const list = useFetch<DatasetMeta[]>("/api/datasets");
  const [open, setOpen] = useState(false);
  const [sel, setSel] = useState<string | null>(null);
  const [split, setSplit] = useState<"train" | "validation" | "test">("train");
  const detail = useFetch<{ meta: DatasetMeta; rows: Json[] }>(sel ? `/api/datasets/${sel}?split=${split}&n=40` : null, [split]);

  const del = async (name: string) => {
    if (!confirm(`Delete dataset ${name}?`)) return;
    await api.del(`/api/datasets/${name}`);
    toast("Dataset deleted");
    setSel(null);
    list.reload();
  };

  const m = detail.data?.meta;
  const labels = m ? Object.entries(m.label_counts) : [];
  const maxCount = Math.max(1, ...labels.map(([, c]) => c));

  return (
    <div>
      <PageHeader title="Datasets" description="System One JSONL: a state, a typed question, an answer. Stored in ~/.snapcall/datasets." actions={<Button variant="primary" icon={<Plus className="size-3.5" />} onClick={() => setOpen(true)}>New dataset</Button>} />
      {list.data?.length === 0 && <Empty icon={<Database className="size-6" />} title="No datasets yet" action={<Button variant="primary" onClick={() => setOpen(true)}>Create a dataset</Button>}>Import Banking77 in one click, upload a CSV, or describe a use case and let an LLM write the data.</Empty>}
      <div className="grid gap-5 lg:grid-cols-[340px_1fr]">
        {!!list.data?.length && (
          <div className="space-y-2">
            {list.data.map((d) => (
              <button key={d.name} onClick={() => setSel(d.name)} className={`w-full rounded-lg border p-3.5 text-left transition cursor-pointer ${sel === d.name ? "border-line-strong bg-hover" : "border-line bg-panel hover:border-line-strong"}`}>
                <div className="flex items-center justify-between gap-2"><span className="truncate text-[13px] font-medium">{d.name}</span><Badge>{d.source.kind}</Badge></div>
                <div className="mt-1.5 flex gap-3 text-[12px] text-muted"><span>{Object.keys(d.question_types).join(", ")}</span><span>{d.num_labels} labels</span><span className="tabular">{(d.counts.train + d.counts.validation + d.counts.test).toLocaleString()} rows</span><span>{fmt.ago(d.created)}</span></div>
              </button>
            ))}
          </div>
        )}
        {m && (
          <div className="space-y-4">
            <Card>
              <CardHeader title={m.name} description={m.example_question?.instructions} right={<div className="flex gap-2"><Button size="sm" variant="primary" icon={<Play className="size-3" />} onClick={() => nav(`/new?dataset=${m.name}`)}>Train on this</Button><Button size="sm" variant="ghost" icon={<Trash2 className="size-3.5" />} onClick={() => del(m.name)} /></div>} />
              <div className="grid gap-4 p-4 md:grid-cols-[1fr_1fr]">
                <div className="grid grid-cols-3 gap-2 text-center">
                  {(["train", "validation", "test"] as const).map((s) => <div key={s} className="rounded-md border border-line bg-bg py-2.5"><div className="text-[18px] font-medium tabular">{m.counts[s].toLocaleString()}</div><div className="text-[11.5px] text-faint">{s}</div></div>)}
                  <div className="col-span-3 mt-1 text-left text-[12px] text-muted">State length p50 {m.state_chars_p50} · p95 {m.state_chars_p95} chars · {m.size_mb ?? "?"} MB · source {m.source.hf_id || m.source.filename || m.source.kind}</div>
                </div>
                <div className="max-h-44 space-y-1 overflow-y-auto pr-1">
                  {labels.slice(0, 60).map(([l, c]) => (
                    <div key={l} className="grid grid-cols-[minmax(0,140px)_1fr_40px] items-center gap-2 text-[12px]">
                      <span className="truncate text-muted">{l}</span>
                      <div className="h-1.5 rounded-full bg-panel-2"><div className="h-1.5 rounded-full bg-accent/70" style={{ width: `${(c / maxCount) * 100}%` }} /></div>
                      <span className="text-right font-mono tabular text-faint">{c}</span>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
            <Card>
              <CardHeader title="Rows" right={<Segmented value={split} onChange={setSplit} options={[{ id: "train", label: "train" }, { id: "validation", label: "validation" }, { id: "test", label: "test" }]} />} />
              <div className="divide-y divide-line">
                {(detail.data?.rows ?? []).map((r, i) => (
                  <div key={i} className="flex gap-3 px-4 py-2 text-[12.5px]">
                    <Badge t="accent" className="shrink-0 self-start">{String(r.answer)}</Badge>
                    <span className="line-clamp-2 text-muted">{typeof r.state === "string" ? r.state : JSON.stringify(r.state)}</span>
                  </div>
                ))}
              </div>
            </Card>
            {m.example_question && <details className="text-[12.5px] text-muted"><summary className="cursor-pointer hover:text-fg">Question schema</summary><Code className="mt-2">{JSON.stringify(m.example_question, null, 2)}</Code></details>}
          </div>
        )}
        {!m && !!list.data?.length && <Card className="flex items-center justify-center p-10 text-[13px] text-muted">Select a dataset to inspect it.</Card>}
      </div>
      <Drawer open={open} onClose={() => setOpen(false)} title="New dataset" width={920}>
        <DatasetCreator onCreated={(meta) => { setOpen(false); list.reload(); setSel(meta.name); }} />
      </Drawer>
    </div>
  );
}
