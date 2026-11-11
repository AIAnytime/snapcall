import { useEffect, useState } from "react";
import { Plus, Play, Trash2 } from "lucide-react";
import { api, type Json } from "../api";
import { Badge, Button, Card, Code, Field, Input, Select, Textarea, useToast } from "./ui";
import { ProbBars } from "./charts";

type Q = { id: string; type: string; instructions: string; options: [string, string][] };

function fromSchema(q: Json | null | undefined, id = "q1"): Q {
  if (!q) return { id, type: "choice", instructions: "", options: [["yes", ""], ["no", ""]] };
  const crit = q.criteria;
  const options: [string, string][] = Array.isArray(crit) ? crit.map((d: string, i: number) => [String(i), d]) : crit ? Object.entries(crit) as [string, string][] : [["no", ""], ["yes", ""]];
  return { id, type: q.type || "choice", instructions: q.instructions || "", options };
}

function toSchema(q: Q): Json {
  if (q.type === "noul") return { type: "noul", instructions: q.instructions };
  if (q.type === "score") return { type: "score", instructions: q.instructions, criteria: q.options.map(([, d]) => d || "level") };
  return { type: "choice", instructions: q.instructions, criteria: Object.fromEntries(q.options.filter(([k]) => k).map(([k, d]) => [k, d || k])) };
}

export default function PlaygroundPanel({ runId, meta }: { runId: string; meta?: Json }) {
  const toast = useToast();
  const [state, setState] = useState("");
  const [qs, setQs] = useState<Q[]>([fromSchema(meta?.question)]);
  const [res, setRes] = useState<Json | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setQs([fromSchema(meta?.question)]);
    setRes(null);
  }, [runId, meta?.question]);

  const body = { state, questions: Object.fromEntries(qs.map((q) => [q.id, toSchema(q)])) };
  const run = async () => {
    setBusy(true);
    try {
      setRes(await api.post(`/api/runs/${runId}/predict`, body));
    } catch (e: any) {
      toast(e.message, "err");
    } finally {
      setBusy(false);
    }
  };
  const upd = (i: number, patch: Partial<Q>) => setQs(qs.map((q, j) => (j === i ? { ...q, ...patch } : q)));

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <div className="space-y-4">
        <Field label="State (text or JSON)">
          <Textarea rows={6} value={state} onChange={(e) => setState(e.target.value)} placeholder="Paste a customer message, a JSON record, a transcript…" onKeyDown={(e) => (e.metaKey || e.ctrlKey) && e.key === "Enter" && run()} />
        </Field>
        {qs.map((q, i) => (
          <Card key={i} className="space-y-3 p-3.5">
            <div className="flex items-center gap-2">
              <Input className="max-w-[120px] font-mono" value={q.id} onChange={(e) => upd(i, { id: e.target.value })} />
              <Select className="max-w-[130px]" value={q.type} onChange={(e) => upd(i, { type: e.target.value, options: e.target.value === "noul" ? [["no", ""], ["yes", ""]] : q.options })} options={["choice", "score", "noul"]} />
              <div className="flex-1" />
              {qs.length > 1 && <Button variant="ghost" size="sm" icon={<Trash2 className="size-3.5" />} onClick={() => setQs(qs.filter((_, j) => j !== i))} />}
            </div>
            <Input value={q.instructions} onChange={(e) => upd(i, { instructions: e.target.value })} placeholder="Instructions / question" />
            {q.type !== "noul" && (
              <div className="space-y-1.5">
                {q.options.map(([n, d], k) => (
                  <div key={k} className="flex gap-1.5">
                    <Input className="max-w-[160px] font-mono text-[12px]" value={n} disabled={q.type === "score"} onChange={(e) => upd(i, { options: q.options.map((o, j) => (j === k ? [e.target.value, o[1]] : o)) })} />
                    <Input className="text-[12px]" value={d} placeholder="description" onChange={(e) => upd(i, { options: q.options.map((o, j) => (j === k ? [o[0], e.target.value] : o)) })} />
                    <button className="px-1 text-faint hover:text-danger cursor-pointer" onClick={() => upd(i, { options: q.options.filter((_, j) => j !== k) })}><Trash2 className="size-3.5" /></button>
                  </div>
                ))}
                <button className="text-[12px] text-muted hover:text-fg cursor-pointer" onClick={() => upd(i, { options: [...q.options, [q.type === "score" ? String(q.options.length) : "", ""]] })}>+ option</button>
              </div>
            )}
          </Card>
        ))}
        <div className="flex justify-between">
          <Button variant="ghost" size="sm" icon={<Plus className="size-3.5" />} onClick={() => setQs([...qs, { ...fromSchema(null, `q${qs.length + 1}`), type: "noul", options: [["no", ""], ["yes", ""]] }])}>Add question</Button>
          <Button variant="primary" icon={<Play className="size-3.5" />} loading={busy} disabled={!state} onClick={run}>Decide <span className="text-[11px] opacity-60">⌘↵</span></Button>
        </div>
      </div>
      <div className="space-y-4">
        {res ? (
          <>
            <div className="flex items-center gap-2 text-[12.5px] text-muted">
              <Badge t="accent">{res.latency_ms} ms</Badge>
              <span>{res.usage?.input_tokens} input tokens · single forward pass</span>
            </div>
            {Object.entries(res.answers).map(([qid, a]: [string, any]) => (
              <Card key={qid} className="p-4">
                <div className="mb-3 flex items-baseline justify-between gap-2">
                  <div className="font-mono text-[12px] text-faint">{qid} · {a.type}</div>
                  <div className="text-[15px] font-medium">
                    {a.type === "choice" && <>{a.choice} <span className="text-[12px] font-normal text-muted">{(a.confidence * 100).toFixed(1)}%</span></>}
                    {a.type === "score" && <>{a.score.toFixed(2)} <span className="text-[12px] font-normal text-muted">expected level</span></>}
                    {a.type === "noul" && <>{(a.noul * 100).toFixed(1)}% <span className="text-[12px] font-normal text-muted">P(yes)</span></>}
                  </div>
                </div>
                <ProbBars probs={a.probabilities ?? { yes: a.noul, no: 1 - a.noul }} top={a.choice ?? (a.type === "score" ? String(Object.entries(a.probabilities).sort((x: any, y: any) => y[1] - x[1])[0][0]) : a.noul >= 0.5 ? "yes" : "no")} />
              </Card>
            ))}
            <details className="text-[12.5px] text-muted">
              <summary className="cursor-pointer hover:text-fg">Raw response</summary>
              <Code className="mt-2">{JSON.stringify(res, null, 2)}</Code>
            </details>
          </>
        ) : (
          <Card className="flex h-full min-h-[200px] items-center justify-center p-6 text-center text-[13px] text-muted">
            Results appear here: a calibrated probability for every option.
          </Card>
        )}
        <div>
          <div className="mb-1.5 text-[12px] text-faint">System One compatible endpoint</div>
          <Code>{`curl -s ${location.origin}/v1/systemone \\\n  -H 'content-type: application/json' \\\n  -d '${JSON.stringify({ model: runId, ...body, state: body.state || "..." })}'`}</Code>
        </div>
      </div>
    </div>
  );
}
