import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Activity, BarChart3, Box, CloudUpload, ExternalLink, FileText, FlaskConical, FolderOpen, Gauge, Square, Terminal, Trash2 } from "lucide-react";
import { api, fmt, waitJob, type Json, type Run } from "../api";
import { Alert, Badge, Button, Card, CardHeader, Code, Empty, Field, Input, Progress, Select, Stat, StatusBadge, Tabs, Textarea, Toggle, useFetch, useToast } from "../components/ui";
import { Confusion, MetricChart, Reliability, type Series } from "../components/charts";
import PlaygroundPanel from "../components/PlaygroundPanel";

type Tab = "metrics" | "logs" | "eval" | "model" | "test" | "publish" | "files";
const C = { a: "var(--chart-1)", b: "var(--chart-2)", c: "var(--chart-3)", d: "var(--chart-4)" };

export default function RunDetail() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const toast = useToast();
  const [run, setRun] = useState<Run | null>(null);
  const [rows, setRows] = useState<Json[]>([]);
  const [log, setLog] = useState("");
  const [tab, setTab] = useState<Tab>("metrics");
  const [smooth, setSmooth] = useState(0.6);
  const offset = useRef(0);

  const loadRun = () => api.get<Run>(`/api/runs/${id}`).then(setRun).catch((e) => toast(e.message, "err"));

  useEffect(() => {
    let es: EventSource | null = null;
    let alive = true;
    offset.current = 0;
    setRows([]);
    setLog("");
    (async () => {
      const r = await api.get<Run>(`/api/runs/${id}`);
      if (!alive) return;
      setRun(r);
      if (r.status.state === "running" || r.status.state === "queued") {
        es = new EventSource(`/api/runs/${id}/stream`);
        es.onmessage = (ev) => {
          const d = JSON.parse(ev.data);
          if (d.metrics?.length) setRows((x) => [...x, ...d.metrics]);
          if (d.log) setLog((x) => (x + d.log).slice(-200000));
          setRun((x) => (x ? { ...x, status: d.status } : x));
          if (["completed", "failed", "stopped"].includes(d.status.state)) {
            es?.close();
            loadRun();
          }
        };
      } else {
        const m = await api.get(`/api/runs/${id}/metrics`);
        setRows(m.rows);
        const l = await api.get(`/api/runs/${id}/logs?tail=2000`);
        setLog(l.text);
      }
    })().catch((e) => toast(e.message, "err"));
    return () => {
      alive = false;
      es?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  if (!run) return <div className="text-muted">Loading…</div>;
  const st = run.status;
  const live = st.state === "running" || st.state === "queued";
  const lastVal = [...rows].reverse().find((r) => r.split === "val");
  const final = run.eval?.test || run.eval?.validation;

  const stop = async () => {
    await api.post(`/api/runs/${id}/stop`);
    toast("Stopping after the current step…");
  };
  const del = async () => {
    if (!confirm(`Delete run ${id} and all its files?`)) return;
    await api.del(`/api/runs/${id}`);
    toast("Run deleted");
    nav("/runs");
  };
  const tb = async () => {
    const r = await api.post("/api/tensorboard");
    window.open(r.url, "_blank");
  };

  return (
    <div>
      <div className="mb-1 text-[12.5px] text-muted"><Link to="/runs" className="hover:text-fg">Runs</Link> / <span className="text-fg">{id}</span></div>
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-[20px] font-medium tracking-tight">{run.config.run_name || id}</h1>
            <StatusBadge state={st.state} />
          </div>
          <div className="mt-1 flex flex-wrap gap-x-3 text-[12.5px] text-muted">
            <span className="font-mono">{run.config.base_model}</span>
            <span>·</span><span>{run.config.data?.dataset}</span>
            <span>·</span><span>{run.config.method} / {run.config.finetune}</span>
            {st.model_info?.device && <><span>·</span><span>{st.model_info.device}</span></>}
          </div>
        </div>
        <div className="flex gap-2">
          <Button size="sm" icon={<BarChart3 className="size-3.5" />} onClick={tb}>TensorBoard</Button>
          {live ? <Button size="sm" variant="danger" icon={<Square className="size-3" />} onClick={stop}>Stop</Button> : <Button size="sm" variant="ghost" icon={<Trash2 className="size-3.5" />} onClick={del}>Delete</Button>}
        </div>
      </div>

      {live && (
        <Card className="mb-6 p-4">
          <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2 text-[13px]">
            <div className="flex items-center gap-2"><span className="font-medium capitalize">{(st.stage || "starting").replace(/-/g, " ")}</span><span className="text-muted tabular">step {st.step ?? 0} / {st.total_steps || "?"}</span></div>
            <div className="flex gap-4 text-[12.5px] text-muted tabular">
              <span>elapsed {fmt.dur(st.elapsed_s)}</span><span>ETA {fmt.dur(st.eta_s)}</span>
              {lastVal && <span>val acc <span className="text-fg">{fmt.num(lastVal.accuracy, 3)}</span> · ece <span className="text-fg">{fmt.num(lastVal.ece, 3)}</span></span>}
            </div>
          </div>
          <Progress value={st.progress || 0} />
        </Card>
      )}
      {st.state === "failed" && <div className="mb-6"><Alert t="danger"><span className="font-medium">Training failed.</span> {st.error} <button className="underline" onClick={() => setTab("logs")}>View logs</button></Alert></div>}

      {final && (
        <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-5">
          <Stat label={`Accuracy (${run.eval?.test ? "test" : "val"})`} value={fmt.pct(final.accuracy)} accent />
          <Stat label="Macro F1" value={fmt.num(final.macro_f1, 3)} />
          <Stat label="ECE ↓" value={fmt.num(final.ece, 3)} sub="calibration error" />
          <Stat label="Brier ↓" value={fmt.num(final.brier, 3)} />
          <Stat label="Training time" value={fmt.dur(st.summary?.elapsed_s)} sub={`T = ${fmt.num(run.eval?.temperature, 2)}`} />
        </div>
      )}

      <Tabs
        value={tab}
        onChange={setTab}
        className="mb-5"
        tabs={[
          { id: "metrics", label: "Metrics", icon: <Activity className="size-3.5" /> },
          { id: "logs", label: "Logs", icon: <Terminal className="size-3.5" /> },
          { id: "eval", label: "Evaluation", icon: <Gauge className="size-3.5" /> },
          { id: "model", label: "Model", icon: <Box className="size-3.5" /> },
          { id: "test", label: "Test", icon: <FlaskConical className="size-3.5" /> },
          { id: "publish", label: "Publish", icon: <CloudUpload className="size-3.5" /> },
          { id: "files", label: "Files", icon: <FolderOpen className="size-3.5" /> },
        ]}
      />

      {tab === "metrics" && <MetricsTab rows={rows} smooth={smooth} setSmooth={setSmooth} />}
      {tab === "logs" && <LogsTab text={log} live={live} />}
      {tab === "eval" && <EvalTab run={run} />}
      {tab === "model" && <ModelTab run={run} />}
      {tab === "test" && (run.has_model ? <PlaygroundPanel runId={id} meta={run.model_meta} /> : <Empty title="No model yet" icon={<FlaskConical />}>The model becomes testable once training finishes.</Empty>)}
      {tab === "publish" && <PublishTab run={run} onPushed={loadRun} />}
      {tab === "files" && <FilesTab run={run} onChange={loadRun} />}
    </div>
  );
}

function MetricsTab({ rows, smooth, setSmooth }: { rows: Json[]; smooth: number; setSmooth: (v: number) => void }) {
  const hasRl = rows.some((r) => r.split === "rlcd");
  const charts: { title: string; series: Series[]; log?: boolean }[] = [
    { title: "Loss", series: [{ key: "loss", split: "train", label: "train", color: C.a }, ...(hasRl ? [{ key: "loss", split: "rlcd", label: "rlcd", color: C.d }] : []), { key: "nll", split: "val", label: "val nll", color: C.b }] },
    { title: "Accuracy", series: [{ key: "accuracy", split: "val", label: "val acc", color: C.a }, { key: "macro_f1", split: "val", label: "val macro F1", color: C.b }] },
    { title: "Calibration", series: [{ key: "ece", split: "val", label: "ECE", color: C.c }, { key: "brier", split: "val", label: "Brier", color: C.d }, { key: "avg_confidence", split: "val", label: "avg conf", color: C.b }] },
    { title: "Learning rate", series: [{ key: "lr", split: "train", label: "sft", color: C.a }, ...(hasRl ? [{ key: "lr", split: "rlcd", label: "rlcd", color: C.d }] : [])] },
    { title: "Gradient norm", series: [{ key: "grad_norm", split: "train", label: "sft", color: C.a }, ...(hasRl ? [{ key: "grad_norm", split: "rlcd", label: "rlcd", color: C.d }] : [])] },
    { title: "Throughput", series: [{ key: "samples_per_s", split: "train", label: "samples/s", color: C.b }] },
    ...(hasRl ? [{ title: "RLCD objective terms", series: [{ key: "brier", split: "rlcd", label: "brier", color: C.c }, { key: "kl_ref", split: "rlcd", label: "KL(ref)", color: C.b }, { key: "consistency", split: "rlcd", label: "order consistency", color: C.d }] }] : []),
    { title: "Memory (GB)", series: [{ key: "memory_gb", split: "train", label: "allocated", color: C.c }] },
  ];
  return (
    <div>
      <div className="mb-4 flex items-center justify-end gap-3 text-[12.5px] text-muted">
        Smoothing
        <input type="range" min={0} max={0.95} step={0.05} value={smooth} onChange={(e) => setSmooth(Number(e.target.value))} className="w-36" />
        <span className="w-8 font-mono text-fg">{smooth.toFixed(2)}</span>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {charts.map((c) => <MetricChart key={c.title} title={c.title} rows={rows} series={c.series} smoothing={smooth} />)}
      </div>
    </div>
  );
}

function LogsTab({ text, live }: { text: string; live: boolean }) {
  const ref = useRef<HTMLPreElement>(null);
  const [follow, setFollow] = useState(true);
  useEffect(() => {
    if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [text, follow]);
  const clean = useMemo(() => text.replace(/\r/g, "\n").split("\n").filter((l) => !/Loading weights:|Writing model shards:/.test(l) || /100%/.test(l)).join("\n"), [text]);
  return (
    <Card>
      <CardHeader title="train.log" right={<Toggle checked={follow} onChange={setFollow} label={<span className="text-[12px] text-muted">Follow {live && "· live"}</span>} />} />
      <pre ref={ref} className="h-[520px] overflow-auto bg-bg p-4 font-mono text-[12px] leading-relaxed text-muted">
        {clean.split("\n").map((l, i) => <div key={i} className={l.startsWith("[snapcall]") ? "text-fg" : /error|traceback/i.test(l) ? "text-danger" : ""}>{l || " "}</div>)}
      </pre>
    </Card>
  );
}

function MetricsTable({ ev }: { ev: Json }) {
  const rows = (["validation_uncalibrated", "validation", "test"] as const).filter((s) => ev[s]);
  return (
    <table className="w-full text-[13px]">
      <thead className="text-left text-[12px] text-faint"><tr className="border-b border-line">{["Split", "Accuracy", "Macro F1", "ECE ↓", "Brier ↓", "NLL ↓", "Avg conf", "N"].map((h) => <th key={h} className="px-4 py-2 font-normal">{h}</th>)}</tr></thead>
      <tbody>
        {rows.map((s) => (
          <tr key={s} className="border-b border-line last:border-0 tabular">
            <td className="px-4 py-2">{s === "validation_uncalibrated" ? <span className="text-muted">validation (T = 1)</span> : s}</td>
            <td className="px-4">{fmt.pct(ev[s].accuracy, 2)}</td><td className="px-4">{fmt.num(ev[s].macro_f1)}</td><td className="px-4">{fmt.num(ev[s].ece)}</td>
            <td className="px-4">{fmt.num(ev[s].brier)}</td><td className="px-4">{fmt.num(ev[s].nll)}</td><td className="px-4">{fmt.num(ev[s].avg_confidence, 3)}</td><td className="px-4 text-muted">{ev[s].n}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function EvalTab({ run }: { run: Run }) {
  const toast = useToast();
  const { data: datasets } = useFetch<Json[]>("/api/datasets");
  const [dsName, setDs] = useState(run.config.data?.dataset || "");
  const [split, setSplit] = useState("test");
  const [res, setRes] = useState<Json | null>(null);
  const [busy, setBusy] = useState(false);
  const ev = run.eval;
  const detail = res || ev?.test || ev?.validation;
  const go = async () => {
    setBusy(true);
    try {
      setRes(await api.post(`/api/runs/${run.id}/evaluate`, { dataset: dsName, split }));
    } catch (e: any) {
      toast(e.message, "err");
    } finally {
      setBusy(false);
    }
  };
  if (!run.has_model) return <Empty title="Evaluation available after training" icon={<Gauge />}>Validation and test metrics, reliability diagrams and confusion matrices appear here.</Empty>;
  return (
    <div className="space-y-5">
      {ev && <Card><CardHeader title="Final metrics" description={`Temperature scaling fitted T = ${fmt.num(ev.temperature, 3)} on validation.`} /><div className="overflow-x-auto"><MetricsTable ev={ev} /></div></Card>}
      <Card className="p-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="w-60"><Field label="Dataset"><Select value={dsName} onChange={(e) => setDs(e.target.value)} options={(datasets ?? []).map((d) => d.name)} /></Field></div>
          <div className="w-36"><Field label="Split"><Select value={split} onChange={(e) => setSplit(e.target.value)} options={["test", "validation", "train"]} /></Field></div>
          <Button variant="primary" loading={busy} onClick={go}>Run evaluation</Button>
          {res && <div className="text-[12.5px] text-muted">acc <span className="text-fg">{fmt.pct(res.accuracy)}</span> · ECE <span className="text-fg">{fmt.num(res.ece, 3)}</span> · {fmt.num(res.latency_ms_per_example, 1)} ms/example</div>}
        </div>
      </Card>
      {detail && (
        <div className="grid gap-5 lg:grid-cols-[400px_1fr]">
          <Card><CardHeader title="Reliability diagram" description="Bars should sit on the diagonal. Red = gap between confidence and accuracy." /><div className="flex justify-center p-4">{detail.reliability ? <Reliability bins={detail.reliability} /> : <span className="text-muted">n/a</span>}</div></Card>
          <Card><CardHeader title="Confusion matrix" /><div className="p-4">{detail.confusion ? <Confusion labels={detail.confusion.labels} matrix={detail.confusion.matrix} /> : <span className="text-[13px] text-muted">Too many labels to display.</span>}</div></Card>
        </div>
      )}
      {res?.mistakes?.length > 0 && (
        <Card>
          <CardHeader title="Mistakes" description="Highest-signal place to improve data or option descriptions." />
          <div className="divide-y divide-line">
            {res!.mistakes.map((m: Json, i: number) => (
              <div key={i} className="grid gap-2 px-4 py-2.5 text-[12.5px] md:grid-cols-[1fr_auto]">
                <div className="text-muted">{m.state}</div>
                <div className="flex items-center gap-1.5 whitespace-nowrap"><Badge t="accent">{m.label}</Badge><span className="text-faint">→</span><Badge t="danger">{m.pred}</Badge><span className="font-mono text-faint">{(m.confidence * 100).toFixed(0)}%</span></div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

function ModelTab({ run }: { run: Run }) {
  const { data: files } = useFetch<Json[]>(`/api/runs/${run.id}/files`);
  if (!run.has_model) return <Empty title="No model saved yet" icon={<Box />} />;
  const m = run.model_meta!;
  const mi = run.status.model_info || {};
  const size = (files ?? []).filter((f) => f.path.startsWith("model/")).reduce((a, f) => a + f.size, 0);
  const q = m.question || {};
  const items: [string, any][] = [
    ["Name", m.name], ["Base model", m.base_model], ["Architecture", `${m.arch} · ${m.pooling} pooling`], ["Fine-tuning", m.finetune],
    ["Parameters", mi.params ? fmt.params(mi.params) : "—"], ["Trainable", mi.trainable_params ? `${fmt.params(mi.trainable_params)} (${((mi.trainable_params / mi.params) * 100).toFixed(2)}%)` : "—"],
    ["Method", m.method], ["Temperature", fmt.num(m.temperature, 3)], ["Max length", m.max_length], ["Dtype", m.dtype],
    ["Dataset", m.dataset], ["Size on disk", fmt.bytes(size)], ["Template", `v${m.template_version}`], ["Snapcall", m.snapcall_version],
    ["Trained", run.status.ended ? new Date(run.status.ended * 1000).toLocaleString() : "—"], ["Device", mi.device],
  ];
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_380px]">
      <Card>
        <CardHeader title="Model metadata" description="Saved with the weights in snapcall_config.json" />
        <div className="grid gap-x-8 p-4 sm:grid-cols-2">
          {items.map(([k, v]) => (
            <div key={k} className="flex justify-between gap-4 border-b border-line py-2 text-[13px] last:border-0"><span className="text-muted">{k}</span><span className="truncate text-right font-mono text-[12.5px]">{v ?? "—"}</span></div>
          ))}
        </div>
      </Card>
      <div className="space-y-4">
        <Card>
          <CardHeader title="Decision schema" description={q.instructions} />
          <div className="p-4"><Code>{JSON.stringify(q, null, 2)}</Code></div>
        </Card>
        <Card className="p-4">
          <div className="mb-2 text-[12.5px] text-muted">Load in Python</div>
          <Code>{`from snapcall import DecisionModel\n\nm = DecisionModel.from_pretrained(\n  "~/.snapcall/runs/${run.id}/model")\nm.decide(state="...", questions={...})`}</Code>
        </Card>
      </div>
    </div>
  );
}

function PublishTab({ run, onPushed }: { run: Run; onPushed: () => void }) {
  const toast = useToast();
  const { data: settings } = useFetch<Json>("/api/settings");
  const [who, setWho] = useState<string>("");
  const [repo, setRepo] = useState("");
  const [priv, setPriv] = useState(true);
  const [card, setCard] = useState("");
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<Json | null>(null);
  useEffect(() => {
    if (settings?.hf_token) api.post("/api/settings/test-hf").then((r) => setWho(r.name)).catch(() => {});
  }, [settings?.hf_token]);
  useEffect(() => {
    if (who && !repo) setRepo(`${who}/${run.config.run_name || run.id.replace(/-\d{8}-\d{6}$/, "")}-snapcall`);
  }, [who, repo, run]);
  useEffect(() => {
    if (!run.has_model) return;
    const t = setTimeout(() => api.get(`/api/runs/${run.id}/card?repo_id=${encodeURIComponent(repo)}`).then((r) => setCard(r.markdown)), 300);
    return () => clearTimeout(t);
  }, [repo, run]);
  if (!run.has_model) return <Empty title="Train a model first" icon={<CloudUpload />} />;
  const push = async () => {
    setBusy(true);
    try {
      const { job: id } = await api.post(`/api/runs/${run.id}/push`, { repo_id: repo, private: priv, card });
      const j = await waitJob(id, setJob);
      toast(`Pushed to ${j.result.url}`);
      onPushed();
    } catch (e: any) {
      toast(e.message, "err");
    } finally {
      setBusy(false);
      setJob(null);
    }
  };
  return (
    <div className="grid gap-5 lg:grid-cols-[360px_1fr]">
      <div className="space-y-4">
        <Card className="space-y-4 p-4">
          {!settings?.hf_token && <Alert t="warn">Add a Hugging Face write token in <Link className="underline" to="/settings">Settings</Link>.</Alert>}
          {who && <div className="text-[12.5px] text-muted">Signed in as <span className="text-fg">{who}</span></div>}
          <Field label="Repository"><Input value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="username/model-name" className="font-mono" /></Field>
          <Toggle checked={priv} onChange={setPriv} label={<span className="text-muted">Private repository</span>} />
          <Button variant="primary" className="w-full" icon={<CloudUpload className="size-3.5" />} onClick={push} loading={busy} disabled={!repo.includes("/") || !settings?.hf_token}>Push to Hugging Face</Button>
          {job && <div className="text-[12px] text-muted">{job.message}</div>}
          <div className="text-[12px] text-faint">Uploads weights, decision head, tokenizer, snapcall_config.json, eval results, training config and this model card.</div>
        </Card>
        {run.pushed && (
          <Card className="p-4">
            <div className="text-[12.5px] text-muted">Last push · {fmt.ago(run.pushed.time)}</div>
            <a href={run.pushed.url} target="_blank" rel="noreferrer" className="mt-1 flex items-center gap-1.5 font-mono text-[13px] text-accent hover:underline">{run.pushed.repo_id}<ExternalLink className="size-3.5" /></a>
          </Card>
        )}
      </div>
      <Card>
        <CardHeader title={<span className="flex items-center gap-2"><FileText className="size-3.5" />README.md · model card</span>} description="Auto-generated from your config and evaluation. Edit freely before pushing." />
        <Textarea rows={30} value={card} onChange={(e) => setCard(e.target.value)} className="rounded-none border-0 font-mono text-[12px] focus:ring-0" spellCheck={false} />
      </Card>
    </div>
  );
}

function FilesTab({ run, onChange }: { run: Run; onChange: () => void }) {
  const toast = useToast();
  const files = useFetch<Json[]>(`/api/runs/${run.id}/files`);
  const ck = (files.data ?? []).filter((f) => f.path.startsWith("checkpoints/"));
  const clean = async (what: Json) => {
    const r = await api.post("/api/cleanup", { run_id: run.id, ...what });
    toast(`Freed ${fmt.bytes(r.freed_bytes)}`);
    files.reload();
    onChange();
  };
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <Button size="sm" disabled={!ck.length || run.status.state === "running"} onClick={() => clean({ checkpoints: true })}>Delete checkpoints ({fmt.bytes(ck.reduce((a, f) => a + f.size, 0))})</Button>
        <Button size="sm" disabled={run.status.state === "running"} onClick={() => clean({ checkpoints: false, tensorboard: true })}>Delete TensorBoard events</Button>
      </div>
      <Card>
        <table className="w-full text-[12.5px]">
          <tbody>
            {(files.data ?? []).map((f) => (
              <tr key={f.path} className="border-b border-line last:border-0"><td className="px-4 py-1.5 font-mono text-muted">{f.path}</td><td className="px-4 text-right font-mono tabular text-faint">{fmt.bytes(f.size)}</td></tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
