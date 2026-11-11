import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { FileUp, Plus, Sparkles, Trash2, Wand2 } from "lucide-react";
import { api, waitJob, type DatasetMeta, type Json } from "../api";
import { Alert, Badge, Button, Card, Code, Field, Input, Progress, Select, Tabs, Textarea, cx, useFetch, useToast } from "./ui";

type Tab = "hf" | "upload" | "synth" | "faker";
const QTYPES = [
  { value: "choice", label: "choice: pick one option" },
  { value: "score", label: "score: ordinal levels" },
  { value: "noul", label: "noul: yes / no" },
];

export default function DatasetCreator({ onCreated }: { onCreated: (m: DatasetMeta) => void }) {
  const [tab, setTab] = useState<Tab>("hf");
  return (
    <div>
      <Tabs
        value={tab}
        onChange={setTab}
        tabs={[
          { id: "hf", label: "Hugging Face", icon: <span className="text-[13px]">🤗</span> },
          { id: "upload", label: "Upload file", icon: <FileUp className="size-3.5" /> },
          { id: "synth", label: "Synthetic (LLM)", icon: <Sparkles className="size-3.5" /> },
          { id: "faker", label: "Faker templates", icon: <Wand2 className="size-3.5" /> },
        ]}
      />
      <div className="pt-5">
        {tab === "hf" && <HF onCreated={onCreated} />}
        {tab === "upload" && <Upload onCreated={onCreated} />}
        {tab === "synth" && <Synth onCreated={onCreated} />}
        {tab === "faker" && <Faker onCreated={onCreated} />}
      </div>
    </div>
  );
}

function JobBar({ job }: { job: { progress: number; message: string } | null }) {
  if (!job) return null;
  return (
    <div className="space-y-1.5">
      <Progress value={job.progress || 0.04} />
      <div className="text-[12px] text-muted">{job.message}</div>
    </div>
  );
}

// ------------------------------------------------------------------ HF
function HF({ onCreated }: { onCreated: (m: DatasetMeta) => void }) {
  const toast = useToast();
  const { data: cat } = useFetch<Json>("/api/catalog");
  const [hfId, setHfId] = useState("");
  const [config, setConfig] = useState<string>("");
  const [peek, setPeek] = useState<Json | null>(null);
  const [form, setForm] = useState<Json>({ name: "", text_col: "", label_col: "", type: "choice", instructions: "", instructions_col: "", max_per_split: 5000, top_labels: "", multi_label: "single" });
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<any>(null);

  const inspect = async (id = hfId, cfg = config) => {
    setBusy(true);
    try {
      const p = await api.get(`/api/hf/peek?hf_id=${encodeURIComponent(id)}${cfg ? `&config=${encodeURIComponent(cfg)}` : ""}`);
      setPeek(p);
      setConfig(p.config);
      const cols = p.columns.map((c: Json) => c.name);
      const guessText = cols.find((c: string) => /text|sentence|content|review|message|passage|utterance/i.test(c)) || cols[0];
      const guessLabel = p.columns.find((c: Json) => c.names)?.name || cols.find((c: string) => /label|category|intent|class|answer/i.test(c)) || cols[1];
      setForm((f) => ({ ...f, name: f.name || id.split("/").pop()!.toLowerCase(), text_col: f.text_col && cols.includes(f.text_col) ? f.text_col : guessText, label_col: f.label_col && cols.includes(f.label_col) ? f.label_col : guessLabel }));
    } catch (e: any) {
      toast(e.message, "err");
    } finally {
      setBusy(false);
    }
  };

  const usePreset = (p: Json) => {
    setHfId(p.hf_id);
    setForm({ name: p.id, text_col: p.text_col, label_col: p.label_col, type: p.type, instructions: p.instructions, instructions_col: p.instructions_col || "", max_per_split: 5000, top_labels: "", multi_label: "single" });
    inspect(p.hf_id, p.config || "");
  };

  const run = async () => {
    setBusy(true);
    try {
      const { job: id } = await api.post("/api/datasets/hf", { hf_id: hfId, config: config || null, ...form, instructions_col: form.instructions_col || null, max_per_split: Number(form.max_per_split) || null, top_labels: Number(form.top_labels) || null });
      const j = await waitJob(id, setJob);
      const prep = Object.entries(j.result.prep || {}).filter(([k, v]) => k !== "labels" && v).map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`);
      toast(`Dataset ${j.result.name} ready${prep.length ? ` · ${prep.join(", ")}` : ""}`);
      onCreated(j.result);
    } catch (e: any) {
      toast(e.message, "err");
    } finally {
      setBusy(false);
      setJob(null);
    }
  };

  const cols: string[] = peek?.columns.map((c: Json) => c.name) ?? [];
  return (
    <div className="space-y-5">
      <div>
        <div className="mb-2 text-[12.5px] text-muted">Presets</div>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {(cat?.hf_datasets ?? []).map((p: Json) => (
            <button key={p.id} onClick={() => usePreset(p)} className={cx("rounded-md border p-3 text-left transition hover:border-line-strong cursor-pointer", hfId === p.hf_id ? "border-accent bg-accent-soft" : "border-line bg-bg")}>
              <div className="flex items-center justify-between gap-2">
                <span className="text-[13px] font-medium">{p.title}</span>
                <Badge>{p.type}</Badge>
              </div>
              <div className="mt-1 line-clamp-2 text-[12px] text-muted">{p.description}</div>
            </button>
          ))}
        </div>
      </div>
      <div className="flex gap-2">
        <Input placeholder="or any dataset id, e.g. PolyAI/banking77" value={hfId} onChange={(e) => setHfId(e.target.value)} onKeyDown={(e) => e.key === "Enter" && inspect()} />
        <Button onClick={() => inspect()} loading={busy && !job} disabled={!hfId}>Inspect</Button>
      </div>
      {peek && (
        <Card className="p-4 space-y-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <Field label="Config"><Select value={config} onChange={(e) => { setConfig(e.target.value); inspect(hfId, e.target.value); }} options={peek.configs} /></Field>
            <Field label="Text column (state)"><Select value={form.text_col} onChange={(e) => setForm({ ...form, text_col: e.target.value })} options={cols} /></Field>
            <Field label="Label column"><Select value={form.label_col} onChange={(e) => setForm({ ...form, label_col: e.target.value })} options={cols} /></Field>
            <Field label="Question type"><Select value={form.type} onChange={(e) => setForm({ ...form, type: e.target.value })} options={QTYPES} /></Field>
            <Field label="Dataset name"><Input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
            <Field label="Max rows per split"><Input type="number" value={form.max_per_split} onChange={(e) => setForm({ ...form, max_per_split: e.target.value })} /></Field>
            <Field label="Keep top N labels" hint="Empty keeps all. Useful for long-tail labels (e.g. CWE)."><Input type="number" min={2} placeholder="all" value={form.top_labels} onChange={(e) => setForm({ ...form, top_labels: e.target.value })} /></Field>
            {peek.columns.find((c: Json) => c.name === form.label_col)?.dtype === "list" && (
              <Field label="Multi-label rows" hint="This label column holds lists.">
                <Select value={form.multi_label} onChange={(e) => setForm({ ...form, multi_label: e.target.value })} options={[{ value: "single", label: "keep single-label rows" }, { value: "first", label: "use the first label" }, { value: "noul", label: "one yes/no question per label" }]} />
              </Field>
            )}
          </div>
          <div className="text-[12px] text-faint">Rows with empty text or labels are skipped automatically. Numeric labels are ordered by value.</div>
          <Field label="Instructions (the question asked of the model)" hint={form.instructions_col ? `Per-row instructions from column "${form.instructions_col}"` : undefined}>
            <Input value={form.instructions} onChange={(e) => setForm({ ...form, instructions: e.target.value })} placeholder="e.g. Which banking intent does this message express?" />
          </Field>
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full text-[12px]">
              <thead className="bg-panel-2 text-muted">
                <tr>{peek.columns.map((c: Json) => <th key={c.name} className="px-2.5 py-1.5 text-left font-medium">{c.name} <span className="text-faint">{c.names ? `class(${c.names.length})` : c.dtype}</span></th>)}</tr>
              </thead>
              <tbody>
                {peek.rows.slice(0, 5).map((r: Json, i: number) => (
                  <tr key={i} className="border-t border-line">{peek.columns.map((c: Json) => <td key={c.name} className="max-w-[320px] truncate px-2.5 py-1.5 text-muted">{String(c.names && typeof r[c.name] === "number" ? c.names[r[c.name]] ?? r[c.name] : r[c.name])}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
          <JobBar job={job} />
          <div className="flex justify-end"><Button variant="primary" onClick={run} loading={busy && !!job} disabled={!form.name || !form.text_col || !form.label_col}>Import dataset</Button></div>
        </Card>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ Upload
function Upload({ onCreated }: { onCreated: (m: DatasetMeta) => void }) {
  const toast = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [info, setInfo] = useState<Json | null>(null);
  const [form, setForm] = useState<Json>({ name: "", text_col: "", label_col: "", type: "choice", instructions: "" });
  const [busy, setBusy] = useState(false);

  const pick = async (f: File) => {
    setFile(f);
    const fd = new FormData();
    fd.append("file", f);
    try {
      const i = await api.post("/api/datasets/upload/inspect", fd);
      setInfo(i);
      const cols: string[] = i.columns;
      setForm((x) => ({ ...x, name: f.name.replace(/\.[^.]+$/, ""), text_col: cols.find((c) => /text|state|message|content|input/i.test(c)) || cols[0], label_col: cols.find((c) => /label|answer|category|class|intent|output/i.test(c)) || cols[1] }));
    } catch (e: any) {
      toast(e.message, "err");
    }
  };
  const submit = async () => {
    if (!file) return;
    setBusy(true);
    const fd = new FormData();
    fd.append("file", file);
    Object.entries(form).forEach(([k, v]) => fd.append(k, String(v)));
    try {
      const m = await api.post("/api/datasets/upload", fd);
      toast(`Dataset ${m.name} ready`);
      if (m.warnings?.length) toast(`${m.warnings.length} rows skipped: ${m.warnings[0]}`, "err");
      onCreated(m);
    } catch (e: any) {
      toast(e.message, "err");
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="space-y-4">
      <label className="flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-line-strong bg-bg px-6 py-10 text-center hover:border-accent">
        <FileUp className="mb-2 size-5 text-faint" />
        <div className="text-[13px]">{file ? file.name : "Drop or choose a .jsonl, .json, .csv or .tsv file"}</div>
        <div className="mt-1 text-[12px] text-faint">Native System One rows or any table with a text and a label column</div>
        <input type="file" accept=".jsonl,.json,.csv,.tsv,.ndjson" className="hidden" onChange={(e) => e.target.files?.[0] && pick(e.target.files[0])} />
      </label>
      <details className="text-[12.5px] text-muted">
        <summary className="cursor-pointer hover:text-fg">Native format</summary>
        <Code className="mt-2">{`{"state": "I was charged twice", "question": {"type": "choice", "instructions": "Route to a team",
  "criteria": {"billing": "Charges, refunds", "shipping": "Delivery"}}, "answer": "billing"}
{"state": {...json state...}, "questions": {"q1": {...}, "q2": {...}}, "answers": {"q1": {"choice": "x"}, "q2": {"noul": 1}}}`}</Code>
      </details>
      {info && (
        <Card className="space-y-4 p-4">
          <div className="flex items-center gap-2 text-[13px]">
            <Badge t={info.native ? "accent" : "info"}>{info.native ? "System One rows" : "table"}</Badge>
            <span className="text-muted">{info.count} rows · columns: {info.columns.join(", ")}</span>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <Field label="Dataset name"><Input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
            {!info.native && (
              <>
                <Field label="Text column"><Select value={form.text_col} onChange={(e) => setForm({ ...form, text_col: e.target.value })} options={info.columns} /></Field>
                <Field label="Label column"><Select value={form.label_col} onChange={(e) => setForm({ ...form, label_col: e.target.value })} options={info.columns} /></Field>
                <Field label="Question type"><Select value={form.type} onChange={(e) => setForm({ ...form, type: e.target.value })} options={QTYPES} /></Field>
              </>
            )}
          </div>
          {!info.native && <Field label="Instructions"><Input value={form.instructions} onChange={(e) => setForm({ ...form, instructions: e.target.value })} placeholder="The question asked of the model" /></Field>}
          <div className="flex justify-end"><Button variant="primary" loading={busy} onClick={submit}>Import</Button></div>
        </Card>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ Synthetic
function OptionsEditor({ options, onChange, type }: { options: [string, string][]; onChange: (o: [string, string][]) => void; type: string }) {
  return (
    <div className="space-y-2">
      {options.map(([n, d], i) => (
        <div key={i} className="flex gap-2">
          <Input className="max-w-[180px] font-mono" value={n} placeholder={type === "score" ? `level ${i}` : "name"} onChange={(e) => onChange(options.map((o, j) => (j === i ? [e.target.value, o[1]] : o)))} />
          <Input value={d} placeholder="When is this the right answer?" onChange={(e) => onChange(options.map((o, j) => (j === i ? [o[0], e.target.value] : o)))} />
          <Button variant="ghost" size="sm" className="h-8.5" onClick={() => onChange(options.filter((_, j) => j !== i))} icon={<Trash2 className="size-3.5" />} />
        </div>
      ))}
      <Button size="sm" variant="ghost" icon={<Plus className="size-3.5" />} onClick={() => onChange([...options, ["", ""]])}>Add option</Button>
    </div>
  );
}

function Synth({ onCreated }: { onCreated: (m: DatasetMeta) => void }) {
  const toast = useToast();
  const { data: settings } = useFetch<Json>("/api/settings");
  const [provider, setProvider] = useState("");
  const [useCase, setUseCase] = useState("");
  const [q, setQ] = useState<{ type: string; instructions: string; options: [string, string][] }>({ type: "choice", instructions: "", options: [] });
  const [n, setN] = useState(30);
  const [language, setLanguage] = useState("English");
  const [style, setStyle] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState<"" | "suggest" | "gen">("");
  const [job, setJob] = useState<any>(null);
  useEffect(() => {
    if (settings && !provider) setProvider(settings.synth_provider || "anthropic");
  }, [settings, provider]);
  const hasKey = provider === "anthropic" ? !!settings?.anthropic_api_key : !!settings?.openai_api_key || /localhost|127\.0\.0\.1/.test(settings?.openai_base_url || "");

  const suggest = async () => {
    setBusy("suggest");
    try {
      const s = await api.post("/api/synth/suggest", { use_case: useCase, provider });
      setQ({ type: s.type || "choice", instructions: s.instructions || "", options: Object.entries(s.options || {}) as [string, string][] });
      if (!name) setName(useCase.split(/\s+/).slice(0, 4).join("-").toLowerCase().replace(/[^a-z0-9-]/g, ""));
    } catch (e: any) {
      toast(e.message, "err");
    } finally {
      setBusy("");
    }
  };
  const gen = async () => {
    setBusy("gen");
    try {
      const { job: id } = await api.post("/api/datasets/synth", { name, use_case: useCase, n_per_option: n, language, style, provider, question: { type: q.type, instructions: q.instructions, options: Object.fromEntries(q.options.filter(([k]) => k)) } });
      const j = await waitJob(id, setJob);
      toast(`Dataset ${j.result.name} ready`);
      onCreated(j.result);
    } catch (e: any) {
      toast(e.message, "err");
    } finally {
      setBusy("");
      setJob(null);
    }
  };
  const total = n * q.options.length;
  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-[200px_1fr]">
        <Field label="Provider">
          <Select value={provider} onChange={(e) => setProvider(e.target.value)} options={[{ value: "anthropic", label: "Anthropic" }, { value: "openai", label: "OpenAI-compatible" }]} />
        </Field>
        <Field label="Model">
          <Input disabled value={provider === "anthropic" ? settings?.anthropic_model ?? "" : `${settings?.openai_model ?? ""} @ ${settings?.openai_base_url ?? ""}`} />
        </Field>
      </div>
      {settings && !hasKey && (
        <Alert t="warn">No API key for this provider. Add one in <Link to="/settings" className="underline">Settings</Link>. Keys stay on this machine in <span className="font-mono">~/.snapcall/settings.json</span>.</Alert>
      )}
      <Field label="Describe your use case">
        <Textarea rows={3} value={useCase} onChange={(e) => setUseCase(e.target.value)} placeholder="e.g. Triage inbound emails for a B2B SaaS: decide whether it's a sales lead, a support ticket, a billing question, partnership, or spam." />
      </Field>
      <div className="flex justify-end">
        <Button onClick={suggest} loading={busy === "suggest"} disabled={!useCase || !hasKey} icon={<Sparkles className="size-3.5" />}>Suggest question & options</Button>
      </div>
      <Card className="space-y-4 p-4">
        <div className="grid gap-3 sm:grid-cols-[200px_1fr]">
          <Field label="Question type"><Select value={q.type} onChange={(e) => setQ({ ...q, type: e.target.value, options: e.target.value === "noul" ? [["no", "No / false"], ["yes", "Yes / true"]] : q.options })} options={QTYPES} /></Field>
          <Field label="Instructions"><Input value={q.instructions} onChange={(e) => setQ({ ...q, instructions: e.target.value })} placeholder="The question the model answers" /></Field>
        </div>
        <Field label={q.type === "score" ? "Levels (lowest first)" : "Options"}>
          <OptionsEditor options={q.options} type={q.type} onChange={(o) => setQ({ ...q, options: o })} />
        </Field>
        <div className="grid gap-3 sm:grid-cols-4">
          <Field label="Examples per option"><Input type="number" min={5} max={1000} value={n} onChange={(e) => setN(Number(e.target.value))} /></Field>
          <Field label="Language"><Input value={language} onChange={(e) => setLanguage(e.target.value)} /></Field>
          <Field label="Style hints"><Input value={style} onChange={(e) => setStyle(e.target.value)} placeholder="emails, 2-5 sentences" /></Field>
          <Field label="Dataset name"><Input value={name} onChange={(e) => setName(e.target.value)} /></Field>
        </div>
        <div className="text-[12px] text-faint">Faker personas (name, job, city, tone) are mixed into every batch for diversity. ≈ {Math.ceil(total / 10)} LLM calls for {total} examples.</div>
        <JobBar job={job} />
        <div className="flex justify-end"><Button variant="primary" onClick={gen} loading={busy === "gen"} disabled={!hasKey || !name || q.options.length < 2 || !q.instructions}>Generate dataset</Button></div>
      </Card>
    </div>
  );
}

// ------------------------------------------------------------------ Faker
const LOCALES = ["en_US", "en_GB", "en_IN", "de_DE", "fr_FR", "es_ES", "it_IT", "pt_BR", "nl_NL", "hi_IN", "ja_JP", "zh_CN", "ko_KR", "ar_AA"];

function Faker({ onCreated }: { onCreated: (m: DatasetMeta) => void }) {
  const toast = useToast();
  const { data: example } = useFetch<Json>("/api/faker/example");
  const [spec, setSpec] = useState("");
  const [n, setN] = useState(60);
  const [locale, setLocale] = useState("en_US");
  const [name, setName] = useState("support-routing");
  const [rows, setRows] = useState<Json[] | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (example && !spec) setSpec(JSON.stringify(example, null, 2));
  }, [example, spec]);
  const parsed = (() => {
    try {
      return JSON.parse(spec);
    } catch {
      return null;
    }
  })();
  const call = async (preview: boolean) => {
    setBusy(true);
    try {
      const r = await api.post("/api/datasets/faker", { name, spec: parsed, n_per_option: n, locale, preview });
      if (preview) setRows(r.rows);
      else {
        toast(`Dataset ${r.name} ready`);
        onCreated(r);
      }
    } catch (e: any) {
      toast(e.message, "err");
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
      <div className="space-y-3">
        <Field label="Spec: options with {placeholder} templates" hint={<>Any Faker provider works as a placeholder: {"{name}"}, {"{company}"}, {"{email}"}, {"{city}"}, {"{iban}"}… plus {"{amount}"}, {"{order_id}"}, {"{product}"}, {"{date}"} and {"{a|b|c}"} alternatives.</>}>
          <Textarea rows={20} className={cx("font-mono text-[12px]", !parsed && spec && "border-danger")} value={spec} onChange={(e) => setSpec(e.target.value)} spellCheck={false} />
        </Field>
      </div>
      <div className="space-y-3">
        <Field label="Dataset name"><Input value={name} onChange={(e) => setName(e.target.value)} /></Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Examples per option"><Input type="number" value={n} onChange={(e) => setN(Number(e.target.value))} /></Field>
          <Field label="Locale"><Select value={locale} onChange={(e) => setLocale(e.target.value)} options={LOCALES} /></Field>
        </div>
        <Alert t="info">No API key needed. Faker also powers PII-safe augmentation at train time (swaps emails, phones, amounts, IDs), so the model learns the decision rather than the entities.</Alert>
        <div className="flex gap-2">
          <Button onClick={() => call(true)} disabled={!parsed} loading={busy}>Preview</Button>
          <Button variant="primary" onClick={() => call(false)} disabled={!parsed || !name} loading={busy}>Create dataset</Button>
        </div>
        {rows && (
          <div className="space-y-1.5 rounded-md border border-line bg-bg p-2.5">
            {rows.slice(0, 9).map((r, i) => (
              <div key={i} className="text-[12px]"><Badge t="accent" className="mr-1.5">{String(r.answer)}</Badge><span className="text-muted">{r.state}</span></div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
