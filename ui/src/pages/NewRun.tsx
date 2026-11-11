import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft, ArrowRight, Check, Cpu, Database, Info, Play, Sparkles, Star } from "lucide-react";
import { api, type DatasetMeta, type Explainer, type Fit, type Json, type Model, type Preset, type System } from "../api";
import { Alert, Badge, Button, Card, CardHeader, Code, Field, Input, PageHeader, Segmented, Select, Toggle, cx, useFetch, useToast } from "../components/ui";
import DatasetCreator from "../components/DatasetCreator";

const STEPS = ["Model", "Data", "Hyperparameters", "Review"] as const;

type FieldDef = { key: string; kind: "num" | "sel" | "bool" | "text"; options?: string[]; step?: number; min?: number };
const FIELDS: FieldDef[] = [
  { key: "method", kind: "sel", options: ["sft+rlcd", "sft", "rlcd"] },
  { key: "finetune", kind: "sel", options: ["full", "lora", "qlora", "head"] },
  { key: "learning_rate", kind: "num", step: 1e-5 },
  { key: "head_learning_rate", kind: "num", step: 1e-4 },
  { key: "epochs", kind: "num", step: 0.5, min: 0.1 },
  { key: "batch_size", kind: "num", step: 1, min: 1 },
  { key: "grad_accum", kind: "num", step: 1, min: 1 },
  { key: "max_length", kind: "num", step: 64, min: 64 },
  { key: "warmup_ratio", kind: "num", step: 0.01 },
  { key: "weight_decay", kind: "num", step: 0.01 },
  { key: "scheduler", kind: "sel", options: ["cosine", "linear", "constant"] },
  { key: "max_grad_norm", kind: "num", step: 0.1 },
  { key: "label_smoothing", kind: "num", step: 0.01 },
  { key: "option_shuffle", kind: "bool" },
  { key: "precision", kind: "sel", options: ["auto", "bf16", "fp16", "fp32"] },
  { key: "gradient_checkpointing", kind: "bool" },
  { key: "lora_r", kind: "num", step: 4, min: 1 },
  { key: "lora_alpha", kind: "num", step: 4, min: 1 },
  { key: "lora_dropout", kind: "num", step: 0.01 },
  { key: "lora_targets", kind: "text" },
  { key: "rlcd_epochs", kind: "num", step: 0.25, min: 0.1 },
  { key: "rlcd_beta", kind: "num", step: 0.01 },
  { key: "rlcd_brier_weight", kind: "num", step: 0.1 },
  { key: "rlcd_consistency_weight", kind: "num", step: 0.1 },
  { key: "temperature_scaling", kind: "bool" },
  { key: "eval_every", kind: "num", step: 10, min: 0 },
  { key: "early_stopping_patience", kind: "num", step: 1, min: 0 },
  { key: "seed", kind: "num", step: 1 },
];

const fitTone = { fits: "accent", tight: "warn", slow: "warn", no: "danger", cpu: "warn", auto: "info" } as const;
const fitLabel = { fits: "Fits", tight: "Tight", slow: "Slow on CPU", no: "Won't fit", cpu: "CPU", auto: "Fits with micro-batching" } as const;

export default function NewRun() {
  const nav = useNavigate();
  const toast = useToast();
  const [params] = useSearchParams();
  const { data: sys } = useFetch<System>("/api/system");
  const { data: cat } = useFetch<{ models: Model[]; presets: Preset[] }>("/api/catalog");
  const { data: hp } = useFetch<{ explainer: Explainer; defaults: Json }>("/api/hparams");
  const ds = useFetch<DatasetMeta[]>("/api/datasets");
  const [step, setStep] = useState(0);
  const [cfg, setCfg] = useState<Json | null>(null);
  const [preset, setPreset] = useState<string | null>(null);
  const [focus, setFocus] = useState("learning_rate");
  const [filter, setFilter] = useState<"all" | "encoder" | "causal" | "decision">("all");
  const [custom, setCustom] = useState("");
  const [creating, setCreating] = useState(false);
  const [mem, setMem] = useState<Json | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (hp && !cfg) {
      const c: Json = { ...hp.defaults, data: { dataset: params.get("dataset") || "", faker_augment: 0, max_train_samples: null, max_eval_samples: null } };
      const rec = sys?.recommendations.presets[0];
      if (rec) {
        Object.assign(c, rec.config);
        setPreset(rec.id);
      }
      setCfg(c);
    }
  }, [hp, sys, cfg, params]);

  const fits = useMemo(() => Object.fromEntries((sys?.recommendations.models ?? []).map((m) => [m.id, m])) as Record<string, Fit>, [sys]);
  const recIds = new Set((sys?.recommendations.presets ?? []).map((p) => p.id));
  const model = cat?.models.find((m) => m.id === cfg?.base_model);
  const dataset = ds.data?.find((d) => d.name === cfg?.data?.dataset);

  const set = (k: string, v: any) => setCfg((c) => ({ ...c!, [k]: v }));
  const applyPreset = (p: Preset) => {
    setPreset(p.id);
    setCfg((c) => ({ ...hp!.defaults, data: c!.data, run_name: c!.run_name, ...p.config }));
  };
  const pickModel = (m: Model | { id: string }) => {
    setPreset(null);
    const f = fits[m.id];
    const c = (cat?.models.find((x) => x.id === m.id) as Model | undefined);
    const ft = f?.finetune || (c && c.params > 1.5 ? "lora" : "full");
    setCfg((x) => ({
      ...x!, base_model: m.id, finetune: ft, trust_remote_code: !!c?.trust_remote_code,
      learning_rate: ft === "full" ? 5e-5 : 2e-4, gradient_checkpointing: (c?.params ?? 0) > 1,
      batch_size: (c?.params ?? 0) > 3 ? 4 : (c?.params ?? 0) > 1 ? 8 : 16,
    }));
  };

  // memory estimate
  useEffect(() => {
    if (!cfg || !model) return setMem(null);
    const t = setTimeout(() => {
      api.post("/api/estimate", { params_b: model.params, finetune: cfg.finetune, batch_size: cfg.batch_size, max_length: cfg.max_length, gradient_checkpointing: cfg.gradient_checkpointing }).then(setMem).catch(() => {});
    }, 250);
    return () => clearTimeout(t);
  }, [cfg?.finetune, cfg?.batch_size, cfg?.max_length, cfg?.gradient_checkpointing, model]);

  if (!cfg || !hp || !cat) return <div className="text-muted">Loading…</div>;

  const canNext = [!!cfg.base_model, !!cfg.data.dataset, true, true][step];
  const ex = hp.explainer;

  const cli = [
    `snapcall train --model ${cfg.base_model} --dataset ${cfg.data.dataset || "<dataset>"}`,
    `  --method ${cfg.method} --finetune ${cfg.finetune} --epochs ${cfg.epochs} --lr ${cfg.learning_rate} --batch-size ${cfg.batch_size}`,
    ...Object.entries(cfg)
      .filter(([k, v]) => !["base_model", "data", "method", "finetune", "epochs", "learning_rate", "batch_size", "run_name"].includes(k) && JSON.stringify(v) !== JSON.stringify(hp.defaults[k]))
      .map(([k, v]) => `  --set ${k}=${typeof v === "string" ? v : JSON.stringify(v)}`),
    ...(cfg.data.faker_augment ? [`  --set data='${JSON.stringify(cfg.data)}'`] : []),
    ...(cfg.run_name ? [`  --name ${cfg.run_name}`] : []),
  ].join(" \\\n");

  const submit = async () => {
    setSubmitting(true);
    try {
      const body = { ...cfg, run_name: cfg.run_name || null, data: { ...cfg.data, max_train_samples: cfg.data.max_train_samples || null, max_eval_samples: cfg.data.max_eval_samples || null } };
      const { id } = await api.post("/api/runs", body);
      toast("Training started");
      nav(`/runs/${id}`);
    } catch (e: any) {
      toast(e.message, "err");
      setSubmitting(false);
    }
  };

  const models = cat.models.filter((m) => filter === "all" || (filter === "decision" ? m.kind === "decision" : m.arch === filter && m.kind === "base"));

  return (
    <div>
      <PageHeader title="New training run" description="Pick a base model, attach data, tune, and train a calibrated decision model." />
      {/* stepper */}
      <div className="mb-6 flex items-center gap-2 overflow-x-auto">
        {STEPS.map((s, i) => (
          <div key={s} className="flex items-center gap-2">
            <button onClick={() => i <= step || (i === step + 1 && canNext) ? setStep(i) : null} className={cx("flex items-center gap-2 rounded-md px-2.5 py-1.5 text-[13px] cursor-pointer", i === step ? "bg-hover text-fg" : i < step ? "text-muted hover:text-fg" : "text-faint")}>
              <span className={cx("flex size-5 items-center justify-center rounded-full border text-[11px]", i < step ? "border-accent bg-accent text-accent-fg" : i === step ? "border-fg" : "border-line-strong")}>
                {i < step ? <Check className="size-3" /> : i + 1}
              </span>
              {s}
            </button>
            {i < STEPS.length - 1 && <div className="h-px w-6 bg-line-strong" />}
          </div>
        ))}
      </div>

      {step === 0 && (
        <div className="space-y-6">
          <div>
            <div className="mb-2 flex items-center gap-2 text-[13px] font-medium">
              Presets
              {sys && <span className="text-[12px] font-normal text-muted">· <Star className="inline size-3 text-accent" /> recommended for {sys.info.accelerator}{sys.info.vram_gb ? ` · ${sys.info.vram_gb} GB` : ""}</span>}
            </div>
            <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
              {[...cat.presets].sort((a, b) => Number(recIds.has(b.id)) - Number(recIds.has(a.id))).map((p) => (
                <button key={p.id} onClick={() => applyPreset(p)} className={cx("rounded-lg border p-3.5 text-left transition cursor-pointer", preset === p.id ? "border-accent bg-accent-soft" : "border-line bg-panel hover:border-line-strong")}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-1.5 text-[13px] font-medium">{recIds.has(p.id) && <Star className="size-3 text-accent fill-current" />}{p.name}</span>
                    <Badge>{p.badge}</Badge>
                  </div>
                  <div className="mt-1.5 text-[12px] leading-relaxed text-muted">{p.description}</div>
                  <div className="mt-2 truncate font-mono text-[11px] text-faint">{p.config.base_model}</div>
                </button>
              ))}
            </div>
          </div>
          <Card>
            <CardHeader
              title="Base model"
              description="Encoders are fastest (Laya-style). LLMs reason over longer states. Decision models continue from an existing System One checkpoint."
              right={<Segmented value={filter} onChange={setFilter} options={[{ id: "all", label: "All" }, { id: "encoder", label: "Encoders" }, { id: "causal", label: "LLMs" }, { id: "decision", label: "Decision models" }]} />}
            />
            <div className="overflow-x-auto">
              <table className="w-full text-[13px]">
                <thead className="text-left text-[12px] text-faint">
                  <tr className="border-b border-line">
                    <th className="px-4 py-2 font-normal">Model</th><th className="px-2 py-2 font-normal">Params</th><th className="px-2 py-2 font-normal">Type</th>
                    <th className="px-2 py-2 font-normal">Languages</th><th className="px-2 py-2 font-normal">On this machine</th><th className="px-4 py-2 font-normal">License</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => {
                    const f = fits[m.id];
                    const sel = cfg.base_model === m.id;
                    return (
                      <tr key={m.id} onClick={() => pickModel(m)} className={cx("cursor-pointer border-b border-line last:border-0 transition", sel ? "bg-accent-soft" : "hover:bg-hover")}>
                        <td className="px-4 py-2.5">
                          <div className="flex items-center gap-2">
                            <span className={cx("flex size-3.5 items-center justify-center rounded-full border", sel ? "border-accent bg-accent" : "border-line-strong")}>{sel && <span className="size-1.5 rounded-full bg-accent-fg" />}</span>
                            <span className="font-medium">{m.name}</span>
                            {m.support === "experimental" && <Badge t="warn">experimental</Badge>}
                          </div>
                          <div className="ml-5.5 mt-0.5 font-mono text-[11.5px] text-faint">{m.id}</div>
                        </td>
                        <td className="px-2 font-mono tabular text-muted">{m.params < 1 ? `${Math.round(m.params * 1000)}M` : `${m.params.toFixed(1)}B`}</td>
                        <td className="px-2"><Badge t={m.arch === "encoder" ? "info" : "neutral"}>{m.arch === "encoder" ? "encoder" : "LLM"}</Badge></td>
                        <td className="px-2 text-[12px] text-muted">{m.langs}</td>
                        <td className="px-2">{f && <span className="flex items-center gap-1.5"><Badge t={fitTone[f.fit]}>{fitLabel[f.fit]}</Badge><span className="text-[12px] text-faint">{f.finetune} · ~{f.memory_gb} GB</span></span>}</td>
                        <td className="px-4 text-[12px] text-muted">{m.license}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="flex gap-2 border-t border-line p-3">
              <Input placeholder="Any Hugging Face model id…" value={custom} onChange={(e) => setCustom(e.target.value)} />
              <Button onClick={() => custom && pickModel({ id: custom.trim() })} disabled={!custom}>Use custom</Button>
            </div>
          </Card>
          {sys?.recommendations.tips.map((t) => <Alert key={t} t="info">{t}</Alert>)}
        </div>
      )}

      {step === 1 && (
        <div className="space-y-6">
          {!creating && (
            <>
              <div className="flex items-center justify-between">
                <div className="text-[13px] font-medium">Choose a dataset</div>
                <Button icon={<Sparkles className="size-3.5" />} onClick={() => setCreating(true)}>Create new dataset</Button>
              </div>
              {ds.data?.length ? (
                <div className="grid gap-2.5 md:grid-cols-2 lg:grid-cols-3">
                  {ds.data.map((d) => (
                    <button key={d.name} onClick={() => setCfg({ ...cfg, data: { ...cfg.data, dataset: d.name } })} className={cx("rounded-lg border p-3.5 text-left transition cursor-pointer", cfg.data.dataset === d.name ? "border-accent bg-accent-soft" : "border-line bg-panel hover:border-line-strong")}>
                      <div className="flex items-center justify-between gap-2">
                        <span className="flex items-center gap-2 truncate text-[13px] font-medium"><Database className="size-3.5 text-faint" />{d.name}</span>
                        <Badge>{d.source.kind}</Badge>
                      </div>
                      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[12px] text-muted">
                        <span>{Object.keys(d.question_types).join(", ")}</span>
                        <span>{d.num_labels} labels</span>
                        <span className="tabular">{d.counts.train}/{d.counts.validation}/{d.counts.test}</span>
                      </div>
                      {d.example_question?.instructions && <div className="mt-1.5 line-clamp-1 text-[12px] text-faint">“{d.example_question.instructions}”</div>}
                    </button>
                  ))}
                </div>
              ) : (
                <Alert t="info">No datasets yet. Create one from Hugging Face, a file, an LLM, or Faker templates.</Alert>
              )}
            </>
          )}
          {creating && (
            <Card className="p-5">
              <div className="mb-3 flex items-center justify-between">
                <div className="text-[14px] font-medium">New dataset</div>
                <Button variant="ghost" size="sm" onClick={() => setCreating(false)}>Cancel</Button>
              </div>
              <DatasetCreator onCreated={(m) => { ds.reload(); setCfg({ ...cfg, data: { ...cfg.data, dataset: m.name } }); setCreating(false); }} />
            </Card>
          )}
          {dataset && !creating && (
            <Card className="grid gap-4 p-4 md:grid-cols-3">
              <Field label="Faker augmentation" hint="Probability of swapping emails, phones, amounts and IDs with fake values per example." onInfo={() => {}}>
                <div className="flex items-center gap-3">
                  <input type="range" min={0} max={0.8} step={0.05} value={cfg.data.faker_augment} onChange={(e) => setCfg({ ...cfg, data: { ...cfg.data, faker_augment: Number(e.target.value) } })} className="w-full" />
                  <span className="w-10 text-right font-mono text-[12px]">{cfg.data.faker_augment}</span>
                </div>
              </Field>
              <Field label="Max training samples" hint={`Train split has ${dataset.counts.train}. Leave empty for all.`}>
                <Input type="number" value={cfg.data.max_train_samples ?? ""} onChange={(e) => setCfg({ ...cfg, data: { ...cfg.data, max_train_samples: e.target.value ? Number(e.target.value) : null } })} />
              </Field>
              <Field label="Suggested max length" hint={`p95 state ≈ ${dataset.state_chars_p95} chars ≈ ${Math.round(dataset.state_chars_p95 / 4)} tokens + options.`}>
                <Button size="sm" onClick={() => set("max_length", Math.min(4096, Math.max(128, Math.ceil((dataset.state_chars_p95 / 4 + dataset.num_labels * 12 + 64) / 64) * 64)))}>Apply suggestion</Button>
              </Field>
            </Card>
          )}
        </div>
      )}

      {step === 2 && (
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          <div className="space-y-5">
            {["Method", "Optimisation", "LoRA", "RLCD", "Hardware", "Evaluation"].map((group) => {
              if (group === "LoRA" && !["lora", "qlora"].includes(cfg.finetune)) return null;
              if (group === "RLCD" && !cfg.method.includes("rlcd")) return null;
              const fs = FIELDS.filter((f) => ex[f.key]?.group === group);
              return (
                <Card key={group}>
                  <CardHeader title={group} description={group === "RLCD" ? "Reinforcement Learning for Calibrated Decisions: confidence that matches accuracy." : undefined} />
                  <div className="grid gap-4 p-4 sm:grid-cols-2 lg:grid-cols-3">
                    {fs.map((f) => (
                      <div key={f.key} onMouseEnter={() => setFocus(f.key)}>
                        <Field label={ex[f.key].label} active={focus === f.key} onInfo={() => setFocus(f.key)}>
                          {f.kind === "sel" ? (
                            <Select value={cfg[f.key]} onFocus={() => setFocus(f.key)} onChange={(e) => set(f.key, e.target.value)} options={f.options!} />
                          ) : f.kind === "bool" ? (
                            <div className="flex h-8.5 items-center"><Toggle checked={!!cfg[f.key]} onChange={(v) => { set(f.key, v); setFocus(f.key); }} label={<span className="text-muted">{cfg[f.key] ? "On" : "Off"}</span>} /></div>
                          ) : f.kind === "text" ? (
                            <Input value={cfg[f.key]} onFocus={() => setFocus(f.key)} onChange={(e) => set(f.key, e.target.value)} />
                          ) : (
                            <Input type="number" step={f.step} min={f.min} value={cfg[f.key] ?? ""} onFocus={() => setFocus(f.key)} onChange={(e) => set(f.key, e.target.value === "" ? null : Number(e.target.value))} className="font-mono" />
                          )}
                        </Field>
                      </div>
                    ))}
                  </div>
                </Card>
              );
            })}
          </div>
          <div className="space-y-4 lg:sticky lg:top-0 lg:self-start">
            <Card className="p-4">
              <div className="mb-1 flex items-center gap-2 text-[12px] text-accent"><Info className="size-3.5" />{ex[focus]?.group}</div>
              <div className="text-[15px] font-medium">{ex[focus]?.label}</div>
              <div className="mt-0.5 font-mono text-[11.5px] text-faint">{focus}</div>
              <p className="mt-3 text-[13px] text-fg">{ex[focus]?.short}</p>
              {ex[focus]?.detail && <p className="mt-2 text-[13px] leading-relaxed text-muted">{ex[focus].detail}</p>}
              <dl className="mt-4 space-y-2 text-[12.5px]">
                <div className="flex gap-2"><dt className="w-16 shrink-0 text-faint">Typical</dt><dd className="font-mono text-fg">{ex[focus]?.typical}</dd></div>
                {ex[focus]?.up && <div className="flex gap-2"><dt className="w-16 shrink-0 text-faint">Higher</dt><dd className="text-muted">{ex[focus].up}</dd></div>}
                {ex[focus]?.down && <div className="flex gap-2"><dt className="w-16 shrink-0 text-faint">Lower</dt><dd className="text-muted">{ex[focus].down}</dd></div>}
              </dl>
              {ex[focus]?.tip && <div className="mt-4"><Alert t="accent">{ex[focus].tip}</Alert></div>}
            </Card>
            {mem && (
              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-[12.5px] text-muted"><Cpu className="size-3.5" /> Estimated peak memory</div>
                  <Badge t={fitTone[mem.fit as keyof typeof fitTone]}>{fitLabel[mem.fit as keyof typeof fitLabel]}</Badge>
                </div>
                <div className="mt-2 text-[22px] font-medium tabular">{mem.memory_gb} GB <span className="text-[13px] font-normal text-faint">/ {mem.budget_gb} GB</span></div>
                <div className="mt-2 h-1.5 rounded-full bg-panel-2"><div className="h-1.5 rounded-full" style={{ width: `${Math.min(100, (mem.memory_gb / Math.max(mem.budget_gb, 0.1)) * 100)}%`, background: mem.fit === "no" ? "var(--danger)" : mem.fit === "fits" ? "var(--accent)" : "var(--warn)" }} /></div>
                <div className="mt-2 text-[12px] text-faint">
                  {mem.fit === "auto"
                    ? <>Snapcall will run micro-batches of {mem.micro_batch} × {Math.round(cfg.batch_size / mem.micro_batch)} accumulation steps, so the effective batch stays {cfg.batch_size * cfg.grad_accum}.</>
                    : <>Effective batch {cfg.batch_size * cfg.grad_accum}.</>} Rough estimate (±30%), based on max length.
                </div>
              </Card>
            )}
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
          <div className="space-y-4">
            <Card>
              <CardHeader title="Summary" />
              <div className="grid gap-x-6 gap-y-3 p-4 text-[13px] sm:grid-cols-2">
                {[
                  ["Base model", cfg.base_model], ["Dataset", cfg.data.dataset], ["Method", cfg.method], ["Fine-tuning", cfg.finetune],
                  ["Epochs", `${cfg.epochs}${cfg.method.includes("rlcd") ? ` + ${cfg.rlcd_epochs} RLCD` : ""}`], ["Learning rate", cfg.learning_rate],
                  ["Batch", `${cfg.batch_size} × ${cfg.grad_accum}`], ["Max length", cfg.max_length],
                  ["Train rows", dataset ? (cfg.data.max_train_samples ? Math.min(cfg.data.max_train_samples, dataset.counts.train) : dataset.counts.train) : "—"],
                  ["Est. steps", dataset ? Math.ceil(((cfg.data.max_train_samples ? Math.min(cfg.data.max_train_samples, dataset.counts.train) : dataset.counts.train) / cfg.batch_size / cfg.grad_accum) * (cfg.epochs + (cfg.method.includes("rlcd") ? cfg.rlcd_epochs : 0))) : "—"],
                ].map(([k, v]) => (
                  <div key={k as string} className="flex justify-between gap-4 border-b border-line pb-2"><span className="text-muted">{k}</span><span className="truncate font-mono text-[12.5px]">{String(v)}</span></div>
                ))}
              </div>
            </Card>
            <Card className="p-4">
              <div className="mb-2 text-[12.5px] text-muted">Equivalent CLI</div>
              <Code>{cli}</Code>
            </Card>
          </div>
          <div className="space-y-4">
            <Card className="p-4 space-y-4">
              <Field label="Run name (optional)"><Input value={cfg.run_name ?? ""} onChange={(e) => set("run_name", e.target.value)} placeholder={cfg.base_model.split("/").pop()} /></Field>
              {mem?.fit === "no" && <Alert t="danger">This configuration likely won't fit in memory even at micro-batch 1. Try LoRA/QLoRA or a shorter max length.</Alert>}
              {model?.support === "experimental" && <Alert t="warn">Experimental model: may need a newer transformers or trust_remote_code.</Alert>}
              <Button variant="primary" className="w-full" icon={<Play className="size-3.5" />} loading={submitting} onClick={submit}>Start training</Button>
              <div className="text-[12px] text-faint">Runs in a background process. You can close this tab; progress is saved under ~/.snapcall/runs.</div>
            </Card>
            <details className="text-[12.5px] text-muted">
              <summary className="cursor-pointer hover:text-fg">Full config JSON</summary>
              <Code className="mt-2">{JSON.stringify(cfg, null, 2)}</Code>
            </details>
          </div>
        </div>
      )}

      <div className="mt-8 flex items-center justify-between border-t border-line pt-5">
        <Button variant="ghost" icon={<ArrowLeft className="size-3.5" />} disabled={step === 0} onClick={() => setStep(step - 1)}>Back</Button>
        <div className="text-[12px] text-faint">{cfg.base_model}{cfg.data.dataset && ` · ${cfg.data.dataset}`}</div>
        {step < 3 ? (
          <Button variant="primary" disabled={!canNext} onClick={() => setStep(step + 1)}>Continue <ArrowRight className="size-3.5" /></Button>
        ) : (
          <Button variant="primary" loading={submitting} onClick={submit} icon={<Play className="size-3.5" />}>Train</Button>
        )}
      </div>
    </div>
  );
}

