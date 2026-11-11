import { Link } from "react-router-dom";
import { ArrowRight, Database, FlaskConical, Gauge, Plus, Sparkles, Terminal } from "lucide-react";
import { fmt, type DatasetMeta, type Run, type System } from "../api";
import { Badge, Button, Card, CardHeader, Code, Stat, useFetch } from "../components/ui";
import { RunsTable } from "./Runs";

export default function Overview() {
  const { data: runs } = useFetch<Run[]>("/api/runs");
  const { data: ds } = useFetch<DatasetMeta[]>("/api/datasets");
  const { data: sys } = useFetch<System>("/api/system");
  const done = (runs ?? []).filter((r) => r.status.state === "completed");
  const running = (runs ?? []).filter((r) => r.status.state === "running");
  const finals = done.map((r) => r.status.summary?.final?.test || r.status.summary?.final?.validation).filter(Boolean);
  const bestAcc = finals.length ? Math.max(...finals.map((f: any) => f.accuracy)) : undefined;
  const bestEce = finals.length ? Math.min(...finals.map((f: any) => f.ece)) : undefined;
  const rec = sys?.recommendations.presets[0];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="mb-2 font-mono text-[12px] text-accent">Train. Evaluate. Deploy.</div>
          <h1 className="text-[22px] font-medium tracking-tight">Build better decision models.</h1>
          <p className="mt-1.5 max-w-2xl text-[13.5px] text-muted">
            Snapcall is an open-source toolkit for training, evaluating and optimizing Jev-style System One decision models: typed questions in, calibrated probabilities out, one forward pass, no invalid outputs.
          </p>
        </div>
        <Link to="/new"><Button variant="primary" icon={<Plus className="size-3.5" />}>New run</Button></Link>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Runs" value={runs?.length ?? "—"} sub={running.length ? `${running.length} training now` : `${done.length} completed`} />
        <Stat label="Datasets" value={ds?.length ?? "—"} sub={ds ? `${ds.reduce((a, d) => a + d.counts.train, 0).toLocaleString()} training rows` : ""} />
        <Stat label="Best accuracy" value={fmt.pct(bestAcc)} accent={bestAcc !== undefined} />
        <Stat label="Best ECE" value={fmt.num(bestEce, 3)} sub="lower is better calibrated" />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {[
          { to: "/datasets", icon: Database, title: "1. Prepare data", body: "Import from Hugging Face, upload JSONL/CSV, generate with an LLM, or template it with Faker." },
          { to: "/new", icon: Gauge, title: "2. Train with RLCD", body: "SFT a typed decision head, then calibrate with RLCD and temperature scaling. Watch it live." },
          { to: "/playground", icon: FlaskConical, title: "3. Test & ship", body: "Probe the model, serve a /v1/systemone API, push to the Hub with an auto model card." },
        ].map((c) => (
          <Link key={c.to} to={c.to}>
            <Card className="group h-full p-4 transition hover:border-line-strong">
              <c.icon className="size-4 text-accent" />
              <div className="mt-3 flex items-center justify-between text-[13.5px] font-medium">{c.title}<ArrowRight className="size-3.5 text-faint transition group-hover:translate-x-0.5 group-hover:text-fg" /></div>
              <div className="mt-1 text-[12.5px] leading-relaxed text-muted">{c.body}</div>
            </Card>
          </Link>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
        <Card>
          <CardHeader title="Recent runs" right={<Link to="/runs" className="text-[12.5px] text-muted hover:text-fg">View all</Link>} />
          {runs?.length ? <RunsTable runs={runs.slice(0, 6)} compact /> : <div className="p-6 text-center text-[13px] text-muted">No runs yet.</div>}
        </Card>
        <div className="space-y-4">
          {rec && (
            <Card className="p-4">
              <div className="flex items-center gap-2 text-[12px] text-accent"><Sparkles className="size-3.5" />Recommended for your machine</div>
              <div className="mt-2 flex items-center justify-between"><span className="text-[14px] font-medium">{rec.name}</span><Badge>{rec.badge}</Badge></div>
              <div className="mt-1 text-[12.5px] text-muted">{rec.description}</div>
              <div className="mt-2 font-mono text-[11.5px] text-faint">{rec.config.base_model}</div>
              <Link to="/new"><Button className="mt-3 w-full" size="sm">Use this preset</Button></Link>
            </Card>
          )}
          <Card className="p-4">
            <div className="mb-2 flex items-center gap-2 text-[12.5px] text-muted"><Terminal className="size-3.5" />Prefer the terminal?</div>
            <Code>{`snapcall doctor\nsnapcall data hf banking77\nsnapcall train -p ${rec?.id ?? "laptop-quick"} -d banking77\nsnapcall push <run> --repo you/model`}</Code>
          </Card>
        </div>
      </div>
    </div>
  );
}
