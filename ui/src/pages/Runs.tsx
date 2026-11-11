import { Link, useNavigate } from "react-router-dom";
import { Activity, Plus } from "lucide-react";
import { useEffect } from "react";
import { fmt, type Run } from "../api";
import { Button, Card, Empty, PageHeader, Progress, StatusBadge, useFetch } from "../components/ui";

export function RunsTable({ runs, compact }: { runs: Run[]; compact?: boolean }) {
  const nav = useNavigate();
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[13px]">
        <thead className="text-left text-[12px] text-faint">
          <tr className="border-b border-line">
            <th className="px-4 py-2 font-normal">Run</th>
            <th className="px-2 py-2 font-normal">Status</th>
            {!compact && <th className="px-2 py-2 font-normal">Base model</th>}
            <th className="px-2 py-2 font-normal">Dataset</th>
            <th className="px-2 py-2 font-normal">Accuracy</th>
            <th className="px-2 py-2 font-normal">ECE</th>
            <th className="px-4 py-2 text-right font-normal">Created</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const fin = r.status.summary?.final?.test || r.status.summary?.final?.validation;
            const best = r.status.best;
            return (
              <tr key={r.id} onClick={() => nav(`/runs/${r.id}`)} className="cursor-pointer border-b border-line last:border-0 hover:bg-hover">
                <td className="px-4 py-2.5">
                  <div className="font-medium">{r.config.run_name || r.id}</div>
                  <div className="text-[11.5px] text-faint">{r.config.method} · {r.config.finetune}</div>
                </td>
                <td className="px-2">
                  <div className="flex items-center gap-2">
                    <StatusBadge state={r.status.state} />
                    {r.status.state === "running" && <Progress value={r.status.progress || 0} className="w-16" />}
                  </div>
                </td>
                {!compact && <td className="px-2 font-mono text-[12px] text-muted">{r.config.base_model}</td>}
                <td className="px-2 text-muted">{r.config.data?.dataset}</td>
                <td className="px-2 tabular">{fin ? fmt.pct(fin.accuracy) : best ? <span className="text-muted">{fmt.pct(best.accuracy)}</span> : "—"}</td>
                <td className="px-2 tabular">{fin ? fmt.num(fin.ece, 3) : "—"}</td>
                <td className="px-4 text-right text-[12px] text-muted">{fmt.ago(r.created)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function Runs() {
  const { data, reload } = useFetch<Run[]>("/api/runs");
  useEffect(() => {
    if (!data?.some((r) => r.status.state === "running" || r.status.state === "queued")) return;
    const t = setInterval(reload, 3000);
    return () => clearInterval(t);
  }, [data, reload]);
  return (
    <div>
      <PageHeader title="Runs" description="Every training run, its status and final calibrated metrics." actions={<Link to="/new"><Button variant="primary" icon={<Plus className="size-3.5" />}>New run</Button></Link>} />
      {data && data.length === 0 ? (
        <Empty icon={<Activity className="size-6" />} title="No runs yet" action={<Link to="/new"><Button variant="primary">Start your first run</Button></Link>}>Pick a preset and a dataset; a laptop run takes minutes.</Empty>
      ) : (
        <Card>{data && <RunsTable runs={data} />}</Card>
      )}
    </div>
  );
}
