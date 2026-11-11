import { useState } from "react";
import { Trash2 } from "lucide-react";
import { api, fmt, type Json } from "../api";
import { Button, Card, CardHeader, PageHeader, Stat, StatusBadge, useFetch, useToast } from "../components/ui";

export default function Cleanup() {
  const toast = useToast();
  const du = useFetch<Json>("/api/cleanup");
  const [busy, setBusy] = useState("");
  const [sel, setSel] = useState<Set<string>>(new Set());
  const go = async (key: string, body: Json, confirmMsg?: string) => {
    if (confirmMsg && !confirm(confirmMsg)) return;
    setBusy(key);
    try {
      const r = await api.post("/api/cleanup", body);
      toast(`Freed ${fmt.bytes(r.freed_bytes)}`);
      setSel(new Set());
      du.reload();
    } catch (e: any) {
      toast(e.message, "err");
    } finally {
      setBusy("");
    }
  };
  const d = du.data;
  return (
    <div className="space-y-6">
      <PageHeader title="Cleanup" description={<>Reclaim disk space. Workspace: <span className="font-mono">{d?.home}</span></>} />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Runs" value={d ? fmt.bytes(d.runs_total) : "—"} sub={`${d?.runs.length ?? 0} runs`} />
        <Stat label="Intermediate checkpoints" value={d ? fmt.bytes(d.checkpoints_total) : "—"} sub="safe to delete" />
        <Stat label="Datasets" value={d ? fmt.bytes(d.datasets) : "—"} />
        <Stat label="Hugging Face cache" value={d ? fmt.bytes(d.hf_cache_total) : "—"} sub={`${d?.hf_cache.length ?? 0} repos`} />
      </div>
      <Card className="flex flex-wrap gap-2 p-4">
        <Button loading={busy === "ck"} onClick={() => go("ck", { checkpoints: true })} icon={<Trash2 className="size-3.5" />}>Delete all checkpoints</Button>
        <Button loading={busy === "tb"} onClick={() => go("tb", { checkpoints: false, tensorboard: true })}>Delete TensorBoard events</Button>
        <Button loading={busy === "failed"} variant="danger" onClick={() => go("failed", { checkpoints: false, failed_runs: true }, "Delete failed and stopped runs that have no saved model?")}>Delete failed runs</Button>
      </Card>
      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <CardHeader title="Runs by size" />
          <table className="w-full text-[12.5px]">
            <tbody>
              {(d?.runs ?? []).map((r: Json) => (
                <tr key={r.id} className="border-b border-line last:border-0">
                  <td className="px-4 py-2"><div className="truncate font-mono">{r.id}</div><div className="text-[11.5px] text-faint">model {fmt.bytes(r.model)} · ckpt {fmt.bytes(r.checkpoints)} · tb {fmt.bytes(r.tensorboard)}</div></td>
                  <td className="px-2"><StatusBadge state={r.state} /></td>
                  <td className="px-4 text-right font-mono tabular">{fmt.bytes(r.total)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card>
          <CardHeader title="Hugging Face cache" description="Downloaded base models and datasets." right={<Button size="sm" variant="danger" disabled={!sel.size} loading={busy === "hf"} onClick={() => go("hf", { checkpoints: false, hf_repos: [...sel] }, `Delete ${sel.size} cached repos? They will re-download when needed.`)}>Delete {sel.size || ""}</Button>} />
          <table className="w-full text-[12.5px]">
            <tbody>
              {(d?.hf_cache ?? []).map((r: Json) => (
                <tr key={r.repo_id + r.type} className="border-b border-line last:border-0">
                  <td className="w-8 pl-4"><input type="checkbox" checked={sel.has(r.repo_id)} onChange={(e) => { const s = new Set(sel); e.target.checked ? s.add(r.repo_id) : s.delete(r.repo_id); setSel(s); }} /></td>
                  <td className="px-2 py-2"><div className="font-mono">{r.repo_id}</div><div className="text-[11.5px] text-faint">{r.type} · used {fmt.ago(r.last_accessed)}</div></td>
                  <td className="px-4 text-right font-mono tabular">{fmt.bytes(r.size)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>
    </div>
  );
}
