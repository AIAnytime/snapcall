import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { FlaskConical } from "lucide-react";
import type { Run } from "../api";
import { Button, Empty, Field, PageHeader, Select, useFetch } from "../components/ui";
import PlaygroundPanel from "../components/PlaygroundPanel";

export default function Playground() {
  const { data } = useFetch<Run[]>("/api/runs");
  const models = (data ?? []).filter((r) => r.has_model);
  const [id, setId] = useState("");
  useEffect(() => {
    if (!id && models.length) setId(models[0].id);
  }, [models, id]);
  const run = models.find((r) => r.id === id);
  return (
    <div>
      <PageHeader title="Playground" description="Ask a trained model typed questions about any state. Multiple questions share one state." />
      {data && !models.length ? (
        <Empty icon={<FlaskConical className="size-6" />} title="No trained models yet" action={<Link to="/new"><Button variant="primary">Train a model</Button></Link>} />
      ) : (
        <>
          <div className="mb-5 max-w-md"><Field label="Model"><Select value={id} onChange={(e) => setId(e.target.value)} options={models.map((m) => ({ value: m.id, label: `${m.config.run_name || m.id} · ${m.config.base_model}` }))} /></Field></div>
          {run && <PlaygroundPanel runId={run.id} meta={run.model_meta} />}
        </>
      )}
    </div>
  );
}
