import { useEffect, useState } from "react";
import { api, type Json } from "../api";
import { Alert, Button, Card, CardHeader, Field, Input, PageHeader, Select, useFetch, useToast } from "../components/ui";

export default function SettingsPage() {
  const toast = useToast();
  const { data, setData } = useFetch<Json>("/api/settings");
  const [form, setForm] = useState<Json>({});
  const [busy, setBusy] = useState(false);
  const [hf, setHf] = useState<string>("");
  useEffect(() => {
    if (data) setForm(data);
  }, [data]);
  const save = async () => {
    setBusy(true);
    try {
      const r = await api.post("/api/settings", form);
      setData(r);
      toast("Settings saved");
    } catch (e: any) {
      toast(e.message, "err");
    } finally {
      setBusy(false);
    }
  };
  const testHf = async () => {
    try {
      const r = await api.post("/api/settings/test-hf");
      setHf(`✓ ${r.name}${r.orgs.length ? ` · orgs: ${r.orgs.join(", ")}` : ""}`);
    } catch (e: any) {
      setHf(`✗ ${e.message}`);
    }
  };
  const f = (k: string) => ({ value: form[k] ?? "", onChange: (e: any) => setForm({ ...form, [k]: e.target.value }) });
  return (
    <div className="max-w-3xl space-y-5">
      <PageHeader title="Settings" description={<>Stored locally in <span className="font-mono">{data?.snapcall_home}/settings.json</span> (chmod 600). Environment variables are used as fallbacks.</>} />
      <Card>
        <CardHeader title="Hugging Face" description="Needed for gated models, private datasets and pushing models. Use a write token." />
        <div className="space-y-3 p-4">
          <Field label="Access token" hint="HF_TOKEN"><Input type="password" placeholder="hf_…" {...f("hf_token")} className="font-mono" /></Field>
          <div className="flex items-center gap-3"><Button size="sm" onClick={testHf} disabled={!data?.hf_token}>Test token</Button><span className="text-[12.5px] text-muted">{hf}</span></div>
        </div>
      </Card>
      <Card>
        <CardHeader title="Synthetic data LLM" description="Used only when you generate synthetic datasets." />
        <div className="space-y-4 p-4">
          <Field label="Default provider"><Select {...f("synth_provider")} options={[{ value: "anthropic", label: "Anthropic" }, { value: "openai", label: "OpenAI-compatible (OpenAI, OpenRouter, Groq, Together, vLLM, Ollama…)" }]} /></Field>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Anthropic API key" hint="ANTHROPIC_API_KEY"><Input type="password" placeholder="sk-ant-…" {...f("anthropic_api_key")} className="font-mono" /></Field>
            <Field label="Anthropic model" hint="claude-haiku-5-5 is a cheaper option for bulk generation"><Input {...f("anthropic_model")} className="font-mono" /></Field>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <Field label="OpenAI-compatible key" hint="OPENAI_API_KEY"><Input type="password" placeholder="sk-…" {...f("openai_api_key")} className="font-mono" /></Field>
            <Field label="Base URL"><Input {...f("openai_base_url")} className="font-mono" /></Field>
            <Field label="Model"><Input {...f("openai_model")} className="font-mono" /></Field>
          </div>
          <Alert t="info">For a local model, point the base URL at e.g. <span className="font-mono">http://localhost:11434/v1</span> (Ollama); no key needed.</Alert>
        </div>
      </Card>
      <div className="flex justify-end"><Button variant="primary" loading={busy} onClick={save}>Save settings</Button></div>
    </div>
  );
}
