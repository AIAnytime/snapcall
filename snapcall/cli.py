"""`snapcall`: the command line."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from . import __version__

app = typer.Typer(help="Snapcall: build better decision models. Train, evaluate and deploy calibrated System One models. Built by AI Anytime.",
                  no_args_is_help=True, rich_markup_mode="rich", add_completion=False)
data_app = typer.Typer(help="Prepare datasets (Hugging Face, files, Faker, LLM synthetic).", no_args_is_help=True)
app.add_typer(data_app, name="data")
con = Console()

ACCENT = "#3ecf8e"


def _gb(b: float) -> str:
    return f"{b / 1e9:.2f} GB" if b > 1e8 else f"{b / 1e6:.1f} MB"


def _parse_set(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


# ---------------------------------------------------------------- basics
@app.command()
def version():
    """Print the version."""
    con.print(f"snapcall [bold]{__version__}[/]")


@app.command()
def ui(port: int = typer.Option(7860, help="Port"), host: str = "127.0.0.1",
       no_browser: bool = typer.Option(False, "--no-browser")):
    """Launch the dashboard."""
    import threading
    import webbrowser

    import uvicorn
    url = f"http://{host}:{port}"
    con.print(Panel.fit(f"[bold]Snapcall[/] v{__version__}\nDashboard  [link={url}]{url}[/]\n"
                        f"[dim]Built by AI Anytime[/]", border_style=ACCENT))
    if not no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run("snapcall.server.app:app", host=host, port=port, log_level="warning")


@app.command()
def doctor():
    """Detect hardware and recommend models, presets and settings."""
    from .compute import detect, recommend
    info = detect()
    rec = recommend(info)
    t = Table(box=box.SIMPLE_HEAD, show_header=False)
    for k in ("platform", "accelerator", "device", "vram_gb", "ram_gb", "cpu_cores", "disk_free_gb", "torch",
              "transformers", "bf16", "qlora"):
        t.add_row(f"[dim]{k}[/]", str(info.get(k)))
    con.print(Panel(t, title="Compute", border_style=ACCENT, expand=False))
    con.print(f"Tier: [bold]{rec['tier']}[/]   memory budget ≈ {rec['memory_budget_gb']} GB\n")
    pt = Table(title="Recommended presets", box=box.SIMPLE_HEAD)
    pt.add_column("preset", style="bold")
    pt.add_column("model")
    pt.add_column("description", style="dim")
    for p in rec["presets"]:
        pt.add_row(p["id"], p["config"]["base_model"], p["description"])
    con.print(pt)
    for tip in rec["tips"]:
        con.print(f"[yellow]•[/] {tip}")
    con.print(f"\nStart: [bold]snapcall train --preset {rec['presets'][0]['id']} --dataset <name>[/]")


@app.command()
def models():
    """List supported base models with a fit estimate for this machine."""
    from .catalog import MODELS
    from .compute import recommend
    fit = {m["id"]: m for m in recommend()["models"]}
    t = Table(box=box.SIMPLE_HEAD)
    for c in ("model", "params", "arch", "kind", "languages", "fits", "mode", "support"):
        t.add_column(c)
    colors = {"fits": "green", "tight": "yellow", "slow": "yellow", "no": "red", "cpu": "yellow"}
    for m in MODELS:
        f = fit[m["id"]]
        t.add_row(m["id"], f"{m['params']:.2f}B", m["arch"], m["kind"], m["langs"],
                  f"[{colors.get(f['fit'], 'white')}]{f['fit']}[/] ~{f['memory_gb']}GB", f["finetune"], m["support"])
    con.print(t)


@app.command()
def presets():
    """List training presets."""
    from .catalog import PRESETS
    for p in PRESETS:
        con.print(f"[bold {ACCENT}]{p['id']}[/]  {p['name']}  [dim]({p['badge']})[/]\n  {p['description']}")
        con.print("  [dim]" + "  ".join(f"{k}={v}" for k, v in p["config"].items()) + "[/]\n")


@app.command()
def explain(param: Optional[str] = typer.Argument(None, help="Hyperparameter name; omit to list all")):
    """Explain hyperparameters: what they do and how to tune them."""
    from .config import HPARAMS
    if not param:
        t = Table(box=box.SIMPLE_HEAD)
        t.add_column("group", style="dim")
        t.add_column("name", style="bold")
        t.add_column("what it does")
        t.add_column("typical", style=ACCENT)
        for k, v in HPARAMS.items():
            t.add_row(v["group"], k, v["short"], v["typical"])
        con.print(t)
        con.print("[dim]snapcall explain <name> for details[/]")
        return
    v = HPARAMS.get(param)
    if not v:
        raise typer.BadParameter(f"unknown hyperparameter {param}")
    body = f"{v['short']}\n\n{v['detail']}\n\n[bold]Typical:[/] {v['typical']}"
    if v.get("up"):
        body += f"\n[bold]Higher:[/] {v['up']}"
    if v.get("down"):
        body += f"\n[bold]Lower:[/] {v['down']}"
    if v.get("tip"):
        body += f"\n\n[{ACCENT}]Tip:[/] {v['tip']}"
    con.print(Panel(body, title=f"{v['label']}  ({param})", border_style=ACCENT, expand=False))


@app.command()
def init(preset: str = typer.Option("laptop-quick", help="Preset to start from"),
         dataset: str = typer.Option("my-dataset"), out: Path = typer.Option(Path("snapcall.yaml"))):
    """Write a starter config file."""
    import yaml

    from .catalog import get_preset
    from .config import DataConfig, TrainConfig
    p = get_preset(preset)
    if not p:
        raise typer.BadParameter(f"unknown preset {preset}")
    cfg = TrainConfig(data=DataConfig(dataset=dataset), **p["config"])
    out.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json()), sort_keys=False))
    con.print(f"Wrote [bold]{out}[/]. Edit it, then run [bold]snapcall train --config {out}[/]")


# ---------------------------------------------------------------- data
@data_app.command("list")
def data_list():
    """List prepared datasets."""
    from .data import list_datasets
    t = Table(box=box.SIMPLE_HEAD)
    for c in ("name", "source", "type", "labels", "train", "val", "test"):
        t.add_column(c)
    for d in list_datasets():
        c = d["counts"]
        t.add_row(d["name"], d["source"].get("kind", ""), ",".join(d.get("question_types", {})),
                  str(d.get("num_labels")), str(c["train"]), str(c["validation"]), str(c["test"]))
    con.print(t)


@data_app.command("show")
def data_show(name: str, n: int = 5, split: str = "train"):
    """Show dataset stats and sample rows."""
    from .data import get_meta, preview
    con.print_json(json.dumps({k: v for k, v in get_meta(name).items() if k != "label_counts"}))
    for r in preview(name, split, n):
        con.print(f"[dim]{json.dumps(r['answer'])}[/]  {str(r['state'])[:160]}")


@data_app.command("hf")
def data_hf(source: str = typer.Argument(..., help="Preset id (banking77, emotion, ...) or a HF dataset id"),
            name: Optional[str] = None, config: Optional[str] = None, text_col: Optional[str] = None,
            label_col: Optional[str] = None, type: str = "choice", instructions: str = "",
            max_per_split: int = 5000):
    """Import a Hugging Face dataset."""
    from .data.hub import HF_PRESETS, import_hf
    from .paths import load_settings
    p = next((x for x in HF_PRESETS if x["id"] == source), None)
    if p:
        source, config = p["hf_id"], config or p["config"]
        text_col, label_col = text_col or p["text_col"], label_col or p["label_col"]
        type, instructions = p["type"], instructions or p["instructions"]
        name = name or p["id"]
        icol = p.get("instructions_col")
    else:
        icol = None
        if not (text_col and label_col):
            raise typer.BadParameter("--text-col and --label-col are required for non-preset datasets")
    with con.status("Importing…") as st:
        meta = import_hf(source, name=name or source.split("/")[-1], text_col=text_col, label_col=label_col,
                         qtype=type, instructions=instructions, instructions_col=icol, config=config,
                         max_per_split=max_per_split, token=load_settings().get("hf_token"),
                         progress=lambda m: st.update(m))
    con.print(f"[green]✓[/] {meta['name']}: {meta['counts']}  ({meta['num_labels']} labels)")


@data_app.command("upload")
def data_upload(file: Path, name: Optional[str] = None, text_col: Optional[str] = None,
                label_col: Optional[str] = None, type: str = "choice", instructions: str = ""):
    """Import a .jsonl / .json / .csv file (native System One rows or a table)."""
    from .data.local import import_rows, parse_bytes
    rows = parse_bytes(file.read_bytes(), file.name)
    meta = import_rows(name or file.stem, rows, text_col=text_col, label_col=label_col, qtype=type,
                       instructions=instructions, filename=file.name)
    con.print(f"[green]✓[/] {meta['name']}: {meta['counts']}")
    for w in meta.get("warnings", [])[:5]:
        con.print(f"[yellow]![/] {w}")


@data_app.command("faker")
def data_faker(name: str, spec: Optional[Path] = typer.Option(None, help="JSON spec with templates"),
               n: int = 50, locale: str = "en_US", example: bool = typer.Option(False, help="Use the bundled example")):
    """Generate data from templates filled by Faker (no API key)."""
    from .data import faker_gen
    if example or not spec:
        if not example:
            con.print("[dim]No --spec given: using the bundled support-routing example. Spec format:[/]")
            con.print_json(json.dumps(faker_gen.FAKER_EXAMPLE))
        sp = faker_gen.FAKER_EXAMPLE
    else:
        sp = json.loads(spec.read_text())
    meta = faker_gen.generate_dataset(name, sp, n, locale)
    con.print(f"[green]✓[/] {meta['name']}: {meta['counts']}")


@data_app.command("synth")
def data_synth(name: str, use_case: str = typer.Option(..., help="Plain-English use case"),
               options: Optional[str] = typer.Option(None, help="name:description;name:description (else suggested)"),
               instructions: Optional[str] = None, type: str = "choice", n: int = 40, language: str = "English",
               provider: Optional[str] = typer.Option(None, help="anthropic | openai (OpenAI-compatible)"),
               model: Optional[str] = None):
    """Generate synthetic data with an LLM for your use case."""
    from .data.synth import LLM, generate_dataset, suggest_schema
    if options:
        opts = dict(o.split(":", 1) if ":" in o else (o, o) for o in options.split(";"))
        q = {"type": type, "instructions": instructions or use_case, "options": opts}
    else:
        with con.status("Designing the question…"):
            q = suggest_schema(use_case, LLM(provider, model))
        con.print_json(json.dumps(q))
    with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                  console=con) as prog:
        task = prog.add_task("Generating", total=n * len(q["options"]))
        meta = generate_dataset(name, use_case, q, n, language, provider=provider, model=model,
                                progress=lambda d, t, m: prog.update(task, completed=d, description=m))
    con.print(f"[green]✓[/] {meta['name']}: {meta['counts']}")


# ---------------------------------------------------------------- train
@app.command()
def train(config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML/JSON config"),
          preset: Optional[str] = typer.Option(None, "--preset", "-p"),
          dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
          model: Optional[str] = typer.Option(None, "--model", "-m", help="Base model id"),
          method: Optional[str] = None, finetune: Optional[str] = None, epochs: Optional[float] = None,
          lr: Optional[float] = typer.Option(None, "--lr"), batch_size: Optional[int] = None,
          name: Optional[str] = None, set: list[str] = typer.Option([], "--set", help="key=value overrides"),
          detach: bool = typer.Option(False, "--detach", help="Start and return immediately")):
    """Fine-tune a decision model. Shows live progress; the run also appears in the UI."""
    from . import runs
    from .catalog import get_preset
    from .config import TrainConfig
    base: dict = {}
    if config:
        base = json.loads(TrainConfig.load(config).model_dump_json())
    if preset:
        p = get_preset(preset)
        if not p:
            raise typer.BadParameter(f"unknown preset {preset}")
        base.update(p["config"])
    over = {"base_model": model, "method": method, "finetune": finetune, "epochs": epochs, "learning_rate": lr,
            "batch_size": batch_size, "run_name": name}
    base.update({k: v for k, v in over.items() if v is not None})
    base.update(_parse_set(set))
    if dataset:
        base["data"] = {**base.get("data", {}), "dataset": dataset}
    if "data" not in base:
        raise typer.BadParameter("--dataset is required (or a config with data.dataset)")
    cfg = TrainConfig.model_validate(base)
    rid = runs.launch(cfg)
    con.print(f"Run [bold]{rid}[/]  ·  {cfg.base_model}  ·  {cfg.method}/{cfg.finetune}")
    if detach:
        con.print(f"Follow with [bold]snapcall logs {rid} -f[/] or open the UI.")
        return
    follow(rid)


def follow(rid: str):
    from . import runs
    from .tracking import RunLogger
    d = runs.run_path(rid)
    off = 0
    last_val: dict = {}
    log_pos = 0
    try:
        with Progress(TextColumn("[bold]{task.fields[stage]}[/]"), BarColumn(complete_style=ACCENT),
                      TextColumn("{task.completed}/{task.total} steps"), TextColumn("{task.fields[metric]}"),
                      TimeRemainingColumn(), console=con) as prog:
            task = prog.add_task("train", total=1, stage="queued", metric="")
            while True:
                st = runs.info(rid)["status"]
                rows, off = RunLogger.read_metrics(d, off)
                for r in rows:
                    if r["split"] == "val":
                        last_val = r
                logp = d / "train.log"
                if logp.exists():
                    txt = logp.read_text(errors="replace")
                    for line in txt[log_pos:].splitlines():
                        if line.startswith("[snapcall]"):
                            prog.console.print(f"[dim]{line}[/]")
                    log_pos = len(txt)
                metric = (f"acc {last_val.get('accuracy', 0):.3f} · ece {last_val.get('ece', 0):.3f}" if last_val else "")
                prog.update(task, total=max(1, st.get("total_steps") or 1), completed=st.get("step", 0),
                            stage=st.get("stage", ""), metric=metric)
                if st.get("state") in ("completed", "failed", "stopped"):
                    break
                time.sleep(1.0)
    except KeyboardInterrupt:
        if typer.confirm("\nStop training (best weights are kept)?", default=True):
            runs.stop(rid)
            con.print("Stop requested.")
        return
    st = runs.info(rid)
    if st["status"].get("state") == "completed":
        _print_summary(st)
    else:
        con.print(f"[red]{st['status'].get('state')}[/] {st['status'].get('error') or ''}\n{runs.logs(rid, 30)}")


def _print_summary(info: dict):
    ev = info.get("eval", {})
    t = Table(title=f"{info['id']}", box=box.SIMPLE_HEAD)
    for c in ("split", "accuracy", "macro F1", "ECE", "Brier", "NLL", "n"):
        t.add_column(c)
    for s in ("validation", "test"):
        m = ev.get(s)
        if m:
            t.add_row(s, f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", f"{m['ece']:.4f}", f"{m['brier']:.4f}",
                      f"{m['nll']:.4f}", str(m["n"]))
    con.print(t)
    con.print(f"temperature {ev.get('temperature', 1):.3f}  ·  next: [bold]snapcall predict {info['id']} --state '...'[/]"
              f"  ·  [bold]snapcall push {info['id']} --repo you/name[/]")


@app.command("runs")
def runs_cmd():
    """List training runs."""
    from . import runs
    t = Table(box=box.SIMPLE_HEAD)
    for c in ("run", "state", "model", "dataset", "progress", "val acc", "val ece"):
        t.add_column(c)
    color = {"completed": "green", "running": ACCENT, "failed": "red", "stopped": "yellow", "queued": "dim"}
    for r in runs.list_runs():
        st, cfg = r["status"], r["config"]
        fin = (st.get("summary") or {}).get("final", {}).get("validation", {})
        t.add_row(r["id"], f"[{color.get(st.get('state'), 'white')}]{st.get('state')}[/]", cfg.get("base_model", ""),
                  cfg.get("data", {}).get("dataset", ""), f"{100 * (st.get('progress') or 0):.0f}%",
                  f"{fin.get('accuracy', 0):.3f}" if fin else "", f"{fin.get('ece', 0):.3f}" if fin else "")
    con.print(t)


@app.command()
def status(run: str):
    """Show a run's status and results."""
    from . import runs
    info = runs.info(run)
    con.print_json(json.dumps(info["status"], default=str))
    if info.get("eval"):
        _print_summary(info)


@app.command()
def logs(run: str, follow_: bool = typer.Option(False, "--follow", "-f"), tail: int = 100):
    """Print training logs."""
    from . import runs
    if follow_:
        follow(run)
    else:
        con.print(runs.logs(run, tail))


@app.command()
def stop(run: str):
    """Gracefully stop a run (keeps the best weights)."""
    from . import runs
    runs.stop(run)
    con.print("Stop requested.")


@app.command("eval")
def eval_cmd(run: str, dataset: Optional[str] = None, split: str = "test", limit: int = 2000):
    """Evaluate a trained run on a dataset split."""
    from . import data as store
    from . import runs
    from .metrics import compute
    from .modeling import DecisionModel
    info = runs.info(run)
    exs = store.load_examples(dataset or info["config"]["data"]["dataset"], split, limit)
    with con.status("Loading model…"):
        m = DecisionModel.from_pretrained(runs.run_path(run) / "model")
    t0 = time.perf_counter()
    res = compute(exs, m.predict_proba(exs))
    res["ms_per_example"] = (time.perf_counter() - t0) * 1000 / max(1, len(exs))
    con.print_json(json.dumps(res))


@app.command()
def predict(model: str = typer.Argument(..., help="Run id, model folder, or HF repo"),
            state: str = typer.Option(..., "--state", "-s"),
            question: Optional[str] = typer.Option(None, help="Question JSON (System One shape)"),
            options: Optional[str] = typer.Option(None, help="Comma-separated options (choice)"),
            instructions: Optional[str] = None):
    """Run a decision with a trained model."""
    from .modeling import DecisionModel
    m = DecisionModel.from_pretrained(_model_path(model))
    if question:
        q = json.loads(question)
    elif options:
        q = {"type": "choice", "instructions": instructions or "", "criteria": {o.strip(): o.strip() for o in options.split(",")}}
    else:
        q = m.meta.get("question")
        if not q:
            raise typer.BadParameter("Pass --question or --options")
    res = m.decide(state, {"q": q})
    a = res["answers"]["q"]
    probs = a.get("probabilities") or {"no": 1 - a["noul"], "yes": a["noul"]}
    for k, v in sorted(probs.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * int(v * 30)
        con.print(f"{k:>24} [{ACCENT}]{bar}[/] {v:.3f}")
    con.print(f"[dim]{res['latency_ms']} ms[/]")


def _model_path(model: str) -> str:
    from . import runs
    try:
        p = runs.run_path(model) / "model"
        if p.exists():
            return str(p)
    except ValueError:
        pass
    return model


@app.command()
def card(run: str, repo: Optional[str] = None, out: Optional[Path] = None):
    """Generate the model card (README.md) for a run."""
    from . import runs
    from .hub_push import model_card
    md = model_card(runs.info(run), repo)
    if out:
        out.write_text(md)
        con.print(f"Wrote {out}")
    else:
        con.print(Markdown(md))


@app.command()
def push(run: str, repo: str = typer.Option(..., help="user/model-name"),
         public: bool = typer.Option(False, "--public")):
    """Push a trained model + auto-generated model card to the Hugging Face Hub."""
    from . import runs
    from .hub_push import push as do_push
    with con.status("Uploading…"):
        r = do_push(runs.info(run), runs.run_path(run), repo, private=not public)
    con.print(f"[green]✓[/] {r['url']}")


@app.command()
def serve(model: str = typer.Argument(..., help="Run id, model folder, or HF repo"), port: int = 8000,
          host: str = "127.0.0.1"):
    """Serve a model behind a System One compatible API: POST /v1/systemone."""
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel

    from .modeling import DecisionModel
    m = DecisionModel.from_pretrained(_model_path(model))
    api = FastAPI(title=f"Snapcall · {model}")

    class Req(BaseModel):
        state: object
        questions: dict
        model: Optional[str] = None

    @api.post("/v1/systemone")
    def decide(r: Req):
        return m.decide(r.state, r.questions)

    @api.get("/health")
    def health():
        return {"ok": True, "model": m.meta.get("name"), "base_model": m.meta["base_model"]}

    con.print(f"Serving [bold]{model}[/] at http://{host}:{port}/v1/systemone")
    uvicorn.run(api, host=host, port=port, log_level="warning")


@app.command()
def clean(run: Optional[str] = typer.Option(None, help="Only this run"),
          checkpoints: bool = typer.Option(True, help="Delete intermediate checkpoints"),
          tensorboard: bool = typer.Option(False, help="Delete TensorBoard event files"),
          failed: bool = typer.Option(False, help="Delete failed/stopped runs without a model"),
          hf_cache: list[str] = typer.Option([], help="HF cache repo ids to delete"),
          dry_run: bool = typer.Option(False, "--dry-run")):
    """Free disk space: checkpoints, event files, failed runs, Hugging Face cache."""
    from . import runs
    du = runs.disk_usage()
    con.print(f"Runs {_gb(du['runs_total'])} (checkpoints {_gb(du['checkpoints_total'])}) · datasets "
              f"{_gb(du['datasets'])} · HF cache {_gb(du['hf_cache_total'])}")
    if dry_run:
        for r in du["hf_cache"][:10]:
            con.print(f"  [dim]hf[/] {r['repo_id']:<50} {_gb(r['size'])}")
        return
    freed = runs.clean(run, checkpoints, tensorboard, failed)["freed_bytes"]
    if hf_cache:
        freed += runs.clean_hf_cache(hf_cache)["freed_bytes"]
    con.print(f"[green]✓[/] freed {_gb(freed)}")


@app.command("tensorboard")
def tensorboard_cmd(port: int = 6006):
    """Open TensorBoard over all runs."""
    import subprocess
    import sys

    from .paths import runs_dir
    subprocess.run([sys.executable, "-m", "tensorboard.main", "--logdir", str(runs_dir()), "--port", str(port)])


def main():
    app()


if __name__ == "__main__":
    main()
