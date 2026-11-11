"""FastAPI app: REST API for the dashboard + System One compatible inference."""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from .. import __version__, catalog, compute, runs
from .. import data as store
from ..config import HPARAMS, TrainConfig
from ..paths import load_settings, masked_settings, runs_dir, save_settings
from ..tracking import RunLogger

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Snapcall", version=__version__)

# Optional password (required when the dashboard is reachable from the internet, e.g. a RunPod proxy URL).
# Browsers get HTTP Basic auth (any username); API clients can send `Authorization: Bearer <password>`.
PASSWORD = os.environ.get("SNAPCALL_PASSWORD") or None


@app.middleware("http")
async def require_password(request: Request, call_next):
    if PASSWORD and request.url.path != "/health":
        h = request.headers.get("authorization", "")
        ok = False
        if h.startswith("Bearer "):
            ok = hmac.compare_digest(h[7:].encode(), PASSWORD.encode())
        elif h.startswith("Basic "):
            try:
                _, _, pw = base64.b64decode(h[6:]).decode().partition(":")
                ok = hmac.compare_digest(pw.encode(), PASSWORD.encode())
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            return Response("Authentication required", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Snapcall"'})
    return await call_next(request)


@app.get("/health")
def health():
    return {"ok": True, "version": __version__}


app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

# ----------------------------------------------------------------- jobs
JOBS: dict[str, dict[str, Any]] = {}


def start_job(kind: str, fn, *args, **kwargs) -> str:
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "kind": kind, "state": "running", "progress": 0.0, "message": "Starting…",
                 "result": None, "error": None, "started": time.time()}

    def runner():
        try:
            JOBS[jid]["result"] = fn(jid, *args, **kwargs)
            JOBS[jid].update(state="completed", progress=1.0, message="Done")
        except Exception as e:  # noqa: BLE001
            JOBS[jid].update(state="failed", error=f"{type(e).__name__}: {e}")
    threading.Thread(target=runner, daemon=True).start()
    return jid


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "job not found")
    return JOBS[jid]


# ----------------------------------------------------------------- system
_SYS_CACHE: dict[str, Any] = {}


@app.on_event("startup")
def _warm():  # torch import + hardware probe take a few seconds; do it before the first page load
    threading.Thread(target=system, daemon=True).start()


@app.get("/api/system")
def system(refresh: bool = False):
    if refresh or "info" not in _SYS_CACHE:
        info = compute.detect()
        _SYS_CACHE["info"] = info
        _SYS_CACHE["rec"] = compute.recommend(info)
    return {"info": _SYS_CACHE["info"], "recommendations": _SYS_CACHE["rec"], "version": __version__}


@app.get("/api/catalog")
def get_catalog():
    return {"models": catalog.MODELS, "presets": catalog.PRESETS, "reference": catalog.REFERENCE_MODELS,
            "hf_datasets": __import__("snapcall.data.hub", fromlist=["HF_PRESETS"]).HF_PRESETS}


@app.get("/api/hparams")
def hparams():
    defaults = TrainConfig(data={"dataset": "x"}).model_dump()
    defaults.pop("data")
    return {"explainer": HPARAMS, "defaults": defaults}


class MemReq(BaseModel):
    params_b: float
    finetune: str
    batch_size: int = 8
    max_length: int = 512
    gradient_checkpointing: bool = False


@app.post("/api/estimate")
def estimate(r: MemReq):
    info = system()["info"]
    req = compute.estimate_memory_gb(r.params_b, r.finetune, r.batch_size, r.max_length, info.get("bf16", False),
                                     r.gradient_checkpointing)
    budget = system()["recommendations"]["memory_budget_gb"]
    out = {"memory_gb": req, "budget_gb": budget, "fit": compute.fit(req, budget), "micro_batch": r.batch_size}
    if out["fit"] == "no":
        # The trainer splits each batch into micro-batches that fit (same effective batch), so find that size.
        mb = r.batch_size
        while mb > 1:
            mb //= 2
            m = compute.estimate_memory_gb(r.params_b, r.finetune, mb, r.max_length, info.get("bf16", False),
                                           r.gradient_checkpointing)
            if m <= budget * 0.85:
                out.update(memory_gb=m, fit="auto", micro_batch=mb)
                break
    return out


# ----------------------------------------------------------------- settings
@app.get("/api/settings")
def get_settings():
    from ..paths import snapcall_home
    return {**masked_settings(), "snapcall_home": str(snapcall_home())}


@app.post("/api/settings")
def post_settings(body: dict[str, Any]):
    save_settings({k: v for k, v in body.items() if k != "snapcall_home"})
    return get_settings()


@app.post("/api/settings/test-hf")
def test_hf():
    from huggingface_hub import HfApi
    tok = load_settings().get("hf_token")
    if not tok:
        raise HTTPException(400, "No token set")
    try:
        who = HfApi(token=tok).whoami()
        return {"ok": True, "name": who.get("name"), "orgs": [o.get("name") for o in who.get("orgs", [])]}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Token rejected: {e}")


# ----------------------------------------------------------------- datasets
@app.get("/api/datasets")
def datasets():
    return store.list_datasets()


@app.get("/api/datasets/{name}")
def dataset(name: str, split: str = "train", n: int = 25):
    try:
        return {"meta": store.get_meta(name), "rows": store.preview(name, split, n)}
    except FileNotFoundError:
        raise HTTPException(404, "dataset not found")


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    store.delete_dataset(name)
    return {"ok": True}


@app.get("/api/hf/peek")
def hf_peek(hf_id: str, config: Optional[str] = None, split: Optional[str] = None):
    from ..data.hub import peek
    try:
        return peek(hf_id, config, split, load_settings().get("hf_token"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read {hf_id}: {e}")


class HFImport(BaseModel):
    hf_id: str
    name: str
    config: Optional[str] = None
    text_col: str | list[str]
    label_col: str
    type: str = "choice"
    instructions: str = ""
    instructions_col: Optional[str] = None
    max_per_split: Optional[int] = 5000
    top_labels: Optional[int] = None
    other_label: Optional[str] = None
    multi_label: str = "single"
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@app.post("/api/datasets/hf")
def import_hf(r: HFImport):
    from ..data.hub import import_hf as do_import

    def job(jid):
        return do_import(r.hf_id, name=r.name, text_col=r.text_col, label_col=r.label_col, qtype=r.type,
                         instructions=r.instructions, instructions_col=r.instructions_col, config=r.config,
                         max_per_split=r.max_per_split, val_ratio=r.val_ratio, test_ratio=r.test_ratio,
                         top_labels=r.top_labels or None, other_label=r.other_label or None,
                         multi_label=r.multi_label,
                         token=load_settings().get("hf_token"),
                         progress=lambda m: JOBS[jid].update(message=m))
    return {"job": start_job("hf-import", job)}


@app.post("/api/datasets/upload/inspect")
async def upload_inspect(file: UploadFile = File(...)):
    from ..data.local import is_native, parse_bytes
    try:
        rows = parse_bytes(await file.read(), file.filename or "")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e))
    cols = list(rows[0].keys()) if rows else []
    return {"rows": rows[:15], "count": len(rows), "columns": cols, "native": is_native(rows)}


@app.post("/api/datasets/upload")
async def upload(file: UploadFile = File(...), name: str = Form(...), text_col: str = Form(""),
                 label_col: str = Form(""), type: str = Form("choice"), instructions: str = Form(""),
                 val_ratio: float = Form(0.1), test_ratio: float = Form(0.1), top_labels: int = Form(0),
                 multi_label: str = Form("single")):
    from ..data.local import import_rows, parse_bytes
    try:
        rows = parse_bytes(await file.read(), file.filename or "")
        return import_rows(name, rows, text_col=text_col or None, label_col=label_col or None, qtype=type,
                           instructions=instructions, val=val_ratio, test=test_ratio, filename=file.filename or "",
                           top_labels=top_labels or None, multi_label=multi_label)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e))


class FakerReq(BaseModel):
    name: str
    spec: dict[str, Any]
    n_per_option: int = 50
    locale: str = "en_US"
    preview: bool = False


@app.get("/api/faker/example")
def faker_example():
    from ..data.faker_gen import FAKER_EXAMPLE
    return FAKER_EXAMPLE


@app.post("/api/datasets/faker")
def faker(r: FakerReq):
    from ..data import faker_gen
    try:
        if r.preview:
            return {"rows": faker_gen.generate(r.spec, min(r.n_per_option, 4), r.locale)}
        return faker_gen.generate_dataset(r.name, r.spec, r.n_per_option, r.locale)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e))


class SuggestReq(BaseModel):
    use_case: str
    provider: Optional[str] = None
    model: Optional[str] = None


@app.post("/api/synth/suggest")
def synth_suggest(r: SuggestReq):
    from ..data.synth import LLM, suggest_schema
    try:
        return suggest_schema(r.use_case, LLM(r.provider, r.model))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e))


class SynthReq(BaseModel):
    name: str
    use_case: str
    question: dict[str, Any]
    n_per_option: int = 40
    language: str = "English"
    style: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None


@app.post("/api/datasets/synth")
def synth(r: SynthReq):
    from ..data.synth import generate_dataset

    def job(jid):
        def prog(done, total, msg):
            JOBS[jid].update(progress=done / max(1, total), message=msg)
        return generate_dataset(r.name, r.use_case, r.question, r.n_per_option, r.language, r.style,
                                r.provider, r.model, progress=prog)
    return {"job": start_job("synth", job)}


# ----------------------------------------------------------------- runs
@app.get("/api/runs")
def list_runs():
    return runs.list_runs()


@app.post("/api/runs")
def create_run(cfg: TrainConfig):
    if not (store.resolve(cfg.data.dataset) / "train.jsonl").exists():
        raise HTTPException(400, f"Dataset '{cfg.data.dataset}' not found")
    return {"id": runs.launch(cfg)}


@app.get("/api/runs/{rid}")
def get_run(rid: str):
    try:
        return runs.info(rid)
    except FileNotFoundError:
        raise HTTPException(404, "run not found")


@app.get("/api/runs/{rid}/metrics")
def run_metrics(rid: str, after: int = 0):
    rows, off = RunLogger.read_metrics(runs.run_path(rid), after)
    return {"rows": rows, "offset": off}


@app.get("/api/runs/{rid}/logs")
def run_logs(rid: str, tail: int = 500):
    return {"text": runs.logs(rid, tail)}


@app.get("/api/runs/{rid}/stream")
async def run_stream(rid: str, after: int = 0):
    d = runs.run_path(rid)

    async def gen():
        off, log_len = after, 0
        while True:
            rows, off = RunLogger.read_metrics(d, off)
            st = runs.info(rid)["status"]
            log_text = (d / "train.log").read_text(errors="replace") if (d / "train.log").exists() else ""
            new_log = log_text[log_len:]
            log_len = len(log_text)
            yield f"data: {json.dumps({'status': st, 'metrics': rows, 'offset': off, 'log': new_log})}\n\n"
            if st.get("state") in ("completed", "failed", "stopped"):
                break
            await asyncio.sleep(1.0)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/runs/{rid}/stop")
def stop_run(rid: str):
    runs.stop(rid)
    return {"ok": True}


@app.delete("/api/runs/{rid}")
def delete_run(rid: str):
    MODELS.pop(rid, None)
    runs.delete(rid)
    return {"ok": True}


@app.get("/api/runs/{rid}/files")
def run_files(rid: str):
    d = runs.run_path(rid)
    out = []
    for f in sorted(d.rglob("*")):
        if f.is_file() and "tensorboard" not in f.parts:
            out.append({"path": str(f.relative_to(d)), "size": f.stat().st_size})
    return out


# -- inference ---------------------------------------------------------------
MODELS: "OrderedDict[str, Any]" = OrderedDict()
_LOAD_LOCK = threading.Lock()


def load_model(rid: str):
    from ..modeling import DecisionModel
    with _LOAD_LOCK:
        if rid in MODELS:
            MODELS.move_to_end(rid)
            return MODELS[rid]
        path = runs.run_path(rid) / "model"
        if not (path / "snapcall_config.json").exists():
            raise HTTPException(400, "This run has no saved model yet")
        m = DecisionModel.from_pretrained(path)
        MODELS[rid] = m
        while len(MODELS) > 2:
            MODELS.popitem(last=False)
        return m


class DecideReq(BaseModel):
    state: Any
    questions: dict[str, dict[str, Any]]
    model: Optional[str] = None


@app.post("/api/runs/{rid}/predict")
def predict(rid: str, r: DecideReq):
    try:
        return load_model(rid).decide(r.state, r.questions)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e))


@app.post("/v1/systemone")
def systemone(r: DecideReq):
    rid = r.model if r.model and r.model not in ("latest", "jev-latest") else (DEFAULT_MODEL or runs.latest_completed())
    if not rid:
        raise HTTPException(404, "No trained model available")
    return predict(rid, r)


DEFAULT_MODEL: Optional[str] = None


class EvalReq(BaseModel):
    dataset: Optional[str] = None
    split: str = "test"
    limit: Optional[int] = 2000


@app.post("/api/runs/{rid}/evaluate")
def evaluate_run(rid: str, r: EvalReq):
    from ..metrics import compute as compute_metrics
    info = runs.info(rid)
    ds = r.dataset or info["config"]["data"]["dataset"]
    exs = store.load_examples(ds, r.split, r.limit)
    if not exs:
        raise HTTPException(400, f"No rows in {ds}/{r.split}")
    m = load_model(rid)
    t0 = time.perf_counter()
    probs = m.predict_proba(exs)
    dt = time.perf_counter() - t0
    res = compute_metrics(exs, probs, with_details=True)
    res["latency_ms_per_example"] = dt * 1000 / len(exs)
    mistakes = []
    for ex, p in zip(exs, probs):
        pred = max(range(len(p)), key=p.__getitem__)
        if pred != ex.label and len(mistakes) < 25:
            mistakes.append({"state": ex.state[:400], "label": ex.question.names[ex.label],
                             "pred": ex.question.names[pred], "confidence": p[pred]})
    res["mistakes"] = mistakes
    res["dataset"], res["split"] = ds, r.split
    return res


# -- publish -----------------------------------------------------------------
@app.get("/api/runs/{rid}/card")
def card(rid: str, repo_id: Optional[str] = None):
    from ..hub_push import model_card
    return {"markdown": model_card(runs.info(rid), repo_id)}


class PushReq(BaseModel):
    repo_id: str
    private: bool = True
    card: Optional[str] = None


@app.post("/api/runs/{rid}/push")
def push(rid: str, r: PushReq):
    from ..hub_push import push as do_push

    def job(jid):
        JOBS[jid]["message"] = "Uploading to the Hub…"
        return do_push(runs.info(rid), runs.run_path(rid), r.repo_id, r.private, card=r.card)
    return {"job": start_job("push", job)}


# -- cleanup -----------------------------------------------------------------
@app.get("/api/cleanup")
def cleanup_info():
    return runs.disk_usage()


class CleanReq(BaseModel):
    run_id: Optional[str] = None
    checkpoints: bool = True
    tensorboard: bool = False
    failed_runs: bool = False
    hf_repos: list[str] = []


@app.post("/api/cleanup")
def cleanup(r: CleanReq):
    freed = runs.clean(r.run_id, r.checkpoints, r.tensorboard, r.failed_runs)["freed_bytes"]
    if r.hf_repos:
        freed += runs.clean_hf_cache(r.hf_repos)["freed_bytes"]
    MODELS.clear()
    return {"freed_bytes": freed}


# -- tensorboard ---------------------------------------------------------------
TB: dict[str, Any] = {}


@app.post("/api/tensorboard")
def tensorboard(port: int = 6006):
    p = TB.get("proc")
    if p is None or p.poll() is not None:
        TB["proc"] = subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", str(runs_dir()),
                                       "--port", str(port), "--bind_all"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        TB["port"] = port
        time.sleep(2.5)
    pod = os.environ.get("RUNPOD_POD_ID")
    if pod:  # RunPod HTTP proxy (port must be exposed on the pod)
        return {"url": f"https://{pod}-{TB['port']}.proxy.runpod.net"}
    return {"url": f"http://localhost:{TB['port']}"}


# -- static UI -------------------------------------------------------------------
@app.get("/{full_path:path}")
def spa(full_path: str):
    if full_path.startswith(("api/", "v1/")):
        raise HTTPException(404)
    f = STATIC / full_path
    if full_path and f.is_file():
        return FileResponse(f)
    index = STATIC / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"message": "Snapcall API is running. UI not built: cd ui && pnpm build"})
