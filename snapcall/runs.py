"""Run management: create, launch (subprocess), list, stop, delete, clean up."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import TrainConfig
from .data import slugify
from .paths import datasets_dir, snapcall_home, runs_dir
from .tracking import RunLogger


def new_run_id(cfg: TrainConfig) -> str:
    base = cfg.run_name or cfg.base_model.split("/")[-1]
    return f"{slugify(base)}-{time.strftime('%Y%m%d-%H%M%S')}"


def run_path(run_id: str) -> Path:
    p = (runs_dir() / run_id).resolve()
    if p.parent != runs_dir().resolve():
        raise ValueError("invalid run id")
    return p


def launch(cfg: TrainConfig, run_id: str | None = None) -> str:
    run_id = run_id or new_run_id(cfg)
    d = run_path(run_id)
    d.mkdir(parents=True, exist_ok=True)
    cfg.save(d / "config.json")
    RunLogger(d, tensorboard=False).status(state="queued", stage="queued", created=time.time(), progress=0.0)
    log = open(d / "train.log", "a")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    # Apple Silicon: by default MPS may allocate well past physical RAM and thrash swap instead of
    # raising OOM. Cap it so the trainer's automatic micro-batch backoff can kick in.
    if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in env:
        env.update(PYTORCH_MPS_HIGH_WATERMARK_RATIO="1.0", PYTORCH_MPS_LOW_WATERMARK_RATIO="0.8")
    proc = subprocess.Popen([sys.executable, "-m", "snapcall.worker", str(d)], stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    (d / "pid").write_text(str(proc.pid))
    return run_id


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def info(run_id: str) -> dict[str, Any]:
    d = run_path(run_id)
    if not d.exists():
        raise FileNotFoundError(run_id)
    status = RunLogger.read_status(d) or {}
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    # a crashed worker leaves status=running; detect it
    if status.get("state") in ("running", "queued") and (d / "pid").exists():
        pid = int((d / "pid").read_text() or 0)
        if pid and not _alive(pid) and time.time() - status.get("updated", status.get("created", 0)) > 5:
            status["state"] = "failed"
            status.setdefault("error", "Worker process exited unexpectedly. See logs.")
    model_dir = d / "model"
    out = {"id": run_id, "status": status, "config": cfg, "has_model": (model_dir / "snapcall_config.json").exists(),
           "created": status.get("created") or d.stat().st_ctime}
    if out["has_model"]:
        out["model_meta"] = json.loads((model_dir / "snapcall_config.json").read_text())
    ev = d / "eval.json"
    if ev.exists():
        out["eval"] = json.loads(ev.read_text())
    if (d / "pushed.json").exists():
        out["pushed"] = json.loads((d / "pushed.json").read_text())
    return out


def list_runs() -> list[dict[str, Any]]:
    out = []
    for d in sorted(runs_dir().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "config.json").exists():
            try:
                i = info(d.name)
                i.pop("eval", None)
                out.append(i)
            except Exception:  # noqa: BLE001
                continue
    return out


def latest_completed() -> str | None:
    for r in list_runs():
        if r["has_model"]:
            return r["id"]
    return None


def stop(run_id: str) -> None:
    d = run_path(run_id)
    (d / "STOP").write_text("1")  # graceful: trainer saves best weights and exits


def kill(run_id: str) -> None:
    d = run_path(run_id)
    if (d / "pid").exists():
        try:
            os.killpg(int((d / "pid").read_text()), signal.SIGTERM)
        except (OSError, ValueError):
            pass


def logs(run_id: str, tail: int = 400) -> str:
    p = run_path(run_id) / "train.log"
    if not p.exists():
        return ""
    lines = p.read_text(errors="replace").splitlines()
    return "\n".join(lines[-tail:])


def delete(run_id: str) -> None:
    kill(run_id)
    shutil.rmtree(run_path(run_id), ignore_errors=True)


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def disk_usage() -> dict[str, Any]:
    runs = []
    for d in runs_dir().iterdir():
        if d.is_dir():
            runs.append({"id": d.name, "total": dir_size(d), "checkpoints": dir_size(d / "checkpoints"),
                         "model": dir_size(d / "model"), "tensorboard": dir_size(d / "tensorboard"),
                         "state": (RunLogger.read_status(d) or {}).get("state")})
    runs.sort(key=lambda r: r["total"], reverse=True)
    hf_cache = []
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in sorted(cache.repos, key=lambda r: r.size_on_disk, reverse=True):
            hf_cache.append({"repo_id": repo.repo_id, "type": repo.repo_type, "size": repo.size_on_disk,
                             "last_accessed": repo.last_accessed})
    except Exception:  # noqa: BLE001
        pass
    return {"home": str(snapcall_home()), "runs": runs, "datasets": dir_size(datasets_dir()),
            "runs_total": sum(r["total"] for r in runs), "checkpoints_total": sum(r["checkpoints"] for r in runs),
            "hf_cache": hf_cache, "hf_cache_total": sum(r["size"] for r in hf_cache)}


def clean(run_id: str | None = None, checkpoints: bool = True, tensorboard: bool = False,
          failed_runs: bool = False) -> dict[str, int]:
    freed = 0
    targets = [run_path(run_id)] if run_id else [d for d in runs_dir().iterdir() if d.is_dir()]
    for d in targets:
        st = (RunLogger.read_status(d) or {}).get("state")
        if st in ("running", "queued"):
            continue
        if failed_runs and st in ("failed", "stopped") and not (d / "model" / "snapcall_config.json").exists():
            freed += dir_size(d)
            shutil.rmtree(d, ignore_errors=True)
            continue
        if checkpoints and (d / "checkpoints").exists():
            freed += dir_size(d / "checkpoints")
            shutil.rmtree(d / "checkpoints", ignore_errors=True)
        if tensorboard and (d / "tensorboard").exists():
            freed += dir_size(d / "tensorboard")
            shutil.rmtree(d / "tensorboard", ignore_errors=True)
    return {"freed_bytes": freed}


def clean_hf_cache(repo_ids: list[str]) -> dict[str, int]:
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()
    hashes = [rev.commit_hash for repo in cache.repos if repo.repo_id in repo_ids for rev in repo.revisions]
    if not hashes:
        return {"freed_bytes": 0}
    strategy = cache.delete_revisions(*hashes)
    freed = strategy.expected_freed_size
    strategy.execute()
    return {"freed_bytes": freed}
