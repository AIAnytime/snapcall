"""Model card generation and push to the Hugging Face Hub."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from . import __version__
from .catalog import get_model
from .paths import load_settings


def _fmt(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def model_card(run: dict[str, Any], repo_id: str | None = None) -> str:
    cfg = run.get("config", {})
    meta = run.get("model_meta", {})
    ev = run.get("eval", {})
    base_full = cfg.get("base_model", meta.get("base_model", ""))
    cat = get_model(base_full) or {}
    base = base_full.split(":")[0]  # Hub id without a variant suffix (e.g. laya:multilingual)
    q = meta.get("question") or {}
    repo = repo_id or f"your-username/{run['id']}"
    ds = cfg.get("data", {}).get("dataset", "")
    hub_ds = None
    try:
        from .data import get_meta
        hub_ds = get_meta(ds).get("source", {}).get("hf_id")
    except Exception:  # noqa: BLE001
        pass
    lic = cat.get("license", "apache-2.0")
    if lic in ("tbd", "lfm1.0"):
        lic = "other"
    metrics_rows = []
    for split in ("validation", "test"):
        m = ev.get(split)
        if m:
            metrics_rows.append(f"| {split} | {_fmt(m.get('accuracy'))} | {_fmt(m.get('macro_f1'))} | "
                                f"{_fmt(m.get('ece'))} | {_fmt(m.get('brier'))} | {_fmt(m.get('nll'))} | {m.get('n')} |")
    test = ev.get("test") or ev.get("validation") or {}
    model_index = ""
    if test:
        model_index = f"""model-index:
- name: {repo.split('/')[-1]}
  results:
  - task:
      type: text-classification
      name: Typed decision ({q.get('type', 'choice')})
    dataset:
      name: {ds}
      type: {hub_ds or ds}
    metrics:
    - type: accuracy
      value: {test.get('accuracy', 0):.4f}
    - type: f1
      name: macro F1
      value: {test.get('macro_f1', 0):.4f}
    - type: ece
      name: expected calibration error
      value: {test.get('ece', 0):.4f}
"""
    crit = q.get("criteria")
    if isinstance(crit, dict):
        opts = "\n".join(f"| `{k}` | {v} |" for k, v in list(crit.items())[:80])
        opts_table = f"| Option | Description |\n|---|---|\n{opts}"
        crit_example = json.dumps(dict(list(crit.items())[:3]), ensure_ascii=False)
    elif isinstance(crit, list):
        opts_table = "| Level | Description |\n|---|---|\n" + "\n".join(f"| {i} | {d} |" for i, d in enumerate(crit))
        crit_example = json.dumps(crit, ensure_ascii=False)
    else:
        opts_table = "Yes / no (`noul`), returns P(yes)."
        crit_example = None
    q_json = {"type": q.get("type", "choice"), "instructions": q.get("instructions", "")}
    if crit_example:
        q_json["criteria"] = json.loads(crit_example)
    st = run.get("status", {})
    summ = st.get("summary", {})
    mi = st.get("model_info", {})
    hp_keys = ["method", "finetune", "epochs", "learning_rate", "batch_size", "grad_accum", "max_length",
               "scheduler", "warmup_ratio", "weight_decay", "lora_r", "lora_alpha", "rlcd_epochs", "rlcd_beta",
               "rlcd_brier_weight", "rlcd_consistency_weight", "seed"]
    hp = "\n".join(f"| `{k}` | {cfg.get(k)} |" for k in hp_keys if k in cfg and
                   not (k.startswith("lora") and cfg.get("finetune") not in ("lora", "qlora")) and
                   not (k.startswith("rlcd") and "rlcd" not in cfg.get("method", "")))
    return f"""---
license: {lic}
base_model: {base}
library_name: snapcall
pipeline_tag: text-classification
tags:
- snapcall
- decision-model
- system-one
- calibrated
- {cfg.get('method', 'sft')}
{f"datasets:{chr(10)}- {hub_ds}{chr(10)}" if hub_ds else ""}{model_index}---

# {repo.split('/')[-1]}

A **System One decision model** fine-tuned from [`{base}`](https://huggingface.co/{base}) with
[Snapcall](https://github.com/AIAnytime/snapcall). Given a *state* (text or JSON) and a typed
question, it returns a calibrated probability for every option in one forward pass. It cannot return an
option you did not offer.

## Decision task

**Question type:** `{q.get('type', 'choice')}`
**Instructions:** {q.get('instructions') or '_(per-example)_'}

{opts_table}

## Evaluation

| Split | Accuracy | Macro F1 | ECE ↓ | Brier ↓ | NLL ↓ | N |
|---|---|---|---|---|---|---|
{chr(10).join(metrics_rows) or '| - | - | - | - | - | - | - |'}

Post-hoc temperature: **{ev.get('temperature', meta.get('temperature', 1.0)):.3f}**. ECE is the 15-bin expected
calibration error: when the model says 0.8 it should be right about 80% of the time.

## Usage

```bash
pip install snapcall
```

```python
from snapcall import DecisionModel

model = DecisionModel.from_pretrained("{repo}")
result = model.decide(
    state="<your input>",
    questions={{"q1": {json.dumps(q_json, ensure_ascii=False)}}},
)
print(result["answers"]["q1"])
```

Serve a System One compatible API:

```bash
snapcall serve {repo} --port 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' \\
  -d '{{"state": "...", "questions": {{"q1": {json.dumps(q_json, ensure_ascii=False)}}}}}'
```

## Training

| | |
|---|---|
| Base model | `{base_full}` ({mi.get('arch', meta.get('arch', ''))}, {meta.get('pooling', '')} pooling) |
| Parameters | {mi.get('params', 0) / 1e6:.1f}M total, {mi.get('trainable_params', 0) / 1e6:.2f}M trainable |
| Train / val / test | {summ.get('train_examples', '?')} / {summ.get('val_examples', '?')} / {summ.get('test_examples', '?')} |
| Training time | {summ.get('elapsed_s', 0) / 60:.1f} min on `{mi.get('device', '?')}` |

{hp}

**Method.** Supervised fine-tuning of a typed decision head (option-span pooling) with option-order
shuffling, followed by **RLCD** (Reinforcement Learning for Calibrated Decisions): a KL-anchored
proper-scoring objective (Brier + log-loss) against the SFT policy plus an option-order consistency term,
then temperature scaling on the validation split. This is Snapcall's open implementation; TypeSafe has
only described its RLCD at a high level.

## Limitations

- Trained on `{ds}`; confidence is calibrated in-distribution, so re-check ECE on your own traffic.
- Text states only (no images/boxes).
- The model judges with the options you give it; vague option descriptions produce vague decisions.

---

Trained with **Snapcall v{__version__}**, the open-source toolkit for building better decision models, built by [AI Anytime](https://www.youtube.com/@AIAnytime).
"""


def push(run: dict[str, Any], run_dir: Path, repo_id: str, private: bool = True, token: str | None = None,
         card: str | None = None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    token = token or load_settings().get("hf_token")
    if not token:
        raise RuntimeError("No Hugging Face token. Add one in Settings or set HF_TOKEN.")
    api = HfApi(token=token)
    model_dir = run_dir / "model"
    stage = run_dir / "_push"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(model_dir, stage)
    (stage / "README.md").write_text(card or model_card(run, repo_id))
    if (run_dir / "eval.json").exists():
        shutil.copy(run_dir / "eval.json", stage / "eval_results.json")
    if (run_dir / "config.json").exists():
        shutil.copy(run_dir / "config.json", stage / "training_config.json")
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    info = api.upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="model",
                             commit_message=f"Upload {run['id']} via Snapcall")
    shutil.rmtree(stage, ignore_errors=True)
    result = {"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}", "private": private,
              "commit": getattr(info, "oid", None), "time": time.time()}
    (run_dir / "pushed.json").write_text(json.dumps(result, indent=2))
    return result
