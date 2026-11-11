"""Dataset store. A prepared dataset is a folder of System One JSONL files:

    SNAPCALL_HOME/datasets/<name>/{train,validation,test}.jsonl + meta.json

Each line: {"state": ..., "question": {"type", "instructions", "criteria"}, "answer": ..., "soft"?: {...}}
"""

from __future__ import annotations

import json
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..paths import datasets_dir
from ..schema import Example, flatten_system_one

SPLITS = ("train", "validation", "test")


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", s).strip("-").lower()[:64] or f"dataset-{int(time.time())}"


def resolve(name_or_path: str) -> Path:
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    return datasets_dir() / name_or_path


def validate_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    ok, errors = [], []
    for i, r in enumerate(records):
        for flat in flatten_system_one(r):
            try:
                Example.from_record(flat)
                ok.append(flat)
            except Exception as e:  # noqa: BLE001
                if len(errors) < 20:
                    errors.append(f"row {i}: {e}")
    return ok, errors


def split_records(records: list[dict[str, Any]], val: float = 0.1, test: float = 0.1,
                  seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Stratified split by answer so rare options appear in every split."""
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(json.dumps(r.get("answer")), []).append(r)
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for g in groups.values():
        rng.shuffle(g)
        n = len(g)
        nt = int(round(n * test)) if n >= 3 else 0
        nv = int(round(n * val)) if n >= 3 else 0
        out["test"] += g[:nt]
        out["validation"] += g[nt:nt + nv]
        out["train"] += g[nt + nv:]
    for s in SPLITS:
        rng.shuffle(out[s])
    return out


def summarize(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    first = next((r for s in SPLITS for r in splits.get(s, [])), None)
    labels = Counter(json.dumps(r["answer"]).strip('"') for r in splits.get("train", []))
    qtypes = Counter(r["question"].get("type", "choice") for s in SPLITS for r in splits.get(s, []))
    lens = sorted(len(str(r["state"])) for r in splits.get("train", [])[:5000])
    return {
        "counts": {s: len(splits.get(s, [])) for s in SPLITS},
        "question_types": dict(qtypes),
        "label_counts": dict(labels.most_common(300)),
        "num_labels": len(labels),
        "state_chars_p50": lens[len(lens) // 2] if lens else 0,
        "state_chars_p95": lens[int(len(lens) * 0.95)] if lens else 0,
        "example_question": first["question"] if first else None,
    }


def save_dataset(name: str, splits: dict[str, list[dict[str, Any]]], source: dict[str, Any]) -> dict[str, Any]:
    name = slugify(name)
    d = datasets_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        with open(d / f"{s}.jsonl", "w") as f:
            for r in splits.get(s, []):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"name": name, "created": time.time(), "source": source, **summarize(splits)}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.extend(flatten_system_one(json.loads(line)))
                if limit and len(out) >= limit:
                    break
    return out


def load_examples(name_or_path: str, split: str, limit: int | None = None) -> list[Example]:
    return [Example.from_record(r) for r in read_jsonl(resolve(name_or_path) / f"{split}.jsonl", limit)]


def list_datasets() -> list[dict[str, Any]]:
    out = []
    for d in sorted(datasets_dir().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        m = d / "meta.json"
        if m.exists():
            meta = json.loads(m.read_text())
            meta["size_mb"] = round(sum(f.stat().st_size for f in d.glob("*")) / 1e6, 2)
            out.append(meta)
    return out


def get_meta(name: str) -> dict[str, Any]:
    d = resolve(name)
    meta = json.loads((d / "meta.json").read_text())
    meta["size_mb"] = round(sum(f.stat().st_size for f in d.glob("*")) / 1e6, 2)
    return meta


def preview(name: str, split: str = "train", n: int = 20) -> list[dict[str, Any]]:
    return read_jsonl(resolve(name) / f"{split}.jsonl", n)


def delete_dataset(name: str) -> None:
    d = datasets_dir() / slugify(name)
    if d.exists() and d.parent == datasets_dir():
        shutil.rmtree(d)
