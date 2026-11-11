"""Import Hugging Face datasets into System One format."""

from __future__ import annotations

import random
from typing import Any, Callable

import httpx

from . import save_dataset, split_records

DS_SERVER = "https://datasets-server.huggingface.co"

HF_PRESETS: list[dict[str, Any]] = [
    {"id": "banking77", "hf_id": "PolyAI/banking77", "config": None, "title": "Banking77",
     "description": "77 fine-grained banking intents. The classic Clef vs Jev benchmark.",
     "type": "choice", "text_col": "text", "label_col": "label", "langs": "en",
     "instructions": "Which banking intent does this customer message express?"},
    {"id": "emotion", "hf_id": "dair-ai/emotion", "config": "split", "title": "Emotion",
     "description": "Six emotions in short English posts.", "type": "choice", "text_col": "text",
     "label_col": "label", "langs": "en", "instructions": "Which emotion does the author express?"},
    {"id": "ag_news", "hf_id": "fancyzhx/ag_news", "config": None, "title": "AG News",
     "description": "News topic routing: World, Sports, Business, Sci/Tech.", "type": "choice",
     "text_col": "text", "label_col": "label", "langs": "en", "instructions": "What is the topic of this article?"},
    {"id": "sst2", "hf_id": "stanfordnlp/sst2", "config": None, "title": "SST-2",
     "description": "Binary sentiment on movie reviews.", "type": "choice", "text_col": "sentence",
     "label_col": "label", "langs": "en", "instructions": "Is the sentiment of this review positive or negative?"},
    {"id": "boolq", "hf_id": "google/boolq", "config": None, "title": "BoolQ",
     "description": "Yes/no questions over Wikipedia passages (noul type).", "type": "noul",
     "text_col": "passage", "label_col": "answer", "instructions_col": "question", "langs": "en",
     "instructions": ""},
    {"id": "yelp_full", "hf_id": "Yelp/yelp_review_full", "config": "yelp_review_full", "title": "Yelp 1-5 stars",
     "description": "Star rating as an ordinal score question.", "type": "score", "text_col": "text",
     "label_col": "label", "langs": "en", "instructions": "How satisfied is this customer?"},
    {"id": "massive_intent", "hf_id": "SetFit/amazon_massive_intent_en-US", "config": None,
     "title": "MASSIVE intents (en)", "description": "60 voice-assistant intents.", "type": "choice",
     "text_col": "text", "label_col": "label_text", "langs": "en",
     "instructions": "Which assistant intent does this utterance express?"},
    {"id": "multilingual_sentiment", "hf_id": "tyqiangz/multilingual-sentiments", "config": "all",
     "title": "Multilingual sentiment", "description": "Positive / neutral / negative across 12 languages.",
     "type": "choice", "text_col": "text", "label_col": "label", "langs": "multilingual",
     "instructions": "What is the sentiment of this text?"},
]


def humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip()


def peek(hf_id: str, config: str | None = None, split: str | None = None, token: str | None = None) -> dict[str, Any]:
    """Columns, splits and first rows via the datasets-server API (no download)."""
    h = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(timeout=30, headers=h) as c:
        sp = c.get(f"{DS_SERVER}/splits", params={"dataset": hf_id})
        sp.raise_for_status()
        splits = sp.json().get("splits", [])
        configs = sorted({s["config"] for s in splits})
        config = config or (configs[0] if configs else "default")
        split_names = [s["split"] for s in splits if s["config"] == config]
        split = split or (split_names[0] if split_names else "train")
        r = c.get(f"{DS_SERVER}/first-rows", params={"dataset": hf_id, "config": config, "split": split})
        r.raise_for_status()
        j = r.json()
    features = {f["name"]: f["type"] for f in j.get("features", [])}
    cols = []
    for name, t in features.items():
        cols.append({"name": name, "dtype": t.get("dtype") or t.get("_type"),
                     "names": t.get("names") if t.get("_type") == "ClassLabel" else None})
    rows = [r["row"] for r in j.get("rows", [])[:20]]
    return {"hf_id": hf_id, "configs": configs, "config": config, "splits": split_names, "columns": cols, "rows": rows}


def import_hf(hf_id: str, *, name: str, text_col: str | list[str], label_col: str, qtype: str = "choice",
              instructions: str = "", instructions_col: str | None = None, config: str | None = None,
              label_descriptions: dict[str, str] | None = None, max_per_split: int | None = 5000,
              val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42, token: str | None = None,
              progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    from datasets import ClassLabel, load_dataset

    say = progress or (lambda _m: None)
    say(f"Downloading {hf_id}…")
    ds = load_dataset(hf_id, config, token=token)
    first = ds[list(ds.keys())[0]]
    feat = first.features.get(label_col)
    if isinstance(feat, ClassLabel):
        label_names = list(feat.names)
    else:
        vals = set()
        for s in ds:
            vals.update(str(v) for v in ds[s][label_col][:200000])
        label_names = sorted(vals, key=lambda v: (len(v), v))
    text_cols = [text_col] if isinstance(text_col, str) else list(text_col)
    rng = random.Random(seed)

    def question(row) -> dict[str, Any]:
        instr = str(row[instructions_col]) if instructions_col else instructions
        if qtype == "noul":
            return {"type": "noul", "instructions": instr}
        if qtype == "score":
            return {"type": "score", "instructions": instr,
                    "criteria": [(label_descriptions or {}).get(n, humanize(n)) for n in label_names]}
        return {"type": "choice", "instructions": instr,
                "criteria": {n: (label_descriptions or {}).get(n, humanize(n)) for n in label_names}}

    def answer(v):
        if qtype == "noul":
            return bool(v) if not isinstance(v, str) else v.lower() in ("true", "yes", "1")
        idx = int(v) if isinstance(feat, ClassLabel) else label_names.index(str(v))
        return idx if qtype == "score" else label_names[idx]

    def convert(split) -> list[dict[str, Any]]:
        idx = list(range(len(split)))
        rng.shuffle(idx)
        if max_per_split:
            idx = idx[:max_per_split]
        out = []
        for i in idx:
            row = split[i]
            v = row[label_col]
            if v is None or (isinstance(feat, ClassLabel) and int(v) < 0):
                continue  # unlabeled row (e.g. hidden benchmark test labels)
            state = "\n\n".join(str(row[c]) for c in text_cols) if len(text_cols) > 1 else str(row[text_cols[0]])
            out.append({"state": state, "question": question(row), "answer": answer(row[label_col])})
        return out

    say("Converting to System One format…")
    conv = {s: convert(ds[s]) for s in ds.keys()}
    conv = {("validation" if s in ("dev", "valid", "val") else s): v for s, v in conv.items() if v}
    if "train" not in conv:
        allrows = [r for v in conv.values() for r in v]
        splits = split_records(allrows, val=val_ratio, test=test_ratio, seed=seed)
    else:
        need_val, need_test = "validation" not in conv, "test" not in conv
        sp = split_records(conv["train"], val=val_ratio if need_val else 0.0,
                           test=test_ratio if need_test else 0.0, seed=seed)
        splits = {"train": sp["train"],
                  "validation": sp["validation"] if need_val else conv["validation"],
                  "test": sp["test"] if need_test else conv["test"]}
    say("Saving…")
    return save_dataset(name, splits, {"kind": "huggingface", "hf_id": hf_id, "config": config,
                                       "text_col": text_cols, "label_col": label_col, "type": qtype})
