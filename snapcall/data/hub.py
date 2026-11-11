"""Import Hugging Face datasets into System One format."""

from __future__ import annotations

import random
from typing import Any, Callable

import httpx

from . import save_dataset, split_records
from .prep import build_records, humanize, is_empty  # noqa: F401  (humanize re-exported)

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
        inner = t[0] if isinstance(t, list) and t else (t.get("feature") if isinstance(t, dict) else None)
        is_list = isinstance(t, list) or (isinstance(t, dict) and t.get("_type") in ("Sequence", "List", "LargeList"))
        if is_list:
            names = inner.get("names") if isinstance(inner, dict) and inner.get("_type") == "ClassLabel" else None
            cols.append({"name": name, "dtype": "list", "names": names})
        else:
            cols.append({"name": name, "dtype": t.get("dtype") or t.get("_type"),
                         "names": t.get("names") if t.get("_type") == "ClassLabel" else None})
    rows = [r["row"] for r in j.get("rows", [])[:20]]
    return {"hf_id": hf_id, "configs": configs, "config": config, "splits": split_names, "columns": cols, "rows": rows}


def class_names_of(feat) -> list[str] | None:
    """Label names for a ClassLabel, or a list/Sequence of ClassLabel (multi-label)."""
    from datasets import ClassLabel
    if isinstance(feat, ClassLabel):
        return list(feat.names)
    inner = getattr(feat, "feature", None)
    if isinstance(inner, ClassLabel):
        return list(inner.names)
    if isinstance(feat, (list, tuple)) and feat and isinstance(feat[0], ClassLabel):
        return list(feat[0].names)
    return None


def convert_dataset(ds, *, text_col: str | list[str], label_col: str, qtype: str = "choice",
                    instructions: str = "", instructions_col: str | None = None,
                    label_descriptions: dict[str, str] | None = None, max_per_split: int | None = 5000,
                    top_labels: int | None = None, other_label: str | None = None, multi_label: str = "single",
                    seed: int = 42) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """DatasetDict (or {split: Dataset}) -> System One records per split."""
    first = ds[list(ds.keys())[0]]
    names = class_names_of(first.features.get(label_col))
    text_cols = [text_col] if isinstance(text_col, str) else list(text_col)
    rng = random.Random(seed)

    def state_of(row) -> str | None:
        parts = [str(row[c]).strip() for c in text_cols if not is_empty(row.get(c))]
        return "\n\n".join(parts) if parts else None

    raw: dict[str, list] = {}
    for split in ds.keys():
        d = ds[split]
        idx = list(range(len(d)))
        rng.shuffle(idx)
        rows = []
        # oversample a little so skipped (empty/rare) rows don't shrink the split below max_per_split;
        # in multi-label "noul" mode each row becomes one record per label, so cap rows exactly
        take = None if not max_per_split else (max_per_split if multi_label == "noul" else int(max_per_split * 1.5))
        for i in idx[:take]:
            row = d[i]
            instr = str(row[instructions_col]) if instructions_col else instructions
            rows.append((state_of(row), instr, row[label_col]))
        name = "validation" if split in ("dev", "valid", "val") else split
        raw[name] = raw.get(name, []) + rows
    recs, stats = build_records(raw, qtype=qtype, class_names=names, top_labels=top_labels,
                                other_label=other_label, multi_label=multi_label,
                                label_descriptions=label_descriptions)
    if max_per_split and multi_label != "noul":
        recs = {k: v[:max_per_split] for k, v in recs.items()}
    return {k: v for k, v in recs.items() if v}, stats


def import_hf(hf_id: str, *, name: str, text_col: str | list[str], label_col: str, qtype: str = "choice",
              instructions: str = "", instructions_col: str | None = None, config: str | None = None,
              label_descriptions: dict[str, str] | None = None, max_per_split: int | None = 5000,
              top_labels: int | None = None, other_label: str | None = None, multi_label: str = "single",
              val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42, token: str | None = None,
              progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    from datasets import load_dataset

    say = progress or (lambda _m: None)
    say(f"Downloading {hf_id}…")
    ds = load_dataset(hf_id, config, token=token)
    say("Converting to System One format…")
    conv, stats = convert_dataset(ds, text_col=text_col, label_col=label_col, qtype=qtype, instructions=instructions,
                                  instructions_col=instructions_col, label_descriptions=label_descriptions,
                                  max_per_split=max_per_split, top_labels=top_labels, other_label=other_label,
                                  multi_label=multi_label, seed=seed)
    if not conv:
        raise ValueError(f"No usable rows. {stats}")
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
    meta = save_dataset(name, splits, {"kind": "huggingface", "hf_id": hf_id, "config": config,
                                       "text_col": [text_col] if isinstance(text_col, str) else list(text_col),
                                       "label_col": label_col, "type": qtype, "top_labels": top_labels,
                                       "multi_label": multi_label, "prep": stats})
    meta["prep"] = stats
    return meta
