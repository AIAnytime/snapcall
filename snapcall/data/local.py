"""Bring your own data: JSONL / JSON / CSV, either native System One rows or a
plain table mapped with (text column, label column)."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from . import save_dataset, split_records, validate_records
from .hub import humanize


def parse_bytes(data: bytes, filename: str) -> list[dict[str, Any]]:
    text = data.decode("utf-8-sig")
    name = filename.lower()
    if name.endswith(".jsonl") or name.endswith(".ndjson"):
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    if name.endswith(".json"):
        obj = json.loads(text)
        return obj if isinstance(obj, list) else obj.get("data", [obj])
    if name.endswith(".csv") or name.endswith(".tsv"):
        return list(csv.DictReader(io.StringIO(text), delimiter="\t" if name.endswith(".tsv") else ","))
    raise ValueError("Supported formats: .jsonl, .json, .csv, .tsv")


def is_native(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and "state" in rows[0] and ("question" in rows[0] or "questions" in rows[0])


def from_table(rows: list[dict[str, Any]], text_col: str, label_col: str, qtype: str = "choice",
               instructions: str = "", label_descriptions: dict[str, str] | None = None) -> list[dict[str, Any]]:
    labels = sorted({str(r[label_col]) for r in rows if r.get(label_col) not in (None, "")})
    if qtype == "score":
        try:
            labels = sorted(labels, key=float)
        except ValueError:
            pass
    desc = label_descriptions or {}
    if qtype == "score":
        q = {"type": "score", "instructions": instructions, "criteria": [desc.get(l, humanize(l)) for l in labels]}
    elif qtype == "noul":
        q = {"type": "noul", "instructions": instructions}
    else:
        q = {"type": "choice", "instructions": instructions, "criteria": {l: desc.get(l, humanize(l)) for l in labels}}
    out = []
    for r in rows:
        v = r.get(label_col)
        if v in (None, ""):
            continue
        v = str(v)
        ans: Any = v if qtype == "choice" else (labels.index(v) if qtype == "score" else v.lower() in ("1", "true", "yes", "y"))
        out.append({"state": str(r[text_col]), "question": q, "answer": ans})
    return out


def import_rows(name: str, rows: list[dict[str, Any]], *, text_col: str | None = None, label_col: str | None = None,
                qtype: str = "choice", instructions: str = "", split_col: str | None = None,
                val: float = 0.1, test: float = 0.1, seed: int = 42, filename: str = "") -> dict[str, Any]:
    if is_native(rows):
        recs, errors = validate_records(rows)
    else:
        if not (text_col and label_col):
            raise ValueError("Tabular data needs text_col and label_col")
        recs, errors = validate_records(from_table(rows, text_col, label_col, qtype, instructions))
    if not recs:
        raise ValueError("No valid rows. " + "; ".join(errors[:5]))
    if split_col and all(split_col in r for r in rows):
        splits: dict[str, list] = {"train": [], "validation": [], "test": []}
        for raw, rec in zip(rows, recs):
            s = str(raw[split_col]).lower()
            splits["validation" if s in ("val", "valid", "dev") else s if s in splits else "train"].append(rec)
    else:
        splits = split_records(recs, val, test, seed)
    meta = save_dataset(name, splits, {"kind": "upload", "filename": filename, "native": is_native(rows)})
    meta["warnings"] = errors
    return meta
