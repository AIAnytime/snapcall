"""Bring your own data: JSONL / JSON / CSV, either native System One rows or a
plain table mapped with (text column, label column)."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from . import save_dataset, split_records, validate_records
from .prep import build_records


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
               instructions: str = "", label_descriptions: dict[str, str] | None = None,
               top_labels: int | None = None, other_label: str | None = None,
               multi_label: str = "single") -> list[dict[str, Any]]:
    """Map a plain table to System One records. Labels may be lists ("a|b" strings or JSON lists) for multi-label."""
    def lab(v):
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        if isinstance(v, str) and "|" in v and multi_label in ("first", "noul"):
            return [x.strip() for x in v.split("|") if x.strip()]
        return v
    raw = {"all": [(r.get(text_col), instructions, lab(r.get(label_col))) for r in rows]}
    recs, _ = build_records(raw, qtype=qtype, top_labels=top_labels, other_label=other_label,
                            multi_label=multi_label, label_descriptions=label_descriptions)
    return recs.get("all", [])


def import_rows(name: str, rows: list[dict[str, Any]], *, text_col: str | None = None, label_col: str | None = None,
                qtype: str = "choice", instructions: str = "",
                top_labels: int | None = None, other_label: str | None = None, multi_label: str = "single",
                val: float = 0.1, test: float = 0.1, seed: int = 42, filename: str = "") -> dict[str, Any]:
    if is_native(rows):
        recs, errors = validate_records(rows)
    else:
        if not (text_col and label_col):
            raise ValueError("Tabular data needs text_col and label_col")
        recs, errors = validate_records(from_table(rows, text_col, label_col, qtype, instructions,
                                                    top_labels=top_labels, other_label=other_label,
                                                    multi_label=multi_label))
    if not recs:
        raise ValueError("No valid rows. " + "; ".join(errors[:5]))
    splits = split_records(recs, val, test, seed)
    meta = save_dataset(name, splits, {"kind": "upload", "filename": filename, "native": is_native(rows)})
    meta["warnings"] = errors
    return meta
