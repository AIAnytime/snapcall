"""Turn raw (text, label) rows into System One records, shared by HF imports and file uploads.

Handles the messy parts of real datasets:
  * rows with empty text or labels are skipped ("", None, "nan", "null", ...)
  * numeric labels are ordered by value ("2" < "10"), and "9.0" is shown as "9"
  * --top-labels N keeps the N most frequent labels (others dropped, or merged into --other-label)
  * multi-label rows (lists) become one choice ("single" / "first") or one yes/no question per label ("noul")
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

EMPTY = {"", "nan", "none", "null", "n/a", "na"}
MULTI_LABEL_MODES = ("single", "first", "noul")


def humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip()


def is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and v != v:  # NaN
        return True
    if isinstance(v, (list, tuple)):
        return len(v) == 0
    return str(v).strip().lower() in EMPTY


def label_str(v: Any) -> str:
    """Canonical label text: 9.0 -> "9", " Cardiology " -> "Cardiology"."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def order_labels(labels: Iterable[str]) -> list[str]:
    labels = list(dict.fromkeys(labels))
    try:
        return sorted(labels, key=float)
    except ValueError:
        return sorted(labels, key=lambda x: x.lower())


def build_records(splits: dict[str, list[tuple[str, str, Any]]], *, qtype: str = "choice",
                  class_names: list[str] | None = None, top_labels: int | None = None,
                  other_label: str | None = None, multi_label: str = "single",
                  label_descriptions: dict[str, str] | None = None,
                  noul_template: str = "Does this text describe: {label}?") -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """splits: name -> [(state, instructions, raw_label)]. Returns (records per split, stats)."""
    if multi_label not in MULTI_LABEL_MODES:
        raise ValueError(f"multi_label must be one of {MULTI_LABEL_MODES}")
    desc = label_descriptions or {}
    stats: Counter = Counter()

    def norm(v: Any) -> list[str] | str | None:
        if isinstance(v, int) and not isinstance(v, bool) and class_names is not None:
            return class_names[v] if 0 <= v < len(class_names) else None  # -1 = unlabeled
        if isinstance(v, (list, tuple)):
            vals = [norm(x) for x in v]
            return [x for x in vals if isinstance(x, str) and x]
        return None if is_empty(v) else label_str(v)

    # pass 1: clean rows
    clean: dict[str, list[tuple[str, str, Any]]] = {}
    for split, rows in splits.items():
        out = []
        for state, instr, raw in rows:
            if is_empty(state):
                stats["skipped_empty_text"] += 1
                continue
            if qtype == "noul" and not isinstance(raw, (list, tuple)):
                if is_empty(raw):
                    stats["skipped_empty_label"] += 1
                    continue
                out.append((str(state), instr, raw))
                continue
            lab = norm(raw)
            if lab is None or lab == []:
                stats["skipped_empty_label"] += 1
                continue
            out.append((str(state), instr, lab))
        clean[split] = out

    # multi-label as one yes/no question per label
    is_multi = any(isinstance(l, list) for rows in clean.values() for _, _, l in rows)
    if is_multi and multi_label == "noul":
        universe = order_labels({x for rows in clean.values() for _, _, l in rows for x in (l if isinstance(l, list) else [l])}) \
            if class_names is None else [n for n in class_names]
        universe = [u for u in universe if u.strip().lower() not in EMPTY]
        if top_labels:
            cnt = Counter(x for _, _, l in clean.get("train", []) or next(iter(clean.values())) for x in (l if isinstance(l, list) else [l]))
            universe = [u for u, _ in cnt.most_common() if u in universe][:top_labels]
        recs: dict[str, list[dict]] = {}
        for split, rows in clean.items():
            r = []
            for state, instr, l in rows:
                have = set(l if isinstance(l, list) else [l])
                for u in universe:
                    q = noul_template.format(label=desc.get(u, humanize(u)))
                    r.append({"state": state, "question": {"type": "noul", "instructions": f"{instr} {q}".strip() if instr else q},
                              "answer": u in have})
            recs[split] = r
        stats["labels"] = len(universe)
        return recs, dict(stats)

    # multi-label rows -> single label
    if is_multi:
        for split, rows in clean.items():
            kept = []
            for state, instr, l in rows:
                if isinstance(l, list):
                    if multi_label == "single" and len(l) != 1:
                        stats["skipped_multi_label"] += 1
                        continue
                    l = l[0]
                kept.append((state, instr, l))
            clean[split] = kept

    if qtype == "noul":
        recs = {s: [{"state": st, "question": {"type": "noul", "instructions": ins},
                     "answer": (raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "yes", "y"))}
                    for st, ins, raw in rows] for s, rows in clean.items()}
        return recs, dict(stats)

    # top-N labels
    counts = Counter(l for rows in clean.values() for _, _, l in rows)
    if top_labels and len(counts) > top_labels:
        keep = {l for l, _ in counts.most_common(top_labels)}
        for split, rows in clean.items():
            new = []
            for state, instr, l in rows:
                if l in keep:
                    new.append((state, instr, l))
                elif other_label:
                    new.append((state, instr, other_label))
                else:
                    stats["skipped_rare_label"] += 1
            clean[split] = new
        stats["labels_dropped"] = len(counts) - len(keep)

    present = {l for rows in clean.values() for _, _, l in rows}
    if class_names is not None and qtype == "score":
        names = [label_str(n) for n in class_names if label_str(n) in present]
    elif class_names is not None:
        names = [label_str(n) for n in class_names if label_str(n) in present] + sorted(present - {label_str(n) for n in class_names})
    else:
        names = order_labels(present)
        if other_label and other_label in names:  # keep the catch-all last
            names = [n for n in names if n != other_label] + [other_label]
    if qtype == "score" and not 2 <= len(names) <= 10:
        raise ValueError(f"score questions need 2-10 levels; this column has {len(names)}. "
                         "Use type=choice, or --top-labels.")
    if qtype == "choice" and len(names) > 255:
        raise ValueError(f"{len(names)} labels: choice questions support up to 255. Use --top-labels 50 (or similar).")
    if qtype == "choice" and len(names) < 2:
        raise ValueError("Need at least 2 distinct labels")

    def question(instr: str) -> dict[str, Any]:
        if qtype == "score":
            return {"type": "score", "instructions": instr, "criteria": [desc.get(n, humanize(n)) for n in names]}
        return {"type": "choice", "instructions": instr, "criteria": {n: desc.get(n, humanize(n)) for n in names}}

    recs = {}
    for split, rows in clean.items():
        recs[split] = [{"state": st, "question": question(ins), "answer": names.index(l) if qtype == "score" else l}
                       for st, ins, l in rows]
    stats["labels"] = len(names)
    return recs, dict(stats)
