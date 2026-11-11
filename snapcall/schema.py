"""System One data types.

Snapcall speaks the same shape as the System One API that Jev, Clef, d1 and
friends expose: a *state* (text or JSON) plus a set of *typed questions*. Every
question resolves to a distribution over a closed set of options, so a trained
model can never return an option that was not offered.

Supported question types
------------------------
choice  -> pick one of N named options (``criteria`` maps name -> description)
score   -> ordinal levels (``criteria`` is an ordered list, index = level)
noul    -> yes / no, answer is P(yes) in [0, 1]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

QUESTION_TYPES = ("choice", "score", "noul")
NOUL_OPTIONS = [("no", "The answer is no / false."), ("yes", "The answer is yes / true.")]


@dataclass
class Question:
    type: str
    instructions: str
    # list of (name, description); for score the name is the level index as str
    options: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Question":
        qtype = d.get("type", "choice")
        if qtype not in QUESTION_TYPES:
            raise ValueError(f"Unsupported question type {qtype!r}; expected one of {QUESTION_TYPES}")
        instr = d.get("instructions", "")
        if not isinstance(instr, str):
            instr = json.dumps(instr, ensure_ascii=False)
        crit = d.get("criteria")
        if qtype == "noul":
            opts = list(NOUL_OPTIONS)
        elif qtype == "score":
            if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
                raise ValueError("score questions need `criteria` as a list of 2-10 level descriptions")
            opts = [(str(i), str(desc)) for i, desc in enumerate(crit)]
        else:
            if isinstance(crit, dict):
                opts = [(str(k), str(v)) for k, v in crit.items()]
            elif isinstance(crit, list):
                opts = [(str(k), str(k)) for k in crit]
            else:
                raise ValueError("choice questions need `criteria` as {name: description}")
            if not 2 <= len(opts) <= 255:
                raise ValueError("choice questions need between 2 and 255 options")
        return cls(type=qtype, instructions=instr, options=opts)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.type == "choice":
            d["criteria"] = {n: desc for n, desc in self.options}
        elif self.type == "score":
            d["criteria"] = [desc for _, desc in self.options]
        return d

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self.options]


@dataclass
class Example:
    """One flattened decision: a state, one question, and its target."""

    state: str
    question: Question
    label: int  # index into question.options
    soft: list[float] | None = None  # optional target distribution (e.g. annotator votes)

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "Example":
        state = rec["state"]
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        q = Question.from_dict(rec["question"])
        ans = rec["answer"]
        names = q.names
        if q.type == "noul":
            if isinstance(ans, bool):
                label = int(ans)
            elif isinstance(ans, (int, float)):
                label = int(float(ans) >= 0.5)
            else:
                label = int(str(ans).strip().lower() in ("yes", "true", "1", "y"))
        elif q.type == "score":
            label = int(round(float(ans)))
        else:
            if str(ans) not in names:
                raise ValueError(f"answer {ans!r} not in options {names[:8]}...")
            label = names.index(str(ans))
        soft = None
        if rec.get("soft"):
            s = rec["soft"]
            soft = [float(s.get(n, 0.0)) for n in names] if isinstance(s, dict) else [float(x) for x in s]
            tot = sum(soft) or 1.0
            soft = [x / tot for x in soft]
        return cls(state=state, question=q, label=label, soft=soft)

    def to_record(self) -> dict[str, Any]:
        q = self.question
        if q.type == "noul":
            ans: Any = bool(self.label)
        elif q.type == "score":
            ans = self.label
        else:
            ans = q.names[self.label]
        rec = {"state": self.state, "question": q.to_dict(), "answer": ans}
        if self.soft:
            rec["soft"] = {n: p for n, p in zip(q.names, self.soft)}
        return rec


def flatten_system_one(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept either a flat record or a multi-question System One record.

    Multi-question shape::

        {"state": ..., "questions": {"qid": {...}}, "answers": {"qid": {"choice": "x"}}}
    """
    if "question" in rec:
        return [rec]
    out = []
    for qid, q in (rec.get("questions") or {}).items():
        a = (rec.get("answers") or {}).get(qid)
        if a is None:
            continue
        if isinstance(a, dict):
            a = a.get("choice", a.get("score", a.get("noul")))
        out.append({"state": rec["state"], "question": q, "answer": a})
    return out


def build_response(question_ids: list[str], questions: list[Question], probs: list[list[float]],
                   model_name: str, usage: dict[str, int] | None = None) -> dict[str, Any]:
    """Shape model outputs like the System One API response."""
    answers: dict[str, Any] = {}
    for qid, q, p in zip(question_ids, questions, probs):
        conf = max(p)
        if q.type == "choice":
            top = q.names[p.index(conf)]
            answers[qid] = {"type": "choice", "choice": top, "confidence": round(conf, 6),
                            "probabilities": {n: round(x, 6) for n, x in zip(q.names, p)}}
        elif q.type == "score":
            exp = sum(i * x for i, x in enumerate(p))
            answers[qid] = {"type": "score", "score": round(exp, 6), "confidence": round(conf, 6),
                            "legend": {str(i): d for i, (_, d) in enumerate(q.options)},
                            "probabilities": {str(i): round(x, 6) for i, x in enumerate(p)}}
        else:
            answers[qid] = {"type": "noul", "noul": round(p[1], 6)}
    return {"model": model_name, "answers": answers, "usage": usage or {"input_tokens": 0, "output_tokens": 0}}
