"""Decision metrics: accuracy, F1, calibration (ECE, Brier, NLL), reliability, confusion."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .schema import Example


def _ece(confs: list[float], correct: list[int], n_bins: int = 15) -> tuple[float, list[dict[str, float]]]:
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confs, correct):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, y))
    ece, rel, n = 0.0, [], max(1, len(confs))
    for i, b in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not b:
            rel.append({"lo": lo, "hi": hi, "count": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += len(b) / n * abs(acc - conf)
        rel.append({"lo": lo, "hi": hi, "count": len(b), "confidence": conf, "accuracy": acc})
    return ece, rel


def compute(examples: list[Example], probs: list[list[float]], with_details: bool = False) -> dict[str, Any]:
    if not examples:
        return {}
    n = len(examples)
    correct, confs, briers, nlls, abs_err = [], [], [], [], []
    per_label: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ex, p in zip(examples, probs):
        pred = max(range(len(p)), key=p.__getitem__)
        y = ex.label
        target = ex.soft or [1.0 if i == y else 0.0 for i in range(len(p))]
        correct.append(int(pred == y))
        confs.append(p[pred])
        briers.append(sum((pi - ti) ** 2 for pi, ti in zip(p, target)))
        nlls.append(-math.log(max(p[y], 1e-12)))
        if ex.question.type == "score":
            abs_err.append(abs(sum(i * x for i, x in enumerate(p)) - y))
        names = ex.question.names
        pn, yn = names[pred], names[y]
        confusion[yn][pn] += 1
        if pred == y:
            per_label[yn]["tp"] += 1
        else:
            per_label[pn]["fp"] += 1
            per_label[yn]["fn"] += 1
    f1s = []
    for c in per_label.values():
        prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    ece, rel = _ece(confs, correct)
    out: dict[str, Any] = {
        "accuracy": sum(correct) / n,
        "macro_f1": sum(f1s) / max(1, len(f1s)),
        "ece": ece,
        "brier": sum(briers) / n,
        "nll": sum(nlls) / n,
        "avg_confidence": sum(confs) / n,
        "n": n,
    }
    if abs_err:
        out["score_mae"] = sum(abs_err) / len(abs_err)
    if with_details:
        out["reliability"] = rel
        labels = sorted(confusion.keys() | {k for v in confusion.values() for k in v})
        if len(labels) <= 40:
            out["confusion"] = {"labels": labels,
                                "matrix": [[confusion[a].get(b, 0) for b in labels] for a in labels]}
    return out


def fit_temperature(logit_rows: list[list[float]], labels: list[int], steps: int = 200) -> float:
    """Fit a single temperature on validation logits by minimising NLL (LBFGS)."""
    import torch
    if not logit_rows:
        return 1.0
    O = max(len(r) for r in logit_rows)
    L = torch.full((len(logit_rows), O), -1e4)
    for i, r in enumerate(logit_rows):
        L[i, :len(r)] = torch.tensor(r)
    y = torch.tensor(labels)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(L / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp())
    return min(max(t, 0.05), 20.0)
