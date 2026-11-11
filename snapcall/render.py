"""Render (state, question, options) into one token sequence with option spans.

A single forward pass scores every option: we pool the hidden states of each
option's span and a small head turns each pooled vector into a logit. Softmax
over the options gives a closed-set, calibrated distribution. No generation, no
parsing, no invalid outputs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .schema import Example

TEMPLATE_VERSION = 1


@dataclass
class Encoded:
    input_ids: list[int]
    spans: list[tuple[int, int]]  # token [start, end) per option, in *presented* order
    order: list[int]  # presented position -> original option index


_SPECIAL_CACHE: dict[int, tuple[list[int], list[int]]] = {}


def special_tokens(tokenizer) -> tuple[list[int], list[int]]:
    """Discover the tokenizer's own prefix/suffix specials ([CLS]/[SEP], <bos>, ...)."""
    key = id(tokenizer)
    if key not in _SPECIAL_CACHE:
        w = tokenizer("decide", add_special_tokens=True)["input_ids"]
        wo = tokenizer("decide", add_special_tokens=False)["input_ids"]
        pre, suf = [], []
        for i in range(len(w) - len(wo) + 1):
            if w[i:i + len(wo)] == wo:
                pre, suf = w[:i], w[i + len(wo):]
                break
        _SPECIAL_CACHE[key] = (pre, suf)
    return _SPECIAL_CACHE[key]


def _option_line(name: str, desc: str, qtype: str) -> str:
    if qtype == "score":
        return f"- level {name}: {desc}\n"
    if name == desc or not desc:
        return f"- {name}\n"
    return f"- {name}: {desc}\n"


def encode(ex: Example, tokenizer, max_length: int, shuffle: bool = False,
           rng: random.Random | None = None, order: list[int] | None = None) -> Encoded:
    q = ex.question
    n = len(q.options)
    if order is None:
        order = list(range(n))
        if shuffle and q.type == "choice":  # ordinal levels keep their order
            (rng or random).shuffle(order)

    head = f"<question type=\"{q.type}\">\n{q.instructions.strip()}\n</question>\n<options>\n"
    lines = [_option_line(*q.options[i], q.type) for i in order]
    tail = "</options>\n<decision>"

    prefix_special, suffix_special = special_tokens(tokenizer)
    bos: list[int] = []

    def tok(s: str) -> list[int]:
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    head_ids = tok(head)
    line_ids = [tok(l) for l in lines]
    tail_ids = tok(tail)
    fixed = len(bos) + len(prefix_special) + len(head_ids) + sum(map(len, line_ids)) + len(tail_ids) + len(suffix_special)
    state_open, state_close = tok("<state>\n"), tok("\n</state>\n")
    budget = max_length - fixed - len(state_open) - len(state_close)
    if budget < 8:
        # options alone overflow: shrink each option line evenly (rare: huge option sets)
        per = max(4, (max_length - 64) // max(1, n))
        line_ids = [l[:per] for l in line_ids]
        budget = 16
    state_ids = tok(ex.state)[: max(budget, 0)]

    ids: list[int] = bos + prefix_special + state_open + state_ids + state_close + head_ids
    spans = []
    for l in line_ids:
        spans.append((len(ids), len(ids) + len(l)))
        ids += l
    ids += tail_ids + suffix_special
    return Encoded(input_ids=ids, spans=spans, order=order)


def collate(encs: list[Encoded], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad a batch and build an option span mask [B, O, T]."""
    B = len(encs)
    T = max(len(e.input_ids) for e in encs)
    O = max(len(e.spans) for e in encs)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    span_mask = torch.zeros((B, O, T), dtype=torch.float32)
    last_idx = torch.zeros((B, O), dtype=torch.long)
    opt_mask = torch.zeros((B, O), dtype=torch.bool)
    for b, e in enumerate(encs):
        L = len(e.input_ids)
        input_ids[b, :L] = torch.tensor(e.input_ids)
        attn[b, :L] = 1
        for o, (s, t) in enumerate(e.spans):
            span_mask[b, o, s:t] = 1.0
            last_idx[b, o] = t - 1
            opt_mask[b, o] = True
    return {"input_ids": input_ids, "attention_mask": attn, "span_mask": span_mask,
            "last_idx": last_idx, "opt_mask": opt_mask}
