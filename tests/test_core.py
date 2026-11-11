"""Fast tests: no model downloads (render test uses a cached tokenizer when available)."""

import math

import pytest

from snapcall.data import split_records
from snapcall.data.faker_gen import FAKER_EXAMPLE, Augmenter, generate
from snapcall.metrics import compute, fit_temperature
from snapcall.schema import Example, Question, build_response, flatten_system_one


def rec(ans="billing"):
    return {"state": "I was charged twice", "answer": ans,
            "question": {"type": "choice", "instructions": "Route", "criteria": {"billing": "b", "shipping": "s"}}}


def test_example_parsing_all_types():
    e = Example.from_record(rec("shipping"))
    assert e.label == 1 and e.question.names == ["billing", "shipping"]
    s = Example.from_record({"state": {"a": 1}, "answer": 2,
                             "question": {"type": "score", "instructions": "x", "criteria": ["lo", "mid", "hi"]}})
    assert s.label == 2 and s.state.startswith("{")
    n = Example.from_record({"state": "x", "answer": True, "question": {"type": "noul", "instructions": "?"}})
    assert n.label == 1 and n.question.names == ["no", "yes"]
    with pytest.raises(ValueError):
        Example.from_record(rec("refunds"))


def test_flatten_multi_question():
    rows = flatten_system_one({"state": "s", "questions": {"a": rec()["question"], "b": {"type": "noul", "instructions": "?"}},
                               "answers": {"a": {"choice": "billing"}, "b": {"noul": 1}}})
    assert len(rows) == 2 and rows[0]["answer"] == "billing"


def test_split_is_stratified():
    recs = [rec("billing") for _ in range(50)] + [rec("shipping") for _ in range(10)]
    sp = split_records(recs, 0.1, 0.2)
    assert sum(map(len, sp.values())) == 60
    assert any(r["answer"] == "shipping" for r in sp["test"])


def test_metrics_perfect_and_calibrated():
    exs = [Example.from_record(rec("billing")), Example.from_record(rec("shipping"))]
    m = compute(exs, [[1.0, 0.0], [0.0, 1.0]], with_details=True)
    assert m["accuracy"] == 1.0 and m["ece"] < 1e-9 and m["brier"] < 1e-9
    m2 = compute(exs, [[0.9, 0.1], [0.9, 0.1]])
    assert m2["accuracy"] == 0.5 and m2["ece"] == pytest.approx(0.4)


def test_temperature_fit_reduces_overconfidence():
    logits = [[8.0, 0.0]] * 50 + [[0.0, 8.0]] * 50 + [[8.0, 0.0]] * 30  # 30 confident mistakes
    labels = [0] * 50 + [1] * 50 + [1] * 30
    assert fit_temperature(logits, labels) > 1.5


def test_build_response_shapes():
    q = [Question.from_dict(rec()["question"]), Question.from_dict({"type": "noul", "instructions": "?"})]
    r = build_response(["a", "b"], q, [[0.2, 0.8], [0.3, 0.7]], "m")
    assert r["answers"]["a"]["choice"] == "shipping" and math.isclose(r["answers"]["b"]["noul"], 0.7)


def test_faker_generation_and_augment():
    rows = generate(FAKER_EXAMPLE, n_per_option=5)
    assert len(rows) == 15 and all("{" not in r["state"] for r in rows)
    aug = Augmenter(prob=1.0)
    out = aug("mail me at a.b@example.com about $12.50")
    assert "a.b@example.com" not in out


def test_render_spans_cover_options():
    tr = pytest.importorskip("transformers")
    try:
        tok = tr.AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", local_files_only=True)
    except Exception:
        pytest.skip("tokenizer not cached")
    from snapcall.render import encode
    ex = Example.from_record(rec())
    enc = encode(ex, tok, 64, shuffle=False)
    assert len(enc.spans) == 2 and all(s < e <= len(enc.input_ids) for s, e in enc.spans)
    long = Example(state="word " * 5000, question=ex.question, label=0)
    assert len(encode(long, tok, 128).input_ids) <= 128
