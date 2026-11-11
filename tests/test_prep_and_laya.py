"""Import fixes (empty rows, numeric order, top-N, multi-label) and the Laya loader. No downloads."""

import json

import pytest
import torch

from snapcall.data.prep import build_records, label_str, order_labels
from snapcall.schema import Example

datasets = pytest.importorskip("datasets")
from snapcall.data.hub import convert_dataset  # noqa: E402


def dd(**cols):
    return datasets.DatasetDict({"train": datasets.Dataset.from_dict(cols)})


def test_empty_text_rows_are_skipped():
    ds = dd(text=["Hail 2 inch", None, "", "nan", "Flash flood on Route 9"], label=["Hail", "Hail", "Tornado", "Hail", "Flood"])
    recs, stats = convert_dataset(ds, text_col="text", label_col="label", max_per_split=None)
    assert sorted(r["state"] for r in recs["train"]) == ["Flash flood on Route 9", "Hail 2 inch"]
    assert stats["skipped_empty_text"] == 3


def test_numeric_labels_ordered_by_value_for_score():
    ratings = [float(x) for x in range(1, 11)] * 3
    ds = dd(review=[f"review {i}" for i in range(len(ratings))], rating=ratings)
    recs, _ = convert_dataset(ds, text_col="review", label_col="rating", qtype="score", max_per_split=None)
    rows = recs["train"]
    crit = rows[0]["question"]["criteria"]
    assert crit == [str(i) for i in range(1, 11)]  # "1".."10", not "1","10","2"
    by_state = {r["state"]: r["answer"] for r in rows}
    assert by_state["review 9"] == 9  # rating 10.0 -> level index 9
    assert by_state["review 1"] == 1  # rating 2.0 -> level index 1
    for r in rows:
        Example.from_record(r)
    assert order_labels(["10", "2", "1"]) == ["1", "2", "10"] and label_str(9.0) == "9"


def test_top_n_labels_drop_or_merge():
    labels = ["CWE-79"] * 10 + ["CWE-89"] * 6 + ["CWE-22"] * 3 + ["CWE-1"]
    ds = dd(text=[f"cve {i}" for i in range(len(labels))], cwe=labels)
    recs, stats = convert_dataset(ds, text_col="text", label_col="cwe", top_labels=2, max_per_split=None)
    assert set(recs["train"][0]["question"]["criteria"]) == {"CWE-79", "CWE-89"}
    assert len(recs["train"]) == 16 and stats["skipped_rare_label"] == 4
    recs, _ = convert_dataset(ds, text_col="text", label_col="cwe", top_labels=2, other_label="other", max_per_split=None)
    assert list(recs["train"][0]["question"]["criteria"])[-1] == "other" and len(recs["train"]) == 20


def _multi():
    feats = datasets.Features({"text": datasets.Value("string"),
                               "labels": datasets.Sequence(datasets.ClassLabel(names=["none", "angiogenesis", "invasion"]))})
    return datasets.DatasetDict({"train": datasets.Dataset.from_dict(
        {"text": ["a", "b", "c", "d"], "labels": [[1], [1, 2], [0], []]}, features=feats)})


def test_multi_label_single_first_noul():
    recs, stats = convert_dataset(_multi(), text_col="text", label_col="labels", multi_label="single", max_per_split=None)
    assert stats["skipped_multi_label"] == 1 and stats["skipped_empty_label"] == 1
    assert {r["state"] for r in recs["train"]} == {"a", "c"}
    recs, _ = convert_dataset(_multi(), text_col="text", label_col="labels", multi_label="first", max_per_split=None)
    assert {r["state"]: r["answer"] for r in recs["train"]}["b"] == "angiogenesis"
    recs, _ = convert_dataset(_multi(), text_col="text", label_col="labels", multi_label="noul", max_per_split=None,
                              instructions="Hallmarks of cancer:")
    rows = recs["train"]
    assert all(r["question"]["type"] == "noul" for r in rows)
    assert len(rows) == 3 * 2  # 3 labelled rows x 2 real hallmarks ("none" is not asked)
    b = {r["question"]["instructions"]: r["answer"] for r in rows if r["state"] == "b"}
    assert sum(b.values()) == 2
    for r in rows:
        Example.from_record(r)


def test_build_records_table_style_labels():
    recs, _ = build_records({"all": [("x", "", "pos"), ("y", "", "neg"), ("z", "", " ")]})
    assert len(recs["all"]) == 2


# ---------------------------------------------------------------- Laya
def test_laya_spec_parsing():
    from snapcall.modeling import laya_spec
    assert laya_spec("convaiinnovations/laya") == ("convaiinnovations/laya", "")
    assert laya_spec("convaiinnovations/laya:multilingual") == ("convaiinnovations/laya", "multilingual/")
    assert laya_spec("answerdotai/ModernBERT-base") is None


def test_load_laya_from_local_checkpoint(tmp_path):
    """A tiny fake Laya repo: encoder.* weights + extra heads in one safetensors file."""
    from safetensors.torch import save_file
    from transformers import ModernBertConfig, ModernBertModel

    from snapcall.modeling import detect_arch, load_backbone
    cfg = ModernBertConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, max_position_embeddings=64, pad_token_id=0)
    src = ModernBertModel(cfg)
    (tmp_path / "encoder").mkdir()
    cfg.save_pretrained(tmp_path / "encoder")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "laya", "architectures": ["LayaTypedDecisions"]}))
    tensors = {f"encoder.{k}": v.contiguous() for k, v in src.state_dict().items()}
    tensors.update({"temperature": torch.ones(3), "head.layers.0.linear1.weight": torch.zeros(4, 32)})
    save_file(tensors, str(tmp_path / "model.safetensors"))

    assert detect_arch(str(tmp_path)) == "encoder"
    m = load_backbone(str(tmp_path), dtype=torch.float32)
    assert isinstance(m, ModernBertModel)
    for k, v in src.state_dict().items():
        assert torch.equal(m.state_dict()[k], v), k


def test_classlabel_names_with_whitespace():
    """galileo-ai/medical_transcription_40 has names like ' Pain Management'."""
    feats = datasets.Features({"text": datasets.Value("string"),
                               "label": datasets.ClassLabel(names=[" Pain Management", " Cardiology / Vascular"])})
    ds = datasets.DatasetDict({"train": datasets.Dataset.from_dict(
        {"text": ["note a", "note b", "note c"], "label": [0, 1, 1]}, features=feats)})
    recs, _ = convert_dataset(ds, text_col="text", label_col="label", max_per_split=None)
    assert set(recs["train"][0]["question"]["criteria"]) == {"Pain Management", "Cardiology / Vascular"}
    for r in recs["train"]:
        Example.from_record(r)
