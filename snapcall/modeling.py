"""DecisionModel: any Hugging Face backbone + a typed decision head."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from . import __version__
from .render import TEMPLATE_VERSION, collate, encode
from .schema import Example, Question, build_response

ENCODER_TYPES = {"bert", "roberta", "xlm-roberta", "modernbert", "deberta", "deberta-v2", "electra",
                 "distilbert", "mpnet", "ettin", "neobert", "laya"}


def resolve_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


def detect_arch(model_id: str, trust_remote_code: bool = False) -> str:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if cfg.model_type in ENCODER_TYPES:
        return "encoder"
    archs = " ".join(getattr(cfg, "architectures", None) or [])
    if "MaskedLM" in archs and "Causal" not in archs:
        return "encoder"
    return "causal"


def hidden_size_of(config) -> int:
    for c in (config, getattr(config, "text_config", None)):
        if c is not None and getattr(c, "hidden_size", None):
            return int(c.hidden_size)
    if hasattr(config, "get_text_config"):
        return int(config.get_text_config().hidden_size)
    raise ValueError("Could not infer hidden size from model config")


class DecisionHead(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DecisionModel(nn.Module):
    def __init__(self, backbone: nn.Module, tokenizer, meta: dict[str, Any]):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.meta = meta
        self.head = DecisionHead(meta["hidden_size"])
        self.temperature = float(meta.get("temperature", 1.0))

    # ------------------------------------------------------------------ build
    @classmethod
    def from_base(cls, base_model: str, arch: str = "auto", pooling: str = "auto", finetune: str = "full",
                  precision: str = "auto", device: str | torch.device = "auto", trust_remote_code: bool = False,
                  gradient_checkpointing: bool = False, lora_r: int = 16, lora_alpha: int = 32,
                  lora_dropout: float = 0.05, lora_targets: str = "all-linear") -> "DecisionModel":
        from transformers import AutoModel, AutoTokenizer

        dev = device if isinstance(device, torch.device) else resolve_device(device)
        if arch == "auto":
            arch = detect_arch(base_model, trust_remote_code)
        if pooling == "auto":
            pooling = "mean" if arch == "encoder" else "last"
        dtype = resolve_dtype(precision, dev)
        if finetune == "full":
            # keep fp32 master weights; the trainer uses autocast for bf16/fp16 compute on CUDA
            dtype = torch.float32
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "dtype": dtype}
        if finetune == "qlora":
            if dev.type != "cuda":
                raise RuntimeError("QLoRA needs a CUDA GPU with bitsandbytes. Use finetune='lora' instead.")
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
            kwargs["device_map"] = {"": 0}
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        backbone = AutoModel.from_pretrained(base_model, **kwargs)
        if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(backbone, "enable_input_require_grads"):
                backbone.enable_input_require_grads()
        if finetune in ("lora", "qlora"):
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if finetune == "qlora":
                backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=gradient_checkpointing)
            targets: Any = lora_targets if lora_targets == "all-linear" else [t.strip() for t in lora_targets.split(",")]
            backbone = get_peft_model(backbone, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                                           target_modules=targets, bias="none"))
        elif finetune == "head":
            for p in backbone.parameters():
                p.requires_grad_(False)
        meta = {"base_model": base_model, "arch": arch, "pooling": pooling, "finetune": finetune,
                "hidden_size": hidden_size_of(backbone.config), "trust_remote_code": trust_remote_code,
                "template_version": TEMPLATE_VERSION, "temperature": 1.0, "dtype": str(dtype).replace("torch.", "")}
        model = cls(backbone, tok, meta)
        if finetune != "qlora":
            model.backbone.to(dev)
        model.head.to(dev)
        model.device_ = dev
        return model

    @property
    def device(self) -> torch.device:
        return getattr(self, "device_", next(self.head.parameters()).device)

    # --------------------------------------------------------------- forward
    def forward(self, input_ids, attention_mask, span_mask, last_idx, opt_mask, **_):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        H = H.float()
        if self.meta["pooling"] == "mean":
            cnt = span_mask.sum(-1, keepdim=True).clamp(min=1.0)
            pooled = torch.bmm(span_mask, H) / cnt
        else:
            idx = last_idx.unsqueeze(-1).expand(-1, -1, H.size(-1))
            pooled = torch.gather(H, 1, idx)
        logits = self.head(pooled)
        return logits.masked_fill(~opt_mask, -1e4)

    def batch(self, examples: list[Example], max_length: int, shuffle: bool = False, rng=None,
              orders: list[list[int]] | None = None):
        encs = [encode(ex, self.tokenizer, max_length, shuffle=shuffle, rng=rng,
                       order=orders[i] if orders else None) for i, ex in enumerate(examples)]
        b = collate(encs, self.tokenizer.pad_token_id)
        return {k: v.to(self.device) for k, v in b.items()}, encs

    def trainable_parameters(self) -> tuple[int, int]:
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return tr, tot

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_proba(self, examples: list[Example], batch_size: int = 16, max_length: int | None = None,
                      temperature: float | None = None) -> list[list[float]]:
        """Probabilities in each question's *original* option order."""
        self.eval()
        T = self.temperature if temperature is None else temperature
        ml = max_length or self.meta.get("max_length", 512)
        out: list[list[float]] = []
        for i in range(0, len(examples), batch_size):
            chunk = examples[i:i + batch_size]
            b, encs = self.batch(chunk, ml)
            logits = self(**b) / T
            probs = torch.softmax(logits, -1).cpu()
            for j, e in enumerate(encs):
                p = [0.0] * len(e.order)
                for pos, orig in enumerate(e.order):
                    p[orig] = float(probs[j, pos])
                out.append(p)
        return out

    def decide(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """System One style call: one state, many typed questions."""
        st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        qids = list(questions)
        qs = [Question.from_dict(questions[q]) for q in qids]
        exs = [Example(state=st, question=q, label=0) for q in qs]
        t0 = time.perf_counter()
        probs = self.predict_proba(exs)
        n_tok = len(self.tokenizer(st, add_special_tokens=False)["input_ids"])
        resp = build_response(qids, qs, probs, self.meta.get("name", self.meta["base_model"]),
                              {"input_tokens": n_tok * len(qs), "output_tokens": 0})
        resp["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return resp

    # ----------------------------------------------------------- save / load
    def save(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> None:
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {**self.meta, "temperature": self.temperature, "snapcall_version": __version__, **(extra_meta or {})}
        (path / "snapcall_config.json").write_text(json.dumps(meta, indent=2))
        save_file({k: v.detach().float().cpu().contiguous() for k, v in self.head.state_dict().items()},
                  str(path / "decision_head.safetensors"))
        ft = meta["finetune"]
        if ft in ("lora", "qlora"):
            self.backbone.save_pretrained(str(path / "adapter"))
        elif ft == "full":
            self.backbone.save_pretrained(str(path / "backbone"), safe_serialization=True)
        self.tokenizer.save_pretrained(str(path))

    @classmethod
    def from_pretrained(cls, path_or_repo: str | Path, device: str = "auto", token: str | None = None,
                        merge: bool = True) -> "DecisionModel":
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        p = Path(path_or_repo)
        if not p.exists():
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(str(path_or_repo), token=token))
        meta = json.loads((p / "snapcall_config.json").read_text())
        dev = resolve_device(device)
        dtype = resolve_dtype("auto", dev) if meta["finetune"] != "full" else getattr(torch, meta.get("dtype", "float32"))
        trc = meta.get("trust_remote_code", False)
        if meta["finetune"] == "full":
            backbone = AutoModel.from_pretrained(str(p / "backbone"), dtype=dtype, trust_remote_code=trc)
        else:
            backbone = AutoModel.from_pretrained(meta["base_model"], dtype=dtype, trust_remote_code=trc)
            if (p / "adapter").exists():
                from peft import PeftModel
                backbone = PeftModel.from_pretrained(backbone, str(p / "adapter"))
                if merge:
                    backbone = backbone.merge_and_unload()
        tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=trc)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = cls(backbone, tok, meta)
        model.head.load_state_dict(load_file(str(p / "decision_head.safetensors")))
        model.backbone.to(dev)
        model.head.to(dev)
        model.device_ = dev
        model.eval()
        return model
