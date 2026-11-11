"""Base models, open decision models, and training presets."""

from __future__ import annotations

from typing import Any

# params in billions. `support`: stable = tested path, experimental = may need newer
# transformers / trust_remote_code. `kind`: base = generic backbone, decision = an existing
# System One model you can continue training from.
MODELS: list[dict[str, Any]] = [
    # --- encoders (Laya-style, fastest) ---
    {"id": "answerdotai/ModernBERT-base", "name": "ModernBERT base", "params": 0.15, "arch": "encoder",
     "kind": "base", "langs": "en", "context": 8192, "license": "apache-2.0", "support": "stable",
     "family": "ModernBERT", "notes": "Fast English encoder. Great first model; trains on a laptop."},
    {"id": "answerdotai/ModernBERT-large", "name": "ModernBERT large", "params": 0.40, "arch": "encoder",
     "kind": "base", "langs": "en", "context": 8192, "license": "apache-2.0", "support": "stable",
     "family": "ModernBERT", "notes": "Same backbone family as Laya English (421M)."},
    {"id": "jhu-clsp/mmBERT-small", "name": "mmBERT small", "params": 0.14, "arch": "encoder",
     "kind": "base", "langs": "multilingual (1800+)", "context": 8192, "license": "mit", "support": "stable",
     "family": "mmBERT", "notes": "Tiny multilingual encoder."},
    {"id": "jhu-clsp/mmBERT-base", "name": "mmBERT base", "params": 0.31, "arch": "encoder",
     "kind": "base", "langs": "multilingual (1800+)", "context": 8192, "license": "mit", "support": "stable",
     "family": "mmBERT", "notes": "Backbone family of Laya multilingual (322M)."},
    {"id": "microsoft/deberta-v3-base", "name": "DeBERTa v3 base", "params": 0.18, "arch": "encoder",
     "kind": "base", "langs": "en", "context": 512, "license": "mit", "support": "stable",
     "family": "DeBERTa", "notes": "Strong classic NLU encoder (used by open-jev-deberta)."},
    # --- small causal LMs ---
    {"id": "HuggingFaceTB/SmolLM2-135M", "name": "SmolLM2 135M", "params": 0.135, "arch": "causal",
     "kind": "base", "langs": "en", "context": 8192, "license": "apache-2.0", "support": "stable",
     "family": "SmolLM", "notes": "Smallest decoder; good for CPU smoke tests."},
    {"id": "HuggingFaceTB/SmolLM2-360M", "name": "SmolLM2 360M", "params": 0.36, "arch": "causal",
     "kind": "base", "langs": "en", "context": 8192, "license": "apache-2.0", "support": "stable",
     "family": "SmolLM", "notes": ""},
    {"id": "LiquidAI/LFM2-350M", "name": "LFM2 350M", "params": 0.35, "arch": "causal",
     "kind": "base", "langs": "multilingual (9)", "context": 32768, "license": "lfm1.0", "support": "stable",
     "family": "LFM2", "notes": "Liquid hybrid architecture, same lineage as d1. Very fast on CPU."},
    {"id": "LiquidAI/LFM2.5-1.2B-Base", "name": "LFM2.5 1.2B Base", "params": 1.17, "arch": "causal",
     "kind": "base", "langs": "multilingual (9)", "context": 32768, "license": "lfm1.0", "support": "stable",
     "family": "LFM2", "notes": "Base (not instruct) checkpoint, ideal for decision heads."},
    {"id": "Qwen/Qwen3-0.6B", "name": "Qwen3 0.6B", "params": 0.75, "arch": "causal",
     "kind": "base", "langs": "multilingual (119)", "context": 32768, "license": "apache-2.0", "support": "stable",
     "family": "Qwen3", "notes": ""},
    {"id": "Qwen/Qwen3-1.7B", "name": "Qwen3 1.7B", "params": 2.03, "arch": "causal",
     "kind": "base", "langs": "multilingual (119)", "context": 32768, "license": "apache-2.0", "support": "stable",
     "family": "Qwen3", "notes": ""},
    {"id": "Qwen/Qwen3-4B", "name": "Qwen3 4B", "params": 4.02, "arch": "causal",
     "kind": "base", "langs": "multilingual (119)", "context": 32768, "license": "apache-2.0", "support": "stable",
     "family": "Qwen3", "notes": ""},
    {"id": "Qwen/Qwen3.5-4B-Base", "name": "Qwen3.5 4B Base", "params": 4.66, "arch": "causal",
     "kind": "base", "langs": "multilingual", "context": 32768, "license": "apache-2.0", "support": "experimental",
     "family": "Qwen3.5", "notes": "Base of Tev1, Decider-4B, kev-4B and JevK5. Vision-language checkpoint; text path is used."},
    {"id": "Qwen/Qwen3.5-9B-Base", "name": "Qwen3.5 9B Base", "params": 9.65, "arch": "causal",
     "kind": "base", "langs": "multilingual", "context": 32768, "license": "apache-2.0", "support": "experimental",
     "family": "Qwen3.5", "notes": "Base of Bespoke Nimble 9B and kev-9B."},
    # --- existing decision models ---
    {"id": "convaiinnovations/laya", "name": "Laya (Convai)", "params": 0.42, "arch": "encoder",
     "kind": "decision", "langs": "en / multilingual", "context": 8192, "license": "apache-2.0",
     "support": "experimental", "family": "Laya", "trust_remote_code": True,
     "notes": "Open encoder decision model on ModernBERT/mmBERT. Custom architecture."},
    {"id": "LiquidAI/d1-omni-600M", "name": "d1 omni 600M (Liquid)", "params": 0.59, "arch": "causal",
     "kind": "decision", "langs": "multilingual", "context": 32768, "license": "lfm1.0",
     "support": "experimental", "family": "d1", "trust_remote_code": True,
     "notes": "Multilingual multimodal System One model. Text path only in Snapcall."},
    {"id": "LiquidAI/d1-3B", "name": "d1 3B (Liquid)", "params": 3.12, "arch": "causal",
     "kind": "decision", "langs": "multilingual", "context": 32768, "license": "lfm1.0",
     "support": "experimental", "family": "d1",
     "notes": "Built on LFM2.5-VL-3B. Best open decision score under 10B (self-reported)."},
    {"id": "togethercomputer/Tev1-4B-experimental", "name": "Tev1 4B (Together)", "params": 4.66, "arch": "causal",
     "kind": "decision", "langs": "multilingual", "context": 32768, "license": "tbd",
     "support": "experimental", "family": "Qwen3.5", "notes": "Weights licence still being finalised."},
    {"id": "Cloudflare/clef-flash", "name": "Clef Flash 9B (Cloudflare)", "params": 9.41, "arch": "causal",
     "kind": "decision", "langs": "multilingual", "context": 32768, "license": "apache-2.0",
     "support": "experimental", "family": "Clef", "notes": "Qwen3.5-based. Needs a 24 GB+ GPU with QLoRA."},
    {"id": "Cloudflare/clef", "name": "Clef 27B (Cloudflare)", "params": 27.36, "arch": "causal",
     "kind": "decision", "langs": "multilingual", "context": 32768, "license": "apache-2.0",
     "support": "experimental", "family": "Clef", "notes": "Needs an 80 GB GPU with QLoRA."},
]

# Reference only (adapters / closed); shown in docs, not trainable directly.
REFERENCE_MODELS = [
    {"id": "jev (TypeSafe)", "open": False, "notes": "Closed, hosted. The original System One model."},
    {"id": "bespokelabs/Bespoke-Nimble-9B-v2", "open": True, "notes": "PEFT adapter on Qwen3.5-9B-Base. Pick Qwen3.5 9B Base to reproduce."},
    {"id": "jaredpalmer/kev-4b", "open": True, "notes": "PEFT adapter on Qwen3.5-4B-Base."},
]

PRESETS: list[dict[str, Any]] = [
    {"id": "laptop-quick", "name": "Laptop quickstart", "badge": "CPU / Apple Silicon",
     "description": "ModernBERT-base encoder, full fine-tune. Minutes on a MacBook.",
     "config": {"base_model": "answerdotai/ModernBERT-base", "finetune": "full", "method": "sft+rlcd",
                "epochs": 3, "learning_rate": 5e-5, "batch_size": 16, "max_length": 384, "rlcd_epochs": 0.5}},
    {"id": "multilingual", "name": "Multilingual", "badge": "d1-style",
     "description": "mmBERT-base encoder for 100+ languages, Laya-multilingual recipe.",
     "config": {"base_model": "jhu-clsp/mmBERT-base", "finetune": "full", "method": "sft+rlcd",
                "epochs": 3, "learning_rate": 4e-5, "batch_size": 16, "max_length": 512}},
    {"id": "tiny-llm-lora", "name": "Tiny LLM + LoRA", "badge": "8 GB",
     "description": "Qwen3-0.6B with LoRA r=16. Reasoning-ish states, still fast.",
     "config": {"base_model": "Qwen/Qwen3-0.6B", "finetune": "lora", "method": "sft+rlcd",
                "epochs": 2, "learning_rate": 2e-4, "batch_size": 8, "max_length": 768}},
    {"id": "liquid-1b", "name": "Liquid 1.2B", "badge": "12 GB",
     "description": "LFM2.5-1.2B-Base + LoRA, the lineage behind Liquid d1.",
     "config": {"base_model": "LiquidAI/LFM2.5-1.2B-Base", "finetune": "lora", "method": "sft+rlcd",
                "epochs": 2, "learning_rate": 2e-4, "batch_size": 8, "max_length": 1024,
                "gradient_checkpointing": True}},
    {"id": "gpu-4b-qlora", "name": "4B QLoRA", "badge": "CUDA 12 GB+",
     "description": "Qwen3-4B in 4-bit with LoRA. Strong accuracy on a single consumer GPU.",
     "config": {"base_model": "Qwen/Qwen3-4B", "finetune": "qlora", "method": "sft+rlcd",
                "epochs": 2, "learning_rate": 1.5e-4, "batch_size": 4, "grad_accum": 4, "max_length": 1024,
                "gradient_checkpointing": True}},
    {"id": "nimble-9b", "name": "Nimble-style 9B", "badge": "CUDA 24 GB+",
     "description": "Qwen3.5-9B-Base QLoRA, the Bespoke Nimble recipe.",
     "config": {"base_model": "Qwen/Qwen3.5-9B-Base", "finetune": "qlora", "method": "sft+rlcd",
                "epochs": 2, "learning_rate": 1e-4, "batch_size": 2, "grad_accum": 8, "max_length": 2048,
                "gradient_checkpointing": True}},
    {"id": "probe", "name": "Linear probe", "badge": "Seconds",
     "description": "Freeze the backbone, train only the decision head. Instant baseline.",
     "config": {"base_model": "answerdotai/ModernBERT-base", "finetune": "head", "method": "sft",
                "epochs": 3, "head_learning_rate": 3e-3, "batch_size": 32, "max_length": 256}},
]


def get_model(model_id: str) -> dict[str, Any] | None:
    return next((m for m in MODELS if m["id"] == model_id), None)


def get_preset(preset_id: str) -> dict[str, Any] | None:
    return next((p for p in PRESETS if p["id"] == preset_id), None)
