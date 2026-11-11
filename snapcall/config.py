"""Training configuration and the hyperparameter explainer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset: str = Field(..., description="Name of a prepared dataset in SNAPCALL_HOME/datasets, or a path to a folder with train/validation/test .jsonl")
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    faker_augment: float = 0.0  # probability of swapping PII-like entities with Faker values


class TrainConfig(BaseModel):
    # identity
    run_name: Optional[str] = None
    base_model: str = "answerdotai/ModernBERT-base"
    arch: Literal["auto", "encoder", "causal"] = "auto"
    pooling: Literal["auto", "mean", "last"] = "auto"
    trust_remote_code: bool = False
    data: DataConfig

    # method
    method: Literal["sft", "rlcd", "sft+rlcd"] = "sft+rlcd"
    finetune: Literal["full", "lora", "qlora", "head"] = "full"

    # optimisation
    epochs: float = 3.0
    learning_rate: float = 5e-5
    head_learning_rate: float = 1e-3
    batch_size: int = 16
    grad_accum: int = 1
    max_length: int = 512
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    option_shuffle: bool = True
    precision: Literal["auto", "fp32", "bf16", "fp16"] = "auto"
    gradient_checkpointing: bool = False
    seed: int = 42

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: str = "all-linear"

    # RLCD
    rlcd_epochs: float = 1.0
    rlcd_learning_rate: Optional[float] = None
    rlcd_beta: float = 0.1
    rlcd_brier_weight: float = 1.0
    rlcd_consistency_weight: float = 0.5

    # eval / checkpoints
    eval_every: int = 0  # 0 = once per epoch
    log_every: int = 5
    early_stopping_patience: int = 0
    temperature_scaling: bool = True
    save_total_limit: int = 2
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    tensorboard: bool = True

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        p = Path(path)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            import yaml
            return cls.model_validate(yaml.safe_load(text))
        return cls.model_validate(json.loads(text))


# ---------------------------------------------------------------------------
# Explainer: every knob, what it does, and what happens when you turn it.
# Shared by `snapcall explain` and the UI side panel.
# ---------------------------------------------------------------------------

HPARAMS: dict[str, dict[str, Any]] = {
    "method": {
        "group": "Method", "label": "Training method",
        "short": "How the decision head is optimised.",
        "detail": "SFT trains on the correct option with cross-entropy. RLCD (Reinforcement Learning for "
                  "Calibrated Decisions) optimises a proper scoring rule (Brier + log-loss) under a KL "
                  "anchor to the SFT policy, plus option-order consistency, so confidences match observed "
                  "accuracy. 'sft+rlcd' runs SFT first, then an RLCD stage.",
        "typical": "sft+rlcd", "up": "", "down": "",
        "tip": "Use sft+rlcd whenever downstream code thresholds on confidence.",
    },
    "finetune": {
        "group": "Method", "label": "Fine-tuning mode",
        "short": "Which weights get updated.",
        "detail": "full = every weight (best for encoders under 1B). lora = low-rank adapters on a frozen "
                  "backbone (4-10x less memory). qlora = LoRA on a 4-bit backbone (CUDA only, biggest "
                  "models on smallest GPUs). head = train only the decision head (fast linear probe).",
        "typical": "full for encoders, lora for LLMs", "up": "", "down": "",
        "tip": "If you run out of memory, step down: full → lora → qlora.",
    },
    "learning_rate": {
        "group": "Optimisation", "label": "Learning rate",
        "short": "Step size of every optimiser update.",
        "detail": "The single most important knob. Full fine-tuning of encoders wants 2e-5 to 8e-5; LoRA "
                  "adapters tolerate 1e-4 to 3e-4 because they start at zero.",
        "typical": "5e-5 (full) · 2e-4 (LoRA)",
        "up": "Faster learning, risk of divergence and loss spikes.",
        "down": "Stabler but slower; may underfit in few epochs.",
        "tip": "Loss flat after warmup? Raise it 2x. Loss spikes or NaN? Halve it.",
    },
    "head_learning_rate": {
        "group": "Optimisation", "label": "Head learning rate",
        "short": "Separate LR for the freshly initialised decision head.",
        "detail": "The decision head starts random, so it can learn much faster than the pretrained backbone.",
        "typical": "1e-3", "up": "Head may overshoot early.", "down": "Slower start.", "tip": "",
    },
    "epochs": {
        "group": "Optimisation", "label": "Epochs",
        "short": "Full passes over the training set (SFT stage).",
        "detail": "Decision tasks usually converge in 2-4 epochs. Watch validation loss: when it rises while "
                  "train loss keeps falling you are overfitting.",
        "typical": "2-4", "up": "More fitting, more overfitting risk.", "down": "May underfit.",
        "tip": "Enable early stopping and set epochs generously.",
    },
    "batch_size": {
        "group": "Optimisation", "label": "Batch size",
        "short": "Examples per forward pass, per device.",
        "detail": "Bounded by memory. Effective batch = batch size x grad accumulation.",
        "typical": "8-32", "up": "Smoother gradients, more memory.", "down": "Less memory, noisier updates.",
        "tip": "Keep effective batch around 16-64; use grad accumulation when memory is tight.",
    },
    "grad_accum": {
        "group": "Optimisation", "label": "Gradient accumulation",
        "short": "Sum gradients over N batches before stepping.",
        "detail": "Simulates a bigger batch without the memory cost. Costs wall-clock time, not memory.",
        "typical": "1-8", "up": "Bigger effective batch.", "down": "", "tip": "",
    },
    "max_length": {
        "group": "Optimisation", "label": "Max sequence length",
        "short": "Token budget for state + question + options.",
        "detail": "Long states are truncated (options are never truncated). Memory grows linearly "
                  "(encoders: quadratically without flash attention) with length.",
        "typical": "256-1024", "up": "Sees more context, more memory.", "down": "Faster, may cut evidence.",
        "tip": "Check the dataset token histogram; pick the ~95th percentile.",
    },
    "warmup_ratio": {
        "group": "Optimisation", "label": "Warmup ratio",
        "short": "Fraction of steps spent ramping the LR up from zero.",
        "detail": "Prevents early large updates from wrecking pretrained features.",
        "typical": "0.03-0.1", "up": "Gentler start.", "down": "Faster start, spikier.", "tip": "",
    },
    "weight_decay": {
        "group": "Optimisation", "label": "Weight decay",
        "short": "L2-style regularisation in AdamW.",
        "detail": "Pulls weights toward zero; mild regulariser.", "typical": "0.0-0.1",
        "up": "More regularisation.", "down": "", "tip": "",
    },
    "scheduler": {
        "group": "Optimisation", "label": "LR schedule",
        "short": "How the learning rate decays after warmup.",
        "detail": "cosine decays smoothly to zero; linear decays linearly; constant keeps it flat.",
        "typical": "cosine", "up": "", "down": "", "tip": "",
    },
    "max_grad_norm": {
        "group": "Optimisation", "label": "Gradient clipping",
        "short": "Caps the gradient norm per step.",
        "detail": "Protects against rare exploding gradients. Watch the grad-norm chart.",
        "typical": "1.0", "up": "", "down": "Stricter clipping, slower learning.", "tip": "",
    },
    "label_smoothing": {
        "group": "Optimisation", "label": "Label smoothing",
        "short": "Spreads a little target mass over wrong options.",
        "detail": "Discourages over-confidence during SFT. RLCD usually makes this unnecessary.",
        "typical": "0.0-0.1", "up": "Less confident outputs.", "down": "", "tip": "",
    },
    "option_shuffle": {
        "group": "Optimisation", "label": "Shuffle option order",
        "short": "Randomise option order every time an example is seen.",
        "detail": "Removes position bias (picking the first option) so the decision depends on the option "
                  "meaning, not its slot.",
        "typical": "on", "up": "", "down": "", "tip": "Leave on unless order carries meaning.",
    },
    "precision": {
        "group": "Hardware", "label": "Precision",
        "short": "Numeric format for weights and activations.",
        "detail": "bf16 halves memory and is stable on Ampere+ GPUs and Apple Silicon. fp16 needs loss "
                  "scaling. fp32 is safest and slowest. auto picks bf16 when supported.",
        "typical": "auto", "up": "", "down": "", "tip": "",
    },
    "gradient_checkpointing": {
        "group": "Hardware", "label": "Gradient checkpointing",
        "short": "Recompute activations in the backward pass.",
        "detail": "Saves 30-60% activation memory at ~25% extra compute.",
        "typical": "off (on for >1B models)", "up": "", "down": "", "tip": "",
    },
    "lora_r": {
        "group": "LoRA", "label": "LoRA rank (r)",
        "short": "Width of the low-rank adapter.",
        "detail": "Higher rank means more trainable capacity and memory. Decision tasks rarely need more than 32.",
        "typical": "8-32", "up": "More capacity.", "down": "Fewer params, faster.", "tip": "",
    },
    "lora_alpha": {
        "group": "LoRA", "label": "LoRA alpha",
        "short": "Scaling of the adapter output (alpha / r).",
        "detail": "Rule of thumb: alpha = 2 x r.", "typical": "2 x r", "up": "", "down": "", "tip": "",
    },
    "lora_dropout": {
        "group": "LoRA", "label": "LoRA dropout",
        "short": "Dropout inside the adapter.", "detail": "Light regularisation for small datasets.",
        "typical": "0.05", "up": "", "down": "", "tip": "",
    },
    "lora_targets": {
        "group": "LoRA", "label": "LoRA target modules",
        "short": "Which linear layers get adapters.",
        "detail": "'all-linear' adapts every linear layer (best quality). Or a comma list like 'q_proj,v_proj'.",
        "typical": "all-linear", "up": "", "down": "", "tip": "",
    },
    "rlcd_epochs": {
        "group": "RLCD", "label": "RLCD epochs",
        "short": "Length of the calibration stage.",
        "detail": "RLCD runs after SFT. One epoch is usually enough to fix confidence without moving accuracy.",
        "typical": "0.5-1", "up": "", "down": "", "tip": "",
    },
    "rlcd_beta": {
        "group": "RLCD", "label": "KL anchor (beta)",
        "short": "How strongly RLCD stays close to the SFT policy.",
        "detail": "KL(policy || reference). The reference is the frozen SFT model. Higher beta keeps "
                  "decisions stable and only reshapes confidence.",
        "typical": "0.05-0.2", "up": "Safer, smaller changes.", "down": "Freer, may drift.", "tip": "",
    },
    "rlcd_brier_weight": {
        "group": "RLCD", "label": "Brier reward weight",
        "short": "Weight of the Brier term in the proper-scoring reward.",
        "detail": "Brier is a strictly proper scoring rule that punishes confident mistakes quadratically; "
                  "combined with log-loss it gives calibrated probabilities.",
        "typical": "1.0", "up": "", "down": "", "tip": "",
    },
    "rlcd_consistency_weight": {
        "group": "RLCD", "label": "Order consistency weight",
        "short": "Penalty when re-ordering options changes the answer.",
        "detail": "Each example is scored under two option orders; their distributions are pulled together "
                  "with a symmetric KL.",
        "typical": "0.5", "up": "", "down": "", "tip": "",
    },
    "temperature_scaling": {
        "group": "Evaluation", "label": "Temperature scaling",
        "short": "Fit one temperature on validation after training.",
        "detail": "Post-hoc calibration that divides logits by T. Never changes the argmax, only confidence.",
        "typical": "on", "up": "", "down": "", "tip": "",
    },
    "eval_every": {
        "group": "Evaluation", "label": "Evaluate every N steps",
        "short": "0 means once per epoch.", "detail": "", "typical": "0", "up": "", "down": "", "tip": "",
    },
    "early_stopping_patience": {
        "group": "Evaluation", "label": "Early stopping patience",
        "short": "Stop after N evaluations without val-loss improvement.",
        "detail": "0 disables it. The best checkpoint is always kept.", "typical": "0-3",
        "up": "", "down": "", "tip": "",
    },
    "seed": {
        "group": "Evaluation", "label": "Seed", "short": "Makes runs reproducible.", "detail": "",
        "typical": "42", "up": "", "down": "", "tip": "",
    },
}
