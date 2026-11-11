"""Hardware detection and compute-aware recommendations."""

from __future__ import annotations

import os
import platform
import shutil
from typing import Any

from .catalog import MODELS, PRESETS


def detect() -> dict[str, Any]:
    import psutil

    info: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "ram_free_gb": round(psutil.virtual_memory().available / 1e9, 1),
        "device": "cpu", "accelerator": "CPU", "vram_gb": 0.0, "gpus": [],
        "bf16": False, "qlora": False,
    }
    from .paths import snapcall_home
    try:
        info["disk_free_gb"] = round(shutil.disk_usage(snapcall_home()).free / 1e9, 1)
    except OSError:
        info["disk_free_gb"] = None
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                gpus.append({"name": p.name, "vram_gb": round(p.total_memory / 1e9, 1)})
            info.update(device="cuda", accelerator=gpus[0]["name"], gpus=gpus,
                        vram_gb=gpus[0]["vram_gb"], bf16=torch.cuda.is_bf16_supported())
            try:
                import bitsandbytes  # noqa: F401
                info["qlora"] = True
            except ImportError:
                pass
        elif torch.backends.mps.is_available():
            # Apple unified memory: Metal can use roughly 65-70% of system RAM.
            info.update(device="mps", accelerator="Apple Silicon (MPS)",
                        vram_gb=round(info["ram_gb"] * 0.65, 1), bf16=True)
    except ImportError:
        info["torch"] = None
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None
    return info


def estimate_memory_gb(params_b: float, finetune: str, batch_size: int = 8, max_length: int = 512,
                       bf16: bool = True, gradient_checkpointing: bool = False) -> float:
    """Rough peak training memory. Good to within ~30%; intended for fit badges."""
    bpp = 2 if bf16 else 4
    if finetune == "full":
        base = params_b * (bpp * 2 + 8)  # weights + grads + AdamW moments (fp32)
    elif finetune == "qlora":
        base = params_b * 0.65
    else:  # lora / head
        base = params_b * bpp
    # activations: measured ~1.6 MB/token for a 150M encoder in fp32 on MPS; scales ~sqrt(params)
    act = batch_size * max_length * 1.6e-3 * (max(params_b, 0.05) / 0.15) ** 0.5
    if finetune != "full":
        act *= 0.6  # frozen / low-precision backbone keeps fewer activations for backward
    if gradient_checkpointing:
        act *= 0.3
    return round(base + act + 0.8, 1)


def fit(required: float, available: float) -> str:
    if available <= 0:
        return "cpu"
    if required <= available * 0.8:
        return "fits"
    if required <= available:
        return "tight"
    return "no"


def recommend(info: dict[str, Any] | None = None) -> dict[str, Any]:
    info = info or detect()
    dev, vram = info["device"], info["vram_gb"]
    mem = vram if dev != "cpu" else info["ram_gb"] * 0.5
    models = []
    for m in MODELS:
        best = None
        for ft in ("full", "lora", "qlora"):
            if ft == "qlora" and not info.get("qlora"):
                continue
            if ft == "full" and m["params"] > 1.5:
                continue
            req = estimate_memory_gb(m["params"], ft, bf16=info.get("bf16", False))
            f = fit(req, mem)
            if f in ("fits", "tight"):
                best = {"finetune": ft, "memory_gb": req, "fit": f}
                break
        if best is None:
            ft = "qlora" if info.get("qlora") else "lora"
            best = {"finetune": ft, "memory_gb": estimate_memory_gb(m["params"], ft), "fit": "no"}
        if dev == "cpu" and m["params"] > 0.5 and best["fit"] != "no":
            best["fit"] = "slow"
        models.append({"id": m["id"], **best})

    if dev == "cuda" and vram >= 40:
        tier, pick = "datacenter", ["nimble-9b", "gpu-4b-qlora", "liquid-1b"]
    elif dev == "cuda" and vram >= 20:
        tier, pick = "workstation", ["gpu-4b-qlora", "nimble-9b", "liquid-1b"]
    elif dev == "cuda" and vram >= 10:
        tier, pick = "consumer-gpu", ["liquid-1b", "gpu-4b-qlora", "tiny-llm-lora"]
    elif dev in ("cuda", "mps"):
        tier, pick = "laptop-gpu", ["laptop-quick", "tiny-llm-lora", "multilingual"]
    else:
        tier, pick = "cpu", ["probe", "laptop-quick", "multilingual"]
    presets = [p for pid in pick for p in PRESETS if p["id"] == pid]
    tips = []
    if dev == "cpu":
        tips.append("No GPU detected: use encoders (ModernBERT/mmBERT) or the linear-probe preset.")
    if dev == "mps":
        tips.append("Apple Silicon: QLoRA (bitsandbytes) is CUDA-only, use LoRA in bf16 instead.")
    if dev == "cuda" and not info.get("qlora"):
        tips.append("Install bitsandbytes to unlock QLoRA: pip install 'snapcall[qlora]'.")
    if dev == "cuda" and not info.get("bf16"):
        tips.append("Your GPU has no bf16 support; fp16 will be used for mixed precision.")
    return {"tier": tier, "presets": presets, "models": models, "tips": tips, "memory_budget_gb": round(mem, 1)}
