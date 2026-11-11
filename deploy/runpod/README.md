# Snapcall on RunPod

```bash
pip install snapcall
snapcall ui --auto-stop 2
```

That's the whole setup. Details and a $2 session plan follow.

## 1. Create the pod

RunPod console → **Pods → Deploy**:

| Setting | Value |
|---|---|
| Cloud | **Community** (about 2-3x cheaper than Secure) |
| GPU | **RTX 4090** 24 GB (~$0.34/h). Cheaper: RTX A5000 24 GB (~$0.16/h) or RTX 3090 (~$0.22/h) |
| Template | **RunPod PyTorch** (official) |
| Container disk | 30 GB |
| Volume disk | 30 GB, mounted at `/workspace` |
| Expose HTTP ports | `8888,7860` (Jupyter, Snapcall). Add `6006` for TensorBoard |

Prices are Community Cloud rates from runpod.io/pricing (November 2026) and change often.

## 2. Install and launch

Open the pod's **Web Terminal** or **JupyterLab → Terminal**:

```bash
pip install snapcall
snapcall ui --auto-stop 2
```

```
╭────────────────────────────────────────────────────────────╮
│ Snapcall v0.1.0                                            │
│ Dashboard  https://<POD_ID>-7860.proxy.runpod.net          │
│ Login      any username · password <generated>             │
│ Workspace  /workspace/.snapcall                            │
│ Auto-stop  pod stops in 2 h                                │
╰────────────────────────────────────────────────────────────╯
```

Open the link and log in. Snapcall detects the pod and:
- Listens on `0.0.0.0` and prints the RunPod proxy URL.
- Protects the dashboard with a password. Proxy URLs are public, and your HF token lives in Settings. Pass `--password` to choose your own; API clients send `Authorization: Bearer <password>`.
- Stores runs, datasets and settings in `/workspace/.snapcall`, and model downloads in `/workspace/.cache/huggingface`, so nothing fills the container disk and everything survives a restart.
- With `--auto-stop 2`, it runs `runpodctl pod stop` after 2 hours to protect a small balance.

To keep the dashboard running after you close the terminal:

```bash
nohup snapcall ui --auto-stop 2 > /workspace/snapcall.log 2>&1 &
tail -f /workspace/snapcall.log     # shows the URL and password
```

PyTorch is already in the image, so pip installs only Snapcall and its light dependencies. To add QLoRA
(bitsandbytes), use `pip install "snapcall[qlora]"`.

## 3. A $2 session plan

Rough estimates for an RTX 4090 at ~$0.34/h (~5.8 GPU-hours for $2):

| Step | What | Time | Cost |
|---|---|---|---|
| Setup | pod start + `pip install` | ~5 min | ~$0.03 |
| Run 1 | `laptop-quick` (ModernBERT) on **Banking77** (77 intents) | ~10 min | ~$0.06 |
| Run 2 | `tiny-llm-lora` (Qwen3-0.6B LoRA) on the same data | ~15 min | ~$0.09 |
| Run 3 | `gpu-4b-qlora` (Qwen3-4B QLoRA, needs `snapcall[qlora]`) | ~40 min | ~$0.23 |
| Ship | evaluate, push the best one to the Hub | ~10 min | ~$0.06 |

That's about $0.50 total. In the dashboard: **Datasets → New dataset → Hugging Face → Banking77**, then
**New run**. The 24 GB GPU unlocks the bigger presets automatically.

From the terminal instead:

```bash
snapcall doctor
snapcall data hf banking77
snapcall train -p laptop-quick -d banking77 --set max_length=768
snapcall push <run-id> --repo <you>/banking77-snapcall
```

## 4. Stop paying

- **Stop** halts GPU billing. A 30 GB volume still costs ~$0.20/day while stopped.
- **Terminate** ends all charges but deletes the volume, so push models to the Hub first.
- Pods without a network volume are **terminated when your balance hits $0**, and their data can't be recovered.
