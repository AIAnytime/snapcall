# Snapcall

**Build better decision models.**
An open-source toolkit for training, evaluating and deploying Jev-style System One decision models.

`Python` · `Open Source` · `CLI + Dashboard` · **Train. Evaluate. Deploy.**

---

Decision models (TypeSafe's Jev, Cloudflare's Clef, Liquid's d1, Convai's Laya, Bespoke Nimble…) don't
generate text. You give them a **state** (text or JSON) and **typed questions**, and they return a
**calibrated probability for every option**, in one forward pass. They can't return an option you didn't
offer, and when they say 0.8 they should be right about 80% of the time.

Snapcall lets you build your own: pick a base model, bring data (or generate it), train with SFT + **RLCD**
calibration, watch it live, evaluate calibration properly, push to the Hub with a generated model card,
and serve it behind a System One compatible `/v1/systemone` API.

```
state ─┐
       ├─► backbone (encoder or LLM) ─► pool each option's span ─► decision head ─► softmax over options
typed  ┘                                                                       (closed set, calibrated)
question + options
```

## Features

| | |
|---|---|
| **Base models** | ModernBERT, mmBERT (multilingual), DeBERTa, SmolLM2, LFM2/LFM2.5, Qwen3, Qwen3.5, or any HF id. Continue from open decision models: Laya, d1, Tev1, Clef (experimental). |
| **Presets** | Laptop quickstart, multilingual, tiny LLM + LoRA, Liquid 1.2B, 4B QLoRA, Nimble-style 9B, linear probe. |
| **Compute-aware** | Detects CUDA / Apple Silicon / CPU, estimates peak memory per model and mode, recommends presets, picks a micro-batch that fits, and recovers from OOM automatically without changing the effective batch. |
| **Methods** | Full fine-tune, LoRA, QLoRA (CUDA), head-only probe. SFT, RLCD, or SFT → RLCD. Temperature scaling. |
| **Hyperparameters with explainers** | Every knob documented: what it does, typical range, what happens when you raise or lower it (`snapcall explain`, UI side panel). |
| **Data** | Hugging Face (presets + any dataset with column mapping), bring your own JSONL/JSON/CSV, LLM synthetic data (Anthropic or any OpenAI-compatible endpoint, Ollama included), Faker templates (no key needed). Stratified train/validation/test splits. |
| **Faker** | 1) template generation, 2) PII-safe augmentation at train time (swaps emails, phones, amounts, IDs), 3) persona seeds that diversify LLM generation. |
| **Observability** | Live loss, accuracy, ECE, Brier, LR, grad norm, throughput, memory, RLCD terms, smoothing like TensorBoard. TensorBoard event files are written too. Reliability diagrams, confusion matrices, mistake browser. |
| **Ship** | Model metadata, playground, re-evaluation on any split, push to Hugging Face with an auto-generated model card, `/v1/systemone` server. |
| **Cleanup** | Delete checkpoints, TensorBoard events, failed runs and Hugging Face cache entries, with sizes shown. |

## Install

```bash
pip install snapcall
snapcall ui
```

Extras: `pip install "snapcall[anthropic]"` adds the Anthropic SDK for synthetic data, and
`pip install "snapcall[qlora]"` adds bitsandbytes for QLoRA on CUDA.

From source (the dashboard has to be built once):

```bash
git clone https://github.com/AIAnytime/snapcall && cd snapcall
cd ui && pnpm install && pnpm build && cd ..   # builds into snapcall/server/static
pip install -e ".[anthropic]"
```

## Quickstart

### Dashboard

```bash
snapcall ui            # http://127.0.0.1:7860
```

**New run** → pick a preset or model (each one shows whether it fits your machine) → pick or create a dataset →
tune hyperparameters with the explainer panel → **Start training**. The run page streams metrics and logs, then
gives you Evaluation, Model metadata, Test (playground), Publish (Hub + model card) and Files tabs.

### CLI

```bash
snapcall doctor                                   # hardware + recommendations
snapcall data hf banking77                        # import a preset HF dataset
snapcall train -p laptop-quick -d banking77       # train with live progress
snapcall runs                                     # list runs
snapcall predict <run> -s "I was charged twice for my coffee"
snapcall eval <run> --split test
snapcall push <run> --repo you/banking77-snapcall
snapcall serve <run> --port 8000                  # System One compatible API
snapcall clean --checkpoints --failed
```

More: `snapcall models`, `snapcall presets`, `snapcall explain [param]`, `snapcall init`, `snapcall logs <run> -f`,
`snapcall stop <run>`, `snapcall card <run>`, `snapcall tensorboard`, `snapcall data {list,show,upload,faker,synth}`.

### Python

```python
from snapcall import Trainer, TrainConfig, DataConfig, DecisionModel

cfg = TrainConfig(
    base_model="answerdotai/ModernBERT-base",
    data=DataConfig(dataset="banking77"),
    method="sft+rlcd", finetune="full", epochs=3, learning_rate=5e-5,
)
summary = Trainer(cfg, run_dir="runs/banking77").train()

model = DecisionModel.from_pretrained("runs/banking77/model")   # or a Hub repo id
model.decide(
    state="My card hasn't arrived yet",
    questions={
        "intent": {"type": "choice", "instructions": "Which banking intent?",
                   "criteria": {"card_arrival": "Card not delivered", "lost_or_stolen_card": "Card lost"}},
        "upset": {"type": "noul", "instructions": "Is the customer upset?"},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
    },
)
```

## Data format

One decision per JSONL line (System One shape):

```json
{"state": "I was charged twice", "question": {"type": "choice", "instructions": "Route to a team",
  "criteria": {"billing": "Charges, refunds", "shipping": "Delivery"}}, "answer": "billing"}
```

Multi-question rows (`"questions": {...}, "answers": {...}`) are flattened automatically. `score` answers are
level indices, `noul` answers are booleans. Optional `"soft": {"billing": 0.7, "shipping": 0.3}` trains on
annotator distributions. CSV/TSV/JSON tables can be mapped with a text column and a label column.

Question types: **choice** (2–255 named options), **score** (2–10 ordered levels, returns the expected level),
**noul** (yes/no, returns P(yes)).

## How training works

1. **Typed decision head.** The state, question and options are rendered into one sequence. Each option's
   token span is pooled (mean for encoders, last token for decoders) and scored by a small head. Softmax over
   options gives a closed-set distribution. Options are shuffled every time an example is seen, so the model
   learns what an option means rather than its position.
2. **SFT.** Cross-entropy on the correct option (or soft targets), AdamW with warmup and a cosine/linear schedule,
   gradient accumulation, clipping, per-epoch evaluation, best-checkpoint restore, early stopping.
3. **RLCD (Reinforcement Learning for Calibrated Decisions).** TypeSafe has only described RLCD at a high level;
   Snapcall's version is an open, KL-regularised **proper-scoring objective**: maximise Brier + log-score
   rewards, plus a `β·KL(π ‖ π_SFT)` anchor to the frozen SFT policy, plus an **option-order consistency**
   term (symmetric KL between two random option orderings). It keeps decisions and reshapes confidence.
4. **Temperature scaling** on the validation split (never changes the argmax).
5. **Evaluation**: accuracy, macro F1, 15-bin ECE, Brier, NLL, average confidence, score MAE, reliability
   bins, confusion matrix.

## Run on a cloud GPU (RunPod)

```bash
pip install snapcall
snapcall ui --auto-stop 2
```

On a RunPod pod, `snapcall ui` detects the pod and:
- prints the proxy URL;
- password-protects the dashboard;
- keeps runs and models on `/workspace`;
- with `--auto-stop`, stops the pod after N hours.

See **[deploy/runpod/README.md](deploy/runpod/README.md)** for pod settings and a $2 session plan.

Anywhere else, expose the dashboard beyond localhost with `snapcall ui --host 0.0.0.0`. A password is
generated automatically, or set your own with `--password` or `SNAPCALL_PASSWORD`.

## Workspace

Everything lives in `~/.snapcall` (override with `SNAPCALL_HOME`): `datasets/`, `runs/<id>/` (`config.json`,
`status.json`, `metrics.jsonl`, `train.log`, `tensorboard/`, `checkpoints/`, `eval.json`, `model/`), and
`settings.json` (API keys, chmod 600; `HF_TOKEN`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`
env vars work as fallbacks). Runs execute in a background process, so the UI and CLI can both follow them,
and closing the browser doesn't stop training.

## Serving

```bash
snapcall serve <run-or-hub-repo> --port 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Where is my new card?",
  "questions": {"intent": {"type": "choice", "instructions": "Which intent?",
                "criteria": {"card_arrival": "Card not delivered", "activate_my_card": "Activation"}}}}'
```

The dashboard server exposes the same endpoint at `/v1/systemone` (`"model": "<run id>"`, or omit it for the
latest model).

## Limitations

- Text states only (no images or `box` questions yet).
- QLoRA needs CUDA + bitsandbytes. On Apple Silicon use LoRA in bf16.
- Qwen3.5, d1, Laya and Clef checkpoints are marked experimental: they may need a newer `transformers` or
  `trust_remote_code`.
- Large option sets (50+) make sequences long; the memory planner shrinks micro-batches automatically, but
  training is slower.

---

Built by **[AI Anytime](https://www.youtube.com/@AIAnytime)**. Apache-2.0.
