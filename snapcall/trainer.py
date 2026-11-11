"""Trainer: SFT -> RLCD -> temperature scaling -> evaluation -> save.

    from snapcall import Trainer, TrainConfig, DataConfig
    cfg = TrainConfig(base_model="answerdotai/ModernBERT-base", data=DataConfig(dataset="banking77"))
    result = Trainer(cfg, run_dir="runs/my-run").train()
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import data as data_store
from .config import TrainConfig
from .metrics import compute as compute_metrics
from .metrics import fit_temperature
from .modeling import DecisionModel, resolve_device, resolve_dtype
from .schema import Example
from .tracking import RunLogger


class StopTraining(Exception):
    pass


def _lr_lambda(kind: str, warmup: int, total: int):
    def f(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        prog = (step - warmup) / max(1, total - warmup)
        if kind == "linear":
            return max(0.0, 1.0 - prog)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return f


class Trainer:
    def __init__(self, config: TrainConfig, run_dir: str | Path):
        self.cfg = config
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        config.save(self.dir / "config.json")
        self.log = RunLogger(self.dir, tensorboard=config.tensorboard)
        self.rng = random.Random(config.seed)
        self.global_step = 0
        self.total_steps = 1
        self.t_start = time.time()
        self.step_times: list[float] = []
        self.micro_bs: int | None = None

    # ---------------------------------------------------------------- helpers
    def say(self, msg: str) -> None:
        print(f"[snapcall] {time.strftime('%H:%M:%S')} {msg}", flush=True)

    def _targets(self, exs: list[Example], encs, O: int) -> torch.Tensor:
        """Target distribution in *presented* option order, [B, O]."""
        t = torch.zeros(len(exs), O)
        eps = self.cfg.label_smoothing
        for b, (ex, e) in enumerate(zip(exs, encs)):
            n = len(e.order)
            if ex.soft:
                for pos, orig in enumerate(e.order):
                    t[b, pos] = ex.soft[orig]
            else:
                t[b, e.order.index(ex.label)] = 1.0
            if eps > 0:
                t[b, :n] = t[b, :n] * (1 - eps) + eps / n
        return t.to(self.model.device)

    def _orig_index(self, encs, O: int) -> torch.Tensor:
        idx = torch.arange(O).repeat(len(encs), 1)
        for b, e in enumerate(encs):
            idx[b, :len(e.order)] = torch.tensor(e.order)
        return idx.to(self.model.device)

    def _autocast(self):
        dev = self.model.device
        if dev.type == "cuda" and self.cfg.finetune == "full":
            dt = resolve_dtype(self.cfg.precision, dev)
            if dt != torch.float32:
                return torch.autocast("cuda", dtype=dt)
        return torch.autocast(dev.type, enabled=False)

    def _tick(self, t0: float, stage: str, epoch: float) -> None:
        self.global_step += 1
        self.step_times.append(time.time() - t0)
        recent = self.step_times[-50:]
        eta = (self.total_steps - self.global_step) * (sum(recent) / len(recent))
        if self.global_step % max(1, self.cfg.log_every) == 0 or self.global_step == self.total_steps:
            self.log.status(state="running", stage=stage, step=self.global_step, total_steps=self.total_steps,
                            epoch=round(epoch, 3), progress=round(self.global_step / self.total_steps, 4),
                            eta_s=round(eta, 1), elapsed_s=round(time.time() - self.t_start, 1))
        if self.log.stop_requested():
            raise StopTraining()

    def _mem(self) -> float:
        dev = self.model.device
        if dev.type == "cuda":
            return torch.cuda.max_memory_allocated() / 1e9
        if dev.type == "mps":
            try:
                return torch.mps.driver_allocated_memory() / 1e9
            except Exception:  # noqa: BLE001
                return 0.0
        return 0.0

    def _batches(self, exs: list[Example], bs: int, shuffle: bool = True):
        idx = list(range(len(exs)))
        if shuffle:
            self.rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield idx[i:i + bs]

    def _optimizer(self, lr: float, head_lr: float, steps: int):
        decay, no_decay, head = [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("head."):
                head.append(p)
            elif p.ndim < 2 or "norm" in n.lower() or "bias" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [{"params": decay, "lr": lr, "weight_decay": self.cfg.weight_decay},
                  {"params": no_decay, "lr": lr, "weight_decay": 0.0},
                  {"params": head, "lr": head_lr, "weight_decay": 0.0}]
        opt = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.999), eps=1e-8)
        warm = int(self.cfg.warmup_ratio * steps)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(self.cfg.scheduler, warm, steps))
        return opt, sched

    def _apply_augment(self, exs: list[Example]) -> list[Example]:
        if self.cfg.data.faker_augment <= 0:
            return exs
        from .data.faker_gen import Augmenter
        aug = self.augmenter = getattr(self, "augmenter", Augmenter(self.cfg.data.faker_augment, seed=self.cfg.seed))
        return [Example(aug(e.state), e.question, e.label, e.soft) for e in exs]

    # ----------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, exs: list[Example], details: bool = False, temperature: float | None = None,
                 return_logits: bool = False):
        self.model.eval()
        T = temperature if temperature is not None else self.model.temperature
        probs, logit_rows = [], []
        bs = max(1, (self.micro_bs or self.cfg.batch_size) * 2)
        for i in range(0, len(exs), bs):
            chunk = exs[i:i + bs]
            b, encs = self.model.batch(chunk, self.cfg.max_length)
            with self._autocast():
                logits = self.model(**b).float()
            p = torch.softmax(logits / T, -1).cpu()
            lc = logits.cpu()
            for j, e in enumerate(encs):
                row_p, row_l = [0.0] * len(e.order), [0.0] * len(e.order)
                for pos, orig in enumerate(e.order):
                    row_p[orig] = float(p[j, pos])
                    row_l[orig] = float(lc[j, pos])
                probs.append(row_p)
                logit_rows.append(row_l)
        m = compute_metrics(exs, probs, with_details=details)
        self.model.train()
        return (m, logit_rows) if return_logits else m

    # ---------------------------------------------------------------- stages
    def _sft(self, train: list[Example], val: list[Example]) -> None:
        cfg = self.cfg
        steps_per_epoch = math.ceil(len(train) / cfg.batch_size / cfg.grad_accum)
        steps = max(1, int(steps_per_epoch * cfg.epochs))
        opt, sched = self._optimizer(cfg.learning_rate, cfg.head_learning_rate, steps)
        eval_every = cfg.eval_every or steps_per_epoch
        best_nll, bad, step = float("inf"), 0, 0
        self.best_state = None
        self.model.train()
        epoch = 0.0
        self.say(f"SFT: {len(train)} examples, {steps} optimizer steps")
        while step < steps:
            batch_iter = list(self._batches(train, cfg.batch_size))
            for bi in range(0, len(batch_iter), cfg.grad_accum):
                t0 = time.time()
                group = batch_iter[bi:bi + cfg.grad_accum]
                acc = self._accumulate(opt, [i for g in group for i in g], lambda idx, w: self._sft_micro(train, idx, w))
                loss_sum, n_tok, n_ex = acc["loss"], acc["n_tok"], acc["n_ex"]
                gn = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                                    cfg.max_grad_norm)
                opt.step()
                sched.step()
                step += 1
                epoch = step / steps_per_epoch
                dt = time.time() - t0
                if step % max(1, cfg.log_every) == 0 or step == 1:
                    self.log.log(self.global_step + 1, "train", {
                        "loss": loss_sum, "lr": sched.get_last_lr()[0], "grad_norm": float(gn),
                        "tokens_per_s": n_tok / max(dt, 1e-6), "samples_per_s": n_ex / max(dt, 1e-6),
                        "memory_gb": self._mem(), "epoch": epoch})
                self._tick(t0, "sft", epoch)
                if step % eval_every == 0 or step == steps:
                    m = self.evaluate(val)
                    self.log.log(self.global_step, "val", {**m, "epoch": epoch})
                    self.say(f"step {self.global_step} val acc={m['accuracy']:.4f} nll={m['nll']:.4f} ece={m['ece']:.4f}")
                    if m["nll"] < best_nll - 1e-4:
                        best_nll, bad = m["nll"], 0
                        self.best_state = self._snapshot()
                        self.log.status(best={"step": self.global_step, **{k: m[k] for k in ("accuracy", "nll", "ece", "brier", "macro_f1")}})
                        self._checkpoint(f"step-{self.global_step}")
                    else:
                        bad += 1
                        if cfg.early_stopping_patience and bad >= cfg.early_stopping_patience:
                            self.say("Early stopping: validation loss stopped improving.")
                            step = steps
                            break
                if step >= steps:
                    break
        if self.best_state:
            self._restore(self.best_state)
            self.say("Restored best SFT checkpoint.")

    def _rlcd(self, train: list[Example], val: list[Example]) -> None:
        """RLCD: proper-scoring reward + KL anchor to the SFT policy + order consistency."""
        cfg = self.cfg
        self.say("RLCD: computing reference policy on the training set…")
        self.log.status(stage="rlcd-reference")
        ref = []
        bs = (self.micro_bs or cfg.batch_size) * 2
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(train), bs):
                chunk = train[i:i + bs]
                b, encs = self.model.batch(chunk, cfg.max_length)
                with self._autocast():
                    p = torch.softmax(self.model(**b).float(), -1).cpu()
                for j, e in enumerate(encs):
                    ref.append([float(p[j, pos]) for pos in range(len(e.order))])  # original order
        self.model.train()
        steps_per_epoch = math.ceil(len(train) / cfg.batch_size / cfg.grad_accum)
        steps = max(1, int(steps_per_epoch * cfg.rlcd_epochs))
        lr = cfg.rlcd_learning_rate or cfg.learning_rate * 0.5
        opt, sched = self._optimizer(lr, cfg.head_learning_rate * 0.3, steps)
        best_score, step = float("inf"), 0
        best = self._snapshot()
        m0 = self.evaluate(val)
        best_score = m0["nll"] + m0["brier"]
        self.say(f"RLCD: {steps} steps, lr={lr:g}, beta={cfg.rlcd_beta}")
        eval_every = max(1, min(cfg.eval_every or steps_per_epoch, max(1, steps // 2)))
        while step < steps:
            batch_iter = list(self._batches(train, cfg.batch_size))
            for bi in range(0, len(batch_iter), cfg.grad_accum):
                t0 = time.time()
                group = batch_iter[bi:bi + cfg.grad_accum]
                logs = self._accumulate(opt, [i for g in group for i in g],
                                        lambda idx, w: self._rlcd_micro(train, ref, idx, w))
                gn = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                                    cfg.max_grad_norm)
                opt.step()
                sched.step()
                step += 1
                if step % max(1, cfg.log_every) == 0 or step == 1:
                    self.log.log(self.global_step + 1, "rlcd", {**logs, "lr": sched.get_last_lr()[0],
                                                                 "grad_norm": float(gn), "memory_gb": self._mem()})
                self._tick(t0, "rlcd", step / steps_per_epoch)
                if step % eval_every == 0 or step == steps:
                    m = self.evaluate(val)
                    self.log.log(self.global_step, "val", {**m, "stage": "rlcd"})
                    self.say(f"RLCD step {step} val acc={m['accuracy']:.4f} ece={m['ece']:.4f} brier={m['brier']:.4f}")
                    score = m["nll"] + m["brier"]
                    if score < best_score:
                        best_score, best = score, self._snapshot()
                if step >= steps:
                    break
        self._restore(best)

    # ---------------------------------------------------------- micro-batching
    def _auto_micro_batch(self, train: list[Example]) -> None:
        """Pick a micro-batch that fits the memory budget, from the real token length of the data."""
        from .compute import detect, estimate_memory_gb, recommend
        cfg = self.cfg
        sample = train[: min(len(train), 64)]
        lengths = sorted(len(self.model.batch([e], cfg.max_length)[0]["input_ids"][0]) for e in sample)
        seq = lengths[int(0.9 * (len(lengths) - 1))] if lengths else cfg.max_length
        info = detect()
        budget = recommend(info)["memory_budget_gb"]
        if budget <= 0 or info["device"] == "cpu":
            return
        _, total = self.model.trainable_parameters()
        params_b = total / 1e9
        mb = cfg.batch_size
        while mb > 1 and estimate_memory_gb(params_b, cfg.finetune, mb, seq, info.get("bf16", False),
                                            cfg.gradient_checkpointing) > budget * 0.85:
            mb //= 2
        self.say(f"Sequences ≈{seq} tokens (p90). Memory budget {budget} GB → micro-batch {mb}"
                 + (f" × {cfg.batch_size // mb} accumulation steps" if mb < cfg.batch_size else ""))
        if mb < cfg.batch_size:
            self.micro_bs = mb
            self.log.status(micro_batch=mb)

    def _free(self) -> None:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _accumulate(self, opt, idx_all: list[int], fn) -> dict[str, float]:
        """Run `fn` over micro-batches of one optimizer step. On out-of-memory, halve the
        micro-batch and retry the step: the effective batch size never changes."""
        while True:
            opt.zero_grad(set_to_none=True)
            mb = min(self.micro_bs or len(idx_all), len(idx_all))
            acc: dict[str, float] = {}
            oom = False
            try:
                for s in range(0, len(idx_all), mb):
                    chunk = idx_all[s:s + mb]
                    for k, v in fn(chunk, len(chunk) / len(idx_all)).items():
                        acc[k] = acc.get(k, 0.0) + v
                return acc
            except RuntimeError as e:  # torch.OutOfMemoryError subclasses RuntimeError
                if "out of memory" not in str(e).lower() or mb <= 1:
                    raise
                oom = True
            if oom:
                opt.zero_grad(set_to_none=True)
                self._free()
                self.micro_bs = max(1, mb // 2)
                self.say(f"Out of memory at micro-batch {mb}; retrying with {self.micro_bs} "
                         f"(gradient accumulation keeps the effective batch at {len(idx_all)}).")
                self.log.status(micro_batch=self.micro_bs)

    def _sft_micro(self, train: list[Example], idx: list[int], w: float) -> dict[str, float]:
        cfg = self.cfg
        exs = self._apply_augment([train[i] for i in idx])
        b, encs = self.model.batch(exs, cfg.max_length, shuffle=cfg.option_shuffle, rng=self.rng)
        with self._autocast():
            logits = self.model(**b).float()
        tgt = self._targets(exs, encs, logits.size(1))
        loss = -(tgt * F.log_softmax(logits, -1)).sum(-1).mean()
        (loss * w).backward()
        return {"loss": loss.item() * w, "n_tok": float(b["attention_mask"].sum()), "n_ex": float(len(exs))}

    def _rlcd_micro(self, train: list[Example], ref: list[list[float]], idx: list[int], w: float) -> dict[str, float]:
        cfg = self.cfg
        exs = self._apply_augment([train[i] for i in idx])
        b1, e1 = self.model.batch(exs, cfg.max_length, shuffle=cfg.option_shuffle, rng=self.rng)
        b2, e2 = self.model.batch(exs, cfg.max_length, shuffle=cfg.option_shuffle, rng=self.rng)
        with self._autocast():
            l1 = self.model(**b1).float()
            l2 = self.model(**b2).float()
        O = l1.size(1)
        total = 0.0
        p_orig = []
        logs = {"brier": 0.0, "nll": 0.0, "kl_ref": 0.0}
        for logits, encs in ((l1, e1), (l2, e2)):
            lp = F.log_softmax(logits, -1)
            p = lp.exp()
            y = self._targets(exs, encs, O)
            valid = (logits > -1e3).float()
            brier = ((p - y) ** 2 * valid).sum(-1).mean()
            nll = -(y * lp).sum(-1).mean()
            r = torch.zeros_like(p)
            for bb, (i, e) in enumerate(zip(idx, encs)):
                for pos, orig in enumerate(e.order):
                    r[bb, pos] = ref[i][orig]
            r = r.clamp(min=1e-8)
            kl = (p * (lp - r.log()) * valid).sum(-1).mean()
            total = total + 0.5 * (cfg.rlcd_brier_weight * brier + nll + cfg.rlcd_beta * kl)
            oi = self._orig_index(encs, O)
            p_orig.append(torch.zeros_like(p).scatter(1, oi, p).clamp(min=1e-8))
            logs["brier"] += 0.5 * brier.item() * w
            logs["nll"] += 0.5 * nll.item() * w
            logs["kl_ref"] += 0.5 * kl.item() * w
        pa, pb = p_orig
        cons = 0.5 * ((pa * (pa.log() - pb.log())).sum(-1) + (pb * (pb.log() - pa.log())).sum(-1)).mean()
        total = total + cfg.rlcd_consistency_weight * cons
        (total * w).backward()
        return {**logs, "consistency": cons.item() * w, "loss": float(total.item()) * w}

    # ------------------------------------------------------------ snapshots
    def _snapshot(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().to("cpu", copy=True) for n, p in self.model.named_parameters() if p.requires_grad}

    def _restore(self, state: dict[str, torch.Tensor]) -> None:
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, v in state.items():
                params[n].copy_(v.to(params[n].device, params[n].dtype))

    def _checkpoint(self, name: str) -> None:
        if self.cfg.save_total_limit <= 0:
            return
        from safetensors.torch import save_file
        d = self.dir / "checkpoints" / name
        d.mkdir(parents=True, exist_ok=True)
        save_file({k: v.float().contiguous() for k, v in (self.best_state or self._snapshot()).items()},
                  str(d / "trainable.safetensors"))
        ckpts = sorted((self.dir / "checkpoints").iterdir(), key=lambda p: p.stat().st_mtime)
        for old in ckpts[:-self.cfg.save_total_limit]:
            shutil.rmtree(old, ignore_errors=True)

    # ------------------------------------------------------------------ main
    def train(self) -> dict[str, Any]:
        cfg = self.cfg
        try:
            torch.manual_seed(cfg.seed)
            self.log.status(state="running", stage="loading-data", step=0, total_steps=0, progress=0.0,
                            started=time.time(), error=None)
            train = data_store.load_examples(cfg.data.dataset, "train", cfg.data.max_train_samples)
            val = data_store.load_examples(cfg.data.dataset, "validation", cfg.data.max_eval_samples)
            test = data_store.load_examples(cfg.data.dataset, "test", cfg.data.max_eval_samples)
            if not train:
                raise ValueError(f"Dataset '{cfg.data.dataset}' has no training rows")
            if not val:
                k = max(1, len(train) // 10)
                val, train = train[:k], train[k:]
            self.say(f"Data: train={len(train)} val={len(val)} test={len(test)}")

            self.log.status(stage="loading-model")
            self.say(f"Loading {cfg.base_model} ({cfg.finetune})…")
            self.model = DecisionModel.from_base(
                cfg.base_model, arch=cfg.arch, pooling=cfg.pooling, finetune=cfg.finetune,
                precision=cfg.precision, device=resolve_device(cfg.device), trust_remote_code=cfg.trust_remote_code,
                gradient_checkpointing=cfg.gradient_checkpointing, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout, lora_targets=cfg.lora_targets)
            self.model.meta["max_length"] = cfg.max_length
            trainable, total = self.model.trainable_parameters()
            self.say(f"Model on {self.model.device}: {total/1e6:.1f}M params, {trainable/1e6:.2f}M trainable "
                     f"(arch={self.model.meta['arch']}, pooling={self.model.meta['pooling']})")
            self.log.status(model_info={"arch": self.model.meta["arch"], "pooling": self.model.meta["pooling"],
                                        "params": total, "trainable_params": trainable,
                                        "device": str(self.model.device)})

            self._auto_micro_batch(train)
            spe = math.ceil(len(train) / cfg.batch_size / cfg.grad_accum)
            sft_steps = max(1, int(spe * cfg.epochs)) if cfg.method in ("sft", "sft+rlcd") else 0
            rl_steps = max(1, int(spe * cfg.rlcd_epochs)) if cfg.method in ("rlcd", "sft+rlcd") else 0
            self.total_steps = sft_steps + rl_steps
            self.log.status(total_steps=self.total_steps)

            baseline = self.evaluate(val[:256])
            self.log.log(0, "val", {**baseline, "epoch": 0.0})

            if sft_steps:
                self._sft(train, val)
            if rl_steps:
                self._rlcd(train, val)

            self.log.status(stage="calibrating")
            val_m, val_logits = self.evaluate(val, return_logits=True, temperature=1.0)
            T = 1.0
            if cfg.temperature_scaling and len(val) >= 20:
                T = fit_temperature(val_logits, [e.label for e in val])
                self.say(f"Temperature scaling: T={T:.3f}")
            self.model.temperature = T

            self.log.status(stage="evaluating")
            results: dict[str, Any] = {"temperature": T, "validation_uncalibrated": val_m,
                                       "validation": self.evaluate(val, details=True)}
            if test:
                results["test"] = self.evaluate(test, details=True)
            (self.dir / "eval.json").write_text(json.dumps(results, indent=2))
            for split in ("validation", "test"):
                if split in results:
                    self.log.log(self.global_step, f"final_{split}",
                                 {k: v for k, v in results[split].items() if isinstance(v, (int, float))})

            self.log.status(stage="saving")
            meta = data_store.get_meta(cfg.data.dataset) if (data_store.resolve(cfg.data.dataset) / "meta.json").exists() else {}
            q = meta.get("example_question") or (train[0].question.to_dict())
            self.model.save(self.dir / "model", extra_meta={
                "name": cfg.run_name or self.dir.name, "question": q, "dataset": cfg.data.dataset,
                "method": cfg.method, "metrics": {s: {k: v for k, v in results[s].items() if isinstance(v, (int, float))}
                                                  for s in ("validation", "test") if s in results}})
            elapsed = time.time() - self.t_start
            summary = {"elapsed_s": round(elapsed, 1), "train_examples": len(train), "val_examples": len(val),
                       "test_examples": len(test), "temperature": T,
                       "final": {s: {k: v for k, v in results[s].items() if isinstance(v, (int, float))}
                                 for s in ("validation", "test") if s in results}}
            self.log.status(state="completed", stage="done", progress=1.0, step=self.global_step, eta_s=0,
                            ended=time.time(), summary=summary)
            self.say(f"Done in {elapsed/60:.1f} min. Model saved to {self.dir / 'model'}")
            return summary
        except StopTraining:
            self.say("Stopped by user.")
            if getattr(self, "model", None) is not None and getattr(self, "best_state", None):
                self._restore(self.best_state)
                self.model.save(self.dir / "model", extra_meta={"name": self.dir.name, "partial": True,
                                                                "dataset": cfg.data.dataset})
            self.log.status(state="stopped", stage="stopped", ended=time.time())
            return {"stopped": True}
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(tb, flush=True)
            self.log.status(state="failed", stage="failed", error=f"{type(e).__name__}: {e}", ended=time.time())
            raise
        finally:
            self.log.close()
