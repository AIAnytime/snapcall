"""Run tracking: status.json + metrics.jsonl (+ TensorBoard event files).

Everything is plain files inside the run folder, so the UI, the CLI and
TensorBoard can all read a run while it trains, and runs survive restarts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, run_dir: str | Path, tensorboard: bool = True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        self.status_path = self.dir / "status.json"
        self._status: dict[str, Any] = self.read_status(self.dir) or {}
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(self.dir / "tensorboard"))
            except Exception:  # noqa: BLE001 - tensorboard optional
                self.tb = None

    # -- metrics
    def log(self, step: int, split: str, values: dict[str, float]) -> None:
        rec = {"step": step, "split": split, "time": time.time(),
               **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in values.items()}}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.tb:
            for k, v in values.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(f"{split}/{k}", v, step)

    def text(self, tag: str, text: str) -> None:
        if self.tb:
            self.tb.add_text(tag, text)

    # -- status
    def status(self, **kw: Any) -> dict[str, Any]:
        self._status.update(kw, updated=time.time(), pid=os.getpid())
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._status, indent=2))
        tmp.replace(self.status_path)
        return self._status

    def close(self) -> None:
        if self.tb:
            self.tb.flush()
            self.tb.close()

    def stop_requested(self) -> bool:
        return (self.dir / "STOP").exists()

    @staticmethod
    def read_status(run_dir: str | Path) -> dict[str, Any] | None:
        p = Path(run_dir) / "status.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def read_metrics(run_dir: str | Path, after: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Return metric lines after line offset `after`, and the new offset."""
        p = Path(run_dir) / "metrics.jsonl"
        if not p.exists():
            return [], 0
        lines = p.read_text().splitlines()
        out = []
        for l in lines[after:]:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                break
        return out, after + len(out)
