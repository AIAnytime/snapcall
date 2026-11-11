"""Subprocess entry point: python -m snapcall.worker <run_dir>"""

import sys
from pathlib import Path

from .config import TrainConfig
from .trainer import Trainer


def main() -> None:
    run_dir = Path(sys.argv[1])
    cfg = TrainConfig.load(run_dir / "config.json")
    try:
        Trainer(cfg, run_dir).train()
    except Exception:  # noqa: BLE001 - status.json already records the failure
        sys.exit(1)


if __name__ == "__main__":
    main()
