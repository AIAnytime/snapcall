"""Snapcall: build better decision models.

Train, evaluate and deploy calibrated System One (Jev-style) decision models.

Built by AI Anytime.
"""

__version__ = "0.1.0"

import logging as _logging
import os as _os

# Cloud GPU pods (RunPod): keep model downloads on the persistent /workspace volume instead of the small,
# ephemeral container disk. Must run before huggingface_hub is imported anywhere.
if _os.environ.get("RUNPOD_POD_ID") and _os.path.isdir("/workspace") and not _os.environ.get("HF_HOME"):
    _os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

_logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(_logging.ERROR)

from .config import DataConfig, TrainConfig  # noqa: E402


def __getattr__(name):  # lazy heavy imports (torch) so the CLI starts fast
    if name == "Trainer":
        from .trainer import Trainer
        return Trainer
    if name == "DecisionModel":
        from .modeling import DecisionModel
        return DecisionModel
    raise AttributeError(name)


__all__ = ["DataConfig", "TrainConfig", "Trainer", "DecisionModel", "__version__"]
