"""Reproducibility helpers: seeding, device selection, run manifest."""
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import transformers
import peft

from config import MANIFEST_NAME, SEED


def set_global_seed(seed: int = SEED) -> None:
    """Seed every RNG used by the pipeline and request deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "cpu" else torch.bfloat16


def environment_versions() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }


def write_manifest(output_dir: Path, manifest: dict) -> None:
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_manifest(adapter_dir: Path) -> dict:
    path = adapter_dir / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"Run manifest not found at {path}; train the adapter first.")
    return json.loads(path.read_text(encoding="utf-8"))
