"""Configuration loading and global seeding."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config; relative paths resolve against the repo root."""
    path = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: str((REPO_ROOT / v).resolve()) for k, v in cfg["paths"].items()}
    rc = cfg.get("rerank", {})
    for k in ("negatives_ltr_oof_dir", "reuse_scores_from"):
        if k in rc:
            rc[k] = str((REPO_ROOT / rc[k]).resolve())
    return cfg


def set_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy and torch, and switch torch to deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects child processes only; this process never relies on
    # set/dict iteration order for results (every ranking sorts explicitly; reruns verified byte-identical)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(int(os.environ.get("TERMMAP_THREADS", "8")))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
