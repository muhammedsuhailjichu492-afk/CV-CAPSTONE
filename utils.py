from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent          # project folder


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(rel: str | Path) -> Path:
    rel = Path(rel)
    return rel if rel.is_absolute() else ROOT / rel


def out_dir(cfg: dict, *parts: str) -> Path:
    d = p(cfg["paths"]["outputs"]).joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def pick_device(setting="auto"):
    if setting != "auto":
        return setting
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return "cpu"


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def banner(title: str):
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t
