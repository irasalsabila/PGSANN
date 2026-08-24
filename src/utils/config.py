import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")


@dataclass
class Config:
    """Thin wrapper over a dict loaded from YAML, with dotted access."""

    data: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    train: Dict[str, Any] = field(default_factory=dict)
    loss: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key in self.data:
            return self.data[key]
        if key in self.model:
            return self.model[key]
        if key in self.train:
            return self.train[key]
        if key in self.loss:
            return self.loss[key]
        if key in self.extra:
            return self.extra[key]
        raise KeyError(key)

    def get(self, section: str, default: Any = None) -> Any:
        section = getattr(self, section, None)
        return section if section is not None else default


def load_config(path: str) -> Config:
    """Load a YAML config. Accepts absolute path or name relative to PROJECT_ROOT."""
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(path):
        # Fall back to configs/<name>
        path = os.path.join(CONFIG_DIR, os.path.basename(path))
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config()
    for section, values in raw.items():
        if not isinstance(values, dict):
            cfg.extra[section] = values
            continue
        if section in ("data", "model", "train", "loss"):
            setattr(cfg, section, values)
        else:
            cfg.extra[section] = values
    return cfg


def save_dict(d: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
