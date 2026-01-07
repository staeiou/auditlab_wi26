from __future__ import annotations

from dataclass import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

def safe_model_dirname(model: str) -> str:
    return model.replace("/", "_")

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing Yaml file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Yaml must be a mapping at top-level: {path}")
    return data

def _require(d: dict, key: str, where: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return d[key]

def _require_type(val: Any, typ, where: str) -> Any:
    if not isinstance(val, typ):
        raise TypeError(f"Expected {typ.__name__} in {where}, got {type(val).__name__}")
    return val

@dataclass(frozen=True)
class ConfigBundle:
    project_root: Path
    config_dir: Path

    config: Dict[str, Any]
    pricing: Dict[str, Any]
    limits: Dict[str, Any]
    logging: Dict[str, Any]

    batch_input_dir: Path
    output_dir: Path
    state_file: Path

    #property
    def model_configs(self) -> List[Dict[str, Any]]:
        return self.config["model_configs"]

    @property
    def pricing_tier(self) -> str:
        return self.config.get("cost_estimation", {}).get("pricing_tier", "standard")

