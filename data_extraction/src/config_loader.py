# project_root/src/config_loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def safe_model_dirname(model: str) -> str:
    """Filesystem-safe directory name for a model id."""
    return model.replace("/", "_")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping at top-level: {path}")
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
    """
    Holds all YAML configs loaded from config/:
      - config.yaml
      - pricing.yaml
      - limits.yaml
      - logging.yaml

    Paths are resolved relative to project_root.
    """
    project_root: Path
    config_dir: Path

    config: Dict[str, Any]
    pricing: Dict[str, Any]
    limits: Dict[str, Any]
    logging: Dict[str, Any]

    # Common resolved paths (from config.yaml)
    batch_input_dir: Path
    output_dir: Path
    state_file: Path

    @property
    def model_configs(self) -> List[Dict[str, Any]]:
        return self.config["model_configs"]

    @property
    def pricing_tier(self) -> str:
        return self.config.get("cost_estimation", {}).get("pricing_tier", "standard")


def load_config_bundle(
    *,
    project_root: Optional[Path] = None,
    config_dir: Optional[Path] = None,
) -> ConfigBundle:
    """
    Load and validate YAML configs.

    Defaults:
      project_root = project_root/src/config_loader.py -> parents[1]
      config_dir    = <project_root>/config
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    if config_dir is None:
        config_dir = project_root / "config"

    cfg = _load_yaml(config_dir / "config.yaml")
    pricing = _load_yaml(config_dir / "pricing.yaml")
    limits = _load_yaml(config_dir / "limits.yaml")
    logging = _load_yaml(config_dir / "logging.yaml")

    # -------------------------
    # Validate config.yaml
    # -------------------------
    paths = _require_type(_require(cfg, "paths", "config.yaml"), dict, "config.yaml:paths")
    openai_batch = _require_type(_require(cfg, "openai_batch", "config.yaml"), dict, "config.yaml:openai_batch")
    model_configs = _require(cfg, "model_configs", "config.yaml")
    _require_type(model_configs, list, "config.yaml:model_configs")
    if not model_configs:
        raise ValueError("config.yaml:model_configs must be a non-empty list")

    for i, mc in enumerate(model_configs):
        _require_type(mc, dict, f"config.yaml:model_configs[{i}]")
        _require(mc, "name", f"config.yaml:model_configs[{i}]")
        # allow additional keys like temperature/max_completion_tokens

    # Required path strings
    batch_input_dir_s = _require(paths, "batch_input_dir", "config.yaml:paths")
    output_dir_s = _require(paths, "output_dir", "config.yaml:paths")
    state_file_s = _require(paths, "state_file", "config.yaml:paths")

    # Required openai_batch keys
    _require(openai_batch, "endpoint", "config.yaml:openai_batch")
    _require(openai_batch, "completion_window", "config.yaml:openai_batch")
    _require(openai_batch, "poll_interval_seconds", "config.yaml:openai_batch")

    # -------------------------
    # Validate pricing.yaml
    # -------------------------
    tiers = _require_type(_require(pricing, "tiers", "pricing.yaml"), dict, "pricing.yaml:tiers")
    # aliases optional
    aliases = pricing.get("aliases", {})
    if aliases is None:
        aliases = {}
    _require_type(aliases, dict, "pricing.yaml:aliases")

    # -------------------------
    # Validate limits.yaml
    # -------------------------
    lim_batch = _require_type(_require(limits, "openai_batch", "limits.yaml"), dict, "limits.yaml:openai_batch")
    for k in ["max_requests_per_file", "max_file_size_mb", "allowed_completion_windows", "require_single_model_per_file"]:
        _require(lim_batch, k, "limits.yaml:openai_batch")

    # completion window allowed?
    allowed = lim_batch["allowed_completion_windows"]
    _require_type(allowed, list, "limits.yaml:openai_batch.allowed_completion_windows")
    cw = openai_batch["completion_window"]
    if cw not in allowed:
        raise ValueError(
            f"config.yaml openai_batch.completion_window={cw!r} not allowed. "
            f"Allowed: {allowed}"
        )

    # -------------------------
    # Validate logging.yaml (light)
    # -------------------------
    _require_type(_require(logging, "artifacts", "logging.yaml"), dict, "logging.yaml:artifacts")
    _require_type(_require(logging, "results_compact", "logging.yaml"), dict, "logging.yaml:results_compact")

    # -------------------------
    # Resolve paths
    # -------------------------
    batch_input_dir = (project_root / str(batch_input_dir_s)).resolve()
    output_dir = (project_root / str(output_dir_s)).resolve()
    state_file = (project_root / str(state_file_s)).resolve()

    return ConfigBundle(
        project_root=project_root,
        config_dir=config_dir,
        config=cfg,
        pricing=pricing,
        limits=limits,
        logging=logging,
        batch_input_dir=batch_input_dir,
        output_dir=output_dir,
        state_file=state_file,
    )

