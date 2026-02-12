# project_root/src/wandb_logging/run_manager.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import wandb

from ..config_loader import ConfigBundle


@dataclass(frozen=True)
class WandbRunConfig:
    project: str
    entity: Optional[str] = None
    job_type: str = "openai-batch"
    tags: Optional[list[str]] = None
    name: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None  # e.g., "online" | "offline" | "disabled"


def init_wandb_run(
    bundle: ConfigBundle,
    *,
    run_cfg: Optional[WandbRunConfig] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> wandb.sdk.wandb_run.Run:
    """
    Initialize a W&B run using config.yaml's `wandb:` section (plus optional overrides).

    - Uses WANDB_API_KEY from environment automatically (wandb handles it).
    - Returns the wandb Run object.
    """
    w = bundle.config.get("wandb", {}) or {}
    tags = w.get("tags", [])
    if tags is None:
        tags = []
    if not isinstance(tags, list):
        tags = [str(tags)]

    # Default from config.yaml
    default = WandbRunConfig(
        project=str(w.get("project", "openai-batch-audit")),
        entity=w.get("entity"),
        job_type=str(w.get("job_type", "openai-batch")),
        tags=[str(t) for t in tags],
    )

    # Allow overrides
    cfg = run_cfg or default

    # Assemble run config payload
    payload: Dict[str, Any] = {
        "paths": {
            "batch_input_dir": str(bundle.batch_input_dir),
            "output_dir": str(bundle.output_dir),
            "state_file": str(bundle.state_file),
        },
        "openai_batch": bundle.config.get("openai_batch", {}),
        "model_configs": bundle.config.get("model_configs", []),
        "cost_estimation": bundle.config.get("cost_estimation", {}),
        "pricing_tier": bundle.pricing_tier,
    }
    if extra_config:
        payload.update(extra_config)

    run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        job_type=cfg.job_type,
        tags=cfg.tags or [],
        name=cfg.name,
        notes=cfg.notes,
        mode=cfg.mode,
        config=payload,
    )
    return run

