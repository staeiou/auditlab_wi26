# project_root/src/wandb_logging/artifacts.py

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

import wandb

from ..config_loader import ConfigBundle, safe_model_dirname


class ArtifactSpecError(ValueError):
    pass


def _now() -> float:
    return time.time()


def _coerce_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    return [str(x)]


def _format_path_template(path_template: str, *, model_dir: Optional[str]) -> str:
    if model_dir is None:
        return path_template
    return path_template.format(model_dir=model_dir)


def _add_files_to_artifact(
    artifact: wandb.Artifact,
    *,
    project_root: Path,
    include_files: List[str],
    model_dir: Optional[str] = None,
) -> int:
    """
    Add explicit files (paths can include {model_dir}) to an artifact.
    Returns number of files added.
    """
    added = 0
    for rel in include_files:
        rel_fmt = _format_path_template(rel, model_dir=model_dir)
        p = (project_root / rel_fmt).resolve()
        if p.exists() and p.is_file():
            artifact.add_file(str(p))
            added += 1
    return added

def _normalize_glob_pattern(pat: str) -> str:
    """
    pathlib.Path.glob() requires '**' to be a whole path component.
    Normalize common mistakes like:
      - 'batch/**_requests.jsonl' -> 'batch/**/*_requests.jsonl'
      - 'outputs/**.jsonl'        -> 'outputs/**/*.jsonl'
      - '**.jsonl'                -> '**/*.jsonl'
    """
    parts = pat.split("/")
    new_parts = []
    for part in parts:
        if "**" in part and part != "**":
            # Handle patterns where '**' is embedded in the component.
            if part.startswith("**"):
                suffix = part[2:]
                if suffix:
                    new_parts.append("**")
                    # ensure suffix component still has a wildcard
                    new_parts.append(("*" + suffix) if not suffix.startswith("*") else suffix)
                else:
                    new_parts.append("**")
            elif part.endswith("**"):
                prefix = part[:-2]
                if prefix:
                    new_parts.append(prefix + "*")
                    new_parts.append("**")
                else:
                    new_parts.append("**")
            else:
                # rare: '**' in the middle of a component -> degrade to single '*'
                new_parts.append(part.replace("**", "*"))
        else:
            new_parts.append(part)
    return "/".join(new_parts)

def _add_globs_to_artifact(artifact, *, project_root: Path, include_globs: Iterable[str]) -> int:
    added = 0
    for raw_pat in include_globs:
        pat = _normalize_glob_pattern(raw_pat)
        try:
            for p in project_root.glob(pat):
                if p.is_file():
                    artifact.add_file(str(p), name=str(p.relative_to(project_root)))
                    added += 1
        except ValueError as e:
            raise ValueError(
                f"Invalid glob pattern '{raw_pat}' (normalized to '{pat}'). "
                f"Note: '**' must be its own path component, e.g. '**/*.jsonl'."
            ) from e
    return added

def _require_key(d: dict, k: str, where: str) -> Any:
    if k not in d:
        raise ArtifactSpecError(f"Missing key '{k}' in {where}")
    return d[k]


@dataclass
class ArtifactUploader:
    """
    W&B artifact uploader driven by config/logging.yaml.

    Intended usage:
      uploader = ArtifactUploader(bundle, run)
      uploader.upload_inputs_once()
      ...
      uploader.upload_state_if_due()
      ...
      uploader.upload_results_for_model_if_present(model_name)

    Rate limiting:
      - Respects limits.yaml wandb min_seconds_between_artifact_uploads if present.
      - Also respects per-artifact upload_every_seconds if present (state).
    """

    bundle: ConfigBundle
    run: wandb.sdk.wandb_run.Run
    project_root: Path

    last_artifact_upload_unix: float = 0.0
    last_state_upload_unix: float = 0.0

    def __post_init__(self) -> None:
        self.project_root = self.project_root.resolve()

    # -------------------------
    # Config helpers
    # -------------------------
    def _wandb_limits(self) -> dict:
        return (self.bundle.limits.get("wandb", {}) or {})

    def _min_artifact_interval(self) -> int:
        lim = self._wandb_limits()
        return int(lim.get("min_seconds_between_artifact_uploads", 60))

    def _artifact_specs(self) -> dict:
        logging_cfg = self.bundle.logging
        artifacts = logging_cfg.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ArtifactSpecError("logging.yaml: artifacts must be a mapping")
        return artifacts

    # -------------------------
    # Upload gating
    # -------------------------
    def _ok_to_upload_artifact(self, force: bool = False) -> bool:
        if force:
            return True
        return (_now() - self.last_artifact_upload_unix) >= self._min_artifact_interval()

    def _mark_artifact_uploaded(self) -> None:
        self.last_artifact_upload_unix = _now()

    # -------------------------
    # Artifact builders
    # -------------------------
    def _build_artifact(self, *, name: str, type_: str, metadata: Optional[dict] = None) -> wandb.Artifact:
        art = wandb.Artifact(name=name, type=type_)
        if metadata:
            art.metadata.update(metadata)
        return art

    # -------------------------
    # Public API
    # -------------------------
    def upload_inputs_once(self, force: bool = False) -> Optional[str]:
        """
        Upload the input artifact as defined in logging.yaml: artifacts.input_artifact
        Returns artifact name if uploaded, else None.
        """
        if not self._ok_to_upload_artifact(force=force):
            return None

        specs = self._artifact_specs()
        spec = _require_key(specs, "input_artifact", "logging.yaml:artifacts")

        name = str(_require_key(spec, "name", "logging.yaml:artifacts.input_artifact"))
        type_ = str(_require_key(spec, "type", "logging.yaml:artifacts.input_artifact"))
        include_globs = _coerce_list(spec.get("include_globs"))
        include_files = _coerce_list(spec.get("include_files"))

        art = self._build_artifact(
            name=name,
            type_=type_,
            metadata={
                "kind": "inputs",
                "batch_input_dir": str(self.bundle.batch_input_dir),
            },
        )

        added = 0
        added += _add_globs_to_artifact(art, project_root=self.project_root, include_globs=include_globs)
        added += _add_files_to_artifact(art, project_root=self.project_root, include_files=include_files)

        if added == 0:
            # Avoid uploading empty artifacts
            return None

        self.run.log_artifact(art)
        self._mark_artifact_uploaded()
        return name

    def upload_state_if_due(self, force: bool = False) -> Optional[str]:
        """
        Upload the state artifact periodically.
        Uses logging.yaml: artifacts.state_artifact.upload_every_seconds (default 60).
        Returns artifact name if uploaded, else None.
        """
        specs = self._artifact_specs()
        spec = _require_key(specs, "state_artifact", "logging.yaml:artifacts")

        every = int(spec.get("upload_every_seconds", 60))
        if not force and (_now() - self.last_state_upload_unix) < every:
            return None
        if not self._ok_to_upload_artifact(force=force):
            return None

        name = str(_require_key(spec, "name", "logging.yaml:artifacts.state_artifact"))
        type_ = str(_require_key(spec, "type", "logging.yaml:artifacts.state_artifact"))
        include_files = _coerce_list(_require_key(spec, "include_files", "logging.yaml:artifacts.state_artifact"))

        art = self._build_artifact(
            name=name,
            type_=type_,
            metadata={"kind": "state"},
        )

        added = _add_files_to_artifact(art, project_root=self.project_root, include_files=include_files)
        if added == 0:
            return None

        self.run.log_artifact(art)
        self._mark_artifact_uploaded()
        self.last_state_upload_unix = _now()
        return name

    def upload_results_for_model_if_present(self, model: str, force: bool = False) -> Optional[str]:
        """
        Upload per-model results artifact if the configured output files exist.
        Uses logging.yaml: artifacts.results_artifact (name_template, include_files).
        Returns artifact name if uploaded, else None.
        """
        if not self._ok_to_upload_artifact(force=force):
            return None

        specs = self._artifact_specs()
        spec = _require_key(specs, "results_artifact", "logging.yaml:artifacts")

        name_template = str(_require_key(spec, "name_template", "logging.yaml:artifacts.results_artifact"))
        type_ = str(_require_key(spec, "type", "logging.yaml:artifacts.results_artifact"))
        include_files = _coerce_list(_require_key(spec, "include_files", "logging.yaml:artifacts.results_artifact"))

        model_dir = safe_model_dirname(model)
        art_name = name_template.format(model_dir=model_dir)

        # Only upload if at least one included file exists
        probe = 0
        for rel in include_files:
            rel_fmt = _format_path_template(rel, model_dir=model_dir)
            p = (self.project_root / rel_fmt).resolve()
            if p.exists() and p.is_file():
                probe += 1
        if probe == 0:
            return None

        art = self._build_artifact(
            name=art_name,
            type_=type_,
            metadata={"kind": "results", "model": model, "model_dir": model_dir},
        )

        _add_files_to_artifact(art, project_root=self.project_root, include_files=include_files, model_dir=model_dir)

        self.run.log_artifact(art)
        self._mark_artifact_uploaded()
        return art_name

