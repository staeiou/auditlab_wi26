from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set


@dataclass(frozen=True)
class BatchInputLimits:
    max_requests_per_file: int = 50_000
    max_file_size_mb: int = 200
    require_single_model_per_file: bool = True


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    n_requests: int
    file_size_mb: float


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield line_no, json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


def validate_batch_jsonl(
    *,
    jsonl_path: Path,
    limits: BatchInputLimits,
    expected_model: Optional[str] = None,
    expected_url: Optional[str] = None,
    expected_method: str = "POST",
    max_model_checks: int = 500,
) -> ValidationResult:
    """
    Validate a Batch input JSONL file against limits + basic schema expectations.

    Checks:
      - file size <= max_file_size_mb
      - request count <= max_requests_per_file
      - each line has: custom_id, method, url, body.model
      - method matches expected_method (default POST)
      - url matches expected_url if provided
      - if require_single_model_per_file: body.model == expected_model (or infer from first line)

    This does NOT validate OpenAI model existence — just structure + constraints.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    size_mb = jsonl_path.stat().st_size / (1024 * 1024)
    if size_mb > limits.max_file_size_mb:
        raise ValueError(
            f"{jsonl_path} is {size_mb:.1f}MB > {limits.max_file_size_mb}MB limit"
        )

    n = 0
    seen_custom_ids: Set[str] = set()

    inferred_model: Optional[str] = None

    for line_no, obj in _read_jsonl(jsonl_path):
        n += 1
        if n > limits.max_requests_per_file:
            raise ValueError(
                f"{jsonl_path} has > {limits.max_requests_per_file} requests (hit at line {line_no})"
            )

        # Basic keys
        if "custom_id" not in obj:
            raise ValueError(f"Missing custom_id at line {line_no} in {jsonl_path}")
        cid = str(obj["custom_id"])
        if cid in seen_custom_ids:
            raise ValueError(f"Duplicate custom_id={cid!r} at line {line_no} in {jsonl_path}")
        seen_custom_ids.add(cid)

        method = obj.get("method")
        if method != expected_method:
            raise ValueError(
                f"Unexpected method={method!r} at line {line_no}; expected {expected_method!r}"
            )

        url = obj.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError(f"Missing/invalid url at line {line_no} in {jsonl_path}")
        if expected_url is not None and url != expected_url:
            raise ValueError(
                f"url mismatch at line {line_no}: got {url!r}, expected {expected_url!r}"
            )

        body = obj.get("body")
        if not isinstance(body, dict):
            raise ValueError(f"Missing/invalid body at line {line_no} in {jsonl_path}")

        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"Missing/invalid body.model at line {line_no} in {jsonl_path}")

        if inferred_model is None:
            inferred_model = model

        # Only check the first N for speed
        if limits.require_single_model_per_file and max_model_checks > 0 and n <= max_model_checks:
            if expected_model is not None and model != expected_model:
                raise ValueError(
                    f"Model mismatch at line {line_no}: got {model!r}, expected {expected_model!r}"
                )
            if expected_model is None and model != inferred_model:
                raise ValueError(
                    f"Multiple models detected at line {line_no}: first={inferred_model!r}, got={model!r}"
                )

    return ValidationResult(ok=True, n_requests=n, file_size_mb=size_mb)

