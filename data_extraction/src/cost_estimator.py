# project_root/src/cost_estimator.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class CostEstimate:
    model: str
    n_requests: int
    est_input_tokens: int
    est_output_tokens: int
    price_input_per_1m: float
    price_output_per_1m: float
    est_cost_usd: float


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _try_count_tokens(text: str, model: str) -> int:
    """
    Best-effort token estimate:
      - if tiktoken installed -> use it
      - else heuristic ~4 chars/token
    """
    try:
        import tiktoken  # type: ignore

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _extract_request_text(obj: dict) -> str:
    """
    Extract prompt text from an OpenAI Batch request line.
    Supports your current format:
      {"body":{"messages":[{"content":...}, ...]}}
    """
    body = obj.get("body", {})
    messages = body.get("messages", [])
    if isinstance(messages, list) and messages:
        parts: List[str] = []
        for m in messages:
            if isinstance(m, dict):
                c = m.get("content")
                if isinstance(c, str):
                    parts.append(c)
        return "\n".join(parts)

    # Fallbacks (in case you switch formats later)
    if isinstance(body.get("input"), str):
        return body["input"]
    return ""


def _base_model_name(model: str, pricing_cfg: dict) -> str:
    aliases = pricing_cfg.get("aliases") or {}
    if not isinstance(aliases, dict):
        aliases = {}
    return aliases.get(model, model)


def _get_prices_per_1m(model: str, tier: str, pricing_cfg: dict) -> Tuple[float, float]:
    tiers = pricing_cfg.get("tiers") or {}
    if not isinstance(tiers, dict) or tier not in tiers:
        raise KeyError(f"pricing.yaml missing tier={tier!r}. Available: {list(tiers.keys())}")

    base = _base_model_name(model, pricing_cfg)
    tier_map = tiers[tier]
    if base not in tier_map:
        raise KeyError(f"pricing.yaml has no pricing for model={model!r} (base={base!r}) in tier={tier!r}")

    entry = tier_map[base]
    try:
        p_in = float(entry["input"])
        p_out = float(entry["output"])
    except Exception as e:
        raise ValueError(f"Bad pricing entry for {base} in tier {tier}: {entry}") from e

    return p_in, p_out


def estimate_cost_for_jsonl(
    *,
    jsonl_path: Path,
    model: str,
    pricing_cfg: dict,
    pricing_tier: str,
    assumed_output_tokens_per_request: int,
    sample_n_requests: int = 200,
) -> CostEstimate:
    """
    Estimate cost for one model's JSONL input.

    Strategy:
      - Count total requests exactly
      - Sample up to N requests to estimate avg input tokens
      - Multiply avg input tokens by total requests
      - Output tokens use assumed_output_tokens_per_request
      - Cost uses pricing.yaml (tier + alias mapping)
    """
    total = 0
    sample_tokens: List[int] = []

    for obj in _read_jsonl(jsonl_path):
        total += 1
        if len(sample_tokens) < sample_n_requests:
            text = _extract_request_text(obj)
            sample_tokens.append(_try_count_tokens(text, model))

    if total == 0:
        p_in, p_out = _get_prices_per_1m(model, pricing_tier, pricing_cfg)
        return CostEstimate(
            model=model,
            n_requests=0,
            est_input_tokens=0,
            est_output_tokens=0,
            price_input_per_1m=p_in,
            price_output_per_1m=p_out,
            est_cost_usd=0.0,
        )

    avg_in = sum(sample_tokens) / max(1, len(sample_tokens))
    est_in = int(avg_in * total)
    est_out = int(assumed_output_tokens_per_request * total)

    p_in, p_out = _get_prices_per_1m(model, pricing_tier, pricing_cfg)
    est_cost = (est_in / 1_000_000) * p_in + (est_out / 1_000_000) * p_out

    return CostEstimate(
        model=model,
        n_requests=total,
        est_input_tokens=est_in,
        est_output_tokens=est_out,
        price_input_per_1m=p_in,
        price_output_per_1m=p_out,
        est_cost_usd=est_cost,
    )


def estimate_costs_for_models(
    *,
    per_model_jsonl: Dict[str, Path],
    pricing_cfg: dict,
    pricing_tier: str,
    assumed_output_tokens_per_request: int,
    sample_n_requests: int = 200,
) -> Dict[str, CostEstimate]:
    """
    Convenience helper to estimate cost for multiple models.

    per_model_jsonl: {model_name: jsonl_path}
    """
    out: Dict[str, CostEstimate] = {}
    for model, path in per_model_jsonl.items():
        out[model] = estimate_cost_for_jsonl(
            jsonl_path=path,
            model=model,
            pricing_cfg=pricing_cfg,
            pricing_tier=pricing_tier,
            assumed_output_tokens_per_request=assumed_output_tokens_per_request,
            sample_n_requests=sample_n_requests,
        )
    return out

