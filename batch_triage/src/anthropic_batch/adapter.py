# project_root/src/anthropic_batch/adapter.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def _content_to_text(content: Any) -> str:
    """
    OpenAI-style message content may be:
      - str
      - list of blocks (e.g., [{"type":"text","text":"..."}, ...])
    We only keep text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for blk in content:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                parts.append(blk["text"])
            elif isinstance(blk, str):
                parts.append(blk)
        return "\n".join([p for p in parts if p])
    # fallback
    return str(content)


def _split_system_messages(
    openai_messages: List[Dict[str, Any]],
    *,
    system_role_strategy: str = "collect_to_system",
    drop_unsupported_roles: bool = True,
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """
    Convert OpenAI chat messages to Anthropic messages.

    Anthropic Messages API supports roles: "user", "assistant".
    "system" is a top-level field (string or list of blocks). We'll store as string.

    Returns: (system_text_or_None, anthropic_messages)
    """
    system_parts: List[str] = []
    out_msgs: List[Dict[str, str]] = []

    for m in openai_messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = _content_to_text(m.get("content"))

        if role == "system":
            if system_role_strategy == "collect_to_system":
                if content.strip():
                    system_parts.append(content.strip())
            else:
                # If you ever want to force system into user, do it here.
                if content.strip():
                    out_msgs.append({"role": "user", "content": content.strip()})
            continue

        if role in ("user", "assistant"):
            out_msgs.append({"role": role, "content": content})
            continue

        # Other roles: tool/function/developer/etc.
        if drop_unsupported_roles:
            continue
        # Otherwise, coerce to user with a tag
        if content.strip():
            out_msgs.append({"role": "user", "content": f"[{role}] {content.strip()}"})

    system_text = "\n\n".join(system_parts).strip() if system_parts else None
    return system_text, out_msgs


def openai_body_to_anthropic_params(
    openai_body: Dict[str, Any],
    *,
    force_model: Optional[str] = None,
    system_role_strategy: str = "collect_to_system",
    max_tokens_field: str = "max_tokens",
    drop_unsupported_fields: bool = True,
    drop_unsupported_roles: bool = True,
) -> Dict[str, Any]:
    """
    Convert an OpenAI /chat/completions-style body into Anthropic Messages API params.
    This is designed to be used inside Anthropic Batch requests.

    Key mappings:
      - model: force_model or openai_body["model"]
      - messages: from openai_body["messages"], with system messages moved to top-level 'system'
      - max_completion_tokens -> max_tokens (if needed)
      - stop -> stop_sequences
    """
    if not isinstance(openai_body, dict):
        openai_body = {}

    model = force_model or openai_body.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Missing model for Anthropic params (force_model or body.model required)")

    # messages
    raw_msgs = openai_body.get("messages", [])
    if not isinstance(raw_msgs, list):
        raw_msgs = []
    system_text, anthropic_messages = _split_system_messages(
        raw_msgs,
        system_role_strategy=system_role_strategy,
        drop_unsupported_roles=drop_unsupported_roles,
    )

    # max_tokens mapping
    max_tokens = openai_body.get(max_tokens_field)
    if max_tokens is None and "max_completion_tokens" in openai_body:
        max_tokens = openai_body.get("max_completion_tokens")

    # Temperature / top_p (Anthropic supports temperature, and top_p for nucleus sampling)
    temperature = openai_body.get("temperature")
    top_p = openai_body.get("top_p")

    # stop sequences mapping
    stop_sequences = None
    if "stop_sequences" in openai_body:
        stop_sequences = openai_body.get("stop_sequences")
    elif "stop" in openai_body:
        stop_sequences = openai_body.get("stop")

    params: Dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
    }
    if system_text:
        params["system"] = system_text
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if isinstance(temperature, (int, float)):
        params["temperature"] = float(temperature)
    if isinstance(top_p, (int, float)):
        params["top_p"] = float(top_p)

    # stop can be string or list in OpenAI; Anthropic expects list[str]
    if stop_sequences is not None:
        if isinstance(stop_sequences, str):
            params["stop_sequences"] = [stop_sequences]
        elif isinstance(stop_sequences, list):
            params["stop_sequences"] = [str(x) for x in stop_sequences if x is not None]

    if drop_unsupported_fields:
        # Intentionally ignore OpenAI-only keys like: response_format, n, logprobs, tools, tool_choice, etc.
        pass

    # Ensure there's at least one user message (Anthropic will error otherwise)
    if not any(m.get("role") == "user" for m in anthropic_messages):
        # If you had only system content, convert it into a user message
        if system_text:
            params.pop("system", None)
            params["messages"] = [{"role": "user", "content": system_text}]
        else:
            params["messages"] = [{"role": "user", "content": ""}]

    return params


def build_anthropic_batch_requests_from_openai_jsonl(
    *,
    input_jsonl_path: Path,
    force_model: Optional[str] = None,
    system_role_strategy: str = "collect_to_system",
    max_tokens_field: str = "max_tokens",
    drop_unsupported_fields: bool = True,
    drop_unsupported_roles: bool = True,
) -> List[Dict[str, Any]]:
    """
    Read your OpenAI-batch-style JSONL (one request per line) and output
    Anthropic batch 'requests' entries: [{"custom_id":..., "params":{...}}, ...]
    """
    out: List[Dict[str, Any]] = []

    for rec in _read_jsonl(input_jsonl_path):
        cid = rec.get("custom_id")
        if not isinstance(cid, str) or not cid:
            continue

        body = rec.get("body") or {}
        if not isinstance(body, dict):
            body = {}

        params = openai_body_to_anthropic_params(
            body,
            force_model=force_model,
            system_role_strategy=system_role_strategy,
            max_tokens_field=max_tokens_field,
            drop_unsupported_fields=drop_unsupported_fields,
            drop_unsupported_roles=drop_unsupported_roles,
        )

        out.append({"custom_id": cid, "params": params})

    return out


def build_anthropic_submit_payload(
    *,
    input_jsonl_path: Path,
    force_model: Optional[str] = None,
    system_role_strategy: str = "collect_to_system",
    max_tokens_field: str = "max_tokens",
    drop_unsupported_fields: bool = True,
    drop_unsupported_roles: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper that returns the exact JSON body you POST to
    POST /v1/messages/batches:

      {"requests": [{"custom_id":..., "params": {...}}, ...]}
    """
    reqs = build_anthropic_batch_requests_from_openai_jsonl(
        input_jsonl_path=input_jsonl_path,
        force_model=force_model,
        system_role_strategy=system_role_strategy,
        max_tokens_field=max_tokens_field,
        drop_unsupported_fields=drop_unsupported_fields,
        drop_unsupported_roles=drop_unsupported_roles,
    )
    return {"requests": reqs}

