# project_root/src/openai_batch/parser.py

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def _strip_code_fences(text: str) -> str:
    """
    Remove ```json ... ``` fences if present.
    """
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Best-effort parse of a JSON object from model output.

    Handles:
      - raw JSON object
      - JSON object inside ``` fences
      - JSON object embedded in extra text (extract first '{'..last '}')
    Returns a dict if successful, else None.
    """
    if not text:
        return None

    t = _strip_code_fences(text)

    # Direct parse
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Substring heuristic
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        sub = t[start : end + 1]
        try:
            obj = json.loads(sub)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


def _parsed_json_string(text: Optional[str]) -> Optional[str]:
    """
    Returns a JSON string of the parsed object if it is a dict, else None.
    Uses ensure_ascii=False to preserve non-English characters.
    """
    parsed = _extract_json_object(text or "")
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    return None


def _extract_chat_completion_text(resp_body: dict) -> Optional[str]:
    """
    Extract assistant text from a /v1/chat/completions response body.

    Expected shape:
      body["choices"][0]["message"]["content"]

    Fallback:
      body["choices"][0]["text"]
    """
    try:
        choices = resp_body.get("choices", [])
        if not choices:
            return None
        c0 = choices[0]
        msg = c0.get("message", {})
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
        if isinstance(c0.get("text"), str):
            return c0["text"]
    except Exception:
        return None
    return None


def _load_input_features_from_csv(
    requests_csv_path: Path,
    include_input_features: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Load per-custom_id input features from the generator CSV.

    Returns: {custom_id: {custom_id:..., feature1:..., ...}}
    """
    if not requests_csv_path.exists():
        return {}

    df = pd.read_csv(requests_csv_path)
    cols = ["custom_id"] + [c for c in include_input_features if c in df.columns]
    df = df[cols]

    out: Dict[str, Dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        cid = str(row["custom_id"])
        out[cid] = row
    return out


def parse_batch_output_jsonl_to_rows(
    *,
    output_jsonl_path: Path,
    model: str,
    input_features_by_custom_id: Optional[Dict[str, Dict[str, Any]]] = None,
    include_input_features: Optional[List[str]] = None,
    include_output_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Parse OpenAI Batch output JSONL into generic "compact" rows.

    This is STUDY-AGNOSTIC: it does NOT hardcode any expected keys inside the
    model's returned JSON (like "urgency_score"). Instead it can store:
      - raw_response: assistant text
      - parsed_json: JSON string of parsed dict if parse succeeds, else None
      - ok: bool
      - error: json string if present

    include_output_fields controls which of the above are included. Unknown fields
    are filled with None (keeps schemas stable if you change config later).

    Returns rows (list of dict).
    """
    if not output_jsonl_path.exists():
        raise FileNotFoundError(f"Output JSONL not found: {output_jsonl_path}")

    include_input_features = include_input_features or []
    include_output_fields = include_output_fields or ["raw_response", "parsed_json", "ok", "error"]

    feats_map = input_features_by_custom_id or {}

    rows: List[Dict[str, Any]] = []

    for obj in _read_jsonl(output_jsonl_path):
        custom_id = str(obj.get("custom_id"))
        resp = obj.get("response")
        err = obj.get("error")

        status_code = None
        resp_text: Optional[str] = None
        ok = False

        if isinstance(resp, dict):
            status_code = resp.get("status_code")
            body = resp.get("body", {})
            if isinstance(body, dict):
                resp_text = _extract_chat_completion_text(body)
            ok = (err is None) and (status_code == 200)

        parsed_json = _parsed_json_string(resp_text)

        row: Dict[str, Any] = {"custom_id": custom_id, "model": model}

        # attach requested input features
        feats = feats_map.get(custom_id, {})
        for k in include_input_features:
            row[k] = feats.get(k)

        # attach requested output fields (generic, study-agnostic)
        for k in include_output_fields:
            if k == "raw_response":
                row[k] = resp_text
            elif k == "parsed_json":
                row[k] = parsed_json
            elif k == "ok":
                row[k] = ok
            elif k == "error":
                row[k] = json.dumps(err, ensure_ascii=False) if err else None
            else:
                row[k] = None

        rows.append(row)

    return rows


def write_compact_results_csv(
    *,
    output_jsonl_path: Path,
    requests_csv_path: Optional[Path],
    out_csv_path: Path,
    model: str,
    include_input_features: List[str],
    include_output_fields: List[str],
) -> Path:
    """
    Write compact results CSV to out_csv_path.

    - Loads requested input features from requests_csv_path (if provided).
    - Parses output_jsonl_path and writes compact rows.

    Returns out_csv_path.
    """
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    feats_map: Dict[str, Dict[str, Any]] = {}
    if requests_csv_path is not None and requests_csv_path.exists():
        feats_map = _load_input_features_from_csv(requests_csv_path, include_input_features)

    rows = parse_batch_output_jsonl_to_rows(
        output_jsonl_path=output_jsonl_path,
        model=model,
        input_features_by_custom_id=feats_map,
        include_input_features=include_input_features,
        include_output_fields=include_output_fields,
    )

    fieldnames = ["custom_id", "model"] + include_input_features + include_output_fields
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return out_csv_path

