#!/usr/bin/env python3
"""
scripts/generate_batch_payloads.py

Generate OpenAI Batch-style JSONL request files (one JSONL per model), plus CSV,
and optional Parquet, under:

  <project_root>/<paths.batch_input_dir>/<safe_model_name>/

Config files expected in config/:
  - config.yaml
  - limits.yaml
  - logging.yaml
  - pricing.yaml

Prompt + variables discovery (robust):
  1) If config.yaml contains `prompt_template` (string), use it.
     Else if config.yaml contains `prompt_file` (path), read it.
     Else try (first existing):
       - <project_root>/prompt.txt
       - <project_root>/prompts/prompt.txt
       - <project_root>/config/prompt.txt

  2) If config.yaml contains `variables` (mapping), use it.
     Else if config.yaml contains `variables_file` (path), read it (YAML with top-level `variables:`).
     Else try (first existing):
       - <project_root>/variables.yaml
       - <project_root>/prompts/variables.yaml
       - <project_root>/config/variables.yaml

Input variables must match $placeholders in prompt_template (Template.safe_substitute is used).

Usage:
  python scripts/generate_batch_payloads.py
  python scripts/generate_batch_payloads.py --write-parquet
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import zipfile
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config_bundle, safe_model_dirname  # type: ignore


# ----------------------------
# Small helpers
# ----------------------------
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML top-level must be a mapping: {path}")
    return data


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _coerce_variables(obj: Any) -> Dict[str, List[str]]:
    if not isinstance(obj, dict) or not obj:
        raise ValueError("variables must be a non-empty mapping of {var_name: [values...]}")
    out: Dict[str, List[str]] = {}
    for k, vals in obj.items():
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"variables.{k} must be a non-empty list")
        out[str(k)] = [str(v) for v in vals]
    return out


def _find_first_existing(candidates: List[Path]) -> Path | None:
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _load_prompt(cfg: dict) -> str:
    # 1) inline prompt
    pt = cfg.get("prompt_template")
    if isinstance(pt, str) and pt.strip():
        return pt

    # 2) prompt file from config.yaml
    pf = cfg.get("prompt_file")
    if isinstance(pf, str) and pf.strip():
        p = (PROJECT_ROOT / pf).resolve()
        if not p.exists():
            raise FileNotFoundError(f"config.yaml prompt_file not found: {p}")
        return _read_text(p)

    # 3) fallback files
    p = _find_first_existing(
        [
            PROJECT_ROOT / "prompt.txt",
            PROJECT_ROOT / "prompts" / "prompt.txt",
            PROJECT_ROOT / "config" / "prompt.txt",
        ]
    )
    if p:
        return _read_text(p)

    raise FileNotFoundError(
        "Prompt template not found.\n"
        "Provide ONE of:\n"
        "  - config.yaml: prompt_template: |\n"
        "  - config.yaml: prompt_file: path/to/prompt.txt\n"
        "  - a file at: prompt.txt, prompts/prompt.txt, or config/prompt.txt"
    )


def _load_variables(cfg: dict) -> Dict[str, List[str]]:
    # 1) inline variables
    v = cfg.get("variables")
    if v is not None:
        return _coerce_variables(v)

    # 2) variables file from config.yaml
    vf = cfg.get("variables_file")
    if isinstance(vf, str) and vf.strip():
        p = (PROJECT_ROOT / vf).resolve()
        data = _load_yaml(p)
        if "variables" not in data:
            raise KeyError(f"{p} must contain top-level key: variables")
        return _coerce_variables(data["variables"])

    # 3) fallback files
    p = _find_first_existing(
        [
            PROJECT_ROOT / "variables.yaml",
            PROJECT_ROOT / "prompts" / "variables.yaml",
            PROJECT_ROOT / "config" / "variables.yaml",
        ]
    )
    if p:
        data = _load_yaml(p)
        if "variables" not in data:
            raise KeyError(f"{p} must contain top-level key: variables")
        return _coerce_variables(data["variables"])

    raise FileNotFoundError(
        "Variables not found.\n"
        "Provide ONE of:\n"
        "  - config.yaml: variables: {var: [..]}\n"
        "  - config.yaml: variables_file: path/to/variables.yaml (with top-level `variables:`)\n"
        "  - a file at: variables.yaml, prompts/variables.yaml, or config/variables.yaml"
    )


def _iter_combinations(variables: Dict[str, List[str]]) -> Tuple[List[str], Iterable[Tuple[str, ...]]]:
    names = list(variables.keys())
    values = [variables[n] for n in names]
    return names, itertools.product(*values)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ----------------------------
# Generation
# ----------------------------
def generate_for_model(
    *,
    model_config: dict,
    endpoint: str,
    prompt_template: str,
    variables: Dict[str, List[str]],
    out_dir: Path,
    write_parquet: bool,
) -> int:
    """
    Writes:
      - <model>_requests.jsonl
      - <model>_requests.zip
      - <model>_requests.csv
      - <model>_requests.parquet (optional)

    Returns: number of records written
    """
    model_name = str(model_config["name"])
    model_dirname = safe_model_dirname(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / f"{model_dirname}_requests.jsonl"
    zip_path = out_dir / f"{model_dirname}_requests.zip"
    csv_path = out_dir / f"{model_dirname}_requests.csv"
    parquet_path = out_dir / f"{model_dirname}_requests.parquet"

    var_names, combos = _iter_combinations(variables)

    # CSV header: custom_id, model, <model params>, <variables>, payload
    model_param_keys = [k for k in model_config.keys() if k != "name"]
    csv_fieldnames = ["custom_id", "model"] + model_param_keys + var_names + ["payload"]

    n_records = 0
    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"

    with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fieldnames)
        writer.writeheader()

        for idx, combo in enumerate(combos):
            var_dict = dict(zip(var_names, combo))

            prompt = Template(prompt_template).safe_substitute(var_dict)
            custom_id = f"request-{idx + 1}"

            payload = {
                "custom_id": custom_id,
                "method": "POST",
                "url": endpoint,
                "body": {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    **{k: v for k, v in model_config.items() if k != "name"},
                },
            }

            jf.write(_json_dumps(payload) + "\n")

            row: Dict[str, Any] = {"custom_id": custom_id, "model": model_name}
            for k in model_param_keys:
                row[k] = model_config.get(k)
            for k in var_names:
                row[k] = var_dict.get(k)
            row["payload"] = _json_dumps(payload)
            writer.writerow(row)

            n_records += 1

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(jsonl_path, arcname=jsonl_path.name)

    parquet_note = "skipped"
    if write_parquet:
        try:
            df = pd.read_csv(csv_path)
            df.to_parquet(parquet_path, index=False)
            parquet_note = "written"
        except Exception as e:
            parquet_note = f"failed ({type(e).__name__})"

    print(f"✓ {out_dir}/")
    print(f"  Model: {model_name}")
    print(f"  Records: {n_records}")
    print(f"  Files: {jsonl_path.name}, {zip_path.name}, {csv_path.name}, parquet={parquet_note}\n")

    return n_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="config", help="Config directory (default: config)")
    ap.add_argument("--write-parquet", action="store_true", help="Also write Parquet (requires pyarrow)")
    args = ap.parse_args()

    config_dir = (PROJECT_ROOT / args.config_dir).resolve()
    bundle = load_config_bundle(project_root=PROJECT_ROOT, config_dir=config_dir)

    cfg = bundle.config

    # Prompt + variables (may live outside config/; discovery logic above)
    prompt_template = _load_prompt(cfg)
    variables = _load_variables(cfg)

    # Output base
    batch_input_dir = Path(cfg.get("paths", {}).get("batch_input_dir", "batch"))
    base_out = (PROJECT_ROOT / batch_input_dir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    endpoint = cfg.get("openai_batch", {}).get("endpoint", "/v1/chat/completions")

    total_combinations = math.prod(len(v) for v in variables.values())
    print("Generating batch payloads...")
    print(f"Variables: {len(variables)} fields")
    print(f"Total combinations: {total_combinations}")
    print(f"Models: {len(bundle.model_configs)}")
    print(f"Endpoint: {endpoint}")
    print(f"Output base: {base_out}\n")

    # Optional: warn if combinations exceed openai batch limits
    lim = (bundle.limits.get("openai_batch", {}) or {})
    max_reqs = int(lim.get("max_requests_per_file", 50_000))
    if total_combinations > max_reqs:
        print(
            f"WARNING: total combinations ({total_combinations}) exceed limits.yaml "
            f"openai_batch.max_requests_per_file ({max_reqs}).\n"
            "You should reduce the variable grid or split into multiple input files per model.\n"
        )

    for mc in bundle.model_configs:
        model = str(mc["name"])
        out_dir = base_out / safe_model_dirname(model)
        generate_for_model(
            model_config=mc,
            endpoint=endpoint,
            prompt_template=prompt_template,
            variables=variables,
            out_dir=out_dir,
            write_parquet=args.write_parquet,
        )

    print(f"Done! Wrote per-model requests under: {base_out}/")


if __name__ == "__main__":
    main()

