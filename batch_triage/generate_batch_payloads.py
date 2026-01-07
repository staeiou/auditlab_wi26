#!/usr/bin/env python3
"""
generate_batch_payloads.py

Generates OpenAI Batch-style JSONL request files (one JSONL per model config),
plus optional CSV/Parquet summaries, under project_root/batch (or the path
configured in config/config.yaml).

Expected repo layout (from your proposed structure):
project_root/
  config/
    config.yaml
    variables.yaml          # variables used for Cartesian product
    prompt.txt              # full prompt template with $placeholders
    # (optional) prompt.yaml instead of prompt.txt, with key: prompt_template
  src/
    generate_batch_payloads.py
  batch/                    # output (auto-created)
"""

from __future__ import annotations

import itertools
import json
import math
import zipfile
from pathlib import Path
from string import Template
from typing import Any, Dict, List

import pandas as pd
import yaml


# -------------------------
# Paths / config loading
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def _safe_model_dirname(model: str) -> str:
    return model.replace("/", "_")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_prompt_template(cfg: dict) -> str:
    """
    Load prompt template from (first match wins):
      1) config/config.yaml key: prompt_template  (backwards-compatible)
      2) config/prompt.yaml key: prompt_template
      3) config/prompt.txt (raw text)
    """
    if "prompt_template" in cfg and isinstance(cfg["prompt_template"], str) and cfg["prompt_template"].strip():
        return cfg["prompt_template"]

    prompt_yaml = CONFIG_DIR / "prompt.yaml"
    if prompt_yaml.exists():
        py = _load_yaml(prompt_yaml)
        if "prompt_template" not in py:
            raise KeyError(f"{prompt_yaml} must contain key: prompt_template")
        return str(py["prompt_template"])

    prompt_txt = CONFIG_DIR / "prompt.txt"
    if prompt_txt.exists():
        return prompt_txt.read_text(encoding="utf-8")

    raise FileNotFoundError(
        "No prompt template found. Provide one of:\n"
        " - config/config.yaml: prompt_template\n"
        " - config/prompt.yaml: prompt_template\n"
        " - config/prompt.txt"
    )


def _load_variables(cfg: dict) -> Dict[str, List[str]]:
    """
    Load variables from (first match wins):
      1) config/variables.yaml key: variables
      2) config/config.yaml key: variables (backwards-compatible)
    """
    var_yaml = CONFIG_DIR / "variables.yaml"
    if var_yaml.exists():
        vy = _load_yaml(var_yaml)
        if "variables" not in vy:
            raise KeyError(f"{var_yaml} must contain key: variables")
        variables = vy["variables"]
    else:
        if "variables" not in cfg:
            raise FileNotFoundError(
                "No variables found. Provide config/variables.yaml with key: variables "
                "or include `variables:` in config/config.yaml."
            )
        variables = cfg["variables"]

    if not isinstance(variables, dict) or not variables:
        raise ValueError("variables must be a non-empty mapping of {var_name: [values...]}")
    # Normalize to strings
    out: Dict[str, List[str]] = {}
    for k, vals in variables.items():
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"variables.{k} must be a non-empty list")
        out[str(k)] = [str(v) for v in vals]
    return out


# -------------------------
# Generation
# -------------------------
def generate_payloads_for_model(
    *,
    model_config: dict,
    prompt_template: str,
    variables: Dict[str, List[str]],
    request_method: str,
    request_url: str,
    jsonl_file: Path,
) -> List[dict]:
    """
    Generate all payloads for a single model configuration.

    Streams payloads to JSONL file to avoid holding everything in memory.
    Returns records list for CSV/Parquet export.
    """
    records: List[dict] = []

    var_names = list(variables.keys())
    var_values = [variables[name] for name in var_names]

    with jsonl_file.open("w", encoding="utf-8") as f:
        for idx, combo in enumerate(itertools.product(*var_values)):
            var_dict = dict(zip(var_names, combo))

            prompt = Template(prompt_template).safe_substitute(var_dict)
            custom_id = f"request-{idx + 1}"

            payload = {
                "custom_id": custom_id,
                "method": request_method,
                "url": request_url,
                "body": {
                    "model": model_config["name"],
                    "messages": [{"role": "user", "content": prompt}],
                    **{k: v for k, v in model_config.items() if k != "name"},
                },
            }

            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            # Record row for analysis (do NOT store full prompt by default)
            record = {"custom_id": custom_id, "model": model_config["name"]}
            for k, v in model_config.items():
                if k != "name":
                    record[k] = v
            record.update(var_dict)
            record["payload"] = json.dumps(payload, ensure_ascii=False)
            records.append(record)

    return records


def main() -> None:
    # Load main config
    cfg = _load_yaml(CONFIG_DIR / "config.yaml")

    # Output location (defaults to project_root/batch)
    batch_input_dir = Path(cfg.get("paths", {}).get("batch_input_dir", "batch"))
    base_dir = (PROJECT_ROOT / batch_input_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    # Batch endpoint for request wrapper
    openai_batch = cfg.get("openai_batch", {}) or {}
    request_url = openai_batch.get("endpoint", "/v1/chat/completions")
    request_method = "POST"

    # Load prompt + variables
    prompt_template = _load_prompt_template(cfg)
    variables = _load_variables(cfg)

    # Models
    model_configs = cfg.get("model_configs")
    if not isinstance(model_configs, list) or not model_configs:
        raise ValueError("config/config.yaml must contain non-empty list: model_configs")

    print("Generating batch payloads...")
    total_combinations = math.prod(len(v) for v in variables.values())
    print(f"Total combinations: {total_combinations}")
    print(f"Model configurations: {len(model_configs)}")
    print(f"Output directory: {base_dir}")
    print(f"Request URL: {request_url}\n")

    for model_config in model_configs:
        if "name" not in model_config:
            raise ValueError(f"Each model_config must contain 'name': {model_config}")

        model_name = _safe_model_dirname(model_config["name"])
        model_dir = base_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        jsonl_file = model_dir / f"{model_name}_requests.jsonl"

        records = generate_payloads_for_model(
            model_config=model_config,
            prompt_template=prompt_template,
            variables=variables,
            request_method=request_method,
            request_url=request_url,
            jsonl_file=jsonl_file,
        )

        # Zip JSONL (max compression)
        zip_file = model_dir / f"{model_name}_requests.zip"
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.write(jsonl_file, arcname=jsonl_file.name)

        # CSV + Parquet (Parquet optional depending on pyarrow)
        df = pd.DataFrame(records)
        csv_file = model_dir / f"{model_name}_requests.csv"
        df.to_csv(csv_file, index=False)

        parquet_file = model_dir / f"{model_name}_requests.parquet"
        try:
            df.to_parquet(parquet_file, index=False)
            parquet_note = "parquet ✓"
        except Exception as e:
            parquet_note = f"parquet skipped ({type(e).__name__})"

        print(f"✓ {model_dir}/")
        print(f"  Model: {model_config['name']}")
        print(f"  Records: {len(records)}")
        print(f"  Files: {jsonl_file.name}, {zip_file.name}, {csv_file.name}, {parquet_note}\n")

    print(f"Done! Generated files for {len(model_configs)} models in {base_dir}/")


if __name__ == "__main__":
    main()

