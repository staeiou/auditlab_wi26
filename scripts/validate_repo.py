#!/usr/bin/env python3
"""Lightweight repository smoke check for grader-facing reproducibility."""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "audit_1_ai_gen_detect/README.md",
    "audit_1_ai_gen_detect/ai_gen_false/README.md",
    "audit_1_ai_gen_detect/ai_gen_false/analysis.ipynb",
    "audit_1_ai_gen_detect/ai_gen_false/experiment.py",
    "audit_1_ai_gen_detect/ai_gen_true/README.md",
    "audit_1_ai_gen_detect/ai_gen_true/analysis.ipynb",
    "audit_1_ai_gen_detect/ai_gen_true/experiment.py",
    "audit_2_employment_screening/README.md",
    "audit_2_employment_screening/analysis.ipynb",
    "audit_2_employment_screening/results.csv",
    "audit_3_legal/README.md",
    "audit_3_legal/judge_audit/README.md",
    "audit_3_legal/judge_audit/analysis.ipynb",
    "audit_3_legal/judge_audit/results.csv",
    "audit_4_mental_health/README.md",
    "audit_4_mental_health/analysis.ipynb",
    "audit_4_mental_health/medical_data.csv",
    "audit_5_service_eval/README.md",
    "audit_5_service_eval/layoffs/README.md",
    "audit_5_service_eval/layoffs/analysis.ipynb",
    "audit_5_service_eval/layoffs/experiment.py",
    "audit_5_service_eval/reward/README.md",
    "audit_5_service_eval/reward/analysis.ipynb",
    "audit_5_service_eval/reward/experiment.py",
]

PYTHON_ENTRYPOINTS = [
    "audit_1_ai_gen_detect/ai_gen_false/experiment.py",
    "audit_1_ai_gen_detect/ai_gen_true/experiment.py",
    "audit_3_legal/judge_audit/model_data/GPT-4o/experiment.py",
    "audit_5_service_eval/layoffs/experiment.py",
    "audit_5_service_eval/reward/experiment.py",
]


def check_required_files() -> list[str]:
    missing = []
    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).exists():
            missing.append(relative_path)
    return missing


def check_legal_model_bundles() -> list[str]:
    problems = []
    model_root = ROOT / "audit_3_legal" / "judge_audit" / "model_data"
    model_dirs = sorted(path for path in model_root.iterdir() if path.is_dir())

    if not model_dirs:
        return ["No model bundles found in audit_3_legal/judge_audit/model_data"]

    for model_dir in model_dirs:
        for expected_name in ("README.md", "experiment.py", "requirements.txt", "trial_config.json"):
            if not (model_dir / expected_name).exists():
                problems.append(f"{model_dir.relative_to(ROOT)}/{expected_name}")
    return problems


def compile_entrypoints() -> list[str]:
    failures = []
    for relative_path in PYTHON_ENTRYPOINTS:
        try:
            py_compile.compile(str(ROOT / relative_path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{relative_path}: {exc.msg}")
    return failures


def main() -> int:
    missing_files = check_required_files()
    model_bundle_problems = check_legal_model_bundles()
    compile_failures = compile_entrypoints()

    if missing_files:
        print("Missing required files:")
        for path in missing_files:
            print(f"  - {path}")

    if model_bundle_problems:
        print("Problems in legal model bundles:")
        for path in model_bundle_problems:
            print(f"  - {path}")

    if compile_failures:
        print("Python compilation failures:")
        for failure in compile_failures:
            print(f"  - {failure}")

    if missing_files or model_bundle_problems or compile_failures:
        return 1

    model_count = len(
        [path for path in (ROOT / "audit_3_legal" / "judge_audit" / "model_data").iterdir() if path.is_dir()]
    )
    print("Repository validation passed.")
    print(f"Checked {len(REQUIRED_FILES)} required files.")
    print(f"Validated {model_count} legal model bundles.")
    print(f"Compiled {len(PYTHON_ENTRYPOINTS)} Python entry points.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
