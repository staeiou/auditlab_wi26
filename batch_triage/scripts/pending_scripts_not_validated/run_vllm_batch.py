#!/usr/bin/env python3
"""
scripts/run_vllm_job.py

Run OpenAI-batch-style JSONL requests using vLLM's OFFLINE batch runner.

Key properties:
- pending_requests.jsonl is built ONCE per model (limit-n applied once)
- dry-run does NOT write any files (true no-side-effects)
- before each run_batch invocation: delete raw_batch_results.jsonl to avoid stale salvage
- append normalized results into output.jsonl, skipping any custom_id already present (no duplicates)
- on run-batch crash: salvage partial raw results if present, then record runner-failure
  only for remaining pending ids (excluding already-done ids)
- errors.jsonl is rewritten from output.jsonl each run (no stale/duplicate contradictions)
- errors_history.jsonl is append-only for debugging/audit trail of runner failures
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config_bundle, safe_model_dirname  # type: ignore
from src.cost_estimator import estimate_cost_for_jsonl  # type: ignore
from src.openai_batch.validator import BatchInputLimits, validate_batch_jsonl  # type: ignore
from src.openai_batch.parser import write_compact_results_csv  # type: ignore
from src.wandb_logging.run_manager import init_wandb_run  # type: ignore


# ---------------------------
# Helpers: extraction / resume
# ---------------------------

def _extract_text_from_chat_completions_body(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        msg = (choices[0] or {}).get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return None


def _extract_text_from_any_body(body: Any) -> Optional[str]:
    if body is None:
        return None
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        t = _extract_text_from_chat_completions_body(body)
        if t is not None:
            return t
        for k in ("output_text", "text"):
            v = body.get(k)
            if isinstance(v, str):
                return v
        v = body.get("content")
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            parts = []
            for blk in v:
                if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                    parts.append(blk["text"])
            if parts:
                return "\n".join(parts)
    return None


def _load_done_custom_ids(output_jsonl: Path) -> Set[str]:
    """
    Resume helper: collect all custom_ids present in output.jsonl.
    (This treats failures as 'done' too. Add a --rerun-failed flag later if desired.)
    Ignores malformed JSON lines (truncated last line etc). Warns on OS errors.
    """
    done: Set[str] = set()
    if not output_jsonl.exists():
        return done
    try:
        with output_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = rec.get("custom_id")
                if isinstance(cid, str) and cid:
                    done.add(cid)
    except OSError as e:
        print(f"[WARN] Could not read {output_jsonl}: {e}", file=sys.stderr)
    return done


def _normalize_body_for_vllm(body: Dict[str, Any], served_model: str) -> Dict[str, Any]:
    """
    Normalize request body for vLLM (deep copy to avoid mutating shared/nested dicts):
      - force model name
      - map max_completion_tokens -> max_tokens if max_tokens missing
    """
    b = copy.deepcopy(body)
    b["model"] = served_model
    if "max_tokens" not in b and "max_completion_tokens" in b:
        b["max_tokens"] = b["max_completion_tokens"]
    return b


def _paths_for_model(batch_input_dir: Path, outputs_root: Path, model: str) -> Dict[str, Path]:
    model_dir_str = safe_model_dirname(model)  # expected to be str
    model_batch_dir = batch_input_dir / model_dir_str
    requests_jsonl = model_batch_dir / f"{model_dir_str}_requests.jsonl"
    requests_csv = model_batch_dir / f"{model_dir_str}_requests.csv"

    out_dir = outputs_root / model_dir_str
    return {
        "model_dir": Path(model_dir_str),
        "batch_dir": model_batch_dir,
        "requests_jsonl": requests_jsonl,
        "requests_csv": requests_csv,
        "out_dir": out_dir,
        "out_jsonl": out_dir / "output.jsonl",
        "err_jsonl": out_dir / "errors.jsonl",                 # derived from output.jsonl (rewritten)
        "err_history_jsonl": out_dir / "errors_history.jsonl", # append-only history
        "compact_csv": out_dir / "results_compact.csv",
        "joined_jsonl": out_dir / "results_joined.jsonl",
        "pending_jsonl": out_dir / "pending_requests.jsonl",
        "raw_results_jsonl": out_dir / "raw_batch_results.jsonl",
        "run_batch_log": out_dir / "run_batch.log",
    }


def write_local_joined_results_jsonl(
    *,
    output_jsonl_path: Path,
    requests_csv_path: Path,
    out_jsonl_path: Path,
    model_label: str,
    include_input_features: List[str],
) -> None:
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    id_to_inputs: Dict[str, Dict[str, Any]] = {}
    if requests_csv_path.exists():
        with requests_csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("custom_id")
                if not cid:
                    continue
                inputs = {k: row.get(k) for k in include_input_features}
                id_to_inputs[cid] = inputs

    with output_jsonl_path.open("r", encoding="utf-8") as fin, out_jsonl_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("custom_id")

            resp = rec.get("response") or {}
            status_code = None
            body = None
            if isinstance(resp, dict):
                status_code = resp.get("status_code")
                body = resp.get("body")

            joined = {
                "custom_id": cid,
                "model_label": model_label,
                "inputs": id_to_inputs.get(cid, {}) if isinstance(cid, str) else {},
                "status_code": status_code,
                "response_text": _extract_text_from_any_body(body),
                "response_body": body,
                "error": rec.get("error"),
            }
            fout.write(json.dumps(joined, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clear_file(path: Path) -> None:
    """Remove a file if it exists (best-effort). Prevents stale data from prior attempts."""
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _count_pending_without_writing(
    *,
    src_requests_jsonl: Path,
    done_ids: Set[str],
    limit_n: Optional[int],
) -> int:
    """True --dry-run: count pending requests without producing pending_jsonl."""
    n = 0
    with src_requests_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cid = rec.get("custom_id")
            if isinstance(cid, str) and cid in done_ids:
                continue
            n += 1
            if limit_n is not None and n >= limit_n:
                break
    return n


def _build_pending_jsonl(
    *,
    src_requests_jsonl: Path,
    dst_pending_jsonl: Path,
    done_ids: Set[str],
    forced_url: str,
    served_model: str,
    limit_n: Optional[int],
) -> int:
    """
    Build pending JSONL ONCE:
    - skips done custom_ids
    - forces method/url
    - normalizes body.model + max_tokens alias
    """
    dst_pending_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with src_requests_jsonl.open("r", encoding="utf-8") as fin, dst_pending_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            cid = rec.get("custom_id")
            if isinstance(cid, str) and cid in done_ids:
                continue
            if limit_n is not None and n_written >= limit_n:
                break

            body = rec.get("body") or {}
            if not isinstance(body, dict):
                body = {}

            rec["method"] = "POST"
            rec["url"] = forced_url
            rec["body"] = _normalize_body_for_vllm(body, served_model=served_model)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    return n_written


def _run_vllm_batch_runner(
    *,
    pending_jsonl: Path,
    raw_results_jsonl: Path,
    model_name: str,
    extra_args: List[str],
    log_path: Path,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.run_batch",
        "-i",
        str(pending_jsonl),
        "-o",
        str(raw_results_jsonl),
        "--model",
        model_name,
    ] + extra_args

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running: {' '.join(cmd)}\n")
        logf.flush()
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=True)


def _append_normalized_results(
    *,
    raw_results_jsonl: Path,
    out_jsonl: Path,
    skip_ids: Set[str],
) -> Set[str]:
    """
    Normalize vLLM run-batch output into output.jsonl schema and append.
    Skips any custom_id already present in skip_ids to prevent duplicates.
    Returns the set of custom_ids newly appended.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    appended_ids: Set[str] = set()

    with raw_results_jsonl.open("r", encoding="utf-8") as fin, out_jsonl.open("a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            cid = rec.get("custom_id")
            if isinstance(cid, str) and cid and cid in skip_ids:
                continue  # prevent duplicates

            resp = rec.get("response")
            err = rec.get("error")

            ok = (err is None) and (resp is not None)
            status_code = 200 if ok else 500
            body = resp if isinstance(resp, (dict, list, str)) else {"raw": resp}

            out_line = {
                "custom_id": cid,
                "response": {"status_code": status_code, "body": body},
                "error": None if ok else err,
            }
            fout.write(json.dumps(out_line, ensure_ascii=False) + "\n")

            if isinstance(cid, str) and cid:
                appended_ids.add(cid)

    return appended_ids


def _rewrite_errors_from_output(*, out_jsonl: Path, err_jsonl: Path) -> int:
    """
    Keep errors.jsonl consistent: rewrite by filtering output.jsonl.
    One line per failed record in output.jsonl.
    """
    err_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_jsonl.open("r", encoding="utf-8") as fin, err_jsonl.open("w", encoding="utf-8") as ferr:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            err = rec.get("error")
            resp = rec.get("response") or {}
            sc = resp.get("status_code") if isinstance(resp, dict) else None
            if err is not None or (isinstance(sc, int) and sc >= 400):
                ferr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


def _write_batch_failure_errors_for_remaining(
    *,
    pending_jsonl: Path,
    err_history_jsonl: Path,
    message: str,
    already_handled_ids: Set[str],
) -> int:
    """
    Append runner-failure errors ONLY for pending requests not already handled.
    Writes to errors_history_jsonl (append-only).
    """
    err_history_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with pending_jsonl.open("r", encoding="utf-8") as fin, err_history_jsonl.open("a", encoding="utf-8") as ferr:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cid = rec.get("custom_id")
            if isinstance(cid, str) and cid in already_handled_ids:
                continue
            out_line = {
                "custom_id": cid,
                "response": {"status_code": 0, "body": {"error": {"message": message}}},
                "error": {"message": message, "type": "batch_runner_failed"},
            }
            ferr.write(json.dumps(out_line, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--vllm-config", default="config/vllm.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--limit-n", type=int, default=None)
    args = ap.parse_args()

    bundle = load_config_bundle(project_root=PROJECT_ROOT, config_dir=PROJECT_ROOT / args.config_dir)

    vcfg_path = PROJECT_ROOT / args.vllm_config
    vcfg = yaml.safe_load(vcfg_path.read_text(encoding="utf-8")) or {}
    vllm_raw = vcfg.get("vllm", {})
    if not isinstance(vllm_raw, dict):
        raise ValueError(f"Expected 'vllm' to be a mapping in {vcfg_path}, got {type(vllm_raw)}")
    vllm_cfg: Dict[str, Any] = vllm_raw

    forced_url: str = str(vllm_cfg.get("batch_url", "/v1/chat/completions"))
    extra_args: List[str] = list(vllm_cfg.get("run_batch_extra_args", []) or [])
    keep_intermediates = bool(vllm_cfg.get("keep_intermediates", False))
    upload_intermediates = bool(vllm_cfg.get("upload_intermediates_to_wandb", False))

    outputs_root = PROJECT_ROOT / "outputs" / "vllm"
    outputs_root.mkdir(parents=True, exist_ok=True)

    lim = bundle.limits.get("openai_batch", {}) or {}
    limits = BatchInputLimits(
        max_requests_per_file=int(lim.get("max_requests_per_file", 50_000)),
        max_file_size_mb=int(lim.get("max_file_size_mb", 200)),
        require_single_model_per_file=bool(lim.get("require_single_model_per_file", True)),
    )

    compact_cfg = bundle.logging.get("results_compact", {}) or {}
    include_input_features = compact_cfg.get("include_input_features", []) or []
    include_output_fields = compact_cfg.get(
        "include_output_fields",
        ["raw_response", "parsed_json", "ok", "error"],
    ) or []

    ce = bundle.config.get("cost_estimation", {}) or {}
    sample_n_cfg = int(ce.get("sample_n_requests", 200))
    assumed_out = int(ce.get("assumed_output_tokens_per_request", 80))
    pricing_tier = bundle.pricing_tier

    batch_input_dir = Path(bundle.config["paths"]["batch_input_dir"])
    if not batch_input_dir.is_absolute():
        batch_input_dir = PROJECT_ROOT / batch_input_dir

    per_model: Dict[str, Dict[str, Path]] = {}
    for mc in bundle.model_configs:
        model = mc["name"]
        p = _paths_for_model(batch_input_dir, outputs_root, model)
        if not p["requests_jsonl"].exists():
            raise FileNotFoundError(f"Missing JSONL for model {model}: {p['requests_jsonl']}")
        per_model[model] = p

    print("\nValidating input JSONL files...")
    nreq_by_model: Dict[str, int] = {}
    for model, p in per_model.items():
        res = validate_batch_jsonl(
            jsonl_path=p["requests_jsonl"],
            limits=limits,
            expected_model=model if limits.require_single_model_per_file else None,
            expected_url=None,
            expected_method="POST",
        )
        nreq_by_model[model] = int(res.n_requests)
        print(f"  ✓ {model}: {res.n_requests} requests, {res.file_size_mb:.1f} MB")

    # ------------------------------------------------------------
    # Build pending ONCE (or compute pending count if dry-run)
    # ------------------------------------------------------------
    print("\nPreparing pending requests (for accurate resume + cost estimate)...")
    n_pending_by_model: Dict[str, int] = {}
    pending_hash_by_model: Dict[str, str] = {}

    for model, p in per_model.items():
        p["out_dir"].mkdir(parents=True, exist_ok=True)
        done_ids = _load_done_custom_ids(p["out_jsonl"])

        if args.dry_run:
            n_pending = _count_pending_without_writing(
                src_requests_jsonl=p["requests_jsonl"],
                done_ids=done_ids,
                limit_n=args.limit_n,
            )
            n_pending_by_model[model] = n_pending
            print(f"  - {model}: pending={n_pending} (dry-run; no files written)")
            continue

        n_pending = _build_pending_jsonl(
            src_requests_jsonl=p["requests_jsonl"],
            dst_pending_jsonl=p["pending_jsonl"],
            done_ids=done_ids,
            forced_url=forced_url,
            served_model=model,
            limit_n=args.limit_n,
        )
        n_pending_by_model[model] = n_pending

        if n_pending == 0:
            print(f"  - {model}: pending=0 (skipping)")
        else:
            ph = _sha256_file(p["pending_jsonl"])
            pending_hash_by_model[model] = ph
            print(f"  - {model}: pending={n_pending} pending_sha256={ph[:12]}...")

    if args.dry_run:
        total_pending = sum(n_pending_by_model.values())
        print(f"\nTOTAL PENDING REQUESTS (dry-run): {total_pending}")
        print("Dry-run: exiting without cost estimation or vLLM execution.")
        return

    # ------------------------------------------------------------
    # Cost estimate: based on pending files (accurate for resume)
    # ------------------------------------------------------------
    print("\nEstimating cost (based on PENDING, not full file)...")
    total_cost = 0.0
    total_pending = 0
    for model, p in per_model.items():
        n_pending = n_pending_by_model.get(model, 0)
        total_pending += n_pending
        if n_pending == 0:
            continue

        sample_n = min(sample_n_cfg, n_pending)
        try:
            est = estimate_cost_for_jsonl(
                jsonl_path=p["pending_jsonl"],
                model=model,
                pricing_cfg=bundle.pricing,
                pricing_tier=pricing_tier,
                assumed_output_tokens_per_request=assumed_out,
                sample_n_requests=sample_n,
            )
            total_cost += est.est_cost_usd
            print(
                f"  - {model}: {est.n_requests} pending | est_in={est.est_input_tokens:,} | "
                f"est_out={est.est_output_tokens:,} | est_cost=${est.est_cost_usd:,.4f}"
            )
        except Exception:
            print(f"  - {model}: {n_pending} pending | est_cost=$0.0000 (no pricing)")

    print(f"\nTOTAL PENDING REQUESTS: {total_pending}")
    print(f"ESTIMATED TOTAL COST: ${total_cost:,.4f} (tier={pricing_tier})")
    print(f"vLLM batch url forced to: {forced_url}")
    if extra_args:
        print(f"vLLM run-batch extra args: {extra_args}")

    if not args.yes:
        if input("Proceed to run vLLM batch jobs? Type 'y' to continue: ").strip().lower() != "y":
            print("Aborted.")
            return

    run: Optional[wandb.sdk.wandb_run.Run] = None
    if not args.no_wandb:
        run = init_wandb_run(
            bundle,
            extra_config={
                "runner": "vllm_run_batch",
                "forced_url": forced_url,
                "estimated_total_cost_usd": total_cost,
                "pricing_tier": pricing_tier,
                "total_pending_requests": total_pending,
                "run_batch_extra_args": extra_args,
            },
        )

    try:
        for model, p in per_model.items():
            n_pending = n_pending_by_model.get(model, 0)
            if n_pending == 0:
                if run:
                    run.log({f"{model}/status": "skipped", f"{model}/pending": 0})
                continue

            # pending_jsonl already built once; do NOT rebuild here
            t0 = time.time()

            # done ids snapshot for duplicate prevention
            done_ids_now = _load_done_custom_ids(p["out_jsonl"])

            # BIG BUG FIX: ensure raw results are not stale from a previous attempt
            _clear_file(p["raw_results_jsonl"])

            appended_ids: Set[str] = set()

            try:
                _run_vllm_batch_runner(
                    pending_jsonl=p["pending_jsonl"],
                    raw_results_jsonl=p["raw_results_jsonl"],
                    model_name=model,
                    extra_args=extra_args,
                    log_path=p["run_batch_log"],
                )
            except subprocess.CalledProcessError:
                msg = f"vLLM run_batch failed for model={model}. See {p['run_batch_log']}"
                print(f"[{model}] ERROR: run_batch failed. Attempting to salvage partial raw results if present.")

                # salvage partial raw results (only new ids; skip duplicates)
                if p["raw_results_jsonl"].exists() and p["raw_results_jsonl"].stat().st_size > 0:
                    try:
                        appended_ids = _append_normalized_results(
                            raw_results_jsonl=p["raw_results_jsonl"],
                            out_jsonl=p["out_jsonl"],
                            skip_ids=done_ids_now,
                        )
                        print(f"[{model}] salvaged {len(appended_ids)} NEW records from partial raw results.")
                    except Exception as e:
                        print(f"[{model}] WARNING: failed to salvage partial raw results: {e}")

                # Do NOT write runner failures for ids already in output (done_ids_now),
                # nor for ids we just salvaged (appended_ids).
                already_handled = done_ids_now | appended_ids
                wrote = _write_batch_failure_errors_for_remaining(
                    pending_jsonl=p["pending_jsonl"],
                    err_history_jsonl=p["err_history_jsonl"],
                    message=msg,
                    already_handled_ids=already_handled,
                )
                print(f"[{model}] wrote {wrote} runner-failure entries to errors_history.jsonl.")

                # Keep errors.jsonl consistent with output.jsonl
                try:
                    n_err = _rewrite_errors_from_output(out_jsonl=p["out_jsonl"], err_jsonl=p["err_jsonl"])
                    print(f"[{model}] rewritten errors.jsonl from output.jsonl (n_errors={n_err}).")
                except Exception as e:
                    print(f"[{model}] WARNING: failed to rewrite errors.jsonl: {e}")

                if run:
                    run.log(
                        {
                            f"{model}/status": "run_batch_failed",
                            f"{model}/pending": n_pending,
                            f"{model}/salvaged_new": len(appended_ids),
                            f"{model}/runner_fail_written": wrote,
                        }
                    )

                # cleanup intermediates (optional)
                if not keep_intermediates:
                    for fp in (p["pending_jsonl"], p["raw_results_jsonl"]):
                        _clear_file(fp)

                continue

            # Success path: normalize output; prevent duplicates against current output
            done_ids_now = _load_done_custom_ids(p["out_jsonl"])  # refresh before append
            appended_ids = _append_normalized_results(
                raw_results_jsonl=p["raw_results_jsonl"],
                out_jsonl=p["out_jsonl"],
                skip_ids=done_ids_now,
            )
            elapsed = time.time() - t0

            # Rewrite errors.jsonl to avoid stale/duplicates
            n_err = _rewrite_errors_from_output(out_jsonl=p["out_jsonl"], err_jsonl=p["err_jsonl"])

            print(f"[{model}] done: appended_new={len(appended_ids)} elapsed={elapsed:.1f}s errors_now={n_err}")

            # Post-processing
            try:
                write_compact_results_csv(
                    output_jsonl_path=p["out_jsonl"],
                    requests_csv_path=p["requests_csv"] if p["requests_csv"].exists() else None,
                    out_csv_path=p["compact_csv"],
                    model=model,
                    include_input_features=include_input_features,
                    include_output_fields=include_output_fields,
                )
            except Exception as e:
                print(f"[{model}] WARNING: failed to write compact csv: {e}")

            try:
                if p["requests_csv"].exists() and p["out_jsonl"].exists() and p["out_jsonl"].stat().st_size > 0:
                    write_local_joined_results_jsonl(
                        output_jsonl_path=p["out_jsonl"],
                        requests_csv_path=p["requests_csv"],
                        out_jsonl_path=p["joined_jsonl"],
                        model_label=model,
                        include_input_features=include_input_features,
                    )
            except Exception as e:
                print(f"[{model}] WARNING: failed to write joined jsonl: {e}")

            if run:
                run.log(
                    {
                        f"{model}/status": "done",
                        f"{model}/pending": n_pending,
                        f"{model}/appended_new": len(appended_ids),
                        f"{model}/elapsed_s": int(elapsed),
                        f"{model}/req_per_s": (len(appended_ids) / elapsed) if elapsed > 0 else 0.0,
                        f"{model}/errors_now": n_err,
                    }
                )

            if not keep_intermediates:
                for fp in (p["pending_jsonl"], p["raw_results_jsonl"]):
                    _clear_file(fp)

        print("\nAll vLLM batch jobs finished.")

    finally:
        if run:
            try:
                artifacts_cfg = (bundle.logging.get("artifacts", {}) or {})
                results_spec = (artifacts_cfg.get("results_artifact", {}) or {})
                name_template = str(results_spec.get("name_template", "results-{model_dir}"))
                type_ = str(results_spec.get("type", "results"))

                for model, p in per_model.items():
                    model_dir = safe_model_dirname(model)
                    base_name = name_template.format(model_dir=model_dir)
                    art_name = f"vllm-{model_dir}-{base_name}"
                    art_type = f"vllm-{type_}"

                    art = wandb.Artifact(name=art_name, type=art_type)
                    added = 0

                    # Upload finals always
                    files = [p["out_jsonl"], p["err_jsonl"], p["compact_csv"], p["joined_jsonl"]]

                    # Optionally include intermediates + history + log
                    if upload_intermediates:
                        files += [p["pending_jsonl"], p["raw_results_jsonl"], p["run_batch_log"], p["err_history_jsonl"]]

                    for fp in files:
                        if fp.exists() and fp.is_file() and fp.stat().st_size > 0:
                            art.add_file(str(fp.resolve()))
                            added += 1

                    if added > 0:
                        run.log_artifact(art)

            except Exception as e:
                print(f"Warning: W&B results artifact upload failed: {e}")

            try:
                run.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()

