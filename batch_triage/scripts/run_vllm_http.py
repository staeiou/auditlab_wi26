#!/usr/bin/env python3
"""
scripts/run_vllm_job.py

Replay locally-generated OpenAI-batch-style JSONL requests against a live vLLM
OpenAI-compatible endpoint (e.g. http://localhost:8000/v1/chat/completions).

Design goals:
- Async concurrency (httpx + worker pool)
- Resume: skip custom_ids already in outputs/vllm/<model>/output.jsonl
- Write outputs locally (output.jsonl, errors.jsonl, results_compact.csv, results_joined.jsonl)
- Log to W&B (run metrics + upload artifacts at the end)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, List

import httpx
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
    Resume helper: read all valid JSON lines; ignore malformed ones (often just a truncated last line).
    If a line is malformed, we may re-run that request on resume (acceptable behavior).
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
                except Exception:
                    # likely a partial/truncated line if we crashed mid-write; skip it
                    continue
                cid = rec.get("custom_id")
                if isinstance(cid, str) and cid:
                    done.add(cid)
    except Exception:
        return done
    return done


def _normalize_body_for_vllm(body: Dict[str, Any], served_model: str) -> Dict[str, Any]:
    """
    vLLM speaks OpenAI-compatible /chat/completions.
    Normalize:
      - force model to served_model
      - map max_completion_tokens -> max_tokens if needed
    """
    b = dict(body)
    b["model"] = served_model
    if "max_tokens" not in b and "max_completion_tokens" in b:
        try:
            b["max_tokens"] = int(b["max_completion_tokens"])
        except Exception:
            b["max_tokens"] = b["max_completion_tokens"]
    return b


def _paths_for_model(batch_input_dir: Path, outputs_root: Path, model: str) -> Dict[str, Path]:
    model_dir = safe_model_dirname(model)
    model_batch_dir = batch_input_dir / model_dir
    requests_jsonl = model_batch_dir / f"{model_dir}_requests.jsonl"
    requests_csv = model_batch_dir / f"{model_dir}_requests.csv"

    out_dir = outputs_root / model_dir
    return {
        "model_dir": Path(model_dir),
        "batch_dir": model_batch_dir,
        "requests_jsonl": requests_jsonl,
        "requests_csv": requests_csv,
        "out_dir": out_dir,
        "out_jsonl": out_dir / "output.jsonl",
        "err_jsonl": out_dir / "errors.jsonl",
        "compact_csv": out_dir / "results_compact.csv",
        "joined_jsonl": out_dir / "results_joined.jsonl",
    }


def write_local_joined_results_jsonl(
    *,
    output_jsonl_path: Path,
    requests_csv_path: Path,
    out_jsonl_path: Path,
    model_label: str,
    include_input_features: List[str],
) -> None:
    """
    Writes JSONL that mirrors the essential fields you'd log:
      - custom_id
      - model_label
      - inputs (subset from requests csv)
      - response_text
      - raw response body
      - status_code
      - error
    """
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


async def _get_served_model_id(client: httpx.AsyncClient, base_url: str) -> Optional[str]:
    url = base_url.rstrip("/") + "/models"
    try:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
            first = data["data"][0]
            if isinstance(first, dict) and isinstance(first.get("id"), str):
                return first["id"]
    except Exception:
        return None
    return None


def _heuristic_token_estimate_from_jsonl(jsonl_path: Path, sample_n: int, assumed_out: int) -> Tuple[int, int, int]:
    n = 0
    sample_chars = 0
    sample_count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            if sample_count < sample_n:
                try:
                    rec = json.loads(line)
                    body = (rec.get("body") or {})
                    msgs = body.get("messages", [])
                    if isinstance(msgs, list):
                        for m in msgs:
                            if isinstance(m, dict) and isinstance(m.get("content"), str):
                                sample_chars += len(m["content"])
                    sample_count += 1
                except Exception:
                    pass
    if n == 0 or sample_count == 0:
        return n, 0, 0
    avg_chars = sample_chars / sample_count
    est_tokens_per_req = int(avg_chars / 4.0)
    est_in = est_tokens_per_req * n
    est_out = assumed_out * n
    return n, est_in, est_out


async def main_async() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--vllm-config", default="config/vllm.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--served-model", default=None, help="Override served model id; otherwise GET /models")
    ap.add_argument("--limit-n", type=int, default=None, help="Only run first N requests per model (testing)")
    args = ap.parse_args()

    bundle = load_config_bundle(project_root=PROJECT_ROOT, config_dir=PROJECT_ROOT / args.config_dir)

    vcfg_path = PROJECT_ROOT / args.vllm_config
    vcfg = yaml.safe_load(vcfg_path.read_text(encoding="utf-8")) or {}
    vllm_cfg = (vcfg.get("vllm", {}) or {})

    base_url: str = str(vllm_cfg.get("base_url", "http://localhost:8000/v1")).rstrip("/")
    endpoint: str = str(vllm_cfg.get("endpoint", "/chat/completions"))
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    full_url = base_url + endpoint

    max_conc = int(vllm_cfg.get("max_concurrent_requests", 16))
    timeout_s = float(vllm_cfg.get("request_timeout_seconds", 120))
    retry_cfg = vllm_cfg.get("retry", {}) or {}
    max_attempts = int(retry_cfg.get("max_attempts", 3))
    backoff_s = float(retry_cfg.get("backoff_seconds", 2))

    outputs_root = PROJECT_ROOT / "outputs" / "vllm"
    outputs_root.mkdir(parents=True, exist_ok=True)

    # reuse openai_batch limits
    lim = bundle.limits.get("openai_batch", {}) or {}
    limits = BatchInputLimits(
        max_requests_per_file=int(lim.get("max_requests_per_file", 50_000)),
        max_file_size_mb=int(lim.get("max_file_size_mb", 200)),
        require_single_model_per_file=bool(lim.get("require_single_model_per_file", True)),
    )
    expected_batch_url = None 

    compact_cfg = bundle.logging.get("results_compact", {}) or {}
    include_input_features = compact_cfg.get("include_input_features", []) or []
    include_output_fields = compact_cfg.get("include_output_fields", ["raw_response", "parsed_json", "ok", "error"]) or []

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
            expected_url=expected_batch_url,
            expected_method="POST",
        )
        nreq_by_model[model] = int(res.n_requests)
        print(f"  ✓ {model}: {res.n_requests} requests, {res.file_size_mb:.1f} MB")

    print("\nEstimating cost (OpenAI-equivalent if pricing exists; otherwise token scale only)...")
    total_cost = 0.0
    total_reqs = 0
    for model, p in per_model.items():
        nreq = nreq_by_model.get(model, 0)
        total_reqs += nreq
        sample_n = min(sample_n_cfg, nreq) if nreq > 0 else 0

        try:
            est = estimate_cost_for_jsonl(
                jsonl_path=p["requests_jsonl"],
                model=model,
                pricing_cfg=bundle.pricing,
                pricing_tier=pricing_tier,
                assumed_output_tokens_per_request=assumed_out,
                sample_n_requests=sample_n,
            )
            total_cost += est.est_cost_usd
            print(
                f"  - {model}: {est.n_requests} req | est_in={est.est_input_tokens:,} | "
                f"est_out={est.est_output_tokens:,} | est_cost=${est.est_cost_usd:,.4f}"
            )
        except Exception:
            n_all, est_in, est_out = _heuristic_token_estimate_from_jsonl(p["requests_jsonl"], max(1, sample_n), assumed_out)
            print(f"  - {model}: {n_all} req | est_in≈{est_in:,} | est_out≈{est_out:,} | est_cost=$0.0000 (no pricing)")

    print(f"\nTOTAL REQUESTS: {total_reqs}")
    print(f"ESTIMATED TOTAL COST: ${total_cost:,.4f} (tier={pricing_tier})")

    if args.dry_run:
        print("Dry-run: exiting without sending requests.")
        return

    if input("Proceed to run vLLM jobs? Type 'y' to continue: ").strip().lower() != "y":
        print("Aborted.")
        return

    run: Optional[wandb.sdk.wandb_run.Run] = None
    if not args.no_wandb:
        run = init_wandb_run(
            bundle,
            extra_config={
                "runner": "vllm",
                "vllm_base_url": base_url,
                "vllm_endpoint": endpoint,
                "estimated_total_cost_usd": total_cost,
                "pricing_tier": pricing_tier,
                "total_requests": total_reqs,
            },
        )

    limits_http = httpx.Limits(max_connections=max_conc * 2, max_keepalive_connections=max_conc)
    timeout = httpx.Timeout(timeout_s)

    async with httpx.AsyncClient(limits=limits_http, timeout=timeout) as client:
        served_model = args.served_model or (await _get_served_model_id(client, base_url)) or "UNKNOWN_VLLM_MODEL"
        print(f"\nvLLM endpoint: {full_url}")
        print(f"Served model id (used for body.model): {served_model}")

        async def send_one(record: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
            custom_id = record.get("custom_id")
            body = record.get("body") or {}
            if not isinstance(body, dict):
                body = {}

            payload = _normalize_body_for_vllm(body, served_model=served_model)

            last_exc: Optional[str] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    r = await client.post(full_url, json=payload)
                    status_code = r.status_code
                    try:
                        resp_body = r.json()
                    except Exception:
                        resp_body = {"raw_text": r.text}

                    out_line = {
                        "custom_id": custom_id,
                        "response": {"status_code": status_code, "body": resp_body},
                        "error": None,
                    }

                    # DESIGN CHOICE:
                    # - output.jsonl stores everything (including HTTP errors)
                    # - errors.jsonl stores only failures
                    if status_code >= 400:
                        err_line = {
                            "custom_id": custom_id,
                            "response": {"status_code": status_code, "body": resp_body},
                            "error": {"message": f"HTTP {status_code}", "type": "http_error"},
                        }
                        return out_line, err_line

                    return out_line, None

                except Exception as e:
                    last_exc = str(e)
                    if attempt < max_attempts:
                        await asyncio.sleep(backoff_s * (2 ** (attempt - 1)) + random.random() * 0.25)
                        continue

                    out_line = {
                        "custom_id": custom_id,
                        "response": {"status_code": 0, "body": {"error": {"message": last_exc}}},
                        "error": {"message": last_exc, "type": "exception"},
                    }
                    return out_line, out_line

            # unreachable
            out_line = {
                "custom_id": custom_id,
                "response": {"status_code": 0, "body": {"error": {"message": last_exc or "unknown"}}},
                "error": {"message": last_exc or "unknown", "type": "exception"},
            }
            return out_line, out_line

        async def run_one_model(model: str, paths: Dict[str, Path]) -> None:
            paths["out_dir"].mkdir(parents=True, exist_ok=True)

            done_ids = _load_done_custom_ids(paths["out_jsonl"])
            total = nreq_by_model.get(model, 0)
            print(f"\n[{model}] starting (resume has {len(done_ids)} done of {total})")

            # BOUNDED queues -> backpressure prevents memory blowup
            qmax = max(1000, max_conc * 100)
            out_q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue(maxsize=qmax)
            err_q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue(maxsize=qmax)

            writer_failed = asyncio.Event()

            async def writer_task(q: asyncio.Queue[Optional[Dict[str, Any]]], out_path: Path) -> None:
                """
                Never crash the whole program:
                - if write fails, set writer_failed and keep draining/discarding until sentinel.
                This prevents deadlocks with bounded queues.
                """
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with out_path.open("a", encoding="utf-8") as f:
                        while True:
                            item = await q.get()
                            if item is None:
                                break
                            if writer_failed.is_set():
                                # discard to keep draining
                                continue
                            try:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                                if q.qsize() == 0:
                                    f.flush()
                            except Exception:
                                writer_failed.set()
                                # keep draining
                                continue
                except Exception:
                    writer_failed.set()
                    # drain to avoid blocking puts
                    while True:
                        item = await q.get()
                        if item is None:
                            break

            w_out = asyncio.create_task(writer_task(out_q, paths["out_jsonl"]))
            w_err = asyncio.create_task(writer_task(err_q, paths["err_jsonl"]))

            work_q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue(maxsize=max_conc * 4)

            finished = 0
            failed = 0
            t0 = time.time()

            async def producer() -> int:
                enqueued = 0
                try:
                    with paths["requests_jsonl"].open("r", encoding="utf-8") as f:
                        for line_no, line in enumerate(f, start=1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except json.JSONDecodeError as e:
                                print(f"[{model}] WARNING invalid JSON on line {line_no}: {e}")
                                continue
                            except Exception as e:
                                print(f"[{model}] WARNING failed to parse line {line_no}: {e}")
                                continue

                            cid = rec.get("custom_id")
                            if isinstance(cid, str) and cid in done_ids:
                                continue
                            if args.limit_n is not None and enqueued >= args.limit_n:
                                break

                            await work_q.put(rec)
                            enqueued += 1
                finally:
                    # Always stop workers, even if producer fails early.
                    for _ in range(max_conc):
                        await work_q.put(None)
                return enqueued
            last_progress_time = 0.0
            last_wandb_time = 0.0
            WANDB_LOG_EVERY_S = 10.0     # adjust if you want (optional)
            CONSOLE_EVERY_S = 5.0        # adjust if you want (optional)

            async def worker() -> None:
                nonlocal finished, failed, last_progress_time, last_wandb_time
                while True:
                    rec = await work_q.get()
                    try:
                        if rec is None:
                            return

                        # If writers broke, stop doing work but keep draining queue so shutdown doesn't hang
                        if writer_failed.is_set():
                            continue

                        out_line, err_line = await send_one(rec)

                        # output gets one line per request (success or failure)
                        await out_q.put(out_line)

                        if err_line is not None:
                            failed += 1
                            await err_q.put(err_line)

                        finished += 1

                        # Console progress (throttled)
                        now = time.time()
                        if (finished % 100 == 0) or ((now - last_progress_time) >= CONSOLE_EVERY_S):
                            elapsed = now - t0
                            rate = finished / elapsed if elapsed > 0 else 0.0
                            print(
                                f"[{model}] Progress: {finished} done ({failed} failed) | "
                                f"{rate:.1f} req/s | elapsed {elapsed:.0f}s"
                            )
                            last_progress_time = now

                        # W&B progress (time-throttled)
                        if run and ((now - last_wandb_time) >= WANDB_LOG_EVERY_S):
                            elapsed = now - t0
                            run.log(
                                {
                                    f"{model}/finished": finished,
                                    f"{model}/failed": failed,
                                    f"{model}/elapsed_s": int(elapsed),
                                    f"{model}/req_per_s": (finished / elapsed) if elapsed > 0 else 0.0,
                                }
                            )
                            last_wandb_time = now

                    finally:
                        work_q.task_done()


            enq_task = asyncio.create_task(producer())
            workers = [asyncio.create_task(worker()) for _ in range(max_conc)]

            enqueued = await enq_task
            await asyncio.gather(*workers)

            # Close writers safely: don't hang if writers already died
            if not w_out.done():
                await out_q.put(None)
            if not w_err.done():
                await err_q.put(None)

            await asyncio.gather(w_out, w_err)

            print(f"[{model}] done: enqueued={enqueued} finished={finished} failed={failed}")

            if writer_failed.is_set():
                print(f"[{model}] WARNING writer failed; outputs may be incomplete.")
                return

            # compact CSV
            try:
                write_compact_results_csv(
                    output_jsonl_path=paths["out_jsonl"],
                    requests_csv_path=paths["requests_csv"] if paths["requests_csv"].exists() else None,
                    out_csv_path=paths["compact_csv"],
                    model=model,
                    include_input_features=include_input_features,
                    include_output_fields=include_output_fields,
                )
            except Exception as e:
                print(f"[{model}] WARNING: failed to write compact csv: {e}")

            # joined JSONL
            try:
                if paths["requests_csv"].exists() and paths["out_jsonl"].exists() and paths["out_jsonl"].stat().st_size > 0:
                    write_local_joined_results_jsonl(
                        output_jsonl_path=paths["out_jsonl"],
                        requests_csv_path=paths["requests_csv"],
                        out_jsonl_path=paths["joined_jsonl"],
                        model_label=model,
                        include_input_features=include_input_features,
                    )
            except Exception as e:
                print(f"[{model}] WARNING: failed to write joined jsonl: {e}")

            if run:
                run.log(
                    {
                        f"{model}/total": total,
                        f"{model}/finished": finished,
                        f"{model}/failed": failed,
                        f"{model}/status": "done",
                    }
                )

        # Run models sequentially (reliable). Within each model, requests are concurrent.
        try:
            for model, paths in per_model.items():
                await run_one_model(model, paths)
        except KeyboardInterrupt:
            print("\nInterrupted (Ctrl+C). Partial outputs remain; rerun to resume.")

        print("\nAll vLLM jobs finished (or interrupted).")

        # Upload results artifacts at the end (best-effort)
        if run:
            try:
                artifacts_cfg = (bundle.logging.get("artifacts", {}) or {})
                results_spec = (artifacts_cfg.get("results_artifact", {}) or {})
                name_template = str(results_spec.get("name_template", "results-{model_dir}"))
                type_ = str(results_spec.get("type", "results"))

                for model, paths in per_model.items():
                    model_dir = safe_model_dirname(model)

                    # Ensure uniqueness even if name_template doesn't include model_dir
                    base_name = name_template.format(model_dir=model_dir)
                    art_name = f"vllm-{model_dir}-{base_name}"
                    art_type = f"vllm-{type_}"

                    art = wandb.Artifact(name=art_name, type=art_type)
                    added = 0
                    for fp in (paths["out_jsonl"], paths["err_jsonl"], paths["compact_csv"], paths["joined_jsonl"]):
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


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

