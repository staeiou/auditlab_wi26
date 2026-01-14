#!/usr/bin/env python3
"""
scripts/run_openai_batch.py

Reliability:
- StateStore writes serialized via store_lock
- Per-model file operations serialized via file_locks[model]
- Blocking calls run in a bounded ThreadPoolExecutor
- Timeouts via asyncio.wait_for are used ONLY for non-file operations (upload/create/get)
- For file-writing ops (download/write CSV/joined), we rely on the OpenAI SDK/HTTP client timeouts
  (we do NOT cancel while holding file locks).
- W&B uploads serialized via wandb_lock
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import functools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config_bundle, safe_model_dirname  # type: ignore
from src.cost_estimator import estimate_cost_for_jsonl  # type: ignore
from src.openai_batch.client import OpenAIBatchClient  # type: ignore
from src.openai_batch.validator import BatchInputLimits, validate_batch_jsonl  # type: ignore
from src.openai_batch.parser import write_compact_results_csv  # type: ignore
from src.state.state_store import StateStore, DONE_STATUSES  # type: ignore
from src.wandb_logging.run_manager import init_wandb_run  # type: ignore
from src.wandb_logging.artifacts import ArtifactUploader  # type: ignore


# -----------------------------------------------------------------------------
# Local "wandb-like" joined results writer (inputs + response)
# -----------------------------------------------------------------------------

def _extract_text_from_chat_completions_body(body: Any) -> Optional[str]:
    """Handles /v1/chat/completions batch outputs."""
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
    """Best-effort text extraction across schemas."""
    if body is None:
        return None
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        t = _extract_text_from_chat_completions_body(body)
        if t is not None:
            return t

        # responses-style fallbacks (if endpoint changes later)
        for k in ("output_text", "text"):
            v = body.get(k)
            if isinstance(v, str):
                return v

        # sometimes body["content"] is list of blocks
        v = body.get("content")
        if isinstance(v, list):
            parts = []
            for blk in v:
                if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                    parts.append(blk["text"])
            if parts:
                return "\n".join(parts)

    return None


def write_local_joined_results_jsonl(
    *,
    output_jsonl_path: Path,
    requests_csv_path: Path,
    out_jsonl_path: Path,
    model: str,
    include_input_features: list[str],
) -> None:
    """
    Writes outputs/openai/<model>/results_joined.jsonl with:
      - custom_id, model
      - inputs (subset from requests CSV)
      - status_code, response_text, response_body, error
    Joined by custom_id.
    """
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Build custom_id -> inputs mapping from requests CSV
    id_to_inputs: Dict[str, Dict[str, Any]] = {}
    if requests_csv_path.exists():
        with requests_csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("custom_id")
                if not cid:
                    continue
                inputs: Dict[str, Any] = {}
                for k in include_input_features:
                    inputs[k] = row.get(k)  # missing -> None
                id_to_inputs[cid] = inputs

    # Stream output.jsonl and write joined lines
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
                "model": model,
                "inputs": id_to_inputs.get(cid, {}) if cid else {},
                "status_code": status_code,
                "response_text": _extract_text_from_any_body(body),
                "response_body": body,
                "error": rec.get("error"),
            }
            fout.write(json.dumps(joined, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# Joined marker helpers (signature-based, robust across restarts)
# -----------------------------------------------------------------------------

def _joined_marker_path(joined_jsonl_path: Path) -> Path:
    return joined_jsonl_path.with_suffix(joined_jsonl_path.suffix + ".ok")


def _marker_payload(output_file_id: Optional[str], output_bytes: int) -> dict:
    return {"output_file_id": output_file_id, "output_bytes": int(output_bytes)}


def _write_marker_tmp(marker_tmp: Path, payload: dict) -> None:
    marker_tmp.parent.mkdir(parents=True, exist_ok=True)
    marker_tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_marker(marker: Path) -> Optional[dict]:
    try:
        if not marker.exists() or marker.stat().st_size == 0:
            return None
        raw = marker.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _count_and_validate_jsonl(path: Path) -> Tuple[int, bool]:
    """
    Returns (line_count, all_lines_parse_as_json).
    Treats blank lines as ignorable.
    """
    n = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                json.loads(line)
                n += 1
        return n, True
    except Exception:
        return n, False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _batch_counts(batch_obj: Any) -> Tuple[int, int, int]:
    counts = _get_attr_or_key(batch_obj, "request_counts", {}) or {}
    if not isinstance(counts, dict):
        total = int(getattr(counts, "total", 0))
        completed = int(getattr(counts, "completed", 0))
        failed = int(getattr(counts, "failed", 0))
        return total, completed, failed
    return int(counts.get("total", 0)), int(counts.get("completed", 0)), int(counts.get("failed", 0))


def _is_done(status: Optional[str]) -> bool:
    return (status or "").lower() in DONE_STATUSES


def _paths_for_model(bundle, model: str) -> Dict[str, Path]:
    model_dirname = safe_model_dirname(model)
    model_batch_dir = bundle.batch_input_dir / model_dirname

    requests_jsonl = model_batch_dir / f"{model_dirname}_requests.jsonl"
    requests_csv = model_batch_dir / f"{model_dirname}_requests.csv"

    out_dir = bundle.output_dir / model_dirname
    out_jsonl = out_dir / "output.jsonl"
    err_jsonl = out_dir / "errors.jsonl"
    compact_csv = out_dir / "results_compact.csv"
    joined_jsonl = out_dir / "results_joined.jsonl"

    return {
        "model_dirname": Path(model_dirname),
        "batch_dir": model_batch_dir,
        "requests_jsonl": requests_jsonl,
        "requests_csv": requests_csv,
        "out_dir": out_dir,
        "out_jsonl": out_jsonl,
        "err_jsonl": err_jsonl,
        "compact_csv": compact_csv,
        "joined_jsonl": joined_jsonl,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

async def main_async() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--poll-seconds", type=int, default=None)
    ap.add_argument("--timeout-seconds", type=int, default=None, help="Override non-file control-call timeout")
    args = ap.parse_args()

    bundle = load_config_bundle(project_root=PROJECT_ROOT, config_dir=PROJECT_ROOT / args.config_dir)

    openai_cfg = bundle.config["openai_batch"]
    endpoint = openai_cfg["endpoint"]
    completion_window = openai_cfg["completion_window"]
    poll_seconds = int(args.poll_seconds or openai_cfg["poll_interval_seconds"])

    lim = bundle.limits.get("openai_batch", {}) or {}
    limits = BatchInputLimits(
        max_requests_per_file=int(lim.get("max_requests_per_file", 50_000)),
        max_file_size_mb=int(lim.get("max_file_size_mb", 200)),
        require_single_model_per_file=bool(lim.get("require_single_model_per_file", True)),
    )

    configured_max_calls = int(lim.get("max_concurrent_control_calls", 4))
    control_timeout_s = float(args.timeout_seconds or lim.get("control_call_timeout_seconds", 120))

    compact_cfg = bundle.logging.get("results_compact", {}) or {}
    include_input_features = compact_cfg.get("include_input_features", []) or []
    include_output_fields = compact_cfg.get("include_output_fields", ["raw_response", "parsed_json", "ok", "error"]) or []

    ce = bundle.config.get("cost_estimation", {}) or {}
    sample_n_cfg = int(ce.get("sample_n_requests", 200))
    assumed_out = int(ce.get("assumed_output_tokens_per_request", 80))
    pricing_tier = bundle.pricing_tier

    per_model: Dict[str, Dict[str, Path]] = {}
    for mc in bundle.model_configs:
        model = mc["name"]
        paths = _paths_for_model(bundle, model)
        if not paths["requests_jsonl"].exists():
            raise FileNotFoundError(f"Missing JSONL for model {model}: {paths['requests_jsonl']}")
        per_model[model] = paths

    # Define executor early (so finally can't crash)
    executor: Optional[ThreadPoolExecutor] = None
    max_workers = max(1, min(configured_max_calls, len(per_model)))
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="openai_ctrl")

    async def _run_blocking(fn, *a, timeout_s: Optional[float], **kw):
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(executor, functools.partial(fn, *a, **kw))  # type: ignore[arg-type]
        if timeout_s is None:
            return await fut
        return await asyncio.wait_for(fut, timeout=timeout_s)

    store = StateStore(bundle.state_file).load()
    store_lock = asyncio.Lock()
    file_locks: Dict[str, asyncio.Lock] = {m: asyncio.Lock() for m in per_model.keys()}
    wandb_lock = asyncio.Lock()

    # per-(model, op) cooldown tracking
    next_retry_at: Dict[Tuple[str, str], float] = {}

    def _should_retry(model: str, op: str) -> bool:
        return time.time() >= next_retry_at.get((model, op), 0.0)

    def _mark_retry_later(model: str, op: str, cooldown_s: int) -> None:
        next_retry_at[(model, op)] = time.time() + cooldown_s

    print("\nValidating batch input JSONL files...")
    validation_counts: Dict[str, int] = {}
    for model, p in per_model.items():
        res = validate_batch_jsonl(
            jsonl_path=p["requests_jsonl"],
            limits=limits,
            expected_model=model if limits.require_single_model_per_file else None,
            expected_url=endpoint,
            expected_method="POST",
        )
        validation_counts[model] = int(res.n_requests)
        print(f"  ✓ {model}: {res.n_requests} requests, {res.file_size_mb:.1f} MB")

    print("\nEstimating cost...")
    total_cost = 0.0
    for model, p in per_model.items():
        nreq = validation_counts.get(model, 0)
        sample_n = min(sample_n_cfg, nreq) if nreq > 0 else 0
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

    print(f"\nESTIMATED TOTAL: ${total_cost:,.4f} (tier={pricing_tier})")
    if args.dry_run:
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        print("Dry-run: exiting without submitting.")
        return

    if input("Proceed to submit OpenAI Batch jobs? Type 'y' to continue: ").strip().lower() != "y":
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        print("Aborted.")
        return

    run = None
    uploader: Optional[ArtifactUploader] = None
    if not args.no_wandb:
        run = init_wandb_run(
            bundle,
            extra_config={
                "estimated_total_cost_usd": total_cost,
                "pricing_tier": pricing_tier,
                "endpoint": endpoint,
                "completion_window": completion_window,
            },
        )
        uploader = ArtifactUploader(bundle=bundle, run=run, project_root=bundle.project_root)
        async with wandb_lock:
            uploader.upload_inputs_once()

    client = OpenAIBatchClient()

    async def _safe_control_call(model: str, op: str, fn, *a, **kw):
        """For non-file operations only: safe to cancel with wait_for."""
        try:
            return await _run_blocking(fn, *a, timeout_s=control_timeout_s, **kw)
        except asyncio.TimeoutError:
            msg = f"{op} timed out after {control_timeout_s}s"
            print(f"[{model}] ERROR {msg}")
            _mark_retry_later(model, op, 120)
            async with store_lock:
                st = store.get(model)
                if st:
                    st.last_error = msg
                    st.mark_updated()
                    store.save()
            return None
        except Exception as e:
            msg = f"{op} failed: {e}"
            print(f"[{model}] ERROR {msg}")
            _mark_retry_later(model, op, 300)
            async with store_lock:
                st = store.get(model)
                if st:
                    st.last_error = msg
                    st.mark_updated()
                    store.save()
            return None

    async def download_with_handling(
        model: str,
        file_id: str,
        target_path: Path,
        op_name: str,
        *,
        allow_empty: bool,
        retry_cooldown_s: int = 300,
    ) -> bool:
        if not _should_retry(model, op_name):
            return False

        # This acquires the per-model file lock (so CSV/join won't read partial writes).
        async with file_locks[model]:
            try:
                if target_path.exists():
                    try:
                        target_path.unlink()
                    except Exception:
                        pass

                # NO asyncio.wait_for here: don't cancel while holding file lock.
                await _run_blocking(client.download_file, file_id, target_path, timeout_s=None)

                if not target_path.exists():
                    raise RuntimeError(f"Downloaded file missing: {target_path}")
                if (not allow_empty) and target_path.stat().st_size == 0:
                    raise RuntimeError(f"Downloaded file is empty: {target_path}")

                return True

            except Exception as e:
                if target_path.exists():
                    try:
                        target_path.unlink()
                    except Exception:
                        pass

                print(f"[{model}] ERROR {op_name}: {e}")
                _mark_retry_later(model, op_name, retry_cooldown_s)

                async with store_lock:
                    st = store.get(model)
                    if st:
                        st.last_error = f"{op_name}: {e}"
                        st.mark_updated()
                        store.save()

                return False

    try:
        async def submit_one(model: str) -> None:
            async with store_lock:
                st = store.ensure(model, per_model[model]["requests_jsonl"])
                if st.batch_id:
                    return

            req_jsonl = per_model[model]["requests_jsonl"]

            uploaded = await _safe_control_call(model, "upload_file", client.upload_file_for_batch, req_jsonl)
            if uploaded is None:
                return

            batch = await _safe_control_call(
                model,
                "create_batch",
                client.create_batch,
                input_file_id=uploaded.file_id,
                endpoint=endpoint,
                completion_window=completion_window,
                metadata={"model": model, "input_jsonl": str(req_jsonl)},
            )
            if batch is None:
                return

            async with store_lock:
                store.mark_submitted(
                    model=model,
                    input_jsonl=req_jsonl,
                    input_file_id=uploaded.file_id,
                    batch_id=_get_attr_or_key(batch, "id"),
                    status=_get_attr_or_key(batch, "status"),
                )
                store.save()

            if run:
                run.log({f"{model}/batch_id": _get_attr_or_key(batch, "id"), f"{model}/status": _get_attr_or_key(batch, "status")})

        print("\nSubmitting jobs (one batch per model)...")
        await asyncio.gather(*(submit_one(m) for m in per_model.keys()))

        async def retrieve_one(model: str) -> Any:
            async with store_lock:
                st = store.get(model)
                batch_id = st.batch_id if st else None
            if not batch_id:
                return None
            return await _safe_control_call(model, "get_batch", client.get_batch, batch_id)

        print("\nPolling batches (Ctrl+C to stop; rerun to resume)...")
        while True:
            batches = await asyncio.gather(*(retrieve_one(m) for m in per_model.keys()))
            statuses: Dict[str, Optional[str]] = {}

            for model, batch_obj in zip(per_model.keys(), batches):
                if batch_obj is None:
                    statuses[model] = None
                    continue

                async with store_lock:
                    store.update_from_batch_object(model=model, batch_obj=batch_obj)
                    store.save()
                    st = store.get(model)

                status = _get_attr_or_key(batch_obj, "status")
                statuses[model] = status
                total, completed, failed = _batch_counts(batch_obj)

                print(f"[{model}] status={status} {completed}/{total} failed={failed}")

                if run:
                    run.log(
                        {
                            f"{model}/status": status,
                            f"{model}/total": total,
                            f"{model}/completed": completed,
                            f"{model}/failed": failed,
                            "poll_time_unix": int(time.time()),
                        }
                    )

                if not st:
                    continue

                if status == "completed":
                    paths = per_model[model]
                    paths["out_dir"].mkdir(parents=True, exist_ok=True)

                    # 1) Download outputs first
                    if st.output_file_id and st.output_jsonl_path is None:
                        ok = await download_with_handling(
                            model, st.output_file_id, paths["out_jsonl"], "download_output", allow_empty=False
                        )
                        if ok:
                            async with store_lock:
                                store.mark_local_outputs(model=model, output_jsonl_path=paths["out_jsonl"])
                                store.save()

                    if st.error_file_id and st.errors_jsonl_path is None:
                        ok = await download_with_handling(
                            model, st.error_file_id, paths["err_jsonl"], "download_errors", allow_empty=True
                        )
                        if ok:
                            async with store_lock:
                                store.mark_local_outputs(model=model, errors_jsonl_path=paths["err_jsonl"])
                                store.save()

                    # Refresh state after possible downloads
                    async with store_lock:
                        st2 = store.get(model)

                    # 2+3) Compact + Joined under ONE file lock (prevents partial reads)
                    async with file_locks[model]:
                        out_ok = paths["out_jsonl"].exists() and paths["out_jsonl"].stat().st_size > 0
                        out_bytes = paths["out_jsonl"].stat().st_size if out_ok else 0
                        out_file_id = getattr(st2, "output_file_id", None) if st2 else None
                        desired_marker = _marker_payload(out_file_id, out_bytes)
                        marker = _joined_marker_path(paths["joined_jsonl"])

                        # 2) Write compact CSV
                        if out_ok and st2 and st2.results_compact_csv_path is None and _should_retry(model, "write_compact"):
                            try:
                                await _run_blocking(
                                    write_compact_results_csv,
                                    output_jsonl_path=paths["out_jsonl"],
                                    requests_csv_path=paths["requests_csv"] if paths["requests_csv"].exists() else None,
                                    out_csv_path=paths["compact_csv"],
                                    model=model,
                                    include_input_features=include_input_features,
                                    include_output_fields=include_output_fields,
                                    timeout_s=None,
                                )
                                if not paths["compact_csv"].exists() or paths["compact_csv"].stat().st_size == 0:
                                    raise RuntimeError("Compact CSV missing/empty after write")

                                async with store_lock:
                                    store.mark_local_outputs(model=model, results_compact_csv_path=paths["compact_csv"])
                                    store.save()

                            except Exception as e:
                                print(f"[{model}] ERROR write_compact: {e}")
                                _mark_retry_later(model, "write_compact", 300)
                                async with store_lock:
                                    stx = store.get(model)
                                    if stx:
                                        stx.last_error = f"write_compact: {e}"
                                        stx.mark_updated()
                                        store.save()

                        # 3) Write joined JSONL + marker robustly
                        if out_ok and _should_retry(model, "write_joined"):
                            marker_obj = _read_marker(marker)

                            joined_done = (
                                marker_obj == desired_marker
                                and paths["joined_jsonl"].exists()
                                and paths["joined_jsonl"].stat().st_size > 0
                            )

                            # If marker missing/mismatch but joined exists, decide whether to adopt or rebuild.
                            if (not joined_done) and paths["joined_jsonl"].exists() and paths["joined_jsonl"].stat().st_size > 0:
                                out_n, out_valid = _count_and_validate_jsonl(paths["out_jsonl"])
                                join_n, join_valid = _count_and_validate_jsonl(paths["joined_jsonl"])
                                # For OUR writer: joined has one line per output line -> counts should match.
                                if out_valid and join_valid and out_n == join_n and out_n > 0:
                                    # Adopt existing joined file: write marker to match current output signature.
                                    tmp_marker = marker.with_suffix(marker.suffix + ".tmp")
                                    _write_marker_tmp(tmp_marker, desired_marker)
                                    os.replace(str(tmp_marker), str(marker))
                                    joined_done = True

                            if not joined_done:
                                try:
                                    tmp_join = paths["joined_jsonl"].with_suffix(paths["joined_jsonl"].suffix + ".tmp")
                                    tmp_marker = marker.with_suffix(marker.suffix + ".tmp")

                                    # clean stale tmp files
                                    for p in (tmp_join, tmp_marker):
                                        if p.exists():
                                            try:
                                                p.unlink()
                                            except Exception:
                                                pass

                                    # write tmp join
                                    await _run_blocking(
                                        write_local_joined_results_jsonl,
                                        output_jsonl_path=paths["out_jsonl"],
                                        requests_csv_path=paths["requests_csv"],
                                        out_jsonl_path=tmp_join,
                                        model=model,
                                        include_input_features=include_input_features,
                                        timeout_s=None,
                                    )
                                    if not tmp_join.exists() or tmp_join.stat().st_size == 0:
                                        raise RuntimeError("Joined JSONL tmp missing/empty after write")

                                    # write tmp marker BEFORE final replace (so we can "double replace")
                                    _write_marker_tmp(tmp_marker, desired_marker)

                                    # atomic-ish publish: joined then marker (both via replace)
                                    os.replace(str(tmp_join), str(paths["joined_jsonl"]))
                                    os.replace(str(tmp_marker), str(marker))

                                except Exception as e:
                                    print(f"[{model}] ERROR write_joined: {e}")
                                    _mark_retry_later(model, "write_joined", 300)
                                    async with store_lock:
                                        stx = store.get(model)
                                        if stx:
                                            stx.last_error = f"write_joined: {e}"
                                            stx.mark_updated()
                                            store.save()

                    # 4) Upload results artifact to W&B (if ready)
                    if uploader:
                        ready = (
                            paths["out_jsonl"].exists()
                            and paths["out_jsonl"].stat().st_size > 0
                            and paths["compact_csv"].exists()
                            and paths["compact_csv"].stat().st_size > 0
                        )
                        if ready:
                            async with wandb_lock:
                                uploader.upload_results_for_model_if_present(model)

                elif status in {"failed", "expired", "cancelled"}:
                    paths = per_model[model]
                    paths["out_dir"].mkdir(parents=True, exist_ok=True)

                    if st.error_file_id and st.errors_jsonl_path is None:
                        ok = await download_with_handling(
                            model, st.error_file_id, paths["err_jsonl"], "download_errors_terminal", allow_empty=True
                        )
                        if ok:
                            async with store_lock:
                                store.mark_local_outputs(model=model, errors_jsonl_path=paths["err_jsonl"])
                                store.save()

            # periodic state upload
            if uploader:
                async with wandb_lock:
                    uploader.upload_state_if_due()

            all_done = True
            for model in per_model.keys():
                s = statuses.get(model)
                if s is None or not _is_done(s):
                    all_done = False
                    break

            if all_done:
                print("\nAll batches reached a terminal state.")
                break

            await asyncio.sleep(poll_seconds)

    except KeyboardInterrupt:
        print("\nInterrupted. You can rerun to resume from state file.")

    finally:
        # 1) Always try to save state
        try:
            store.save()
        except Exception:
            pass

        # 2) Final W&B sync (best effort)
        if uploader:
            try:
                # "force" without changing artifacts.py: reset gating timestamps
                uploader.last_artifact_upload_unix = 0.0
                uploader.last_state_upload_unix = 0.0

                async with wandb_lock:
                    uploader.upload_state_if_due(force=True)
                    for model, paths in per_model.items():
                        if (
                            paths["out_jsonl"].exists()
                            and paths["out_jsonl"].stat().st_size > 0
                            and paths["compact_csv"].exists()
                            and paths["compact_csv"].stat().st_size > 0
                        ):
                            uploader.upload_results_for_model_if_present(model, force=True)
            except Exception as e:
                print(f"Warning: Final W&B upload failed: {e}")

        # 3) Finish W&B run
        if run:
            try:
                run.finish()
            except Exception:
                pass

        # 4) Shutdown thread pool
        if executor:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
