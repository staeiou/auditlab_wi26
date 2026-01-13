from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI


@dataclass(frozen=True)
class UploadedFile:
    file_id: str
    filename: str


class OpenAIBatchClient:
    """
    Thin wrapper around the OpenAI Python SDK for Batch-related operations.

    Notes on timeouts:
      - Official OpenAI SDKs have a default request timeout (docs mention 10 minutes),
        and allow overriding via `timeout` plus per-request overrides via `with_options(timeout=...)`. :contentReference[oaicite:1]{index=1}
      - For large output downloads, we use a longer timeout and write atomically to avoid partial files.

    If you pass a pre-configured OpenAI client, this class will not override its timeouts/retries.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        *,
        timeout_s: float = 600.0,          # 10 minutes (matches docs default mention) :contentReference[oaicite:2]{index=2}
        download_timeout_s: float = 1200.0, # 20 minutes for large file downloads
        max_retries: int = 2,              # OpenAI SDK retries some transient failures by default :contentReference[oaicite:3]{index=3}
    ):
        self.client = client or OpenAI(timeout=timeout_s, max_retries=max_retries)
        self._timeout_s = timeout_s
        self._download_timeout_s = download_timeout_s
        self._max_retries = max_retries

    def upload_file_for_batch(self, jsonl_path: Path) -> UploadedFile:
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Input JSONL not found: {jsonl_path}")
        with jsonl_path.open("rb") as f:
            uploaded = self.client.files.create(file=f, purpose="batch")
        return UploadedFile(file_id=uploaded.id, filename=jsonl_path.name)

    def create_batch(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Any:
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata or {},
        )

    def get_batch(self, batch_id: str) -> Any:
        return self.client.batches.retrieve(batch_id)

    def download_file(self, file_id: str, out_path: Path) -> Path:
        """
        Download a file from OpenAI Files API to out_path.

        Safety properties:
          - Uses a longer timeout for downloads (per-request override).
          - Writes to a temporary .part file then atomically replaces out_path,
            so downstream parsers never see a partially-written JSONL.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")

        # Ensure stale partial file doesn't confuse us
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

        # Use a per-request timeout override for large downloads. :contentReference[oaicite:4]{index=4}
        dl_client = self.client
        if hasattr(self.client, "with_options"):
            dl_client = self.client.with_options(timeout=self._download_timeout_s)

        try:
            content = dl_client.files.content(file_id)

            # openai-python often provides write_to_file
            if hasattr(content, "write_to_file"):
                content.write_to_file(str(tmp_path))
            elif hasattr(content, "read"):
                tmp_path.write_bytes(content.read())
            else:
                tmp_path.write_bytes(bytes(content))

            # Basic sanity check: file exists after write
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded file is empty or missing: {tmp_path}")

            # Atomic replace
            tmp_path.replace(out_path)
            return out_path

        except Exception:
            # Clean up partial download so next retry starts cleanly
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise

