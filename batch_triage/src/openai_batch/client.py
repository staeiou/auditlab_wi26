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

    Methods:
      - upload_file_for_batch(jsonl_path) -> UploadedFile
      - create_batch(input_file_id, endpoint, completion_window, metadata) -> batch object
      - get_batch(batch_id) -> batch object
      - download_file(file_id, out_path) -> Path

    Notes:
      - Assumes OPENAI_API_KEY is set in env unless you pass a configured OpenAI client.
      - Works with OpenAI-compatible endpoints too, if you construct OpenAI(base_url=..., api_key=...).
    """

    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client or OpenAI()

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
        # returns OpenAI SDK batch object
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
        Compatible with openai-python's content streaming interface.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.client.files.content(file_id)

        # openai-python often provides write_to_file
        if hasattr(content, "write_to_file"):
            content.write_to_file(str(out_path))
            return out_path

        # fallback: file-like .read()
        if hasattr(content, "read"):
            out_path.write_bytes(content.read())
            return out_path

        # fallback: bytes-like
        out_path.write_bytes(bytes(content))
        return out_path

