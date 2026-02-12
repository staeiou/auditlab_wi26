# project_root/src/wandb_logging/tables.py

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import wandb


def _now() -> float:
    return time.time()


@dataclass
class TableLogger:
    """
    Optional W&B Table logger (best for SMALL runs).

    This is deliberately conservative:
      - buffers rows
      - flushes at most every `log_every_seconds`
      - flushes at most `chunk_size` rows per flush

    For large-scale batch outputs, prefer artifacts (CSV/JSONL) instead of tables.
    """

    run: wandb.sdk.wandb_run.Run
    table_name: str = "results_table"
    columns: Optional[List[str]] = None
    chunk_size: int = 500
    log_every_seconds: int = 120

    _table: wandb.Table = field(init=False)
    _buffer: List[Dict[str, Any]] = field(default_factory=list)
    _last_flush_unix: float = 0.0

    def __post_init__(self) -> None:
        self._table = wandb.Table(columns=self.columns) if self.columns else wandb.Table()

    def add_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        for r in rows:
            self._buffer.append(r)

    def flush_if_due(self, force: bool = False) -> bool:
        """
        Flush buffered rows to W&B (as a single Table log).
        Returns True if flushed, else False.
        """
        if not self._buffer:
            return False

        if not force:
            if (_now() - self._last_flush_unix) < self.log_every_seconds:
                return False

        # Log up to chunk_size rows per flush
        chunk = self._buffer[: self.chunk_size]
        self._buffer = self._buffer[self.chunk_size :]

        # Ensure stable columns if columns specified
        if self.columns:
            for r in chunk:
                self._table.add_data(*[r.get(c) for c in self.columns])
        else:
            # If no columns specified, infer keys from first row and keep adding dicts as JSON
            for r in chunk:
                self._table.add_data(r)

        self.run.log({self.table_name: self._table})
        self._last_flush_unix = _now()
        return True

    def flush_all(self) -> None:
        while self._buffer:
            self.flush_if_due(force=True)

