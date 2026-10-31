from __future__ import annotations

import csv
import json
import queue
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .shared_state import SharedState


@dataclass(frozen=True)
class WorkerError:
    name: str
    message: str
    traceback_text: str


def utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing OpenCV Python bindings. Install with: python3 -m pip install opencv-python"
        ) from exc
    return cv2


def worker_wrapper(
    name: str,
    errors: queue.Queue[WorkerError],
    shared: SharedState,
    init_done,
    target,
    *args,
) -> None:
    try:
        target(*args, init_done=init_done)
    except Exception as exc:
        errors.put(
            WorkerError(
                name=name,
                message=str(exc),
                traceback_text=traceback.format_exc(),
            )
        )
        shared.shutdown.set()
        init_done.set()
