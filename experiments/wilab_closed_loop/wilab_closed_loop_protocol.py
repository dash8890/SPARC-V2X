"""Shared file protocol for the isolated MATLAB/Python WiLab bridge."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def wait_json(path: Path, timeout_s: float = 300.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(0.001)
    raise TimeoutError(f"Timed out waiting for {path}")
