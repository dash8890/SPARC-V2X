"""Shared file protocol for the isolated MATLAB/Python WiLab bridge."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), default=_json_default), encoding="utf-8")
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
