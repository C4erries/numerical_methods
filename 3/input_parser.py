#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8")


def _parse_vector(text: str) -> np.ndarray:
    clean = text.strip()
    if clean.startswith("[") and clean.endswith("]"):
        clean = clean[1:-1]
    clean = clean.replace(",", " ")
    parts = [p for p in clean.split() if p]
    if not parts:
        raise ValueError("Vector value is empty")
    return np.array([float(v) for v in parts], dtype=float)


def _parse_float_list(text: str) -> list[float]:
    clean = text.replace(",", " ")
    parts = [p for p in clean.split() if p]
    if not parts:
        raise ValueError("Expected non-empty list of numbers")
    return [float(v) for v in parts]


def _parse_key_value(line: str) -> tuple[str, str] | None:
    if ":" in line:
        key, val = line.split(":", 1)
        return key.strip().lower(), val.strip()
    if "=" in line:
        key, val = line.split("=", 1)
        return key.strip().lower(), val.strip()
    return None


def _as_scalar_or_vector(arr: np.ndarray) -> float | np.ndarray:
    vec = np.asarray(arr, dtype=float).reshape(-1)
    if vec.size == 1:
        return float(vec[0])
    return vec


def read_input_config(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)

    cfg: dict[str, str] = {}
    for line in lines:
        parsed = _parse_key_value(line)
        if parsed is None:
            continue
        key, val = parsed
        cfg[key] = val

    required = ("t0", "y0", "t_end")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    if "h_values" in cfg:
        h_values = _parse_float_list(cfg["h_values"])
    elif "h" in cfg:
        h_values = [float(cfg["h"])]
    else:
        raise ValueError("Missing step specification: provide h or h_values")

    if any(h == 0.0 for h in h_values):
        raise ValueError("All h values must be non-zero")

    result = {
        "t0": float(cfg["t0"]),
        "y0": _as_scalar_or_vector(_parse_vector(cfg["y0"])),
        "t_end": float(cfg["t_end"]),
        "h_values": h_values,
        "tol": float(cfg.get("tol", "1e-10")),
        "max_iter": int(cfg.get("max_iter", "20")),
        "exact_h": float(cfg["exact_h"]) if "exact_h" in cfg else None,
    }
    return result
