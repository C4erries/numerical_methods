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


def _parse_float_list(text: str) -> list[float]:
    clean = text.replace(",", " ")
    parts = [p for p in clean.split() if p]
    if not parts:
        raise ValueError("Expected non-empty list of numbers")
    return [float(v) for v in parts]


def _parse_int_list(text: str) -> list[int]:
    result: list[int] = []
    for value in _parse_float_list(text):
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError("n values must be integers")
        n = int(value)
        if n <= 0:
            raise ValueError("n values must be positive")
        result.append(n)
    return result


def _parse_key_value(line: str) -> tuple[str, str] | None:
    if ":" in line:
        key, val = line.split(":", 1)
        return key.strip().lower(), val.strip()
    if "=" in line:
        key, val = line.split("=", 1)
        return key.strip().lower(), val.strip()
    return None


def _n_values_from_h(a: float, b: float, h_values: list[float]) -> list[int]:
    interval = b - a
    n_values: list[int] = []
    for h in h_values:
        if h <= 0.0 or not np.isfinite(h):
            raise ValueError("h values must be positive and finite")
        n_float = interval / h
        n_round = int(round(n_float))
        if n_round <= 0 or abs(n_float - n_round) > 1e-10 * max(1.0, abs(n_float)):
            raise ValueError(f"h={h:g} does not split [{a}, {b}] into an integer number of intervals")
        n_values.append(n_round)
    return n_values


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

    required = ("a", "b", "y_left", "dy_right")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    a = float(cfg["a"])
    b = float(cfg["b"])
    if not np.isfinite([a, b]).all():
        raise ValueError("a and b must be finite")
    if b <= a:
        raise ValueError("expected a < b")

    if "n_values" in cfg:
        n_values = _parse_int_list(cfg["n_values"])
    elif "n" in cfg:
        n_values = _parse_int_list(cfg["n"])
    elif "h_values" in cfg:
        n_values = _n_values_from_h(a, b, _parse_float_list(cfg["h_values"]))
    elif "h" in cfg:
        n_values = _n_values_from_h(a, b, [float(cfg["h"])])
    else:
        raise ValueError("Missing grid specification: provide n, n_values, h, or h_values")

    exact_n = int(float(cfg.get("exact_n", str(max(n_values) * 10))))
    if exact_n <= 0:
        raise ValueError("exact_n must be positive")

    boundary_order = int(float(cfg.get("boundary_order", "2")))
    if boundary_order not in (1, 2):
        raise ValueError("boundary_order must be 1 or 2")

    return {
        "a": a,
        "b": b,
        "y_left": float(cfg["y_left"]),
        "dy_right": float(cfg["dy_right"]),
        "n_values": n_values,
        "exact_n": exact_n,
        "boundary_order": boundary_order,
    }
