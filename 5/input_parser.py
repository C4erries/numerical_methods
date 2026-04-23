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
        if n < 2:
            raise ValueError("n values must be at least 2")
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


def _get_float(cfg: dict[str, str], *keys: str) -> float:
    for key in keys:
        if key in cfg:
            return float(cfg[key])
    raise ValueError(f"Missing required field: {'/'.join(keys)}")


def _n_values_from_h(x0: float, x1: float, y0: float, y1: float, h_values: list[float]) -> list[int]:
    x_len = x1 - x0
    y_len = y1 - y0
    n_values: list[int] = []
    for h in h_values:
        if h <= 0.0 or not np.isfinite(h):
            raise ValueError("h values must be positive and finite")
        nx_float = x_len / h
        ny_float = y_len / h
        nx = int(round(nx_float))
        ny = int(round(ny_float))
        if nx != ny:
            raise ValueError("h must give the same n in x and y for this version")
        if nx < 2 or abs(nx_float - nx) > 1e-10 * max(1.0, abs(nx_float)):
            raise ValueError(f"h={h:g} does not split x interval into an integer number of intervals")
        if abs(ny_float - ny) > 1e-10 * max(1.0, abs(ny_float)):
            raise ValueError(f"h={h:g} does not split y interval into an integer number of intervals")
        n_values.append(nx)
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

    x0 = _get_float(cfg, "x0", "a")
    x1 = _get_float(cfg, "x1", "b")
    y0 = _get_float(cfg, "y0", "c")
    y1 = _get_float(cfg, "y1", "d")
    if not np.isfinite([x0, x1, y0, y1]).all():
        raise ValueError("rectangle bounds must be finite")
    if x1 <= x0:
        raise ValueError("expected x0 < x1")
    if y1 <= y0:
        raise ValueError("expected y0 < y1")

    if "n_values" in cfg:
        n_values = _parse_int_list(cfg["n_values"])
    elif "n" in cfg:
        n_values = _parse_int_list(cfg["n"])
    elif "h_values" in cfg:
        n_values = _n_values_from_h(x0, x1, y0, y1, _parse_float_list(cfg["h_values"]))
    elif "h" in cfg:
        n_values = _n_values_from_h(x0, x1, y0, y1, [float(cfg["h"])])
    else:
        raise ValueError("Missing grid specification: provide n, n_values, h, or h_values")

    exact_n = int(float(cfg.get("exact_n", str(max(n_values) * 10))))
    if exact_n < 2:
        raise ValueError("exact_n must be at least 2")

    return {
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "n_values": n_values,
        "exact_n": exact_n,
    }
