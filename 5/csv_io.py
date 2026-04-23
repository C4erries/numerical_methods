#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _validate_grid(x: np.ndarray, y: np.ndarray, values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    val_arr = np.asarray(values, dtype=float)
    if val_arr.shape != (y_arr.size, x_arr.size):
        raise ValueError(f"{name} shape must be (len(y), len(x))")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)) or not np.all(np.isfinite(val_arr)):
        raise FloatingPointError(f"{name}/grid contains NaN or inf")
    return x_arr, y_arr, val_arr


def write_grid_csv(path: Path, x: np.ndarray, y: np.ndarray, values: np.ndarray, value_name: str = "u") -> None:
    x_arr, y_arr, val_arr = _validate_grid(x, y, values, value_name)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", value_name])
        for j, yj in enumerate(y_arr):
            for i, xi in enumerate(x_arr):
                writer.writerow([f"{xi:.12f}", f"{yj:.12f}", f"{val_arr[j, i]:.12e}"])


def write_num_exact_delta_grid_csv(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    u_num: np.ndarray,
    u_exact: np.ndarray,
) -> tuple[float, float]:
    x_arr, y_arr, u_num_arr = _validate_grid(x, y, u_num, "u_num")
    _, _, u_exact_arr = _validate_grid(x_arr, y_arr, u_exact, "u_exact")

    delta = np.abs(u_num_arr - u_exact_arr)
    mean_err = float(np.mean(delta))
    max_err = float(np.max(delta))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "u_num", "u_exact", "delta"])
        for j, yj in enumerate(y_arr):
            for i, xi in enumerate(x_arr):
                writer.writerow(
                    [
                        f"{xi:.12f}",
                        f"{yj:.12f}",
                        f"{u_num_arr[j, i]:.12e}",
                        f"{u_exact_arr[j, i]:.12e}",
                        f"{delta[j, i]:.12e}",
                    ]
                )

    return mean_err, max_err
