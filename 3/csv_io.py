#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_xy_csv(path: Path, x: np.ndarray, y: np.ndarray, component: int = 0) -> None:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    if y_arr.ndim != 2:
        raise ValueError("y must be 1D or 2D array")
    if x_arr.size != y_arr.shape[0]:
        raise ValueError("x and y lengths mismatch")
    if component < 0 or component >= y_arr.shape[1]:
        raise ValueError("invalid component index")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise FloatingPointError("x/y contains NaN/inf")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        dim = y_arr.shape[1]
        if dim == 1:
            writer.writerow(["x", "y"])
            for xi, yi in zip(x_arr, y_arr[:, component]):
                writer.writerow([f"{xi:.12f}", f"{yi:.12e}"])
        else:
            writer.writerow(["x", *[f"y{k + 1}" for k in range(dim)]])
            for i in range(x_arr.size):
                writer.writerow([f"{x_arr[i]:.12f}", *[f"{v:.12e}" for v in y_arr[i, :]]])


def write_num_exact_delta_csv(
    path: Path,
    x: np.ndarray,
    y_num: np.ndarray,
    y_exact: np.ndarray,
    component: int = 0,
) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_num_arr = np.asarray(y_num, dtype=float)
    y_exact_arr = np.asarray(y_exact, dtype=float)

    if y_num_arr.ndim == 1:
        y_num_arr = y_num_arr.reshape(-1, 1)
    if y_exact_arr.ndim == 1:
        y_exact_arr = y_exact_arr.reshape(-1, 1)
    if y_num_arr.ndim != 2 or y_exact_arr.ndim != 2:
        raise ValueError("y_num and y_exact must be 1D or 2D arrays")
    if y_num_arr.shape != y_exact_arr.shape:
        raise ValueError("y_num and y_exact shapes mismatch")
    if x_arr.size != y_num_arr.shape[0]:
        raise ValueError("x and y arrays lengths mismatch")
    if component < 0 or component >= y_num_arr.shape[1]:
        raise ValueError("invalid component index")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_num_arr)) or not np.all(np.isfinite(y_exact_arr)):
        raise FloatingPointError("x/y contains NaN/inf")

    diff = y_num_arr - y_exact_arr
    if diff.shape[1] == 1:
        delta = np.abs(diff[:, 0])
    else:
        delta = np.max(np.abs(diff), axis=1)
    mean_err = float(np.mean(delta))
    max_err = float(np.max(delta))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        dim = y_num_arr.shape[1]
        if dim == 1:
            writer.writerow(["x", "y_num", "y_exact", "delta"])
            for xi, yn, ye, de in zip(x_arr, y_num_arr[:, component], y_exact_arr[:, component], delta):
                writer.writerow([f"{xi:.12f}", f"{yn:.12e}", f"{ye:.12e}", f"{de:.12e}"])
        else:
            header = (
                ["x"]
                + [f"y_num{k + 1}" for k in range(dim)]
                + [f"y_exact{k + 1}" for k in range(dim)]
                + ["delta"]
            )
            writer.writerow(header)
            for i in range(x_arr.size):
                writer.writerow(
                    [f"{x_arr[i]:.12f}"]
                    + [f"{v:.12e}" for v in y_num_arr[i, :]]
                    + [f"{v:.12e}" for v in y_exact_arr[i, :]]
                    + [f"{delta[i]:.12e}"]
                )

    return mean_err, max_err
