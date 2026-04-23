#!/usr/bin/env python3
from __future__ import annotations

from typing import Callable

import numpy as np


Scalar2D = Callable[[float, float], float]


def _validate_rectangle(x0: float, x1: float, y0: float, y1: float, n: int) -> tuple[float, float, float, float, int]:
    x0 = float(x0)
    x1 = float(x1)
    y0 = float(y0)
    y1 = float(y1)
    n_float = float(n)
    if not np.isfinite([x0, x1, y0, y1]).all():
        raise ValueError("rectangle bounds must be finite")
    if x1 <= x0:
        raise ValueError("expected x0 < x1")
    if y1 <= y0:
        raise ValueError("expected y0 < y1")
    if not np.isfinite(n_float) or not n_float.is_integer():
        raise ValueError("n must be an integer")
    n = int(n_float)
    if n < 2:
        raise ValueError("n must be at least 2 for a 2D interior grid")
    return x0, x1, y0, y1, n


def _eval_scalar2d(func: Scalar2D, x: float, y: float, name: str) -> float:
    arr = np.asarray(func(float(x), float(y)), dtype=float)
    if arr.ndim != 0:
        raise ValueError(f"{name}({x}, {y}) must return a scalar")
    value = float(arr)
    if not np.isfinite(value):
        raise FloatingPointError(f"{name}({x}, {y}) is not finite")
    return value


def build_uniform_grid_2d(x0: float, x1: float, y0: float, y1: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1, n = _validate_rectangle(x0, x1, y0, y1, n)
    x = np.linspace(x0, x1, n + 1, dtype=float)
    y = np.linspace(y0, y1, n + 1, dtype=float)
    return x, y


def _interior_index(i: int, j: int, interior_count: int) -> int:
    return (j - 1) * interior_count + (i - 1)


def build_poisson_dirichlet_system(
    f: Scalar2D,
    boundary: Scalar2D,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0, x1, y0, y1, n = _validate_rectangle(x0, x1, y0, y1, n)
    x, y = build_uniform_grid_2d(x0, x1, y0, y1, n)
    hx = (x1 - x0) / n
    hy = (y1 - y0) / n
    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)

    interior_count = n - 1
    size = interior_count * interior_count
    matrix = np.zeros((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)

    for j in range(1, n):
        for i in range(1, n):
            row = _interior_index(i, j, interior_count)
            xi = x[i]
            yj = y[j]

            matrix[row, row] = -2.0 * inv_hx2 - 2.0 * inv_hy2
            rhs[row] = _eval_scalar2d(f, xi, yj, "f")

            neighbors = (
                (i - 1, j, inv_hx2),
                (i + 1, j, inv_hx2),
                (i, j - 1, inv_hy2),
                (i, j + 1, inv_hy2),
            )
            for ni, nj, coeff in neighbors:
                if ni == 0 or ni == n or nj == 0 or nj == n:
                    rhs[row] -= coeff * _eval_scalar2d(boundary, x[ni], y[nj], "boundary")
                else:
                    col = _interior_index(ni, nj, interior_count)
                    matrix[row, col] = coeff

    return x, y, matrix, rhs


def solve_poisson_dirichlet(
    f: Scalar2D,
    boundary: Scalar2D,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    x, y, matrix, rhs = build_poisson_dirichlet_system(
        f=f,
        boundary=boundary,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        n=n,
    )

    interior = np.linalg.solve(matrix, rhs)
    residual = matrix @ interior - rhs

    u = np.zeros((n + 1, n + 1), dtype=float)
    for j, yj in enumerate(y):
        for i, xi in enumerate(x):
            if i == 0 or i == n or j == 0 or j == n:
                u[j, i] = _eval_scalar2d(boundary, xi, yj, "boundary")

    interior_count = n - 1
    for j in range(1, n):
        for i in range(1, n):
            u[j, i] = interior[_interior_index(i, j, interior_count)]

    if not np.all(np.isfinite(u)):
        raise FloatingPointError("solution contains NaN or inf")

    info = {
        "method": "np.linalg.solve",
        "n": int(n),
        "hx": float((x1 - x0) / n),
        "hy": float((y1 - y0) / n),
        "points": int((n + 1) * (n + 1)),
        "unknowns": int((n - 1) * (n - 1)),
        "residual_norm": float(np.linalg.norm(residual, ord=np.inf)),
        "message": "ok",
    }
    return x, y, u, info


solve_poisson_dense = solve_poisson_dirichlet
