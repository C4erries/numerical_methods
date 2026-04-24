#!/usr/bin/env python3
from __future__ import annotations

from typing import Callable

import numpy as np


Scalar2D = Callable[[float, float], float]
LinearMethod = str


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


def solve_linear_system_gauss_seidel(
    matrix: np.ndarray,
    rhs: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 10000,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float).reshape(-1)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    if b.size != a.shape[0]:
        raise ValueError("rhs length must match matrix size")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("matrix and rhs must contain finite values")
    if tol <= 0.0 or not np.isfinite(tol):
        raise ValueError("tol must be positive and finite")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    n = b.size
    diagonal = np.diag(a)
    if np.any(diagonal == 0.0):
        raise ZeroDivisionError("zero diagonal element in Gauss-Seidel method")
    row_entries: list[list[tuple[int, float]]] = []
    for i in range(n):
        entries: list[tuple[int, float]] = []
        for j in np.flatnonzero(a[i]):
            if j != i:
                entries.append((int(j), float(a[i, j])))
        row_entries.append(entries)

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1).copy()
        if x.size != n:
            raise ValueError("x0 length must match rhs length")
        if not np.all(np.isfinite(x)):
            raise ValueError("x0 must contain finite values")

    converged = False
    iterations = 0
    residual_norm = float(np.linalg.norm(a @ x - b, ord=np.inf))
    correction_norm = float("inf")

    for iteration in range(1, max_iter + 1):
        old = x.copy()
        for i in range(n):
            row_sum = 0.0
            for j, coeff in row_entries[i]:
                row_sum += coeff * x[j]
            x[i] = (b[i] - row_sum) / diagonal[i]

        correction_norm = float(np.linalg.norm(x - old, ord=np.inf))
        iterations = iteration
        if correction_norm < tol:
            residual_norm = float(np.linalg.norm(a @ x - b, ord=np.inf))
            converged = True
            break

    if not np.all(np.isfinite(x)):
        raise FloatingPointError("Gauss-Seidel method produced NaN or inf")
    if not converged:
        residual_norm = float(np.linalg.norm(a @ x - b, ord=np.inf))

    info = {
        "converged": converged,
        "iterations": iterations,
        "correction_norm": correction_norm,
        "linear_residual_norm": residual_norm,
        "message": "ok" if converged else "Gauss-Seidel did not converge within max_iter.",
    }
    return x, info


def solve_linear_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
    method: LinearMethod = "dense",
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> tuple[np.ndarray, dict]:
    method_key = method.strip().lower()
    if method_key in ("dense", "numpy", "np.linalg.solve", "linalg"):
        solution = np.linalg.solve(matrix, rhs)
        residual = matrix @ solution - rhs
        return solution, {
            "method": "np.linalg.solve",
            "converged": True,
            "iterations": 1,
            "correction_norm": 0.0,
            "linear_residual_norm": float(np.linalg.norm(residual, ord=np.inf)),
            "message": "ok",
        }
    if method_key in ("seidel", "gauss-seidel", "gauss_seidel", "gs"):
        solution, info = solve_linear_system_gauss_seidel(matrix, rhs, tol=tol, max_iter=max_iter)
        info["method"] = "Gauss-Seidel"
        return solution, info
    raise ValueError("unknown linear solver method: use 'dense' or 'seidel'")


def solve_poisson_dirichlet(
    f: Scalar2D,
    boundary: Scalar2D,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    n: int,
    method: LinearMethod = "dense",
    tol: float = 1e-10,
    max_iter: int = 10000,
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

    interior, linear_info = solve_linear_system(matrix, rhs, method=method, tol=tol, max_iter=max_iter)
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
        "method": linear_info["method"],
        "converged": bool(linear_info["converged"]),
        "iterations": int(linear_info["iterations"]),
        "correction_norm": float(linear_info["correction_norm"]),
        "n": int(n),
        "hx": float((x1 - x0) / n),
        "hy": float((y1 - y0) / n),
        "points": int((n + 1) * (n + 1)),
        "unknowns": int((n - 1) * (n - 1)),
        "residual_norm": float(np.linalg.norm(residual, ord=np.inf)),
        "message": linear_info["message"],
    }
    return x, y, u, info


solve_poisson_dense = solve_poisson_dirichlet
solve_linear_system_seidel = solve_linear_system_gauss_seidel
