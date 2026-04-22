#!/usr/bin/env python3
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


Coefficient = Callable[[float], float] | Sequence[float] | np.ndarray | float | int
VectorLike = Sequence[float] | np.ndarray


def _as_float_vector(values: VectorLike, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr.astype(float, copy=True)


def _validate_interval(a: float, b: float, n: int) -> tuple[float, float, int]:
    a = float(a)
    b = float(b)
    n_float = float(n)
    if not np.isfinite([a, b]).all():
        raise ValueError("a and b must be finite")
    if b <= a:
        raise ValueError("expected a < b")
    if not np.isfinite(n_float) or not n_float.is_integer():
        raise ValueError("n must be an integer")
    n = int(n_float)
    if n < 1:
        raise ValueError("n must be positive")
    return a, b, n


def build_uniform_grid(a: float, b: float, n: int) -> np.ndarray:
    a, b, n = _validate_interval(a, b, n)
    return np.linspace(a, b, n + 1, dtype=float)


def _eval_coefficient(coef: Coefficient, x: np.ndarray, name: str) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).reshape(-1)

    if callable(coef):
        values: list[float] = []
        for xi in x_arr:
            val = np.asarray(coef(float(xi)), dtype=float)
            if val.ndim != 0:
                raise ValueError(f"{name}(x) must return a scalar")
            values.append(float(val))
        arr = np.asarray(values, dtype=float)
    else:
        arr = np.asarray(coef, dtype=float)
        if arr.ndim == 0 or arr.size == 1:
            arr = np.full(x_arr.size, float(arr.reshape(-1)[0]), dtype=float)
        elif arr.ndim == 1 and arr.size == x_arr.size:
            arr = arr.astype(float, copy=True)
        else:
            raise ValueError(f"{name} must be scalar or have length {x_arr.size}")

    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"{name} contains NaN or inf")
    return arr


def _validate_tridiagonal(
    lower: VectorLike,
    diagonal: VectorLike,
    upper: VectorLike,
    rhs: VectorLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower_arr = _as_float_vector(lower, "lower")
    diagonal_arr = _as_float_vector(diagonal, "diagonal")
    upper_arr = _as_float_vector(upper, "upper")
    rhs_arr = _as_float_vector(rhs, "rhs")

    m = diagonal_arr.size
    if m == 0:
        raise ValueError("diagonal must not be empty")
    if rhs_arr.size != m:
        raise ValueError("rhs length must match diagonal length")
    if lower_arr.size != max(0, m - 1):
        raise ValueError("lower length must be len(diagonal) - 1")
    if upper_arr.size != max(0, m - 1):
        raise ValueError("upper length must be len(diagonal) - 1")
    return lower_arr, diagonal_arr, upper_arr, rhs_arr


def thomas_algorithm(lower: VectorLike, diagonal: VectorLike, upper: VectorLike, rhs: VectorLike) -> np.ndarray:
    """Solve a tridiagonal linear system by the Thomas sweep algorithm."""
    lower_arr, diagonal_arr, upper_arr, rhs_arr = _validate_tridiagonal(lower, diagonal, upper, rhs)
    m = diagonal_arr.size

    c_prime = np.zeros(max(0, m - 1), dtype=float)
    d_prime = np.zeros(m, dtype=float)

    scale = max(
        1.0,
        float(np.max(np.abs(lower_arr))) if lower_arr.size else 0.0,
        float(np.max(np.abs(diagonal_arr))),
        float(np.max(np.abs(upper_arr))) if upper_arr.size else 0.0,
    )
    pivot_tol = 10.0 * np.finfo(float).eps * scale

    pivot = diagonal_arr[0]
    if abs(pivot) <= pivot_tol:
        raise ZeroDivisionError("zero pivot in Thomas algorithm at row 0")
    if m > 1:
        c_prime[0] = upper_arr[0] / pivot
    d_prime[0] = rhs_arr[0] / pivot

    for i in range(1, m):
        pivot = diagonal_arr[i] - lower_arr[i - 1] * c_prime[i - 1]
        if abs(pivot) <= pivot_tol:
            raise ZeroDivisionError(f"zero pivot in Thomas algorithm at row {i}")
        if i < m - 1:
            c_prime[i] = upper_arr[i] / pivot
        d_prime[i] = (rhs_arr[i] - lower_arr[i - 1] * d_prime[i - 1]) / pivot

    solution = np.zeros(m, dtype=float)
    solution[-1] = d_prime[-1]
    for i in range(m - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]

    if not np.all(np.isfinite(solution)):
        raise FloatingPointError("Thomas algorithm produced NaN or inf")
    return solution


def tridiagonal_matvec(lower: VectorLike, diagonal: VectorLike, upper: VectorLike, values: VectorLike) -> np.ndarray:
    """Multiply a tridiagonal matrix by a vector using its three diagonals."""
    values_arr = _as_float_vector(values, "values")
    lower_arr, diagonal_arr, upper_arr, _ = _validate_tridiagonal(lower, diagonal, upper, values_arr)

    result = diagonal_arr * values_arr
    if values_arr.size > 1:
        result[1:] += lower_arr * values_arr[:-1]
        result[:-1] += upper_arr * values_arr[1:]
    return result


def build_finite_difference_system(
    p: Coefficient,
    q: Coefficient,
    f: Coefficient,
    a: float,
    b: float,
    n: int,
    y_left: float,
    dy_right: float,
    boundary_order: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the finite-difference system for Dirichlet-left/Neumann-right BVP."""
    a, b, n = _validate_interval(a, b, n)
    y_left = float(y_left)
    dy_right = float(dy_right)
    if not np.isfinite([y_left, dy_right]).all():
        raise ValueError("boundary values must be finite")
    if boundary_order not in (1, 2):
        raise ValueError("boundary_order must be 1 or 2")

    x = build_uniform_grid(a, b, n)
    h = (b - a) / n
    inv_h = 1.0 / h
    inv_h2 = inv_h * inv_h

    p_values = _eval_coefficient(p, x, "p")
    q_values = _eval_coefficient(q, x, "q")
    f_values = _eval_coefficient(f, x, "f")

    size = n + 1
    lower = np.zeros(size - 1, dtype=float)
    diagonal = np.zeros(size, dtype=float)
    upper = np.zeros(size - 1, dtype=float)
    rhs = np.zeros(size, dtype=float)

    diagonal[0] = 1.0
    rhs[0] = y_left

    for i in range(1, n):
        lower[i - 1] = inv_h2 - 0.5 * p_values[i] * inv_h
        diagonal[i] = -2.0 * inv_h2 + q_values[i]
        upper[i] = inv_h2 + 0.5 * p_values[i] * inv_h
        rhs[i] = f_values[i]

    if boundary_order == 1:
        lower[n - 1] = -inv_h
        diagonal[n] = inv_h
        rhs[n] = dy_right
    else:
        lower[n - 1] = 2.0 * inv_h2
        diagonal[n] = -2.0 * inv_h2 + q_values[n]
        rhs[n] = f_values[n] - p_values[n] * dy_right - 2.0 * dy_right * inv_h

    return x, lower, diagonal, upper, rhs


def solve_finite_difference_bvp(
    p: Coefficient,
    q: Coefficient,
    f: Coefficient,
    a: float,
    b: float,
    n: int,
    y_left: float,
    dy_right: float,
    boundary_order: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve y'' + p(x)y' + q(x)y = f(x), y(a)=y_left, y'(b)=dy_right."""
    x, lower, diagonal, upper, rhs = build_finite_difference_system(
        p=p,
        q=q,
        f=f,
        a=a,
        b=b,
        n=n,
        y_left=y_left,
        dy_right=dy_right,
        boundary_order=boundary_order,
    )
    y = thomas_algorithm(lower, diagonal, upper, rhs)
    residual = tridiagonal_matvec(lower, diagonal, upper, y) - rhs

    info = {
        "n": int(n),
        "h": float((b - a) / int(n)),
        "boundary_order": int(boundary_order),
        "residual_norm": float(np.linalg.norm(residual, ord=np.inf)),
        "message": "ok",
    }
    return x, y, info


def solve_bvp_fdm_dirichlet_neumann(
    p: Coefficient,
    q: Coefficient,
    f: Coefficient,
    a: float,
    b: float,
    n: int,
    y_left: float,
    dy_right: float,
    boundary_order: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    return solve_finite_difference_bvp(
        p=p,
        q=q,
        f=f,
        a=a,
        b=b,
        n=n,
        y_left=y_left,
        dy_right=dy_right,
        boundary_order=boundary_order,
    )


solve_tridiagonal = thomas_algorithm
sweep_method = thomas_algorithm
solve_boundary_value_problem = solve_finite_difference_bvp
finite_difference_method = solve_finite_difference_bvp
