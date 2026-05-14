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


def _sample_boundary_grid(boundary: Scalar2D, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_y = y.size
    n_x = x.size
    u_boundary = np.zeros((n_y, n_x), dtype=float)
    for j in range(n_y):
        for i in range(n_x):
            if i == 0 or i == n_x - 1 or j == 0 or j == n_y - 1:
                u_boundary[j, i] = _eval_scalar2d(boundary, x[i], y[j], "boundary")
    return u_boundary


def _sample_rhs_interior(f: Scalar2D, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.size - 1
    interior_count = n - 1
    rhs = np.zeros((interior_count, interior_count), dtype=float)
    for j in range(1, n):
        for i in range(1, n):
            rhs[j - 1, i - 1] = _eval_scalar2d(f, x[i], y[j], "f")
    return rhs


def _five_point_residual(u: np.ndarray, f_interior: np.ndarray, hx: float, hy: float) -> float:
    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)
    interior = u[1:-1, 1:-1]
    left = u[1:-1, :-2]
    right = u[1:-1, 2:]
    down = u[:-2, 1:-1]
    up = u[2:, 1:-1]
    lap = (left - 2.0 * interior + right) * inv_hx2 + (down - 2.0 * interior + up) * inv_hy2
    return float(np.linalg.norm(lap - f_interior, ord=np.inf))


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


def build_poisson_dirichlet_system_sparse(
    f: Scalar2D,
    boundary: Scalar2D,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    n: int,
):
    from scipy.sparse import csr_matrix

    x0, x1, y0, y1, n = _validate_rectangle(x0, x1, y0, y1, n)
    x, y = build_uniform_grid_2d(x0, x1, y0, y1, n)
    hx = (x1 - x0) / n
    hy = (y1 - y0) / n
    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)

    interior_count = n - 1
    size = interior_count * interior_count
    rhs = np.zeros(size, dtype=float)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for j in range(1, n):
        for i in range(1, n):
            row = _interior_index(i, j, interior_count)
            rows.append(row)
            cols.append(row)
            data.append(-2.0 * inv_hx2 - 2.0 * inv_hy2)
            rhs[row] = _eval_scalar2d(f, x[i], y[j], "f")

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
                    rows.append(row)
                    cols.append(col)
                    data.append(coeff)

    matrix = csr_matrix((data, (rows, cols)), shape=(size, size))
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


def _thomas_solve(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = diag.size
    cp = np.empty(n - 1, dtype=float)
    dp = np.empty(n, dtype=float)

    beta = diag[0]
    if beta == 0.0:
        raise ZeroDivisionError("zero pivot in Thomas algorithm")
    cp[0] = sup[0] / beta
    dp[0] = rhs[0] / beta
    for i in range(1, n):
        denom = diag[i] - sub[i - 1] * (cp[i - 1] if i - 1 < n - 1 else 0.0)
        if denom == 0.0:
            raise ZeroDivisionError("zero pivot in Thomas algorithm")
        if i < n - 1:
            cp[i] = sup[i] / denom
        dp[i] = (rhs[i] - sub[i - 1] * dp[i - 1]) / denom

    x = np.empty(n, dtype=float)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def solve_poisson_adi(
    f: Scalar2D,
    boundary: Scalar2D,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    n: int,
    tol: float = 1e-8,
    max_iter: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    x0, x1, y0, y1, n = _validate_rectangle(x0, x1, y0, y1, n)
    x, y = build_uniform_grid_2d(x0, x1, y0, y1, n)
    hx = (x1 - x0) / n
    hy = (y1 - y0) / n
    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)

    m = n - 1  # interior count along each axis
    u_full = _sample_boundary_grid(boundary, x, y)
    f_int = _sample_rhs_interior(f, x, y)

    u = u_full[1:-1, 1:-1].copy()

    # boundary slices used in every iteration
    left_col = u_full[1:-1, 0]
    right_col = u_full[1:-1, -1]
    bottom_row = u_full[0, 1:-1]
    top_row = u_full[-1, 1:-1]

    # Single-tau Peaceman-Rachford: optimal tau ~ sqrt(L*h)/pi for square Poisson;
    # using tau = h gives convergence factor ~ (1 - 2*pi*h) per iteration on unit square.
    h = max(hx, hy)
    L = max(x1 - x0, y1 - y0)
    tau = float(np.sqrt(L * h) / np.pi)
    inv_tau = 1.0 / tau

    # x-sweep tridiagonal: row j, unknowns u_{i,j}, i = 1..m
    # (I/tau - L_x) u^{1/2} = (I/tau + L_y) u^k - f
    # diag = inv_tau + 2*inv_hx2, off = -inv_hx2
    diag_x = np.full(m, inv_tau + 2.0 * inv_hx2)
    sub_x = np.full(m - 1, -inv_hx2)
    sup_x = np.full(m - 1, -inv_hx2)

    diag_y = np.full(m, inv_tau + 2.0 * inv_hy2)
    sub_y = np.full(m - 1, -inv_hy2)
    sup_y = np.full(m - 1, -inv_hy2)

    converged = False
    iterations = 0
    correction_norm = float("inf")

    for iteration in range(1, max_iter + 1):
        old = u.copy()

        # half-step: implicit in x, explicit in y
        # RHS for row j: (inv_tau - 2*inv_hy2) u_{i,j} + inv_hy2 (u_{i,j-1} + u_{i,j+1}) - f_{i,j}
        # boundary contribution adds -(-inv_hx2)*u_left at i=1 and i=m
        u_half = np.empty_like(u)
        explicit_y = (inv_tau - 2.0 * inv_hy2) * u
        # add u_{i, j-1} and u_{i, j+1}
        explicit_y[:, :] += inv_hy2 * (
            np.vstack([bottom_row[np.newaxis, :], u[:-1, :]])
            + np.vstack([u[1:, :], top_row[np.newaxis, :]])
        )
        rhs_x = explicit_y - f_int
        rhs_x[:, 0] = rhs_x[:, 0] + inv_hx2 * left_col
        rhs_x[:, -1] = rhs_x[:, -1] + inv_hx2 * right_col

        for j in range(m):
            u_half[j, :] = _thomas_solve(sub_x, diag_x, sup_x, rhs_x[j, :])

        # half-step: implicit in y, explicit in x
        explicit_x = (inv_tau - 2.0 * inv_hx2) * u_half
        explicit_x[:, :] += inv_hx2 * (
            np.hstack([left_col[:, np.newaxis], u_half[:, :-1]])
            + np.hstack([u_half[:, 1:], right_col[:, np.newaxis]])
        )
        rhs_y = explicit_x - f_int
        rhs_y[0, :] = rhs_y[0, :] + inv_hy2 * bottom_row
        rhs_y[-1, :] = rhs_y[-1, :] + inv_hy2 * top_row

        u_new = np.empty_like(u)
        for i in range(m):
            u_new[:, i] = _thomas_solve(sub_y, diag_y, sup_y, rhs_y[:, i])

        u = u_new
        correction_norm = float(np.linalg.norm(u - old, ord=np.inf))
        iterations = iteration
        if correction_norm < tol:
            converged = True
            break

    u_full[1:-1, 1:-1] = u

    if not np.all(np.isfinite(u_full)):
        raise FloatingPointError("ADI solver produced NaN or inf")

    residual_norm = _five_point_residual(u_full, f_int, hx, hy)
    info = {
        "method": "ADI Peaceman-Rachford",
        "converged": converged,
        "iterations": iterations,
        "correction_norm": correction_norm,
        "n": int(n),
        "hx": float(hx),
        "hy": float(hy),
        "points": int((n + 1) * (n + 1)),
        "unknowns": int(m * m),
        "residual_norm": float(residual_norm),
        "message": "ok" if converged else "ADI did not converge within max_iter.",
    }
    return x, y, u_full, info


def solve_poisson_fft(
    f: Scalar2D,
    boundary: Scalar2D,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    from scipy.fft import dst

    x0, x1, y0, y1, n = _validate_rectangle(x0, x1, y0, y1, n)
    x, y = build_uniform_grid_2d(x0, x1, y0, y1, n)
    hx = (x1 - x0) / n
    hy = (y1 - y0) / n
    if not np.isclose(hx, hy, rtol=1e-12, atol=0.0):
        raise ValueError("FFT/DST-I solver requires hx == hy")
    h = hx
    inv_h2 = 1.0 / (h * h)

    m = n - 1
    u_full = _sample_boundary_grid(boundary, x, y)
    f_int = _sample_rhs_interior(f, x, y)

    # boundary contribution shifted to RHS: subtract neighbor boundary / h^2
    rhs_grid = f_int.copy()
    rhs_grid[:, 0] -= inv_h2 * u_full[1:-1, 0]
    rhs_grid[:, -1] -= inv_h2 * u_full[1:-1, -1]
    rhs_grid[0, :] -= inv_h2 * u_full[0, 1:-1]
    rhs_grid[-1, :] -= inv_h2 * u_full[-1, 1:-1]

    # DST-I along both axes; scipy dst type=1, default norm "backward" -> inverse via the same dst divided by 2*(N+1)
    f_hat = dst(dst(rhs_grid, type=1, axis=0), type=1, axis=1)

    # eigenvalues of discrete Laplacian for DST-I basis
    k = np.arange(1, m + 1)
    lam_x = 2.0 * inv_h2 * (np.cos(np.pi * k / n) - 1.0)
    lam_y = 2.0 * inv_h2 * (np.cos(np.pi * k / n) - 1.0)
    Lam = lam_x[np.newaxis, :] + lam_y[:, np.newaxis]

    u_hat = f_hat / Lam

    # inverse DST-I: applying dst type=1 twice yields scaling by (2*(N+1))^2 with N+1 = n
    u_int = dst(dst(u_hat, type=1, axis=0), type=1, axis=1) / (2.0 * n) ** 2

    u_full[1:-1, 1:-1] = u_int

    if not np.all(np.isfinite(u_full)):
        raise FloatingPointError("FFT solver produced NaN or inf")

    residual_norm = _five_point_residual(u_full, f_int, hx, hy)
    info = {
        "method": "FFT/DST-I",
        "converged": True,
        "iterations": 1,
        "correction_norm": 0.0,
        "n": int(n),
        "hx": float(hx),
        "hy": float(hy),
        "points": int((n + 1) * (n + 1)),
        "unknowns": int(m * m),
        "residual_norm": float(residual_norm),
        "message": "ok",
    }
    return x, y, u_full, info


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
    method_key = method.strip().lower()

    if method_key in ("adi", "peaceman-rachford", "peaceman_rachford"):
        return solve_poisson_adi(
            f=f, boundary=boundary,
            x0=x0, x1=x1, y0=y0, y1=y1, n=n,
            tol=tol, max_iter=max_iter,
        )

    if method_key in ("fft", "dst", "dst-i"):
        return solve_poisson_fft(
            f=f, boundary=boundary,
            x0=x0, x1=x1, y0=y0, y1=y1, n=n,
        )

    if method_key in ("sparse", "scipy", "spsolve"):
        from scipy.sparse.linalg import spsolve

        x, y, matrix, rhs = build_poisson_dirichlet_system_sparse(
            f=f, boundary=boundary, x0=x0, x1=x1, y0=y0, y1=y1, n=n,
        )
        interior = spsolve(matrix.tocsc(), rhs)
        residual = matrix @ interior - rhs
        linear_info = {
            "method": "scipy.sparse.spsolve",
            "converged": True,
            "iterations": 1,
            "correction_norm": 0.0,
            "linear_residual_norm": float(np.linalg.norm(residual, ord=np.inf)),
            "message": "ok",
        }
    else:
        x, y, matrix, rhs = build_poisson_dirichlet_system(
            f=f, boundary=boundary, x0=x0, x1=x1, y0=y0, y1=y1, n=n,
        )
        interior, linear_info = solve_linear_system(matrix, rhs, method=method, tol=tol, max_iter=max_iter)
        residual = matrix @ interior - rhs

    n_int = int(n)
    u = np.zeros((n_int + 1, n_int + 1), dtype=float)
    for j, yj in enumerate(y):
        for i, xi in enumerate(x):
            if i == 0 or i == n_int or j == 0 or j == n_int:
                u[j, i] = _eval_scalar2d(boundary, xi, yj, "boundary")

    interior_count = n_int - 1
    for j in range(1, n_int):
        for i in range(1, n_int):
            u[j, i] = interior[_interior_index(i, j, interior_count)]

    if not np.all(np.isfinite(u)):
        raise FloatingPointError("solution contains NaN or inf")

    info = {
        "method": linear_info["method"],
        "converged": bool(linear_info["converged"]),
        "iterations": int(linear_info["iterations"]),
        "correction_norm": float(linear_info["correction_norm"]),
        "n": int(n_int),
        "hx": float((x1 - x0) / n_int),
        "hy": float((y1 - y0) / n_int),
        "points": int((n_int + 1) * (n_int + 1)),
        "unknowns": int((n_int - 1) * (n_int - 1)),
        "residual_norm": float(np.linalg.norm(residual, ord=np.inf)) if hasattr(residual, "__len__") else float(residual),
        "message": linear_info["message"],
    }
    return x, y, u, info


solve_poisson_dense = solve_poisson_dirichlet
solve_linear_system_seidel = solve_linear_system_gauss_seidel
