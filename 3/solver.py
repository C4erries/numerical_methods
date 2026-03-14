#!/usr/bin/env python3
from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np


ArrayLike = np.ndarray | Sequence[float] | float | int


def _as_state_vector(y: ArrayLike) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError("y must be scalar or 1D array")
    if arr.size == 0:
        raise ValueError("y must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("y contains NaN or inf")
    return arr


def _eval_f(f: Callable[[float, np.ndarray], ArrayLike], t: float, y: np.ndarray, dim: int) -> np.ndarray:
    fy = np.asarray(f(float(t), y.copy()), dtype=float)
    if fy.ndim == 0:
        fy = fy.reshape(1)
    elif fy.ndim != 1:
        raise ValueError("f(t, y) must return scalar or 1D array")
    if fy.size != dim:
        raise ValueError(f"f(t, y) returned size {fy.size}, expected {dim}")
    if not np.all(np.isfinite(fy)):
        raise FloatingPointError(f"f(t, y) is not finite at t={t}")
    return fy


def rk4_step(f: Callable[[float, np.ndarray], ArrayLike], t: float, y: ArrayLike, h: float) -> np.ndarray:
    yv = _as_state_vector(y)
    dim = yv.size

    k1 = _eval_f(f, t, yv, dim)
    k2 = _eval_f(f, t + 0.5 * h, yv + 0.5 * h * k1, dim)
    k3 = _eval_f(f, t + 0.5 * h, yv + 0.5 * h * k2, dim)
    k4 = _eval_f(f, t + h, yv + h * k3, dim)

    y_next = yv + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if not np.all(np.isfinite(y_next)):
        raise FloatingPointError(f"RK4 produced NaN/inf at t={t + h}")
    return y_next


def solve_adams_moulton4(
    f: Callable[[float, np.ndarray], ArrayLike],
    t0: float,
    y0: ArrayLike,
    t_end: float,
    h: float,
    tol: float = 1e-10,
    max_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not np.isfinite([t0, t_end, h, tol]).all():
        raise ValueError("t0, t_end, h, tol must be finite")
    if h == 0.0:
        raise ValueError("h must be non-zero")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    y0v = _as_state_vector(y0)
    dim = y0v.size
    interval = float(t_end - t0)
    if interval == 0.0:
        return np.array([t0], dtype=float), y0v.reshape(1, -1), {
            "converged": True,
            "non_converged_steps": [],
            "iterations_per_step": [],
            "used_short_last_step": False,
            "message": "Trivial interval: t_end == t0.",
        }

    if interval * h < 0.0:
        raise ValueError("h sign must point from t0 to t_end")

    direction = 1.0 if interval > 0 else -1.0
    h_abs = abs(h)
    h_base = direction * h_abs

    n_full = int(np.floor(abs(interval) / h_abs + 1e-14))
    remainder = interval - n_full * h_base
    eps_t = 10.0 * np.finfo(float).eps * max(1.0, abs(t0), abs(t_end))
    if abs(remainder) <= eps_t:
        remainder = 0.0

    t_values: list[float] = [float(t0)]
    y_values: list[np.ndarray] = [y0v.copy()]
    f_values: list[np.ndarray] = [_eval_f(f, t0, y0v, dim)]

    iterations_per_step: list[int] = []
    non_converged_steps: list[int] = []

    startup_steps = min(3, n_full)
    for _ in range(startup_steps):
        t_cur = t_values[-1]
        y_cur = y_values[-1]
        y_next = rk4_step(f, t_cur, y_cur, h_base)
        t_next = t_cur + h_base

        t_values.append(float(t_next))
        y_values.append(y_next)
        f_values.append(_eval_f(f, t_next, y_next, dim))

    for n in range(3, n_full):
        tn = t_values[n]
        yn = y_values[n]
        tnp1 = tn + h_base

        fn = f_values[n]
        fn1 = f_values[n - 1]
        fn2 = f_values[n - 2]
        fn3 = f_values[n - 3]

        # AB4 predictor:
        # y_{n+1}^{(0)} = y_n + h/24 * (55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3})
        y_pred = yn + (h_base / 24.0) * (55.0 * fn - 59.0 * fn1 + 37.0 * fn2 - 9.0 * fn3)
        if not np.all(np.isfinite(y_pred)):
            raise FloatingPointError(f"Predictor produced NaN/inf at step {n + 1}")

        y_old = y_pred
        converged = False
        it_used = 0

        for m in range(max_iter):
            f_np1_old = _eval_f(f, tnp1, y_old, dim)
            # AM4 corrector fixed-point iteration:
            # y^{m+1} = y_n + h/720 * (251 f(t_{n+1}, y^m) + 646 f_n - 264 f_{n-1} + 106 f_{n-2} - 19 f_{n-3})
            y_new = yn + (h_base / 720.0) * (
                251.0 * f_np1_old + 646.0 * fn - 264.0 * fn1 + 106.0 * fn2 - 19.0 * fn3
            )
            if not np.all(np.isfinite(y_new)):
                raise FloatingPointError(f"Corrector produced NaN/inf at step {n + 1}")

            it_used = m + 1
            if np.linalg.norm(y_new - y_old, ord=np.inf) < tol:
                converged = True
                y_old = y_new
                break
            y_old = y_new

        if not converged:
            non_converged_steps.append(n + 1)
            warnings.warn(
                f"AM4 corrector did not converge at step {n + 1} (t={tnp1:.6g})",
                RuntimeWarning,
            )

        y_next = y_old
        f_next = _eval_f(f, tnp1, y_next, dim)
        t_values.append(float(tnp1))
        y_values.append(y_next)
        f_values.append(f_next)
        iterations_per_step.append(it_used)

    used_short_last_step = False
    if remainder != 0.0:
        t_cur = t_values[-1]
        y_cur = y_values[-1]
        y_next = rk4_step(f, t_cur, y_cur, remainder)
        t_values.append(float(t_end))
        y_values.append(y_next)
        used_short_last_step = True

    t_arr = np.array(t_values, dtype=float)
    y_arr = np.vstack(y_values).astype(float)

    if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(y_arr)):
        raise FloatingPointError("Solution contains NaN/inf")

    info = {
        "converged": len(non_converged_steps) == 0,
        "non_converged_steps": non_converged_steps,
        "iterations_per_step": iterations_per_step,
        "used_short_last_step": used_short_last_step,
        "message": "ok" if len(non_converged_steps) == 0 else "Corrector did not converge on some steps.",
    }
    return t_arr, y_arr, info
