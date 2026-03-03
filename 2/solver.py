#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence

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

        y_pred = yn + (h_base / 24.0) * (55.0 * fn - 59.0 * fn1 + 37.0 * fn2 - 9.0 * fn3)
        if not np.all(np.isfinite(y_pred)):
            raise FloatingPointError(f"Predictor produced NaN/inf at step {n + 1}")

        y_old = y_pred
        converged = False
        it_used = 0

        for m in range(max_iter):
            f_np1_old = _eval_f(f, tnp1, y_old, dim)
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


def build_uniform_grid(t0: float, t_end: float, h: float) -> np.ndarray:
    if not np.isfinite([t0, t_end, h]).all():
        raise ValueError("t0, t_end, h must be finite")
    if h <= 0.0:
        raise ValueError("h for uniform grid must be positive")

    interval = float(t_end - t0)
    if interval == 0.0:
        return np.array([t0], dtype=float)

    direction = 1.0 if interval > 0 else -1.0
    step = direction * h
    n_full = int(np.floor(abs(interval) / h + 1e-14))
    remainder = interval - n_full * step
    eps_t = 10.0 * np.finfo(float).eps * max(1.0, abs(t0), abs(t_end))
    if abs(remainder) <= eps_t:
        remainder = 0.0

    t_values = [float(t0)]
    for _ in range(n_full):
        t_values.append(t_values[-1] + step)
    if remainder != 0.0:
        t_values.append(float(t_end))
    else:
        t_values[-1] = float(t_end)
    return np.array(t_values, dtype=float)


def build_problem_from_code(
    t0: float, y0: ArrayLike
) -> tuple[Callable[[float, np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], str]:
    """
    Equation is configured here.
    Active variant: y' = y, y(t0) = y0.
    """
    y0v = _as_state_vector(y0)
    if y0v.size != 1:
        raise ValueError("Current hardcoded equation expects scalar y0.")

    y00 = float(y0v[0])

    def f_exp(_t: float, y: np.ndarray) -> np.ndarray:
        return y

    def exact_exp(t_arr: np.ndarray) -> np.ndarray:
        return (y00 * np.exp(t_arr - t0)).reshape(-1, 1)

    problem_name = "y' = y"
    return f_exp, exact_exp, problem_name


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
        writer.writerow(["x", "y"])
        for xi, yi in zip(x_arr, y_arr[:, component]):
            writer.writerow([f"{xi:.12f}", f"{yi:.12e}"])


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

    delta = np.abs(y_num_arr[:, component] - y_exact_arr[:, component])
    mean_err = float(np.mean(delta))
    max_err = float(np.max(delta))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y_num", "y_exact", "delta"])
        for xi, yn, ye, de in zip(x_arr, y_num_arr[:, component], y_exact_arr[:, component], delta):
            writer.writerow([f"{xi:.12f}", f"{yn:.12e}", f"{ye:.12e}", f"{de:.12e}"])

    return mean_err, max_err


def _read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8")


def _parse_vector(text: str) -> np.ndarray:
    clean = text.replace(",", " ")
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


def read_input_config(path: Path) -> dict:
    text = _read_text(path)
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)

    cfg: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        cfg[key.strip().lower()] = val.strip()

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
        "y0": _parse_vector(cfg["y0"]),
        "t_end": float(cfg["t_end"]),
        "h_values": h_values,
        "tol": float(cfg.get("tol", "1e-10")),
        "max_iter": int(cfg.get("max_iter", "20")),
        "exact_h": float(cfg["exact_h"]) if "exact_h" in cfg else None,
    }
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="ODE solver for Adams-Moulton 4-step, writes x,y CSV file.")
    parser.add_argument("-i", "--input", type=Path, default=base / "in.txt", help="Input config file")
    parser.add_argument("-o", "--output", type=Path, default=base / "out_xy.csv", help="Output x,y CSV file")
    parser.add_argument("--h", type=float, default=None, help="Step for numerical solution")
    parser.add_argument("--exact", action="store_true", help="Write exact solution on uniform grid")
    parser.add_argument("--grid-h", type=float, default=None, help="Grid step for exact solution")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    cfg = read_input_config(args.input)
    f, exact_fn, _ = build_problem_from_code(cfg["t0"], cfg["y0"])

    if args.exact:
        base_h = min(abs(h) for h in cfg["h_values"])
        grid_h = args.grid_h if args.grid_h is not None else (cfg["exact_h"] or base_h / 10.0)
        t = build_uniform_grid(cfg["t0"], cfg["t_end"], abs(grid_h))
        y = exact_fn(t)
        write_xy_csv(args.output, t, y)
        return

    h = args.h if args.h is not None else cfg["h_values"][0]
    t, y, _ = solve_adams_moulton4(
        f=f,
        t0=cfg["t0"],
        y0=cfg["y0"],
        t_end=cfg["t_end"],
        h=h,
        tol=cfg["tol"],
        max_iter=cfg["max_iter"],
    )
    write_xy_csv(args.output, t, y)


if __name__ == "__main__":
    main()
