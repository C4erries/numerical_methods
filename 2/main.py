#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence

import matplotlib.pyplot as plt
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
    """
    One RK4 step for y' = f(t, y).
    """
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
    """
    Solve y' = f(t, y) with AB4 predictor + implicit AM4 corrector (fixed-point iterations).

    Returns:
      t : ndarray, shape (N,)
      y : ndarray, shape (N, dim)
      info : dict with convergence flags and diagnostics
    """
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

    # Clean tiny floating-point remainder.
    eps_t = 10.0 * np.finfo(float).eps * max(1.0, abs(t0), abs(t_end))
    if abs(remainder) <= eps_t:
        remainder = 0.0

    t_values: list[float] = [float(t0)]
    y_values: list[np.ndarray] = [y0v.copy()]
    f_values: list[np.ndarray] = [_eval_f(f, t0, y0v, dim)]

    iterations_per_step: list[int] = []
    non_converged_steps: list[int] = []

    # Start values y1..y3 by RK4 on full steps.
    startup_steps = min(3, n_full)
    for _ in range(startup_steps):
        t_cur = t_values[-1]
        y_cur = y_values[-1]
        y_next = rk4_step(f, t_cur, y_cur, h_base)
        t_next = t_cur + h_base

        t_values.append(float(t_next))
        y_values.append(y_next)
        f_values.append(_eval_f(f, t_next, y_next, dim))

    # Multistep phase for remaining full steps.
    # History layout at step n: f_values[n-3], f_values[n-2], f_values[n-1], f_values[n].
    for n in range(3, n_full):
        tn = t_values[n]
        yn = y_values[n]
        tnp1 = tn + h_base

        fn = f_values[n]
        fn1 = f_values[n - 1]
        fn2 = f_values[n - 2]
        fn3 = f_values[n - 3]

        # AB4 predictor:
        # y_{n+1}^{(0)} = y_n + h/24*(55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3})
        y_pred = yn + (h_base / 24.0) * (55.0 * fn - 59.0 * fn1 + 37.0 * fn2 - 9.0 * fn3)
        if not np.all(np.isfinite(y_pred)):
            raise FloatingPointError(f"Predictor produced NaN/inf at step {n + 1}")

        # Fixed-point AM4 corrector iterations:
        # y^{m+1} = y_n + h/720*(251 f(t_{n+1}, y^m) + 646 f_n - 264 f_{n-1} + 106 f_{n-2} - 19 f_{n-3})
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
        # Last shortened step: keep integration robust with RK4.
        t_cur = t_values[-1]
        y_cur = y_values[-1]
        y_next = rk4_step(f, t_cur, y_cur, remainder)
        t_values.append(float(t_end))
        y_values.append(y_next)
        f_values.append(_eval_f(f, t_end, y_next, dim))
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


def read_input_file(path: Path) -> dict:
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
        k, v = line.split(":", 1)
        cfg[k.strip().lower()] = v.strip()

    required = ("t0", "y0", "t_end", "h")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    result = {
        "t0": float(cfg["t0"]),
        "y0": _parse_vector(cfg["y0"]),
        "t_end": float(cfg["t_end"]),
        "h": float(cfg["h"]),
        "tol": float(cfg.get("tol", "1e-10")),
        "max_iter": int(cfg.get("max_iter", "20")),
    }
    return result


def build_problem_from_code(
    t0: float, y0: ArrayLike
) -> tuple[Callable[[float, np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], str]:
    """
    Equation is configured in code here (not in input file).
    Active problem below: y' = y, y(t0) = y0.
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


def write_output_file(path: Path, t: np.ndarray, y: np.ndarray, exact: np.ndarray, info: dict) -> None:
    err_inf = np.max(np.abs(y - exact), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Adams-Moulton 4-step (p=5) with AB4 predictor")
    lines.append(f"# converged: {info['converged']}")
    lines.append(f"# non_converged_steps: {info['non_converged_steps']}")
    lines.append(f"# used_short_last_step: {info['used_short_last_step']}")

    dim = y.shape[1]
    header = ["t"] + [f"y{i + 1}" for i in range(dim)] + [f"y_exact{i + 1}" for i in range(dim)] + ["err_inf"]
    lines.append("\t".join(header))

    for i in range(t.size):
        row = [f"{t[i]:.12f}"]
        row.extend(f"{v:.12e}" for v in y[i])
        row.extend(f"{v:.12e}" for v in exact[i])
        row.append(f"{err_inf[i]:.12e}")
        lines.append("\t".join(row))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_solution_and_error(path: Path, t: np.ndarray, y: np.ndarray, exact: np.ndarray, title: str) -> None:
    err_inf = np.max(np.abs(y - exact), axis=1)
    err_plot = np.maximum(err_inf, np.finfo(float).tiny)
    dim = y.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax0, ax1 = axes

    for j in range(dim):
        ax0.plot(t, y[:, j], label=f"y{j + 1} num")
        ax0.plot(t, exact[:, j], "--", label=f"y{j + 1} exact")
    ax0.set_ylabel("solution")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")
    ax0.set_title(title)

    ax1.plot(t, err_plot, color="crimson")
    ax1.set_ylabel("||error||_inf")
    ax1.set_xlabel("t")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_files(input_path: Path, output_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    cfg = read_input_file(input_path)
    f, exact_fn, problem_name = build_problem_from_code(cfg["t0"], cfg["y0"])

    t, y, info = solve_adams_moulton4(
        f=f,
        t0=cfg["t0"],
        y0=cfg["y0"],
        t_end=cfg["t_end"],
        h=cfg["h"],
        tol=cfg["tol"],
        max_iter=cfg["max_iter"],
    )

    exact = exact_fn(t)
    write_output_file(output_path, t, y, exact, info)
    plot_solution_and_error(
        output_path.with_suffix(".png"),
        t,
        y,
        exact,
        title=f"{problem_name} | Adams-Moulton 4-step",
    )
    return t, y, info


def demo() -> None:
    """
    Two demos:
      1) y' = y, y(0)=1, exact e^t
      2) harmonic oscillator x' = v, v' = -w^2 x
    Results are written to files in the script directory.
    """
    base = Path(__file__).resolve().parent

    # Demo 1: exponential growth.
    f1 = lambda _t, y: y
    t1, y1, info1 = solve_adams_moulton4(f1, t0=0.0, y0=1.0, t_end=3.0, h=0.1, tol=1e-12, max_iter=30)
    exact1 = np.exp(t1).reshape(-1, 1)
    write_output_file(base / "demo_exp_out.txt", t1, y1, exact1, info1)
    plot_solution_and_error(base / "demo_exp_plot.png", t1, y1, exact1, "Demo: y' = y")

    # Demo 2: harmonic oscillator.
    omega = 2.0

    def f2(_t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], -(omega * omega) * y[0]], dtype=float)

    t2, y2, info2 = solve_adams_moulton4(
        f2, t0=0.0, y0=np.array([1.0, 0.0]), t_end=10.0, h=0.05, tol=1e-12, max_iter=40
    )
    exact2 = np.column_stack((np.cos(omega * t2), -omega * np.sin(omega * t2)))
    write_output_file(base / "demo_osc_out.txt", t2, y2, exact2, info2)
    plot_solution_and_error(base / "demo_osc_plot.png", t2, y2, exact2, "Demo: harmonic oscillator")

    summary = [
        "Demo finished.",
        f"exp: converged={info1['converged']}, max_err={np.max(np.abs(y1 - exact1)):.3e}",
        f"osc: converged={info2['converged']}, max_err={np.max(np.abs(y2 - exact2)):.3e}",
        "Files:",
        "  demo_exp_out.txt, demo_exp_plot.png",
        "  demo_osc_out.txt, demo_osc_plot.png",
    ]
    (base / "demo_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Adams-Moulton 4-step ODE solver (equation is set in code)."
    )
    parser.add_argument("-i", "--input", type=Path, default=base / "in.txt", help="Input config file")
    parser.add_argument("-o", "--output", type=Path, default=base / "out.txt", help="Output table file")
    parser.add_argument("--demo", action="store_true", help="Run built-in demos and save plots/tables")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.demo:
        demo()
        return

    process_files(args.input, args.output)


if __name__ == "__main__":
    main()
