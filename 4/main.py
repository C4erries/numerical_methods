#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from csv_io import write_num_exact_delta_csv, write_xy_csv
from input_parser import read_input_config
from plotter import plot_xy_files
from solver import build_uniform_grid, solve_finite_difference_bvp



PROBLEM_NAME = "hard"

def exact_scalar(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return np.exp(-3.0 * x_arr) * np.sin(10.0 * x_arr) + 0.2 * np.cos(5.0 * x_arr) + x_arr * x_arr

def exact_derivative(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return (
        np.exp(-3.0 * x_arr) * (10.0 * np.cos(10.0 * x_arr) - 3.0 * np.sin(10.0 * x_arr))
        - np.sin(5.0 * x_arr)
        + 2.0 * x_arr
    )

def exact_second_derivative(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return np.exp(-3.0 * x_arr) * (-60.0 * np.cos(10.0 * x_arr) - 91.0 * np.sin(10.0 * x_arr)) - 5.0 * np.cos(5.0 * x_arr) + 2.0

def p(x: float) -> float:
    return float(np.sin(2.0 * x) + 0.5 * x)

def q(x: float) -> float:
    return 1.0 + x * x

def rhs(x: float) -> float:
    return float(
        exact_second_derivative(x)
        + p(x) * exact_derivative(x)
        + q(x) * exact_scalar(x)
    )

def exact(x: np.ndarray, a: float, b: float, y_left: float, dy_right: float) -> np.ndarray:
    _ = a, b, y_left, dy_right
    return exact_scalar(x).reshape(-1, 1)

##
###
####
def _deduplicate_n(n_values: Sequence[int]) -> list[int]:
    unique: list[int] = []
    for n in n_values:
        n_int = int(n)
        if n_int <= 0:
            raise ValueError("n values must be positive")
        if n_int not in unique:
            unique.append(n_int)
    return unique


def _n_values_from_h(a: float, b: float, h_values: Sequence[float]) -> list[int]:
    interval = b - a
    result: list[int] = []
    for h in h_values:
        h = float(h)
        if h <= 0.0 or not np.isfinite(h):
            raise ValueError("h values must be positive and finite")
        n_float = interval / h
        n = int(round(n_float))
        if n <= 0 or abs(n_float - n) > 1e-10 * max(1.0, abs(n_float)):
            raise ValueError(f"h={h:g} does not split [{a}, {b}] into an integer number of intervals")
        result.append(n)
    return result


def _format_error_ratio(prev_error: float | None, cur_error: float) -> str:
    if prev_error is None or cur_error == 0.0:
        return ""
    return f"{prev_error / cur_error:.12g}"


def run_application(
    input_path: Path,
    out_dir: Path,
    n_override: Optional[Sequence[int]] = None,
    h_override: Optional[Sequence[float]] = None,
    show_window: bool = True,
) -> None:
    cfg = read_input_config(input_path)

    a = float(cfg["a"])
    b = float(cfg["b"])
    y_left = float(cfg["y_left"])
    dy_right = float(cfg["dy_right"])
    boundary_order = int(cfg["boundary_order"])

    if n_override is not None and len(n_override) > 0:
        n_values = [int(n) for n in n_override]
    elif h_override is not None and len(h_override) > 0:
        n_values = _n_values_from_h(a, b, h_override)
    else:
        n_values = list(cfg["n_values"])
        if len(n_values) == 1:
            n_values.append(n_values[0] * 2)
    n_values = _deduplicate_n(n_values)

    out_dir.mkdir(parents=True, exist_ok=True)

    exact_n = max(int(cfg["exact_n"]), max(n_values) * 10)
    x_exact = build_uniform_grid(a, b, exact_n)
    y_exact = exact(x_exact, a, b, y_left, dy_right)
    exact_path = out_dir / "exact_xy.csv"
    write_xy_csv(exact_path, x_exact, y_exact)

    solution_series: list[tuple[Path, str]] = [(exact_path, "exact")]
    error_series: list[tuple[Path, str]] = []
    summary_rows: list[list[str]] = []
    prev_mean_err: float | None = None
    prev_max_err: float | None = None

    for n in n_values:
        x_num, y_num, info = solve_finite_difference_bvp(
            p=p,
            q=q,
            f=rhs,
            a=a,
            b=b,
            n=n,
            y_left=y_left,
            dy_right=dy_right,
            boundary_order=boundary_order,
        )
        h = info["h"]
        num_path = out_dir / f"num_n_{n}.csv"
        write_xy_csv(num_path, x_num, y_num)

        y_ref = exact(x_num, a, b, y_left, dy_right)
        detail_path = out_dir / f"num_vs_exact_n_{n}.csv"
        mean_err, max_err = write_num_exact_delta_csv(detail_path, x_num, y_num, y_ref)

        solution_series.append((num_path, f"n={n}, h={h:g}"))
        error_series.append((detail_path, f"n={n}, h={h:g}"))
        summary_rows.append(
            [
                str(n),
                f"{h:.12g}",
                str(x_num.size),
                str(boundary_order),
                f"{mean_err:.12e}",
                f"{max_err:.12e}",
                _format_error_ratio(prev_mean_err, mean_err),
                _format_error_ratio(prev_max_err, max_err),
                f"{info['residual_norm']:.12e}",
            ]
        )
        prev_mean_err = mean_err
        prev_max_err = max_err
    
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["problem", PROBLEM_NAME])
        writer.writerow(["interval", f"[{a}, {b}]"])
        writer.writerow(["y_left", f"{y_left:.12g}"])
        writer.writerow(["dy_right", f"{dy_right:.12g}"])
        writer.writerow(["boundary_order", str(boundary_order)])
        writer.writerow(["exact_n", str(exact_n)])
        writer.writerow([])
        writer.writerow(
            [
                "n",
                "h",
                "points",
                "boundary_order",
                "mean_abs_error",
                "max_abs_error",
                "mean_error_ratio_prev",
                "max_error_ratio_prev",
                "residual_norm",
            ]
        )
        writer.writerows(summary_rows)

    plot_path = out_dir / "comparison.png"
    plot_xy_files(
        solution_series,
        plot_path,
        title=f"{PROBLEM_NAME}: exact and finite difference",
        show_window=show_window,
    )

    error_plot_path = out_dir / "errors.png"
    plot_xy_files(
        error_series,
        error_plot_path,
        title=f"{PROBLEM_NAME}: absolute error",
        show_window=show_window,
        y_column="delta",
        y_label="|y_num - y_exact|",
    )

    

    print(f"Done. Output directory: {out_dir.resolve()}")
    print(f"Exact data: {exact_path.name}")
    for n in n_values:
        print(f"Numeric data for n={n}: num_n_{n}.csv")
        print(f"Num-vs-exact for n={n}: num_vs_exact_n_{n}.csv")
    print(f"Combined plot: {plot_path.name}")
    print(f"Error plot: {error_plot_path.name}")
    print(f"Summary: {summary_path.name}")


def _build_arg_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run finite-difference BVP solver for several grids and plot exact + numerical curves."
    )
    parser.add_argument("-i", "--input", type=Path, default=base / "in.txt", help="Input config file")
    parser.add_argument(
        "-d",
        "--out-dir",
        type=Path,
        default=base / "out",
        help="Directory for CSV data files and combined plot",
    )
    parser.add_argument(
        "--n",
        dest="n_override",
        type=int,
        nargs="+",
        default=None,
        help="Optional override list of n values, e.g. --n 10 20 40",
    )
    parser.add_argument(
        "--h",
        dest="h_override",
        type=float,
        nargs="+",
        default=None,
        help="Optional override list of h values, e.g. --h 0.1 0.05 0.025",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=False,
        help="Do not open interactive window, only save PNG.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.n_override is not None and args.h_override is not None:
        raise ValueError("Use either --n or --h, not both")

    show_window = not args.no_show
    run_application(
        args.input,
        args.out_dir,
        n_override=args.n_override,
        h_override=args.h_override,
        show_window=show_window,
    )


if __name__ == "__main__":
    main()
