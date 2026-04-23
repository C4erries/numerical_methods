#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from csv_io import write_grid_csv, write_num_exact_delta_grid_csv
from input_parser import read_input_config
from plotter import plot_grid_files
from solver import build_uniform_grid_2d, solve_poisson_dirichlet


PROBLEM_NAME = "Poisson Dirichlet: u = sin(pi*x) sin(pi*y)"


def rhs(x: float, y: float) -> float:
    return -2.0 * float(np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y))


def exact_value(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    return np.sin(np.pi * np.asarray(x, dtype=float)) * np.sin(np.pi * np.asarray(y, dtype=float))


def boundary(x: float, y: float) -> float:
    return float(exact_value(x, y))


def exact_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return exact_value(xx, yy)


def _deduplicate_n(n_values: Sequence[int]) -> list[int]:
    unique: list[int] = []
    for n in n_values:
        n_int = int(n)
        if n_int < 2:
            raise ValueError("n values must be at least 2")
        if n_int not in unique:
            unique.append(n_int)
    return unique


def _n_values_from_h(x0: float, x1: float, y0: float, y1: float, h_values: Sequence[float]) -> list[int]:
    x_len = x1 - x0
    y_len = y1 - y0
    result: list[int] = []
    for h in h_values:
        h = float(h)
        if h <= 0.0 or not np.isfinite(h):
            raise ValueError("h values must be positive and finite")
        nx_float = x_len / h
        ny_float = y_len / h
        nx = int(round(nx_float))
        ny = int(round(ny_float))
        if nx != ny:
            raise ValueError("h must give the same n in x and y for this version")
        if nx < 2 or abs(nx_float - nx) > 1e-10 * max(1.0, abs(nx_float)):
            raise ValueError(f"h={h:g} does not split x interval into an integer number of intervals")
        if abs(ny_float - ny) > 1e-10 * max(1.0, abs(ny_float)):
            raise ValueError(f"h={h:g} does not split y interval into an integer number of intervals")
        result.append(nx)
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

    x0 = float(cfg["x0"])
    x1 = float(cfg["x1"])
    y0 = float(cfg["y0"])
    y1 = float(cfg["y1"])

    if n_override is not None and len(n_override) > 0:
        n_values = [int(n) for n in n_override]
    elif h_override is not None and len(h_override) > 0:
        n_values = _n_values_from_h(x0, x1, y0, y1, h_override)
    else:
        n_values = list(cfg["n_values"])
        if len(n_values) == 1:
            n_values.append(n_values[0] * 2)
    n_values = _deduplicate_n(n_values)

    out_dir.mkdir(parents=True, exist_ok=True)

    exact_n = max(int(cfg["exact_n"]), max(n_values) * 4)
    x_exact, y_exact = build_uniform_grid_2d(x0, x1, y0, y1, exact_n)
    u_exact = exact_grid(x_exact, y_exact)
    exact_path = out_dir / "exact_grid.csv"
    write_grid_csv(exact_path, x_exact, y_exact, u_exact)

    solution_series: list[tuple[Path, str]] = [(exact_path, "exact")]
    error_series: list[tuple[Path, str]] = []
    summary_rows: list[list[str]] = []
    prev_mean_err: float | None = None
    prev_max_err: float | None = None

    for n in n_values:
        x_num, y_num, u_num, info = solve_poisson_dirichlet(
            f=rhs,
            boundary=boundary,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            n=n,
        )
        num_path = out_dir / f"num_n_{n}.csv"
        write_grid_csv(num_path, x_num, y_num, u_num)

        u_ref = exact_grid(x_num, y_num)
        detail_path = out_dir / f"num_vs_exact_n_{n}.csv"
        mean_err, max_err = write_num_exact_delta_grid_csv(detail_path, x_num, y_num, u_num, u_ref)

        solution_series.append((num_path, f"n={n}, hx={info['hx']:.4g}, hy={info['hy']:.4g}"))
        error_series.append((detail_path, f"n={n}"))
        summary_rows.append(
            [
                str(n),
                f"{info['hx']:.12g}",
                f"{info['hy']:.12g}",
                str(info["points"]),
                str(info["unknowns"]),
                str(info["method"]),
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
        writer.writerow(["equation", "u_xx + u_yy = f(x, y)"])
        writer.writerow(["rectangle", f"[{x0}, {x1}] x [{y0}, {y1}]"])
        writer.writerow(["boundary", "u = exact on rectangle boundary"])
        writer.writerow(["exact_n", str(exact_n)])
        writer.writerow([])
        writer.writerow(
            [
                "n",
                "hx",
                "hy",
                "points",
                "unknowns",
                "method",
                "mean_abs_error",
                "max_abs_error",
                "mean_error_ratio_prev",
                "max_error_ratio_prev",
                "residual_norm",
            ]
        )
        writer.writerows(summary_rows)

    plot_path = out_dir / "comparison.png"
    plot_grid_files(
        solution_series,
        plot_path,
        title=f"{PROBLEM_NAME}: exact and finite difference",
        show_window=show_window,
        value_label="u",
    )

    error_plot_path = out_dir / "errors.png"
    plot_grid_files(
        error_series,
        error_plot_path,
        title=f"{PROBLEM_NAME}: absolute error",
        show_window=show_window,
        value_column="delta",
        value_label="|u_num - u_exact|",
        cmap="magma",
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
        description="Run finite-difference Poisson-Dirichlet solver for several grids."
    )
    parser.add_argument("-i", "--input", type=Path, default=base / "in.txt", help="Input config file")
    parser.add_argument(
        "-d",
        "--out-dir",
        type=Path,
        default=base / "out",
        help="Directory for CSV data files and plots",
    )
    parser.add_argument(
        "--n",
        dest="n_override",
        type=int,
        nargs="+",
        default=None,
        help="Optional override list of n values, e.g. --n 8 16 32",
    )
    parser.add_argument(
        "--h",
        dest="h_override",
        type=float,
        nargs="+",
        default=None,
        help="Optional override list of h values, e.g. --h 0.125 0.0625",
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

    run_application(
        args.input,
        args.out_dir,
        n_override=args.n_override,
        h_override=args.h_override,
        show_window=not args.no_show,
    )


if __name__ == "__main__":
    main()
