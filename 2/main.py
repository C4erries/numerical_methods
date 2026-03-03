#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from plotter import plot_xy_files
from solver import (
    build_problem_from_code,
    build_uniform_grid,
    read_input_config,
    solve_adams_moulton4,
    write_num_exact_delta_csv,
    write_xy_csv,
)


def _format_h_for_name(h: float) -> str:
    txt = f"{h:.12g}"
    txt = txt.replace("-", "m").replace(".", "p")
    return txt


def _deduplicate_h(h_values: Sequence[float]) -> list[float]:
    unique: list[float] = []
    for h in h_values:
        if not any(abs(h - prev) <= 1e-15 * max(1.0, abs(h), abs(prev)) for prev in unique):
            unique.append(float(h))
    return unique


def run_application(
    input_path: Path,
    out_dir: Path,
    h_override: Optional[Sequence[float]] = None,
    show_window: bool = True,
) -> None:
    cfg = read_input_config(input_path)

    if h_override is not None and len(h_override) > 0:
        h_values = [float(h) for h in h_override]
    else:
        h_values = list(cfg["h_values"])
        if len(h_values) == 1:
            h_values.append(h_values[0] / 2.0)
    h_values = _deduplicate_h(h_values)

    t0 = float(cfg["t0"])
    y0 = cfg["y0"]
    t_end = float(cfg["t_end"])
    tol = float(cfg["tol"])
    max_iter = int(cfg["max_iter"])

    f, exact_fn, problem_name = build_problem_from_code(t0, y0)

    out_dir.mkdir(parents=True, exist_ok=True)
    min_h_abs = min(abs(h) for h in h_values)
    exact_h = cfg["exact_h"] if cfg["exact_h"] is not None else min_h_abs / 10.0
    exact_h = abs(float(exact_h))
    if exact_h == 0.0:
        raise ValueError("exact_h must be non-zero")

    t_exact = build_uniform_grid(t0, t_end, exact_h)
    y_exact = exact_fn(t_exact)
    exact_path = out_dir / "exact_xy.csv"
    write_xy_csv(exact_path, t_exact, y_exact)

    series_for_plot: list[tuple[Path, str]] = [(exact_path, "exact")]
    summary_rows: list[list[str]] = []

    for h in h_values:
        t_num, y_num, info = solve_adams_moulton4(
            f=f,
            t0=t0,
            y0=y0,
            t_end=t_end,
            h=h,
            tol=tol,
            max_iter=max_iter,
        )
        num_path = out_dir / f"num_h_{_format_h_for_name(h)}.csv"
        write_xy_csv(num_path, t_num, y_num)

        y_ref = exact_fn(t_num)
        detail_path = out_dir / f"num_vs_exact_h_{_format_h_for_name(h)}.csv"
        mean_err, max_err = write_num_exact_delta_csv(detail_path, t_num, y_num, y_ref)

        series_for_plot.append((num_path, f"h={h:g}"))
        summary_rows.append(
            [
                f"{h:.12g}",
                str(t_num.size),
                str(info["converged"]),
                f"{mean_err:.12e}",
                f"{max_err:.12e}",
                " ".join(str(v) for v in info["non_converged_steps"]),
            ]
        )

    plot_path = out_dir / "comparison.png"
    plot_xy_files(
        series_for_plot,
        plot_path,
        title=f"{problem_name}: exact and Adams-Moulton",
        show_window=show_window,
    )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", problem_name])
        writer.writerow(["interval", f"[{t0}, {t_end}]"])
        writer.writerow(["y0", " ".join(f"{v:.12g}" for v in np.asarray(y0, dtype=float).reshape(-1))])
        writer.writerow(["tol", f"{tol:.12e}"])
        writer.writerow(["max_iter", str(max_iter)])
        writer.writerow(["exact_h", f"{exact_h:.12g}"])
        writer.writerow([])
        writer.writerow(["h", "points", "converged", "mean_abs_error", "max_abs_error", "non_converged_steps"])
        writer.writerows(summary_rows)

    print(f"Done. Output directory: {out_dir.resolve()}")
    print(f"Exact data: {exact_path.name}")
    for h in h_values:
        print(f"Numeric data for h={h:g}: num_h_{_format_h_for_name(h)}.csv")
        print(f"Num-vs-exact for h={h:g}: num_vs_exact_h_{_format_h_for_name(h)}.csv")
    print(f"Combined plot: {plot_path.name}")
    print(f"Summary: {summary_path.name}")


def _build_arg_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run solver for several h values and plot exact + all numerical curves."
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
        help="Do not open interactive window, only save PNG.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    run_application(
        args.input,
        args.out_dir,
        h_override=args.h_override,
        show_window=not args.no_show,
    )


if __name__ == "__main__":
    main()
