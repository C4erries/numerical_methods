#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def read_xy_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    x_values: list[float] = []
    y_values: list[float] = []

    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        x_values.append(float(parts[0]))
        y_values.append(float(parts[1]))

    if not x_values:
        raise ValueError(f"No numeric x y rows in file: {path}")

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise FloatingPointError(f"NaN/inf detected in file: {path}")
    return x, y


def plot_xy_files(
    file_label_pairs: Sequence[tuple[Path, str]],
    output_path: Path,
    title: str = "Solutions comparison",
    show_window: bool = True,
) -> None:
    if not file_label_pairs:
        raise ValueError("No series provided for plotting")

    series: list[tuple[np.ndarray, np.ndarray, str]] = []
    for path, label in file_label_pairs:
        x, y = read_xy_file(path)
        series.append((x, y, label))

    exact_idx: int | None = None
    for i, (_, _, label) in enumerate(series):
        if "exact" in label.lower():
            exact_idx = i
            break

    if exact_idx is None:
        fig, ax_sol = plt.subplots(1, 1, figsize=(10, 5))
        axes = [ax_sol]
    else:
        fig, (ax_sol, ax_err) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes = [ax_sol, ax_err]

    for i, (x, y, label) in enumerate(series):
        if exact_idx is not None and i == exact_idx:
            ax_sol.plot(x, y, color="black", linewidth=2.2, label=label)
        else:
            ax_sol.plot(x, y, linewidth=1.7, label=label)

    ax_sol.set_title(title)
    ax_sol.set_ylabel("y")
    ax_sol.grid(True, alpha=0.35)
    ax_sol.legend(loc="best")

    if exact_idx is not None:
        x_exact, y_exact, _ = series[exact_idx]
        order = np.argsort(x_exact)
        x_exact_sorted = x_exact[order]
        y_exact_sorted = y_exact[order]

        for i, (x, y, label) in enumerate(series):
            if i == exact_idx:
                continue
            y_ref = np.interp(x, x_exact_sorted, y_exact_sorted)
            err = np.abs(y - y_ref)
            err = np.maximum(err, np.finfo(float).tiny)
            ax_err.plot(x, err, linewidth=1.7, label=f"|{label} - exact|")

        ax_err.set_yscale("log")
        ax_err.set_ylabel("abs error")
        ax_err.set_xlabel("x")
        ax_err.grid(True, alpha=0.35)
        ax_err.legend(loc="best")
    else:
        ax_sol.set_xlabel("x")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show_window:
        plt.show()
    plt.close(fig)


def _parse_series_arg(value: str) -> tuple[Path, str]:
    if "|" in value:
        path_part, label_part = value.split("|", 1)
        path = Path(path_part.strip())
        label = label_part.strip() or path.stem
        return path, label
    path = Path(value.strip())
    return path, path.stem


def _build_arg_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot multiple x y files in one figure.")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Series descriptor: path or path|label. Repeat for multiple curves.",
    )
    parser.add_argument("-o", "--output", type=Path, default=base / "comparison.png", help="Output PNG file")
    parser.add_argument("--title", type=str, default="Solutions comparison", help="Plot title")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive window, only save PNG.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    pairs = [_parse_series_arg(item) for item in args.series]
    plot_xy_files(pairs, args.output, title=args.title, show_window=not args.no_show)


if __name__ == "__main__":
    main()
