#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_xy_file(path: Path, y_column: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        fields = [name.strip() for name in fieldnames]
        if "x" not in fields:
            raise ValueError(f"CSV {path} must contain column 'x'")

        selected = y_column
        if selected is None:
            for cand in ("y", "y_num"):
                if cand in fields:
                    selected = cand
                    break
        if selected is None or selected not in fields:
            raise ValueError(f"CSV {path} must contain requested y column")

        x_values: list[float] = []
        y_values: list[float] = []
        for row in reader:
            x_values.append(float(row["x"]))
            y_values.append(float(row[selected]))

    if not x_values:
        raise ValueError(f"No numeric rows in file: {path}")

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
    y_column: str | None = None,
    y_label: str = "y",
) -> None:
    if not file_label_pairs:
        raise ValueError("No series provided for plotting")

    sns.set_theme(style="whitegrid")
    series: list[tuple[np.ndarray, np.ndarray, str]] = []
    for path, label in file_label_pairs:
        x, y = read_xy_file(path, y_column=y_column)
        series.append((x, y, label))

    exact_idx: int | None = None
    for i, (_, _, label) in enumerate(series):
        if "exact" in label.lower():
            exact_idx = i
            break

    fig, ax_sol = plt.subplots(1, 1, figsize=(10, 6))

    for i, (x, y, label) in enumerate(series):
        if exact_idx is not None and i == exact_idx:
            sns.lineplot(x=x, y=y, ax=ax_sol, color="black", linewidth=2.2, label=label)
        else:
            sns.lineplot(x=x, y=y, ax=ax_sol, linewidth=1.7, label=label)

    ax_sol.set_title(title)
    ax_sol.set_ylabel(y_label)
    ax_sol.set_xlabel("x")
    ax_sol.grid(True, alpha=0.35)
    ax_sol.legend(loc="best")

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
    parser = argparse.ArgumentParser(description="Plot multiple CSV files in one figure.")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Series descriptor: csv_path or csv_path|label. Repeat for multiple curves.",
    )
    parser.add_argument("-o", "--output", type=Path, default=base / "comparison.png", help="Output PNG file")
    parser.add_argument("--title", type=str, default="Solutions comparison", help="Plot title")
    parser.add_argument("--y-column", type=str, default=None, help="CSV y column to plot")
    parser.add_argument("--y-label", type=str, default="y", help="Y axis label")
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
    plot_xy_files(
        pairs,
        args.output,
        title=args.title,
        show_window=not args.no_show,
        y_column=args.y_column,
        y_label=args.y_label,
    )


if __name__ == "__main__":
    main()
