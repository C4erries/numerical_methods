#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _sorted_component_columns(fields: list[str], prefix: str) -> list[str]:
    cols = [name for name in fields if re.fullmatch(rf"{re.escape(prefix)}\d+", name)]
    cols.sort(key=lambda name: int(name[len(prefix) :]))
    return cols


def read_xy_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        fields = [name.strip() for name in fieldnames]
        if "x" not in fields:
            raise ValueError(f"CSV {path} must contain column 'x'")

        if "y" in fields:
            y_columns = ["y"]
        elif "y_num" in fields:
            y_columns = ["y_num"]
        else:
            y_columns = _sorted_component_columns(fields, "y")
            if not y_columns:
                y_columns = _sorted_component_columns(fields, "y_num")
        if not y_columns:
            raise ValueError(f"CSV {path} must contain 'y'/'y_num' or component columns like y1,y2,...")

        x_values: list[float] = []
        y_rows: list[list[float]] = []
        for row in reader:
            x_values.append(float(row["x"]))
            y_rows.append([float(row[col]) for col in y_columns])

    if not x_values:
        raise ValueError(f"No numeric rows in file: {path}")

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    if y.ndim != 2:
        raise ValueError(f"Invalid y shape in {path}")
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

    sns.set_theme(style="whitegrid")
    series: list[tuple[np.ndarray, np.ndarray, str]] = []
    for path, label in file_label_pairs:
        x, y = read_xy_file(path)
        series.append((x, y, label))

    dims = {y.shape[1] for _, y, _ in series}
    if len(dims) != 1:
        raise ValueError("All series must have the same dimension")
    dim = next(iter(dims))

    exact_idx: int | None = None
    for i, (_, _, label) in enumerate(series):
        if "exact" in label.lower():
            exact_idx = i
            break

    fig, ax_sol = plt.subplots(1, 1, figsize=(10, 6))

    if dim == 1:
        for i, (x, y, label) in enumerate(series):
            if exact_idx is not None and i == exact_idx:
                sns.lineplot(x=x, y=y[:, 0], ax=ax_sol, color="black", linewidth=2.2, label=label)
            else:
                sns.lineplot(x=x, y=y[:, 0], ax=ax_sol, linewidth=1.7, label=label)
    else:
        for i, (x, y, label) in enumerate(series):
            is_exact = exact_idx is not None and i == exact_idx
            style = "-" if is_exact else "--"
            alpha = 1.0 if is_exact else 0.9
            lw = 2.2 if is_exact else 1.5
            for k in range(dim):
                sns.lineplot(
                    x=x,
                    y=y[:, k],
                    ax=ax_sol,
                    linestyle=style,
                    linewidth=lw,
                    alpha=alpha,
                    label=f"{label}: y{k + 1}",
                )

    ax_sol.set_title(title)
    ax_sol.set_ylabel("y")
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
    parser = argparse.ArgumentParser(description="Plot multiple CSV files (x,y) in one figure.")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Series descriptor: csv_path or csv_path|label. Repeat for multiple curves.",
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
