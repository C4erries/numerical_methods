#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_grid_file(path: Path, value_column: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        fields = [name.strip() for name in fieldnames]
        if "x" not in fields or "y" not in fields:
            raise ValueError(f"CSV {path} must contain columns 'x' and 'y'")

        selected = value_column
        if selected is None:
            for cand in ("u", "u_num", "u_exact", "delta"):
                if cand in fields:
                    selected = cand
                    break
        if selected is None or selected not in fields:
            raise ValueError(f"CSV {path} must contain requested value column")

        rows: list[tuple[float, float, float]] = []
        for row in reader:
            rows.append((float(row["x"]), float(row["y"]), float(row[selected])))

    if not rows:
        raise ValueError(f"No numeric rows in file: {path}")

    x = np.array(sorted({r[0] for r in rows}), dtype=float)
    y = np.array(sorted({r[1] for r in rows}), dtype=float)
    z = np.full((y.size, x.size), np.nan, dtype=float)

    x_pos = {val: i for i, val in enumerate(x)}
    y_pos = {val: j for j, val in enumerate(y)}
    for xi, yi, zi in rows:
        z[y_pos[yi], x_pos[xi]] = zi

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(z)):
        raise FloatingPointError(f"NaN/inf detected in file: {path}")
    return x, y, z


def plot_grid_files(
    file_label_pairs: Sequence[tuple[Path, str]],
    output_path: Path,
    title: str = "Grid comparison",
    show_window: bool = True,
    value_column: str | None = None,
    value_label: str = "u",
    cmap: str = "viridis",
) -> None:
    if not file_label_pairs:
        raise ValueError("No grids provided for plotting")

    sns.set_theme(style="white")
    series: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    for path, label in file_label_pairs:
        x, y, z = read_grid_file(path, value_column=value_column)
        series.append((x, y, z, label))

    count = len(series)
    ncols = 2 if count > 1 else 1
    nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.8 * nrows), squeeze=False)
    flat_axes = axes.reshape(-1)

    for ax, (x, y, z, label) in zip(flat_axes, series):
        image = ax.imshow(
            z,
            origin="lower",
            extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.85, label=value_label)

    for ax in flat_axes[count:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description="Plot multiple 2D grid CSV files.")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Series descriptor: csv_path or csv_path|label. Repeat for multiple grids.",
    )
    parser.add_argument("-o", "--output", type=Path, default=base / "comparison.png", help="Output PNG file")
    parser.add_argument("--title", type=str, default="Grid comparison", help="Plot title")
    parser.add_argument("--value-column", type=str, default=None, help="CSV value column to plot")
    parser.add_argument("--value-label", type=str, default="u", help="Colorbar label")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib color map")
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
    plot_grid_files(
        pairs,
        args.output,
        title=args.title,
        show_window=not args.no_show,
        value_column=args.value_column,
        value_label=args.value_label,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
