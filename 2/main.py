#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Parsing helpers
# -----------------------------

def _read_text(path: Path) -> str:
    """Read text from a file trying UTF-8 (with BOM support) first and cp1251 second."""
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # If everything fails, re-raise using UTF-8 for a clearer error message
    return path.read_text(encoding="utf-8")


def _parse_numbers(line: str) -> np.ndarray:
    """Parse a line like 'x: 0 1 2' into a numpy array of floats."""
    return np.array([float(tok) for tok in line.strip().split()[1:]], dtype=float)


def parse_input(text: str) -> Dict[str, Any]:
    """
    Parse the textual description of the problem.
    Required keys: x:, y:, xmin:, xmax:.
    Optional keys: n_plot:, conditions:, dy0:, dyn:.
    """
    clean_lines = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            clean_lines.append(line)

    kv: Dict[str, Any] = {}
    for line in clean_lines:
        if line.startswith("x:"):
            kv["x"] = _parse_numbers(line)
        elif line.startswith("y:"):
            kv["y"] = _parse_numbers(line)
        else:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            kv[key.strip().lower()] = val.strip()

    required = ["x", "y", "xmin", "xmax"]
    missing = [name for name in required if name not in kv]
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}")

    kv["xmin"] = float(kv["xmin"])
    kv["xmax"] = float(kv["xmax"])
    if kv["xmin"] >= kv["xmax"]:
        raise ValueError("xmin must be strictly less than xmax")

    kv["n_plot"] = int(kv.get("n_plot", 400))
    kv["conditions"] = str(kv.get("conditions", "natural")).strip().lower()
    if kv["conditions"] not in {"natural", "clamped"}:
        raise ValueError("conditions must be 'natural' or 'clamped'")

    if kv["conditions"] == "clamped":
        if "dy0" not in kv or "dyn" not in kv:
            raise ValueError("dy0 and dyn are required for clamped boundary conditions")
        kv["dy0"] = float(kv["dy0"])
        kv["dyn"] = float(kv["dyn"])

    x = kv["x"]
    y = kv["y"]
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    if x.size < 2:
        raise ValueError("At least two data points are required")

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    if np.any(np.diff(x_sorted) == 0):
        raise ValueError("x values must be unique")

    kv["x"] = x_sorted
    kv["y"] = y_sorted
    return kv


# -----------------------------
# Cubic spline implementation
# -----------------------------

@dataclass
class CubicSpline1D:
    x: np.ndarray
    y: np.ndarray
    M: np.ndarray
    h: np.ndarray

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the spline at arbitrary points."""
        xq = np.asarray(points, dtype=float)
        x, y, M, h = self.x, self.y, self.M, self.h
        n = x.size

        idx = np.searchsorted(x, xq, side="right") - 1
        idx = np.clip(idx, 0, n - 2)

        xi = x[idx]
        xi1 = x[idx + 1]
        hi = h[idx]
        t = (xq - xi) / hi
        a = 1.0 - t
        b = t

        Mi = M[idx]
        Mi1 = M[idx + 1]
        yi = y[idx]
        yi1 = y[idx + 1]

        term1 = Mi * ((xi1 - xq) ** 3) / (6.0 * hi)
        term2 = Mi1 * ((xq - xi) ** 3) / (6.0 * hi)
        term3 = (yi - Mi * hi * hi / 6.0) * a
        term4 = (yi1 - Mi1 * hi * hi / 6.0) * b
        return term1 + term2 + term3 + term4


def _build_tridiagonal(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(n - 1, dtype=float),
        np.zeros(n, dtype=float),
        np.zeros(n - 1, dtype=float),
    )


def _solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = b.size
    ac = a.copy()
    bc = b.copy()
    cc = c.copy()
    dc = d.copy()

    for i in range(1, n):
        w = ac[i - 1] / bc[i - 1]
        bc[i] = bc[i] - w * cc[i - 1]
        dc[i] = dc[i] - w * dc[i - 1]

    x = np.zeros(n, dtype=float)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x


def cubic_spline(
    x: Sequence[float],
    y: Sequence[float],
    *,
    conditions: Literal["natural", "clamped"] = "natural",
    dy0: Optional[float] = None,
    dyn: Optional[float] = None,
) -> CubicSpline1D:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2:
        raise ValueError("Need at least two nodes")

    h = np.diff(x_arr)
    if np.any(h <= 0):
        raise ValueError("x values must be strictly increasing")

    rhs = np.zeros_like(x_arr)
    a, b, c = _build_tridiagonal(x_arr.size)

    if conditions == "natural":
        b[0] = b[-1] = 1.0
        for i in range(1, x_arr.size - 1):
            a[i - 1] = h[i - 1]
            b[i] = 2.0 * (h[i - 1] + h[i])
            c[i] = h[i]
            rhs[i] = 6.0 * (
                (y_arr[i + 1] - y_arr[i]) / h[i] - (y_arr[i] - y_arr[i - 1]) / h[i - 1]
            )
    elif conditions == "clamped":
        if dy0 is None or dyn is None:
            raise ValueError("dy0 and dyn must be provided for clamped conditions")
        b[0] = 2.0 * h[0]
        c[0] = h[0]
        rhs[0] = 6.0 * ((y_arr[1] - y_arr[0]) / h[0] - dy0)

        for i in range(1, x_arr.size - 1):
            a[i - 1] = h[i - 1]
            b[i] = 2.0 * (h[i - 1] + h[i])
            c[i] = h[i]
            rhs[i] = 6.0 * (
                (y_arr[i + 1] - y_arr[i]) / h[i] - (y_arr[i] - y_arr[i - 1]) / h[i - 1]
            )

        a[-1] = h[-1]
        b[-1] = 2.0 * h[-1]
        rhs[-1] = 6.0 * (dyn - (y_arr[-1] - y_arr[-2]) / h[-1])
    else:
        raise ValueError("conditions must be natural or clamped")

    moments = _solve_tridiagonal(a, b, c, rhs)
    return CubicSpline1D(x=x_arr, y=y_arr, M=moments, h=h)


# -----------------------------
# File IO helpers
# -----------------------------

def sample_spline(
    spline: CubicSpline1D, xmin: float, xmax: float, n_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(xmin, xmax, int(n_points))
    values = spline.evaluate(grid)
    return grid, values


def write_output_file(
    path: Path,
    cfg: Dict[str, Any],
    samples_x: Sequence[float],
    samples_y: Sequence[float],
) -> None:
    lines = [
        "# Cubic spline interpolation result",
        f"# nodes: {cfg['x'].size}",
        f"# xmin: {cfg['xmin']}",
        f"# xmax: {cfg['xmax']}",
        f"# n_plot (samples): {cfg['n_plot']}",
        "# columns: x  spline(x)",
    ]
    for x_val, y_val in zip(samples_x, samples_y):
        lines.append(f"{x_val:.10f}\t{y_val:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# CLI
# -----------------------------

def process_files(input_path: Path, output_path: Path) -> None:
    text = _read_text(input_path)
    cfg = parse_input(text)
    spline = cubic_spline(
        cfg["x"],
        cfg["y"],
        conditions=cfg["conditions"],
        dy0=cfg.get("dy0"),
        dyn=cfg.get("dyn"),
    )
    samples_x, samples_y = sample_spline(spline, cfg["xmin"], cfg["xmax"], cfg["n_plot"])
    write_output_file(output_path, cfg, samples_x, samples_y)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cubic spline interpolation helper.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("in.txt"),
        help="Path to the input file (default: in.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("out.txt"),
        help="Where to store the sampled spline values (default: out.txt)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    process_files(args.input, args.output)


if __name__ == "__main__":
    main()
