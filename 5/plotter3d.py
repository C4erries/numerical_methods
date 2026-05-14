#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from plotter import read_grid_file


def _coarsen(x: np.ndarray, y: np.ndarray, z: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.size <= max_side and y.size <= max_side:
        return x, y, z
    step_x = max(1, int(np.ceil(x.size / max_side)))
    step_y = max(1, int(np.ceil(y.size / max_side)))
    return x[::step_x], y[::step_y], z[::step_y, ::step_x]


def build_interactive_surface(
    exact_path: Path,
    numeric_pairs: Sequence[tuple[int, Path, Path]],
    output_html: Path,
    title: str,
    exact_max_side: int = 200,
) -> Path:
    """Build an interactive Plotly HTML with a slider over (mode, n) states.

    numeric_pairs: list of (n, num_csv, num_vs_exact_csv).
    """
    import plotly.graph_objects as go

    x_e, y_e, z_e = read_grid_file(exact_path)
    x_e, y_e, z_e = _coarsen(x_e, y_e, z_e, exact_max_side)

    z_min = float(np.min(z_e))
    z_max = float(np.max(z_e))

    numeric_data: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    err_max_global = 0.0
    for n, num_path, delta_path in numeric_pairs:
        x_n, y_n, z_n = read_grid_file(num_path, value_column="u")
        x_d, y_d, z_d = read_grid_file(delta_path, value_column="delta")
        if x_n.shape != x_d.shape or y_n.shape != y_d.shape:
            raise ValueError(f"grid mismatch between {num_path.name} and {delta_path.name}")
        z_min = min(z_min, float(np.min(z_n)))
        z_max = max(z_max, float(np.max(z_n)))
        err_max_global = max(err_max_global, float(np.max(z_d)))
        numeric_data.append((n, x_n, y_n, z_n, z_d))

    frames: list = []
    step_labels: list[str] = []

    # Mode 1: exact only.
    frames.append(go.Frame(
        name="exact",
        data=[go.Surface(z=z_e, x=x_e, y=y_e, colorscale="Viridis",
                          cmin=z_min, cmax=z_max, name="exact",
                          colorbar=dict(title="u"))],
    ))
    step_labels.append("exact")

    for n, x_n, y_n, z_n, _z_d in numeric_data:
        frames.append(go.Frame(
            name=f"num n={n}",
            data=[go.Surface(z=z_n, x=x_n, y=y_n, colorscale="Viridis",
                              cmin=z_min, cmax=z_max, name=f"num n={n}",
                              colorbar=dict(title="u"))],
        ))
        step_labels.append(f"num, n={n}")

    for n, x_n, y_n, _z_n, z_d in numeric_data:
        frames.append(go.Frame(
            name=f"err n={n}",
            data=[go.Surface(z=z_d, x=x_n, y=y_n, colorscale="Magma",
                              cmin=0.0, cmax=err_max_global,
                              name=f"|u_num - u_exact| n={n}",
                              colorbar=dict(title="|err|"))],
        ))
        step_labels.append(f"|error|, n={n}")

    fig = go.Figure(data=list(frames[0].data), frames=frames)

    slider_steps = []
    for frame, label in zip(frames, step_labels):
        slider_steps.append(dict(
            method="animate",
            label=label,
            args=[[frame.name], dict(
                mode="immediate",
                frame=dict(duration=0, redraw=True),
                transition=dict(duration=0),
            )],
        ))

    fig.update_layout(
        title=title,
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="View: ", font=dict(size=14)),
            pad=dict(t=40),
            steps=slider_steps,
        )],
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="u",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=720,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True)
    return output_html


def _discover_pairs(out_dir: Path) -> tuple[Path, list[tuple[int, Path, Path]]]:
    exact_path = out_dir / "exact_grid.csv"
    if not exact_path.exists():
        raise FileNotFoundError(f"missing {exact_path}")

    pairs: list[tuple[int, Path, Path]] = []
    for num_path in sorted(out_dir.glob("num_n_*.csv")):
        stem = num_path.stem  # num_n_<N>
        try:
            n = int(stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        delta_path = out_dir / f"num_vs_exact_n_{n}.csv"
        if not delta_path.exists():
            continue
        pairs.append((n, num_path, delta_path))
    pairs.sort(key=lambda item: item[0])
    if not pairs:
        raise FileNotFoundError(f"no num_n_*.csv pairs in {out_dir}")
    return exact_path, pairs


def main(argv: Sequence[str] | None = None) -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build interactive 3D Plotly HTML from solver output CSVs.",
    )
    parser.add_argument("-d", "--out-dir", type=Path, default=base / "out",
                        help="Directory containing exact_grid.csv and num_n_*.csv")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output HTML path (defaults to <out-dir>/surface.html)")
    parser.add_argument("--title", type=str, default="Poisson Dirichlet: interactive 3D surface")
    args = parser.parse_args(argv)

    exact_path, pairs = _discover_pairs(args.out_dir)
    output = args.output or (args.out_dir / "surface.html")
    written = build_interactive_surface(exact_path, pairs, output, args.title)
    print(f"Interactive 3D plot: {written.resolve()}")


if __name__ == "__main__":
    main()
