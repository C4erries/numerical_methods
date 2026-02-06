#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Ввод и разбор данных
# -----------------------------

def _read_text(path: Path) -> str:
    """
    Считать текст из файла, пробуя несколько кодировок:
    сначала UTF‑8 (включая UTF‑8 с BOM), затем cp1251.
    """
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # Если всё не удалось — ещё раз UTF‑8, чтобы получить понятное исключение.
    return path.read_text(encoding="utf-8")


def _parse_numbers(line: str) -> np.ndarray:
    """Разобрать строку вида 'x: 0 1 2' в массив float."""
    return np.array([float(tok) for tok in line.strip().split()[1:]], dtype=float)


def parse_input(text: str) -> Dict[str, Any]:
    """
    Разобрать текстовое описание задачи интерполяции.

    Обязательные поля: x:, y:, xmin:, xmax:
    Необязательные:    n_plot:, conditions:, dy0:, dyn:
    """
    # Убираем комментарии после '#', игнорируем пустые строки.
    clean_lines: list[str] = []
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
        raise ValueError(f"Отсутствуют обязательные поля: {', '.join(missing)}")

    kv["xmin"] = float(kv["xmin"])
    kv["xmax"] = float(kv["xmax"])
    if kv["xmin"] >= kv["xmax"]:
        raise ValueError("Должно выполняться неравенство xmin < xmax")

    kv["n_plot"] = int(kv.get("n_plot", 400))
    kv["conditions"] = str(kv.get("conditions", "natural")).strip().lower()
    if kv["conditions"] not in {"natural", "clamped"}:
        raise ValueError("Поле conditions должно быть 'natural' или 'clamped'")

    if kv["conditions"] == "clamped":
        if "dy0" not in kv or "dyn" not in kv:
            raise ValueError("Для условий типа clamped необходимо задать dy0 и dyn")
        kv["dy0"] = float(kv["dy0"])
        kv["dyn"] = float(kv["dyn"])

    x = kv["x"]
    y = kv["y"]
    if x.shape != y.shape:
        raise ValueError("Массивы x и y должны иметь одинаковую длину")
    if x.size < 2:
        raise ValueError("Должно быть не менее двух узлов (точек данных)")

    # Сортируем точки по возрастанию x и проверяем, что нет повторов.
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    if np.any(np.diff(x_sorted) == 0):
        raise ValueError("Значения x должны быть попарно различными (без повторов)")

    kv["x"] = x_sorted
    kv["y"] = y_sorted
    return kv


# -----------------------------
# Кубический сплайн
# -----------------------------

@dataclass
class CubicSpline1D:
    x: np.ndarray          # узлы
    y: np.ndarray          # значения в узлах
    M: np.ndarray          # вторые производные в узлах
    h: np.ndarray          # шаги h_i = x_{i+1} - x_i

    def evaluate(self, xq: np.ndarray) -> np.ndarray:
        """Вычислить значение сплайна в точках xq."""
        xq = np.asarray(xq, dtype=float)
        x, y, M, h = self.x, self.y, self.M, self.h
        n = x.size

        # Находим интервал [x_i, x_{i+1}] для каждой точки.
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
    a = np.zeros(n - 1, dtype=float)  # поддиагональ
    b = np.zeros(n, dtype=float)      # диагональ
    c = np.zeros(n - 1, dtype=float)  # наддиагональ
    return a, b, c


def _solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Решить трёхдиагональную СЛАУ методом прогонки."""
    n = b.size
    ac = a.copy()
    bc = b.copy()
    cc = c.copy()
    dc = d.copy()

    # прямой ход
    for i in range(1, n):
        w = ac[i - 1] / bc[i - 1]
        bc[i] = bc[i] - w * cc[i - 1]
        dc[i] = dc[i] - w * dc[i - 1]

    # обратный ход
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
    """Построить кубический сплайн по узлам (x, y)."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2:
        raise ValueError("Для построения сплайна требуется как минимум два узла")

    h = np.diff(x_arr)
    if np.any(h <= 0):
        raise ValueError("Значения x должны строго возрастать")

    rhs = np.zeros_like(x_arr)
    a, b, c = _build_tridiagonal(x_arr.size)

    if conditions == "natural":
        # естественные граничные условия: M0 = Mn-1 = 0
        b[0] = 1.0
        b[-1] = 1.0
        rhs[0] = 0.0
        rhs[-1] = 0.0

        for i in range(1, x_arr.size - 1):
            a[i - 1] = h[i - 1]
            b[i] = 2.0 * (h[i - 1] + h[i])
            c[i] = h[i]
            rhs[i] = 6.0 * (
                (y_arr[i + 1] - y_arr[i]) / h[i] - (y_arr[i] - y_arr[i - 1]) / h[i - 1]
            )
    elif conditions == "clamped":
        if dy0 is None or dyn is None:
            raise ValueError("Для условий типа clamped необходимо задать dy0 и dyn")

        # левая граница
        b[0] = 2.0 * h[0]
        c[0] = h[0]
        rhs[0] = 6.0 * ((y_arr[1] - y_arr[0]) / h[0] - dy0)

        # внутренние узлы
        for i in range(1, x_arr.size - 1):
            a[i - 1] = h[i - 1]
            b[i] = 2.0 * (h[i - 1] + h[i])
            c[i] = h[i]
            rhs[i] = 6.0 * (
                (y_arr[i + 1] - y_arr[i]) / h[i] - (y_arr[i] - y_arr[i - 1]) / h[i - 1]
            )

        # правая граница
        a[-1] = h[-1]
        b[-1] = 2.0 * h[-1]
        rhs[-1] = 6.0 * (dyn - (y_arr[-1] - y_arr[-2]) / h[-1])
    else:
        raise ValueError("Параметр conditions должен быть 'natural' или 'clamped'")

    moments = _solve_tridiagonal(a, b, c, rhs)
    return CubicSpline1D(x=x_arr, y=y_arr, M=moments, h=h)


# -----------------------------
# Выборка сплайна и вывод
# -----------------------------

def sample_spline(
    spline: CubicSpline1D, xmin: float, xmax: float, n_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Построить равномерную сетку по [xmin, xmax] и вычислить S(x)."""
    grid = np.linspace(xmin, xmax, int(n_points))
    values = spline.evaluate(grid)
    return grid, values


def write_output_file(
    path: Path,
    samples_x: Sequence[float],
    samples_y: Sequence[float],
) -> None:
    """
    Записать точки (x, S(x)) в файл.
    В файле только данные: по одной паре x, S(x) на строку, без заголовка.
    """
    lines: list[str] = []
    for x_val, y_val in zip(samples_x, samples_y):
        lines.append(f"{x_val:.10f}\t{y_val:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(
    output_path: Path,
    samples_x: Sequence[float],
    samples_y: Sequence[float],
    nodes_x: Sequence[float],
    nodes_y: Sequence[float],
) -> None:
    """
    Сохранить график сплайна и узлов в PNG.

    Картинка сохраняется рядом с выходным файлом, с тем же
    именем, но расширением `.png` (например, `out.txt` → `out.png`).
    """
    img_path = output_path.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(samples_x, samples_y, label="spline", color="C0")
    ax.scatter(nodes_x, nodes_y, label="узлы", color="C1", zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

    # гарантируем наличие директории для вывода
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_output_file(output_path, samples_x, samples_y)
    save_plot(output_path, samples_x, samples_y, cfg["x"], cfg["y"])


def build_arg_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Интерполяция табличной функции кубическим сплайном."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=base_dir / "in.txt",
        help="Путь к входному файлу (по умолчанию: in.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=base_dir / "out" / "out.txt",
        help="Путь к выходному файлу (по умолчанию: out/out.txt)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    process_files(args.input, args.output)


if __name__ == "__main__":
    main()

