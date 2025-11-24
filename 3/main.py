#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Задача 3. Поиск действительных корней многочлена методом Бернулли
с уточнением методом Ньютона и дефляцией.

Формат работы программы:
  - коэффициенты многочлена читаются из текстового файла (по умолчанию: in.txt);
  - действительные корни записываются в файл в папку out (по умолчанию: out/out.txt);
  - в консоль выводятся найденные корни и значения |p(r)|.

Форматы входного файла:
  1) JSON:
       {"coefficients": [1, -6, 11, -6]}
  2) Одна строка с коэффициентами:
       1 -6 11 -6
     или:
       1, -6, 11, -6
  3) Несколько строк, по одному числу в строке.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

Number = float


# ------------------------------
# Вспомогательные операции с многочленами
# ------------------------------

def strip_leading_zeros(coeffs: Sequence[Number]) -> List[Number]:
    """
    Удалить ведущие нулевые коэффициенты слева.
    Пример: [0, 0, 1, -3, 2] -> [1, -3, 2].
    """
    coeffs = list(coeffs)
    i = 0
    n = len(coeffs)
    while i < n - 1 and abs(coeffs[i]) == 0:
        i += 1
    return [Number(c) for c in coeffs[i:]]


def poly_eval(coeffs: Sequence[Number], x: Number) -> Number:
    """
    Вычислить значение многочлена p(x) в точке x.
    Коэффициенты заданы как [a0, a1, ..., ad] по убыванию степеней:
    p(x) = a0 x^d + a1 x^{d-1} + ... + ad.
    """
    res = 0.0
    for c in coeffs:
        res = res * x + c
    return res


def poly_eval_derivative(coeffs: Sequence[Number], x: Number) -> Number:
    """
    Вычислить значение производной p'(x) в точке x.
    Коэффициенты p заданы как [a0, a1, ..., ad] по убыванию степеней.
    """
    n = len(coeffs) - 1
    if n <= 0:
        return 0.0
    res = 0.0
    for i, c in enumerate(coeffs[:-1]):
        degree = n - i
        res = res * x + degree * c
    return res


def synthetic_division(coeffs: Sequence[Number], r: Number) -> Tuple[List[Number], Number]:
    """
    Синтетическое деление многочлена p(x) на (x - r).

    Возвращает:
      q_coeffs - коэффициенты частного q(x),
      remainder - остаток p(r).
    """
    coeffs = list(coeffs)
    n = len(coeffs) - 1
    if n <= 0:
        raise ValueError("Для деления на (x - r) степень многочлена должна быть >= 1.")

    b: List[Number] = [Number(coeffs[0])]
    for c in coeffs[1:]:
        b.append(b[-1] * r + c)
    remainder = b.pop()
    return b, Number(remainder)


def trailing_zeros_deflation(coeffs: Sequence[Number]) -> Tuple[List[Number], int]:
    """
    Удалить нулевые коэффициенты при младших степенях (справа).
    Если a_n = 0, то x = 0 - корень. Возвращает (новые_коэффициенты, кратность_нуля).
    """
    coeffs = list(coeffs)
    k = 0
    while len(coeffs) > 1 and abs(coeffs[-1]) == 0:
        coeffs.pop()
        k += 1
    return coeffs, k


def normalize_polynomial(coeffs: Sequence[Number]) -> List[Number]:
    """
    Нормировать многочлен так, чтобы старший коэффициент был равен 1.
    Бросает ValueError, если многочлен нулевой или ведущий коэффициент равен 0.
    """
    coeffs = strip_leading_zeros(coeffs)
    if not coeffs:
        raise ValueError("Многочлен пустой - нет коэффициентов.")
    lead = coeffs[0]
    if lead == 0:
        raise ValueError("Ведущий коэффициент многочлена равен 0 - нормировка невозможна.")
    if lead != 1.0:
        coeffs = [c / lead for c in coeffs]
    return coeffs


# ------------------------------
# Метод Бернулли + уточнение Ньютона
# ------------------------------

def bernoulli_dominant_root(
    coeffs: Sequence[Number],
    max_iter: int = 20000,
    tol: float = 1e-12,
    patience: int = 10,
) -> Tuple[Optional[Number], Dict[str, object]]:
    """
    Найти доминирующий по модулю корень многочлена методом Бернулли.

    Пусть p(x) = x^d + a_{d-1} x^{d-1} + ... + a_0 (после нормировки a0 = 1).
    Тогда последовательность x_n задаётся рекуррентно:
        x_{n} = - (a_{d-1} x_{n-1} + ... + a_0 x_{n-d}),
    а отношения q_n = x_n / x_{n-1} сходятся к одному из корней.
    """
    info: Dict[str, object] = {
        "converged": False,
        "iterations": 0,
        "reason": "",
        "last_ratio": None,
    }

    a = normalize_polynomial(coeffs)  # теперь a[0] == 1
    d = len(a) - 1
    if d <= 0:
        info["reason"] = "Степень многочлена не положительна."
        return None, info

    # окно значений x_{-d+1}, ..., x_0
    window: List[Number] = [0.0] * (d - 1) + [1.0]
    last_x = window[-1]
    last_ratio: Optional[Number] = None
    stable_hits = 0

    big = 1e150
    small = 1e-150

    for it in range(1, max_iter + 1):
        # рекуррентное вычисление x_next
        acc = 0.0
        for k in range(1, d + 1):
            acc += a[k] * window[-k]
        x_next = -acc

        # масштабирование, чтобы не улететь в переполнение/подпоток
        maxabs = max(1.0, max(abs(v) for v in window + [x_next]))
        if maxabs > big:
            scale = maxabs
            window = [v / scale for v in window]
            x_next /= scale
        elif maxabs < small:
            scale = 1.0 / small
            window = [v * scale for v in window]
            x_next *= scale

        # отслеживаем стабилизацию отношения q_n = x_n / x_{n-1}
        if last_x != 0.0:
            ratio = x_next / last_x
            if last_ratio is not None:
                rel = abs(ratio - last_ratio) / max(1.0, abs(ratio))
                if rel < tol:
                    stable_hits += 1
                else:
                    stable_hits = 0
            last_ratio = ratio
        else:
            ratio = None

        window = window[1:] + [x_next]
        last_x = x_next

        if ratio is not None and stable_hits >= patience:
            info["converged"] = True
            info["iterations"] = it
            info["last_ratio"] = last_ratio
            info["reason"] = "Отношения q_n стабилизировались по относительной погрешности."
            return float(last_ratio), info

    info["iterations"] = max_iter
    info["last_ratio"] = last_ratio
    info["reason"] = (
        "Не удалось добиться стабильности q_n за заданное число итераций "
        "(возможна плохая обусловленность или близкие по модулю корни)."
    )
    return None, info


def newton_refine(
    coeffs: Sequence[Number],
    x0: Number,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Tuple[Number, Dict[str, object]]:
    """
    Уточнить приближение корня методом Ньютона.
    Возвращает пару (x, info), где info содержит флаг сходимости и причину остановки.
    """
    x = float(x0)
    info: Dict[str, object] = {"converged": False, "iterations": 0, "reason": ""}

    for it in range(1, max_iter + 1):
        fx = poly_eval(coeffs, x)
        dfx = poly_eval_derivative(coeffs, x)
        if dfx == 0.0:
            info["iterations"] = it
            info["reason"] = "Производная занулилась - шаг Ньютона невозможен."
            return x, info

        step = fx / dfx
        x -= step
        if abs(step) <= tol * max(1.0, abs(x)):
            info["converged"] = True
            info["iterations"] = it
            info["reason"] = "Шаги Ньютона стали достаточно малы."
            return x, info

    info["iterations"] = max_iter
    info["reason"] = "Метод Ньютона не сошёлся за заданное число итераций."
    return x, info


def find_all_real_roots_bernoulli(
    coeffs: Sequence[Number],
    tol_ratio: float = 1e-12,
    tol_newton: float = 1e-12,
    max_iter_ratio: int = 20000,
) -> Tuple[List[Number], Dict[str, object]]:
    """
    Найти все действительные корни многочлена:
      1) методом Бернулли для доминирующего корня;
      2) уточнением методом Ньютона;
      3) последовательной дефляцией.

    Для степени 1 и 2 используются явные формулы (линейная и квадратная).
    Возвращает (список_корней, summary) с описанием шагов и остаточной степенью.
    """
    coeffs = strip_leading_zeros(coeffs)
    if len(coeffs) < 2:
        return [], {"reason": "Степень многочлена 0 - действительных корней нет.", "degree_left": 0, "steps": []}

    roots: List[Number] = []
    summary: Dict[str, object] = {"steps": []}

    # Сначала вытащим корни x = 0 (если есть)
    coeffs, zeros = trailing_zeros_deflation(coeffs)
    roots.extend([0.0] * zeros)
    if zeros:
        summary["steps"].append(
            f"Удалены {zeros} нулевых коэффициентов при младших степенях -> корень x=0 с кратностью {zeros}."
        )

    while len(coeffs) > 1:
        degree = len(coeffs) - 1

        if degree == 1:
            # линейный случай: a0 x + a1 = 0
            a0, a1 = coeffs
            if a0 == 0.0:
                break
            r = -a1 / a0
            roots.append(r)
            coeffs = [1.0]
            summary["steps"].append("Обработка линейного остатка - найден ещё один корень.")
            break

        if degree == 2:
            # квадратный случай: a0 x^2 + a1 x + a2
            a0, a1, a2 = coeffs
            if a0 == 0.0:
                # вырождается в линейный случай
                if a1 != 0.0:
                    r = -a2 / a1
                    roots.append(r)
                    summary["steps"].append("Квадратный случай выродился в линейный - найден корень.")
                break
            D = a1 * a1 - 4.0 * a0 * a2
            if D < 0.0:
                summary["steps"].append("Дискриминант < 0 - действительных корней у оставшегося квадратичного множителя нет.")
                break
            sqrtD = math.sqrt(D)
            r1 = (-a1 + sqrtD) / (2.0 * a0)
            r2 = (-a1 - sqrtD) / (2.0 * a0)
            roots.append(r1)
            roots.append(r2)
            coeffs = [1.0]
            summary["steps"].append("Квадратный остаток решён по формуле - найдено ещё два корня.")
            break

        # степень >= 3 - используем Бернулли + Ньютон
        r0, info_b = bernoulli_dominant_root(coeffs, max_iter=max_iter_ratio, tol=tol_ratio)
        summary["steps"].append(f"Бернулли: {info_b}")
        if r0 is None:
            summary["reason"] = "Метод Бернулли не дал устойчивой оценки корня - дальнейший поиск прекращён."
            break

        r, info_n = newton_refine(coeffs, r0, tol=tol_newton)
        summary["steps"].append(f"Ньютон: {info_n}")

        # проверяем невязку
        resid = abs(poly_eval(coeffs, r))
        if resid > math.sqrt(tol_newton):
            summary["steps"].append(
                f"Предупреждение: невязка |p(r)|={resid:.3e} довольно велика - корень может быть неточным."
            )

        # деление на (x - r)
        q, rem = synthetic_division(coeffs, r)
        if abs(rem) > 1e-6 * (1 + sum(abs(c) for c in coeffs)):
            summary["steps"].append(
                f"Предупреждение: остаток при делении p(x)/(x - r) равен {rem:.3e}, "
                "что указывает на неточный корень."
            )
        coeffs = q
        roots.append(r)

        # подчистим очень маленькие коэффициенты и снова вынесем возможные корни x=0
        coeffs = [0.0 if abs(c) < 1e-16 else c for c in coeffs]
        coeffs, z = trailing_zeros_deflation(coeffs)
        if z:
            roots.extend([0.0] * z)
            summary["steps"].append(
                f"После дефляции обнаружены ещё {z} корней x=0 (по нулевым коэффициентам)."
            )

    summary["degree_left"] = len(coeffs) - 1
    summary.setdefault("reason", "Алгоритм завершён: дальнейшая дефляция не требуется.")
    return roots, summary


# ------------------------------
# Ввод/вывод коэффициентов и корней
# ------------------------------

def read_coeffs_from_file(path: Path) -> List[Number]:
    """
    Прочитать коэффициенты многочлена из файла в одном из поддерживаемых форматов.
    """
    text = path.read_text(encoding="utf-8").strip()

    # Попытка интерпретировать как JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "coefficients" in obj and isinstance(obj["coefficients"], list):
            return [float(c) for c in obj["coefficients"]]
    except Exception:
        pass

    # Текстовые форматы
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Файл с коэффициентами пуст.")

    if len(lines) == 1:
        parts = [p for p in lines[0].replace(",", " ").split() if p]
        return [float(p) for p in parts]
    else:
        return [float(ln.replace(",", " ")) for ln in lines]


def save_roots(path: Path, roots: Sequence[Number]) -> None:
    """
    Сохранить действительные корни в текстовый файл (по одному числу в строке).
    """
    lines = [f"{float(r):.16g}" for r in roots]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ------------------------------
# CLI-обёртка
# ------------------------------

def process_file(
    input_path: Path,
    output_path: Path,
    *,
    tol_ratio: float = 1e-12,
    tol_newton: float = 1e-12,
    max_iter_ratio: int = 20000,
) -> Tuple[List[Number], Dict[str, object]]:
    """
    Прочитать многочлен из файла, найти действительные корни и сохранить их в выходной файл.
    Возвращает (список_корней, summary).
    """
    coeffs = read_coeffs_from_file(input_path)
    roots, summary = find_all_real_roots_bernoulli(
        coeffs,
        tol_ratio=tol_ratio,
        tol_newton=tol_newton,
        max_iter_ratio=max_iter_ratio,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_roots(output_path, roots)
    return roots, summary


def build_arg_parser() -> argparse.ArgumentParser:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Поиск действительных корней многочлена методом Бернулли с дефляцией."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=base_dir / "in.txt",
        help="Путь к входному файлу с коэффициентами (по умолчанию: in.txt).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=base_dir / "out" / "out.txt",
        help="Путь к выходному файлу с корнями (по умолчанию: out/out.txt).",
    )
    parser.add_argument(
        "--tol-ratio",
        type=float,
        default=1e-12,
        help="Точность стабилизации отношений Бернулли (по умолчанию 1e-12).",
    )
    parser.add_argument(
        "--tol-newton",
        type=float,
        default=1e-12,
        help="Точность метода Ньютона (по умолчанию 1e-12).",
    )
    parser.add_argument(
        "--max-iter-ratio",
        type=int,
        default=20000,
        help="Максимальное число итераций для метода Бернулли (по умолчанию 20000).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    output_path: Path = args.output

    roots, summary = process_file(
        input_path,
        output_path,
        tol_ratio=args.tol_ratio,
        tol_newton=args.tol_newton,
        max_iter_ratio=args.max_iter_ratio,
    )

    coeffs = read_coeffs_from_file(input_path)

    print("Найдённые действительные корни многочлена:")
    if not roots:
        print("  (действительных корней не найдено)")
    for i, r in enumerate(roots, 1):
        print(f"  r{i} = {r:.16g}   |p(r)| = {abs(poly_eval(coeffs, r)):.3e}")

    degree_left = summary.get("degree_left", 0)
    if degree_left and degree_left > 0:
        print(
            f"\nВнимание: после дефляции остался многочлен степени {degree_left} - "
            "возможно наличие дополнительных комплексных корней или численных проблем."
        )

    print("\nДиагностика шагов алгоритма:")
    for s in summary.get("steps", []):
        print(" -", s)
    if "reason" in summary:
        print("\nПричина остановки:", summary["reason"])

    print(f"\nКорни записаны в файл: {output_path}")


if __name__ == "__main__":
    main()

