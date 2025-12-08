#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Задача 3. Поиск действительных корней многочлена методом Бернулли
с дефляцией (без уточнения методом Ньютона).

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
    Функция оставлена для возможного расширения (например, уточнение корней методом Ньютона).
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
      q_coeffs -- коэффициенты частного q(x),
      remainder -- остаток p(r).
    """
    coeffs = list(coeffs)
    n = len(coeffs) - 1
    if n <= 0:
        raise ValueError("Для деления на (x - r) степень многочлена должна быть не меньше 1.")

    b: List[Number] = [Number(coeffs[0])]
    for c in coeffs[1:]:
        b.append(b[-1] * r + c)
    remainder = b.pop()
    return b, Number(remainder)


def trailing_zeros_deflation(coeffs: Sequence[Number]) -> Tuple[List[Number], int]:
    """
    Удалить нулевые коэффициенты при младших степенях (справа).
    Если a_n = 0, то x = 0 является корнем.
    Возвращает (новые_коэффициенты, кратность_нуля).
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
    Функция может использоваться перед применением других численных методов.
    """
    coeffs = strip_leading_zeros(coeffs)
    if not coeffs:
        raise ValueError("Многочлен пустой -- нет коэффициентов.")
    lead = coeffs[0]
    if lead == 0:
        raise ValueError("Ведущий коэффициент многочлена равен 0 -- нормировка невозможна.")
    if lead != 1.0:
        coeffs = [c / lead for c in coeffs]
    return coeffs


# ------------------------------
# Вспомогательные функции для степенных подстановок
# ------------------------------

def compress_power_polynomial(coeffs: Sequence[Number]) -> Tuple[List[Number], int]:
    """
    Проверить, являются ли степени всех ненулевых членов кратными одному и тому же
    целому числу g, и при положительном ответе построить вспомогательный многочлен
    по переменной y = x^g.

    Возвращает (новые_коэффициенты, g). При g == 1 подстановка не выполняется.
    """
    coeffs = strip_leading_zeros(coeffs)
    degree = len(coeffs) - 1
    if degree <= 0:
        return coeffs, 1

    exponents = [degree - idx for idx, c in enumerate(coeffs) if c != 0]
    if len(exponents) <= 1:
        return coeffs, 1

    g = 0
    for exp in exponents:
        g = math.gcd(g, int(abs(exp)))
    if g <= 1:
        return coeffs, 1

    new_degree = degree // g
    new_coeffs = [0.0] * (new_degree + 1)
    for idx, c in enumerate(coeffs):
        if c == 0:
            continue
        exp = degree - idx
        target_deg = exp // g
        target_idx = new_degree - target_deg
        new_coeffs[target_idx] = c
    return new_coeffs, g


def expand_power_roots(
    roots: Sequence[Number],
    power: int,
    steps: Optional[List[str]] = None,
    tol: float = 1e-12,
) -> List[Number]:
    """
    Преобразовать корни вспомогательного многочлена по переменной y обратно
    к корням по переменной x после подстановки y = x^power.
    Возвращаются только действительные корни.
    """
    if power <= 1:
        return [float(r) for r in roots]

    res: List[Number] = []
    zero_eps = max(1e-14, tol)

    if power % 2 == 1:
        for y in roots:
            if abs(y) < zero_eps:
                res.extend([0.0] * power)
                continue
            x = math.copysign(abs(y) ** (1.0 / power), y)
            res.append(float(x))
        if steps is not None:
            steps.append(f"Корни вспомогательного многочлена преобразованы с учётом подстановки y = x^{power}.")
        return res

    skipped = 0
    for y in roots:
        if abs(y) < zero_eps:
            res.extend([0.0] * power)
            continue
        if y < 0.0:
            skipped += 1
            continue
        base = abs(y) ** (1.0 / power)
        res.append(float(base))
        res.append(float(-base))

    if steps is not None:
        steps.append(
            f"Каждый вспомогательный корень y >= 0 дал два действительных корня ±|y|^(1/{power}) после обратной подстановки."
        )
        if skipped:
            steps.append(
                f"{skipped} вспомогательных корней с y < 0 не дали действительных решений уравнения x^{power} = y."
            )
    return res


# ------------------------------
# Метод Бернулли (без Ньютона)
# ------------------------------

def bernoulli_dominant_root(
    coeffs: Sequence[Number],
    max_iter: int = 20000,
    eps: float = 1e-12,
    multiple_roots: bool = False,
) -> Tuple[Optional[Number], Dict[str, object]]:
    """
    Найти доминирующий по модулю корень многочлена методом Бернулли.

    Схема:
      - строится рекуррентная последовательность u_n;
      - на каждом шаге оценивается корень как root = u_n / u_{n-1};
      - контролируется малость |p(root)| и |root - prev_root|.

    Если multiple_roots=True, используются более строгие критерии сходимости.
    """
    info: Dict[str, object] = {
        "converged": False,
        "iterations": 0,
        "reason": "",
        "root": None,
        "residual": None,
    }

    coeffs = strip_leading_zeros(coeffs)
    n = len(coeffs) - 1
    if n <= 0:
        info["reason"] = "Степень многочлена должна быть положительной."
        return None, info

    a0 = coeffs[0]
    if a0 == 0.0:
        info["reason"] = "Ведущий коэффициент равен 0 -- метод Бернулли неприменим."
        return None, info

    # Вектор u длины n: u[0..n-1], начальные значения равны 0, последний элемент = 1.
    u: List[Number] = [0.0] * n
    u[-1] = 1.0

    prev_root: Optional[Number] = None
    root: Number = 0.0

    for it in range(1, max_iter + 1):
        s = 0.0
        for j in range(n):
            s += (coeffs[j + 1] * u[n - 1 - j]) / a0

        u = u[1:] + [-s]

        # Оценка корня как отношение двух последних членов последовательности u.
        if u[-2] == 0.0:
            # На текущем шаге отношение вычислить нельзя.
            continue
        root = u[-1] / u[-2]

        # Периодическое масштабирование (каждые 10 итераций) для уменьшения переполнений.
        if it % 10 == 0:
            scale = u[-1]
            if scale != 0.0:
                u = [val / scale for val in u]

        f_root = poly_eval(coeffs, root)
        thresh = eps * eps if multiple_roots else eps

        # Критерий сходимости:
        # |p(root)| < thresh и |root - prev_root| < thresh
        if prev_root is not None and abs(f_root) < thresh and abs(root - prev_root) < thresh:
            info["converged"] = True
            info["iterations"] = it
            info["root"] = float(root)
            info["residual"] = float(f_root)
            info["reason"] = "Условия по |p(r)| и |Δr| выполнены."
            return float(root), info

        prev_root = root

    # Если сюда дошли -- сходимость по основному критерию не достигнута.
    f_root = poly_eval(coeffs, root)
    info["iterations"] = max_iter
    info["root"] = float(root)
    info["residual"] = float(f_root)

    # При |p(root)| >= eps считаем, что требуемая точность не достигнута.
    if abs(f_root) >= eps:
        info["reason"] = (
            "Не удалось достичь заданной точности: |p(r)| >= eps; "
            "возможно, отсутствует действительный корень или выполнено недостаточное число итераций."
        )
        return None, info
    else:
        info["reason"] = "Достигнут лимит итераций, но |p(r)| < eps; корень принят."
        info["converged"] = True
        return float(root), info


# ------------------------------
# Поиск всех действительных корней (Бернулли + дефляция)
# ------------------------------

def find_all_real_roots_bernoulli(
    coeffs: Sequence[Number],
    tol_ratio: float = 1e-12,
    max_iter_ratio: int = 20000,
    multiple_roots: bool = False,
) -> Tuple[List[Number], Dict[str, object]]:
    """
    Найти все действительные корни многочлена, используя метод Бернулли
    с последовательной дефляцией многочлена.
    """
    coeffs = strip_leading_zeros(coeffs)
    if len(coeffs) < 2:
        return [], {
            "reason": "Степень многочлена меньше 1; действительные корни не извлекаются.",
            "degree_left": 0,
            "steps": [],
        }

    actual_roots: List[Number] = []
    aux_roots: List[Number] = []
    summary: Dict[str, object] = {"steps": []}

    coeffs, zeros = trailing_zeros_deflation(coeffs)
    actual_roots.extend([0.0] * zeros)
    if zeros:
        summary["steps"].append(
            f"Удалены {zeros} нулевых коэффициентов при младших степенях; корень x = 0 добавлен с соответствующей кратностью."
        )

    coeffs, power_factor = compress_power_polynomial(coeffs)
    if power_factor > 1 and len(coeffs) > 1:
        summary["steps"].append(
            f"Обнаружено, что степени всех ненулевых членов кратны {power_factor}; используется вспомогательный многочлен при подстановке y = x^{power_factor}."
        )

    while len(coeffs) > 1:
        degree = len(coeffs) - 1

        if degree == 1:
            a0, a1 = coeffs
            if a0 == 0.0:
                break
            r = -a1 / a0
            aux_roots.append(r)
            coeffs = [1.0]
            summary["steps"].append("Оставшийся линейный множитель решён аналитически.")
            break

        if degree == 2:
            a0, a1, a2 = coeffs
            if a0 == 0.0:
                if a1 != 0.0:
                    r = -a2 / a1
                    aux_roots.append(r)
                    summary["steps"].append(
                        "Квадратичный множитель выродился в линейный; найден соответствующий корень."
                    )
                break
            D = a1 * a1 - 4.0 * a0 * a2
            eps_D = 1e-14 * (1.0 + abs(a1) * abs(a1) + abs(a0) * abs(a2))
            if D < -eps_D:
                summary["steps"].append(
                    "Дискриминант отрицателен; действительных корней на этом шаге нет."
                )
                break
            if D < 0.0:
                D = 0.0
            sqrtD = math.sqrt(D)
            r1 = (-a1 + sqrtD) / (2.0 * a0)
            r2 = (-a1 - sqrtD) / (2.0 * a0)
            aux_roots.append(r1)
            aux_roots.append(r2)
            coeffs = [1.0]
            summary["steps"].append(
                "Квадратичный множитель решён по формуле; оба корня добавлены."
            )
            break

        r, info_b = bernoulli_dominant_root(
            coeffs,
            max_iter=max_iter_ratio,
            eps=tol_ratio,
            multiple_roots=multiple_roots,
        )
        summary["steps"].append(f"Этап метода Бернулли: {info_b}")
        if r is None:
            summary["reason"] = "Итерационный процесс Бернулли не достиг заданной точности."
            break

        resid = abs(poly_eval(coeffs, r))
        if resid > tol_ratio:
            summary["steps"].append(
                f"Предупреждение: |p(r)| = {resid:.3e} превышает заданную точность tol_ratio = {tol_ratio:.1e}."
            )

        q, rem = synthetic_division(coeffs, r)
        if abs(rem) > 1e-6 * (1 + sum(abs(c) for c in coeffs)):
            summary["steps"].append(
                f"Предупреждение: остаток синтетического деления {rem:.3e} по величине значителен."
            )
        coeffs = q
        aux_roots.append(r)

        coeffs = [0.0 if abs(c) < 1e-16 else c for c in coeffs]
        coeffs, z = trailing_zeros_deflation(coeffs)
        if z:
            aux_roots.extend([0.0] * z)
            summary["steps"].append(
                f"После дефляции появились {z} дополнительных нулевых коэффициентов; добавлены корни y = 0 с этой кратностью."
            )

    summary["degree_left"] = power_factor * (len(coeffs) - 1)
    summary.setdefault(
        "reason",
        "Алгоритм завершён; дальнейшая дефляция не требуется."
    )
    expanded_roots = expand_power_roots(aux_roots, power_factor, summary["steps"], tol_ratio)
    all_roots = actual_roots + expanded_roots
    return all_roots, summary


def read_coeffs_from_file(path: Path) -> List[Number]:
    """
    Прочитать коэффициенты многочлена из файла в одном из поддерживаемых форматов.
    """
    text = path.read_text(encoding="utf-8").strip()

    # Попытка интерпретировать как JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "coefficients" in obj and isinstance(obj["coefficients"], list):
            return [float(c) for c in obj["coefficients"]]
    except Exception:
        pass

    # Текстовые форматы.
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
    max_iter_ratio: int = 20000,
    multiple_roots: bool = False,
) -> Tuple[List[Number], Dict[str, object]]:
    """
    Прочитать многочлен из файла, найти действительные корни
    и сохранить их в выходной файл.
    Возвращает (список_корней, summary).
    """
    coeffs = read_coeffs_from_file(input_path)
    roots, summary = find_all_real_roots_bernoulli(
        coeffs,
        tol_ratio=tol_ratio,
        max_iter_ratio=max_iter_ratio,
        multiple_roots=multiple_roots,
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
        default=1e-8,
        help="Запрошенная точность по методу Бернулли (eps, по умолчанию 1e-8).",
    )
    parser.add_argument(
        "--max-iter-ratio",
        type=int,
        default=20000,
        help="Максимальное число итераций для метода Бернулли (по умолчанию 20000).",
    )
    parser.add_argument(
        "--multiple-roots",
        action="store_true",
        help="Режим поиска совпадающих корней (используются более строгие критерии сходимости).",
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
        max_iter_ratio=args.max_iter_ratio,
        multiple_roots=args.multiple_roots,
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
            f"\nВнимание: после дефляции остался многочлен степени {degree_left} -- "
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
