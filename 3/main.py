#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Метод Бернулли для корней многочленов (реальные корни).
Алгоритм: Бернулли (доминирующий корень) -> уточнение Ньютоном -> дефляция -> повтор.
Ограничения: базовый Бернулли сходится к корню с наибольшим модулем, если он единственный по модулю.
"""

from __future__ import annotations
import json
import math
from typing import List, Tuple, Dict, Optional

Number = float

# ------------------------------
# Вспомогательные процедуры
# ------------------------------

def strip_leading_zeros(coeffs: List[Number]) -> List[Number]:
    """Удаляет ведущие нули (коэффициенты заданы от старшей степени к свободному члену)."""
    i = 0
    while i < len(coeffs) - 1 and abs(coeffs[i]) == 0:
        i += 1
    return coeffs[i:]


def normalize_polynomial(coeffs: List[Number]) -> List[Number]:
    """Нормирует многочлен так, чтобы старший коэффициент был 1."""
    coeffs = strip_leading_zeros(coeffs)
    if not coeffs:
        raise ValueError("Пустой список коэффициентов.")
    lead = coeffs[0]
    if lead == 0:
        raise ValueError("Старший коэффициент равен 0 -- недопустимый многочлен.")
    if lead != 1.0:
        coeffs = [c / lead for c in coeffs]
    return coeffs


def poly_eval(coeffs: List[Number], x: Number) -> Number:
    """Вычисляет p(x) по схеме Горнера. coeffs: [a0, a1, ..., ad]."""
    res = 0.0
    for c in coeffs:
        res = res * x + c
    return res


def poly_eval_derivative(coeffs: List[Number], x: Number) -> Number:
    """Вычисляет p'(x) по Горнеру (без явного построения коэффициентов p')."""
    n = len(coeffs) - 1
    if n <= 0:
        return 0.0
    res = 0.0
    for i, c in enumerate(coeffs[:-1]):
        degree = n - i
        res = res * x + degree * c
    return res


def synthetic_division(coeffs: List[Number], r: Number) -> Tuple[List[Number], Number]:
    """Деление многочлена на (x - r) по Горнеру. Возвращает (коэффициенты частного, остаток)."""
    n = len(coeffs) - 1
    if n <= 0:
        raise ValueError("Степень многочлена должна быть >= 1 для дефляции.")
    b = [coeffs[0]]
    for c in coeffs[1:]:
        b.append(b[-1] * r + c)
    remainder = b.pop()
    return b, remainder


def trailing_zeros_deflation(coeffs: List[Number]) -> Tuple[List[Number], int]:
    """Удаляет нулевые свободные члены: если a_n = 0, то x=0 -- корень. Возвращает (новые коэф., сколько удалили)."""
    k = 0
    while len(coeffs) > 1 and abs(coeffs[-1]) == 0:
        coeffs = coeffs[:-1]
        k += 1
    return coeffs, k

# ------------------------------
# Метод Бернулли + Ньютон
# ------------------------------

def bernoulli_dominant_root(coeffs: List[Number],
                            max_iter: int = 20000,
                            tol: float = 1e-12,
                            patience: int = 10) -> Tuple[Optional[Number], Dict[str, object]]:
    """
    Оценивает доминирующий по модулю корень методом Бернулли.
    Рекуррентность для нормированных коэффициентов a: x_n = - (a1 x_{n-1} + ... + a_d x_{n-d}), a0=1.
    Отношение q_n = x_n / x_{n-1} -> r при выполнении условий сходимости.
    """
    info = {"converged": False, "iterations": 0, "reason": "", "last_ratio": None}

    a = normalize_polynomial(coeffs)  # теперь a0 == 1
    d = len(a) - 1
    if d <= 0:
        info["reason"] = "Полином нулевой степени."
        return None, info

    window = [0.0] * (d - 1) + [1.0]  # x_{-d+1},...,x_0
    last_x = window[-1]
    last_ratio = None
    stable_hits = 0

    big = 1e150
    small = 1e-150

    for it in range(1, max_iter + 1):
        acc = 0.0
        for k in range(1, d + 1):
            acc += a[k] * window[-k]
        x_next = -acc

        # масштабирование
        maxabs = max(1.0, max(abs(v) for v in window + [x_next]))
        if maxabs > big:
            scale = maxabs
            window = [v / scale for v in window]
            x_next /= scale
        elif maxabs < small:
            scale = 1.0 / small
            window = [v * scale for v in window]
            x_next *= scale

        # отношение
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
            info["reason"] = "Критерий сходимости по отношению выполнен."
            return float(last_ratio), info

    info["iterations"] = max_iter
    info["last_ratio"] = last_ratio
    info["reason"] = "Нет уверенной сходимости (возможна комплексная доминирующая пара/кратность)."
    return None, info


def newton_refine(coeffs: List[Number], x0: Number,
                  tol: float = 1e-12, max_iter: int = 100) -> Tuple[Number, Dict[str, object]]:
    """Уточнение корня методом Ньютона (вещественная итерация)."""
    x = float(x0)
    info = {"converged": False, "iterations": 0, "reason": ""}
    for it in range(1, max_iter + 1):
        fx = poly_eval(coeffs, x)
        dfx = poly_eval_derivative(coeffs, x)
        if dfx == 0.0:
            info["iterations"] = it
            info["reason"] = "Нулевая производная -- вероятен кратный корень."
            return x, info
        step = fx / dfx
        x -= step
        if abs(step) <= tol * max(1.0, abs(x)):
            info["converged"] = True
            info["iterations"] = it
            info["reason"] = "Критерий по шагу выполнен."
            return x, info
    info["iterations"] = max_iter
    info["reason"] = "Лимит итераций Ньютона."
    return x, info


def find_all_real_roots_bernoulli(coeffs: List[Number],
                                  tol_ratio: float = 1e-12,
                                  tol_newton: float = 1e-12,
                                  max_iter_ratio: int = 20000) -> Tuple[List[Number], Dict[str, object]]:
    """
    Ищет все РЕАЛЬНЫЕ корни: Бернулли (доминирующий) -> Ньютон -> дефляция, пока возможно.
    Возвращает (список корней, сводка).
    """
    coeffs = strip_leading_zeros(coeffs)
    if len(coeffs) < 2:
        return [], {"reason": "Степень 0.", "degree_left": 0, "steps": []}

    roots: List[Number] = []
    summary: Dict[str, object] = {"steps": []}

    # x=0 корни
    coeffs, zeros = trailing_zeros_deflation(coeffs)
    roots.extend([0.0] * zeros)
    if zeros:
        summary["steps"].append(f"Нулевые свободные члены -> корень 0 с кратностью {zeros}.")

    while len(coeffs) > 1:
        if len(coeffs) == 2:
            a0, a1 = coeffs
            if a0 == 0.0:
                break
            r = -a1 / a0
            roots.append(r)
            coeffs = [1.0]
            summary["steps"].append("Линейный случай -- корень найден аналитически.")
            break

        r0, info_b = bernoulli_dominant_root(coeffs, max_iter=max_iter_ratio, tol=tol_ratio)
        summary["steps"].append(f"Бернулли: {info_b}")
        if r0 is None:
            summary["reason"] = "Бернулли не сошёлся (равные модули доминирующих корней/комплексная пара)."
            break

        r, info_n = newton_refine(coeffs, r0, tol=tol_newton)
        summary["steps"].append(f"Ньютон: {info_n}")

        resid = abs(poly_eval(coeffs, r))
        if resid > math.sqrt(tol_newton):
            summary["steps"].append(f"Предупреждение: |p(r)|={resid:.3e} -- возможна неточность.")

        q, rem = synthetic_division(coeffs, r)
        if abs(rem) > 1e-6 * (1 + sum(abs(c) for c in coeffs)):
            summary["steps"].append(f"Предупреждение: остаток после дефляции {rem:.3e}.")
        coeffs = q
        roots.append(r)

        # чистим почти-нули и снова выносим x=0, если возникли
        coeffs = [0.0 if abs(c) < 1e-16 else c for c in coeffs]
        coeffs, z = trailing_zeros_deflation(coeffs)
        if z:
            roots.extend([0.0] * z)
            summary["steps"].append(f"После дефляции: дополнительный нулевой корень кратности {z}.")

    summary["degree_left"] = len(coeffs) - 1
    summary.setdefault("reason", "Готово.")
    return roots, summary

# ------------------------------
# Ввод/вывод
# ------------------------------

def read_coeffs_from_file(path: str) -> List[Number]:
    """
    Поддерживаемые форматы:
    - JSON: {"coefficients":[...]}
    - Одна строка: числа через пробел/запятые
    - По одному числу на строку
    Порядок: от старшей степени к свободному члену.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "coefficients" in obj and isinstance(obj["coefficients"], list):
            return [float(c) for c in obj["coefficients"]]
    except Exception:
        pass

    # Простой текст
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1:
        parts = [p for p in lines[0].replace(",", " ").split() if p]
        return [float(p) for p in parts]
    else:
        return [float(ln.replace(",", " ")) for ln in lines]


def save_roots(path: str, roots: List[Number]) -> None:
    """Сохраняет корни (по одному на строку)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in roots:
            f.write(f"{r:.16g}\n")


def main():
    # --------- ПАРАМЕТРЫ ДЛЯ ТЕБЯ ---------
    INPUT_PATH  = "example_poly.txt"   # путь к файлу с коэффициентами
    TOL         = 1e-12                # точность критерия Бернулли (по отношению)
    NEWTON_TOL  = 1e-12                # точность Ньютона (по шагу)
    MAX_ITER    = 20000                # максимум итераций Бернулли
    OUT_PATH    = ""                   # "" чтобы не сохранять; или путь для записи корней
    # --------------------------------------

    coeffs = read_coeffs_from_file(INPUT_PATH)
    roots, summary = find_all_real_roots_bernoulli(coeffs,
                                                   tol_ratio=TOL,
                                                   tol_newton=NEWTON_TOL,
                                                   max_iter_ratio=MAX_ITER)

    print("Найденные реальные корни (приближения):")
    for i, r in enumerate(roots, 1):
        print(f"  r{i} = {r:.16g}   |p(r)| = {abs(poly_eval(coeffs, r)):.3e}")

    if summary.get("degree_left", 0) > 0:
        print(f"\nВнимание: осталось неразложенной степень: {summary['degree_left']}")
        print("Скорее всего, оставшиеся корни комплексные или с одинаковым доминирующим модулем.")

    print("\nСводка шагов:")
    for s in summary.get("steps", []):
        print(" -", s)
    if "reason" in summary:
        print("Итог:", summary["reason"])

    if OUT_PATH:
        save_roots(OUT_PATH, roots)
        print(f"\nКорни сохранены в файл: {OUT_PATH}")


if __name__ == "__main__":
    main()
