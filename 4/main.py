import math


def F(x):
    """
    Пример:
        f1(x, y) = x^2 + y^2 - 1 = 0
        f2(x, y) = x - y = 0

    Тогда:
        f1 = x[0]**2 + x[1]**2 - 1
        f2 = x[0] - x[1]
    """
    # ---------- ПРИМЕР, УДАЛИ И ПРОПИШИ СВОЁ ----------
    f1 = x[0]**2 + x[1]**2 - 1.0
    f2 = x[0] - x[1]
    return [f1, f2]
    # -------------------------------------------------
    # Для своей системы верни список [f1, f2, ..., fn]


def J(x):
    """
    Якоби F'(x).

    Для примера выше:
        f1 = x^2 + y^2 - 1
          df1/dx = 2x, df1/dy = 2y
        f2 = x - y
          df2/dx = 1,  df2/dy = -1

        J = [[2x, 2y],
             [ 1, -1]]
    """
    return [
        [2.0 * x[0], 2.0 * x[1]],
        [1.0,        -1.0]
    ]

def norm(v):
    """Евклидова норма вектора."""
    return math.sqrt(sum(vi * vi for vi in v))


def gauss_seidel(A, b, y0, eps_inner, max_inner):
    """
    Метод Зейделя для решения A y = b.
    Возвращает (y, число_итераций).
    """
    n = len(A)
    y = y0[:]

    for k in range(max_inner):
        y_old = y[:]
        for i in range(n):
            s1 = sum(A[i][j] * y[j] for j in range(i))           # j < i -- уже обновлённые
            s2 = sum(A[i][j] * y_old[j] for j in range(i + 1, n))  # j > i -- старые
            y[i] = (b[i] - s1 - s2) / A[i][i]

        diff = [y[i] - y_old[i] for i in range(n)]
        if norm(diff) < eps_inner:
            return y, k + 1

    return y, max_inner


def newton_gauss_seidel(x0, eps_outer, eps_inner, max_outer, max_inner):
    """
    Внешний метод Ньютона:
        J(x^k) * delta^k = -F(x^k)
        x^{k+1} = x^k + delta^k

    Внутри линейную систему решаем методом Зейделя.
    """
    x = x0[:]
    n = len(x)

    for k in range(max_outer):
        Fx = F(x)
        res = norm(Fx)
        if res < eps_outer:
            # Условие остановки по невязке
            return x, k, res

        Jx = J(x)
        b = [-v for v in Fx]          # правая часть для Jx * delta = -F
        delta0 = [0.0] * n            # начальное приближение для delta

        # ВНУТРЕННИЕ ИТЕРАЦИИ ЗЕЙДЕЛЯ
        delta, inner_used = gauss_seidel(Jx, b, delta0, eps_inner, max_inner)

        # ОБНОВЛЕНИЕ НЬЮТОНА
        x = [x[i] + delta[i] for i in range(n)]

    # Если вышли по ограничению по числу итераций Ньютона
    Fx = F(x)
    return x, max_outer, norm(Fx)


def solve_from_file(in_filename='in.txt', out_filename='out.txt'):
    # ----- ЧТЕНИЕ ВХОДНЫХ ДАННЫХ -----
    with open(in_filename, 'r', encoding='utf-8') as f:
        n = int(f.readline().strip())
        eps_outer, eps_inner = map(float, f.readline().split())
        max_outer, max_inner = map(int, f.readline().split())
        x0 = list(map(float, f.readline().split()))

    if len(x0) != n:
        raise ValueError("Число компонент начального приближения не совпадает с n")

    # ----- РЕШЕНИЕ -----
    x, outer_used, res_norm = newton_gauss_seidel(
        x0, eps_outer, eps_inner, max_outer, max_inner
    )

    # ----- ЗАПИСЬ РЕЗУЛЬТАТА -----
    with open(out_filename, 'w', encoding='utf-8') as f:
        f.write('Решение системы:\n')
        for i, xi in enumerate(x, start=1):
            f.write(f'x{i} = {xi:.10f}\n')
        f.write(f'\nЧисло внешних итераций (Ньютон): {outer_used}\n')
        f.write(f'Норма невязки ||F(x)|| = {res_norm:.6e}\n')


if __name__ == "__main__":
    solve_from_file("in.txt", "out.txt")
