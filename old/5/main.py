import math


def f(x: float) -> float:
    """
    Функция f(x) под интегралом.
    """
    return x**2 + 1.0

def intergate(h: float, a: float, x:float) -> float:
    s = 0.5 * (f(a) + f(x))
    for k in range(1, int((x-a)/h)):
        t = a + k * h
        s += f(t)
    return s * h

def integrate_trap(a: float, x: float, eps: float, m: int, *, max_refine: int = 30) -> float:
    """
    Численное вычисление интеграла ∫_a^x f(t) dt
    по формуле трапеций с m начальными разбиениями.

    Корректно работает и если x < a (учитывается знак).
    """
    if x == a:
        return 0.0

    # Если x < a, интегрируем от x до a и меняем знак
    if x < a:
        return -integrate_trap(x, a, eps, m)

    # Эффективное уточнение трапеций: при каждом делении шага пополам
    # переиспользуем уже посчитанный интеграл, добавляя только новые точки.
    n = max(1, int(m))
    h = (x - a) / n

    s = 0.5 * (f(a) + f(x))
    for i in range(1, n):
        s += f(a + i * h)
    T = s * h

    for _ in range(max_refine):
        # Новые точки - середины старых отрезков
        h *= 0.5
        mid_sum = 0.0
        for i in range(1, n + 1):
            mid_sum += f(a + (2 * i - 1) * h)

        T2 = 0.5 * T + h * mid_sum
        if abs(T2 - T) <= (4 - 1) * eps:
            return T2

        T = T2
        n *= 2

    return T


def bisection_solve(a: float,
                    b_val: float,
                    left: float,
                    right: float,
                    n_trap: int,
                    eps: float = 1e-6,
                    max_iter: int = 200,
                    *,
                    eps_int: float | None = None,
                    max_refine_int: int = 10):
    """
    Решение уравнения Φ(x) = ∫_a^x f(y) dy - b_val = 0 методом дихотомии
    на отрезке [left, right].

    Возвращает кортеж (x, F(x), Φ(x), k), где k - число итераций.
    """
    l, r = (left, right) if left <= right else (right, left)

    # Для дихотомии достаточно грубее считать интеграл, иначе будет слишком медленно:
    if eps_int is None:
        eps_int = max(1e-8, eps * 100.0)
    n_trap_bis = max(10, min(int(n_trap), 500))

    F_l = integrate_trap(a, l, eps_int, n_trap_bis, max_refine=max_refine_int)
    phi_l = F_l - b_val
    if abs(phi_l) < eps:
        return l, F_l, phi_l, 0

    F_r = integrate_trap(a, r, eps_int, n_trap_bis, max_refine=max_refine_int)
    phi_r = F_r - b_val
    if abs(phi_r) < eps:
        return r, F_r, phi_r, 0

    if phi_l * phi_r > 0:
        raise ValueError(
            f"На отрезке [{l}, {r}] нет смены знака Φ(x): "
            f"Φ(left)={phi_l:.6e}, Φ(right)={phi_r:.6e}"
        )

    for k in range(1, max_iter + 1):
        mid = 0.5 * (l + r)
        F_mid = integrate_trap(a, mid, eps_int, n_trap_bis, max_refine=max_refine_int)
        phi_mid = F_mid - b_val

        if abs(phi_mid) < eps or 0.5 * abs(r - l) < eps:
            return mid, F_mid, phi_mid, k

        if phi_l * phi_mid <= 0:
            r = mid
            phi_r = phi_mid
        else:
            l = mid
            phi_l = phi_mid

    return mid, F_mid, phi_mid, max_iter


def newton_solve(a: float,
                 b_val: float,
                 x0: float,
                 n_trap: int,
                 eps: float = 1e-6,
                 max_iter: int = 50):
    """
    Решение уравнения ∫_a^x f(y) dy = b_val методом Ньютона.

    Φ(x) = ∫_a^x f(y) dy - b_val
    Φ'(x) = f(x)

    Возвращает кортеж (x, F(x), Φ(x), k),
    где k -- число сделанных итераций.
    """
    x = x0
    k = 0
    for k in range(1, max_iter + 1):
        F = integrate_trap(a, x, eps, n_trap)  # ≈ ∫_a^x f
        phi = F - b_val
        if abs(phi) < eps:
            return x, F, phi, k

        dphi = f(x)
        if dphi == 0:
            # Производная 0 -- дальше Ньютон не идёт
            break

        x = x - phi / dphi

    # После выхода по числу итераций или из-за dphi == 0
    F = integrate_trap(a, x, eps, n_trap)
    phi = F - b_val
    return x, F, phi, k


def main():
    use_bisection = False  # False -> запускать только Ньютон с x0 = (a + c) / 2

    # Чтение данных из in.txt
    with open("in.txt", "r", encoding="utf-8") as f_in:
        data = f_in.read().split()

    if len(data) < 4:
        raise ValueError(
            "В in.txt должно быть минимум 4 числа: a b c n [eps] [max_iter]"
        )

    a = float(data[0])
    b_val = float(data[1])
    c = float(data[2])
    n_trap = int(float(data[3]))

    # Дополнительные параметры: eps и max_iter (необязательны)
    if len(data) >= 5:
        eps = float(data[4])
    else:
        eps = 1e-6

    if len(data) >= 6:
        max_iter = int(float(data[5]))
    else:
        max_iter = 50

    if use_bisection:
        # Сначала находим корень на [a, c] дихотомией, затем уточняем Ньютоном
        x_bis, _, _, bis_iters = bisection_solve(
            a=a,
            b_val=b_val,
            left=a,
            right=c,
            n_trap=n_trap,
            eps=eps,
            max_iter=200,
        )
    else:
        # Только Ньютон: стартуем с середины отрезка [a, c]
        x_bis = 0.5 * (a + c)
        bis_iters = 0

    x_root, F_root, phi_root, iters = newton_solve(
        a=a,
        b_val=b_val,
        x0=x_bis,
        n_trap=n_trap,
        eps=eps,
        max_iter=max_iter
    )

    # Запись результата в out.txt
    with open("out.txt", "w", encoding="utf-8") as f_out:
        f_out.write(f"x_bisect = {x_bis:.10f}\n")
        f_out.write(f"bisection_iterations = {bis_iters}\n")
        f_out.write(f"x = {x_root:.10f}\n")
        f_out.write(f"integral = {F_root:.10f}\n")
        f_out.write(f"Phi(x) = integral - b = {phi_root:.10e}\n")
        f_out.write(f"iterations = {iters}\n")


if __name__ == "__main__":
    main()
