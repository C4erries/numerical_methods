import math


def f(x: float) -> float:
    """
    Функция f(x) под интегралом.
    """
    return 1.0 / (1.0 + x**2)

def intergate(h: float, a: float, x:float) -> float:
    s = 0.5 * (f(a) + f(x))
    for k in range(1, int((x-a)/h)):
        t = a + k * h
        s += f(t)
    return s * h

def integrate_trap(a: float, x: float, eps: float, m: int) -> float:
    """
    Численное вычисление интеграла ∫_a^x f(t) dt
    по формуле трапеций с m разбиениями.

    Корректно работает и если x < a (учитывается знак).
    """
    if x == a:
        return 0.0

    # Если x < a, интегрируем от x до a и меняем знак
    if x < a:
        return -integrate_trap(x, a, eps, m)
    
    h = (x - a) / m
    I = intergate(h, a, x)
    I2 = intergate(h/2, a, x)
    while abs(I - I2) > (2**2 - 1)*eps:
        I=I2
        h=h/2
        I2 = intergate(h/2,a, x)
        
    
    return I2


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

    # Начальное приближение: середина отрезка [a, c]
    x0 = 0.5 * (a + c)

    x_root, F_root, phi_root, iters = newton_solve(
        a=a,
        b_val=b_val,
        x0=x0,
        n_trap=n_trap,
        eps=eps,
        max_iter=max_iter
    )

    # Запись результата в out.txt
    with open("out.txt", "w", encoding="utf-8") as f_out:
        f_out.write(f"x = {x_root:.10f}\n")
        f_out.write(f"integral = {F_root:.10f}\n")
        f_out.write(f"Phi(x) = integral - b = {phi_root:.10e}\n")
        f_out.write(f"iterations = {iters}\n")


if __name__ == "__main__":
    main()
