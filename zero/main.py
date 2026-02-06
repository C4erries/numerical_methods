import numpy as np


def power_method(A, eps=1e-6, max_iter=1000, delta=1e-12):
    n = A.shape[0]
    rng = np.random.default_rng()
    y = rng.random(n)
    norm_y = np.max(np.abs(y))
    if norm_y == 0:
        return 0.0, y, 0
    x = y / norm_y
    lambda_prev = None

    for k in range(1, max_iter + 1):
        y = A @ x
        norm_y = np.max(np.abs(y))
        if norm_y == 0:
            return 0.0, x, k
        x_new = y / norm_y

        mask = np.abs(x) > delta
        if np.any(mask):
            lambda_est = float(np.mean(y[mask] / x[mask]))
        else:
            lambda_est = float((x @ y) / (x @ x))

        if lambda_prev is not None:
            if abs(lambda_est - lambda_prev) <= eps * max(1.0, abs(lambda_est)):
                return lambda_est, x_new, k

        x = x_new
        lambda_prev = lambda_est

    return (lambda_prev if lambda_prev is not None else 0.0), x, max_iter


def solve_from_file(in_filename="in.txt", out_filename="out.txt"):
    with open(in_filename, "r", encoding="utf-8") as f:
        data = f.read().split()

    if not data:
        raise ValueError("Пустой файл ввода")

    n = int(float(data[0]))
    if n <= 0:
        raise ValueError("n должно быть больше нуля")
    need = 1 + n * n
    if len(data) < need:
        raise ValueError("Недостаточно данных для матрицы")

    a_vals = list(map(float, data[1:need]))
    A = np.array(a_vals, dtype=float).reshape((n, n))

    rest = data[need:]
    eps = float(rest[0]) if len(rest) >= 1 else 1e-6
    max_iter = int(float(rest[1])) if len(rest) >= 2 else 1000
    delta = float(rest[2]) if len(rest) >= 3 else 1e-12
    if delta <= 0:
        raise ValueError("дельта должна быть больше нуля")

    lam, vec, iters = power_method(A, eps=eps, max_iter=max_iter, delta=delta)

    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(f"lambda = {lam:.10f}\n")
        f.write("eigenvector:\n")
        f.write(" ".join(f"{v:.10f}" for v in vec))
        f.write("\n")
        f.write(f"iterations = {iters}\n")


if __name__ == "__main__":
    solve_from_file("in.txt", "out.txt")
