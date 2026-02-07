import sys
import numpy as np


def read_matrix(path: str) -> np.ndarray:
    """
    Читает матрицу из текстового файла: по строке на строку, числа через пробел.
    Проверяет, что матрица квадратная.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        raise ValueError("Пустой файл ввода")

    rows = []
    for line in lines:
        parts = line.split()
        rows.append([float(x) for x in parts])

    n = len(rows)
    m = len(rows[0])
    if any(len(r) != m for r in rows):
        raise ValueError("Строки матрицы имеют разную длину")
    if n != m:
        raise ValueError("Матрица должна быть квадратной")

    return np.array(rows, dtype=float)


def write_results(path: str, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                  iterations: int, max_offdiag: float) -> None:
    """
    Записывает собственные значения, собственные векторы и диагностику.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("eigenvalues:\n")
        f.write(" ".join(f"{v:.12f}" for v in eigenvalues))
        f.write("\n")
        f.write("eigenvectors:\n")
        for i in range(eigenvectors.shape[0]):
            f.write(" ".join(f"{v:.12f}" for v in eigenvectors[i, :]))
            f.write("\n")
        f.write(f"iterations = {iterations}\n")

def jacobi_eigen(A: np.ndarray, eps: float = 1e-10, max_iter: int | None = None):
    """
    Метод вращений Якоби для симметричной матрицы.
    Возвращает (eigenvalues, eigenvectors, iterations, max_offdiag).
    """
    n = A.shape[0]
    if max_iter is None:
        max_iter = 100 * n * n

    # Проверка симметричности: при нарушении — ошибка (не симметризуем).
    if np.max(np.abs(A - A.T)) > 1e-12:
        raise ValueError("Матрица должна быть симметричной")

    A = A.copy()
    X = np.eye(n, dtype=float)

    for it in range(1, max_iter + 1):
        off = np.abs(A)
        np.fill_diagonal(off, 0.0)
        p, q = np.unravel_index(np.argmax(off), off.shape)
        max_offdiag = off[p, q]

        if max_offdiag <= eps:
            return np.diag(A).copy(), X, it - 1, max_offdiag

        apq = A[p, q]
        if apq == 0.0:
            continue

        # Устойчивая формула вычисления c и s
        tau = (A[q, q] - A[p, p]) / (2.0 * apq)
        t = np.sign(tau) / (abs(tau) + np.sqrt(1.0 + tau * tau))
        c = 1.0 / np.sqrt(1.0 + t * t)
        s = t * c

        app = A[p, p]
        aqq = A[q, q]

        # Обновление строк/столбцов p и q
        for i in range(n):
            if i == p or i == q:
                continue
            aip = A[i, p]
            aiq = A[i, q]
            A[i, p] = c * aip - s * aiq
            A[p, i] = A[i, p]
            A[i, q] = s * aip + c * aiq
            A[q, i] = A[i, q]

        A[p, p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        A[q, q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        A[p, q] = 0.0
        A[q, p] = 0.0

        # Обновление матрицы собственных векторов
        for i in range(n):
            xip = X[i, p]
            xiq = X[i, q]
            X[i, p] = c * xip - s * xiq
            X[i, q] = s * xip + c * xiq

    off = np.abs(A)
    np.fill_diagonal(off, 0.0)
    return np.diag(A).copy(), X, max_iter, float(np.max(off))


def main():
    in_path = 'in.txt'
    out_path = 'out.txt'

    A = read_matrix(in_path)
    eigenvalues, eigenvectors, iterations, max_offdiag = jacobi_eigen(A)

    # Нормировка собственных векторов
    for i in range(eigenvectors.shape[1]):
        norm = np.linalg.norm(eigenvectors[:, i])
        if norm > 0:
            eigenvectors[:, i] /= norm

    # Сортировка по возрастанию собственных значений
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    write_results(out_path, eigenvalues, eigenvectors, iterations, max_offdiag)


if __name__ == "__main__":
    main()
