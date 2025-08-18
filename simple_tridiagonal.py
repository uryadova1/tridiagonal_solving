import numpy as np


def thomas_algorithm2(A, B, C, F, n):
    new_b = np.zeros(n, dtype=float)
    new_f = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)

    new_b[0] = B[0]
    new_f[0] = F[0]

    for i in range(1, n):
        den = A[i] / new_b[i - 1]
        new_b[i] = B[i] - den * C[i - 1]
        new_f[i] = F[i] - den * new_f[i - 1]

    x[-1] = new_f[-1] / new_b[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (new_f[i] - C[i] * x[i + 1]) / new_b[i]

    return x


def thomas_algorithm1(A, B, C, F, n):
    alpha = np.zeros(n + 1, dtype=float)
    beta = np.zeros(n + 1, dtype=float)
    x = np.zeros(n, dtype=float)

    alpha[1] = -C[0] / B[0]
    beta[1] = F[0] / B[0]

    for i in range(1, n - 1):
        den = A[i] * alpha[i] + B[i]
        alpha[i + 1] = -C[i] / den
        beta[i + 1] = (F[i] - A[i] * beta[i]) / den

    x[-1] = (F[-1] - A[-1] * beta[-1]) / (B[-1] + A[-1] * alpha[-1])
    # x[-1] = beta[-1]

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]

    return x


if __name__ == "__main__":
    n = 10
    A = np.array([0, 9, 3, 1, 6, 1, 1, 6, 4, 8])
    B = np.array([2, 7, 1, 1, 9, 8, 6, 8, 5, 4])
    C = np.array([7, 6, 7, 9, 0, 7, 4, 3, 8, 0])
    F = np.array([5, 6, 1, 6, 9, 2, 0, 0, 6, 1])


    # print(f"A: {A}\nB: {B}\nC: {C}\nF: {F}")
    #
    # sol = thomas_algorithm1(A, B, C, F, n)
    # print(f"X: {sol}")
    # print()

    print(f"A: {A}\nB: {B}\nC: {C}\nF: {D}")

    sol = thomas_algorithm2(A, B, C, D, n)
    print(f"X: {sol}")