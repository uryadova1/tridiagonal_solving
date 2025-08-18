import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


A = np.array([0, 9, 3, 1, 6, 1, 1, 6, 4, 8])
B = np.array([2, 7, 1, 1, 9, 8, 6, 8, 5, 4])
C = np.array([7, 6, 7, 9, 0, 7, 4, 3, 8, 0])
F = np.array([5, 6, 1, 6, 9, 2, 0, 0, 6, 1])


n = len(B)
diagonals = [A[1:], B, C[:-1]]
offsets = [-1, 0, 1]
matrix = diags(diagonals, offsets, shape=(n, n), format='csr')

# Решаем систему
solution = spsolve(matrix, F)

print("Решение SciPy:", solution)