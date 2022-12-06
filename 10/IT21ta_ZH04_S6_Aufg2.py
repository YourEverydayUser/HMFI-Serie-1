from unittest import skip
import numpy as np


def gauss(A, b):
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    #b = b.reshape(4, 1)
    if (A.shape[0] != A.shape[1]):
        raise Exception("Keine n x n Matrix")
    for i in range(A.shape[0]):
        if A[i, i] == 0:
            if only_zeros_below(A, i) == -1:
                raise Exception("Matrix hat nicht vollen Rang")
            non_zero_index = only_zeros_below(A, i)
            A[(i, non_zero_index),] = A[(non_zero_index, i),]
            b[(i, non_zero_index),] = b[(non_zero_index, i),]
    A, b = make_zeros(A, b)

    return A, b


def only_zeros_below(A, column):
    for j in range((column + 1), A.shape[1]):
        if A[j, column] != 0:
            return j
    return -1


def calculate_det(A, b):
    sign = 1
    der = 1
    for i in range(A.shape[0]):
        der *= sign * A[i, i]
        sign *= -1
    return der


def calculate_system(A, b):
    result = np.zeros(A.shape[0])
    for i in reversed(range(A.shape[0])):  # Durch Zeilen iterieren, von unten nach oben
        result[i] = b[i]
        for j in reversed(range(i, A.shape[0])):  # Durch Spalten iterieren, von rechts nach links
            if j == i:
                continue
            result[i] -= A[i, j] * result[j]
        result[i] = result[i] / A[i, i]
    return result


def make_zeros(A, b):
    A = np.array(A, dtype=np.float64)
    for i in range(A.shape[0]):  # Durch Diagonale iterieren
        if A[i, i] == 0:
            continue
        for j in range(i + 1, A.shape[0]):  # Durch Zeilen iterieren
            if A[j, i] == 0:
                continue
            factor = A[j, i] / A[i, i]
            b[j] -= b[i] * factor
            for k in range(i, A.shape[0]):  # Durch Spalten iterieren
                A[j, k] = A[j, k] - A[i, k] * factor

    return A, b


def IT21ta_ZH04_S6_Aufg2(A, b):
    A, b = gauss(A, b)
    return calculate_system(A, b)


A = np.array([5, 11, 1, 6, 0, 3, 1, 1, -2, 1, -1, 2, 3, 4, 1, -2]).reshape(4, 4)
print(A)
b = np.array(np.arange(4))
print(b)

print(IT21ta_ZH04_S6_Aufg2(A, b))

res = np.linalg.solve(A, b)
print(res)

A2 = np.array([[6,8,1,12,-2],[9,14,7,4,4], [6,11,-2,9,-11],[6,1,-12,9,-1], [4,22,-4,9,-11]])
b2 = np.array([9, -2, 4,-1,3])

res2 = np.linalg.solve(A2, b2)
print(res2)
print(IT21ta_ZH04_S6_Aufg2(A2, b2))