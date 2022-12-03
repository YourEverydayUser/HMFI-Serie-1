import math
import timeit

import numpy as np

A = np.array([[8, 5, 2],
              [5, 9, 1],
              [4, 2, 7]])

b = np.array([[19], [5], [34]])

x0 = np.array([[1], [-1], [3]])


def apriori(B, tol, x1, x0):
    Bnorm = np.linalg.norm(B, np.inf)
    xNorm = np.linalg.norm(x1-x0, np.inf)
    numerator = (tol*(1 - Bnorm)) / xNorm

    return np.ceil(math.log(numerator) / math.log(Bnorm))


def jacobi(R, L, negativeDInverse, x0, b, tol):
    B = negativeDInverse@(L+R)
    xnext = B@x0 - negativeDInverse@b
    xn = B@xnext - negativeDInverse@b
    counter = 2
    n2 = apriori(B, tol, xnext, x0)

    while np.linalg.norm(np.abs(xn - xnext), np.inf) > tol:
        xnext = xn
        xn = B@xnext - negativeDInverse@b
        counter += 1

    return xn, counter, n2


def gauss_seidel(R, L, D, x0, b, tol):
    B = -1 * np.linalg.inv(D + L)@R
    DLInverse = np.linalg.inv(D + L)
    xnext = (-1 * DLInverse)@R@x0 + DLInverse@b
    xn = (-1 * DLInverse)@R@xnext + DLInverse@b
    counter = 2
    n2 = apriori(B, tol, xnext, x0)

    while np.linalg.norm(np.abs(xn - xnext), np.inf) > tol:
        xnext = xn
        xn = (-1 * DLInverse)@R@xnext + DLInverse@b
        counter += 1

    return xn, counter, n2


def IT21ta_ZH04_Aufg3a(A, b, x0, tol, opt):
    R = np.triu(A) - np.diag(np.diag(A))
    L = np.tril(A) - np.diag(np.diag(A))
    D = np.diag(np.diag(A))
    negativeDInverse = -1 * np.linalg.inv(np.diag(np.diag(A)))

    if opt == 0:
        solution = jacobi(R, L, negativeDInverse, x0, b, tol)
        xn = solution[0]
        n = solution[1]
        n2 = solution[2]
    else:
        solution = gauss_seidel(R, L, D, x0, b, tol)
        xn = solution[0]
        n = solution[1]
        n2 = solution[2]

    return xn, n, n2


print(IT21ta_ZH04_Aufg3a(A, b, x0, 10**-4, 0))
print(IT21ta_ZH04_Aufg3a(A, b, x0, 10**-4, 1))

dim = 3000
A = np.diag(np.diag(np.ones((dim, dim))*4000)) + np.ones((dim, dim))
dum1 = np.arange(1, int(dim/2+1), dtype=np.float64).reshape((int(dim/2), 1))
dum2 = np.arange(int(dim/2), 0, -1, dtype=np.float64).reshape((int(dim/2), 1))
x = np.append(dum1, dum2, axis=0)
b = A@x
x0 = np.zeros((dim, 1))
tol = 1*np.e**-4


def IT21ta_ZH04_Aufg3b(A, b, x0, tol):
    timeJacopi = timeit.timeit("IT21ta_ZH04_Aufg3a(A, b, x0, tol, 0)", "from __main__ import IT21ta_ZH04_Aufg3a, A, b, x0, tol")
    timeGauss = timeit.timeit("IT21ta_ZH04_Aufg3a(A, b, x0, tol, 1)", "from __main__ import IT21ta_ZH04_Aufg3a, A, b, x0, tol")

    return timeJacopi, timeGauss


#print(IT21ta_ZH04_Aufg3b(A, b, x0, tol))
