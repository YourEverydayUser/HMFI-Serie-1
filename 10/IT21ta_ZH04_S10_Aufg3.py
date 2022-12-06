import timeit
import numpy as np
import matplotlib.pyplot as plt
from IT21ta_ZH04_S10_Aufg1 import jacobi as jacobi
from IT21ta_ZH04_S10_Aufg1 import number_iterations_apriori as apriori_jacobi
from IT21ta_ZH04_S10_Aufg2 import gauss_seidel as gauss_seidel
from IT21ta_ZH04_S10_Aufg2 import number_iterations_apriori as apriori_gauss_seidel
from IT21ta_ZH04_S6_Aufg2 import IT21ta_ZH04_S6_Aufg2

print("------------------------------------------")
ATest = np.array([[8, 5, 2], [5, 9, 1], [4, 2, 7]])
bTest = np.array([19, 5, 34])
x0Test = np.array([1, -1, 3])
tolTest = 10 ** -4


# Aufgabe 3a)

def IT21ta_ZH04_S10_Aufg3a(A, b, x0, tol, opt):
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    L = np.tril(A) - D
    R = np.triu(A) - D

    n = 1
    xalt = x0
    if opt == 0:
        B_jacobi = -D_inv @ (L + R)
        xneu = jacobi(A, b, xalt)
        while np.linalg.norm(xneu - xalt, np.inf) > tol:
            xalt = xneu
            xneu = jacobi(A, b, xalt)
            n += 1
        n2 = apriori_jacobi(A, b, B_jacobi, x0, tol)
    elif opt == 1:
        B_gauss_seidel = -np.linalg.inv(D + L) @ R
        xneu = gauss_seidel(A, b, xalt)
        while np.linalg.norm(xneu - xalt, np.inf) > tol:
            xalt = xneu
            xneu = gauss_seidel(A, b, xalt)
            n += 1
        n2 = apriori_gauss_seidel(A, b, B_gauss_seidel, x0, tol)
    else:
        print("Enter '0' for jacobi, or '1' for gauss-seidel")
        return
    return xneu, n, n2


print("Aufgabe 3a) [xn, n, n2] = ")
print(IT21ta_ZH04_S10_Aufg3a(ATest, bTest, x0Test, tolTest, 1))

# Aufgabe 3b)

dim = 3000
A = np.diag(np.diag(np.ones((dim, dim)) * 4000)) + np.ones((dim, dim))
dum1 = np.arange(1, np.int(dim / 2 + 1), dtype=np.float64).reshape((np.int(dim / 2), 1))
dum2 = np.arange(np.int(dim / 2), 0, -1, dtype=np.float64).reshape((np.int(dim / 2), 1))
x = np.append(dum1, dum2, axis=0)
b = A @ x
x0 = np.zeros((dim, 1))
tol = 1e-4

start = timeit.default_timer()
solutionLinalg = np.linalg.solve(A, b)
end = timeit.default_timer()
print("linalg.solve(): ")
print(end - start)

start = timeit.default_timer()
solutionGaussSeidel = IT21ta_ZH04_S10_Aufg3a(A, b, x0, tol, 1)[0]
end = timeit.default_timer()
print("Gauss-Seidel: ")
print(end - start)

#start = timeit.default_timer()
#IT21ta_ZH04_S6_Aufg2(A, b)
#end = timeit.default_timer()
#print("Gauss: ")
#print(end - start)

# Aufgabe 3c)
solutionJacobi = IT21ta_ZH04_S10_Aufg3a(A, b, x0, tol, 0)[0]
print(solutionJacobi)
print(solutionLinalg)
absErrorLinalg = abs(x - solutionLinalg)
absErrorJacobi = abs(x - solutionJacobi)
absErrorGaussSeidel = abs(x - solutionGaussSeidel)
plt.plot(absErrorLinalg)
plt.plot(absErrorJacobi)
plt.plot(absErrorGaussSeidel)
plt.show()


