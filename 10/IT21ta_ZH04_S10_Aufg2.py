import numpy as np

ATest = np.array([[8, 5, 2], [5, 9, 1], [4, 2, 7]])
bTest = np.array([19, 5, 34])

print("--------------------------------------------------")
# AUFGABE 2

def B_gauss_seidel(A):
    D = np.diag(np.diag(A))
    L = np.tril(A) - D
    R = np.triu(A) - D
    return -np.linalg.inv(D + L) @ R


def gauss_seidel(A, b, xstart):
    #xstart = np.transpose(xstart)
    D = np.diag(np.diag(A))
    L = np.tril(A) - D
    R = np.triu(A) - D
    DL_inv = np.linalg.inv(D + L)
    return (-DL_inv @ R) @ xstart + DL_inv @ b


def iterate_gauss_seidel(A, b, x_start, number_of_iterations):
    xneu = x_start
    for i in range(0, number_of_iterations):
        xalt = xneu
        xneu = gauss_seidel(A, b, xalt)
    return xneu

def number_of_iterations_gauss_seidel(A, b, x_start, tol):
    xneu = x_start
    n = 1
    xalt = xneu
    xneu = gauss_seidel(A, b, xalt)
    while np.linalg.norm(xalt - xneu) > tol:
        xalt = xneu
        xneu = gauss_seidel(A, b, xalt)
        n += 1
    return n


# Aufgabe 2b)
xstart = np.array([1, -1, 3])
print("Aufgabe 2b) x(3) = ")
#print(iterate_gauss_seidel(ATest, bTest, xstart, 3))


# Aufgabe 2 c)
def a_posteriori(A, b, B, n):
    return np.linalg.norm(B, np.inf) / (1 - np.linalg.norm(B, np.inf)) * np.linalg.norm(iterate_gauss_seidel(A, b, xstart, n) - iterate_gauss_seidel(A, b, xstart, n - 1), np.inf)

print("Aufgabe 2c) Absoluter fehler für x(3) = ")
print(a_posteriori(ATest, bTest, B_gauss_seidel(ATest), 3))


# Aufgabe 2 d)
def number_iterations_apriori(A, b, B, xstart, tol):
    res = np.log(tol * (1 - np.linalg.norm(B, np.inf)) / (np.linalg.norm(iterate_gauss_seidel(A, b, xstart, 1), np.inf))) * 1 / np.log(np.linalg.norm(B, np.inf))
    return np.ceil(res)

xstart_zero = np.zeros(3)
tol = 10**-4

number_of_iterations = number_iterations_apriori(ATest, bTest, B_gauss_seidel(ATest), xstart_zero, tol)
print("Aufgabe 2d) n = ")
print(number_of_iterations)


# Aufgabe 2 e)
xstart2 = np.array([1, -1, 3])
tol = 10**-4
number_of_iterations2 = number_iterations_apriori(ATest, bTest, B_gauss_seidel(ATest), xstart2, tol)
print("Aufgabe 2e) n = ")
print(number_of_iterations2)








