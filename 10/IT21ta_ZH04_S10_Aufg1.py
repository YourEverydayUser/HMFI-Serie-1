import numpy as np

ATest = np.array([[8, 5, 2], [5, 9, 1], [4, 2, 7]])
bTest = np.array([19, 5, 34])

def B_jacobi(A):
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    L = np.tril(A) - D
    R = np.triu(A) - D
    return -D_inv @ (L + R)

def jacobi(A, b, xstart):
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    L = np.tril(A) - D
    R = np.triu(A) - D
    B = -D_inv @ (L + R)
    return B @ xstart + D_inv @ b

def iterate_jacobi(A, b, x_start, number_of_iterations):
    xneu = x_start
    for i in range(0, number_of_iterations):
        xalt = xneu
        xneu = jacobi(A, b, xalt)
    return xneu

# Aufgabe 1 b)
xstart = np.array([1, -1, 3])
print("Aufgabe 1b) x(3) = ")
print(iterate_jacobi(ATest, bTest, xstart, 3))


# Aufgabe 1 c)
def a_posteriori(A, b, B, n):
    return np.linalg.norm(B, np.inf) / (1 - np.linalg.norm(B, np.inf)) * np.linalg.norm(iterate_jacobi(A, b, xstart, n) - iterate_jacobi(A, b, xstart, n - 1), np.inf)


print("Aufgabe 1c) Absoluter fehler für x(3) = ")
print(a_posteriori(ATest, bTest, B_jacobi(ATest), 3))


# Aufgabe 1 d)
def number_iterations_apriori(A, b, B, xstart, tol):
    res = np.log(tol * (1 - np.linalg.norm(B, np.inf)) / (np.linalg.norm(iterate_jacobi(A, b, xstart, 1), np.inf))) * 1 / np.log(np.linalg.norm(B_jacobi(A), np.inf))
    return np.ceil(res)

xstart_zero = np.zeros(3)
tol = 10**-4

number_of_iterations = number_iterations_apriori(ATest, bTest, B_jacobi(ATest), xstart_zero, tol)
print("Aufgabe 1d) n = ")
print(number_of_iterations)

#x = np.linalg.norm(iterate_jacobi(A, b, xstart_zero, 40) - np.array([2, -1, 4]), np.inf)
#print(x)
#print(np.linalg.norm(B, np.inf))
#result2 = np.linalg.norm(B, np.inf)**97 / (1 - np.linalg.norm(B, np.inf)) * np.linalg.norm(iterate_jacobi(A, b, xstart_zero, 1) - xstart_zero, np.inf)
#print(result2)


# Aufgabe 1 e)
xstart2 = np.array([1, -1, 3])
tol = 10**-4
number_of_iterations2 = number_iterations_apriori(ATest, bTest, B_jacobi(ATest), xstart2, tol)
print("Aufgabe 1e) n = ")
print(number_of_iterations2)








