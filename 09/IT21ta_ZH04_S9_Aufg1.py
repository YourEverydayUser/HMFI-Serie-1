import numpy as np

# AUFGABE 1

A = np.array([[1, 0, 2], [0, 1, 0], [10**(-4), 0, 10**(-4)]])
A = np.array(A, dtype=np.float64)
b = np.array([1, 1, 0])
b = np.array(b, dtype=np.float64)

# Aufgabe 1a)
A_inv = np.linalg.inv(A)
A_norm = np.linalg.norm(A, np.inf)
A_inv_norm = np.linalg.norm(A_inv, np.inf)
cond_A = A_norm*A_inv_norm
print("1a) Cond(A) = ")
print(cond_A)

# Aufgabe 1b)
epsilon = 1/100/cond_A
print("1b) Epsilon = ")
print(epsilon)

# Aufgabe 1c)
b_tilde = np.array([1, 1, epsilon])
x = np.linalg.solve(A, b)
x_tilde = np.linalg.solve(A, b_tilde)

rel_error = np.linalg.norm((x - x_tilde), np.inf) / np.linalg.norm(x, np.inf)
print("1c) Relative error of x = ")
print(rel_error)

# Aufgabe 1d)
abs_error_A_error = 1*10**-7 * 3
rel_error_A = abs_error_A_error / A_norm
rel_error_b = np.linalg.norm((b - b_tilde), np.inf) / np.linalg.norm(b, np.inf)

rel_error2 = cond_A/(1-cond_A*abs_error_A_error/A_norm) * (rel_error_A + rel_error_b)
print("1d) Relative error of x = ")
print(rel_error2)











