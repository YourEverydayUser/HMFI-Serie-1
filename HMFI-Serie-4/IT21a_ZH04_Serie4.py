import numpy as np
import matplotlib.pyplot as plt

def f_next(x):
    return (1/221) * (230*x**4 + 18 * x**3 + 9 * x**2 -9)

x = -0.25
x_1 = f_next(x)

while np.absolute(x) - np.absolute(x_1) > 10 ** -7:
    x = x_1
    print(x_1)
    x_1 = f_next(x)

x2 = 1.25
x2_1 = f_next(x2)

while np.absolute(x2) - np.absolute(x2_1) > 10 ** -15:
    x2 = x2_1
    print(x2_1)
    x2_1 = f_next(x2)

print(x_1)
print(x2_1)