import numpy as np

A = np.array([[8, 5, 2],
              [5, 9, 1],
              [4, 2, 7]])

b = np.array([[19], [5], [34]])

D = np.array([[8, 0, 0],
              [0, 9, 0],
              [0, 0, 7]])

DinvNegative = -1 * np.linalg.inv(D)

L = np.array([[0, 0, 0],
              [5, 0, 0],
              [4, 2, 0]])

R = np.array([[0, 5, 2],
              [0, 0, 1],
              [0, 0, 0]])

xNext = np.array([[1], [-1], [3]])

for k in range(3):
    xNext = DinvNegative@(L + R)@xNext - DinvNegative@b
    print(xNext)


