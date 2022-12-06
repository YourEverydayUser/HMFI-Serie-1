import numpy
import numpy as np
import numpy.linalg.linalg as lg
import scipy.linalg

A = np.array([[1, 2, -1],
             [4, -2, 6],
             [3, 1, 0]])

H1 = np.array([[-0.1824, 0.9126, -0.365],
               [0.9126, 0.2956, 0.28178],
               [-0.365, 0.2816, 0.8874]])


#Aufgabe 3
B = np.array([[0.02, 0.03, 0.01],
              [0.01, 0.017, 0.006],
              [0.002, 0.003, 0.002]])

B_wrong = np.array([[0.02, 0.03, 0.01],
                    [0.01, 0.017, 0.006],
                    [0.002, 0.003, 0.002]]) - 0.0001

b_wrong = np.array([5.82, 0.43, 0.936])

print(lg.solve(B_wrong, b_wrong))



print(lg.cond(B, p=numpy.inf))
print(lg.inv(B))