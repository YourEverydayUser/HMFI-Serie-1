# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:26:09 2020

Höhere Mathematik 1, Serie 8, Gerüst für Aufgabe 2

Description: calculates the QR factorization of A so that A = QR
Input Parameters: A: array, n*n matrix
Output Parameters: Q : n*n orthogonal matrix
                   R : n*n upper right triangular matrix            
Remarks: none
Example: A = np.array([[1,2,-1],[4,-2,6],[3,1,0]]) 
        [Q,R]=Serie8_Aufg2(A)

@author: knaa
"""
import numpy as np

def Serie8_Aufg2(A):
    
    A = np.copy(A)                       #necessary to prevent changes in the original matrix A_in
    A = A.astype('float64')              #change to float
    
    n = np.shape(A)[0]
    
    if n != np.shape(A)[1]:
        raise Exception('Matrix is not square') 
    
    Q = np.eye(n)
    Q = Q.astype('int32')
    R = A
    
    for j in np.arange(0, n-1):

        a = np.copy(R)[j:, j].reshape(n-j, 1)
        e = np.eye(n)[j:, j].reshape(n-j, 1)
        length_a = np.linalg.norm(a)
        if a[0] >= 0: sig = 1
        else: sig = -1
        v = a + sig * length_a * e
        u = (1/np.linalg.norm(v)) * v
        H = np.eye(n)[j:, j:] - 2 * u @ np.transpose(u)
        Qi = np.eye(n)
        Qi[j:, j:] = H
        R = Qi @ R
        Q = Q @ np.transpose(Qi)

    return(Q,R)


A = np.array([[1, -2, 3],
              [-5, 4, 1],
              [2, -1, 3]])

B = np.array([[0.0199, 0.0299, 0.0099],
             [0.0099, 0.0169, 0.0059],
             [0.0019, 0.0029, 0.0019]])

b = np.array([5.82, 3.4, 0.936])

B1 = np.array([[19900, 29900, 9900],
              [9900, 16900, 5900],
              [1900, 2900, 1900]])

b1 = np.array([5820000, 3400000, 936000])

#Aufgabe 1.b
print(Serie8_Aufg2(A))

#Aufgabe 2.c
import timeit

TestMatrix = np.random.rand(100,100)
t1=timeit.repeat("Serie8_Aufg2(TestMatrix)", "from __main__ import Serie8_Aufg2, TestMatrix", number=100)
t2=timeit.repeat("np.linalg.qr(TestMatrix)", "from __main__ import np, TestMatrix", number=100)
#t1=timeit.repeat("Serie8_Aufg2(A)", "from __main__ import Serie8_Aufg2, A", number=100)
#t2=timeit.repeat("np.linalg.qr(A)", "from __main__ import np, A", number=100)
avg_t1 = np.average(t1)/100
avg_t2 = np.average(t2)/100
print(avg_t1)
print(avg_t2)




#Aufgabe 3.c Lösungsvektor x
print(np.linalg.solve(B, b))



