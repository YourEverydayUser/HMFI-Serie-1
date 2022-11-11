import numpy as np
import matplotlib.pyplot as plt
from numpy import float64
import sys

sys.path.insert(1, "C://Users//Fabian//Desktop//OneDrive Privat//OneDrive//Desktop//Höhere Mathematik für "
                   "Informatiker//HMFI-Serie-6")

from IT21a_ZH04_S6_Aufg2 import IT21a_ZH04_S6_Aufg2


#Teilaufgabe a

A4 = np.array([[1, 0, 0, 0],
               [1, 2, 4, 8],
               [1, 9, 81, 729],
               [1, 13, 169, 2197]], dtype=float64)

b4 = np.array([150, 104, 172, 152], dtype=float64)

x_values = np.arange(0, 17, 0.1)
y_values = []

p = IT21a_ZH04_S6_Aufg2(A4, b4)[5]
p = p[::-1]

for x in x_values:
    y_values.append(np.polyval(p, x))

markers_on = [0, 20, 90, 130]
plt.plot(x_values, y_values, '-o', markevery=markers_on, color="green")
plt.grid()
plt.xticks(np.arange(0, 17), np.arange(1997, 2014), rotation=70)
plt.ylim(75, 220)
plt.xlim(0, 17)


#Teilaufgabe b

print(y_values[60], y_values[70])

#Teilaufgabe c

p_polyfit = np.polyfit([0, 2, 9, 13], [150, 104, 172, 152], 3)
y_values_polyfit = []

for x in x_values:
    y_values_polyfit.append(np.polyval(p_polyfit, x))

print(y_values_polyfit[60], y_values_polyfit[70])
plt.plot(x_values, y_values_polyfit, color="red")
plt.show()

