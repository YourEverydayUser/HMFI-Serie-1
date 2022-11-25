import numpy as np
import matplotlib.pyplot as plt
from IT21ta_ZH04_S9_Aufg2 import IT21ta_ZH04_S9_Aufg2 as myfunc

xaxis = np.arange(0, 1000, 1)
dxmax_values = []
dxobs_values = []
ratio_values = []

for i in range(1000):
    A = np.random.rand(100, 100)
    b = np.random.rand(100, 1)
    A_tilde = A + np.random.rand(100, 100)/10**5
    b_tilde = b + np.random.rand(100, 1)/10**5
    dxmax, dxobs = myfunc(A, A_tilde, b, b_tilde)[2:4]
    dxmax_values.append(dxmax)
    dxobs_values.append(dxobs)
    ratio_values.append(dxmax/dxobs)

plt.semilogy(xaxis, dxmax_values, label="dxmax", alpha=0.5)
plt.semilogy(xaxis, dxobs_values, label="dxobs", alpha=0.5)
plt.semilogy(xaxis, ratio_values, label="dxmax/dxobs", alpha=0.5)
plt.legend()
plt.show()

# Aufgabe 3: Es lässt sich beobachten, dass die beobachteten Werte der relativen Fehler (dx_obs) stets
# unterhalb der berechneten Schranke (dx_max) sind. Daher erfüllt dx_max die Funktion einer oberen Schranke.
# Das Verhältnis von dx_max zu dx_obs ist jedoch oft eher gross (um ca. 10^3). D.h. die obere Schranke ist
# oft um einen Faktor 1000 zu hoch. Dies ist vermutlich durch die zufällig generierten Werte zu erklären.