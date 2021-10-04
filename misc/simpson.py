from scipy import integrate
import numpy as np

x1 = [1.0, 2.0, 3.0]
x2 = [1.0, 1.1, 3.0]
x3 = [1.0, 2.9, 3.0]
y = [-1.0, 1.0, 0.0]
mean = np.mean(y)
a1 = integrate.trapezoid(y=y, x=x1) / (x1[-1] - x1[0])
a2 = integrate.trapezoid(y=y, x=x2) / (x2[-1] - x2[0])
a3 = integrate.trapezoid(y=y, x=x3) / (x3[-1] - x3[0])
print(f"mean = {mean:.4f}")
print(f"a1   = {a1:.4f}")
print(f"a2   = {a2:.4f}")
print(f"a3   = {a3:.4f}")
