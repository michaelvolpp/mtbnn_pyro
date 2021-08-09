import numpy as np
from matplotlib import pyplot as plt


def f(pt):
    return np.sin(pt[:, 0] ** 2 + pt[:, 1] ** 2)


x = np.array([1.5, 3.5, 4.5])
y = np.array([0.5, 2.1])
xx, yy = np.meshgrid(x, y)
print(x)
print(y)
print(xx)
print(xx.ravel())
print(xx.ravel().reshape(xx.shape))
print(yy)
print(yy.ravel())
print(yy.ravel().reshape(yy.shape))
z = f(np.stack([xx.ravel(), yy.ravel()], axis=1)).reshape(xx.shape)
print(z)
plt.contourf(xx, yy, z)
plt.show()
