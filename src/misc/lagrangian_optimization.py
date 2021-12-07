from matplotlib import pyplot as plt
import numpy as np

w1 = np.arange(-3.0, 3.0, 0.1)
w2 = np.arange(-3.0, 3.0, 0.1)
ww1, ww2 = np.meshgrid(w1, w2)
objective = ww1 ** 2 + ww2 ** 2
constraint1 = ww2 <= 0.5
constraint2 = ww1 + ww2 >= 2

fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
ax = axes[0, 0]
ax.contour(ww1, ww2, objective)
ax.scatter(ww1[constraint1], ww2[constraint1], alpha=0.1)
ax.scatter(ww1[constraint2], ww2[constraint2], alpha=0.1)
ax.axis("scaled")
ax.grid()
plt.show()
