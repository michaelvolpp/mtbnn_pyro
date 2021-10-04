import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = a
b = b[None, :, :]

print(a.shape)
print(b.shape)
