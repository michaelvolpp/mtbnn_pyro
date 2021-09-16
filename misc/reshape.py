import torch

a = torch.rand((7, 5, 4))
print(a)
b = a.reshape((7, 5, 2, 2))
print(b)
