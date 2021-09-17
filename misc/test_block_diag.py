import torch

a = torch.rand(3, 3)
b = torch.rand(2, 2)
c = torch.block_diag(a, b)
print(a)
print(b)
print(c)
