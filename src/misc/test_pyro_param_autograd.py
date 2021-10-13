import pyro
import torch
from torch.distributions import constraints

a = pyro.param("a", init_tensor=torch.tensor([3.0]), constraint=constraints.real)
b = a ** 2
b.backward()
print(a.grad)
