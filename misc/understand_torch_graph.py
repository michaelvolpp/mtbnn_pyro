import torch
from torchviz import make_dot

x1 = torch.tensor(1.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True)
f = x1 * x2 + torch.sin(x1)
make_dot(f, params={"x1": x1, "x2": x2}, show_attrs=True, show_saved=True).render()
