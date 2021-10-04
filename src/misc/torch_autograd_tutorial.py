# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

import torch
import torchvision

# ## standard torch NN training workflow
# # load resnet and create random 3x64x64 image and corresponding random label
# model = torchvision.models.resnet18(pretrained=True)
# data = torch.rand(1, 3, 64, 64)
# labels = torch.rand(1, 1000)

# # forward pass
# prediction = model(data)

# # compute loss and do backward pass
# loss = (prediction - labels).sum()
# loss.backward()

# # take optimizer step
# optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# optim.step()

# ## autograd details
# # define parameters x = (x1, x2)
# x1 = torch.tensor([2.0, 3.0], requires_grad=True)
# x2 = torch.tensor([6.0, 4.0], requires_grad=True)
# print(f"After defining x1 and x2:")
# print(f"x1.grad    = {x1.grad}")
# print(f"x2.grad    = {x2.grad}")
# print(f"x1.grad_fn = {x1.grad_fn}")
# print(f"x2.grad_fn = {x2.grad_fn}")
# # define a function depending on x: y = f(x)
# y = 3 * x1 ** 2 - x2 ** 2
# print(f"After computing y = f(x1, x2):")
# print(f"y         = {y}")
# print(f"x1.grad    = {x1.grad}")
# print(f"x2.grad    = {x2.grad}")
# print(f"x1.grad_fn = {x1.grad_fn}")
# print(f"x2.grad_fn = {x2.grad_fn}")
# # we want to compute the Jacobian J = dy / dx
# # autograd computes J^T * dl / dy = dy / dx * dl / dy = dl / dy
# # we have to provide dl / dy as the gradient argument to backward
# # here, l = g(y) = y, i.e., dl / dy = 1 -> J = dy / dx
# ext_grad = torch.tensor([1.0, 1.0])  # has to be provided because Q is vector-valued
# y.backward(gradient=ext_grad)
# print(f"After calling y.backward():")
# print(f"x1.grad    = {x1.grad}")
# print(f"6 * x1     = {6 * x1}")
# print(f"x2.grad    = {x2.grad}")
# print(f"-2 * x2    = {-2 * x2}")
# print(f"x1.grad_fn = {x1.grad_fn}")
# print(f"x2.grad_fn = {x2.grad_fn}")

## another test 
x = torch.tensor([-2.0], requires_grad=False)
p = torch.tensor([3.0], requires_grad=True)
y = p * x
l = (p * 2.0 - y) ** 2
print(" *** Before y.backward() ***")
print(f"x.requires_grad = {x.requires_grad}")
print(f"y.requires_grad = {y.requires_grad}")
print(f"l.requires_grad = {l.requires_grad}")
print(f"p.requires_grad = {p.requires_grad}")
print(f"x.is_leaf       = {x.is_leaf}")
print(f"y.is_leaf       = {y.is_leaf}")
print(f"l.is_leaf       = {l.is_leaf}")
print(f"p.is_leaf       = {p.is_leaf}")
print(f"x.grad_fn       = {x.grad_fn}")
print(f"y.grad_fn       = {y.grad_fn}")
print(f"l.grad_fn       = {l.grad_fn}")
print(f"p.grad_fn       = {p.grad_fn}")
print(f"x.grad          = {x.grad}")
print(f"y.grad          = {y.grad}")
print(f"l.grad          = {l.grad}")
print(f"p.grad          = {p.grad}")
l.backward()
print(" *** After y.backward() ***")
print(f"x.requires_grad = {x.requires_grad}")
print(f"y.requires_grad = {y.requires_grad}")
print(f"l.requires_grad = {l.requires_grad}")
print(f"p.requires_grad = {p.requires_grad}")
print(f"x.is_leaf       = {x.is_leaf}")
print(f"y.is_leaf       = {y.is_leaf}")
print(f"l.is_leaf       = {l.is_leaf}")
print(f"p.is_leaf       = {p.is_leaf}")
print(f"x.grad_fn       = {x.grad_fn}")
print(f"y.grad_fn       = {y.grad_fn}")
print(f"l.grad_fn       = {l.grad_fn}")
print(f"p.grad_fn       = {p.grad_fn}")
print(f"x.grad          = {x.grad}")
print(f"y.grad          = {y.grad}")
print(f"l.grad          = {l.grad}")
print(f"p.grad          = {p.grad}")
