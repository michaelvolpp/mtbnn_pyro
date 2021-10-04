from pyro import distributions as dist
import torch

d1 = dist.Normal(torch.tensor([0.0, 0.5]), torch.tensor([1.0, 0.5]))
d2 = dist.MultivariateNormal(
    torch.tensor([0.0, 0.5]), scale_tril=torch.tensor([[1.0, 0.0], [0.9, 1.0]])
)
print(d1.sample())
print(d2.sample())
print(d1.log_prob(torch.tensor([1.0, 2.0])))
print(d2.log_prob(torch.tensor([1.0, 2.0])))
print(d1.event_dim)
print(d2.event_dim)
