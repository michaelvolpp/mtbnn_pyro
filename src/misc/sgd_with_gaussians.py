import torch
import math
from matplotlib import pyplot as plt


# # log-sigma space
# mean = torch.tensor(0.0, requires_grad=True)
# log_var = torch.tensor(math.log(init_var), requires_grad=True)
# theta = torch.tensor(0.5, requires_grad=True)
# gaussian = torch.distributions.Normal(mean, torch.sqrt(torch.exp(log_var)))
# log_prob = gaussian.log_prob(theta)
# log_prob.backward()
# print(mean.grad)
# print(theta.grad)
# print(log_var.grad)
# print(torch.exp(log_var) ** (-1) * (theta - mean))
# print(-(torch.exp(log_var) ** (-1)) * (theta - mean))
# print(-0.5 + 0.5 * torch.exp(log_var) ** (-1) * (theta - mean) ** 2)

# # sigma space
# mean = torch.tensor(init_mean, requires_grad=True)
# var = torch.tensor(init_var, requires_grad=True)
# theta = torch.tensor(init_theta, requires_grad=True)
# gaussian = torch.distributions.Normal(mean, torch.sqrt(var))
# log_prob = gaussian.log_prob(theta)
# log_prob.backward()
# print(mean.grad)
# print(theta.grad)
# print(var.grad)
# print(init_var)
# print(init_beta)
# print(var ** (-1) * init_beta)
# print(-(var ** (-1)) * init_beta)
# print(0.5 * var ** (-1) * (init_beta ** 2 - 1))

# init_std = 1e-6
# init_mean = 1.0
# init_theta = 3.0
# init_var = init_std ** 2
# init_beta = (init_theta - init_mean) / init_std
# theta = torch.tensor(init_theta, requires_grad=False)
# mean = torch.tensor(init_mean, requires_grad=False)
# var = torch.tensor(init_std ** 2, requires_grad=True)
# # optim = torch.optim.LBFGS(lr=0.01, params=[mean, var])
# # optim = torch.optim.Adam(lr=0.01, params=[mean, var])
# optim = torch.optim.SGD(lr=0.1, params=[mean, var])
# sigmas = []
# betas = []
# iters = []
# n_iter = 100000
# for i in range(n_iter):

#     def closure():
#         optim.zero_grad()
#         log_prob = torch.distributions.Normal(
#             loc=mean,
#             scale=torch.sqrt(var),
#         ).log_prob(theta)
#         loss = -log_prob
#         loss.backward()
#         return loss

#     sigma = (torch.sqrt(var)).item()
#     beta = (theta - mean).item()
#     sigmas.append(sigma)
#     betas.append(beta)
#     iters.append(i)
#     try:
#         loss = closure()
#         optim.step(closure)
#     except ValueError:
#         print(f"Value error after {i:0d} iterations! Final loss = {loss:.4f}")
#         break
#     if i % 100 == 0:
#         print(f"iter = {i:0d} | loss = {loss:.4f}")

# sigma = (torch.sqrt(var)).item()
# beta = (theta - mean).item()
# iters.append(i)
# sigmas.append(sigma)
# betas.append(beta)

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), squeeze=False)
# ax = axes[0, 0]
# ax.scatter(betas, sigmas, c=iters)
# ax.plot(torch.linspace(0.0, 10.0), torch.linspace(0.0, 10.0))
# plt.show()

# generate data
torch.manual_seed(123)
n_data = 100
sigma_n = 0.1
mu_true = 1.0
thetas = mu_true + sigma_n * torch.randn((n_data, 1))
mu_ML = torch.mean(thetas)
var_ML = torch.mean((thetas - mu_ML) ** 2)

# initialize model
mu_init = 3 * mu_true
sigma_init = 1.0
mu = torch.tensor(mu_init, requires_grad=True)
var = torch.tensor(sigma_init ** 2, requires_grad=True)

# optimize
n_iter = 10000
lr = 0.01
# optim = torch.optim.SGD(params=[mu, var], lr=lr)
# optim = torch.optim.LBFGS(params=[mu, var], lr=lr)
optim = torch.optim.Adam(params=[mu, var], lr=lr)
mus = [mu.item()]
vars = [var.item()]
iters = [0]
for i in range(n_iter):

    def closure():
        optim.zero_grad()
        log_probs = torch.distributions.Normal(
            loc=mu,
            scale=torch.sqrt(var),
        ).log_prob(thetas)
        loss = -log_probs.sum()
        # cf. https://openreview.net/pdf?id=aPOpXlnV1T
        # sigma = sigma_n if isinstance(sigma_n, float) else sigma_n.detach()
        # loss = loss * sigma**2
        loss.backward()
        return loss

    try:
        loss = closure()
        optim.step(closure)
    except ValueError:
        print(f"Value error after {i:0d} iterations! Final loss = {loss:.4f}")
        break

    # log
    mus.append(mu.item())
    vars.append(var.item())
    iters.append(i + 1)
    if i % 100 == 0:
        print(f"iter = {i:0d} | loss = {loss:.4f}")


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), squeeze=False)
ax = axes[0, 0]
ax.scatter(mus, vars, c=iters)
ax.axvline(mu_ML)
ax.axhline(var_ML)
ax.grid()
ax.set_xlabel("mu")
ax.set_ylabel("var")
plt.show()
