import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns

# seed
torch.manual_seed(123)

# generate Gaussian
N = 2
sigma_n = 0.001
sigma_z = 0.1
mu = torch.arange(N, dtype=torch.float)
# Sigma = torch.eye(L) * sigma_n ** 2 + torch.ones(L) * sigma_z ** 2
theta = math.pi / 4
R = torch.tensor(
    [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
)
Sigma = torch.diag(torch.tensor([sigma_n ** 2 + N * sigma_z ** 2, sigma_n ** 2]))
Sigma = R @ Sigma @ R.T
evals, evecs = torch.linalg.eig(Sigma)
largest_eval = torch.max(torch.abs(evals))

# sample
L = 10000
gaussian = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=Sigma)
samples = gaussian.sample_n(n=L)
flattened_samples = samples.flatten()
std_y = flattened_samples.std()
print(f"std_y                          = {std_y:.4f}")
print(f"semi-major axis                = {torch.sqrt(largest_eval):.4f}")
print(
    f"projection of semi-major axis  = {math.cos(math.pi/4)*torch.sqrt(largest_eval):.4f}"
)
print(f"sqrt diagonal entries of Sigma = {torch.sqrt(Sigma[0,0]):.4f}")


# plot
fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(16, 8))
ax = axes[0, 0]
if N == 2:
    # plot samples
    ax.scatter(x=samples[:, 0], y=samples[:, 1], marker="x", s=1)

    # plot 1-sigma contours
    phis = torch.linspace(0.0, 2.0 * torch.pi, 100)
    x = torch.sqrt(evals[0]) * torch.cos(phis)
    y = torch.sqrt(evals[1]) * torch.sin(phis)
    xy = torch.row_stack((x, y))
    R = torch.row_stack((evecs[0].T, evecs[1].T))
    xy = R.matmul(xy)
    xy = (xy.T + mu).T
    ax.plot(xy[0, :], xy[1, :], "r")
    ax.scatter(x=mu[0], y=mu[1], color="r", marker="x", s=25)
    ax.grid()

# plot "flattened samples"
ax = axes[0, 1]
flattened_samples = samples.flatten()
ax.scatter(x=samples.flatten(), y=torch.zeros(flattened_samples.shape), marker="x", s=1)
sns.kdeplot(flattened_samples, ax=ax)
ax.grid()

# plot function
ax = axes[0, 2]
max_samples = 10
ax.plot(torch.arange(N), samples[:max_samples].T, "b", alpha=0.3)
ax.plot(torch.arange(N), mu.T, "r")
ax.grid()

fig.tight_layout()
plt.show()
