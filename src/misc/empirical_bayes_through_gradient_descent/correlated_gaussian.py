import torch
import matplotlib.pyplot as plt
import math
import seaborn as sns

# seed
torch.manual_seed(123)

# generate Gaussian
D = 20
sigma_n = 0.2
sigma_z = 2.0
mu = torch.zeros(D)
Sigma = torch.eye(D) * sigma_n ** 2 + torch.ones(D) * sigma_z ** 2
gaussian = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=Sigma)

# sample
S = 10000
samples = gaussian.sample_n(n=S)
flattened_samples = samples.flatten()
evals, evecs = torch.linalg.eig(Sigma)
largest_eval = torch.max(torch.abs(evals))
std_y = flattened_samples.std()
print(f"std_y                          = {std_y:.4f}")
print(f"semi-major axis                = {torch.sqrt(largest_eval):.4f}")
print(
    f"projection of semi-major axis  = {math.cos(math.pi/4)*torch.sqrt(largest_eval):.4f}"
)
print(f"sqrt diagonal entries of Sigma = {torch.sqrt(Sigma[0,0]):.4f}")


# plot
fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
ax = axes[0, 0]
if D == 2:
    # plot samples
    ax.scatter(x=samples[:, 0], y=samples[:, 1], marker="x", s=1)

    # plot 1-sigma contours
    phis = torch.linspace(0.0, 2.0 * torch.pi, 100)
    x = torch.sqrt(evals[0]) * torch.cos(phis)
    y = torch.sqrt(evals[1]) * torch.sin(phis)
    xy = torch.row_stack((x, y))
    R = torch.row_stack((evecs[0].T, evecs[1].T))
    xy = R.matmul(xy)
    ax.plot(xy[0, :], xy[1, :], "r")
    # ax.set_xlim(
    #     [-1.5 * torch.sqrt(evals[0].float()), 1.5 * torch.sqrt(evals[0].float())]
    # )
    # ax.set_ylim(
    #     [-1.5 * torch.sqrt(evals[0].float()), 1.5 * torch.sqrt(evals[0].float())]
    # )
    ax.grid()

# plot "flattened samples"
ax = axes[0, 0]
flattened_samples = samples.flatten()
ax.scatter(x=samples.flatten(), y=torch.zeros(flattened_samples.shape), marker="x", s=1)
sns.kdeplot(flattened_samples, ax=ax)
# ax.grid()

fig.tight_layout()
plt.show()
