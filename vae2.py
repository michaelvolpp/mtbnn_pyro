import numpy as np
import pyro
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from pyro import distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.optim import ClippedAdam
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d
from torch.distributions import constraints


class VAE(PyroModule):
    def __init__(self, d_x: int, d_z: int, n_hidden: int, d_hidden: int, eps: float):
        super().__init__()

        self.d_x = d_x
        self.d_z = d_z

        # generator
        assert n_hidden > 1
        modules = []
        modules.append(PyroModule[Linear](in_features=d_z, out_features=d_hidden))
        modules.append(PyroModule[ReLU]())
        for _ in range(n_hidden - 1):
            modules.append(
                PyroModule[Linear](in_features=d_hidden, out_features=d_hidden)
            )
            modules.append(PyroModule[ReLU]())
        modules.append(PyroModule[Linear](in_features=d_hidden, out_features=d_x))
        self.decoder = PyroModule[Sequential](*modules)

        # noise
        self.eps = eps

    def forward(self, x=None):
        eps = (
            pyro.sample("eps", dist.Uniform(0.0, 1.0)) if self.eps is None else self.eps
        )
        n_data = x.shape[0] if x is not None else 1
        with pyro.plate("data", n_data):
            z_prior_loc = torch.zeros(n_data, self.d_z)
            z_prior_scale = torch.ones(n_data, self.d_z)
            z = pyro.sample("z", dist.Normal(z_prior_loc, z_prior_scale).to_event(1))
            x_gen = self.decoder(z)
            pyro.sample("obs", dist.Normal(x_gen, eps).to_event(1), obs=x)

        return x_gen

    def guide_diag_normal(self, x):
        # TODO: reconstruct results from AutoDiagNormal!
        n_data = x.shape[0]
        if self.eps is None:
            eps_posterior_loc = pyro.param(
                "eps_posterior_loc",
                torch.tensor(0.5),
                constraint=constraints.positive,
            )
            eps_posterior_scale = pyro.param(
                "eps_posterior_scale",
                torch.tensor(0.1),
                constraint=constraints.positive,
            )
            eps_posterior = pyro.sample(
                "eps", dist.Normal(eps_posterior_loc, eps_posterior_scale)
            )
        with pyro.plate("data", n_data):
            z_posterior_loc = pyro.param(
                "z_posterior_loc",
                torch.zeros(n_data, self.d_z),
                constraint=constraints.real,
            )
            z_posterior_scale = pyro.param(
                "z_posterior_scale",
                0.1 * torch.ones(n_data, self.d_z),
                constraint=constraints.positive,
            )
            z_posterior = pyro.sample(
                "z", dist.Normal(z_posterior_loc, z_posterior_scale).to_event(1)
            )


def generate_circle_data(n_points, noise):
    angles = np.random.uniform(low=0, high=1 * np.pi, size=(n_points,))
    # x1 = angles * np.cos(angles) + noise * np.random.randn(n_points)
    # x2 = angles * np.sin(angles) + noise * np.random.randn(n_points)
    x1 = np.cos(angles) + noise * np.random.randn(n_points)
    x2 = np.sin(angles) + noise * np.random.randn(n_points)
    x = np.stack((x1, x2), axis=1)
    return x


def predict(model, guide, x, n_samples: int):
    # TODO: understand shapes of svi_samples
    predictive = Predictive(
        model=model,
        guide=guide,
        num_samples=n_samples,
        return_sites=(
            "z",
            "obs",
            "_RETURN",
        ),
    )
    svi_samples = predictive(x=x)
    svi_samples = {k: v for k, v in svi_samples.items()}

    return svi_samples


def plot_x(x, ax, z=None, label=None):
    d_z = z.shape[-1] if z is not None else None
    hue = z if d_z == 1 else None
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=hue, ax=ax, label=label)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid()


def plot_z(z, ax, label=None):
    d_z = z.shape[-1] if z.ndim > 1 else 1
    if d_z == 1:
        sns.distplot(x=z, ax=ax, label=label)
        ax.set_xlabel("z")
    elif d_z == 2:
        sns.kdeplot(x=z[:, 0], y=z[:, 1], ax=ax, label=label, fill=False)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
    else:
        raise ValueError("Plotting of latent distribution only possible for d_z=1,2!")
    ax.grid()


def print_parameters():
    for name, value in pyro.get_param_store().items():
        if not "decoder" in name:
            print(
                f"\n\nname  = {name}"
                f"\nshape = {pyro.param(name).shape}"
                f"\nvalue = {pyro.param(name)}"
            )


if __name__ == "__main__":
    # TODO: try MultivariateNormal guide -> correlations between different z samples
    # constants
    pyro.set_rng_seed(1236)
    smoke_test = True 
    n_train = 1000
    noise = 0.01
    n_iter = 5000 if not smoke_test else 1
    n_pred = 5000 if not smoke_test else 100
    initial_lr = 0.1
    gamma = 0.0001  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / n_iter)
    adam = ClippedAdam({"lr": initial_lr, "lrd": lrd})
    n_hidden = 2
    d_hidden = 128
    d_z = 2

    # data
    x = generate_circle_data(n_points=n_train, noise=noise)

    # perform inference
    vae = VAE(d_x=2, d_z=d_z, n_hidden=n_hidden, d_hidden=d_hidden, eps=noise)
    vae.train()
    # guide = AutoDiagonalNormal(model=vae)
    guide = vae.guide_diag_normal
    svi = SVI(model=vae, guide=guide, optim=adam, loss=Trace_ELBO())
    pyro.clear_param_store()
    print_parameters()
    for n in range(n_iter):
        elbo = svi.step(x=torch.tensor(x))
        print_parameters()
        if n % 100 == 0:
            print(f"[iter {n:04d}] elbo = {elbo:.4f}")

    # generate data
    vae.eval()
    prior_samples = predict(model=vae, guide=None, x=None, n_samples=n_pred)
    # TODO: understand what predict does (n_samples vs. x.shape[0])
    # -> this is the reason we need to squeeze in mtblr.py!
    posterior_samples = predict(model=vae, guide=guide, x=torch.tensor(x), n_samples=1)

    # plot predictions
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(12, 4), sharex=True, sharey=True
    )
    fig.suptitle("Data and Samples")
    ax = axes[0]
    ax.set_title("Data")
    plot_x(x=x, z=None, ax=ax)
    ax = axes[1]
    ax.set_title("Prior Samples")
    plot_x(
        prior_samples["_RETURN"].squeeze().numpy(),
        z=prior_samples["z"].squeeze().numpy(),
        ax=ax,
    )
    ax = axes[2]
    ax.set_title("Posterior Samples")
    plot_x(
        posterior_samples["_RETURN"].squeeze().numpy(),
        z=posterior_samples["z"].squeeze().numpy(),
        ax=ax,
    )

    # plot distributions
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    fig.suptitle("Latent Distribution")
    plot_z(prior_samples["z"].squeeze().numpy(), ax=ax, label="Prior")
    plot_z(posterior_samples["z"].squeeze().numpy(), ax=ax, label="Posterior")
    ax.legend()
    plt.show()
