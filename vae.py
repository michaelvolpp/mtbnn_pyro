from typing import Optional

import numpy as np
import pyro
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from pyro import distributions as dist
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.optim import ClippedAdam
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch.distributions import Independent


class MLP(PyroModule):
    def __init__(self, d_in: int, d_out: int, n_hidden: int, d_hidden: int):
        super().__init__()

        assert n_hidden > 0
        modules = []
        modules.append(PyroModule[Linear](in_features=d_in, out_features=d_hidden))
        modules.append(PyroModule[ReLU]())
        for _ in range(n_hidden - 1):
            modules.append(
                PyroModule[Linear](in_features=d_hidden, out_features=d_hidden)
            )
            modules.append(PyroModule[ReLU]())
        modules.append(PyroModule[Linear](in_features=d_hidden, out_features=d_out))
        self.mlp = PyroModule[Sequential](*modules)

    def forward(self, x):
        return self.mlp(x)


class VAE(PyroModule):
    def __init__(
        self,
        d_x: int,
        d_z: int,
        n_hidden: int,
        d_hidden: int,
        noise_stddev: Optional[float] = None,
    ):
        super().__init__()

        # dimensions
        self.d_x = d_x
        self.d_z = d_z

        # decoder
        self.decoder = MLP(d_in=d_z, d_out=d_x, n_hidden=n_hidden, d_hidden=d_hidden)

        # prior
        self._z_prior = dist.Normal(0.0, 1.0).expand([self.d_z]).to_event(1)
        self.z = PyroSample(self._z_prior)
        self.last_N = None

        # noise
        self.noise_stddev_prior = (
            dist.Uniform(0.0, 1.0) if noise_stddev is None else None
        )
        self.noise_stddev = (
            PyroSample(self.noise_stddev_prior)
            if noise_stddev is None
            else noise_stddev
        )

    def forward(self, x):
        noise_stddev = self.noise_stddev  # sample noise
        self.last_N = x.shape[0]
        with pyro.plate("data", x.shape[0]):
            z = self.z  # sample x.shape[0] independent latent variables from prior
            x_gen = self.decoder(z)  # generate corresponding observations
            pyro.sample(
                "obs",
                dist.Normal(x_gen, noise_stddev).to_event(1),
                obs=x,
            )  # score x_gen against x

        return x_gen

    def median(self):
        # convenience function s.t. self.median() behaves as guide.median()
        if self.last_N is None:
            return None

        result = {}
        result["z"] = self._z_prior.expand([self.last_N]).mean.detach()
        if self.noise_stddev_prior is not None:
            result["noise_stddev"] = self.noise_stddev_prior.mean.detach()

        return result

    def quantiles(self, quantiles: list):
        # convenience function s.t. self.quantiles() behaves as guide.quantiles()
        if self.last_N is None:
            return None

        result = {}
        z_prior = self._z_prior.expand([self.last_N]).base_dist
        result["z"] = torch.stack(
            [z_prior.icdf(torch.tensor(q)).detach() for q in quantiles], dim=0
        )
        if self.noise_stddev_prior is not None:
            result["noise_stddev"] = torch.stack(
                [
                    self.noise_stddev_prior.icdf(torch.tensor(q)).detach()
                    for q in quantiles
                ],
                dim=0,
            )

        return result


def generate_spiral_data(
    n_points: int,
    noise_stddev: float,
    angle_low: float = 0.0,
    angle_high: float = np.pi,
    spiral: bool = False,
):
    angles = np.random.uniform(low=angle_low, high=angle_high, size=(n_points,))
    x1 = np.cos(angles)
    x2 = np.sin(angles)
    if spiral:
        x1, x2 = angles * x1, angles * x2
    x1 = x1 + noise_stddev * np.random.randn(n_points)
    x2 = x2 + noise_stddev * np.random.randn(n_points)
    x = np.stack((x1, x2), axis=1)
    return x


def predict(model, guide, x: torch.tensor, n_samples: int):
    predictive = Predictive(
        model=model,
        guide=guide,
        num_samples=n_samples,
        return_sites=(
            "z",
            "obs",
            "eps",
            "_RETURN",
        ),
    )
    # shape of samples: (n_samples, n_points_train, d_x/d_z)
    svi_samples = predictive(x=x)
    svi_samples = {
        k: v.reshape(n_samples * x.shape[0], -1) for k, v in svi_samples.items()
    }

    return svi_samples


def plot_x(x, ax, log_p_z=None, label=None):
    x = x.numpy()
    log_p_z = log_p_z.numpy() if log_p_z is not None else None
    sns.scatterplot(
        x=x[:, 0], y=x[:, 1], hue=log_p_z, hue_norm=(-10, 2), ax=ax, label=label
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def plot_z_samples(z, ax, label=None):
    z = z.numpy()
    d_z = z.shape[-1] if z.ndim > 1 else 1
    if d_z == 1:
        sns.distplot(x=z, ax=ax, label=label)
        ax.set_xlabel("z")
    elif d_z == 2:
        sns.kdeplot(x=z[:, 0], y=z[:, 1], ax=ax, label=label, fill=False)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
    else:
        return


def plot_latent_dist(means, stddevs, ax, label=None):
    assert means.ndim == stddevs.ndim
    d_z = means.shape[-1]
    if d_z == 1:
        for i in range(means.shape[0]):
            cur_mean, cur_std = means[i], stddevs[i]
            cur_low = (cur_mean - 5 * cur_std).squeeze()
            cur_high = (cur_mean + 5 * cur_std).squeeze()
            cur_z = torch.linspace(cur_low, cur_high, 100)
            cur_p = torch.exp(
                torch.distributions.Normal(cur_mean, cur_std).log_prob(cur_z)
            )
            line = ax.plot(
                cur_z.numpy(),
                cur_p.numpy(),
                color=line.get_color() if i > 0 else None,
                label=label if i == 0 else None,
            )[0]
            ax.set_xlabel("z")
    elif d_z == 2:
        color = next(ax._get_lines.prop_cycler)["color"]
        for i in range(means.shape[0]):
            cur_mean, cur_std = means[i].numpy(), stddevs[i].numpy()
            cur_ellipse = Ellipse(
                cur_mean,
                width=2 * cur_std[0],  # ellipse half width = 1 stddevs
                height=2 * cur_std[1],  # ellipse half height = 1 stddevs
                facecolor="none",
                edgecolor=color,
            )
            ax.add_patch(cur_ellipse)
            if i == 0:  # add legend entry
                ax.plot([], [], color=color, label=label)
            ax.set_xlabel("z1")
            ax.set_ylabel("z2")
    else:
        return


def marginal_latent_log_prob(z, means, stddevs):
    # check shapes
    assert z.ndim == means.ndim == stddevs.ndim
    assert means.shape == stddevs.shape
    assert means.shape[1] == z.shape[1]
    sample_shape = z.shape[0]
    event_shape = z.shape[1]
    batch_shape = means.shape[0]

    normal = torch.distributions.Normal(means, stddevs)
    normal = Independent(normal, 1)
    emp_marg_dist = normal.log_prob(z[:, None, :])
    emp_marg_dist = torch.logsumexp(emp_marg_dist, dim=1)  # sum batch_dim
    emp_marg_dist -= np.log(batch_shape)

    assert emp_marg_dist.shape == (sample_shape,)
    return emp_marg_dist


def plot_marginal_latent_dist(means, stddevs, ax, label=None):
    assert means.ndim == stddevs.ndim == 2
    assert means.shape == stddevs.shape
    d_z = means.shape[-1]
    if d_z == 1:
        low = (means.min() - 5 * stddevs.max()).squeeze()
        high = (means.max() + 5 * stddevs.max()).squeeze()
        z = torch.linspace(low, high, 1000).reshape(1000, 1)
        d = torch.exp(marginal_latent_log_prob(z=z, means=means, stddevs=stddevs))
        ax.plot(z.numpy(), d.numpy(), label=label)
        ax.set_xlabel("z")
    elif d_z == 2:
        color = next(ax._get_lines.prop_cycler)["color"]
        low_z1 = (means[:, 0].min() - 5 * stddevs[:, 0].max()).item()
        high_z1 = (means[:, 0].max() + 5 * stddevs[:, 0].max()).item()
        low_z2 = (means[:, 1].min() - 5 * stddevs[:, 1].max()).item()
        high_z2 = (means[:, 1].max() + 5 * stddevs[:, 1].max()).item()
        z1, z2 = np.mgrid[low_z1:high_z1:100j, low_z2:high_z2:100j]
        z = torch.tensor(np.stack([z1.ravel(), z2.ravel()], axis=1))
        d = torch.exp(marginal_latent_log_prob(z=z, means=means, stddevs=stddevs))
        d = d.reshape(z1.shape)
        ax.contour(z1, z2, d.numpy(), colors=[color])
        ax.plot([], [], color=color, label=label)
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
    else:
        return


def print_posterior_parameters():
    for name in pyro.get_param_store().keys():
        if not "decoder" in name:
            print(
                f"\n\nname  = {name}"
                f"\nshape = {pyro.param(name).shape}"
                f"\nvalue = {pyro.param(name)}"
            )


if __name__ == "__main__":
    # TODO: try MultivariateNormal guide -> correlations between different z samples
    ## constants
    pyro.set_rng_seed(1236)
    smoke_test = False
    # data
    N = 1000
    noise_stddev = 0.01
    # architecture
    d_z = 2
    n_hidden = 1
    d_hidden = 16
    # training
    n_iter_train = 5000 if not smoke_test else 100
    initial_lr = 0.1
    gamma = 0.0001  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / n_iter_train)
    adam = ClippedAdam({"lr": initial_lr, "lrd": lrd})
    # evaluation
    n_samples_pred = 100 if not smoke_test else 1

    # data
    x = torch.tensor(generate_spiral_data(n_points=N, noise_stddev=noise_stddev, spiral=True, angle_high=3*np.pi))

    # perform inference
    vae = VAE(d_x=2, d_z=d_z, n_hidden=n_hidden, d_hidden=d_hidden, noise_stddev=None)
    vae.train()
    guide = AutoDiagonalNormal(model=vae)
    svi = SVI(model=vae, guide=guide, optim=adam, loss=Trace_ELBO())
    pyro.clear_param_store()
    for n in range(n_iter_train):
        elbo = svi.step(x=x)
        if n % 100 == 0:
            print(f"[iter {n:04d}] elbo = {elbo:.4f}")

    # print posterior parameters
    print("***** Posterior Parameters *****")
    print_posterior_parameters()
    print("********************************")

    # generate data
    vae.eval()
    prior_samples = predict(model=vae, guide=None, x=x, n_samples=n_samples_pred)
    posterior_samples = predict(model=vae, guide=guide, x=x, n_samples=n_samples_pred)

    # obtain prior and posterior distribution parameters
    prior_means = vae.median()["z"]
    posterior_means = guide.median()["z"]
    one_sigma_q = dist.Normal(0.0, 1.0).cdf(torch.tensor(1.0)).item()
    prior_stddevs = vae.quantiles([one_sigma_q])["z"].squeeze(0) - prior_means
    posterior_stddevs = guide.quantiles([one_sigma_q])["z"].squeeze(0) - posterior_means

    # plot predictions
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("VAE Visualization")

    ax = plt.subplot(2, 3, 1)
    ax.set_title(f"Training Data\n(N = {N})")
    plot_x(x=x, log_p_z=None, ax=ax)
    ax.grid()

    ax = plt.subplot(2, 3, 2, sharey=ax, sharex=ax)
    ax.set_title(f"Prior Data Samples\n(N = {N} * {n_samples_pred})")
    p_post_of_prior_samples = marginal_latent_log_prob(
        prior_samples["z"], posterior_means, posterior_stddevs
    )
    plot_x(prior_samples["_RETURN"], log_p_z=p_post_of_prior_samples, ax=ax)
    ax.grid()

    ax = plt.subplot(2, 3, 3, sharey=ax, sharex=ax)
    ax.set_title(f"Posterior Data Samples\n(N = {N} * {n_samples_pred})")
    p_post_of_posterior_samples = marginal_latent_log_prob(
        posterior_samples["z"], posterior_means, posterior_stddevs
    )
    plot_x(posterior_samples["_RETURN"], log_p_z=p_post_of_posterior_samples, ax=ax)
    ax.grid()

    # plot distributions
    ax = plt.subplot(2, 3, 4)
    ax.set_title(f"Latent samples + KDE\n(N = {N} * {n_samples_pred})")
    plot_z_samples(prior_samples["z"], ax=ax, label="Prior")
    plot_z_samples(posterior_samples["z"], ax=ax, label="Posterior")
    ax.legend()
    ax.grid()

    if d_z == 1:
        ax = plt.subplot(2, 3, 5)
    else:
        ax = plt.subplot(2, 3, 5, sharex=ax, sharey=ax)
    ax.set_title("'Marginal' Latent Distribution")
    plot_marginal_latent_dist(
        means=prior_means,
        stddevs=prior_stddevs,
        ax=ax,
        label="Prior",
    )
    plot_marginal_latent_dist(
        means=posterior_means,
        stddevs=posterior_stddevs,
        ax=ax,
        label="Posterior",
    )
    ax.legend()
    ax.grid()

    if d_z == 1:
        ax = plt.subplot(2, 3, 6)
    else:
        ax = plt.subplot(2, 3, 6, sharex=ax, sharey=ax)
    ax.set_title(f"Latent Distributions\n(N = {N}, plotted into one CoSy)")
    plot_latent_dist(
        means=prior_means,
        stddevs=prior_stddevs,
        ax=ax,
        label="Prior",
    )
    plot_latent_dist(
        means=posterior_means,
        stddevs=posterior_stddevs,
        ax=ax,
        label="Posterior",
    )
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()

# TODO: color coding according to p_marginal(z)
# TODO: 2D plotting
# TODO: likelihood computation
