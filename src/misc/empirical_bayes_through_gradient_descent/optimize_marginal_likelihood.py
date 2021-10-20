from numpy import dtype
import pyro
import torch
from torch.distributions import constraints
from pyro import distributions as dist
from pyro import poutine
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.infer import EmpiricalMarginal, Importance, Predictive
import seaborn as sns
from matplotlib import pyplot as plt
from mtutils.mtutils import print_pyro_parameters

SIGMA_N = 0.01


class Model(PyroModule):
    def __init__(self, init_sigma, correlated):
        super().__init__()

        self.correlated = correlated

        # prior
        self.mu = PyroParam(
            init_value=torch.tensor([0.0]),
            constraint=constraints.real,
        )
        # self.mu.requires_grad = False
        self.sigma = PyroParam(
            init_value=torch.tensor([init_sigma]),
            constraint=constraints.positive,
        )
        self.prior = lambda self: dist.Normal(loc=self.mu, scale=self.sigma).to_event(1)
        self.z = PyroSample(self.prior)

    def forward(self, dataset_size=None, y=None):
        """
        p(y, z) = p(y|z) * p(z)
        p(z) = N(z | mu, sigma^2)
        p(y|z) = N(y | z, sigma_n^2)
        -> marginal mean: 0.0
        -> marginal std: sqrt(sigma^2 + sigma_n^2)
        """
        if dataset_size is None:
            assert y is not None
            dataset_size = y.shape[0]
        if y is not None:
            assert y.ndim == 2
            assert y.shape[0] == dataset_size

        if self.correlated:
            z = self.z
        with pyro.plate("data", size=dataset_size, dim=-1):
            if not self.correlated:
                z = self.z
            likelihood = dist.Normal(loc=z, scale=SIGMA_N).to_event(1)
            obs = pyro.sample("obs", fn=likelihood, obs=y)

        return obs


def predict(model, dataset_size, n_samples):
    predictive = Predictive(
        model=model,
        guide=None,
        num_samples=n_samples,
        parallel=True,
        return_sites=("obs",),
    )
    samples = predictive(dataset_size=dataset_size, y=None)
    samples = {k: v.detach().cpu().numpy() for k, v in samples.items()}

    return samples


def marginal_log_likelihood(model, y, n_samples):
    n_pts = y.shape[0]

    # obtain vectorized model trace
    vectorized_model = pyro.plate("batch", size=n_samples, dim=-2)(model)
    model_trace = poutine.trace(vectorized_model).get_trace(
        y=torch.tensor(y, dtype=torch.float32)
    )

    # compute log-likelihood for the observation sites
    obs_site = model_trace.nodes["obs"]
    log_prob = obs_site["fn"].log_prob(obs_site["value"])  # reduces event-dims
    assert log_prob.shape == (n_samples, n_pts)

    # compute predictive likelihood
    if model.correlated:
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)  # sum pts-per-task dim
        log_prob = torch.logsumexp(log_prob, dim=0, keepdim=True)  # reduce sample dim
        assert log_prob.shape == (1, 1)
        log_prob = log_prob.squeeze()
        log_prob = log_prob - torch.log(torch.tensor(n_samples))
    else:
        log_prob = torch.logsumexp(log_prob, dim=0, keepdim=True)  # reduce sample dim
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)  # sum task dim
        assert log_prob.shape == (1, 1)
        log_prob = log_prob.squeeze()
        log_prob = log_prob - n_pts * torch.log(torch.tensor(n_samples))

    # normalize w.r.t. number of datapoints
    log_prob = log_prob / n_pts

    return log_prob


def true_marginal_log_likelihood(model, y):
    n_pts = y.shape[0]
    if model.correlated:
        mu = model.mu * torch.ones((n_pts,))
        Sigma = model.sigma ** 2 * torch.ones((n_pts, n_pts))
        Sigma = Sigma + SIGMA_N ** 2 * torch.eye(n_pts)
        dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=Sigma)

        log_prob = dist.log_prob(y.squeeze()) / n_pts
    else:
        mu = model.mu
        sigma = torch.sqrt(model.sigma ** 2 + SIGMA_N ** 2)
        dist = torch.distributions.Normal(loc=mu, scale=sigma)

        log_prob = dist.log_prob(y.squeeze()).sum() / n_pts

    return log_prob


def plot_samples(samples, ax, label):
    sns.kdeplot(x=samples, ax=ax, label=label)


def main():
    pyro.set_rng_seed(123)
    # data
    true_mean = 0.0
    true_std = 1.0
    dataset_size = 100
    y = torch.randn((dataset_size, 1)) * true_std + true_mean
    y = y + torch.randn((dataset_size, 1)) * SIGMA_N
    print(f"true mean = {y.mean():.4f}")
    print(f"true std  = {y.std():.4f}")

    # model
    correlated = False

    # # TODO: understand this!
    # print("*" * 100)
    # model = Model(init_sigma=1.0, correlated=correlated)
    # print(
    #     f"model1-ll (sampled) = {marginal_log_likelihood(model=model, y=y, n_samples=10000)}"
    # )
    # print(f"model1-ll (true)    = {true_marginal_log_likelihood(model=model, y=y)}")
    # print_pyro_parameters()
    # pyro.clear_param_store()
    # print("*" * 100)

    # print("*" * 100)
    # model = Model(init_sigma=0.1, correlated=correlated)
    # print(
    #     f"model2-ll (sampled) = {marginal_log_likelihood(model=model, y=y, n_samples=1000000)}"
    # )
    # print(f"model2-ll (true)    = {true_marginal_log_likelihood(model=model, y=y)}")
    # print_pyro_parameters()
    # pyro.clear_param_store()
    # print("*" * 100)

    # model
    model = Model(init_sigma=1.0, correlated=correlated)

    # plot prior prediction
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
    ax = axes[0, 0]
    samples = predict(model=model, dataset_size=dataset_size, n_samples=10000)
    samples = samples["obs"].squeeze().flatten()
    plot_samples(y.squeeze(), ax=ax, label="data")
    plot_samples(samples, ax=ax, label="before training")
    print(f"predictive mean before training = {samples.mean():.4f}")
    print(f"predictive std before training  = {samples.std():.4f}")

    # optimize
    optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())
    n_epochs = 10000
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = -marginal_log_likelihood(model=model, y=y, n_samples=1000)

        if epoch % 100 == 0:
            # print_pyro_parameters()
            print(
                f"epoch = {epoch:04d}"
                f" | loss = {loss.item():.4f}"
                f" | mu = {model.mu.item():.4f}"
                f" | sigma = {model.sigma.item():.4f}"
            )

        loss.backward()
        optimizer.step()

    # plot prior prediction
    samples = predict(model=model, dataset_size=dataset_size, n_samples=10000)
    samples = samples["obs"].squeeze().flatten()
    plot_samples(samples, ax=ax, label="after training")
    print(f"predictive mean after training = {samples.mean():.4f}")
    print(f"predictive std after training  = {samples.std():.4f}")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
