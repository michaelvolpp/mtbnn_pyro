from typing import Optional

import numpy as np
import pyro
import torch
from matplotlib import pyplot as plt
from metalearning_benchmarks import Affine1D, Linear1D, Quadratic1D
from pyro import distributions as dist
from pyro import poutine
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
from pyro.optim import ClippedAdam
from torch import nn
from torch.distributions import constraints


class BNN(PyroModule):
    def __init__(
        self,
        d_x: int,
        d_y: int,
        n_hidden: int,
        prior_type: str = "diagonal",
        d_hidden: Optional[int] = None,
        noise_stddev: Optional[float] = None,
    ):
        super().__init__()
        assert prior_type == "diagonal"

        # create BNN
        modules = create_bnn_modules(
            d_in=d_x, d_out=d_y, n_hidden=n_hidden, d_hidden=d_hidden
        )
        self.bnn = PyroModule[nn.Sequential](*modules)
        self.noise_stddev_prior = (
            dist.Uniform(0.0, 1.0) if noise_stddev is None else None
        )
        self.noise_stddev = (
            PyroSample(self.noise_stddev_prior)
            if noise_stddev is None
            else noise_stddev
        )

    def forward(self, x, y=None):
        # TODO: vectorize model -> torch.Linear does not work with batched weights
        # shape
        assert x.ndim == 2
        if y is not None:
            assert y.ndim == 2
        n_points = x.shape[0]

        noise_stddev = self.noise_stddev
        mean = self.bnn(x)
        with pyro.plate("data", n_points):
            if noise_stddev.nelement() > 1:
                # noise stddev can have a sample dimension! -> expand to mean's shape
                noise_stddev = noise_stddev.reshape([-1] + [1] * (mean.ndim - 1))
                noise_stddev = noise_stddev.expand(mean.shape)
            obs = pyro.sample("obs", dist.Normal(mean, noise_stddev).to_event(1), obs=y)
        return mean


def create_bnn_modules(
    d_in: int, d_out: int, n_hidden: int, d_hidden: Optional[int] = None
):
    is_linear_model = n_hidden == 0
    if is_linear_model:
        assert d_hidden is None
        d_hidden = d_out

    modules = []

    # input layer
    input_layer = PyroModule[nn.Linear](d_in, d_hidden)
    input_layer.weight = PyroSample(
        dist.Normal(torch.zeros(d_hidden, d_in), torch.ones(d_hidden, d_in))
        .expand([d_hidden, d_in])
        .to_event(2)
    )
    input_layer.bias = PyroSample(
        dist.Normal(torch.zeros(d_hidden), torch.ones(d_hidden))
        .expand([d_hidden])
        .to_event(1)
    )
    modules.append(input_layer)
    if is_linear_model:
        return modules
    modules.append(PyroModule[nn.Tanh]())

    # hidden layers
    for _ in range(n_hidden - 1):
        hidden_layer = PyroModule[nn.Linear](d_hidden, d_hidden)
        hidden_layer.weight = PyroSample(
            dist.Normal(torch.zeros(d_hidden, d_hidden), torch.ones(d_hidden, d_hidden))
            .expand([d_hidden, d_hidden])
            .to_event(2)
        )
        hidden_layer.bias = PyroSample(
            dist.Normal(torch.zeros(d_hidden), torch.ones(d_hidden))
            .expand([d_hidden])
            .to_event(1)
        )
        modules.append(hidden_layer)
        modules.append(PyroModule[nn.Tanh]())

    # output layer
    output_layer = PyroModule[nn.Linear](d_hidden, d_out)
    output_layer.weight = PyroSample(
        dist.Normal(torch.zeros(d_out, d_hidden), torch.ones(d_out, d_hidden))
        .expand([d_out, d_hidden])
        .to_event(2)
    )
    output_layer.bias = PyroSample(
        dist.Normal(torch.zeros(d_out), torch.ones(d_out)).expand([d_out]).to_event(1)
    )
    modules.append(output_layer)

    return modules


def compute_log_likelihood_vectorized(model, guide, x, y, n_samples):
    """
    Computes predictive log-likelihood using latent samples from guide using Predictive.
    """
    # obtain vectorized model trace
    x, y = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    assert x.ndim == 2
    predictive = Predictive(
        model=model,
        guide=guide,
        num_samples=n_samples,
        parallel=True,
    )
    model_trace = predictive.get_vectorized_trace(x=x, y=y)

    # compute log-likelihood for the observation sites
    obs_site = model_trace.nodes["obs"]
    log_prob = obs_site["fn"].log_prob(obs_site["value"])  # reduces event-dims
    n_pts = x.shape[0]
    assert log_prob.shape == (n_samples, n_pts)

    # compute predictive likelihood
    log_prob = torch.sum(log_prob, dim=1, keepdim=True)  # sum pts-per-task dim
    log_prob = torch.logsumexp(log_prob, dim=0, keepdim=True)  # reduce sample dim
    assert log_prob.shape == (1, 1)
    log_prob = log_prob.squeeze_()
    log_prob = log_prob - torch.log(torch.tensor(n_samples))

    # normalize w.r.t. number of datapoints
    log_prob = log_prob / n_pts

    return log_prob


def compute_log_likelihood(model, guide, x, y, n_samples):
    """
    Computes predictive log-likelihood using latent samples from guide using Predictive.
    TODO: use vectorized version!
    """
    x, y = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    assert x.ndim == 2
    n_pts = x.shape[0]

    # sequentially compute the log probs
    log_prob = torch.zeros((n_samples, n_pts))
    for s in range(n_samples):
        # obtain model trace
        if guide is not None:
            guide_trace = poutine.trace(guide).get_trace(x=x, y=y)
            replayed_model = poutine.replay(model, trace=guide_trace)
            model_trace = poutine.trace(replayed_model).get_trace(x=x, y=y)
        else:
            model_trace = poutine.trace(model).get_trace(x=x, y=y)
        # compute log-likelihood for the observation sites
        obs_site = model_trace.nodes["obs"]
        log_prob[s] = obs_site["fn"].log_prob(obs_site["value"])  # reduces event-dims

    # compute predictive likelihood
    log_prob = torch.sum(log_prob, dim=1, keepdim=True)  # sum pts-per-task dim
    log_prob = torch.logsumexp(log_prob, dim=0, keepdim=True)  # reduce sample dim
    assert log_prob.shape == (1, 1)
    log_prob = log_prob.squeeze_()
    log_prob = log_prob - torch.log(torch.tensor(n_samples))

    # normalize w.r.t. number of datapoints
    log_prob = log_prob / n_pts

    return log_prob


def train_model(model, guide, x, y, n_iter, initial_lr, final_lr=None):
    model.train()

    # optimizer
    optim_args = {}
    optim_args["lr"] = initial_lr
    if final_lr is not None:
        gamma = final_lr / initial_lr  # final learning rate will be gamma * initial_lr
        optim_args["lrd"] = gamma ** (1 / n_iter)
    optim = ClippedAdam(optim_args=optim_args)

    # SVI
    svi = SVI(model=model, guide=guide, optim=optim, loss=Trace_ELBO())

    # training loop
    pyro.clear_param_store()
    for i in range(n_iter):
        elbo = -svi.step(
            x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y, dtype=torch.float)
        )
        if i % 100 == 0 or i == len(range(n_iter)) - 1:
            print(f"[iter {i:04d}] elbo = {elbo:.4f}")

    model.eval()


def split_task(x, y, n_context):
    x_context, y_context = x[:n_context, :], y[:n_context, :]
    # TODO: use all data as target?
    x_target, y_target = x, y

    return x_context, y_context, x_target, y_target


def plot_metrics(lls, lls_context, n_contexts):
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
    fig.suptitle("Metrics")

    ax = axes[0, 0]
    ax.set_title("Marginal Log-Likelihood")
    ax.set_xlabel("n_context")
    ax.set_xticks(n_contexts)
    ax.set_ylabel("marginal ll")
    ax.plot(n_contexts, lls, label="all data")
    ax.plot(n_contexts, lls_context, label="context only")
    ax.legend()
    ax.grid()

    fig.tight_layout()


def main():
    ## flags, constants
    pyro.set_rng_seed(123)
    plot = True
    smoke_test = False 
    # benchmark
    bm = Quadratic1D
    n_points_per_task = 16
    noise_stddev = 0.01
    # model
    n_hidden = 1
    d_hidden = 8
    infer_noise_stddev = True
    prior_type = "diagonal"
    # training
    n_iter = 5000 if not smoke_test else 100
    initial_lr = 0.1
    final_lr = 0.00001
    # evaluation
    n_contexts = (
        np.array([0, 1, 2, 5, 10, n_points_per_task])
        if not smoke_test
        else np.array([0, 5, 10])
    )
    n_pred = 100
    n_samples = 1000 if not smoke_test else 100
    max_plot_tasks = 5

    # test benchmark
    bm_test = bm(
        n_task=1,
        n_datapoints_per_task=n_points_per_task,
        output_noise=noise_stddev,
        seed_task=1235,
        seed_x=1236,
        seed_noise=1237,
    )
    x_pred_test = np.linspace(-1.5, 1.5, n_pred)[:, None]

    # create model
    bnn = BNN(
        d_x=bm.d_x,
        d_y=bm.d_y,
        n_hidden=n_hidden,
        d_hidden=d_hidden,
        noise_stddev=None if infer_noise_stddev else noise_stddev,
        prior_type=prior_type,
    )
    bnn.eval()

    ## do inference on test task
    lls = np.zeros(n_contexts.shape)
    lls_context = np.zeros(n_contexts.shape)
    guides = []
    task = bm_test.get_task_by_index(0)
    for i, n_context in enumerate(n_contexts):
        print("\n**************************************************************")
        print(
            f"*** Performing inference on test task (n_context = {n_context:3d})... ***"
        )
        print("**************************************************************")
        x_context, y_context, x_target, y_target = split_task(
            x=task.x, y=task.y, n_context=n_context
        )
        if n_context != 0:
            cur_guide = AutoDiagonalNormal(model=bnn)
            train_model(
                model=bnn,
                guide=cur_guide,
                x=x_context,
                y=y_context,
                n_iter=n_iter,
                initial_lr=initial_lr,
                final_lr=final_lr,
            )
        else:
            cur_guide = None
        guides.append(cur_guide)
        lls[i] = compute_log_likelihood(
            model=bnn,
            guide=cur_guide,
            x=x_target,
            y=y_target,
            n_samples=n_samples,
        )
        if n_context != 0:
            lls_context[i] = compute_log_likelihood(
                model=bnn,
                guide=cur_guide,
                x=x_context,
                y=y_context,
                n_samples=n_samples,
            )
        else:
            lls_context[i] = np.nan
        print("*******************************")

    ## plot
    plot_metrics(lls=lls, lls_context=lls_context, n_contexts=n_contexts)
    plt.show()


if __name__ == "__main__":
    main()
