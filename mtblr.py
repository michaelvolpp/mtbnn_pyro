import copy
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pyro
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from metalearning_benchmarks import Affine1D, Linear1D, Quadratic1D
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
from numpy.core.fromnumeric import sort
from pyro import distributions as dist
from pyro.distributions import constraints
from pyro.infer import SVI, Predictive, Trace_ELBO, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.autoguide.guides import AutoMultivariateNormal
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.optim import Adam, ClippedAdam
from torch import nn


def prior(obj):
    prior_dist = (
        dist.Normal(obj.weight_loc_prior, obj.weight_scale_prior)
        .expand([obj.out_features, obj.in_features])
        .to_event(2)
    )
    return prior_dist


class BayesianLinear(PyroModule):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight_loc_prior = PyroParam(
            init_value=torch.tensor(0.0),
            constraint=constraints.real,
        )
        self.weight_scale_prior = PyroParam(
            init_value=torch.tensor(1.0),
            constraint=constraints.positive,
        )
        # self.weight_loc_prior = torch.tensor(0.0)
        # self.weight_scale_prior = torch.tensor(1.0)
        # https://docs.pyro.ai/en/dev/nn.html#pyro.nn.module.PyroSample
        # https://forum.pyro.ai/t/getting-estimates-of-parameters-that-use-pyrosample/2901/2
        self.weight = PyroSample(prior)
        self.freeze_parameters()

        if bias:
            self.bias_loc_prior = PyroParam(
                init_value=torch.tensor(0.0),
                constraint=constraints.real,
            )
            self.bias_scale_prior = PyroParam(
                init_value=torch.tensor(1.0),
                constraint=constraints.positive,
            )
            # https://docs.pyro.ai/en/dev/nn.html#pyro.nn.module.PyroSample
            # https://forum.pyro.ai/t/getting-estimates-of-parameters-that-use-pyrosample/2901/2
            self.bias = PyroSample(
                lambda self: dist.Normal(self.bias_loc_prior, self.bias_scale_prior)
                .expand([self.out_features])
                .to_event(1)
            )

    def forward(self, x):
        assert x.ndim == 3
        n_tasks = x.shape[1]
        assert self.weight.shape == (n_tasks, self.out_features, self.in_features)
        assert self.bias.shape == (n_tasks, self.out_features)
        # TODO: why do we need x.float() here as soon as we have more than zero layers?
        y = torch.einsum("lyx,nlx->nly", self.weight, x.float())
        y = y + self.bias[None, :, :]
        return y

    def freeze_parameters(self):
        # self.weight_loc_prior = self.weight_loc_prior.clone().detach()
        self.weight_loc_prior = torch.tensor(0.0, requires_grad=False)
        # self.weight_loc_prior = 0.0


class MTBLR(PyroModule):
    def __init__(self, d_x: int, d_y: int, n_hidden: int, d_hidden: int):
        super().__init__()

        if n_hidden == 1:
            modules = []
            modules.append(
                BayesianLinear(in_features=d_x, out_features=d_hidden, bias=True)
            )
            modules.append(PyroModule[nn.Tanh]())
            modules.append(
                BayesianLinear(in_features=d_hidden, out_features=d_y, bias=True)
            )
            self.mtbnn = PyroModule[nn.Sequential](*modules)
        elif n_hidden == 0:
            self.mtbnn = BayesianLinear(in_features=d_x, out_features=d_y, bias=True)
        else:
            raise NotImplementedError

        # TODO: learn noise prior?
        # self.sigma_upper_bound = PyroParam(
        #     init_value=torch.tensor(1.0),
        #     constraint=constraints.positive,
        # )
        # self.sigma = PyroSample(lambda self: dist.Uniform(0.0, self.sigma_upper_bound))

    def forward(self, x: torch.tensor, d_y: int, y: Optional[torch.tensor] = None):
        check_shapes(x=x, y=y)
        n_points = x.shape[0]
        n_tasks = x.shape[1]
        d_x = x.shape[2]
        if y is not None:
            assert y.shape == (n_points, n_tasks, d_y)

        # wrap this in a softplus?
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 1.0))
        with pyro.plate("tasks", n_tasks):
            mean = self.mtbnn(x)
            assert mean.shape == (n_points, n_tasks, d_y)
            with pyro.plate("data", n_points):
                obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=y)
                assert obs.shape == (n_points, n_tasks, d_y)
        return mean


def check_shapes(x, y):
    assert x.ndim == 3  # (n_points, n_tasks, d_x)
    assert x.shape[-1] == 1  # d_x = 1
    if y is not None:
        assert x.shape == y.shape  # d_y = 1


# def mtblr_guide(x: torch.tensor, d_y: int, y: Optional[torch.tensor] = None):
# check_shapes(x=x, y=y)
# n_points = x.shape[0]
# n_tasks = x.shape[1]
# d_x = x.shape[2]
# if y is not None:
#     assert y.shape == (n_points, n_tasks, d_y)

# sigma_scale = 1e-4
# sigma_loc = pyro.param(
#     "sigma_loc", torch.tensor(1.0), constraint=constraints.positive
# )
# pyro.sample("sigma", dist.Normal(max(sigma_loc, 5 * sigma_scale), sigma_scale))
# with pyro.plate("tasks", n_tasks):
#     m_loc = pyro.param("m_loc", torch.randn((n_tasks, d_y, d_x)))
#     m_scale = pyro.param(
#         "m_scale", torch.ones((n_tasks, d_y, d_x)), constraint=constraints.positive
#     )
#     m = pyro.sample("m", dist.Normal(m_loc, m_scale).to_event(2))
#     assert m.shape == (n_tasks, d_y, d_x)


def predict(model, guide, x: np.ndarray, d_y: int, n_samples: int):
    # TODO: understand shapes of svi_samples
    predictive = Predictive(
        model=model,
        guide=guide,
        num_samples=n_samples,
        return_sites=(
            # if model is linear, those sites are available
            "mtbnn.weight",
            "mtbnn.bias",
            # those sites are always available
            "obs",
            "sigma",
            "_RETURN",
        ),
    )
    svi_samples = predictive(x=torch.tensor(x), d_y=d_y, y=None)
    svi_samples = {k: v for k, v in svi_samples.items()}
    pred_summary = summary(svi_samples)

    return pred_summary, svi_samples


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0).detach().cpu().numpy(),
            "std": torch.std(v, 0).detach().cpu().numpy(),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0].detach().cpu().numpy(),
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0].detach().cpu().numpy(),
        }
    return site_stats


def plot_one_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    pred_summary: dict,
    plot_obs: bool,
    ax: None,
    max_tasks: int,
):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    x_pred = x_pred.squeeze(-1)
    means_plt = pred_summary["_RETURN"]["mean"].squeeze(-1)
    means_perc5_plt = pred_summary["_RETURN"]["5%"].squeeze(-1)
    means_perc95_plt = pred_summary["_RETURN"]["95%"].squeeze(-1)
    obs_plt = pred_summary["obs"]["mean"].squeeze(-1)
    obs_perc5_plt = pred_summary["obs"]["5%"].squeeze(-1)
    obs_perc95_plt = pred_summary["obs"]["95%"].squeeze(-1)
    for l in range(n_task):
        if l == max_tasks:
            break
        base_line = ax.plot(x_pred[:, l], means_plt[:, l])[0]
        ax.scatter(x_train[:, l], y_train[:, l], color=base_line.get_color())
        if not plot_obs:
            ax.fill_between(
                x_pred[:, l],
                means_perc5_plt[:, l],
                means_perc95_plt[:, l],
                alpha=0.3,
                color=base_line.get_color(),
            )
        else:
            ax.fill_between(
                x_pred[:, l],
                obs_perc5_plt[:, l],
                obs_perc95_plt[:, l],
                alpha=0.3,
                color=base_line.get_color(),
            )
    ax.grid()


def plot_predictions(
    x_train,
    y_train,
    x_pred,
    pred_summary_prior_untrained,
    pred_summary_prior,
    pred_summary_posterior,
    max_tasks=3,
):
    assert x_train.shape[-1] == y_train.shape[-1] == x_pred.shape[-1]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), sharey=True)
    fig.suptitle(f"Prior and Posterior Predictions")

    ax = axes[0, 0]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Mean (untrained)")
    plot_one_prediction(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_pred,
        pred_summary=pred_summary_prior_untrained,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    ax = axes[0, 1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Mean (trained)")
    plot_one_prediction(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_pred,
        pred_summary=pred_summary_prior,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    ax = axes[0, 2]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Mean")
    plot_one_prediction(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_pred,
        pred_summary=pred_summary_posterior,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    ax = axes[1, 0]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Observation (untrained)")
    plot_one_prediction(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_pred,
        pred_summary=pred_summary_prior_untrained,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )

    ax = axes[1, 1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Observation (trained)")
    plot_one_prediction(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_pred,
        pred_summary=pred_summary_prior,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )

    ax = axes[1, 2]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Observation")
    plot_one_prediction(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_pred,
        pred_summary=pred_summary_posterior,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )


def plot_distributions(
    site_name,
    bm,
    bm_param_idx,
    samples_prior_untrained,
    samples_prior,
    samples_posterior,
    max_tasks=3,
):
    # TODO: why do we need to squeeze the slope samples?
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), sharex=True)
    fig.suptitle(f"Prior and Posterior Distributions of site '{site_name}'")

    for l, task in enumerate(bm):
        if l == max_tasks:
            break
        ax = axes[0]
        ax.set_title("Prior distribution (untrained)")
        sns.distplot(
            samples_prior_untrained[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_param_idx is not None:
            ax.axvline(x=task.param[bm_param_idx], color=sns.color_palette()[l])

        ax = axes[1]
        ax.set_title("Prior distribution (trained)")
        sns.distplot(
            samples_prior[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_param_idx is not None:
            ax.axvline(x=task.param[bm_param_idx], color=sns.color_palette()[l])

        ax = axes[2]
        ax.set_title("Posterior distribution")
        sns.distplot(
            samples_posterior[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_param_idx is not None:
            ax.axvline(x=task.param[bm_param_idx], color=sns.color_palette()[l])
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[0].set_xlabel("m")
    axes[1].set_xlabel("m")
    axes[2].set_xlabel("m")


def collate_data(bm: MetaLearningBenchmark):
    x = np.zeros((bm.n_points_per_task, bm.n_task, bm.d_x))
    y = np.zeros((bm.n_points_per_task, bm.n_task, bm.d_y))
    for l, task in enumerate(bm):
        x[:, l] = task.x
        y[:, l] = task.y
    return x, y


def print_parameters():
    for name, value in pyro.get_param_store().items():
        print(f"{name}\n{pyro.param(name)}")


def freeze_parameters():
    for name, value in pyro.get_param_store().items():
        pyro.param[name] = pyro.param[name].detach()


if __name__ == "__main__":
    # TODO: use more expressive prior
    # TODO: implement adaptation
    # TODO: multivariate normal guide
    # seed
    pyro.set_rng_seed(123)

    ## flags, constants
    plot = True
    smoke_test = True
    # benchmark
    n_task = 16
    n_points_per_task = 32
    output_noise = 0.01
    # model
    n_hidden = 0
    d_hidden = 4
    # training
    n_iter = 5000 if not smoke_test else 1000
    initial_lr = 0.1
    gamma = 0.00001  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / n_iter)
    adam = ClippedAdam({"lr": initial_lr, "lrd": lrd})
    # evaluation
    n_pred = 100
    x_pred = np.linspace(-1.5, 1.5, n_pred)[:, None, None].repeat(n_task, axis=1)
    n_samples = 1000
    max_plot_tasks = 5

    # create benchmark
    bm = Affine1D(
        n_task=n_task,
        n_datapoints_per_task=n_points_per_task,
        output_noise=output_noise,
        seed_task=1234,
        seed_x=1235,
        seed_noise=1236,
    )
    x, y = collate_data(bm=bm)

    # create model
    mtblr = MTBLR(d_x=bm.d_x, d_y=bm.d_y, n_hidden=n_hidden, d_hidden=d_hidden)

    # obtain predictions before training
    mtblr.eval()  # TODO: do we need this?
    with torch.no_grad():
        pred_summary_prior_untrained, samples_prior_untrained = predict(
            model=mtblr, guide=None, x=x_pred, d_y=bm.d_y, n_samples=n_samples
        )

    print("\n************************")
    print("*** Prior parameters ***")
    print("************************")
    print_parameters()
    print("************************")

    # do inference
    print("\n*******************************")
    print("*** Performing inference... ***")
    print("*******************************")
    mtblr.train()  # TODO: do we need this?
    guide = AutoDiagonalNormal(model=mtblr)
    svi = SVI(model=mtblr, guide=guide, optim=adam, loss=Trace_ELBO())
    pyro.clear_param_store()
    for i in range(n_iter):
        elbo = svi.step(x=torch.tensor(x), d_y=bm.d_y, y=torch.tensor(y))
        if i % 100 == 0:
            print(f"[iter {i:04d}] elbo = {elbo:.4f}")
    print("*******************************")

    # print learned parameters
    print("\n****************************")
    print("*** Posterior parameters ***")
    print("****************************")
    print_parameters()
    print("****************************")

    # obtain prior predictions
    pred_summary_prior, samples_prior = predict(
        model=mtblr, guide=None, x=x_pred, d_y=bm.d_y, n_samples=n_samples
    )

    # obtain posterior predictions
    pred_summary_posterior, samples_posterior = predict(
        model=mtblr, guide=guide, x=x_pred, d_y=bm.d_y, n_samples=n_samples
    )

    # plot predictions
    if plot:
        plot_predictions(
            x_train=x,
            y_train=y,
            x_pred=x_pred,
            pred_summary_prior_untrained=pred_summary_prior_untrained,
            pred_summary_prior=pred_summary_prior,
            pred_summary_posterior=pred_summary_posterior,
            max_tasks=max_plot_tasks,
        )

        if n_hidden == 0:
            # plot prior and posterior distributions

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_distributions(
                    site_name="mtbnn.weight",
                    bm=bm,
                    bm_param_idx=0
                    if isinstance(bm, Linear1D) or isinstance(bm, Affine1D)
                    else None,
                    samples_prior_untrained=samples_prior_untrained,
                    samples_prior=samples_prior,
                    samples_posterior=samples_posterior,
                )

                plot_distributions(
                    site_name="mtbnn.bias",
                    bm=bm,
                    bm_param_idx=1 if isinstance(bm, Affine1D) else None,
                    samples_prior_untrained=samples_prior_untrained,
                    samples_prior=samples_prior,
                    samples_posterior=samples_posterior,
                )

        plt.show()
