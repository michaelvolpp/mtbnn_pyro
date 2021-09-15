import copy
import warnings
from typing import List, Optional

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
from pyro import poutine
from pyro.distributions import constraints
from pyro.infer import SVI, Predictive, Trace_ELBO, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoNormal
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.optim import ClippedAdam
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam as AdamTorch
from torch.optim.lr_scheduler import ExponentialLR

### PyroNotes:
## Learned PyroParams
# https://docs.pyro.ai/en/dev/nn.html#pyro.nn.module.PyroSample
# https://forum.pyro.ai/t/getting-estimates-of-parameters-that-use-pyrosample/2901/2
## Plates with explicit independent dimensions
# https://pyro.ai/examples/tensor_shapes.html#Declaring-independent-dims-with-plate


def _generate_mtbnn_module(
    d_x: int, d_y: int, n_hidden: int, d_hidden: int, prior_type: str
) -> PyroModule:

    """
    Generate a multi-task Bayesian neural network Pyro module.
    """
    modules = []
    if n_hidden == 0:
        modules.append(
            MultiTaskBayesianLinear(
                in_features=d_x,
                out_features=d_y,
                bias=True,
                prior_type=prior_type,
            )
        )
    else:
        modules.append(
            MultiTaskBayesianLinear(
                in_features=d_x,
                out_features=d_hidden,
                bias=True,
                prior_type=prior_type,
            )
        )
        modules.append(PyroModule[nn.Tanh]())
        for _ in range(n_hidden - 1):
            modules.append(
                MultiTaskBayesianLinear(
                    in_features=d_hidden,
                    out_features=d_hidden,
                    bias=True,
                    prior_type=prior_type,
                )
            )
            modules.append(PyroModule[nn.Tanh]())
        modules.append(
            MultiTaskBayesianLinear(
                in_features=d_hidden,
                out_features=d_y,
                bias=True,
                prior_type=prior_type,
            )
        )
    net = PyroModule[nn.Sequential](*modules)

    return net


def _kl_regularizer(model: PyroModule):
    prior = model.get_prior_distribution()
    regularizer = torch.tensor(0.0)
    for prior_factor in prior:
        assert len(prior_factor.batch_shape) == 0
        normal = (
            dist.Normal(0.0, 1.0)
            .expand(prior_factor.event_shape)
            .to_event(len(prior_factor.event_shape))
        )
        kl = kl_divergence(prior_factor, normal)
        regularizer = regularizer + kl
    return regularizer


def _train_model_svi(
    model: PyroModule,
    guide: PyroModule,
    x: torch.tensor,
    y: torch.tensor,
    n_epochs: int,
    alpha_reg: float,
    initial_lr: float,
    final_lr: Optional[float] = None,
) -> torch.tensor:
    ## get parameters
    pyro.clear_param_store()  # to forget old guide shapes
    params_model = list(model.parameters())
    guide(x=x, y=y)
    params_guide = list(guide.parameters())
    params = params_model + params_guide

    ## optimizer
    # use the same tweaks as Pyro's ClippedAdam: LR decay and gradient clipping
    # (gradient clipping is implemented in the training loop itself)
    optim = AdamTorch(params=params, lr=initial_lr)
    if final_lr is not None:
        gamma = final_lr / initial_lr  # final learning rate will be gamma * initial_lr
        lr_decay = gamma ** (1 / n_epochs)
        lr_scheduler = ExponentialLR(optimizer=optim, gamma=lr_decay)
    else:
        lr_scheduler = None

    ## loss
    regularizer_fn = _kl_regularizer
    loss_fn = Trace_ELBO().differentiable_loss

    ## training loop
    train_losses = []
    for i in range(n_epochs):
        optim.zero_grad()

        # compute loss
        elbo = -loss_fn(
            model=model,
            guide=guide,
            x=x,
            y=y,
        )
        loss = -elbo

        # add regularizer
        if regularizer_fn is not None:
            regularizer = regularizer_fn(model=model)
            loss = loss + alpha_reg * regularizer

        # compute gradients and step
        loss.backward()
        clip_grad_norm_(params, max_norm=10.0)  # gradient clipping
        optim.step()

        # adapt lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # logging
        train_losses.append(loss.item())
        if i % 100 == 0 or i == len(range(n_epochs)) - 1:
            print(f"[iter {i:04d}] elbo = {elbo:.4e} | reg = {regularizer:.4e}")

    return torch.tensor(train_losses)


def _marginal_log_likelihood(
    model: PyroModule,
    guide: PyroModule,
    x: torch.tensor,
    y: torch.tensor,
    n_samples: int,
) -> torch.tensor:
    """
    Computes predictive log-likelihood using latent samples from guide using Predictive.
    """
    # obtain vectorized model trace
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
    n_task = x.shape[0]
    n_pts = x.shape[1]
    assert log_prob.shape == (n_samples, n_task, n_pts)

    # compute predictive likelihood
    log_prob = torch.sum(log_prob, dim=2, keepdim=True)  # sum pts-per-task dim
    log_prob = torch.logsumexp(log_prob, dim=0, keepdim=True)  # reduce sampledim
    log_prob = torch.sum(log_prob, dim=1, keepdim=True)  # sum task dim
    assert log_prob.shape == (1, 1, 1)
    log_prob = log_prob.squeeze_()
    log_prob = log_prob - n_task * torch.log(torch.tensor(n_samples))

    # normalize w.r.t. number of datapoints
    log_prob = log_prob / n_task / n_pts

    return log_prob


def summarize_samples(samples: dict) -> dict:
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": np.mean(v, axis=0),
            "std": np.std(v, axis=0),
            "5%": np.percentile(v, 5, axis=0),
            "95%": np.percentile(v, 95, axis=0),
        }
    return site_stats


class MultiTaskBayesianLinear(PyroModule):
    """
    A multi-task Bayesian linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_type: str = "isotropic",
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_type = prior_type

        ## weight prior
        if self.prior_type == "isotropic":
            self.weight_prior_loc = PyroParam(
                init_value=torch.tensor(0.0),
                constraint=constraints.real,
            )
            self.weight_prior_scale = PyroParam(
                init_value=torch.tensor(1.0),
                constraint=constraints.positive,
            )
            self.weight_prior = (
                lambda self: dist.Normal(self.weight_prior_loc, self.weight_prior_scale)
                .expand([self.out_features, self.in_features])
                .to_event(2)
            )
        elif self.prior_type == "diagonal":
            self.weight_prior_loc = PyroParam(
                init_value=torch.zeros(self.out_features, self.in_features),
                constraint=constraints.real,
            )
            self.weight_prior_scale = PyroParam(
                init_value=torch.ones(self.out_features, self.in_features),
                constraint=constraints.positive,
            )
            self.weight_prior = lambda self: dist.Normal(
                self.weight_prior_loc, self.weight_prior_scale
            ).to_event(2)
        else:
            raise ValueError(f"Unknown prior specification '{self.prior_type}'!")
        self.weight = PyroSample(self.weight_prior)

        ## bias prior
        if bias:
            if self.prior_type == "isotropic":
                self.bias_prior_loc = PyroParam(
                    init_value=torch.tensor(0.0),
                    constraint=constraints.real,
                )
                self.bias_prior_scale = PyroParam(
                    init_value=torch.tensor(1.0),
                    constraint=constraints.positive,
                )
                self.bias_prior = (
                    lambda self: dist.Normal(self.bias_prior_loc, self.bias_prior_scale)
                    .expand([self.out_features])
                    .to_event(1)
                )
            elif self.prior_type == "diagonal":
                self.bias_prior_loc = PyroParam(
                    init_value=torch.zeros(self.out_features),
                    constraint=constraints.real,
                )
                self.bias_prior_scale = PyroParam(
                    init_value=torch.ones(self.out_features),
                    constraint=constraints.positive,
                )
                self.bias_prior = lambda self: dist.Normal(
                    self.bias_prior_loc, self.bias_prior_scale
                ).to_event(1)
            else:
                raise ValueError(f"Unknown prior specification '{self.prior_type}'!")
            self.bias = PyroSample(self.bias_prior)
        else:
            self.bias = None

    def forward(self, x: torch.tensor) -> torch.tensor:
        weight, bias = self.weight, self.bias  # we will create views below

        ## check shapes
        # weight.event_shape == (self.out_features, self.in_features)
        # weight.batch_shape depends on whether a sample dimension is added, (e.g.,
        #  by Predictive)
        has_sample_dim = len(self.weight.shape) == 5
        if not has_sample_dim:
            # add sample dim
            n_samples = 1
            weight = weight[None, :, :, :, :]
        else:
            n_samples = weight.shape[0]

        # x.shape = (n_tasks, n_points, d_x) for the input layer
        # x.shape = (n_samples, n_tasks, n_points, d_layer) for hidden layers
        if x.ndim == 3:  # input layer
            n_tasks = x.shape[0]
            n_points = x.shape[1]
            # expand x to sample shape
            x_expd = x.expand(torch.Size([n_samples]) + x.shape)
            x_expd = x_expd.float()
        else:
            assert x.ndim == 4
            n_tasks = x.shape[1]
            n_points = x.shape[2]
            x_expd = x

        assert x_expd.shape == (n_samples, n_tasks, n_points, self.in_features)
        assert weight.shape == (
            n_samples,
            n_tasks,
            1,  # because the n_pts plate is nested inside of the n_tsk plate
            self.out_features,
            self.in_features,
        )
        # squeeze n_pts batch dimension
        weight = weight.squeeze(2)

        ## compute the linear transformation
        y = torch.einsum("slyx,slnx->slny", weight, x_expd)

        if bias is not None:
            ## check shapes
            # bias.event_shape = (self.out_features)
            # bias.batch_shape = (n_tasks, 1) or (n_samples, n_tasks, 1) (cf. above)
            if not has_sample_dim:
                # add sample dim
                bias = bias[None, :, :, :]
            assert bias.shape == (n_samples, n_tasks, 1, self.out_features)
            # squeeze the n_pts batch dimension
            bias = bias.squeeze(2)
            assert bias.shape == (n_samples, n_tasks, self.out_features)

            ## add the bias
            y = y + bias[:, :, None, :]

        if not has_sample_dim:
            # if we do not have a sample dimension, we must not return one
            y.squeeze_(0)

        return y

    def get_prior_distribution(self) -> List:
        return [self.weight_prior(self), self.bias_prior(self)]


class MultiTaskBayesianNeuralNetwork(PyroModule):
    """
    A multi-task Bayesian neural network.
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        n_hidden: int,
        d_hidden: int,
        prior_type: str,
        noise_stddev: Optional[float] = None,
    ):
        super().__init__()

        ## the mean network
        self.net = _generate_mtbnn_module(
            d_x=d_x,
            d_y=d_y,
            n_hidden=n_hidden,
            d_hidden=d_hidden,
            prior_type=prior_type,
        )

        ## noise stddev
        # TODO: learn noise prior?
        self.noise_stddev_prior = (
            dist.Uniform(0.0, 1.0) if noise_stddev is None else None
        )
        self.noise_stddev = (
            PyroSample(self.noise_stddev_prior)
            if noise_stddev is None
            else noise_stddev
        )

        ## the current guide
        self._meta_guide = None
        self._guide = None

        ## set evaluation mode
        self.freeze_prior()
        self.eval()

    def freeze_prior(self) -> None:
        """
        Freeze the unconstrained parameters.
        -> those are the leaf variables of the autograd graph
        -> those are the registered parameters of self
        """
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_prior(self) -> None:
        """
        Unfreeze the unconstrained parameters.
        -> those are the leaf variables of the autograd graph
        -> those are the registered parameters of self
        """
        for p in self.parameters():
            p.requires_grad = True

    def get_prior_distribution(self) -> List:
        prior_distribution = []
        for module in self.net:
            if hasattr(module, "get_prior_distribution"):
                prior_distribution += module.get_prior_distribution()
        return prior_distribution

    @staticmethod
    def print_parameters() -> None:
        first = True
        for name in pyro.get_param_store().keys():
            if first:
                first = False
            else:
                print("\n")
            print(
                f"name  = {name}"
                f"\nshape = {pyro.param(name).shape}"
                f"\nvalue = {pyro.param(name)}"
            )

    def forward(
        self, x: torch.tensor, y: Optional[torch.tensor] = None
    ) -> torch.tensor:
        # shapes
        assert x.ndim == 3
        if y is not None:
            assert y.ndim == 3
        n_tasks = x.shape[0]
        n_points = x.shape[1]

        noise_stddev = self.noise_stddev  # (sample) noise stddev
        with pyro.plate("tasks", n_tasks, dim=-2):
            mean = self.net(x)  # sample weights and compute mean pred
            if noise_stddev.nelement() > 1:
                # noise stddev can have a sample dimension! -> expand to mean's shape
                noise_stddev = noise_stddev.reshape([-1] + [1] * (mean.ndim - 1))
                noise_stddev = noise_stddev.expand(mean.shape)
            with pyro.plate("data", n_points, dim=-1):
                obs = pyro.sample(
                    "obs", dist.Normal(mean, noise_stddev).to_event(1), obs=y
                )  # score mean predictions against ground truth
        return mean

    def get_guide(self, guide: str) -> Optional[PyroModule]:
        assert guide == "meta" or guide == "test" or guide == "prior"
        if guide == "meta":
            guide = self._meta_guide
        elif guide == "test":
            guide = self._guide
        else:
            guide = None

        return guide

    def meta_train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_epochs: int,
        initial_lr: float,
        final_lr: float,
        alpha_reg: float,
    ) -> np.ndarray:
        self.unfreeze_prior()
        self.train()

        meta_guide = AutoDiagonalNormal(model=self)
        epoch_losses = _train_model_svi(
            model=self,
            guide=meta_guide,
            x=torch.tensor(x, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.float),
            n_epochs=n_epochs,
            initial_lr=initial_lr,
            final_lr=final_lr,
            alpha_reg=alpha_reg,
        )

        self._meta_guide = meta_guide
        self.eval()
        self.freeze_prior()

        return epoch_losses.numpy()

    def adapt(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_epochs: int,
        initial_lr: float,
        final_lr: float,
    ) -> np.ndarray:
        self.train()

        if x.size == 0:
            guide = None
            epoch_losses = np.array([])
        else:
            guide = AutoDiagonalNormal(model=self)
            epoch_losses = _train_model_svi(
                model=self,
                guide=guide,
                x=torch.tensor(x, dtype=torch.float),
                y=torch.tensor(y, dtype=torch.float),
                n_epochs=n_epochs,
                initial_lr=initial_lr,
                final_lr=final_lr,
                alpha_reg=0.0,
            ).numpy()

        self._guide = guide
        self.eval()

        return epoch_losses

    @torch.no_grad()
    def marginal_log_likelihood(
        self, x: np.ndarray, y: np.ndarray, n_samples: int, guide_choice: str
    ):
        guide = self.get_guide(guide=guide_choice)
        if guide is not None:
            assert guide.plates["tasks"].size == x.shape[0]

        if x.size > 0:
            marg_ll = _marginal_log_likelihood(
                model=self,
                guide=guide,
                x=torch.tensor(x, dtype=torch.float),
                y=torch.tensor(y, dtype=torch.float),
                n_samples=n_samples,
            ).numpy()
        else:
            marg_ll = np.nan

        return marg_ll

    @torch.no_grad()
    def predict(self, x: np.ndarray, n_samples: int, guide: str) -> dict:
        guide = self.get_guide(guide=guide)
        if guide is not None:
            assert guide.plates["tasks"].size == x.shape[0], (
                f"x and guide have different numbers of tasks! "
                f"x.shape[0] = {x.shape[0]:d}, "
                f"guide.plates['tasks'].size = {guide.plates['tasks'].size:d}"
            )

        predictive = Predictive(
            model=self,
            guide=guide,
            num_samples=n_samples,
            parallel=True,  # our model is vectorized
            return_sites=(
                # if model is linear, those sites are available
                "net.0.weight",
                "net.0.bias",
                # those sites are always available
                "obs",
                "sigma",
                "_RETURN",
            ),
        )
        samples = predictive(x=torch.tensor(x, dtype=torch.float), y=None)
        samples = {k: v.detach().cpu().numpy() for k, v in samples.items()}

        return samples


def plot_predictions_for_one_set_of_tasks(
    x: np.ndarray,
    y: np.ndarray,
    x_pred: np.ndarray,
    pred_summary: dict,
    plot_obs: bool,
    ax: None,
    max_tasks: int,
    n_context: Optional[np.ndarray] = None,
):
    ## prepare data
    means_plt = pred_summary["_RETURN"]["mean"]
    means_perc5_plt = pred_summary["_RETURN"]["5%"]
    means_perc95_plt = pred_summary["_RETURN"]["95%"]
    obs_perc5_plt = pred_summary["obs"]["5%"]
    obs_perc95_plt = pred_summary["obs"]["95%"]

    # assert that all inputs have the same number of dimensions
    assert (
        x.ndim
        == y.ndim
        == x_pred.ndim
        == means_plt.ndim
        == means_perc5_plt.ndim
        == means_perc95_plt.ndim
        == obs_perc5_plt.ndim
        == obs_perc95_plt.ndim
    )
    # check that all inputs are one-dimensional
    assert (
        x.shape[-1]
        == y.shape[-1]
        == x_pred.shape[-1]
        == means_plt.shape[-1]
        == means_perc5_plt.shape[-1]
        == means_perc95_plt.shape[-1]
        == obs_perc5_plt.shape[-1]
        == obs_perc95_plt.shape[-1]
        == 1
    )
    # check that all inputs have the same number of tasks
    assert (
        x.shape[0]
        == y.shape[0]
        == means_plt.shape[0]
        == means_perc5_plt.shape[0]
        == means_perc95_plt.shape[0]
        == obs_perc5_plt.shape[0]
        == obs_perc95_plt.shape[0]
    )
    # squeeze data dimension
    (
        x,
        y,
        x_pred,
        means_plt,
        means_perc5_plt,
        means_perc95_plt,
        obs_perc5_plt,
        obs_perc95_plt,
    ) = (
        x.squeeze(-1),
        y.squeeze(-1),
        x_pred.squeeze(-1),
        means_plt.squeeze(-1),
        means_perc5_plt.squeeze(-1),
        means_perc95_plt.squeeze(-1),
        obs_perc5_plt.squeeze(-1),
        obs_perc95_plt.squeeze(-1),
    )

    ## prepare plot
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    ## plot
    n_task = x.shape[0]
    for l in range(n_task):
        if l == max_tasks:
            break
        line = ax.plot(x_pred[l, :], means_plt[l, :])[0]
        ax.scatter(x[l, :], y[l, :], color=line.get_color())
        if n_context is not None:
            x_context, y_context, _, _ = split_tasks(
                x=x[l : l + 1, :][:, :, None],
                y=y[l : l + 1, :][:, :, None],
                n_context=n_context,
            )
            ax.scatter(
                x_context,
                y_context,
                marker="x",
                s=100,
                color=line.get_color(),
            )
        if not plot_obs:
            ax.fill_between(
                x_pred[l, :],
                means_perc5_plt[l, :],
                means_perc95_plt[l, :],
                alpha=0.3,
                color=line.get_color(),
            )
        else:
            ax.fill_between(
                x_pred[l, :],
                obs_perc5_plt[l, :],
                obs_perc95_plt[l, :],
                alpha=0.3,
                color=line.get_color(),
            )
    ax.grid()


def plot_predictions(
    x_meta,
    y_meta,
    x_pred_meta,
    x_test,
    y_test,
    n_contexts_test,
    x_pred_test,
    pred_summary_prior_meta_untrained,
    pred_summary_prior_meta_trained,
    pred_summary_posterior_meta,
    pred_summaries_posterior_test,
    max_tasks=3,
):
    fig, axes = plt.subplots(
        nrows=2, ncols=3 + len(n_contexts_test), figsize=(16, 8), sharey=True
    )
    fig.suptitle(f"Prior and Posterior Predictions")

    ax = axes[0, 0]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Mean + Meta Data")
    plot_predictions_for_one_set_of_tasks(
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        pred_summary=pred_summary_prior_meta_untrained,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    ax = axes[0, 1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Mean\n(trained on meta data)")
    plot_predictions_for_one_set_of_tasks(
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        pred_summary=pred_summary_prior_meta_trained,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    ax = axes[0, 2]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Mean\n(meta data)")
    plot_predictions_for_one_set_of_tasks(
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        pred_summary=pred_summary_posterior_meta,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    for i, n_context in enumerate(n_contexts_test):
        ax = axes[0, 3 + i]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Posterior Mean\n(test data, n_ctx={n_context:d})")
        plot_predictions_for_one_set_of_tasks(
            x=x_test,
            y=y_test,
            x_pred=x_pred_test,
            n_context=n_context,
            pred_summary=pred_summaries_posterior_test[i],
            ax=ax,
            plot_obs=False,
            max_tasks=max_tasks,
        )

    ax = axes[1, 0]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Observation + Meta Data")
    plot_predictions_for_one_set_of_tasks(
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        pred_summary=pred_summary_prior_meta_untrained,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )

    ax = axes[1, 1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Observation\n(trained on meta data)")
    plot_predictions_for_one_set_of_tasks(
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        pred_summary=pred_summary_prior_meta_trained,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )

    ax = axes[1, 2]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Observation\n(meta data)")
    plot_predictions_for_one_set_of_tasks(
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        pred_summary=pred_summary_posterior_meta,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )

    for i, n_context in enumerate(n_contexts_test):
        ax = axes[1, 3 + i]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Posterior Observation\n(test data, n_ctx={n_context:d})")
        plot_predictions_for_one_set_of_tasks(
            x=x_test,
            y=y_test,
            x_pred=x_pred_test,
            n_context=n_context,
            pred_summary=pred_summaries_posterior_test[i],
            ax=ax,
            plot_obs=True,
            max_tasks=max_tasks,
        )
    fig.tight_layout()


def plot_distributions(
    site_name,
    bm_meta_params,
    bm_test_params,
    samples_prior_meta_untrained,
    samples_prior_meta_trained,
    samples_posterior_meta,
    samples_posteriors_test,
    n_contexts_test,
    max_tasks=3,
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3 + len(n_contexts_test),
        figsize=(12, 6),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(f"Prior and Posterior Distributions of site '{site_name}'")

    n_meta_tasks = samples_prior_meta_untrained[site_name].shape[1]
    n_test_tasks = samples_posteriors_test[0][site_name].shape[1]

    for l in range(n_meta_tasks):
        if l == max_tasks:
            break
        ax = axes[0, 0]
        ax.set_title("Prior distribution\n(untrained)")
        sns.distplot(
            samples_prior_meta_untrained[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_meta_params is not None:
            ax.axvline(x=bm_meta_params[l], color=sns.color_palette()[l])

        ax = axes[0, 1]
        ax.set_title("Prior distribution\n(trained on meta data)")
        sns.distplot(
            samples_prior_meta_trained[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_meta_params is not None:
            ax.axvline(x=bm_meta_params[l], color=sns.color_palette()[l])

        ax = axes[0, 2]
        ax.set_title("Posterior distribution\n(meta data)")
        sns.distplot(
            samples_posterior_meta[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_meta_params is not None:
            ax.axvline(x=bm_meta_params[l], color=sns.color_palette()[l])

    for i, n_context in enumerate(n_contexts_test):
        for l in range(n_test_tasks):
            if l == max_tasks:
                break
            ax = axes[0, 3 + i]
            ax.set_title(f"Posterior distribution\n(test data, n_ctx={n_context:d})")
            sns.distplot(
                samples_posteriors_test[i][site_name].squeeze()[:, l],
                kde_kws={"label": f"Test Task {l}"},
                ax=ax,
            )
            if bm_test_params is not None:
                ax.axvline(x=bm_test_params[l], color=sns.color_palette()[l])

    for ax in axes[0]:
        ax.legend()
        ax.set_xlabel(site_name)
    fig.tight_layout()


def plot_metrics(
    learning_curve_meta, learning_curves_test, lls, lls_context, n_contexts
):
    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=False)
    fig.suptitle("Metrics")

    ax = axes[0, 0]
    ax.set_title("Learning Curve (meta training)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("symlog")  # TODO: think about symlog scaling
    if learning_curve_meta is not None:
        ax.plot(
            np.arange(len(learning_curve_meta)),
            learning_curve_meta,
        )
        ax.legend()
    ax.grid()

    ax = axes[0, 1]
    ax.set_title("Learning Curve (adaptation)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("symlog")  # TODO: think about symlog scaling
    for learning_curve, n_context in zip(learning_curves_test, n_contexts):
        ax.plot(
            np.arange(len(learning_curve)),
            learning_curve,
            label=f"n_ctx={n_context:d}",
        )
    ax.legend()
    ax.grid()

    ax = axes[0, 2]
    ax.set_title("Marginal Log-Likelihood")
    ax.set_xlabel("n_context")
    ax.set_xticks(n_contexts)
    ax.set_ylabel("marginal ll")
    ax.plot(n_contexts, lls, label="all data")
    ax.plot(n_contexts, lls_context, label="context only")
    ax.legend()
    ax.grid()

    fig.tight_layout()


def collate_data(bm: MetaLearningBenchmark):
    x = np.zeros((bm.n_task, bm.n_points_per_task, bm.d_x))
    y = np.zeros((bm.n_task, bm.n_points_per_task, bm.d_y))
    for l, task in enumerate(bm):
        x[l, :] = task.x
        y[l, :] = task.y
    return x, y


def split_tasks(x, y, n_context):
    x_context, y_context = x[:, :n_context, :], y[:, :n_context, :]
    # TODO: use all data as target?
    x_target, y_target = x, y

    return x_context, y_context, x_target, y_target


def main():
    # TODO: sample functions
    # TODO: use exact prior/posterior distributions, not the KDE (e.g., prior is task-independent!)
    # TODO: implement more complex priors (e.g., not factorized across layers?)

    ## flags, constants
    pyro.set_rng_seed(123)
    plot = True
    smoke_test = False
    # benchmarks
    bm = Affine1D
    noise_stddev = 0.01
    n_tasks_meta = 8
    n_points_per_task_meta = 16
    n_tasks_test = 128
    n_points_per_task_test = 128
    # model
    n_hidden = 0
    d_hidden = 8
    infer_noise_stddev = True
    prior_type = "diagonal"
    # training
    do_meta_training = True
    n_epochs = 2000 if not smoke_test else 100
    initial_lr = 0.1
    final_lr = 0.00001
    alpha_reg = 0.0
    # evaluation
    n_contexts = (
        np.array([0, 1, 2, 5, 10, n_points_per_task_test])
        if not smoke_test
        else np.array([0, 5, 10])
    )
    n_pred = 100
    n_samples = 1000 if not smoke_test else 100
    max_plot_tasks = 5

    ## create benchmarks
    # meta benchmark
    bm_meta = bm(
        n_task=n_tasks_meta,
        n_datapoints_per_task=n_points_per_task_meta,
        output_noise=noise_stddev,
        seed_task=1234,
        seed_x=1235,
        seed_noise=1236,
    )
    x_meta, y_meta = collate_data(bm=bm_meta)
    x_pred_meta = np.linspace(-1.5, 1.5, n_pred)[None, :, None].repeat(
        n_tasks_meta, axis=0
    )
    # test benchmark
    bm_test = bm(
        n_task=n_tasks_test,
        n_datapoints_per_task=n_points_per_task_test,
        output_noise=noise_stddev,
        seed_task=1235,
        seed_x=1236,
        seed_noise=1237,
    )
    x_test, y_test = collate_data(bm=bm_test)
    x_pred_test = np.linspace(-1.5, 1.5, n_pred)[None, :, None].repeat(
        n_tasks_test, axis=0
    )

    ## create model
    mtbnn = MultiTaskBayesianNeuralNetwork(
        d_x=bm_meta.d_x,
        d_y=bm_meta.d_y,
        n_hidden=n_hidden,
        d_hidden=d_hidden,
        noise_stddev=None if infer_noise_stddev else noise_stddev,
        prior_type=prior_type,
    )

    ## obtain predictions on meta data before meta training
    samples_prior_meta_untrained = mtbnn.predict(
        x=x_pred_meta, n_samples=n_samples, guide="prior"
    )
    pred_summary_prior_meta_untrained = summarize_samples(
        samples=samples_prior_meta_untrained
    )

    ## print prior parameters
    print("\n************************")
    print("*** Prior parameters ***")
    print("************************")
    mtbnn.print_parameters()
    print("************************")

    ## do inference
    print("\n*******************************")
    print("*** Performing inference... ***")
    print("*******************************")
    if do_meta_training:
        learning_curve_meta = mtbnn.meta_train(
            x=x_meta,
            y=y_meta,
            n_epochs=n_epochs,
            initial_lr=initial_lr,
            final_lr=final_lr,
            alpha_reg=alpha_reg,
        )
    else:
        learning_curve_meta = None
    print("*******************************")

    ## print learned parameters
    print("\n****************************")
    print("*** Posterior parameters ***")
    print("****************************")
    mtbnn.print_parameters()
    print("****************************")

    ## obtain predictions on meta data after training
    # obtain prior predictions
    samples_prior_meta_trained = mtbnn.predict(
        x=x_pred_meta, n_samples=n_samples, guide="prior"
    )
    pred_summary_prior_meta_trained = summarize_samples(
        samples=samples_prior_meta_trained
    )
    # obtain posterior predictions
    samples_posterior_meta = mtbnn.predict(
        x=x_pred_meta, n_samples=n_samples, guide="meta"
    )
    pred_summary_posterior_meta = summarize_samples(samples=samples_posterior_meta)

    # print freezed parameters
    print("\n**************************************")
    print("*** Posterior parameters (freezed) ***")
    print("**************************************")
    mtbnn.print_parameters()
    print("**************************************")

    ## do inference on test task
    lls = np.zeros(n_contexts.shape)
    lls_context = np.zeros(n_contexts.shape)
    pred_summaries_posteriors_test, samples_posteriors_test = [], []
    learning_curves_test = []
    for i, n_context in enumerate(n_contexts):
        print("\n**************************************************************")
        print(
            f"*** Performing inference on test tasks (n_context = {n_context:3d})... ***"
        )
        print("**************************************************************")
        x_context, y_context, x_target, y_target = split_tasks(
            x=x_test, y=y_test, n_context=n_context
        )
        learning_curves_test.append(
            mtbnn.adapt(
                x=x_context,
                y=y_context,
                n_epochs=n_epochs,
                initial_lr=initial_lr,
                final_lr=final_lr,
            )
        )
        lls[i] = mtbnn.marginal_log_likelihood(
            x=x_target,
            y=y_target,
            n_samples=n_samples,
            guide_choice="test",
        )
        lls_context[i] = mtbnn.marginal_log_likelihood(
            x=x_context,
            y=y_context,
            n_samples=n_samples,
            guide_choice="test",
        )
        cur_samples_posterior_test = mtbnn.predict(
            x=x_pred_test, n_samples=n_samples, guide="test"
        )
        cur_pred_summary_posterior_test = summarize_samples(
            samples=cur_samples_posterior_test
        )
        pred_summaries_posteriors_test.append(cur_pred_summary_posterior_test)
        samples_posteriors_test.append(cur_samples_posterior_test)
        print("*******************************")

    # print freezed parameters (make sure adaptation step did not change them)
    print("\n**************************************")
    print("*** Posterior parameters (freezed) ***")
    print("**************************************")
    mtbnn.print_parameters()
    print("**************************************")

    # plot predictions
    if plot:
        plot_metrics(
            learning_curve_meta=learning_curve_meta,
            learning_curves_test=learning_curves_test,
            lls=lls,
            lls_context=lls_context,
            n_contexts=n_contexts,
        )
        plot_predictions(
            x_meta=x_meta,
            y_meta=y_meta,
            x_pred_meta=x_pred_meta,
            x_test=x_test,
            y_test=y_test,
            x_pred_test=x_pred_test,
            n_contexts_test=n_contexts,
            pred_summary_prior_meta_untrained=pred_summary_prior_meta_untrained,
            pred_summary_prior_meta_trained=pred_summary_prior_meta_trained,
            pred_summary_posterior_meta=pred_summary_posterior_meta,
            pred_summaries_posterior_test=pred_summaries_posteriors_test,
            max_tasks=max_plot_tasks,
        )

        if n_hidden == 0:
            # plot prior and posterior distributions

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if isinstance(bm_meta, Affine1D):
                    bm_meta_params = np.zeros(n_tasks_meta)
                    bm_test_params = np.zeros(n_tasks_test)
                    for l, task in enumerate(bm_meta):
                        bm_meta_params[l] = task.param[0]
                    for l, task in enumerate(bm_test):
                        bm_test_params[l] = task.param[1]
                else:
                    bm_meta_params, bm_test_params = None, None
                plot_distributions(
                    site_name="net.0.weight",
                    bm_meta_params=bm_meta_params,
                    bm_test_params=bm_test_params,
                    samples_prior_meta_untrained=samples_prior_meta_untrained,
                    samples_prior_meta_trained=samples_prior_meta_trained,
                    samples_posterior_meta=samples_posterior_meta,
                    samples_posteriors_test=samples_posteriors_test,
                    n_contexts_test=n_contexts,
                )

                if isinstance(bm_meta, Affine1D):
                    bm_meta_params = np.zeros(n_tasks_meta)
                    bm_test_params = np.zeros(n_tasks_test)
                    for l, task in enumerate(bm_meta):
                        bm_meta_params[l] = task.param[1]
                    for l, task in enumerate(bm_test):
                        bm_test_params[l] = task.param[1]
                else:
                    bm_meta_params, bm_test_params = None, None
                plot_distributions(
                    site_name="net.0.bias",
                    bm_meta_params=bm_meta_params,
                    bm_test_params=bm_test_params,
                    samples_prior_meta_untrained=samples_prior_meta_untrained,
                    samples_prior_meta_trained=samples_prior_meta_trained,
                    samples_posterior_meta=samples_posterior_meta,
                    samples_posteriors_test=samples_posteriors_test,
                    n_contexts_test=n_contexts,
                )

        plt.show()


if __name__ == "__main__":
    main()
