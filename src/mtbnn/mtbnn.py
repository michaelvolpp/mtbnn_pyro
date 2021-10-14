"""
Implementation of a multi-task Bayesian neural network.
"""

import math
from typing import List, Optional, Union

import numpy as np
import pyro
import torch
from mtutils.mtutils import BatchedLinear, BatchedSequential, broadcast_xwb
from pyro import distributions as dist
from pyro import poutine
from pyro.distributions import constraints
from pyro.infer import Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
from pyro.nn import PyroModule, PyroParam, PyroSample
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam as AdamTorch
from torch.optim.lr_scheduler import ExponentialLR

_allowed_prior_types = [
    # "isotropic_normal",
    "factorized_normal",
    # "factorized_multivariate_normal",
    # "block_diagonal_multivariate_normal",
    # "multivariate_normal",
]

_allowed_prior_inits = [
    "standard_normal",
    "as_pytorch_linear",
]

_allowed_posterior_inits = [
    "pyro_standard",
    "set_to_prior",
]


def _create_batched_bnn(
    d_x: int,
    d_y: int,
    n_hidden: int,
    d_hidden: int,
) -> PyroModule:
    """
    Generate a multi-task Bayesian neural network Pyro module.
    """

    layers = []
    if n_hidden == 0:  # linear model
        layers.append(PyroModule[BatchedLinear](in_features=d_x, out_features=d_y))
    else:  # fully connected MLP
        layers.append(PyroModule[BatchedLinear](in_features=d_x, out_features=d_hidden))
        layers.append(PyroModule[nn.Tanh]())
        for _ in range(n_hidden - 1):
            layers.append(
                PyroModule[BatchedLinear](in_features=d_hidden, out_features=d_hidden)
            )
            layers.append(PyroModule[nn.Tanh]())
        layers.append(PyroModule[BatchedLinear](in_features=d_hidden, out_features=d_y))
    net = PyroModule[BatchedSequential](*layers)

    return net


def _compute_kl_regularizer(model: PyroModule) -> torch.tensor:
    def _compute_kl_to_standard_normal(distribution) -> torch.tensor:
        if hasattr(distribution, "base_dist") and isinstance(
            distribution.base_dist, dist.Normal
        ):
            standard_normal = (
                dist.Normal(0.0, 1.0)
                .expand(distribution.event_shape)
                .to_event(len(distribution.event_shape))
            )
        elif isinstance(distribution, dist.MultivariateNormal):
            standard_normal = dist.MultivariateNormal(
                torch.zeros(distribution.event_shape),
                covariance_matrix=torch.eye(distribution.event_shape[0]),
            )
        else:
            raise NotImplementedError

        return kl_divergence(distribution, standard_normal)

    kl = _compute_kl_to_standard_normal(model.prior_wb)

    return kl


def _train_model_svi(
    model: PyroModule,
    guide: PyroModule,
    x: torch.tensor,
    y: torch.tensor,
    n_epochs: int,
    alpha_reg: float,
    initial_lr: float,
    wandb_run,
    log_identifier: str,
    final_lr: Optional[float] = None,
) -> torch.tensor:
    # watch model exectuion with wandb
    wandb_run.watch(model, log="all")

    ## get parameters
    params_model = list(model.parameters())
    guide(x=x, y=y)  # "create guide"
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
    loss_fn = Trace_ELBO().differentiable_loss
    regularizer_fn = _compute_kl_regularizer

    ## training loop
    train_losses = []
    for i in range(n_epochs):
        optim.zero_grad()

        # compute loss
        elbo = -loss_fn(model=model, guide=guide, x=x, y=y)
        loss = -elbo

        # add regularizer
        if regularizer_fn is not None:
            regularizer = regularizer_fn(model=model)
            loss = loss + alpha_reg * regularizer
        else:
            regularizer = torch.tensor(0.0)

        # compute gradients and step
        loss.backward()
        clip_grad_norm_(params, max_norm=10.0)  # gradient clipping
        optim.step()

        # adapt lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # logging
        train_losses.append(loss.item())
        # TODO: find neater way to log parametric learning curve
        n_context = x.shape[-2]
        wandb_run.log(
            {
                f"{log_identifier}/epoch": i,
                f"{log_identifier}/loss_n_context_{n_context:03d}": loss,
                f"{log_identifier}/elbo_n_context_{n_context:03d}": elbo,
                f"{log_identifier}/regularizer_n_context_{n_context:03d}": regularizer,
            }
        )
        if i % 100 == 0 or i == len(range(n_epochs)) - 1:
            print(f"[iter {i:04d}] elbo = {elbo:.4e} | reg = {regularizer:.4e}")

    return torch.tensor(train_losses)


def _train_model_monte_carlo(
    model: PyroModule,
    x: torch.tensor,
    y: torch.tensor,
    n_samples: int,
    n_epochs: int,
    initial_lr: float,
    wandb_run,
    log_identifier: str,
    final_lr: Optional[float] = None,
) -> torch.tensor:
    # watch model exectuion with wandb
    wandb_run.watch(model, log="all")

    ## get parameters
    params = list(model.parameters())

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

    ## training loop
    train_losses = []
    for i in range(n_epochs):
        optim.zero_grad()

        # compute loss
        loss = -_differentiable_prior_marginal_log_likelihood(
            model=model, x=x, y=y, n_samples=n_samples
        )

        # compute gradients and step
        loss.backward()
        clip_grad_norm_(params, max_norm=10.0)  # gradient clipping
        optim.step()

        # adapt lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # logging
        train_losses.append(loss.item())
        # TODO: find neater way to log parametric learning curve
        n_context = x.shape[-2]
        wandb_run.log(
            {
                f"{log_identifier}/epoch": i,
                f"{log_identifier}/loss_n_context_{n_context:03d}": loss,
            }
        )
        if i % 100 == 0 or i == len(range(n_epochs)) - 1:
            print(f"[iter {i:04d}] marg_ll = {-loss:.4e}")

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


def _differentiable_prior_marginal_log_likelihood(
    model: PyroModule,
    x: torch.tensor,
    y: torch.tensor,
    n_samples: int,
) -> torch.tensor:
    """
    Computes predictive log-likelihood using latent samples from prior.
    Does not use Predictive as Predictive wraps execution in torch.no_grad() and we
    require gradients here
    """
    # obtain trace for n_samples model executions
    vectorized_model = pyro.plate("samples", size=n_samples, dim=-3)(model)
    model_trace = poutine.trace(vectorized_model).get_trace(x=x, y=y)

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
        prior_init: str,
        posterior_init: str,
        noise_stddev: Optional[float] = None,
        device: str = "cpu",
    ):
        super().__init__()

        ## arguments
        self._d_x = d_x
        self._d_y = d_y
        self._n_hidden = n_hidden
        self._d_hidden = d_hidden
        self._device = device
        assert (
            prior_type in _allowed_prior_types
        ), f"Unknown prior type '{prior_type}'! "
        assert (
            prior_init in _allowed_prior_inits
        ), f"Unknown prior initialization '{prior_init}'!"
        assert (
            posterior_init in _allowed_posterior_inits
        ), f"Unknown posterior initialization '{posterior_init}'!"
        self._prior_type = prior_type
        self._prior_init = prior_init
        self._posterior_init = posterior_init

        ## clear everything
        # TODO: this should not be necessary if we properly spawn a process for each wandb run?
        pyro.clear_param_store()

        ## the mean network
        self._bnn = _create_batched_bnn(
            d_x=d_x,
            d_y=d_y,
            n_hidden=n_hidden,
            d_hidden=d_hidden,
        )

        ## latent variables
        self._prior_wb = self._create_bnn_priors()
        self._wb = PyroSample(self._prior_wb)
        self._infer_noise_stddev = noise_stddev is None
        if self._infer_noise_stddev:
            # TODO: learn noise prior?
            self._prior_noise_stddev = dist.Uniform(0.0 + 1e-6, 1.0)
            self._noise_stddev = PyroSample(self._prior_noise_stddev)
        else:
            self._noise_stddev = torch.tensor(noise_stddev)

        ## set evaluation mode
        self.freeze_prior()
        self.eval()

        ## set device
        self.to(self._device)
        if self._infer_noise_stddev:
            self._prior_noise_stddev.low = self._prior_noise_stddev.low.to(self._device)
            self._prior_noise_stddev.high = self._prior_noise_stddev.high.to(
                self._device
            )
        else:
            self._noise_stddev = self._noise_stddev.to(self._device)

    def _create_bnn_priors(self) -> dist.Distribution:
        if self._prior_type == "isotropic_normal":
            self.prior_wb_loc = PyroParam(
                init_value=torch.tensor(0.0),
                constraint=constraints.real,
            )
            self.prior_wb_scale = PyroParam(
                init_value=torch.tensor(1.0),
                constraint=constraints.positive,
            )
            prior_wb = (
                lambda self: dist.Normal(self.prior_wb_loc, self.prior_wb_scale)
                .expand([self._bnn.size_w + self._bnn.size_b])
                .to_event(1)
            )
            return prior_wb

        if self._prior_type == "factorized_normal":
            self.prior_wb_loc = PyroParam(
                init_value=torch.zeros(self._bnn.size_w + self._bnn.size_b),
                constraint=constraints.real,
            )
            self.prior_wb_scale = PyroParam(
                init_value=self._prior_wb_scale_init,
                constraint=constraints.positive,
            )
            prior_wb = lambda self: dist.Normal(
                self.prior_wb_loc, self.prior_wb_scale
            ).to_event(1)

            return prior_wb

        if self._prior_type == "factorized_multivariate_normal":
            self.prior_wb_loc = PyroParam(
                init_value=torch.zeros(self._bnn.size_w + self._bnn.size_b),
                constraint=constraints.real,
            )
            self.prior_wb_scale_diagonal = PyroParam(
                init_value=torch.ones(self._bnn.size_w + self._bnn.size_b),
                constraint=constraints.positive,
            )
            prior_wb = lambda self: dist.MultivariateNormal(
                self.prior_wb_loc,
                scale_tril=torch.diag(self.prior_wb_scale_diagonal),
            )
            return prior_wb

        if self._prior_type == "block_diagonal_multivariate_normal":
            self.prior_wb_loc = PyroParam(
                init_value=torch.zeros(self._bnn.size_w + self._bnn.size_b),
                constraint=constraints.real,
            )
            self.prior_w_scale_tril = PyroParam(
                init_value=torch.eye(self._bnn.size_w),
                constraint=constraints.lower_cholesky,
            )
            self.prior_b_scale_tril = PyroParam(
                init_value=torch.eye(self._bnn.size_b),
                constraint=constraints.lower_cholesky,
            )
            prior_wb = lambda self: dist.MultivariateNormal(
                loc=self.prior_wb_loc,
                scale_tril=torch.block_diag(
                    self.prior_w_scale_tril,
                    self.prior_b_scale_tril,
                ),
            )
            return prior_wb

        if self._prior_type == "multivariate_normal":
            self.prior_wb_loc = PyroParam(
                init_value=torch.zeros(self._bnn.size_w + self._bnn.size_b),
                constraint=constraints.real,
            )
            self.prior_wb_scale_tril = PyroParam(
                init_value=torch.eye(self._bnn.size_w + self._bnn.size_b),
                constraint=constraints.lower_cholesky,
            )
            prior_wb = lambda self: dist.MultivariateNormal(
                loc=self.prior_wb_loc,
                scale_tril=self.prior_wb_scale_tril,
            )
            return prior_wb

    def _prior_wb_scale_init(self) -> torch.tensor:
        assert self._prior_type == "factorized_normal"
        if self._prior_init == "standard_normal":
            return torch.ones((self._bnn.size_w + self._bnn.size_b))
        if self._prior_init == "as_pytorch_linear":
            return self._prior_wb_scale_init_pytorch_linear()

    def _prior_wb_scale_init_pytorch_linear(self) -> torch.tensor:
        """
        Initialize the prior scale according to torch.Linear.
        torch.Linear initializes the weights of a layer with in_features according to
        U(-sqrt(k), +sqrt(k)) where k = 1/sqrt(in_features).
        We use a Gaussian prior with the same standard deviation.
        The standard deviation of U(-sqrt(k), sqrt(k)) is sqrt(1/12)*(2*sqrt(k)).
        """

        def compute_scale(in_features):
            k = 1 / in_features
            scale = math.sqrt(1 / 12) * (2 * math.sqrt(k))
            return scale

        init_w = []
        init_b = []
        # input layer
        init_w.append(compute_scale(self._d_x) * torch.ones(self._d_x * self._d_hidden))
        init_b.append(compute_scale(self._d_x) * torch.ones(self._d_hidden))
        # hidden layers
        for _ in range(self._n_hidden - 1):
            init_w.append(
                compute_scale(self._d_hidden) * torch.ones(self._d_hidden ** 2)
            )
            init_b.append(compute_scale(self._d_hidden) * torch.ones(self._d_hidden))
        # output layer
        init_w.append(
            compute_scale(self._d_hidden) * torch.ones(self._d_hidden * self._d_y)
        )
        init_b.append(compute_scale(self._d_hidden) * torch.ones(self._d_y))

        init_w = torch.cat(init_w)
        init_b = torch.cat(init_b)
        init_scale = torch.cat([init_w, init_b])
        return init_scale

    def _initialize_guide(
        self,
        x: torch.tensor,
        y: torch.tensor,
    ):
        assert self._prior_type == "factorized_normal"
        guide = AutoNormal(self).to(self._device)

        # run guide once with dummy data (x, y could be empty)
        guide(
            x=torch.randn((x.shape[0], 1, x.shape[2]), device=self._device),
            y=torch.randn((y.shape[0], 1, y.shape[2]), device=self._device),
        )

        if self._posterior_init == "set_to_prior":
            # TODO: why are the guide-values not changed if I set them in a loop _wb[i][0] = ...
            guide.locs._wb = (
                self.prior_wb_loc.expand(guide.locs._wb.shape).detach().clone()
            )
            guide.scales._wb = (
                self.prior_wb_scale.expand(guide.scales._wb.shape).detach().clone()
            )

        return guide

    @property
    def prior_wb(self) -> dist.Distribution:
        return self._prior_wb(self)

    @property
    def prior_noise_stddev(self) -> dist.Distribution:
        return self._prior_noise_stddev if self._infer_noise_stddev else None

    def set_noise_stddev(self, noise_stddev: float) -> None:
        assert not self._infer_noise_stddev
        self._noise_stddev = torch.tensor(noise_stddev, device=self._device)

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

    def forward(
        self, x: torch.tensor, y: Optional[torch.tensor] = None
    ) -> torch.tensor:
        # shapes
        assert x.ndim == 3
        if y is not None:
            assert y.ndim == 3
        n_tasks = x.shape[0]
        n_points = x.shape[1]

        noise_stddev = self._noise_stddev  # (sample) noise stddev
        with pyro.plate("tasks", n_tasks, dim=-2):
            # sample weights and biases
            wb = self._wb
            w = wb[..., : self._bnn.size_w]
            b = wb[..., self._bnn.size_w :]
            # add samples- and tasks-dim
            x, w, b = broadcast_xwb(x=x, w=w, b=b)
            mean = self._bnn(x=x, w=w, b=b)

            if noise_stddev.nelement() > 1:
                # noise stddev can have a sample dimension! -> expand to mean's shape
                noise_stddev = noise_stddev.reshape([-1] + [1] * (mean.ndim - 1))
                noise_stddev = noise_stddev.expand(mean.shape)
            with pyro.plate("data", n_points, dim=-1):
                obs = pyro.sample(
                    "obs", dist.Normal(mean, noise_stddev).to_event(1), obs=y
                )  # score mean predictions against ground truth
        return mean

    def export_onnx(self, f) -> None:
        raise NotImplementedError  # not all ops we need are supported by onnx yet
        torch.onnx.export(
            model=self,
            args=(torch.randn((2, 3, 4)), torch.randn((2, 3, 4))),
            f=f,
        )

    def meta_train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_epochs: int,
        initial_lr: float,
        final_lr: float,
        alpha_reg: float,
        wandb_run,
    ) -> np.ndarray:
        self.unfreeze_prior()
        self.train()

        # generate meta-guide
        pyro.clear_param_store()  # to forget old guide shapes
        guide = self._initialize_guide(
            x=torch.tensor(x, dtype=torch.float, device=self._device),
            y=torch.tensor(y, dtype=torch.float, device=self._device),
        )
        epoch_losses = _train_model_svi(
            model=self,
            guide=guide,
            x=torch.tensor(x, dtype=torch.float, device=self._device),
            y=torch.tensor(y, dtype=torch.float, device=self._device),
            n_epochs=n_epochs,
            initial_lr=initial_lr,
            final_lr=final_lr,
            alpha_reg=alpha_reg,
            wandb_run=wandb_run,
            log_identifier="meta_train",
        )

        self.eval()
        self.freeze_prior()

        return epoch_losses.numpy(), guide

    def meta_train_monte_carlo(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_epochs: int,
        n_samples: int,
        initial_lr: float,
        final_lr: float,
        wandb_run,
    ) -> np.ndarray:
        self.unfreeze_prior()
        self.train()

        pyro.clear_param_store()
        epoch_losses = _train_model_monte_carlo(
            model=self,
            x=torch.tensor(x, dtype=torch.float, device=self._device),
            y=torch.tensor(y, dtype=torch.float, device=self._device),
            n_epochs=n_epochs,
            n_samples=n_samples,
            initial_lr=initial_lr,
            final_lr=final_lr,
            wandb_run=wandb_run,
            log_identifier="meta_train_monte_carlo",
        )

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
        wandb_run,
    ) -> np.ndarray:
        self.train()

        # generate new guide
        pyro.clear_param_store()  # to forget old guide shapes
        guide = self._initialize_guide(
            x=torch.tensor(x, dtype=torch.float, device=self._device),
            y=torch.tensor(y, dtype=torch.float, device=self._device),
        )

        n_context = x.shape[-2]
        if n_context == 0:
            epoch_losses = np.array([])
        else:
            epoch_losses = _train_model_svi(
                model=self,
                guide=guide,
                x=torch.tensor(x, dtype=torch.float, device=self._device),
                y=torch.tensor(y, dtype=torch.float, device=self._device),
                n_epochs=n_epochs,
                initial_lr=initial_lr,
                final_lr=final_lr,
                alpha_reg=0.0,
                wandb_run=wandb_run,
                log_identifier=f"adapt",
            ).numpy()

        self.eval()

        return epoch_losses, guide

    @torch.no_grad()
    def marginal_log_likelihood(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        guide: Optional[PyroModule],
    ):
        if guide is not None:
            assert guide.plates["tasks"].size == x.shape[0]

        if x.size > 0:
            marg_ll = (
                _marginal_log_likelihood(
                    model=self,
                    guide=guide,
                    x=torch.tensor(x, dtype=torch.float, device=self._device),
                    y=torch.tensor(y, dtype=torch.float, device=self._device),
                    n_samples=n_samples,
                )
                .cpu()
                .numpy()
            )
        else:
            marg_ll = np.nan

        return marg_ll

    @torch.no_grad()
    def predict(
        self,
        x: np.ndarray,
        n_samples: int,
        guide: Optional[PyroModule],
    ) -> dict:
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
                "obs",
                "_RETURN",
                "_wb",
                "_noise_stddev",
            ),
        )
        samples = predictive(
            x=torch.tensor(x, dtype=torch.float, device=self._device), y=None
        )
        samples = {k: v.detach().cpu().numpy() for k, v in samples.items()}

        return samples
