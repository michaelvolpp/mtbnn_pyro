"""
Implementation of a multi-task Bayesian neural network.
"""

from typing import List, Optional, Union

import numpy as np
import pyro
import torch
from pyro import distributions as dist
from pyro.distributions import constraints
from pyro.infer import Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroParam, PyroSample
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam as AdamTorch
from torch.optim.lr_scheduler import ExponentialLR

_allowed_prior_types = [
    "isotropic_normal",
    "factorized_normal",
    "diagonal_multivariate_normal",
    "block_diagonal_multivariate_normal",
    "multivariate_normal",
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
    regularizer_fn = _compute_kl_regularizer
    loss_fn = Trace_ELBO().differentiable_loss

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


def _broadcast_xwb(
    x: torch.tensor,
    w: torch.tensor,
    b: torch.tensor,
) -> Union[torch.tensor, torch.tensor, torch.tensor]:
    ## check inputs
    assert x.ndim == 3
    n_tasks = x.shape[0]
    n_points = x.shape[1]
    assert w.ndim == 3 or w.ndim == 4
    has_sample_dim = w.ndim == 4
    assert w.ndim == b.ndim
    assert w.shape[-2] == b.shape[-2] == 1  # singleton pts-dim present due to plates
    n_samples = w.shape[0] if has_sample_dim else 1

    if has_sample_dim:
        ## add sample dim
        x = x[None, ...]

        ## broadcast
        x = x.expand([n_samples, n_tasks, n_points, -1])
        w = w.expand([n_samples, n_tasks, n_points, -1])
        b = b.expand([n_samples, n_tasks, n_points, -1])
    else:
        ## broadcast
        x = x.expand([n_tasks, n_points, -1])
        w = w.expand([n_tasks, n_points, -1])
        b = b.expand([n_tasks, n_points, -1])

    return x, w, b


class BatchedSequential(nn.Sequential):
    """
    A container akin to nn.Sequential supporting BatchedLinear layers.
    """

    def __init__(
        self,
        *args,
    ):
        super().__init__(*args)

    @property
    def size_w(self):
        size = 0
        for module in self:
            if isinstance(module, BatchedLinear):
                size += module.size_w
        return size

    @property
    def size_b(self):
        size = 0
        for module in self:
            if isinstance(module, BatchedLinear):
                size += module.size_b
        return size

    def forward(
        self, x: torch.tensor, w: torch.tensor, b: torch.tensor
    ) -> torch.tensor:
        w_pos, b_pos = 0, 0
        for module in self:
            if isinstance(module, BatchedLinear):
                x = module(
                    x=x,
                    w=w[..., w_pos : w_pos + module.size_w],
                    b=b[..., b_pos : b_pos + module.size_b] if b is not None else None,
                )
                w_pos += module.size_w
                b_pos += module.size_b
            else:
                x = module(x)

        assert w_pos == self.size_w
        if b is not None:
            assert b_pos == self.size_b

        return x


class BatchedLinear(nn.Module):
    """
    A linear layer with batched weights and bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.size_w = self.in_features * self.out_features
        self.size_b = self.out_features

    def forward(
        self,
        x: torch.tensor,
        w: torch.tensor,
        b: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        ## reshape weight vector to a weight matrix
        w = w.reshape(tuple(w.shape[:-1]) + (self.out_features, self.in_features))

        ## check dimensions
        assert x.ndim == w.ndim - 1
        assert x.shape[:-1] == w.shape[:-2]  # same batch dimensions
        if b is not None:
            assert x.ndim == b.ndim
            assert x.shape[:-1] == b.shape[:-1]  # same batch dimensions
            assert b.shape[-1] == w.shape[-2]  # b and w are compatible

        ## compute the output
        h = torch.einsum("...hx,...x->...h", w, x)
        if b is not None:
            h = h + b

        return h


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
        self._bnn = _create_batched_bnn(
            d_x=d_x,
            d_y=d_y,
            n_hidden=n_hidden,
            d_hidden=d_hidden,
        )

        ## latent variables
        self._prior_wb = self._create_bnn_priors(prior_type)
        self._wb = PyroSample(self._prior_wb)
        # TODO: learn noise prior?
        if noise_stddev is None:
            self._prior_noise_stddev = dist.Uniform(0.0, 1.0)
            self._noise_stddev = PyroSample(self._prior_noise_stddev)
        else:
            self._noise_stddev = noise_stddev

        ## the guides
        self._meta_guide = None
        self._guide = None

        ## set evaluation mode
        self.freeze_prior()
        self.eval()

    def _create_bnn_priors(self, type) -> dist.Distribution:
        assert type in _allowed_prior_types

        if type == "isotropic_normal":
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

        if type == "factorized_normal":
            self.prior_wb_loc = PyroParam(
                init_value=torch.zeros(self._bnn.size_w + self._bnn.size_b),
                constraint=constraints.real,
            )
            self.prior_wb_scale = PyroParam(
                init_value=torch.ones(self._bnn.size_w + self._bnn.size_b),
                constraint=constraints.positive,
            )
            prior_wb = lambda self: dist.Normal(
                self.prior_wb_loc, self.prior_wb_scale
            ).to_event(1)

            return prior_wb

        if type == "diagonal_multivariate_normal":
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

        if type == "block_diagonal_multivariate_normal":
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
                self.prior_wb_loc,
                scale_tril=torch.block_diag(
                    self.prior_w_scale_tril,
                    self.prior_b_scale_tril,
                ),
            )
            return prior_wb

        if type == "multivariate_normal":
            self.prior_wb_loc = PyroParam(
                init_value=torch.zeros(self._bnn.size_w + self._bnn.size_b),
                constraint=constraints.real,
            )
            self.prior_wb_scale_tril = PyroParam(
                init_value=torch.eye(self._bnn.size_w + self._bnn.size_b),
                constraint=constraints.lower_cholesky,
            )
            prior_wb = lambda self: dist.MultivariateNormal(
                self.prior_wb_loc, scale_tril=self.prior_wb_scale_tril
            )
            return prior_wb

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

    @property
    def prior_wb(self) -> dist.Distribution:
        return self._prior_wb(self)

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
            x, w, b = _broadcast_xwb(x=x, w=w, b=b)
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
