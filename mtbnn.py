"""
Implementation of a multi-task Bayesian neural network.
"""

from typing import List, Optional

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

### PyroNotes:
## Learned PyroParams
# https://docs.pyro.ai/en/dev/nn.html#pyro.nn.module.PyroSample
# https://forum.pyro.ai/t/getting-estimates-of-parameters-that-use-pyrosample/2901/2
## Plates with explicit independent dimensions
# https://pyro.ai/examples/tensor_shapes.html#Declaring-independent-dims-with-plate


def _create_mtbnn_module(
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


def _compute_kl_regularizer(model: PyroModule):
    # TODO: make prior a proper distribution, not a list of factor distributions!
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
    regularizer_fn = _compute_kl_regularizer
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


class MultiTaskBayesianLinear(PyroModule):
    """
    A multi-task Bayesian linear layer.
    TODO: adapt for variable numbers of batch dimensions
          -> use also for single-task case.
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
        self.net = _create_mtbnn_module(
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

        ## the guides
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
