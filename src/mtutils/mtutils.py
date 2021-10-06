"""
Utility functions for (multi-task) Bayesian neural networks.
"""
import math
from typing import Optional, Union

import numpy as np
import pyro
import torch
from metalearning_benchmarks import Affine1D, Quadratic1D, Sinusoid
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
from scipy import integrate
from torch import nn

BM_DICT = {"Affine1D": Affine1D, "Quadratic1D": Quadratic1D, "Sinusoid1D": Sinusoid}


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
        self,
        x: torch.tensor,
        # provide this if the BatchedLinears have manual_wb == True
        w: Optional[torch.tensor] = None,
        b: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        assert not (b is None and w is not None)
        assert not (w is None and b is not None)

        w_pos, b_pos = 0, 0
        for module in self:
            if isinstance(module, BatchedLinear) and w is not None:
                x = module(
                    x=x,
                    w=w[..., w_pos : w_pos + module.size_w],
                    b=b[..., b_pos : b_pos + module.size_b],
                )
                w_pos += module.size_w
                b_pos += module.size_b
            else:
                x = module(x)

        if w is not None:
            assert w_pos == self.size_w
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
        manual_wb: bool = True,
        n_task: Optional[int] = None,
    ):
        """
        If manual_wb == False, the layer will manage its weights and biases
        automatically. I.p. it will take care of state-of-the-art initialization.
        Then, wb_batch_size has to be provided.
        If manual_wb == True, the weights and biases have to be provided in each
        forward pass.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.size_w = self.in_features * self.out_features
        self.size_b = self.out_features
        self.manual_wb = manual_wb
        self.n_task = n_task

        if not self.manual_wb:
            init_w, init_b = self._get_init_wb()
            self._w = torch.nn.Parameter(init_w, requires_grad=True)
            self._b = torch.nn.Parameter(init_b, requires_grad=True)
        else:
            assert self.n_task is None
            self._w, self._b = None, None

    def _get_init_wb(self):
        """
        Initialize weights and biases in the same way as is standard for torch.nn.Linear
        https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        """
        k = 1 / self.in_features
        # the singleton dimension is the n_pts dimension
        init_w = (torch.rand(self.n_task, 1, self.size_w) - 0.5) * math.sqrt(k)
        init_b = (torch.rand(self.n_task, 1, self.size_b) - 0.5) * math.sqrt(k)
        return init_w, init_b

    def forward(
        self,
        x: torch.tensor,
        w: Optional[torch.tensor] = None,
        b: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        """
        Explanation of dimensions:
         x = (n_task, n_pts, d_x)
         w = (n_task, 1, d_w) 
         b = (n_task, 1, d_b) 
        """
        if not self.manual_wb:
            assert (w is None) and (b is None)
            x, w, b = broadcast_xwb(x=x, w=self._w, b=self._b)

        ## reshape weight vector to a weight matrix
        w = w.reshape(tuple(w.shape[:-1]) + (self.out_features, self.in_features))

        ## check dimensions
        assert x.ndim == w.ndim - 1
        assert x.shape[:-1] == w.shape[:-2]  # same batch dimensions
        assert x.ndim == b.ndim
        assert x.shape[:-1] == b.shape[:-1]  # same batch dimensions
        assert b.shape[-1] == w.shape[-2]  # b and w are compatible

        ## compute the output
        h = torch.einsum("...hx,...x->...h", w, x)
        h = h + b

        return h


def collate_data(bm: MetaLearningBenchmark):
    x = np.zeros((bm.n_task, bm.n_points_per_task, bm.d_x))
    y = np.zeros((bm.n_task, bm.n_points_per_task, bm.d_y))
    for l, task in enumerate(bm):
        x[l, :] = task.x
        y[l, :] = task.y
    return x, y


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


def print_pyro_parameters() -> None:
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


def split_tasks(x, y, n_context):
    x_context, y_context = x[:, :n_context, :], y[:, :n_context, :]
    # TODO: use all data as target?
    x_target, y_target = x, y

    return x_context, y_context, x_target, y_target


def print_headline_string(string):
    string = "*** " + string + " ***"
    starline = "*" * len(string)

    print("\n")
    print(starline)
    print(string)
    print(starline)


def norm_area_under_curve(x, y):
    """
    Computes the area under the curve f(x) = y using the trapeziodal rule,
    normalized by the length of the domain.
    """
    if not len(np.unique(x)) == len(x):
        print(f"Warning (in norm_area_under_curve): x contains duplicate values!")
        return np.nan
    domain_length = max(x) - min(x)
    if domain_length == 0.0:
        return None

    area = integrate.trapezoid(x=x, y=y)
    normalized_area = area / domain_length

    return normalized_area


def broadcast_xwb(
    x: torch.tensor,
    w: torch.tensor,
    b: torch.tensor,
) -> Union[torch.tensor, torch.tensor, torch.tensor]:
    """
    Explanation of dimensions:
     x = (n_task, n_pts, d_x)
     w = (n_batch, n_task, 1, d_x)  # the batch dim. is optional
     b = (n_batch, n_task, 1, d_x)  # the batch dim. is optional
    """
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
        # x = x.expand([n_tasks, n_points, -1])  # x already has this shape
        w = w.expand([n_tasks, n_points, -1])
        b = b.expand([n_tasks, n_points, -1])

    return x, w, b
