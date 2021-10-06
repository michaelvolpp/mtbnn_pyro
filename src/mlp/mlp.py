from typing import Optional

import numpy as np
import torch
from torch.nn import Linear, Module, MSELoss, Sequential, Tanh
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


def _create_mlp(
    d_x: int,
    d_y: int,
    n_hidden: int,
    d_hidden: int,
) -> Module:
    """
    Generate a vanilla MLP Torch module.
    """

    layers = []
    if n_hidden == 0:  # linear model
        layers.append(Linear(in_features=d_x, out_features=d_y))
    else:  # fully connected MLP
        layers.append(Linear(in_features=d_x, out_features=d_hidden))
        layers.append(Tanh())
        for _ in range(n_hidden - 1):
            layers.append(Linear(in_features=d_hidden, out_features=d_hidden))
            layers.append(Tanh())
        layers.append(Linear(in_features=d_hidden, out_features=d_y))
    net = Sequential(*layers)

    return net


def _train_model_mse(
    model: Module,
    x: torch.tensor,
    y: torch.tensor,
    n_epochs: int,
    initial_lr: float,
    final_lr: Optional[float],
    wandb_run,
    log_identifier: str,
) -> torch.tensor:
    # watch model exectuion with wandb
    wandb_run.watch(model, log="all")

    ## optimizer
    params = list(model.parameters())
    optim = torch.optim.Adam(params=params, lr=initial_lr)
    if final_lr is not None:
        gamma = final_lr / initial_lr  # final learning rate will be gamma * initial_lr
        lr_decay = gamma ** (1 / n_epochs)
        lr_scheduler = ExponentialLR(optimizer=optim, gamma=lr_decay)
    else:
        lr_scheduler = None

    ## loss
    loss_fn = MSELoss()
    regularizer_fn = None

    ## training loop
    train_losses = []
    for i in range(n_epochs):
        optim.zero_grad()

        # loss
        pred = model(x)
        mse = loss_fn(pred, y)
        loss = mse

        # regularizer
        if regularizer_fn is not None:
            raise NotImplementedError
        else:
            regularizer = torch.tensor(0.0)

        # compute gradients and step
        loss.backward()
        clip_grad_norm_(params, max_norm=10.0)
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
                f"{log_identifier}/mse_n_context_{n_context:03d}": mse,
                f"{log_identifier}/regularizer_n_context_{n_context:03d}": regularizer,
            }
        )
        if i % 100 == 0 or i == len(range(n_epochs)) - 1:
            print(f"[iter {i:04d}] mse = {mse:.4e} | reg = {regularizer:.4e}")

    return torch.tensor(train_losses)


def _mse(model: Module, x: torch.tensor, y: torch.tensor) -> torch.tensor:
    """
    Computes predictive MSE of model on data (x, y).
    """
    pred = model(x)
    mse = MSELoss()(pred, y)

    return mse


class MultiLayerPerceptron(Module):
    def __init__(
        self,
        d_x: int,
        d_y: int,
        n_hidden: int,
        d_hidden: int,
    ):
        super().__init__()
        self.d_x, self.d_y, self.n_hidden, self.d_hidden = d_x, d_y, n_hidden, d_hidden
        self._reset()
        self.eval()

    def _reset(self):
        # TODO: is this correctly registered?
        self._mlp = _create_mlp(
            d_x=self.d_x,
            d_y=self.d_y,
            n_hidden=self.n_hidden,
            d_hidden=self.d_hidden,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.ndim == 3
        assert x.shape[0] == 1, "Only n_task == 1 is supported!"
        pred = self._mlp(x)
        return pred

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

        n_context = x.shape[-2]
        if n_context == 0:
            epoch_losses = np.array([])
        else:
            epoch_losses = _train_model_mse(
                model=self,
                x=torch.tensor(x, dtype=torch.float),
                y=torch.tensor(y, dtype=torch.float),
                n_epochs=n_epochs,
                initial_lr=initial_lr,
                final_lr=final_lr,
                wandb_run=wandb_run,
                log_identifier="adapt",
            ).numpy()

        self.eval()

        return epoch_losses

    @torch.no_grad()
    def mse(self, x: np.ndarray, y: np.ndarray):
        if x.size > 0:
            mse = _mse(
                model=self,
                x=torch.tensor(x, dtype=torch.float),
                y=torch.tensor(y, dtype=torch.float),
            ).numpy()
        else:
            mse = np.nan

        return mse

    @torch.no_grad()
    def predict(self, x: np.ndarray):
        return self(torch.tensor(x, dtype=torch.float))
