import pyro
import torch
from torch.nn import Module
from torch.distributions import constraints
from pyro.nn import PyroModule, PyroParam, PyroSample
from torch.nn.modules.loss import MSELoss
from torch.nn.parameter import Parameter
from torch.optim import Adam as AdamTorch
from pyro.optim import Adam as AdamPyro


class TestModuleTorch(Module):
    def __init__(self):
        super().__init__()
        self.m = Parameter(data=torch.tensor(0.5), requires_grad=True)
        self.b = Parameter(data=torch.tensor(0.5), requires_grad=True)

    def forward(self, x):
        return self.m * x + self.b

    def freeze_m(self):
        self.m.requires_grad = False


class TestModulePyro(PyroModule):
    def __init__(self):
        super().__init__()
        self.m = PyroParam(init_value=torch.tensor(0.5), constraint=constraints.real)
        self.b = PyroParam(init_value=torch.tensor(0.5), constraint=constraints.real)

    def forward(self, x):
        return self.m * x + self.b

    def freeze_m(self):
        self.m.requires_grad = False


if __name__ == "__main__":
    # mode
    mode = "pyro"

    # data
    m1, m2 = 1.0, -1.0
    b1, b2 = 2.0, -2.0
    x = torch.linspace(-1.0, 1.0, 100)
    y1 = m1 * x + b1
    y2 = m2 * x + b2

    # model/optimizer
    lr = 0.1
    module = TestModulePyro() if mode == "pyro" else TestModuleTorch()
    optim = AdamTorch(params=module.parameters(), lr=lr)

    # print parameters
    print("Before training:")
    for param in module.parameters():
        print(param)
    print()

    # train
    n_iter = 100
    loss_fn = MSELoss()
    for _ in range(n_iter):
        optim.zero_grad()
        loss = loss_fn(y1, module(x))
        loss.backward()
        optim.step()

    # print parameters
    print("After training on y1:")
    for param in module.parameters():
        print(param)
    print()

    # freeze slope paramter
    module.freeze_m()

    # print parameters
    print("After freezing 'm':")
    for param in module.parameters():
        print(param)
    print()

    # train on y2
    optim = AdamTorch(
        params=module.parameters(), lr=lr
    )  # avoids changes due to momentum!?
    n_iter = 100
    loss_fn = MSELoss()
    for _ in range(n_iter):
        optim.zero_grad()
        loss = loss_fn(y2, module(x))
        loss.backward()
        optim.step()

    # print parameters
    print("After training on y2:")
    for param in module.parameters():
        print(param)
    print()
