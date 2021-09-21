import torch
import pyro
from pyro import distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample, pyro_method
from pyro.infer import SVI
from torch.optim import Adam
from torch.distributions import constraints
from torchviz import make_dot


def print_parameters():
    for p, v in pyro.get_param_store().items():
        print(f"{p} = {v}")


class TestModule(PyroModule):
    def __init__(self):
        super().__init__()

        self.p1 = PyroParam(
            init_value=torch.eye(3),
            constraint=constraints.lower_cholesky,
        )
        self.p2 = PyroParam(
            init_value=torch.eye(2),
            constraint=constraints.lower_cholesky,
        )
        # self.p = PyroParam(init_value=torch.block_diag(self.p1, self.p2))
        self.prior = lambda self: pyro.distributions.Normal(
            loc=self.loc,
            scale=0.0001,
        )
        self.s = PyroSample(self.prior)

    @property
    def loc(self):
        return torch.block_diag(self.p1, self.p2)

    def forward(self):
        return self.s

    def train(self):
        optim = Adam(params=self.parameters(), lr=0.1)
        for i in range(1000):
            optim.zero_grad()
            pred = self()
            loss = ((pred - 0.1 * torch.ones((5, 5))) ** 2).sum()
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print(f"{i:03d} | loss = {loss.item():.8f}")


def main():
    pyro.set_rng_seed(123)

    m = TestModule()
    print_parameters()
    print("after training:")
    m.train()
    print_parameters()


if __name__ == "__main__":
    main()
