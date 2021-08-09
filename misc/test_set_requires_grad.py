import pyro
import torch
from torch.distributions import constraints
from pyro.nn import PyroModule, PyroParam, PyroSample


class TestModule(PyroModule):
    def __init__(self):
        super().__init__()
        self.p = PyroParam(
            init_tensor=torch.tensor(0.5), constraint=constraints.real
        )
        print(self.p)
        # self.freeze_parameters()
        print(self.p)

    def forward(self, x):
        return self.p * x

    def freeze_parameters(self):
        self.p.requires_grad = False


if __name__ == "__main__":
    m = TestModule()
    x = torch.tensor(2.0)
    print(m(x))
