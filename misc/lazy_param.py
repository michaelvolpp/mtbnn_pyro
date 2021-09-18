import torch
import pyro
from pyro import distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample, pyro_method
from pyro.infer import SVI
from torch.optim import Adam


def print_parameters():
    for p, v in pyro.get_param_store().items():
        print(f"{p} = {v}")


class TestModule(PyroModule):
    def __init__(self):
        super().__init__()

        self.init()
        # self.p = PyroParam(init_value=torch.ones(2))
        # self.s1 = PyroSample(prior=lambda self: dist.Normal(loc=self.p, scale=0.0001))
        # self.s2 = PyroSample(prior=dist.Normal(loc=self.p, scale=0.0001))

    @pyro_method
    def init(self):
        self.p = PyroParam(init_value=torch.ones(2))
        self.s1 = PyroSample(prior=lambda self: dist.Normal(loc=self.p, scale=0.0001))
        self.s2 = PyroSample(prior=dist.Normal(loc=self.p, scale=0.0001))

    def forward(self):
        return self.s1

    @pyro_method
    def trigger_param_store(self):
        self.p

    def train(self):
        optim = Adam(params=self.parameters(), lr=0.01)
        for i in range(1000):
            optim.zero_grad()
            loss = ((self() - torch.tensor([-0.5, 0.5])) ** 2).sum()
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print(f"{i:03d} | loss = {loss.item():.4f}")


def main():
    pyro.set_rng_seed(123)

    m = TestModule()
    print("before calling m:")
    print_parameters()

    print("after calling m:")
    m()
    print_parameters()

    print("after triggering param store:")
    m.trigger_param_store()
    print_parameters()

    print("after training:")
    m.train()
    print_parameters()

    # print("before:")
    # print_parameters()

    # s = m()
    # print("after:")
    # print_parameters()
    # print(s)

    # m.change_p()
    # s = m()
    # print("after2:")
    # print_parameters()
    # print(s)
    # pass

    # p = pyro.param(name="p", init_tensor=torch.ones(2))
    # s1 = pyro.sample(name="s1", fn=dist.Normal(loc=p, scale=0.0001))
    # s2 = pyro.sample(name="s2", fn=dist.Normal(loc=param_store["p"], scale=0.0001))
    # s3 = pyro.sample(
    #     name="s3", fn=lambda: dist.Normal(loc=param_store["p"], scale=0.0001)
    # )
    # print(p)
    # print(param_store["p"])
    # print(s1)
    # print(s2)
    # print(s3.loc)
    # print()

    # param_store["p"] = 2 * torch.ones(2)
    # print(p)
    # print(param_store["p"])
    # print(s1)
    # print(s2)
    # print(s3.loc)


if __name__ == "__main__":
    main()
