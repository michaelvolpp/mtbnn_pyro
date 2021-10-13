import pyro
import torch
from mtutils.mtutils import print_pyro_parameters
from pyro import distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.nn.module import pyro_method
from pyro.optim import Adam
from torch._C import dtype
from torch.distributions import constraints

# TODO: use EasyGuide?

PRIOR_WB_MEAN = [1.0, 0.5]
PRIOR_WB_STDDEV = [2.0, 0.1]


class BLR(PyroModule):
    def __init__(self):
        super().__init__()

        self.prior_wb_mean = PyroParam(torch.tensor(PRIOR_WB_MEAN))
        self.prior_wb_stddev = PyroParam(torch.tensor(PRIOR_WB_STDDEV))
        self.freeze_parameters()
        self.prior_wb = lambda self: dist.Normal(
            self.prior_wb_mean,
            self.prior_wb_stddev,
        ).to_event(1)
        self.wb = PyroSample(self.prior_wb)

    def forward(self, *args, **kwargs):
        self.model(*args, **kwargs)

    @pyro_method
    def model(self, x, y):
        ## check arguments
        assert x.ndim == 2
        n_data = x.shape[0]
        d_x = x.shape[1]
        assert d_x == 1

        ## generate data
        wb = self.wb
        w, b = wb[0], wb[1]
        with pyro.plate("data", size=n_data):
            pred_mean = w * x + b
            pred_stddev = 0.1
            out_dist = dist.Normal(loc=pred_mean, scale=pred_stddev).to_event(1)
            obs = pyro.sample("obs", out_dist, obs=y)

        return pred_mean

    @pyro_method
    def freeze_parameters(self) -> None:
        """
        Freeze the unconstrained parameters.
        -> those are the leaf variables of the autograd graph
        -> those are the registered parameters of self
        """
        for p in self.parameters():
            p.requires_grad = False


def reset_guide_to_prior(model, guide):
    guide.locs.wb = model.prior_wb_mean.detach().clone()
    guide.scales.wb = model.prior_wb_stddev.detach().clone()
    pass


def main():
    n_pts = 128
    w = 2.0
    b = 3.0
    x = torch.rand((n_pts, 1))
    y = w * x + b + 0.01 * torch.randn(x.shape)

    n_epochs = 2000
    blr = BLR()
    guide = AutoNormal(model=blr)
    guide(x=x, y=y)
    print("**** Before resetting:")
    print_pyro_parameters()
    reset_guide_to_prior(model=blr, guide=guide)
    print("**** After resetting:")
    print_pyro_parameters()
    svi = SVI(
        model=blr,
        guide=guide,
        loss=Trace_ELBO(),
        optim=Adam({"lr": 0.005}),
    )
    for epoch in range(n_epochs):
        loss = svi.step(x=x, y=y)

        if epoch == 0:
            print("**** After first epoch:")
            print_pyro_parameters()
        if epoch % 100 == 0:
            print(f"{epoch = :05d} | {loss = :.4f}")

    print("**** After training:")
    print_pyro_parameters()
    reset_guide_to_prior(model=blr, guide=guide)
    print("**** After resetting:")
    print_pyro_parameters()


if __name__ == "__main__":
    main()
