import pyro
import torch
from pyro import poutine
from pyro import distributions as dist

def model(x):
    s = pyro.param("s", torch.tensor(0.5))
    z = pyro.sample("z", dist.Normal(x, s))
    return z**2

def main():
    old_trace = pyro.poutine.trace(model).get_trace(1.0)
    replayed_model = pyro.poutine.replay(model, trace=old_trace)
    print(bool(replayed_model(1e6) == old_trace.nodes["_RETURN"]["value"]))
    pass


if __name__ == "__main__":
    main()
