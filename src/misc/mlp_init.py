import torch
from matplotlib import pyplot as plt
from torch.nn import Linear, Module, ReLU, Sequential, MSELoss
from torch.optim import Adam


class MLP(Module):
    def __init__(self, d_x: int, d_y: int, n_hidden: int, d_hidden: int):
        super().__init__()

        assert n_hidden > 0
        modules = []
        modules.append(Linear(in_features=d_x, out_features=d_hidden))
        modules.append(ReLU())
        for _ in range(n_hidden - 1):
            modules.append(Linear(in_features=d_hidden, out_features=d_hidden))
            modules.append(ReLU())
        modules.append(Linear(in_features=d_hidden, out_features=d_y))

        self.mlp_layers = []
        for layer in range(n_hidden):
            self.mlp_layers.append(Sequential(*modules[: 2 * (layer + 1)]))
        self.mlp_layers.append(Sequential(*modules))
        self.mlp = Sequential(*modules)

    def forward(self, x, layer=-1):
        return self.mlp_layers[layer](x)


def plot(model, x_plt, layer, ax):
    ax.plot(x_plt, model(x_plt, layer=layer).detach().numpy(), label="Prediction")
    ax.scatter(x, y, label="Data")
    ax.grid()
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")


if __name__ == "__main__":
    # constants
    n_iter = 1000
    lr = 0.01
    n_hidden = 1
    d_hidden = 8
    n_data = 10
    n_plt = 100
    x_plt = torch.linspace(-3.0, 3.0, n_plt).reshape(n_plt, 1)

    # data
    x = (torch.rand((n_data, 1)).reshape(n_data, 1) - 0.5) * 2
    y = 2 * x

    # model, optimizer, loss
    model = MLP(d_x=1, d_y=1, n_hidden=n_hidden, d_hidden=d_hidden)
    optim = Adam(params=model.parameters())
    loss_fn = MSELoss()

    # plot
    fig, axes = plt.subplots(nrows=1, ncols=n_hidden + 1, squeeze=False)
    fig.suptitle("Before Training")
    for i in range(n_hidden + 1):
        ax = axes[0, i]
        plot(model=model, x_plt=x_plt, layer=i, ax=ax)

    # train
    for i in range(n_iter):
        optim.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optim.step()
        if i % 10 == 0 or i == len(range(n_iter)) - 1:
            print(f"[iter {i:04d}] loss = {loss.item():.4f}")

    # plot
    fig, axes = plt.subplots(nrows=1, ncols=n_hidden + 1, squeeze=False)
    fig.suptitle("After Training")
    for i in range(n_hidden + 1):
        ax = axes[0, i]
        plot(model=model, x_plt=x_plt, layer=i, ax=ax)
    plt.show()
