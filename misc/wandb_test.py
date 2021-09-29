import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt

# model
class MyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w = torch.nn.Parameter(
            data=torch.tensor(config["w_init"]),
            requires_grad=True,
        )
        self.b = torch.nn.Parameter(
            data=torch.tensor(0.0),
            requires_grad=True,
        )
        self.eval()

    def forward(self, x):
        return self.w * x + self.b


# data generation
def get_data(config):
    x_train = torch.rand(size=(config["n_train_points"],))
    y_train = config["w_true"] * x_train + config["b_true"]
    x_val = torch.rand(size=(config["n_test_points"],))
    y_val = config["w_true"] * x_val + config["b_true"]
    x_test = torch.rand(size=(config["n_test_points"],))
    y_test = config["w_true"] * x_test + config["b_true"]

    return x_train, y_train, x_val, y_val, x_test, y_test


# a simple training loop
def train(model, x_train, y_train, x_val, y_val, config, wandb_run):
    wandb.watch(models=model, log="all", log_freq=10)

    model.train()
    optim = torch.optim.Adam(params=model.parameters(), lr=config["lr"])
    mse = torch.nn.MSELoss()
    for i in range(config["n_epochs"]):
        optim.zero_grad()
        pred = model(x_train)
        loss = mse(pred, y_train)
        loss.backward()
        optim.step()
        wandb_run.log({"loss": loss.item()}, step=i)

        if i % 100 == 0:
            val_loss = test(model, x=x_val, y=y_val, config=config)
            wandb_run.log({"validation_loss": val_loss.item()}, step=i)
            print(f"Epoch = {i:04d} | Loss = {loss:.4f}")

    model.eval()


# evaluate model
@torch.no_grad()
def test(model, x, y, config):
    mse = torch.nn.MSELoss()
    pred = model(x)
    test_loss = mse(pred, y)
    return test_loss


# predict
@torch.no_grad()
def predict(model, x, config):
    pred = model(x).numpy()
    return pred


def run_experiment():
    # config
    config = {
        "a_string": "asdf",
        "n_epochs": 10000,
        "lr": 1e-3,
        "w_init": 0.0,
        "n_train_points": 5,
        "n_test_points": 50,
        "w_true": 2.0,
        "b_true": -1.0,
    }

    wandb_mode = "online"
    # wandb_mode="disabled"
    if wandb_mode != "disabled":
        wandb.login()
    with wandb.init(project="wandb_test", config=config, mode=wandb_mode) as wandb_run:
        config = wandb.config

        # data
        x_train, y_train, x_val, y_val, x_test, y_test = get_data(config=config)
        x_pred = torch.linspace(-1.0, 1.0, 100)

        # generate model
        model = MyModel(config=config)

        # train and save model
        train(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            config=config,
            wandb_run=wandb_run,
        )
        with open("model.onnx", "wb") as f:
            torch.onnx.export(model=model, args=(x_train,), f=f)
        wandb_run.save("model.onnx")

        # evaluate model
        test_mse = test(model=model, x=x_test, y=y_test, config=config)
        wandb_run.log({"test_mse": test_mse.item()})

        # predict something
        pred = predict(model=model, x=x_pred, config=config)
        data = np.stack((x_pred, pred), axis=1)
        pred_table = wandb.Table(columns=["x", "prediction"], data=data)
        wandb_run.log({"predictions": pred_table})

        # plot
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8), squeeze=False)
        ax = axes[0, 0]
        ax.plot(data[:, 0], data[:, 1], label="prediction")
        ax.scatter(x_test, y_test, label="ground truth")
        ax.legend()
        ax.grid()
        wandb_run.log({"plot_predictions": fig})
        wandb_run.log({"plot_predictions_img": wandb.Image(fig)})


def main():
    # set up sweep
    sweep_config = {
        "name": "n_epochs_grid_search",
        "method": "grid",
        "parameters": {
            "a_string": {"values": ["bsdf", "csdf"]},
            "n_epochs": {"values": [1000, 10000]},
            "n_train_points": {"values": [5, 10]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="wandb_test")
    wandb.agent(sweep_id=sweep_id, function=run_experiment)


if __name__ == "__main__":
    main()
