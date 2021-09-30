import multiprocessing
import wandb

COUNT = -1


def run_experiment():
    global COUNT
    COUNT += 1
    config = {"p": 0.1}

    wandb_mode = "offline"
    if wandb_mode == "online":
        wandb.login()
    with wandb.init(project="wandb_test", config=config, mode=wandb_mode) as wandb_run:
        config = wandb.config

        print(f"metric = {COUNT:2d}")
        wandb.log({"metric": COUNT})


def main():
    run_experiment()

    # sweep_config = {
    #     "name": "test_sweep",
    #     "program": "wandb_multiprocessing.py",
    #     "method": "grid",
    #     "parameters": {"lr": {"values": [0.1, 0.2, 0.3]}},
    # }
    # sweep_id = wandb.sweep(sweep_config, project="wandb_test")
    # wandb.agent(sweep_id=sweep_id, function=run_experiment)


if __name__ == "__main__":
    main()
