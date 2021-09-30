import wandb
import random
import os
import pprint
from multiprocessing import Pool


def run_experiment(config, wandb_run):
    wandb_run.log({"metric": random.random()})


def main():
    default_config = {
        "bm": "bm1",
        "prior_type": "type1",
        "seed": 123,
    }
    with wandb.init(
        project=os.getenv("WANDB_PROJECT", "test_project"),
        mode=os.getenv("WANDB_MODE", "online"),
        group=os.getenv("WANDB_RUN_GROUP", None),
        job_type=os.getenv("WANDB_JOB_TYPE", None),
        config=default_config,
    ) as wandb_run:
        config = wandb_run.config
        run_experiment(config=config, wandb_run=wandb_run)


if __name__ == "__main__":
    main()
