import wandb


def main():
    wandb.login()

    with wandb.init(project="custom_metrics", mode="online") as wandb_run:
        # define custom metrics
        wandb_run.define_metric(name="train/epoch")
        wandb_run.define_metric(name="train/*", step_metric="train/epoch")
        wandb_run.define_metric(name="adapt/epoch")
        wandb_run.define_metric(name="adapt/*", step_metric="adapt/epoch")
        wandb_run.define_metric(name="eval/n_context")
        wandb_run.define_metric(name="eval/*", step_metric="eval/n_context")

        # train loop
        for i in range(10):
            wandb_run.log(
                {
                    "train/epoch": i,
                    "train/loss": 1 / (i + 1),
                    "train/accuracy": 1 - (1 / (i + 1)),
                }
            )

        # adapt loops
        for n in range(5):
            for i in range(10):
                wandb_run.log(
                    {
                        "adapt/epoch": i,
                        f"adapt/loss_{n:d}": n / (i + 1),
                        f"adapt/accuracy_{n:d}": n - (n / (i + 1)),
                    }
                )

        # eval loops
        for n in range(5):
            wandb_run.log(
                {
                    "eval/n_context": n,
                    "eval/metric1": n,
                    "eval/metric2": 2 * n,
                }
            )


if __name__ == "__main__":
    main()
