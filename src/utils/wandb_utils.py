import os
from datetime import datetime

from loguru import logger

import wandb


def initialize_wandb(args):
    """
    Initializes WandB. Resource monitoring (GPU/CPU/RAM) starts automatically.
    """
    if not args.use_wandb:
        logger.info("WandB is disabled for this run.")
        # mode="disabled" is the standard way to turn off wandb
        # without changing the rest of your code.
        run = wandb.init(mode="disabled")
        return run

    logger.info("Initializing WandB...")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # format timestamp example: 2024_06_15_14_30_45

    # Create a descriptive name based on the config file or job name
    config_name = os.path.basename(args.config).replace(".yaml", "")
    run_name = f"{config_name}-{args.input_dir.split('/')[-1]}-{timestamp}"

    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        config=vars(args),  # Log all hyperparameters
        job_type="inference",
        tags=[f"debug_{args.debug}", f"cp_{args.cp_model_path.split('/')[-1]}"],
    )

    logger.info(f"WandB run initialized: {run.name}")
    return run
