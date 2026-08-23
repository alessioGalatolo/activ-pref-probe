"""Weights & Biases logging utilities for APO."""

from typing import Dict
import wandb
from accelerate import Accelerator
from config import APOConfig


def init_wandb(config: APOConfig):
    """Initialize wandb logging."""
    if not config.use_wandb:
        return None

    accelerator = Accelerator()
    if not accelerator.is_main_process:
        return None

    run_name = config.wandb_run_name or f"apo_{config.po_method}_{config.model_name.split('/')[-1]}"

    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=run_name,
        tags=config.wandb_tags or [config.po_method, config.probe_type],
        config={
            "model_name": config.model_name,
            "po_method": config.po_method,
            "probe_type": config.probe_type,
            "probe_layers": config.probe_layers,
            "probe_subset_size": config.probe_subset_size,
            "po_max_samples": config.po_max_samples,
            "po_epochs": config.po_epochs,
            "beta": config.beta,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "virtual_batch_size": config.virtual_batch_size,
            "max_length": config.max_length,
            "do_sft": config.do_sft,
            "seed": config.seed,
        },
    )
    return run


def log_metrics(metrics: Dict, step: int = None, prefix: str = ""):
    """Log metrics to wandb if active."""
    if wandb.run is None:
        return
    log_dict = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
    if step is not None:
        wandb.log(log_dict, step=step)
    else:
        wandb.log(log_dict)
