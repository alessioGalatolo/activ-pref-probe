"""Configuration for Activation-Based Preference Optimization (APO)."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class APOConfig:
    debug: bool = False

    # Model settings
    model_name: str = "meta-llama/Llama-3.2-1B"
    use_4bit: bool = True

    # SFT settings
    do_sft: bool = False
    sft_dataset: str = "tatsu-lab/alpaca"
    sft_max_samples: int = 1000
    sft_epochs: int = 1

    # Probe settings
    probe_dataset: Optional[str] = None  # If None, uses po_dataset
    probe_dataset_args: Optional[str] = None  # Arguments for probe dataset
    probe_layers: List[int] = field(default_factory=lambda: [8, 12, 16])
    probe_subset_size: int = 1000
    probe_type: str = "logistic"  # logistic, mlp
    probe_filter_length_outliers: bool = True  # Filter probe training data to reduce length bias
    probe_confidence_threshold: float = 0.0  # Only relabel if |prob_chosen - prob_rejected| > threshold (0.0 = no filtering)

    # Preference optimization settings
    po_method: str = "dpo"  # dpo, kto, cpo, ipo
    po_dataset: str = "Anthropic/hh-rlhf"
    po_dataset_args: Optional[str] = None  # e.g., language code for multi-language datasets
    po_max_samples: int = 5000
    po_epochs: int = 1
    beta: float = 0.1
    dpo_label_smoothing: float = 0.0

    # Training options
    train_probe_only: bool = False  # Only train probe model (skip baseline)
    train_baseline_only: bool = False  # Only train baseline model (skip probe)

    # Evaluation
    baseline: str = "original"  # What to compare probe against: "original", "random", "sft"
    flip_probability: float = 0.5  # When using random baseline, probability of flipping each label
    eval_samples: int = 100
    judge_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    generate_bs: int = 4  # Batch size for LLM-as-a-judge generation

    # Checkpoint-based evaluation
    enable_checkpoint_eval: bool = False
    checkpoint_intervals: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    checkpoint_eval_samples: int = 30

    # General
    output_dir: str = "./apo_output"
    seed: int = 42
    max_length: int = None
    batch_size: int = 4
    virtual_batch_size: int = 64
    learning_rate: float = 2e-5

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "activation-preference-optimization"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
