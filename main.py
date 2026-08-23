"""
Activation-Based Preference Optimization (APO)
Train a linear probe on model activations to predict preferences,
then use predicted labels for preference optimization.

Now supports AfriSenti sentiment classification dataset.
"""

import argparse
import json
import os
import random
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from trl import DPOTrainer
import wandb
import warnings
warnings.filterwarnings("ignore")

from config import APOConfig
from dataset_utils import load_preference_dataset
from wandb_utils import init_wandb
from training import run_sft, run_preference_optimization
from probe_training import train_preference_probe, label_dataset_with_probe, random_label_dataset
from evaluation import llm_judge_evaluate, ground_truth_evaluate, evaluate_checkpoints


# ============================================================================
# Main Pipeline
# ============================================================================

def main(config: APOConfig):
    """Run the full APO pipeline."""
    # Validate baseline choice
    assert config.baseline in ["original", "random", "sft"], \
        f"Invalid baseline: {config.baseline}. Must be 'original', 'random', or 'sft'."
    assert not (config.baseline == "sft" and not config.do_sft), \
        "Cannot use SFT baseline if SFT is not enabled. Use --do-sft flag."

    # Validate training options
    assert not (config.train_probe_only and config.train_baseline_only), \
        "Cannot use both --train-probe-only and --train-baseline-only. Choose one or neither."

    print("="*60)
    print("Activation-Based Preference Optimization (APO)")
    print("="*60)
    print("\nConfig:")
    print(f"  Model: {config.model_name}")
    print(f"  PO Method: {config.po_method}")
    print(f"  Probe dataset: {config.probe_dataset or config.po_dataset}")
    if config.probe_dataset_args:
        print(f"  Probe dataset args: {config.probe_dataset_args}")
    print(f"  Probe layers: {config.probe_layers}")
    print(f"  Probe subset size: {config.probe_subset_size}")
    print(f"  Probe confidence threshold: {config.probe_confidence_threshold}")
    print(f"  PO dataset: {config.po_dataset}")
    if config.po_dataset_args:
        print(f"  PO dataset args: {config.po_dataset_args}")
    print(f"  Baseline: {config.baseline}")

    # Set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize wandb
    init_wandb(config)

    # Load model and tokenizer
    print(f"\nLoading model: {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.chat_template = "{{- bos_token }}\n{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="cpu" if config.debug else "auto",
        dtype="auto",
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.stop_strings = ["<|im_end|>", "<|im_start|>"]

    if model.config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.keys():
        if hasattr(tokenizer, "_tokenizer"):
            tokenizer.tokenizer = tokenizer._tokenizer
        DPOTrainer.process_row = staticmethod(DPOTrainer.tokenize_row)

    # Determine probe dataset source
    probe_dataset_name = config.probe_dataset if config.probe_dataset else config.po_dataset
    probe_uses_separate_dataset = config.probe_dataset is not None

    # Check if SFT and PO datasets are the same (need non-overlapping splits)
    datasets_are_same = config.do_sft and (config.sft_dataset == config.po_dataset)

    # Optional SFT
    if config.do_sft:
        if datasets_are_same:
            print("\nSFT and PO datasets are the same - using non-overlapping splits")
            sft_offset = 0
        else:
            sft_offset = 0

        model = run_sft(config, model, tokenizer, sft_offset=sft_offset)
        model = model.merge_and_unload()

    # Load probe dataset (if separate from PO dataset)
    if probe_uses_separate_dataset:
        print(f"\nLoading probe dataset: {probe_dataset_name}")
        probe_full_dataset = load_preference_dataset(
            probe_dataset_name,
            max_samples=config.probe_subset_size,
            dataset_args=config.probe_dataset_args,
        )
        random.shuffle(probe_full_dataset)
        probe_dataset = probe_full_dataset[:config.probe_subset_size]

    # Load PO preference dataset
    if datasets_are_same and not probe_uses_separate_dataset:
        # All three (SFT, Probe, PO) use the same dataset - need non-overlapping splits
        total_samples = config.sft_max_samples + config.probe_subset_size + config.po_max_samples
        po_offset = config.sft_max_samples

        print(f"\nLoading {total_samples} total samples (SFT: {config.sft_max_samples}, Probe: {config.probe_subset_size}, PO: {config.po_max_samples})")

        full_dataset = load_preference_dataset(
            config.po_dataset,
            max_samples=total_samples,
            dataset_args=config.po_dataset_args,
        )

        po_full_dataset = full_dataset[po_offset:]
        random.shuffle(po_full_dataset)
        probe_dataset = po_full_dataset[:config.probe_subset_size]
        po_dataset = po_full_dataset[config.probe_subset_size:config.probe_subset_size + config.po_max_samples]
    elif not probe_uses_separate_dataset:
        # Probe and PO use same dataset, but different from SFT (or no SFT)
        print(f"\nLoading PO dataset: {config.po_dataset}")
        full_dataset = load_preference_dataset(
            config.po_dataset,
            max_samples=config.po_max_samples + config.probe_subset_size,
            dataset_args=config.po_dataset_args,
        )
        random.shuffle(full_dataset)
        probe_dataset = full_dataset[:config.probe_subset_size]
        po_dataset = full_dataset[config.probe_subset_size:config.probe_subset_size + config.po_max_samples]
    else:
        # Probe uses separate dataset, just load PO dataset
        print(f"\nLoading PO dataset: {config.po_dataset}")
        full_dataset = load_preference_dataset(
            config.po_dataset,
            max_samples=config.po_max_samples,
            dataset_args=config.po_dataset_args,
        )
        random.shuffle(full_dataset)
        po_dataset = full_dataset[:config.po_max_samples]

    print("\nDataset split:")
    if datasets_are_same and not probe_uses_separate_dataset:
        print(f"  SFT training: {config.sft_max_samples} (samples 0-{config.sft_max_samples-1})")
        print(f"  Probe training: {len(probe_dataset)} (from {config.po_dataset}, non-overlapping with SFT)")
        print(f"  PO training: {len(po_dataset)} (from {config.po_dataset}, non-overlapping with SFT and probe)")
    elif probe_uses_separate_dataset:
        print(f"  Probe training: {len(probe_dataset)} (from {probe_dataset_name})")
        print(f"  PO training: {len(po_dataset)} (from {config.po_dataset})")
    else:
        print(f"  Probe training: {len(probe_dataset)} (from {config.po_dataset})")
        print(f"  PO training: {len(po_dataset)} (from {config.po_dataset})")

    # Train preference probe (unless baseline-only mode)
    if not config.train_baseline_only:
        relabeled_path = f"{config.output_dir}/relabeled_data.json"
        if os.path.exists(relabeled_path):
            print(f"\nLoading existing relabeled dataset from: {relabeled_path}")
            with open(relabeled_path, "r") as f:
                probe_relabeled_dataset = json.load(f)
        else:
            probe, probe_layers = train_preference_probe(config, model, tokenizer, probe_dataset)

            # Relabel dataset with probe
            probe_relabeled_dataset = label_dataset_with_probe(
                config, model, tokenizer, probe, probe_layers, po_dataset
            )
    else:
        probe, probe_layers = None, None
        probe_relabeled_dataset = None

    # Train two models: one with probe labels, one with baseline labels
    print("\n" + "="*60)
    if config.train_probe_only:
        print(f"Training Probe Model Only")
    elif config.train_baseline_only:
        print(f"Training {config.baseline.capitalize()} Baseline Model Only")
    else:
        print(f"Training Models for Comparison: Probe vs {config.baseline.capitalize()}")
    print("="*60)

    # Train with probe labels (unless baseline-only mode)
    if not config.train_baseline_only:
        model_probe, probe_checkpoint_paths = run_preference_optimization(
            config, model, tokenizer, probe_relabeled_dataset,
            use_probe_labels=True, suffix="_probe"
        )
    else:
        model_probe = None
        probe_checkpoint_paths = []

    # Train with baseline labels (unless probe-only mode)
    if not config.train_probe_only:
        if config.baseline == "sft":
            # Use SFT model directly without additional training
            model_baseline = model
            baseline_checkpoint_paths = []
        elif config.baseline == "random":
            # Relabel dataset randomly
            random_relabeled_dataset = random_label_dataset(config, po_dataset, config.flip_probability)
            model_baseline, baseline_checkpoint_paths = run_preference_optimization(
                config, model, tokenizer, random_relabeled_dataset,
                use_probe_labels=False, suffix="_random"
            )
        else:  # baseline == "original"
            # Use original labels
            model_baseline, baseline_checkpoint_paths = run_preference_optimization(
                config, model, tokenizer, po_dataset,
                use_probe_labels=False, suffix="_original"
            )
    else:
        model_baseline = None
        baseline_checkpoint_paths = []

    # Prepare evaluation prompts (skip if training only one model)
    eval_data = []
    if config.train_probe_only or config.train_baseline_only:
        print("\n" + "="*60)
        print("Skipping evaluation (only one model trained)")
        print("="*60)
        results = {"note": "Evaluation skipped - only one model trained"}
    else:
        eval_data = random.sample(
            full_dataset, min(config.eval_samples, len(full_dataset))
        )

        use_ground_truth = "afrisenti" in config.po_dataset.lower()
        compare_with_sft = (config.baseline == "sft")

        if use_ground_truth:
            print("\n" + "="*60)
            print("Using Ground Truth Evaluation (Classification Dataset)")
            print("="*60)
            results = ground_truth_evaluate(
                config, model_probe, model_baseline, tokenizer, eval_data
            )
        else:
            print("\n" + "="*60)
            print("Using LLM-as-a-Judge Evaluation (Preference Dataset)")
            print("="*60)
            eval_prompts = [item["prompt"] for item in eval_data]
            results = llm_judge_evaluate(
                config, model_probe, model_baseline, compare_with_sft, tokenizer, eval_prompts, batched_generate=config.generate_bs
            )

    # Save final evaluation results
    results_path = f"{config.output_dir}/results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal evaluation results saved to: {results_path}")

    # Save artifact to wandb
    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"apo_results_{config.po_method}",
            type="results",
            description=f"APO results using {config.po_method} with {config.probe_type} probe",
        )
        artifact.add_file(results_path)
        wandb.log_artifact(artifact)

        if "probe_win_rate" in results:
            wandb.summary["final_probe_win_rate"] = results["probe_win_rate"]
            wandb.summary["final_original_win_rate"] = results["original_win_rate"]
            wandb.summary["final_tie_rate"] = results["tie_rate"]

    # Checkpoint-based evaluation if enabled
    checkpoint_results = None
    if config.enable_checkpoint_eval and probe_checkpoint_paths and eval_data:
        print(f"\n{'='*60}")
        print(f"Starting checkpoint evaluation with {len(probe_checkpoint_paths)} probe checkpoints")
        print(f"{'='*60}")

        base_model_for_eval = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="cpu" if config.debug else "auto",
            dtype="auto",
        )

        if config.do_sft:
            base_model_for_eval = model

        checkpoint_results = evaluate_checkpoints(
            config,
            probe_checkpoint_paths,
            baseline_checkpoint_paths if baseline_checkpoint_paths else ["sft"] * len(probe_checkpoint_paths),
            tokenizer,
            eval_data,
            base_model_for_eval,
            batched_generate=config.generate_bs,
        )

        checkpoint_results_path = f"{config.output_dir}/checkpoint_results.json"
        with open(checkpoint_results_path, "w") as f:
            json.dump(checkpoint_results, f, indent=2)
        print(f"\nCheckpoint results saved to: {checkpoint_results_path}")

    if wandb.run is not None:
        wandb.finish()

    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Activation-Based Preference Optimization")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode with CPU and verbose logging")

    # Model
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--use-4bit", action="store_true", default=False)

    # SFT
    parser.add_argument("--do-sft", action="store_true")
    parser.add_argument("--sft-dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--sft-max-samples", type=int, default=1000)
    parser.add_argument("--sft-epochs", type=int, default=1)

    # Probe
    parser.add_argument("--probe-dataset", type=str, default=None,
                        help="Dataset for probe training. If not specified, uses --po-dataset")
    parser.add_argument("--probe-dataset-args", type=str, default=None,
                        help="Arguments for probe dataset (e.g., language code for multi-language datasets)")
    parser.add_argument("--probe-layers", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--probe-subset-size", type=int, default=1000)
    parser.add_argument("--probe-type", type=str, choices=["logistic", "mlp"], default="logistic")
    parser.add_argument("--probe-filter-length-outliers", action="store_true", default=True,
                        help="Filter probe training data to reduce length bias")
    parser.add_argument("--no-probe-filter-length-outliers", dest="probe_filter_length_outliers",
                        action="store_false", help="Disable length outlier filtering")
    parser.add_argument("--probe-confidence-threshold", type=float, default=0.0,
                        help="Only relabel if |prob_chosen - prob_rejected| > threshold. 0.0 = no filtering (relabel all)")

    # PO
    parser.add_argument("--po-method", type=str, choices=["dpo", "kto", "cpo", "ipo"], default="dpo")
    parser.add_argument("--po-dataset", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--po-dataset-args", type=str, default=None,
                        help="Arguments for PO dataset (e.g., language code for multi-language datasets)")
    parser.add_argument("--po-max-samples", type=int, default=5000)
    parser.add_argument("--po-epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dpo-label-smoothing", type=float, default=0.0)

    # Training options
    parser.add_argument("--train-probe-only", action="store_true",
                        help="Only train the probe model (skip baseline training)")
    parser.add_argument("--train-baseline-only", action="store_true",
                        help="Only train the baseline model (skip probe training)")

    # Eval
    parser.add_argument("--baseline", type=str, choices=["original", "random", "sft"], default="original",
                        help="Baseline to compare probe against: 'original' (human labels), 'random' (random labels), or 'sft' (no PO)")
    parser.add_argument("--flip-probability", type=float, default=0.5, help="When using random baseline, probability of flipping each label")
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--generate-bs", type=int, default=4, help="Batch size for LLM-as-a-judge generation")

    # Checkpoint-based evaluation
    parser.add_argument("--enable-checkpoint-eval", action="store_true",
                        help="Enable checkpoint-based time-series evaluation")
    parser.add_argument("--checkpoint-intervals", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0],
                        help="Training progress intervals to save checkpoints (e.g., 0.25 0.5 0.75 1.0)")
    parser.add_argument("--checkpoint-eval-samples", type=int, default=30,
                        help="Number of samples to use for checkpoint evaluations")

    # General
    parser.add_argument("--output-dir", type=str, default="./apo_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--virtual-batch-size", type=int, default=64, help="Effective batch size via gradient accumulation")
    parser.add_argument("--learning-rate", type=float, default=2e-5)

    # Wandb
    parser.add_argument("--use-wandb", type=bool, default=True)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="activation-preference-optimization")
    parser.add_argument("--wandb-entity", type=str, default=os.getenv("WANDB_ENTITY", None))
    parser.add_argument("--wandb-run-name", type=str, default=os.environ.get("RUN_NAME", None))
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=[])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Handle no_wandb flag
    if hasattr(args, 'no_wandb'):
        if args.no_wandb:
            args.use_wandb = False
        delattr(args, 'no_wandb')
    config = APOConfig(**vars(args))
    results = main(config)
