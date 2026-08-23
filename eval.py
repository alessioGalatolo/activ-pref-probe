import argparse
from contextlib import nullcontext
import json
import os
import random
import re
from typing import List, Dict, Optional
import numpy as np
import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftModelForCausalLM
from peft.helpers import check_if_peft_model
from config import APOConfig
from dataset_utils import load_preference_dataset

from evaluation import evaluate_checkpoints


# Sort by checkpoint percentage
def get_checkpoint_percentage(path):
    basename = os.path.basename(path)
    # Extract number from checkpoint_XX_probe or checkpoint_XX_original
    match = re.search(r'checkpoint_(\d+)', basename)
    if match:
        return int(match.group(1))
    return 0


def find_checkpoints(checkpoint_dir: str, po_method: str, suffix: str = "") -> List[str]:
    """Find all checkpoint directories with the given suffix."""
    # folder structure is:
    # checkpoint_dir/
    #   X_suffix/
    #     checkpoint_Y_suffix/
    checkpoints = []
    checkpoint_dir_level_1 = os.path.join(checkpoint_dir, f"{po_method}{suffix}")
    # # Find first-level checkpoint directories
    # for item in os.listdir(checkpoint_dir):
    #     item_path = os.path.join(checkpoint_dir, item)
    #     if os.path.isdir(item_path) and item.endswith(suffix):
    #         checkpoint_dir_level_1 = item_path
            # break
    if not checkpoint_dir_level_1 or not os.path.isdir(checkpoint_dir_level_1):
        return []

    # Look for checkpoint directories
    for item in os.listdir(checkpoint_dir_level_1):
        item_path = os.path.join(checkpoint_dir_level_1, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint_") and item.endswith(suffix):
            checkpoints.append(item_path)

    checkpoints.sort(key=get_checkpoint_percentage)
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints from a previous APO run")

    # Required arguments
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory containing the checkpoints (e.g., ./apo_output/dpo_probe)")
    parser.add_argument("--baseline-checkpoint-dir", type=str, default=None,
                        help="Directory containing baseline checkpoints (defaults to same as --checkpoint-dir)")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Base model name (e.g., meta-llama/Llama-3.2-1B)")
    parser.add_argument("--po-dataset", type=str, required=True,
                       help="Dataset to use for evaluation (e.g., Anthropic/hh-rlhf)")

    # Optional arguments
    parser.add_argument("--baseline", type=str, default="original", choices=["original", "sft", "random"],
                       help="Baseline type to compare against (original, sft, or random)")
    parser.add_argument("--po-dataset-args", type=str, default=None,
                       help="Arguments for the preference optimization dataset (e.g., 'amh' for AfriSenti)")
    parser.add_argument("--eval-samples", type=int, default=250,
                       help="Number of samples to use for evaluation")
    parser.add_argument("--judge-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                       help="Judge model for LLM-as-a-judge evaluation")
    parser.add_argument("--output-dir", type=str, default="./checkpoint_eval_output",
                       help="Directory to save evaluation results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--only-final", action="store_true",
                       help="Only evaluate the final checkpoint")
    parser.add_argument("--save-generations", type=str, default=None,
                       help="If given, save all generations for the final checkpoint pair to this file (JSONL)")
    parser.add_argument("--po-method", type=str, required=True)

    args = parser.parse_args()

    # Default baseline checkpoint dir to same as probe checkpoint dir
    if args.baseline_checkpoint_dir is None:
        args.baseline_checkpoint_dir = args.checkpoint_dir

    # Do not set seed (allows us to do different eval runs to add std if necessary)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    print("="*60)
    print("APO Checkpoint Evaluation")
    print("="*60)
    print(f"\nProbe checkpoint directory: {args.checkpoint_dir}")
    if args.baseline_checkpoint_dir != args.checkpoint_dir:
        print(f"Baseline checkpoint directory: {args.baseline_checkpoint_dir}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.po_dataset}")
    if args.po_dataset_args:
        print(f"Dataset Args: {args.po_dataset_args}")

    # Find checkpoint paths
    probe_checkpoint_paths = find_checkpoints(args.checkpoint_dir, args.po_method, suffix="_probe")

    if not probe_checkpoint_paths:
        print(f"\nERROR: No probe checkpoints found in {args.checkpoint_dir}")
        return
    probe_checkpoint_paths = [os.path.join(x, "po_probe") for x in probe_checkpoint_paths]

    if args.only_final:
        probe_checkpoint_paths = [probe_checkpoint_paths[-1]]

    print(f"\nFound {len(probe_checkpoint_paths)} probe checkpoints:")
    for path in probe_checkpoint_paths:
        print(f"  - {path}")

    # Find SFT path
    # sft is in: checkpoint_dir/sft/checkpoint-Y
    # need to sort descending by Y
    sft_dir = os.path.join(args.checkpoint_dir, "sft")
    sft_checkpoints = []
    for item in os.listdir(sft_dir):
        item_path = os.path.join(sft_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            sft_checkpoints.append(item_path)
    sft_checkpoints.sort(key=lambda x: int(re.search(r'checkpoint-(\d+)', os.path.basename(x)).group(1)), reverse=True)
    sft_path = sft_checkpoints[0]

    # Find original checkpoints
    original_checkpoint_paths = []
    if args.baseline != "sft":
        original_checkpoint_paths = find_checkpoints(args.baseline_checkpoint_dir, args.po_method, suffix=f"_{args.baseline}")
        original_checkpoint_paths = [os.path.join(x, f"po_{args.baseline}") for x in original_checkpoint_paths]
    else:
        # Use SFT as baseline
        original_checkpoint_paths = [sft_path] * len(probe_checkpoint_paths)

    if args.only_final:
        # take the highest checkpoint that is shared between both lists
        original_percentages = [get_checkpoint_percentage(p.split("/")[-2]) for p in original_checkpoint_paths]
        probe_percentages = [get_checkpoint_percentage(p.split("/")[-2]) for p in probe_checkpoint_paths]
        max_shared_percentage = min(max(original_percentages), max(probe_percentages))
        original_checkpoint_paths = [p for p in original_checkpoint_paths if get_checkpoint_percentage(p.split("/")[-2]) == max_shared_percentage]
        probe_checkpoint_paths = [p for p in probe_checkpoint_paths if get_checkpoint_percentage(p.split("/")[-2]) == max_shared_percentage]
        print(f"\nOnly evaluating final checkpoint at {max_shared_percentage}%.")

    if not original_checkpoint_paths:
        print("\nERROR: Found no baseline checkpoints")
        return

    if original_checkpoint_paths:
        print(f"\nFound {len(original_checkpoint_paths)} baseline checkpoints:")
        for path in original_checkpoint_paths:
            print(f"  - {path}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.chat_template = "{{- bos_token }}\n{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    print(f"\nLoading base model: {args.model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        dtype="auto",
    )

    if check_if_peft_model(sft_path):
        print(f"Detected PEFT model at SFT path: {sft_path}")
        base_model = PeftModelForCausalLM.from_pretrained(
            base_model,
            sft_path,
            device_map="auto",
            dtype="auto",
        )
        base_model = base_model.merge_and_unload()
    else:
        print(f"Loading full model from SFT path: {sft_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            sft_path,
            device_map="auto",
            dtype="auto",
        )
    base_model.generation_config.stop_strings = ["<|im_end|>", "<|im_start|>"]

    # Load evaluation dataset
    print(f"\nLoading evaluation dataset: {args.po_dataset}")
    eval_data = load_preference_dataset(
        args.po_dataset,
        split="train",
        max_samples=args.eval_samples,
        dataset_args=args.po_dataset_args,
    )
    print(f"Loaded {len(eval_data)} evaluation samples")

    # Determine evaluation type
    use_ground_truth = "afrisenti" in args.po_dataset.lower()
    eval_type = "Ground Truth (Classification)" if use_ground_truth else "LLM-as-a-Judge (Preference)"
    print(f"\nEvaluation type: {eval_type}")

    # Run evaluation
    # create mock config, only these params are used
    config = APOConfig(
        po_dataset=args.po_dataset,
        judge_model=args.judge_model,
        checkpoint_eval_samples=args.eval_samples,
        checkpoint_intervals=[1.0 * (i + 1) / len(probe_checkpoint_paths) for i in range(len(probe_checkpoint_paths))],
    )
    evaluate_checkpoints(
        config,
        probe_checkpoint_paths,
        original_checkpoint_paths,
        tokenizer,
        eval_data,
        base_model,
        batched_generate=args.batch_size,
        save_generations=args.save_generations,
    )

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
