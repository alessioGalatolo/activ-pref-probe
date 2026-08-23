"""
Fine-tuned LoRA classifier baseline for APO.

Companion to `probe_gold_eval.py`, which freezes a base model and fits a
lightweight linear/MLP probe on its activations. This script instead takes a
small base LM (e.g. Qwen3-0.6B-Base), attaches a randomly-initialized
sequence-classification head, and fine-tunes it end-to-end with LoRA to
directly predict "chosen" vs. "rejected" from the raw text, using the
dataset's original preference labels as ground truth.

The scoring/metrics/plots (pointwise + pairwise accuracy, calibration, error
analysis) are shared with `probe_gold_eval.py` so the two baselines (frozen
probe vs. fine-tuned classifier) are directly comparable.

Example:
    python lora_classifier_eval.py --model-name Qwen/Qwen3-0.6B-Base 
        --dataset Anthropic/hh-rlhf --train-samples 1000 --eval-samples 500
"""

import argparse
import os
import random
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import json
import warnings
warnings.filterwarnings('ignore')

from probe_gold_eval import (
    load_and_split_dataset,
    compute_pointwise_metrics,
    compute_pairwise_metrics,
    compute_pairwise_pr_curve,
    compute_calibration,
    error_analysis,
    plot_roc_pr_curves,
    plot_pairwise_pr_curve,
    plot_calibration,
    plot_confidence_distribution,
    plot_confusion_matrix,
)


def build_classification_dataset(
    data: List[Dict], tokenizer, max_length: int
) -> HFDataset:
    """Turn each preference pair into two pointwise examples: chosen -> label 1,
    rejected -> label 0, using the dataset's original labels."""
    texts, labels = [], []
    for item in data:
        chosen_full = deepcopy(item["prompt"]) + item["chosen"]
        rejected_full = deepcopy(item["prompt"]) + item["rejected"]
        texts.append(tokenizer.apply_chat_template(chosen_full, tokenize=False))
        labels.append(1)
        texts.append(tokenizer.apply_chat_template(rejected_full, tokenize=False))
        labels.append(0)

    ds = HFDataset.from_dict({"text": texts, "label": labels})
    ds = ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )
    return ds


def score_eval_set(
    model,
    tokenizer,
    eval_data: List[Dict],
    batch_size: int,
    max_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the fine-tuned classifier over a held-out set, returning
    prob(chosen)/prob(rejected) per pair (mirrors `probe_gold_eval.score_eval_set`)."""
    device = next(model.parameters()).device
    model.eval()

    prob_chosen = np.zeros(len(eval_data))
    prob_rejected = np.zeros(len(eval_data))

    num_batches = (len(eval_data) + batch_size - 1) // batch_size
    for b in tqdm(range(num_batches), desc="Scoring eval set"):
        start = b * batch_size
        end = min(start + batch_size, len(eval_data))
        batch = eval_data[start:end]

        chosen_texts = [
            tokenizer.apply_chat_template(deepcopy(item["prompt"]) + item["chosen"], tokenize=False)
            for item in batch
        ]
        rejected_texts = [
            tokenizer.apply_chat_template(deepcopy(item["prompt"]) + item["rejected"], tokenize=False)
            for item in batch
        ]

        chosen_enc = tokenizer(
            chosen_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        rejected_enc = tokenizer(
            rejected_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            chosen_probs = torch.softmax(model(**chosen_enc).logits.float(), dim=-1)[:, 1]
            rejected_probs = torch.softmax(model(**rejected_enc).logits.float(), dim=-1)[:, 1]

        prob_chosen[start:end] = chosen_probs.cpu().numpy()
        prob_rejected[start:end] = rejected_probs.cpu().numpy()

    return prob_chosen, prob_rejected


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuned classifier baseline vs. gold labels")

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--dataset", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--dataset-language", type=str, default=None,
                         help="Dataset-specific arg (e.g. language code for oasst2/prism)")
    parser.add_argument("--train-samples", type=int, default=1000,
                         help="Number of preference pairs used for fine-tuning (2x pointwise examples)")
    parser.add_argument("--eval-samples", type=int, default=500,
                         help="Number of held-out pairs used to evaluate against gold labels")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Decision threshold for pointwise precision/recall/F1")
    parser.add_argument("--pairwise-confidence-threshold", type=float, default=0.0,
                         help="Confidence threshold |P(chosen)-P(rejected)| for the headline "
                              "pairwise precision/recall (0.0 = every pair retained)")
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--top-k-errors", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="./lora_classifier_eval")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("LORA-FINE-TUNED CLASSIFIER VS. GOLD LABEL EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Train pairs: {args.train_samples} | Eval pairs: {args.eval_samples}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    train_data, eval_data = load_and_split_dataset(
        args.dataset, args.dataset_language, args.train_samples, args.eval_samples, args.seed
    )
    print(f"Loaded {len(train_data)} train pairs / {len(eval_data)} eval pairs (disjoint)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.chat_template = "{{- bos_token }}\n{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # The classification head is randomly initialized (not part of the pretrained
    # checkpoint) so it must be fully trained, not LoRA-adapted.
    classifier_head_name = "score" if hasattr(model, "score") else "classifier"

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        modules_to_save=[classifier_head_name],
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = build_classification_dataset(train_data, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()

    prob_chosen, prob_rejected = score_eval_set(
        model, tokenizer, eval_data, args.batch_size, args.max_length
    )

    # --- Metrics (shared with probe_gold_eval.py) ---
    pointwise_metrics, y_true, y_prob = compute_pointwise_metrics(prob_chosen, prob_rejected, args.threshold)
    pairwise_metrics, diff = compute_pairwise_metrics(
        prob_chosen, prob_rejected, args.pairwise_confidence_threshold
    )
    pairwise_pr_curve = compute_pairwise_pr_curve(prob_chosen, prob_rejected)
    calibration_metrics, bin_stats = compute_calibration(y_true, y_prob, args.ece_bins)
    error_stats = error_analysis(eval_data, prob_chosen, prob_rejected, args.output_dir, args.top_k_errors)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("-- Pointwise (chosen/rejected scored as independent instances) --")
    print(f"Accuracy:           {pointwise_metrics['accuracy']:.4f}")
    print(f"Precision:          {pointwise_metrics['precision']:.4f}")
    print(f"Recall:             {pointwise_metrics['recall']:.4f}")
    print(f"F1:                 {pointwise_metrics['f1']:.4f}")
    print(f"ROC-AUC:            {pointwise_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:             {pointwise_metrics['pr_auc']:.4f}")
    print(f"Brier score:        {pointwise_metrics['brier_score']:.4f}")
    print(f"ECE:                {calibration_metrics['ece']:.4f}")
    print(f"MCE:                {calibration_metrics['mce']:.4f}")
    print(f"\n-- Pairwise (does the classifier rank gold-chosen above gold-rejected?) "
          f"[confidence threshold={args.pairwise_confidence_threshold}] --")
    print(f"Accuracy (unconditional): {pairwise_metrics['pairwise_accuracy']:.4f}  "
          f"(ties: {pairwise_metrics['tie_rate']:.4f})")
    print(f"Coverage:           {pairwise_metrics['coverage']:.4f}")
    print(f"Precision:          {pairwise_metrics['pairwise_precision']:.4f}")
    print(f"Recall:             {pairwise_metrics['pairwise_recall']:.4f}")
    print(f"F1:                 {pairwise_metrics['pairwise_f1']:.4f}")
    print(f"Pairwise error rate: {error_stats['pairwise_error_rate']:.4f} "
          f"({error_stats['n_errors']}/{error_stats['n_total']})")

    # --- Plots (shared with probe_gold_eval.py) ---
    plot_roc_pr_curves(y_true, y_prob, args.output_dir)
    plot_pairwise_pr_curve(pairwise_pr_curve, args.output_dir)
    plot_calibration(bin_stats, y_prob, args.output_dir)
    plot_confidence_distribution(prob_chosen, prob_rejected, diff > 0, args.output_dir)
    plot_confusion_matrix(pointwise_metrics['confusion_matrix'], args.output_dir)

    # --- Save everything ---
    results = {
        'config': {
            'model_name': args.model_name,
            'dataset': args.dataset,
            'train_samples': len(train_data),
            'eval_samples': len(eval_data),
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'epochs': args.epochs,
            'threshold': args.threshold,
            'pairwise_confidence_threshold': args.pairwise_confidence_threshold,
        },
        'pointwise_metrics': pointwise_metrics,
        'pairwise_metrics': pairwise_metrics,
        'pairwise_pr_curve': pairwise_pr_curve,
        'calibration': {**calibration_metrics, 'bins': bin_stats},
        'error_analysis': error_stats,
    }
    with open(os.path.join(args.output_dir, 'gold_comparison_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Saved metrics to: {args.output_dir}/gold_comparison_metrics.json")
    print(f"Saved worst errors to: {args.output_dir}/worst_errors.json")
    print("Plots: roc_pr_curves.pdf, pairwise_precision_recall_curve.pdf, calibration.pdf, "
          "confidence_distribution.pdf, confusion_matrix.pdf, error_length_bias.pdf")
    print("=" * 70)


if __name__ == "__main__":
    main()
