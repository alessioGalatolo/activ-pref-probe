"""
LLM-as-a-judge relabelling analysis for APO.

Mirrors the core question probe_analysis.py asks of a linear probe -- "given
a (chosen, rejected) pair, can it correctly identify which one is chosen?"
-- but asks an LLM judge instead of training a probe on activations.

For each (prompt, chosen, rejected) triple in a preference dataset, the judge
is shown both responses (in a randomised A/B order to cancel position bias)
and picks a winner. Accuracy is how often the judge's pick agrees with the
dataset's own chosen/rejected labels -- the same "relabel agreement" metric
computed by probe_training.label_dataset_with_probe, just with an LLM judge
standing in for the probe.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dataset_utils import load_preference_dataset
from evaluation import generate_response

# The judge is split into a system-level "instructions" component and a
# per-example user template. Keeping them separate lets gepa_judge_optimization.py
# evolve just the instructions while the example-specific text stays fixed --
# GEPA optimizes named text *components*, and the instructions are the only
# component that makes sense to evolve here.
JUDGE_INSTRUCTIONS = """You are an impartial judge evaluating the quality of AI assistant responses.

Given a prompt and two responses (A and B), determine which response is better based on:
1. Helpfulness and relevance
2. Accuracy and factuality
3. Clarity and coherence
4. Safety and harmlessness

Which response is better? Answer with ONLY "A", "B"."""

JUDGE_USER_TEMPLATE = """Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}"""


def message_text(messages: List[Dict]) -> str:
    """Flatten a list of chat messages into plain text."""
    return "\n".join(m["content"] for m in messages)


def prompt_text(messages: List[Dict]) -> str:
    """Flatten prompt turns into a readable transcript."""
    if not messages:
        return ""
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def build_afrisenti_pairs(dataset: List[Dict]) -> List[Dict]:
    """Pair up positive/negative AfriSenti tweets into chosen/rejected, same
    logic probe_analysis.py uses for this dataset."""
    pos_pending, neg_pending = [], []
    pairs = []
    for point in dataset:
        if point["label"] == "positive":
            pos_pending.append(point)
        elif point["label"] == "negative":
            neg_pending.append(point)
        if pos_pending and neg_pending:
            pairs.append({
                "prompt": [],
                "chosen": [{"role": "assistant", "content": pos_pending.pop()["tweet"]}],
                "rejected": [{"role": "assistant", "content": neg_pending.pop()["tweet"]}],
            })
    return pairs


def judge_dataset(
    judge_model,
    judge_tokenizer,
    dataset: List[Dict],
    batch_size: int = 4,
    max_new_tokens: int = 16,
    seed: int = 42,
    instructions: Optional[str] = None,
) -> Dict:
    """Run the LLM judge over every pair and score it against ground truth.

    `instructions` overrides the judge's system prompt -- e.g. with the text
    GEPA evolved in gepa_judge_optimization.py. Defaults to JUDGE_INSTRUCTIONS.
    """
    if instructions is None:
        instructions = JUDGE_INSTRUCTIONS

    rng = random.Random(seed)
    correct = 0
    ties = 0
    details = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Judging"):
        batch = dataset[i:i + batch_size]

        judge_inputs = []
        batch_info = []
        for item in batch:
            p_text = prompt_text(item["prompt"])
            chosen_text = message_text(item["chosen"])
            rejected_text = message_text(item["rejected"])

            if rng.random() < 0.5:
                resp_a, resp_b = chosen_text, rejected_text
                mapping = {"A": "chosen", "B": "rejected"}
            else:
                resp_a, resp_b = rejected_text, chosen_text
                mapping = {"A": "rejected", "B": "chosen"}

            user_content = JUDGE_USER_TEMPLATE.format(prompt=p_text, response_a=resp_a, response_b=resp_b)
            judge_inputs.append([
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_content},
            ])
            batch_info.append({
                "prompt": p_text,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "mapping": mapping,
            })

        verdicts = generate_response(
            judge_model, judge_tokenizer, judge_inputs,
            max_new_tokens=max_new_tokens, batched=True,
        )

        for info, verdict in zip(batch_info, verdicts):
            verdict = verdict.strip()
            mapping = info["mapping"]

            if verdict.lower().startswith("a"):
                winner = mapping["A"]
            elif verdict.lower().startswith("b"):
                winner = mapping["B"]
            else:
                winner = "tie"

            if winner == "tie":
                ties += 1
            elif winner == "chosen":
                correct += 1

            details.append({
                "prompt": info["prompt"][:500],
                "chosen": info["chosen"][:500],
                "rejected": info["rejected"][:500],
                "verdict": verdict,
                "winner": winner,
            })

    total = len(dataset)
    non_ties = total - ties
    results = {
        "total": total,
        "correct": correct,
        "ties": ties,
        "accuracy": correct / total if total else 0.0,
        "tie_rate": ties / total if total else 0.0,
        "accuracy_excluding_ties": correct / non_ties if non_ties else 0.0,
        "details": details,
    }

    chosen_lengths = [len(d["chosen"]) for d in details]
    rejected_lengths = [len(d["rejected"]) for d in details]
    avg_chosen_len = float(np.mean(chosen_lengths)) if chosen_lengths else 0.0
    avg_rejected_len = float(np.mean(rejected_lengths)) if rejected_lengths else 0.0
    results["avg_chosen_length"] = avg_chosen_len
    results["avg_rejected_length"] = avg_rejected_len

    return results


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-judge relabelling accuracy analysis for APO")
    parser.add_argument("--judge-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset", type=str, nargs="+", default=["Anthropic/hh-rlhf"],
                        help="One or more preference datasets to evaluate (space separated).")
    parser.add_argument("--dataset-language", type=str, default=None,
                        help="Dataset-specific argument (e.g. AfriSenti language code), applied to every --dataset.")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-dir", type=str, default="./llm_judge_analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-4bit", action="store_true", help="Load the judge model in 4-bit.")
    parser.add_argument("--judge-instructions-file", type=str, default=None,
                        help="Path to a text file with custom judge instructions "
                             "(e.g. output/optimized_instructions.txt from gepa_judge_optimization.py). "
                             "Defaults to the built-in JUDGE_INSTRUCTIONS.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    instructions = JUDGE_INSTRUCTIONS
    if args.judge_instructions_file:
        with open(args.judge_instructions_file) as f:
            instructions = f.read()

    print("=" * 70)
    print("LLM-AS-A-JUDGE RELABELLING ACCURACY ANALYSIS")
    print("=" * 70)
    print(f"Judge model: {args.judge_model}")
    print(f"Datasets: {args.dataset}")
    print(f"Samples per dataset: {args.max_samples}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        quantization_config=bnb_config,
        dtype="auto",
        device_map="auto",
    )
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token
    judge_model.eval()

    all_results = {}

    for dataset_name in args.dataset:
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

        dataset = load_preference_dataset(
            dataset_name, max_samples=args.max_samples, dataset_args=args.dataset_language
        )
        if "afrisenti" in dataset_name.lower():
            dataset = build_afrisenti_pairs(dataset)

        results = judge_dataset(
            judge_model, judge_tokenizer, dataset,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            instructions=instructions,
        )

        print(f"\nJudge-original label agreement (accuracy): {results['accuracy']:.4f}")
        print(f"  Correct: {results['correct']}/{results['total']}")
        print(f"  Ties:    {results['ties']}/{results['total']} ({results['tie_rate']:.2%})")
        print(f"  Accuracy excluding ties: {results['accuracy_excluding_ties']:.4f}")
        print(f"  Avg chosen length:   {results['avg_chosen_length']:.1f}")
        print(f"  Avg rejected length: {results['avg_rejected_length']:.1f}")
        if results["avg_chosen_length"] < results["avg_rejected_length"] * 0.5:
            print("  WARNING: Chosen responses are much shorter than rejected! Possible length bias.")
        elif results["avg_rejected_length"] < results["avg_chosen_length"] * 0.5:
            print("  WARNING: Rejected responses are much shorter than chosen! Possible length bias.")

        safe_name = dataset_name.replace("/", "_")
        with open(os.path.join(args.output_dir, f"{safe_name}_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        all_results[dataset_name] = {k: v for k, v in results.items() if k != "details"}

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    if len(args.dataset) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY ACROSS DATASETS")
        print("=" * 70)
        print(f"{'Dataset':<40}{'Accuracy':>12}{'Tie rate':>12}")
        for dataset_name, res in all_results.items():
            print(f"{dataset_name:<40}{res['accuracy']:>12.4f}{res['tie_rate']:>12.2%}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
