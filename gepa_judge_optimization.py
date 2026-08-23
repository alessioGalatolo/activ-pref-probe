"""
GEPA-based optimization of the LLM-judge instructions used by
llm_judge_analysis.py / evaluation.py.

llm_judge_analysis.py measures how often a fixed LLM-judge prompt agrees with
a preference dataset's own chosen/rejected labels -- the LLM-judge analogue
of the probe's classification_accuracy. This script treats that same
agreement rate as a fitness function and uses GEPA (reflective prompt
evolution) to evolve the judge's system instructions to maximize it: GEPA
proposes new instructions, the local judge model is shown each (prompt,
response A, response B) example under those instructions, and every miss is
turned into textual feedback ("the judge picked B but the dataset preferred
A because ...") that a reflection LM uses to rewrite the instructions.

Only the instructions component is optimized -- the per-example data
(JUDGE_USER_TEMPLATE) stays fixed, since that's the part that shouldn't
change between examples.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Sequence, TypedDict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from dataset_utils import load_preference_dataset
from evaluation import generate_response
from llm_judge_analysis import (
    JUDGE_INSTRUCTIONS,
    JUDGE_USER_TEMPLATE,
    build_afrisenti_pairs,
    judge_dataset,
    message_text,
    prompt_text,
)


class JudgeDataInst(TypedDict):
    user_content: str
    correct_letter: str
    prompt: str
    chosen: str
    rejected: str


def build_data_insts(dataset: List[Dict], seed: int) -> List[JudgeDataInst]:
    """Turn (prompt, chosen, rejected) triples into GEPA rollout inputs, with a
    fixed A/B order per example so scoring is stable across candidates."""
    rng = random.Random(seed)
    insts: List[JudgeDataInst] = []
    for item in dataset:
        p_text = prompt_text(item["prompt"])
        chosen_text = message_text(item["chosen"])
        rejected_text = message_text(item["rejected"])

        if rng.random() < 0.5:
            resp_a, resp_b, correct_letter = chosen_text, rejected_text, "A"
        else:
            resp_a, resp_b, correct_letter = rejected_text, chosen_text, "B"

        insts.append({
            "user_content": JUDGE_USER_TEMPLATE.format(prompt=p_text, response_a=resp_a, response_b=resp_b),
            "correct_letter": correct_letter,
            "prompt": p_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        })
    return insts


class LocalChatModel:
    """Wraps a local HF causal LM as a callable GEPA can use for either the
    task LM (called with a chat messages list) or the reflection LM (called
    with a plain string prompt)."""

    def __init__(self, model, tokenizer, max_new_tokens: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        return generate_response(
            self.model, self.tokenizer, messages,
            max_new_tokens=self.max_new_tokens, batched=False,
        )


def parse_verdict(response: str) -> str:
    verdict = response.strip().lower()
    if verdict.startswith("a"):
        return "A"
    if verdict.startswith("b"):
        return "B"
    return "TIE"


class JudgeInstructionAdapter(GEPAAdapter):
    """GEPA adapter that evolves the judge's system instructions against
    agreement-with-dataset-labels, the same metric judge_dataset() reports."""

    COMPONENT_NAME = "judge_instructions"

    def __init__(self, chat_model: LocalChatModel):
        self.chat_model = chat_model

    def evaluate(self, batch: Sequence[JudgeDataInst], candidate: Dict[str, str], capture_traces: bool = False):
        instructions = candidate[self.COMPONENT_NAME]
        outputs, scores = [], []
        trajectories = [] if capture_traces else None

        for data in batch:
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": data["user_content"]},
            ]
            response = self.chat_model(messages)
            winner = parse_verdict(response)
            score = 1.0 if winner == data["correct_letter"] else 0.0

            outputs.append(response)
            scores.append(score)
            if capture_traces:
                trajectories.append({"data": data, "response": response, "winner": winner})

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = []
        for traj in eval_batch.trajectories:
            data = traj["data"]
            winner = traj["winner"]
            correct = data["correct_letter"]
            if winner == correct:
                feedback = (
                    f"Correct: the judge picked response {winner}, which matches the "
                    "dataset's preferred (chosen) response."
                )
            else:
                feedback = (
                    f"Incorrect: the judge picked response {winner}, but the dataset's preferred "
                    f"(chosen) response was {correct}.\n"
                    f"Chosen response: {data['chosen'][:500]!r}\n"
                    f"Rejected response: {data['rejected'][:500]!r}\n"
                    "Revise the instructions so the judge weighs whatever actually separates "
                    "the chosen response from the rejected one in cases like this."
                )
            records.append({
                "Inputs": {"prompt": data["prompt"][:500], "response_a_vs_b": data["user_content"][:1500]},
                "Generated Outputs": traj["response"],
                "Feedback": feedback,
            })
        return {self.COMPONENT_NAME: records}


def main():
    parser = argparse.ArgumentParser(description="Optimize the LLM-judge instructions with GEPA")
    parser.add_argument("--judge-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Local HF model acting as the judge whose prompt is optimized (GEPA's task_lm).")
    parser.add_argument("--reflection-lm", type=str, default=None,
                        help="litellm model string for the teacher/reflection LM "
                             "(e.g. 'anthropic/claude-sonnet-4-5', requires the matching API key "
                             "in the environment and `pip install litellm`). If omitted, falls back "
                             "to --judge-model itself as a fully local (but weaker) reflector.")
    parser.add_argument("--dataset", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--dataset-language", type=str, default=None)
    parser.add_argument("--train-samples", type=int, default=150)
    parser.add_argument("--val-samples", type=int, default=50)
    parser.add_argument("--test-samples", type=int, default=500,
                        help="Held-out samples used only for the final before/after accuracy report.")
    parser.add_argument("--max-metric-calls", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=1600, help="Max new tokens for judge verdicts.")
    parser.add_argument("--reflection-max-new-tokens", type=int, default=2048,
                        help="Max new tokens for the local fallback reflector (ignored if --reflection-lm is set).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the final before/after report.")
    parser.add_argument("--output-dir", type=str, default="./gepa_judge_optimization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-4bit", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("GEPA JUDGE-INSTRUCTION OPTIMIZATION")
    print("=" * 70)
    print(f"Judge model (task_lm): {args.judge_model}")
    print(f"Reflection LM: {args.reflection_lm or f'{args.judge_model} (local fallback)'}")
    print(f"Dataset: {args.dataset}")
    print(f"Train/val/test samples: {args.train_samples}/{args.val_samples}/{args.test_samples}")
    print(f"Max metric calls: {args.max_metric_calls}")
    print("=" * 70)

    total_samples = args.train_samples + args.val_samples + args.test_samples
    dataset = load_preference_dataset(args.dataset, max_samples=total_samples, dataset_args=args.dataset_language)
    if "afrisenti" in args.dataset.lower():
        dataset = build_afrisenti_pairs(dataset)

    random.Random(args.seed).shuffle(dataset)
    train_raw = dataset[:args.train_samples]
    val_raw = dataset[args.train_samples:args.train_samples + args.val_samples]
    test_raw = dataset[args.train_samples + args.val_samples:total_samples]

    trainset = build_data_insts(train_raw, seed=args.seed)
    valset = build_data_insts(val_raw, seed=args.seed + 1)

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"\nLoading judge model: {args.judge_model}")
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

    chat_model = LocalChatModel(judge_model, judge_tokenizer, max_new_tokens=args.max_new_tokens)
    adapter = JudgeInstructionAdapter(chat_model)

    reflection_lm: Optional[object] = args.reflection_lm
    if reflection_lm is None:
        print("No --reflection-lm given; using the judge model itself as the reflector (fully local, but weaker).")
        reflection_lm = LocalChatModel(judge_model, judge_tokenizer, max_new_tokens=args.reflection_max_new_tokens)

    seed_candidate = {JudgeInstructionAdapter.COMPONENT_NAME: JUDGE_INSTRUCTIONS}

    print("\n" + "=" * 70)
    print("RUNNING GEPA OPTIMIZATION")
    print("=" * 70)
    print(f"Train examples: {len(trainset)}  |  Val examples: {len(valset)}")

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=args.max_metric_calls,
        run_dir=args.output_dir,
        display_progress_bar=True,
        seed=args.seed,
    )

    optimized_instructions = result.best_candidate[JudgeInstructionAdapter.COMPONENT_NAME]
    best_val_score = result.val_aggregate_scores[result.best_idx]

    print("\n" + "=" * 70)
    print("GEPA OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_score:.4f}")
    print(f"Candidates explored: {result.num_candidates}")
    print("\nOptimized judge instructions:\n" + "-" * 40)
    print(optimized_instructions)

    with open(os.path.join(args.output_dir, "optimized_instructions.txt"), "w") as f:
        f.write(optimized_instructions)

    # Held-out before/after comparison, reusing llm_judge_analysis.py's accuracy metric
    print("\n" + "=" * 70)
    print(f"HELD-OUT COMPARISON ({len(test_raw)} samples)")
    print("=" * 70)

    print("\nBaseline (seed) instructions:")
    baseline_results = judge_dataset(
        judge_model, judge_tokenizer, test_raw,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        seed=args.seed, instructions=JUDGE_INSTRUCTIONS,
    )
    print(f"  Accuracy: {baseline_results['accuracy']:.4f}  (tie rate {baseline_results['tie_rate']:.2%})")

    print("\nGEPA-optimized instructions:")
    optimized_results = judge_dataset(
        judge_model, judge_tokenizer, test_raw,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        seed=args.seed, instructions=optimized_instructions,
    )
    print(f"  Accuracy: {optimized_results['accuracy']:.4f}  (tie rate {optimized_results['tie_rate']:.2%})")

    summary = {
        "seed_instructions": JUDGE_INSTRUCTIONS,
        "optimized_instructions": optimized_instructions,
        "best_val_accuracy": best_val_score,
        "held_out_baseline_accuracy": baseline_results["accuracy"],
        "held_out_optimized_accuracy": optimized_results["accuracy"],
        "held_out_samples": len(test_raw),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
