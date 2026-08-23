"""
Evaluation script for PRISM-specific APO runs.

Generates responses from probe and original models on PRISM prompts, then
compares them against a ground-truth embedding centroid computed from the
chosen responses of a specific PRISM group (dataset_args). Metric: cosine
similarity to the group centroid.
"""

import argparse
import json
import os
import random
import re
from contextlib import nullcontext
from typing import List, Dict, Optional

import numpy as np
import torch
from peft import PeftModel, PeftModelForCausalLM
from peft.helpers import check_if_peft_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_utils import get_formatted_prism
from evaluation import generate_response
from eval import find_checkpoints, get_checkpoint_percentage


# ──────────────────────────────────────────────────────────────
# PRISM data helpers
# ──────────────────────────────────────────────────────────────

def load_prism_eval_data(
    target_group: Optional[str],
    eval_samples: Optional[int],
    seed: int = 42,
) -> tuple[List[Dict], List[str]]:
    """
    Returns:
        eval_data   – list of preference dicts used as prompts (whole dataset or subset)
        gt_texts    – chosen-response texts from the target group (ground truth)
    """
    prism = get_formatted_prism()
    available_groups = list(prism.keys())

    # Ground-truth: chosen responses from the target group
    if target_group is None:
        raise ValueError(
            f"--po-dataset-args is required for PRISM evaluation. "
            f"Available groups: {available_groups}"
        )
    if target_group not in prism:
        raise ValueError(
            f"Group '{target_group}' not found in PRISM. "
            f"Available groups: {available_groups}"
        )

    gt_items = prism[target_group]
    gt_texts = [item["chosen"][0]["content"] for item in gt_items if "chosen" in item]
    print(f"Ground-truth group '{target_group}': {len(gt_texts)} chosen responses")

    # Eval prompts: whole dataset concatenated
    all_items = [item for items in prism.values() for item in items]

    rng = random.Random(seed)
    if eval_samples and eval_samples < len(all_items):
        all_items = rng.sample(all_items, eval_samples)

    print(f"Eval prompts: {len(all_items)} samples from all PRISM groups")
    return all_items, gt_texts


# ──────────────────────────────────────────────────────────────
# Embedding helpers
# ──────────────────────────────────────────────────────────────

def compute_centroid(texts: List[str], embed_model: SentenceTransformer) -> np.ndarray:
    """Return the mean embedding (L2-normalised) of a list of texts."""
    embeddings = embed_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    centroid = embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    return centroid  # shape (D,)


def cosine_sim_to_centroid(texts: List[str], centroid: np.ndarray, embed_model: SentenceTransformer) -> np.ndarray:
    """Cosine similarity of each text to the centroid. Returns shape (N,)."""
    embeddings = embed_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    sims = cosine_similarity(embeddings, centroid.reshape(1, -1)).squeeze(1)
    return sims  # shape (N,)


# ──────────────────────────────────────────────────────────────
# Generation helpers
# ──────────────────────────────────────────────────────────────

def _clean(response: str) -> str:
    if "<|im_end|>" in response or "<|im_start|>" in response:
        response = re.split(r"<\|im_end\|>|<\|im_start\|>", response)[0]
    return response.strip()


def generate_all_responses(
    model,
    tokenizer,
    prompts: List[List[Dict]],
    batch_size: int,
    desc: str = "Generating",
) -> List[str]:
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch = prompts[i : i + batch_size]
        if batch_size == 1:
            resp = generate_response(model, tokenizer, batch[0])
            responses.append(_clean(resp))
        else:
            batch_resp = generate_response(model, tokenizer, batch, batched=True)
            responses.extend([_clean(r) for r in batch_resp])
    return responses


# ──────────────────────────────────────────────────────────────
# Per-checkpoint evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_checkpoint_pair(
    probe_path: str,
    original_path: str,
    base_model,
    tokenizer,
    eval_prompts: List[List[Dict]],
    gt_centroid: np.ndarray,
    embed_model: SentenceTransformer,
    batch_size: int,
) -> Dict:
    """Generate probe and original responses, score vs ground-truth centroid."""

    # ── Probe responses ──────────────────────────────────────
    model_probe = PeftModel.from_pretrained(
        base_model, probe_path, adapter_name="po_probe"
    )
    model_probe.set_adapter("po_probe")
    model_probe.eval()
    probe_responses = generate_all_responses(
        model_probe, tokenizer, eval_prompts, batch_size, desc="Probe"
    )
    model_probe.unload()
    del model_probe
    torch.cuda.empty_cache()

    # ── Original responses ───────────────────────────────────
    if original_path == "sft":
        # base_model is already the SFT model; disable any remaining adapters
        model_original = base_model
        original_responses = generate_all_responses(
            model_original, tokenizer, eval_prompts, batch_size, desc="Original (SFT)"
        )
    else:
        model_original = PeftModel.from_pretrained(
            base_model, original_path, adapter_name="po_original"
        )
        model_original.set_adapter("po_original")
        model_original.eval()
        original_responses = generate_all_responses(
            model_original, tokenizer, eval_prompts, batch_size, desc="Original"
        )
        model_original.unload()
        del model_original
        torch.cuda.empty_cache()

    # ── Embedding similarity ─────────────────────────────────
    print("Computing probe embedding similarities...")
    probe_sims = cosine_sim_to_centroid(probe_responses, gt_centroid, embed_model)
    print("Computing original embedding similarities...")
    orig_sims = cosine_sim_to_centroid(original_responses, gt_centroid, embed_model)

    probe_mean = float(probe_sims.mean())
    orig_mean = float(orig_sims.mean())
    improvement = probe_mean - orig_mean

    per_sample = [
        {
            "prompt": eval_prompts[i],
            "probe_response": probe_responses[i],
            "original_response": original_responses[i],
            "probe_sim": float(probe_sims[i]),
            "original_sim": float(orig_sims[i]),
        }
        for i in range(len(eval_prompts))
    ]

    print(f"\n  Probe    cosine sim to GT centroid: {probe_mean:.4f}")
    print(f"  Original cosine sim to GT centroid: {orig_mean:.4f}")
    print(f"  Improvement (probe − original):     {improvement:+.4f}")

    return {
        "probe_mean_sim": probe_mean,
        "original_mean_sim": orig_mean,
        "improvement": improvement,
        "probe_sims": probe_sims.tolist(),
        "original_sims": orig_sims.tolist(),
        "per_sample": per_sample,
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PRISM embedding-similarity evaluation for APO checkpoints"
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory containing the checkpoints")
    parser.add_argument("--baseline-checkpoint-dir", type=str, default=None,
                        help="Directory containing baseline checkpoints (defaults to --checkpoint-dir)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Base model name (e.g., meta-llama/Llama-3.2-1B)")
    parser.add_argument("--po-dataset-args", type=str, required=True,
                        help="PRISM group to use as ground truth (e.g., '18-24_Africa')")
    parser.add_argument("--baseline", type=str, default="original",
                        choices=["original", "sft", "random"],
                        help="Baseline type to compare against")
    parser.add_argument("--eval-samples", type=int, default=None,
                        help="Number of PRISM prompts to evaluate (default: all)")
    parser.add_argument("--embed-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-Transformers model for embeddings")
    parser.add_argument("--output-dir", type=str, default="./prism_eval_output",
                        help="Directory to save evaluation results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Generation batch size")
    parser.add_argument("--only-final", action="store_true",
                        help="Only evaluate the final checkpoint")
    parser.add_argument("--po-method", type=str, required=True)
    args = parser.parse_args()

    if args.baseline_checkpoint_dir is None:
        args.baseline_checkpoint_dir = args.checkpoint_dir

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("APO PRISM Embedding-Similarity Evaluation")
    print("=" * 60)
    print(f"Probe checkpoint dir : {args.checkpoint_dir}")
    print(f"Model                : {args.model_name}")
    print(f"Target PRISM group   : {args.po_dataset_args}")
    print(f"Embedding model      : {args.embed_model}")

    # ── Load PRISM data ──────────────────────────────────────
    eval_data, gt_texts = load_prism_eval_data(
        target_group=args.po_dataset_args,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )
    eval_prompts = [item["prompt"] for item in eval_data]

    # ── Build ground-truth centroid ──────────────────────────
    print(f"\nLoading embedding model: {args.embed_model}")
    embed_model = SentenceTransformer(args.embed_model)
    print(f"Computing ground-truth centroid from {len(gt_texts)} chosen responses...")
    gt_centroid = compute_centroid(gt_texts, embed_model)

    # ── Find checkpoints ─────────────────────────────────────
    probe_checkpoint_paths = find_checkpoints(args.checkpoint_dir, args.po_method, suffix="_probe")
    if not probe_checkpoint_paths:
        print(f"\nERROR: No probe checkpoints found in {args.checkpoint_dir}")
        return
    probe_checkpoint_paths = [os.path.join(x, "po_probe") for x in probe_checkpoint_paths]

    if args.only_final:
        probe_checkpoint_paths = [probe_checkpoint_paths[-1]]

    # Find SFT path
    sft_dir = os.path.join(args.checkpoint_dir, "sft")
    sft_checkpoints = []
    for item in os.listdir(sft_dir):
        item_path = os.path.join(sft_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            sft_checkpoints.append(item_path)
    sft_checkpoints.sort(
        key=lambda x: int(re.search(r'checkpoint-(\d+)', os.path.basename(x)).group(1)),
        reverse=True,
    )
    sft_path = sft_checkpoints[0]

    original_checkpoint_paths: List[str]
    if args.baseline == "sft":
        original_checkpoint_paths = ["sft"] * len(probe_checkpoint_paths)
    else:
        original_checkpoint_paths = find_checkpoints(
            args.baseline_checkpoint_dir, args.po_method, suffix=f"_{args.baseline}"
        )
        original_checkpoint_paths = [
            os.path.join(x, f"po_{args.baseline}") for x in original_checkpoint_paths
        ]
        if args.only_final:
            original_checkpoint_paths = [original_checkpoint_paths[-1]]

    if not original_checkpoint_paths:
        print("\nERROR: Found no baseline checkpoints")
        return

    print(f"\nFound {len(probe_checkpoint_paths)} probe checkpoint(s):")
    for p in probe_checkpoint_paths:
        print(f"  {p}")
    print(f"Found {len(original_checkpoint_paths)} baseline checkpoint(s):")
    for p in original_checkpoint_paths:
        print(f"  {p}")

    # ── Load tokenizer + base model ──────────────────────────
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.chat_template = (
        "{{- bos_token }}\n"
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base model: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", dtype="auto"
    )

    if check_if_peft_model(sft_path):
        print(f"Detected PEFT model at SFT path: {sft_path}")
        base_model = PeftModelForCausalLM.from_pretrained(
            base_model, sft_path, device_map="auto", dtype="auto"
        )
        base_model = base_model.merge_and_unload()
    else:
        print(f"Loading full model from SFT path: {sft_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            sft_path, device_map="auto", dtype="auto"
        )
    base_model.generation_config.stop_strings = ["<|im_end|>", "<|im_start|>"]

    # ── Run per-checkpoint evaluation ────────────────────────
    all_results = []
    n = min(len(probe_checkpoint_paths), len(original_checkpoint_paths))

    for i in range(n):
        probe_path = probe_checkpoint_paths[i]
        orig_path = original_checkpoint_paths[i]
        probe_pct = get_checkpoint_percentage(probe_path.split("/")[-2])
        print(f"\n{'='*60}")
        print(f"[{i+1}/{n}] Checkpoint {probe_pct}%")
        print(f"  Probe    : {probe_path}")
        print(f"  Original : {orig_path}")
        print("=" * 60)

        result = evaluate_checkpoint_pair(
            probe_path=probe_path,
            original_path=orig_path,
            base_model=base_model,
            tokenizer=tokenizer,
            eval_prompts=eval_prompts,
            gt_centroid=gt_centroid,
            embed_model=embed_model,
            batch_size=args.batch_size,
        )
        result["probe_path"] = probe_path
        result["original_path"] = orig_path
        result["checkpoint_pct"] = probe_pct
        all_results.append(result)

        # Save incremental results
        out_path = os.path.join(args.output_dir, f"checkpoint_{probe_pct}.json")
        with open(out_path, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "per_sample"}, f, indent=2)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary — cosine similarity to ground-truth centroid")
    print(f"  Target group : {args.po_dataset_args}")
    print(f"  GT responses : {len(gt_texts)}")
    print(f"  Eval prompts : {len(eval_prompts)}")
    print("=" * 60)
    header = f"{'Checkpoint':<14}  {'Probe sim':>10}  {'Orig sim':>10}  {'Δ (probe−orig)':>14}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        pct = r["checkpoint_pct"]
        print(
            f"  {pct:>3}%          "
            f"  {r['probe_mean_sim']:>10.4f}"
            f"  {r['original_mean_sim']:>10.4f}"
            f"  {r['improvement']:>+14.4f}"
        )

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {
        "target_group": args.po_dataset_args,
        "embed_model": args.embed_model,
        "n_gt_texts": len(gt_texts),
        "n_eval_prompts": len(eval_prompts),
        "checkpoints": [
            {
                "checkpoint_pct": r["checkpoint_pct"],
                "probe_path": r["probe_path"],
                "original_path": r["original_path"],
                "probe_mean_sim": r["probe_mean_sim"],
                "original_mean_sim": r["original_mean_sim"],
                "improvement": r["improvement"],
            }
            for r in all_results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
