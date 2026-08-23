"""Evaluation utilities for APO."""

import random
import re
from contextlib import nullcontext
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from peft import PeftModelForCausalLM, PeftModel
from tqdm import tqdm
from sklearn.metrics import classification_report
import wandb

from config import APOConfig
from wandb_utils import log_metrics


JUDGE_PROMPT = """You are an impartial judge evaluating the quality of AI assistant responses.

Given a prompt and two responses (A and B), determine which response is better based on:
1. Helpfulness and relevance
2. Accuracy and factuality
3. Clarity and coherence
4. Safety and harmlessness

Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Answer with ONLY "A", "B", or "TIE" (if roughly equal).
"""


def generate_response(model, tokenizer: PreTrainedTokenizerBase, prompt: str, max_new_tokens: int = 256, batched: bool = False) -> str:
    """Generate a response from the model."""
    if batched:
        inputs = [tokenizer.apply_chat_template(p, return_tensors="pt", add_generation_prompt=True, return_dict=True, enable_thinking=False).to(model.device) for p in prompt]
        # left pad to longest
        max_len = max([input['input_ids'].shape[1] for input in inputs])
        for i in range(len(inputs)):
            pad_len = max_len - inputs[i]['input_ids'].shape[1]
            if pad_len > 0:
                pad_tensor = torch.full((inputs[i]['input_ids'].shape[0], pad_len), tokenizer.pad_token_id, dtype=inputs[i]['input_ids'].dtype).to(model.device)
                inputs[i]['input_ids'] = torch.cat([pad_tensor, inputs[i]['input_ids']], dim=1)
                inputs[i]['attention_mask'] = torch.cat([torch.zeros((inputs[i]['attention_mask'].shape[0], pad_len), dtype=inputs[i]['attention_mask'].dtype).to(model.device), inputs[i]['attention_mask']], dim=1)
        input_ids = torch.cat([input['input_ids'] for input in inputs], dim=0)
        input_attention_mask = torch.cat([input['attention_mask'] for input in inputs], dim=0)
        input_dict = {
            "input_ids": input_ids,
            "attention_mask": input_attention_mask,
        }
    else:
        input_dict = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True, return_dict=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **input_dict,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
        )

    if batched:
        responses = []
        for i, outputs_i in enumerate(outputs):
            response = tokenizer.decode(outputs_i[input_dict["input_ids"][i].shape[0]:], skip_special_tokens=True)
            if "</think>" in response:
                response = response.split("</think>")[1]
            responses.append(response.strip())
        return responses
    else:
        response = tokenizer.decode(outputs[0][input_dict["input_ids"].shape[1]:], skip_special_tokens=True)
        if "</think>" in response:
            response = response.split("</think>")[1]
        return response.strip()


def llm_judge_evaluate(
    config: APOConfig,
    model_probe: PeftModelForCausalLM,
    model_original: PeftModelForCausalLM,
    compare_with_sft: bool,
    tokenizer,
    eval_prompts: List[str],
    batched_generate: Optional[int] = None,  # None means not batching
) -> Dict:
    """Evaluate models using LLM-as-a-judge."""
    print("\n" + "="*50)
    print("LLM-as-a-Judge Evaluation")
    print("="*50)

    results = {"probe_wins": 0, "original_wins": 0, "ties": 0, "details": []}

    print("Generating probe responses...")
    probe_responses = []
    model_probe.set_adapter("po_probe")
    model_probe.eval()
    if batched_generate is not None:
        for i in tqdm(range(0, len(eval_prompts), batched_generate), desc="Probe responses (batched)"):
            batch_prompts = eval_prompts[i:i+batched_generate]
            responses = generate_response(
                model_probe,
                tokenizer,
                batch_prompts,
                batched=batched_generate is not None
            )
            for response in responses:
                if "<|im_end|>" in response or "<|im_start|>" in response:
                    response = re.split(r"<\|im_end\|>|<\|im_start\|>", response)[0]
                probe_responses.append(response.strip())
    else:
        for prompt in tqdm(eval_prompts, desc="Probe responses"):
            response = generate_response(model_probe, tokenizer, prompt)
            # split in case of "<|im_end|>" or "<|im_start|>"
            # (should not happen because of stop_strings, but just in case)
            if "<|im_end|>" in response or "<|im_start|>" in response:
                response = re.split(r"<\|im_end\|>|<\|im_start\|>", response)[0]
            probe_responses.append(response)
    print("Example probe response to check for degeneration issues:")
    print(probe_responses[0] if probe_responses else "No probe responses generated.")
    print("Generating original responses...")
    original_responses = []
    if hasattr(model_original, "set_adapter") and "po_original" in model_original.peft_config:
        model_original.set_adapter("po_original")
    model_original.eval()
    # model original is not a peft model so it should not have any adapters. To be 100% safe we disable any adapter from the probe model
    original_context = model_probe.disable_adapter() if compare_with_sft else nullcontext()
    with original_context:
        if batched_generate is not None:
            for i in tqdm(range(0, len(eval_prompts), batched_generate), desc="Original responses (batched)"):
                batch_prompts = eval_prompts[i:i+batched_generate]
                responses = generate_response(
                    model_original,
                    tokenizer,
                    batch_prompts,
                    batched=batched_generate is not None
                )
                for response in responses:
                    if "<|im_end|>" in response or "<|im_start|>" in response:
                        response = re.split(r"<\|im_end\|>|<\|im_start\|>", response)[0]
                    original_responses.append(response.strip())
        else:
            for prompt in tqdm(eval_prompts, desc="Original responses"):
                response = generate_response(model_original, tokenizer, prompt)
                # split in case of "<|im_end|>" or "<|im_start|>"
                if "<|im_end|>" in response or "<|im_start|>" in response:
                    response = re.split(r"<\|im_end\|>|<\|im_start\|>", response)[0]
                original_responses.append(response)
    print("Example original response to check for degeneration issues:")
    print(original_responses[0] if original_responses else "No original responses generated.")
    print("Unloading adapters...")
    if hasattr(model_probe, 'unload'):
        model_probe.unload()
    if hasattr(model_original, 'unload'):
        model_original.unload()
    del model_probe
    del model_original
    torch.cuda.empty_cache()

    print(f"Loading judge model: {config.judge_model}")
    judge_tokenizer = AutoTokenizer.from_pretrained(config.judge_model)
    judge_model = AutoModelForCausalLM.from_pretrained(
        config.judge_model,
        dtype="auto",
        device_map="auto",
    )

    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token

    print("Judging all responses...")
    if batched_generate is not None:
        # here, given judge prompts are larger, we use half the batch size
        batched_judge_size = max(1, batched_generate // 2)
    else:
        batched_judge_size = 1

    for i in tqdm(range(0, len(eval_prompts), batched_judge_size), desc="Judging responses"):
        batch_prompts = eval_prompts[i:i+batched_judge_size]
        batch_probe_responses = probe_responses[i:i+batched_judge_size]
        batch_original_responses = original_responses[i:i+batched_judge_size]

        judge_inputs = []
        mappings = []
        for j in range(len(batch_prompts)):
            prompt = batch_prompts[j]
            response_probe = batch_probe_responses[j]
            response_original = batch_original_responses[j]

            if random.random() > 0.5:
                resp_a, resp_b = response_probe, response_original
                mapping = {"A": "probe", "B": "original"}
            else:
                resp_a, resp_b = response_original, response_probe
                mapping = {"A": "original", "B": "probe"}

            judge_prompt = JUDGE_PROMPT.format(
                prompt=prompt,
                response_a=resp_a,
                response_b=resp_b,
            )
            judge_inputs.append([{"role": "user", "content": judge_prompt}])
            mappings.append(mapping)

        verdicts = generate_response(
            judge_model,
            judge_tokenizer,
            judge_inputs,
            max_new_tokens=2048,
            batched=True
        )

        for j, verdict in enumerate(verdicts):
            verdict = verdict.strip()
            mapping = mappings[j]
            prompt = batch_prompts[j]
            response_probe = batch_probe_responses[j]
            response_original = batch_original_responses[j]

            if verdict.lower().startswith("a"):
                winner = mapping["A"]
            elif verdict.lower().startswith("b"):
                winner = mapping["B"]
            else:
                winner = "tie"

            if winner == "probe":
                results["probe_wins"] += 1
            elif winner == "original":
                results["original_wins"] += 1
            else:
                results["ties"] += 1

            results["details"].append({
                "prompt": prompt,
                "response_probe": response_probe,
                "response_original": response_original,
                "verdict": verdict,
                "winner": winner,
            })

        continue  # skip to next batch

    total = len(eval_prompts)
    results["probe_win_rate"] = results["probe_wins"] / total
    results["original_win_rate"] = results["original_wins"] / total
    results["tie_rate"] = results["ties"] / total

    print("\nResults:")
    print(f"  Probe-labeled model wins: {results['probe_wins']} ({results['probe_win_rate']:.2%})")
    print(f"  Original-labeled model wins: {results['original_wins']} ({results['original_win_rate']:.2%})")
    print(f"  Ties: {results['ties']} ({results['tie_rate']:.2%})")

    log_metrics({
        "eval/probe_wins": results["probe_wins"],
        "eval/original_wins": results["original_wins"],
        "eval/ties": results["ties"],
        "eval/probe_win_rate": results["probe_win_rate"],
        "eval/original_win_rate": results["original_win_rate"],
        "eval/tie_rate": results["tie_rate"],
        "eval/total_samples": total,
    })

    if wandb.run is not None:
        table = wandb.Table(columns=["prompt", "response_probe", "response_original", "winner"])
        for detail in results["details"]:
            table.add_data(
                detail["prompt"][:200],
                detail["response_probe"][:500],
                detail["response_original"][:500],
                detail["winner"],
            )
        wandb.log({"eval/comparison_table": table})

    del judge_model
    torch.cuda.empty_cache()

    return results


def ground_truth_evaluate(
    config: APOConfig,
    model_probe: PeftModelForCausalLM,
    model_original: PeftModelForCausalLM,
    tokenizer,
    eval_dataset: List[Dict],
) -> Dict:
    """Evaluate models against ground truth labels for classification tasks."""
    print("\n" + "="*50)
    print("Ground Truth Evaluation")
    print("="*50)

    results = {
        "probe_correct": 0,
        "original_correct": 0,
        "probe_predictions": [],
        "original_predictions": [],
        "ground_truth": [],
        "details": []
    }

    def extract_label(response_text: str) -> str:
        """Extract the label from a response."""
        if isinstance(response_text, list) and len(response_text) > 0:
            response_text = response_text[0].get("content", "")

        patterns = [
            r"sentiment.*?is\s+(\w+)",
            r"is\s+(\w+)\s*\.?\s*$",
            r"^(\w+)\s*\.?\s*$",
        ]

        response_lower = response_text.lower().strip()
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                return match.group(1)

        for sentiment in ["positive", "negative", "neutral"]:
            if sentiment in response_lower:
                return sentiment

        return "unknown"

    print("Generating probe model predictions...")
    for item in tqdm(eval_dataset, desc="Probe predictions"):
        prompt = item["prompt"]
        ground_truth_label = extract_label(item["chosen"])

        response = generate_response(model_probe, tokenizer, prompt, max_new_tokens=50)
        predicted_label = extract_label(response)

        results["probe_predictions"].append(predicted_label)
        results["ground_truth"].append(ground_truth_label)

        if predicted_label == ground_truth_label:
            results["probe_correct"] += 1

    print("Generating original model predictions...")
    for item in tqdm(eval_dataset, desc="Original predictions"):
        prompt = item["prompt"]
        ground_truth_label = extract_label(item["chosen"])

        response = generate_response(model_original, tokenizer, prompt, max_new_tokens=50)
        predicted_label = extract_label(response)

        results["original_predictions"].append(predicted_label)

        if predicted_label == ground_truth_label:
            results["original_correct"] += 1

        results["details"].append({
            "prompt": prompt,
            "ground_truth": ground_truth_label,
            "probe_prediction": results["probe_predictions"][len(results["original_predictions"])-1],
            "original_prediction": predicted_label,
            "probe_correct": results["probe_predictions"][len(results["original_predictions"])-1] == ground_truth_label,
            "original_correct": predicted_label == ground_truth_label,
        })

    total = len(eval_dataset)
    results["probe_accuracy"] = results["probe_correct"] / total if total > 0 else 0
    results["original_accuracy"] = results["original_correct"] / total if total > 0 else 0
    results["total_samples"] = total

    print("\nResults:")
    print(f"  Probe model accuracy: {results['probe_correct']}/{total} ({results['probe_accuracy']:.2%})")
    print(f"  Original model accuracy: {results['original_correct']}/{total} ({results['original_accuracy']:.2%})")
    print(f"  Improvement: {results['probe_accuracy'] - results['original_accuracy']:.2%}")

    try:
        print("\n--- Probe Model Classification Report ---")
        probe_report = classification_report(
            results["ground_truth"],
            results["probe_predictions"],
            output_dict=True,
            zero_division=0
        )
        print(classification_report(results["ground_truth"], results["probe_predictions"], zero_division=0))

        print("\n--- Original Model Classification Report ---")
        original_report = classification_report(
            results["ground_truth"],
            results["original_predictions"],
            output_dict=True,
            zero_division=0
        )
        print(classification_report(results["ground_truth"], results["original_predictions"], zero_division=0))

        results["probe_report"] = probe_report
        results["original_report"] = original_report

    except Exception as e:
        print(f"Could not generate classification report: {e}")

    log_metrics({
        "eval/probe_accuracy": results["probe_accuracy"],
        "eval/original_accuracy": results["original_accuracy"],
        "eval/accuracy_improvement": results["probe_accuracy"] - results["original_accuracy"],
        "eval/probe_correct": results["probe_correct"],
        "eval/original_correct": results["original_correct"],
        "eval/total_samples": total,
    })

    if wandb.run is not None:
        table = wandb.Table(columns=[
            "prompt", "ground_truth", "probe_prediction",
            "original_prediction", "probe_correct", "original_correct"
        ])
        for detail in results["details"]:
            table.add_data(
                str(detail["prompt"])[:200],
                detail["ground_truth"],
                detail["probe_prediction"],
                detail["original_prediction"],
                detail["probe_correct"],
                detail["original_correct"],
            )
        wandb.log({"eval/predictions_table": table})

        if "probe_report" in results:
            for label, metrics in results["probe_report"].items():
                if isinstance(metrics, dict):
                    log_metrics({
                        f"eval/probe_{label}_precision": metrics.get("precision", 0),
                        f"eval/probe_{label}_recall": metrics.get("recall", 0),
                        f"eval/probe_{label}_f1": metrics.get("f1-score", 0),
                    })
            for label, metrics in results["original_report"].items():
                if isinstance(metrics, dict):
                    log_metrics({
                        f"eval/original_{label}_precision": metrics.get("precision", 0),
                        f"eval/original_{label}_recall": metrics.get("recall", 0),
                        f"eval/original_{label}_f1": metrics.get("f1-score", 0),
                    })

    if hasattr(model_probe, 'unload'):
        model_probe.unload()
    if hasattr(model_original, 'unload'):
        model_original.unload()
    del model_probe
    del model_original
    torch.cuda.empty_cache()

    return results


def evaluate_checkpoint_pair(
    config: APOConfig,
    probe_checkpoint_path: str,
    original_checkpoint_path: str,
    tokenizer,
    eval_data: List[Dict],
    base_model,
    use_ground_truth: bool = False,
    batched_generate: Optional[int] = None,
) -> Dict:
    """Evaluate a single pair of checkpoints."""
    print(f"\nEvaluating checkpoint pair:")
    print(f"  Probe: {probe_checkpoint_path}")
    print(f"  Original: {original_checkpoint_path}")

    model_probe = PeftModel.from_pretrained(
        base_model,
        probe_checkpoint_path,
        adapter_name="po_probe",
    )

    if original_checkpoint_path != "sft":
        model_original = PeftModel.from_pretrained(
            base_model,
            original_checkpoint_path,
            adapter_name="po_original",
        )
        compare_with_sft = False
    else:
        model_original = base_model
        compare_with_sft = True

    if use_ground_truth:
        results = ground_truth_evaluate(
            config, model_probe, model_original, tokenizer, eval_data
        )
    else:
        eval_prompts = [item["prompt"] for item in eval_data]
        results = llm_judge_evaluate(
            config, model_probe, model_original, compare_with_sft, tokenizer, eval_prompts, batched_generate=batched_generate
        )

    return results


def evaluate_checkpoints(
    config: APOConfig,
    probe_checkpoint_paths: List[str],
    original_checkpoint_paths: List[str],
    tokenizer,
    eval_data: List[Dict],
    base_model,
    batched_generate: Optional[int] = None,
    save_generations: Optional[str] = None,
) -> Dict:
    """Evaluate all checkpoint cross-combinations (matrix evaluation)."""
    print("\n" + "="*60)
    print("Checkpoint Cross-Comparison Matrix Evaluation")
    print("="*60)

    if config.checkpoint_eval_samples < len(eval_data):
        checkpoint_eval_data = random.sample(eval_data, config.checkpoint_eval_samples)
        print(f"Using {config.checkpoint_eval_samples} samples for checkpoint evaluation")
    else:
        checkpoint_eval_data = eval_data

    use_ground_truth = "afrisenti" in config.po_dataset.lower()

    probe_checkpoints = [(config.checkpoint_intervals[i], path) for i, path in enumerate(probe_checkpoint_paths)]
    original_checkpoints = [(config.checkpoint_intervals[i], path) for i, path in enumerate(original_checkpoint_paths)] if original_checkpoint_paths else [(1.0, "sft")]

    results = {
        "intervals": config.checkpoint_intervals,
        "matrix_results": [],
        "probe_fixed_comparisons": {},
        "original_fixed_comparisons": {},
    }

    total_comparisons = len(probe_checkpoints) * len(original_checkpoints)
    print(f"\nEvaluating {total_comparisons} checkpoint pairs:")
    print(f"  {len(probe_checkpoints)} probe checkpoints × {len(original_checkpoints)} original checkpoints\n")

    comparison_count = 0

    for probe_interval, probe_path in probe_checkpoints:
        probe_results = []

        print(f"\n{'='*60}")
        print(f"Evaluating Probe@{probe_interval:.0%} against all Original checkpoints")
        print(f"{'='*60}")

        for orig_interval, orig_path in original_checkpoints:
            comparison_count += 1
            print(f"\n[{comparison_count}/{total_comparisons}] Probe@{probe_interval:.0%} vs Original@{orig_interval:.0%}")

            checkpoint_results = evaluate_checkpoint_pair(
                config,
                probe_path,
                orig_path,
                tokenizer,
                checkpoint_eval_data,
                base_model,
                use_ground_truth=use_ground_truth,
                batched_generate=batched_generate,
            )

            if use_ground_truth:
                probe_score = checkpoint_results.get("probe_accuracy", 0)
                original_score = checkpoint_results.get("original_accuracy", 0)
                metric_name = "accuracy"
            else:
                probe_score = checkpoint_results.get("probe_win_rate", 0)
                original_score = checkpoint_results.get("original_win_rate", 0)
                metric_name = "win_rate"

            result_entry = {
                "probe_interval": probe_interval,
                "original_interval": orig_interval,
                "probe_path": probe_path,
                "original_path": orig_path,
                "probe_score": probe_score,
                "original_score": original_score,
                "improvement": probe_score - original_score,
                "full_results": checkpoint_results,
            }

            probe_results.append(result_entry)
            results["matrix_results"].append(result_entry)

            log_metrics({
                f"checkpoint_matrix/probe_{int(probe_interval*100)}_vs_original_{int(orig_interval*100)}_probe_{metric_name}": probe_score,
                f"checkpoint_matrix/probe_{int(probe_interval*100)}_vs_original_{int(orig_interval*100)}_original_{metric_name}": original_score,
                f"checkpoint_matrix/probe_{int(probe_interval*100)}_vs_original_{int(orig_interval*100)}_improvement": probe_score - original_score,
            })

        results["probe_fixed_comparisons"][f"probe_{int(probe_interval*100)}"] = probe_results

    if save_generations is not None:
        max_probe_interval = max(pi for pi, _ in probe_checkpoints)
        max_orig_interval = max(oi for oi, _ in original_checkpoints)
        final_result = next(
            (r for r in results["matrix_results"]
             if r["probe_interval"] == max_probe_interval and r["original_interval"] == max_orig_interval),
            None,
        )
        if final_result is not None:
            details = final_result["full_results"].get("details", [])
            import json as _json
            with open(save_generations, "w") as f:
                for entry in details:
                    f.write(_json.dumps({
                        "prompt": entry["prompt"],
                        "response_probe": entry["response_probe"],
                        "response_original": entry["response_original"],
                    }) + "\n")
            print(f"\nSaved {len(details)} generations to {save_generations}")
        else:
            print("\nWARNING: Could not find final checkpoint result to save generations.")

    print("\n" + "="*80)
    print("Cross-Comparison Matrix Summary")
    print("="*80)

    orig_intervals = [oi for oi, _ in original_checkpoints]
    col_label = "Probe↓ \\ Orig→"
    header = f"{col_label:<15}"
    for oi in orig_intervals:
        header += f"  {oi:>6.0%}"
    print(header)
    print("-" * 80)

    for probe_interval, _ in probe_checkpoints:
        row = f"  {probe_interval:>6.0%}         "
        for orig_interval in orig_intervals:
            result = next((r for r in results["matrix_results"]
                          if r["probe_interval"] == probe_interval and r["original_interval"] == orig_interval), None)
            if result:
                improvement = result["improvement"]
                row += f"  {improvement:>+6.3f}"
            else:
                row += "      --"
        print(row)

    print("\n(Values show improvement: probe_score - original_score)")

    if wandb.run is not None:
        table_data = []
        for result in results["matrix_results"]:
            table_data.append([
                f"{result['probe_interval']:.0%}",
                f"{result['original_interval']:.0%}",
                result['probe_score'],
                result['original_score'],
                result['improvement'],
            ])

        table = wandb.Table(
            columns=["Probe Checkpoint", "Original Checkpoint", "Probe Score", "Original Score", "Improvement"],
            data=table_data
        )
        wandb.log({"checkpoint_eval/cross_comparison_table": table})

    return results
