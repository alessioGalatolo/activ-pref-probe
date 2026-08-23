"""Probe training and dataset labeling utilities."""

import json
import os
import random
from copy import deepcopy
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from config import APOConfig
from activation_extractor import ActivationExtractor
from probes import PreferenceProbe
from model_utils import get_model_layers
from wandb_utils import log_metrics


def train_preference_probe(config: APOConfig, model, tokenizer, dataset: List[Dict]):
    """Train a linear probe on model activations to predict preferences."""
    print("\n" + "="*50)
    print("Training Preference Probe")
    print("="*50)

    probe_data = random.sample(dataset, min(config.probe_subset_size * 2, len(dataset)))

    if config.probe_filter_length_outliers:
        filtered_probe_data = []
        for item in probe_data:
            chosen_len = len(item["chosen"][0]["content"])
            rejected_len = len(item["rejected"][0]["content"])

            length_ratio = max(chosen_len, rejected_len) / max(min(chosen_len, rejected_len), 1)
            if length_ratio <= 3.0:
                filtered_probe_data.append(item)

            if len(filtered_probe_data) >= config.probe_subset_size:
                break

        removed = len(probe_data) - len(filtered_probe_data)
        if removed > 0:
            print(f"Filtered {removed} probe examples with extreme length differences (>{removed/len(probe_data):.1%})")

        probe_data = filtered_probe_data

    num_layers = len(get_model_layers(model))
    # config probe layers imagine a model with 16 layers, adjust accordingly
    ratio = num_layers / 16
    probe_layers = [max(1, min(num_layers - 1, int(l * ratio))) for l in config.probe_layers]
    probe_layers = [l for l in probe_layers if l < num_layers]
    if not probe_layers:
        probe_layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]

    print(f"Using layers: {probe_layers}")

    extractor = ActivationExtractor(model, probe_layers)

    activations_chosen = []
    activations_rejected = []

    model.eval()
    for item in tqdm(probe_data, desc="Extracting activations"):
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_text = deepcopy(prompt)
        chosen_text.extend(chosen)
        chosen_enc = tokenizer.apply_chat_template(
            chosen_text,
            return_tensors="pt",
        ).to(model.device)

        rejected_text = deepcopy(prompt)
        rejected_text.extend(rejected)
        rejected_enc = tokenizer.apply_chat_template(
            rejected_text,
            return_tensors="pt",
        ).to(model.device)

        act_chosen = extractor.extract(chosen_enc)
        act_rejected = extractor.extract(rejected_enc)

        activations_chosen.append(act_chosen.float().numpy())
        activations_rejected.append(act_rejected.float().numpy())

    extractor.remove_hooks()

    X_chosen = np.vstack(activations_chosen)
    X_rejected = np.vstack(activations_rejected)

    X = np.vstack([X_chosen, X_rejected])
    y = np.concatenate([np.ones(len(X_chosen)), np.zeros(len(X_rejected))])

    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    input_dim = X.shape[1]
    probe = PreferenceProbe(config.probe_type, input_dim)
    if len(X) < 5:
        return probe, probe_layers
    acc = probe.train(X, y, model.device)

    print(f"Probe input dimension: {input_dim}")
    print(f"Probe training samples: {len(X)}")

    return probe, probe_layers


def label_dataset_with_probe(
    config: APOConfig,
    model,
    tokenizer,
    probe: PreferenceProbe,
    probe_layers: List[int],
    dataset: List[Dict],
) -> List[Dict]:
    """Use the trained probe to relabel the dataset."""
    print("\n" + "="*50)
    print("Relabeling Dataset with Probe")
    print("="*50)
    if config.probe_confidence_threshold > 0.0:
        print(f"Confidence threshold: {config.probe_confidence_threshold} (only relabel if |prob_chosen - prob_rejected| > {config.probe_confidence_threshold})")

    extractor = ActivationExtractor(model, probe_layers)
    model.eval()

    relabeled = []
    correct = 0
    total = 0
    low_confidence_count = 0

    batch_size = config.generate_bs
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Relabeling"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]

        # Prepare all chosen and rejected texts for the batch
        chosen_texts = []
        rejected_texts = []
        for item in batch:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            chosen_text = deepcopy(prompt)
            chosen_text.extend(chosen)
            chosen_texts.append(chosen_text)

            rejected_text = deepcopy(prompt)
            rejected_text.extend(rejected)
            rejected_texts.append(rejected_text)

        # Tokenize batches
        chosen_enc = tokenizer.apply_chat_template(
            chosen_texts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        rejected_enc = tokenizer.apply_chat_template(
            rejected_texts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        # Extract activations for the batch
        act_chosen = extractor.extract(chosen_enc).float().cpu().numpy()
        act_rejected = extractor.extract(rejected_enc).float().cpu().numpy()

        # Get probabilities for the batch
        prob_chosen = probe.predict_proba(act_chosen, model.device)
        prob_rejected = probe.predict_proba(act_rejected, model.device)

        # Process each item in the batch
        for i, item in enumerate(batch):
            prob_c = prob_chosen[i]
            prob_r = prob_rejected[i]

            # Calculate confidence as absolute difference in probabilities
            confidence = abs(prob_c - prob_r)

            # confidence exceeds threshold, relabel; otherwise throw away
            if confidence > config.probe_confidence_threshold:
                if prob_c >= prob_r:
                    new_chosen, new_rejected = item["chosen"], item["rejected"]
                    if prob_c > prob_r:
                        correct += 1
                else:
                    new_chosen, new_rejected = item["rejected"], item["chosen"]
            else:
                continue
                # Low confidence - keep original labels
                new_chosen, new_rejected = item["chosen"], item["rejected"]
                low_confidence_count += 1
                correct += 1  # Count as agreement since we kept original

            total += 1

            relabeled.append({
                "prompt": item["prompt"],
                "chosen": new_chosen,
                "rejected": new_rejected,
                "probe_score_chosen": float(prob_c),
                "probe_score_rejected": float(prob_r),
                "confidence": float(confidence),
            })

    extractor.remove_hooks()

    agreement = correct / total if total > 0 else 0
    print(f"Probe-original label agreement: {agreement:.4f}")

    # Calculate average confidence
    avg_confidence = np.mean([item["confidence"] for item in relabeled])
    print(f"Average confidence: {avg_confidence:.4f}")

    if config.probe_confidence_threshold > 0.0:
        print(f"Low confidence examples (kept original): {low_confidence_count}/{total} ({low_confidence_count/total:.2%})")
        print(f"High confidence examples (relabeled): {total - low_confidence_count}/{total} ({(total - low_confidence_count)/total:.2%})")

    chosen_lengths = [len(item["chosen"][0]["content"]) for item in relabeled]
    rejected_lengths = [len(item["rejected"][0]["content"]) for item in relabeled]
    avg_chosen_len = np.mean(chosen_lengths)
    avg_rejected_len = np.mean(rejected_lengths)
    print(f"Average chosen length: {avg_chosen_len:.1f}")
    print(f"Average rejected length: {avg_rejected_len:.1f}")

    if avg_chosen_len < avg_rejected_len * 0.5:
        print("WARNING: Chosen responses are much shorter than rejected! Possible length bias.")
    elif avg_rejected_len < avg_chosen_len * 0.5:
        print("WARNING: Rejected responses are much shorter than chosen! Possible length bias.")

    log_metrics({
        "probe/label_agreement": agreement,
        "probe/relabeled_samples": total,
        "probe/flipped_labels": total - correct,
        "probe/avg_confidence": avg_confidence,
        "probe/low_confidence_count": low_confidence_count,
        "probe/low_confidence_rate": low_confidence_count / total if total > 0 else 0,
        "probe/high_confidence_count": total - low_confidence_count,
        "probe/confidence_threshold": config.probe_confidence_threshold,
        "probe/avg_chosen_length": avg_chosen_len,
        "probe/avg_rejected_length": avg_rejected_len,
    })

    def is_valid_response(content: str) -> bool:
        """Check if a response is well-formed."""
        if len(content.strip()) < 10:
            return False

        malformed_patterns = [
            '<|im_end|>',
            '<|im_start|>',
            '<noinput>',
            '<|eot_id|>',
        ]
        content_lower = content.lower().strip()

        if any(pattern in content_lower and len(content.strip()) < 50 for pattern in malformed_patterns):
            return False

        if len(content.strip()) > 0:
            special_char_ratio = sum(1 for c in content if not c.isalnum() and c not in ' \n\t.,!?-') / len(content)
            if special_char_ratio > 0.5:
                return False

        return True

    filtered_relabeled = []
    filtered_count = 0

    for item in relabeled:
        chosen_content = item["chosen"][0]["content"]
        rejected_content = item["rejected"][0]["content"]

        if is_valid_response(chosen_content) and is_valid_response(rejected_content):
            filtered_relabeled.append(item)
        else:
            filtered_count += 1

    if filtered_count > 0:
        print(f"Filtered out {filtered_count} malformed examples ({filtered_count/len(relabeled):.1%})")

    if config.output_dir:
        relabeled_path = f"{config.output_dir}/relabeled_data.json"
        os.makedirs(config.output_dir, exist_ok=True)
        with open(relabeled_path, "w") as f:
            json.dump(filtered_relabeled, f, indent=2)
        print(f"Saved relabeled data to: {relabeled_path}")

    return filtered_relabeled


def random_label_dataset(
    config: APOConfig,
    dataset: List[Dict],
    flip_probability: float = 0.5,
) -> List[Dict]:
    """Randomly relabel the dataset by flipping chosen/rejected with given probability.

    Args:
        config: APOConfig object
        dataset: List of preference pairs
        flip_probability: Probability of flipping labels (default 0.5 for complete randomness)

    Returns:
        Dataset with randomly flipped labels
    """
    print("\n" + "="*50)
    print("Randomly Relabeling Dataset")
    print("="*50)
    print(f"Flip probability: {flip_probability}")

    relabeled = []
    flipped_count = 0

    for item in tqdm(dataset, desc="Random relabeling"):
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Randomly flip with given probability
        if random.random() < flip_probability:
            new_chosen, new_rejected = rejected, chosen
            flipped_count += 1
        else:
            new_chosen, new_rejected = chosen, rejected

        relabeled.append({
            "prompt": prompt,
            "chosen": new_chosen,
            "rejected": new_rejected,
        })

    flip_rate = flipped_count / len(dataset) if len(dataset) > 0 else 0
    print(f"Flipped {flipped_count}/{len(dataset)} labels ({flip_rate:.2%})")

    log_metrics({
        "random/flipped_labels": flipped_count,
        "random/total_samples": len(dataset),
        "random/flip_rate": flip_rate,
    })

    # Save random relabeled data for analysis
    if config.output_dir:
        random_path = f"{config.output_dir}/random_relabeled_data.json"
        os.makedirs(config.output_dir, exist_ok=True)
        with open(random_path, "w") as f:
            json.dump(relabeled, f, indent=2)
        print(f"Saved random relabeled data to: {random_path}")

    return relabeled
