"""
Evaluate a trained preference probe against gold (human-annotated) labels.

Trains the probe using the same code path as the main pipeline
(`probe_training.train_preference_probe`, `probes.PreferenceProbe`) on a training
split, then scores a disjoint held-out split and reports how well the probe's
"chosen"/"rejected" predictions line up with the dataset's original gold labels:

  - Pairwise accuracy (does prob(chosen) > prob(rejected) for the gold pair?)
  - Pointwise precision / recall / F1 / accuracy / ROC-AUC / PR-AUC
    (treat gold-chosen activations as the positive class, gold-rejected as negative)
  - Calibration: reliability diagram, Brier score, Expected Calibration Error (ECE)
  - Confidence distribution (correct vs. incorrect pairwise predictions)
  - Error analysis: length bias in errors, worst high-confidence mistakes dumped
    to JSON for manual inspection

Example:
    python probe_gold_eval.py --model-name meta-llama/Llama-3.2-1B \\
        --dataset Anthropic/hh-rlhf --train-samples 1000 --eval-samples 500
"""

import argparse
import json
import os
import random
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')

from activation_extractor import ActivationExtractor
from config import APOConfig
from dataset_utils import load_preference_dataset
from probe_training import train_preference_probe

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)

COLORS = {
    'correct': '#2ecc71',
    'incorrect': '#e74c3c',
    'chosen': '#3498db',
    'rejected': '#e74c3c',
    'neutral': '#95a5a6',
}
PLOT_CONFIG = {'dpi': 150, 'bbox_inches': 'tight', 'facecolor': 'white', 'edgecolor': 'none'}


# =============================================================================
# Data loading / probe scoring
# =============================================================================

def load_and_split_dataset(
    dataset_name: str,
    dataset_args: str,
    train_samples: int,
    eval_samples: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Load a preference dataset and split it into disjoint train/eval sets."""
    total_needed = train_samples + eval_samples
    data = load_preference_dataset(dataset_name, max_samples=total_needed, dataset_args=dataset_args)

    if len(data) < total_needed:
        print(f"WARNING: dataset only has {len(data)} usable examples, "
              f"requested {total_needed} (train={train_samples} + eval={eval_samples}).")

    random.Random(seed).shuffle(data)
    train_data = data[:train_samples]
    eval_data = data[train_samples:train_samples + eval_samples]
    return train_data, eval_data


def score_eval_set(
    model,
    tokenizer,
    probe,
    probe_layers: List[int],
    eval_data: List[Dict],
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the trained probe over a held-out set, returning prob(chosen)/prob(rejected)
    per pair using the exact same activation path as `probe_training.label_dataset_with_probe`."""
    extractor = ActivationExtractor(model, probe_layers)
    model.eval()

    prob_chosen = np.zeros(len(eval_data))
    prob_rejected = np.zeros(len(eval_data))

    num_batches = (len(eval_data) + batch_size - 1) // batch_size
    for b in tqdm(range(num_batches), desc="Scoring eval set"):
        start = b * batch_size
        end = min(start + batch_size, len(eval_data))
        batch = eval_data[start:end]

        chosen_texts = [deepcopy(item["prompt"]) + item["chosen"] for item in batch]
        rejected_texts = [deepcopy(item["prompt"]) + item["rejected"] for item in batch]

        chosen_enc = tokenizer.apply_chat_template(
            chosen_texts, return_tensors="pt", padding=True
        ).to(model.device)
        rejected_enc = tokenizer.apply_chat_template(
            rejected_texts, return_tensors="pt", padding=True
        ).to(model.device)

        act_chosen = extractor.extract(chosen_enc).float().cpu().numpy()
        act_rejected = extractor.extract(rejected_enc).float().cpu().numpy()

        prob_chosen[start:end] = probe.predict_proba(act_chosen, model.device)
        prob_rejected[start:end] = probe.predict_proba(act_rejected, model.device)

    extractor.remove_hooks()
    return prob_chosen, prob_rejected


# =============================================================================
# Metrics
# =============================================================================

def compute_pointwise_metrics(
    prob_chosen: np.ndarray, prob_rejected: np.ndarray, threshold: float = 0.5
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Treat each gold-chosen activation as a positive example and each gold-rejected
    activation as a negative example, and score the probe as a binary classifier."""
    y_true = np.concatenate([np.ones_like(prob_chosen), np.zeros_like(prob_rejected)])
    y_prob = np.concatenate([prob_chosen, prob_rejected])
    y_pred = (y_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    metrics = {
        'threshold': threshold,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
        'brier_score': float(brier_score_loss(y_true, y_prob)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics, y_true, y_prob


def compute_pairwise_metrics(
    prob_chosen: np.ndarray, prob_rejected: np.ndarray, confidence_threshold: float = 0.0
) -> Tuple[Dict, np.ndarray]:
    """The metric that actually matters for relabeling: does the probe rank the gold
    chosen response above the gold rejected response for the same prompt?

    Pairwise precision/recall are defined via the same selective-prediction framing as
    `config.probe_confidence_threshold`: a pair is only "retained" (i.e. the probe makes
    a call) if |P(chosen) - P(rejected)| > confidence_threshold. Precision is then the
    accuracy among retained pairs (how trustworthy the probe's confident calls are), and
    recall is the fraction of *all* pairs that end up both retained and correct (how much
    of the dataset the probe can confidently and correctly relabel). At the default
    threshold of 0.0, recall reduces to plain pairwise accuracy.
    """
    diff = prob_chosen - prob_rejected
    abs_diff = np.abs(diff)
    correct = diff > 0
    ties = diff == 0
    n = len(diff)

    retained = abs_diff > confidence_threshold
    n_retained = int(retained.sum())
    n_correct_retained = int((correct & retained).sum())

    precision = n_correct_retained / n_retained if n_retained > 0 else float('nan')
    recall = n_correct_retained / n
    f1 = 2 * precision * recall / (precision + recall) if n_retained > 0 and (precision + recall) > 0 else 0.0

    metrics = {
        'confidence_threshold': confidence_threshold,
        'pairwise_accuracy': float(correct.sum() / n),
        'tie_rate': float(ties.sum() / n),
        'coverage': float(n_retained / n),
        'pairwise_precision': float(precision),
        'pairwise_recall': float(recall),
        'pairwise_f1': float(f1),
        'mean_confidence_gap': float(abs_diff.mean()),
        'median_confidence_gap': float(np.median(abs_diff)),
    }
    return metrics, diff


def compute_pairwise_pr_curve(
    prob_chosen: np.ndarray, prob_rejected: np.ndarray, n_points: int = 50
) -> Dict[str, List[float]]:
    """Sweep the confidence threshold and trace out precision/recall/coverage — a
    risk-coverage curve for the probe's pairwise (relabeling) decisions."""
    diff = prob_chosen - prob_rejected
    abs_diff = np.abs(diff)
    correct = diff > 0
    n = len(diff)

    thresholds = np.unique(np.quantile(abs_diff, np.linspace(0.0, 0.99, n_points)))

    curve = {'threshold': [], 'precision': [], 'recall': [], 'coverage': []}
    for t in thresholds:
        retained = abs_diff > t
        n_retained = int(retained.sum())
        if n_retained == 0:
            continue
        n_correct_retained = int((correct & retained).sum())
        curve['threshold'].append(float(t))
        curve['precision'].append(n_correct_retained / n_retained)
        curve['recall'].append(n_correct_retained / n)
        curve['coverage'].append(n_retained / n)

    return curve


def compute_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[Dict, List[Dict]]:
    """Reliability-diagram bin statistics plus Expected/Maximum Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)

    n = len(y_true)
    ece = 0.0
    mce = 0.0
    bin_stats = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            bin_stats.append({
                'bin_lower': float(bin_edges[b]), 'bin_upper': float(bin_edges[b + 1]),
                'count': 0, 'confidence': None, 'accuracy': None,
            })
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        gap = abs(conf - acc)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bin_stats.append({
            'bin_lower': float(bin_edges[b]), 'bin_upper': float(bin_edges[b + 1]),
            'count': count, 'confidence': conf, 'accuracy': acc,
        })

    return {'ece': float(ece), 'mce': float(mce), 'n_bins': n_bins}, bin_stats


# =============================================================================
# Plots
# =============================================================================

def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, color=COLORS['chosen'], linewidth=2.5, label=f'ROC (AUC={auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color=COLORS['neutral'], linestyle='--', linewidth=1.5, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontweight='bold')
    axes[0].set_title('ROC Curve', fontweight='bold')
    axes[0].legend(frameon=True)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[1].plot(recall, precision, color='#f39c12', linewidth=2.5, label=f'PR (AP={ap:.3f})')
    axes[1].axhline(y=0.5, color=COLORS['neutral'], linestyle='--', linewidth=1.5, label='Random')
    axes[1].set_xlabel('Recall', fontweight='bold')
    axes[1].set_ylabel('Precision', fontweight='bold')
    axes[1].set_title('Precision-Recall Curve', fontweight='bold')
    axes[1].legend(frameon=True)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_pr_curves.pdf'), **PLOT_CONFIG)
    plt.close()


def plot_pairwise_pr_curve(curve: Dict[str, List[float]], output_dir: str):
    """Risk-coverage view of the pairwise relabeling decision: as the confidence
    threshold rises, coverage drops but precision (accuracy on retained pairs) rises."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    axes[0].plot(curve['recall'], curve['precision'], color=COLORS['chosen'],
                 linewidth=2.5, marker='o', markersize=4)
    axes[0].set_xlabel('Recall (correct & retained / total pairs)', fontweight='bold')
    axes[0].set_ylabel('Precision (correct / retained pairs)', fontweight='bold')
    axes[0].set_title('Pairwise Precision-Recall Curve', fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].plot(curve['coverage'], curve['precision'], color=COLORS['incorrect'],
                 linewidth=2.5, marker='o', markersize=4)
    axes[1].set_xlabel('Coverage (fraction of pairs retained)', fontweight='bold')
    axes[1].set_ylabel('Precision (accuracy on retained pairs)', fontweight='bold')
    axes[1].set_title('Risk-Coverage Curve', fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairwise_precision_recall_curve.pdf'), **PLOT_CONFIG)
    plt.close()


def plot_calibration(bin_stats: List[Dict], y_prob: np.ndarray, output_dir: str):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), facecolor='white',
                              gridspec_kw={'height_ratios': [3, 1]})

    valid = [b for b in bin_stats if b['count'] > 0]
    centers = [(b['bin_lower'] + b['bin_upper']) / 2 for b in valid]
    confidences = [b['confidence'] for b in valid]
    accuracies = [b['accuracy'] for b in valid]

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=COLORS['neutral'], linestyle='--', linewidth=1.5, label='Perfect calibration')
    ax.bar(centers, accuracies, width=1.0 / len(bin_stats), alpha=0.7,
           color=COLORS['chosen'], edgecolor='white', label='Empirical accuracy')
    ax.plot(centers, confidences, marker='o', color=COLORS['incorrect'],
            linewidth=2, markersize=6, label='Mean predicted confidence')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted probability', fontweight='bold')
    ax.set_ylabel('Empirical accuracy', fontweight='bold')
    ax.set_title('Reliability Diagram', fontweight='bold', pad=12)
    ax.legend(frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax2 = axes[1]
    ax2.hist(y_prob, bins=len(bin_stats), range=(0, 1), color=COLORS['neutral'], alpha=0.8, edgecolor='white')
    ax2.set_xlabel('Predicted probability', fontweight='bold')
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration.pdf'), **PLOT_CONFIG)
    plt.close()


def plot_confidence_distribution(
    prob_chosen: np.ndarray, prob_rejected: np.ndarray, pairwise_correct: np.ndarray, output_dir: str
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    ax = axes[0]
    ax.hist(prob_chosen, bins=30, range=(0, 1), alpha=0.6, density=True,
            color=COLORS['chosen'], label='Gold chosen')
    ax.hist(prob_rejected, bins=30, range=(0, 1), alpha=0.6, density=True,
            color=COLORS['rejected'], label='Gold rejected')
    ax.set_xlabel('Probe P(preferred)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Confidence Distribution by Gold Label', fontweight='bold')
    ax.legend(frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    gap = prob_chosen - prob_rejected
    ax2 = axes[1]
    ax2.hist(gap[pairwise_correct], bins=30, alpha=0.6, density=True,
             color=COLORS['correct'], label='Correct pairwise')
    ax2.hist(gap[~pairwise_correct], bins=30, alpha=0.6, density=True,
             color=COLORS['incorrect'], label='Incorrect pairwise')
    ax2.axvline(x=0, color='black', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('P(chosen) - P(rejected)', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('Confidence Gap: Correct vs. Incorrect', fontweight='bold')
    ax2.legend(frameon=True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.pdf'), **PLOT_CONFIG)
    plt.close()


def plot_confusion_matrix(cm: List[List[int]], output_dir: str):
    cm = np.array(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor='white')
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Rejected', 'Chosen'], fontweight='bold')
    ax.set_yticklabels(['Rejected', 'Chosen'], fontweight='bold')
    ax.set_xlabel('Predicted', fontweight='bold', labelpad=10)
    ax.set_ylabel('Gold', fontweight='bold', labelpad=10)
    ax.set_title('Confusion Matrix (row-normalized)', fontweight='bold', pad=12)

    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.1%})',
                    ha='center', va='center', color=text_color, fontweight='bold', fontsize=12)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.pdf'), **PLOT_CONFIG)
    plt.close()


# =============================================================================
# Error analysis
# =============================================================================

def compute_length_features(eval_data: List[Dict], tokenizer=None) -> Dict[str, np.ndarray]:
    """Length of chosen/rejected responses, in tokens if `tokenizer` is given
    (more faithful to what the probe actually "sees"), else characters."""
    if tokenizer is not None:
        chosen_len = np.array([len(tokenizer.encode(item["chosen"][0]["content"])) for item in eval_data])
        rejected_len = np.array([len(tokenizer.encode(item["rejected"][0]["content"])) for item in eval_data])
    else:
        chosen_len = np.array([len(item["chosen"][0]["content"]) for item in eval_data])
        rejected_len = np.array([len(item["rejected"][0]["content"]) for item in eval_data])

    return {
        'chosen_len': chosen_len,
        'rejected_len': rejected_len,
        'len_diff': (chosen_len - rejected_len).astype(float),
        'length_ratio': chosen_len / np.maximum(rejected_len, 1),
        'log_length_ratio': np.log(np.maximum(chosen_len, 1) / np.maximum(rejected_len, 1)),
    }


def error_analysis(
    eval_data: List[Dict],
    prob_chosen: np.ndarray,
    prob_rejected: np.ndarray,
    output_dir: str,
    top_k: int = 20,
    length_feats: Dict[str, np.ndarray] = None,
) -> Dict:
    """Characterize pairwise errors (probe ranks gold-rejected above gold-chosen):
    length bias and the most confidently wrong examples, dumped for manual review."""
    diff = prob_chosen - prob_rejected
    error_idx = np.where(diff <= 0)[0]
    correct_idx = np.where(diff > 0)[0]

    if length_feats is None:
        length_feats = compute_length_features(eval_data)
    chosen_len = length_feats['chosen_len']
    rejected_len = length_feats['rejected_len']
    length_ratio = length_feats['length_ratio']

    stats = {
        'n_total': len(eval_data),
        'n_errors': int(len(error_idx)),
        'n_correct': int(len(correct_idx)),
        'pairwise_error_rate': float(len(error_idx) / len(eval_data)),
        'mean_length_ratio_errors': float(length_ratio[error_idx].mean()) if len(error_idx) else None,
        'mean_length_ratio_correct': float(length_ratio[correct_idx].mean()) if len(correct_idx) else None,
        'mean_chosen_len_errors': float(chosen_len[error_idx].mean()) if len(error_idx) else None,
        'mean_chosen_len_correct': float(chosen_len[correct_idx].mean()) if len(correct_idx) else None,
    }

    # Most confidently wrong first (most negative gap = rejected scored far above chosen)
    worst_order = error_idx[np.argsort(diff[error_idx])]
    worst_examples = []
    for idx in worst_order[:top_k]:
        item = eval_data[idx]
        worst_examples.append({
            'index': int(idx),
            'prob_chosen': float(prob_chosen[idx]),
            'prob_rejected': float(prob_rejected[idx]),
            'confidence_gap': float(diff[idx]),
            'prompt': item['prompt'],
            'chosen': item['chosen'][0]['content'][:500],
            'rejected': item['rejected'][0]['content'][:500],
        })

    with open(os.path.join(output_dir, 'worst_errors.json'), 'w') as f:
        json.dump(worst_examples, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    if len(correct_idx):
        ax.hist(length_ratio[correct_idx], bins=30, alpha=0.6, density=True,
                color=COLORS['correct'], label='Correct')
    if len(error_idx):
        ax.hist(length_ratio[error_idx], bins=30, alpha=0.6, density=True,
                color=COLORS['incorrect'], label='Error')
    ax.set_xlabel('Chosen / Rejected length ratio', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Length Ratio: Correct vs. Probe Errors', fontweight='bold')
    ax.legend(frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_length_bias.pdf'), **PLOT_CONFIG)
    plt.close()

    return stats


# =============================================================================
# Length-confound analysis
# =============================================================================

def analyze_length_confound(
    eval_data: List[Dict],
    prob_chosen: np.ndarray,
    prob_rejected: np.ndarray,
    length_feats: Dict[str, np.ndarray],
    output_dir: str,
) -> Dict:
    """Quantify how much of the probe's judgment is explained by response length
    rather than content.

    Three complementary angles:
      1. Correlation between the probe's confidence gap (and raw pointwise scores)
         and length — does the probe just score "longer" as "better"?
      2. A length-only baseline ("the longer response is chosen") vs. the probe's
         actual pairwise accuracy — an upper bound on how far length alone gets you.
      3. The key causal check: pairwise accuracy split by whether length *agrees*
         with the gold label (chosen is the longer one) or *disagrees* (chosen is
         shorter). If accuracy collapses toward chance on the "disagrees" slice,
         the probe is substituting length for content judgment; if it stays high,
         the probe is robust to length and tracks content.
    """
    diff = prob_chosen - prob_rejected
    correct = diff > 0
    len_diff = length_feats['len_diff']
    log_ratio = length_feats['log_length_ratio']

    # --- 1. Correlations ---
    r_diff_lendiff, p_diff_lendiff = pearsonr(diff, len_diff)
    rho_diff_lendiff, _ = spearmanr(diff, len_diff)
    r_diff_logratio, p_diff_logratio = pearsonr(diff, log_ratio)
    rho_diff_logratio, _ = spearmanr(diff, log_ratio)

    # Pointwise: pool chosen+rejected activations and see if raw P(preferred) tracks
    # raw length, independent of which one is gold-chosen.
    all_probs = np.concatenate([prob_chosen, prob_rejected])
    all_lens = np.concatenate([length_feats['chosen_len'], length_feats['rejected_len']]).astype(float)
    r_prob_len, p_prob_len = pearsonr(all_probs, all_lens)
    rho_prob_len, _ = spearmanr(all_probs, all_lens)

    # --- 2. Length-only baseline ---
    is_longer_chosen = len_diff > 0
    is_shorter_chosen = len_diff < 0
    is_tie = len_diff == 0
    length_only_accuracy = float(is_longer_chosen.mean() + 0.5 * is_tie.mean())

    # How often does the probe's *own* preferred response happen to be the longer
    # one, regardless of whether that's actually correct? Near 1.0 => the probe is
    # essentially a length detector; near 0.5 => length isn't driving its choices.
    non_tie = ~is_tie
    probe_prefers_longer = (diff[non_tie] > 0) == is_longer_chosen[non_tie]
    probe_length_alignment = float(probe_prefers_longer.mean()) if non_tie.any() else float('nan')

    # --- 3. Stratified pairwise accuracy: does the probe still work when length
    # points the wrong way? ---
    n_agree = int(is_longer_chosen.sum())
    n_disagree = int(is_shorter_chosen.sum())
    n_tie = int(is_tie.sum())

    acc_length_agrees = float(correct[is_longer_chosen].mean()) if n_agree else None
    acc_length_disagrees = float(correct[is_shorter_chosen].mean()) if n_disagree else None
    acc_length_tie = float(correct[is_tie].mean()) if n_tie else None

    stats = {
        'correlation': {
            'pearson_confidence_gap_vs_len_diff': float(r_diff_lendiff),
            'pearson_confidence_gap_vs_len_diff_pvalue': float(p_diff_lendiff),
            'spearman_confidence_gap_vs_len_diff': float(rho_diff_lendiff),
            'pearson_confidence_gap_vs_log_length_ratio': float(r_diff_logratio),
            'pearson_confidence_gap_vs_log_length_ratio_pvalue': float(p_diff_logratio),
            'spearman_confidence_gap_vs_log_length_ratio': float(rho_diff_logratio),
            'pearson_pointwise_prob_vs_len': float(r_prob_len),
            'pearson_pointwise_prob_vs_len_pvalue': float(p_prob_len),
            'spearman_pointwise_prob_vs_len': float(rho_prob_len),
        },
        'length_only_baseline_accuracy': length_only_accuracy,
        'probe_length_alignment_rate': probe_length_alignment,
        'stratified_by_length': {
            'n_length_agrees_with_gold': n_agree,
            'n_length_disagrees_with_gold': n_disagree,
            'n_tie': n_tie,
            'pairwise_accuracy_length_agrees': acc_length_agrees,
            'pairwise_accuracy_length_disagrees': acc_length_disagrees,
            'pairwise_accuracy_length_tie': acc_length_tie,
            'accuracy_gap_agrees_minus_disagrees': (
                acc_length_agrees - acc_length_disagrees
                if acc_length_agrees is not None and acc_length_disagrees is not None
                else None
            ),
        },
    }

    plot_length_confound(diff, len_diff, correct, stats, output_dir)
    return stats


def plot_length_confound(
    diff: np.ndarray,
    len_diff: np.ndarray,
    correct: np.ndarray,
    stats: Dict,
    output_dir: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    ax = axes[0]
    ax.scatter(len_diff[correct], diff[correct], s=12, alpha=0.5,
               color=COLORS['correct'], label='Correct pairwise')
    ax.scatter(len_diff[~correct], diff[~correct], s=12, alpha=0.5,
               color=COLORS['incorrect'], label='Incorrect pairwise')
    if len(len_diff) > 1:
        slope, intercept = np.polyfit(len_diff, diff, 1)
        xs = np.linspace(len_diff.min(), len_diff.max(), 100)
        ax.plot(xs, slope * xs + intercept, color='black', linewidth=1.5, linestyle='--',
                label=f"r={stats['correlation']['pearson_confidence_gap_vs_len_diff']:.2f}")
    ax.axhline(y=0, color='black', linewidth=1, alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=1, alpha=0.3)
    ax.set_xlabel('Length(chosen) - Length(rejected)', fontweight='bold')
    ax.set_ylabel('P(chosen) - P(rejected)', fontweight='bold')
    ax.set_title('Probe Confidence Gap vs. Length Gap', fontweight='bold')
    ax.legend(frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    strat = stats['stratified_by_length']
    labels, accs, ns = [], [], []
    if strat['pairwise_accuracy_length_agrees'] is not None:
        labels.append('Length agrees\nwith gold')
        accs.append(strat['pairwise_accuracy_length_agrees'])
        ns.append(strat['n_length_agrees_with_gold'])
    if strat['pairwise_accuracy_length_disagrees'] is not None:
        labels.append('Length disagrees\nwith gold')
        accs.append(strat['pairwise_accuracy_length_disagrees'])
        ns.append(strat['n_length_disagrees_with_gold'])
    if strat['pairwise_accuracy_length_tie'] is not None:
        labels.append('Length tie')
        accs.append(strat['pairwise_accuracy_length_tie'])
        ns.append(strat['n_tie'])

    ax2 = axes[1]
    bar_colors = [COLORS['chosen'], COLORS['incorrect'], COLORS['neutral']][:len(labels)]
    bars = ax2.bar(labels, accs, color=bar_colors, alpha=0.85, edgecolor='white')
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Chance')
    ax2.axhline(y=stats['length_only_baseline_accuracy'], color='#f39c12', linestyle=':',
                linewidth=1.5, label='Length-only baseline')
    for bar, n in zip(bars, ns):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                  f'n={n}', ha='center', fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Pairwise accuracy', fontweight='bold')
    ax2.set_title('Pairwise Accuracy Stratified by Length Agreement', fontweight='bold')
    ax2.legend(frameon=True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_confound.pdf'), **PLOT_CONFIG)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained probe against gold labels")

    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--dataset-language", type=str, default=None,
                         help="Dataset-specific arg (e.g. language code for oasst2/prism)")
    parser.add_argument("--train-samples", type=int, default=1000,
                         help="Number of pairs used to train the probe")
    parser.add_argument("--eval-samples", type=int, default=500,
                         help="Number of held-out pairs used to evaluate against gold labels")
    parser.add_argument("--probe-layers", type=int, nargs='+', default=[8])
    parser.add_argument("--probe-type", type=str, choices=["logistic", "mlp"], default="logistic")
    parser.add_argument("--no-filter-length-outliers", action="store_true",
                         help="Disable filtering of extreme chosen/rejected length ratios during probe training")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Decision threshold for pointwise precision/recall/F1")
    parser.add_argument("--pairwise-confidence-threshold", type=float, default=0.0,
                         help="Confidence threshold |P(chosen)-P(rejected)| for the headline "
                              "pairwise precision/recall (mirrors config.probe_confidence_threshold; "
                              "0.0 = every pair is retained, recall reduces to pairwise accuracy)")
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--top-k-errors", type=int, default=20)
    parser.add_argument("--length-unit", type=str, choices=["chars", "tokens"], default="chars",
                         help="Unit used for length-confound analysis. 'tokens' uses the "
                              "model's own tokenizer and is more faithful to what the probe sees.")
    parser.add_argument("--output-dir", type=str, default="./probe_gold_eval")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PROBE VS. GOLD LABEL EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Train samples: {args.train_samples} | Eval samples: {args.eval_samples}")
    print(f"Probe: {args.probe_type} on layers {args.probe_layers}")
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
    # Left-padding is required here: ActivationExtractor reads hidden[:, -1, :], which
    # must be the true last token, not a pad token, for every sequence in the batch.
    tokenizer.padding_side = "left"

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    config = APOConfig(
        model_name=args.model_name,
        probe_layers=args.probe_layers,
        probe_subset_size=len(train_data),
        probe_type=args.probe_type,
        probe_filter_length_outliers=not args.no_filter_length_outliers,
        output_dir=args.output_dir,
        seed=args.seed,
        use_wandb=False,
    )

    probe, probe_layers = train_preference_probe(config, model, tokenizer, train_data)

    prob_chosen, prob_rejected = score_eval_set(
        model, tokenizer, probe, probe_layers, eval_data, batch_size=args.batch_size
    )

    # --- Metrics ---
    pointwise_metrics, y_true, y_prob = compute_pointwise_metrics(prob_chosen, prob_rejected, args.threshold)
    pairwise_metrics, diff = compute_pairwise_metrics(
        prob_chosen, prob_rejected, args.pairwise_confidence_threshold
    )
    pairwise_pr_curve = compute_pairwise_pr_curve(prob_chosen, prob_rejected)
    calibration_metrics, bin_stats = compute_calibration(y_true, y_prob, args.ece_bins)

    length_tokenizer = tokenizer if args.length_unit == "tokens" else None
    length_feats = compute_length_features(eval_data, length_tokenizer)
    error_stats = error_analysis(
        eval_data, prob_chosen, prob_rejected, args.output_dir, args.top_k_errors, length_feats
    )
    length_confound_stats = analyze_length_confound(
        eval_data, prob_chosen, prob_rejected, length_feats, args.output_dir
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("-- Pointwise (chosen/rejected activations scored as independent instances) --")
    print(f"Accuracy:           {pointwise_metrics['accuracy']:.4f}")
    print(f"Precision:          {pointwise_metrics['precision']:.4f}")
    print(f"Recall:             {pointwise_metrics['recall']:.4f}")
    print(f"F1:                 {pointwise_metrics['f1']:.4f}")
    print(f"ROC-AUC:            {pointwise_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:             {pointwise_metrics['pr_auc']:.4f}")
    print(f"Brier score:        {pointwise_metrics['brier_score']:.4f}")
    print(f"ECE:                {calibration_metrics['ece']:.4f}")
    print(f"MCE:                {calibration_metrics['mce']:.4f}")
    print(f"\n-- Pairwise (does the probe rank gold-chosen above gold-rejected?) "
          f"[confidence threshold={args.pairwise_confidence_threshold}] --")
    print(f"Accuracy (unconditional): {pairwise_metrics['pairwise_accuracy']:.4f}  "
          f"(ties: {pairwise_metrics['tie_rate']:.4f})")
    print(f"Coverage:           {pairwise_metrics['coverage']:.4f}")
    print(f"Precision:          {pairwise_metrics['pairwise_precision']:.4f}  "
          f"(accuracy among retained/confident pairs)")
    print(f"Recall:             {pairwise_metrics['pairwise_recall']:.4f}  "
          f"(correct & retained / all pairs)")
    print(f"F1:                 {pairwise_metrics['pairwise_f1']:.4f}")
    print(f"Pairwise error rate: {error_stats['pairwise_error_rate']:.4f} "
          f"({error_stats['n_errors']}/{error_stats['n_total']})")
    if error_stats['mean_length_ratio_errors'] is not None:
        print(f"Mean length ratio (chosen/rejected) — errors: {error_stats['mean_length_ratio_errors']:.2f}, "
              f"correct: {error_stats['mean_length_ratio_correct']:.2f}")

    strat = length_confound_stats['stratified_by_length']
    print(f"\n-- Length confound (unit: {args.length_unit}) --")
    print(f"Pearson r(confidence gap, len diff):     {length_confound_stats['correlation']['pearson_confidence_gap_vs_len_diff']:.4f} "
          f"(p={length_confound_stats['correlation']['pearson_confidence_gap_vs_len_diff_pvalue']:.2e})")
    print(f"Spearman rho(confidence gap, len diff):  {length_confound_stats['correlation']['spearman_confidence_gap_vs_len_diff']:.4f}")
    print(f"Pearson r(pointwise P, length):           {length_confound_stats['correlation']['pearson_pointwise_prob_vs_len']:.4f} "
          f"(p={length_confound_stats['correlation']['pearson_pointwise_prob_vs_len_pvalue']:.2e})")
    print(f"Length-only baseline accuracy (longer=chosen): {length_confound_stats['length_only_baseline_accuracy']:.4f}")
    print(f"Probe-length alignment rate (probe prefers the longer response): "
          f"{length_confound_stats['probe_length_alignment_rate']:.4f}")
    print(f"Pairwise accuracy when length AGREES with gold (n={strat['n_length_agrees_with_gold']}):    "
          f"{strat['pairwise_accuracy_length_agrees']:.4f}")
    print(f"Pairwise accuracy when length DISAGREES with gold (n={strat['n_length_disagrees_with_gold']}): "
          f"{strat['pairwise_accuracy_length_disagrees']:.4f}")
    if strat['accuracy_gap_agrees_minus_disagrees'] is not None:
        print(f"Accuracy gap (agrees - disagrees): {strat['accuracy_gap_agrees_minus_disagrees']:.4f}  "
              f"(large gap => probe leans on length; near 0 => robust to length)")

    # --- Plots ---
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
            'probe_type': args.probe_type,
            'probe_layers': probe_layers,
            'threshold': args.threshold,
            'pairwise_confidence_threshold': args.pairwise_confidence_threshold,
            'length_unit': args.length_unit,
        },
        'pointwise_metrics': pointwise_metrics,
        'pairwise_metrics': pairwise_metrics,
        'pairwise_pr_curve': pairwise_pr_curve,
        'calibration': {**calibration_metrics, 'bins': bin_stats},
        'error_analysis': error_stats,
        'length_confound': length_confound_stats,
    }
    with open(os.path.join(args.output_dir, 'gold_comparison_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Saved metrics to: {args.output_dir}/gold_comparison_metrics.json")
    print(f"Saved worst errors to: {args.output_dir}/worst_errors.json")
    print("Plots: roc_pr_curves.pdf, pairwise_precision_recall_curve.pdf, calibration.pdf, "
          "confidence_distribution.pdf, confusion_matrix.pdf, error_length_bias.pdf, "
          "length_confound.pdf")
    print("=" * 70)


if __name__ == "__main__":
    main()
