"""
Preference-geometry analysis for APO: characterises how chosen/rejected
activations are organised in a model's representation space, independent of
the main probe-labelling pipeline in probe_training.py.

This is the script behind Section 3 of the paper ("Preference Geometry in
Activations"): for one or more training stages (pretrained, and optionally
an SFT and/or PO checkpoint of the same model), it extracts per-layer
chosen/rejected activations and computes, per layer:
  - Separation directions (mean-difference and/or logistic-regression) and
    classification accuracy (in-distribution or cross-dataset)
  - Effect size (Cohen's d) and significance (PC1 t-test / Mann-Whitney U)
  - A kernel two-sample test (MMD, RBF kernel, permutation p-value)

It then produces cross-stage comparisons (layer-wise accuracy/Cohen's-d
evolution, PCA cluster visualisations, direction-transfer heatmaps,
feature-importance overlap, a summary dashboard) as PDFs, plus the
underlying metrics as JSON in --output-dir. `--shuffle-labels` reruns the
whole pipeline with chosen/rejected randomly swapped as a sanity check
(paper Appendix A.1): a genuine signal should collapse to chance under
shuffling.
"""

import argparse
import json
import os
from copy import deepcopy
import random
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import ttest_ind, mannwhitneyu, spearmanr, shapiro, levene, t as t_dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

from dataset_utils import load_preference_dataset

# Set better default style with improved aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)
sns.set_palette("Set2")

# Custom color palette
COLORS = {
    'pretrained': '#3498db',  # Blue
    'sft': '#2ecc71',         # Green
    'po': '#e74c3c',          # Red
    'chosen': '#3498db',      # Blue
    'rejected': '#e74c3c',    # Red
    'direction': '#f39c12',   # Orange
    'neutral': '#95a5a6'      # Gray
}

# Plot configuration
PLOT_CONFIG = {
    'dpi': 150,
    'bbox_inches': 'tight',
    'facecolor': 'white',
    'edgecolor': 'none'
}


def get_model_layers(model):
    """Get the transformer layers from a model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return model.model.decoder.layers
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        return model.language_model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'transformer') and hasattr(model.model.transformer, 'blocks'):
        return model.model.transformer.blocks
    else:
        raise ValueError("Could not find transformer layers in model architecture")


def get_completion_token_indices(tokenizer, full_messages: List[Dict], prompt_messages: List[Dict]) -> Tuple[int, int]:
    """Identify which tokens belong to the completion (assistant response)."""
    full_tokens = tokenizer.apply_chat_template(
        full_messages, 
        return_tensors="pt", 
        add_generation_prompt=False
    )
    prompt_tokens = tokenizer.apply_chat_template(
        prompt_messages, 
        return_tensors="pt", 
        add_generation_prompt=True
    )

    return prompt_tokens.shape[1], full_tokens.shape[1]


def extract_activations_single_layer(
    model,
    tokenizer,
    dataset: List[Dict],
    layer_idx: int,
    max_samples: int = 500,
    position: str = "completion_mean"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract activations from a single layer (memory efficient)."""
    layers = get_model_layers(model)
    
    chosen_activations = []
    rejected_activations = []
    
    activations_cache = {}
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations_cache['output'] = output[0].detach()
        else:
            activations_cache['output'] = output.detach()
    
    hook = layers[layer_idx].register_forward_hook(hook_fn)
    
    model.eval()
    samples = dataset[:max_samples]

    for item in samples:
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        prompt_only = deepcopy(prompt)

        # Process chosen
        chosen_full = deepcopy(prompt)
        chosen_full.extend(chosen)
        completion_start, completion_end = get_completion_token_indices(tokenizer, chosen_full, prompt_only)
        
        chosen_enc = tokenizer.apply_chat_template(chosen_full, return_tensors="pt").to(model.device)
        
        activations_cache.clear()
        with torch.no_grad():
            model(input_ids=chosen_enc)
        
        hidden_states = activations_cache['output'].squeeze(0)
        
        if position == "completion_mean":
            activation = hidden_states[completion_start:completion_end].mean(dim=0)
        elif position == "last_token":
            activation = hidden_states[-1]
        elif position == "max":
            activation = hidden_states[completion_start:completion_end].max(dim=0).values
        elif position == "last_prompt":
            activation = hidden_states[completion_start - 1]
        else:  # mean
            activation = hidden_states.mean(dim=0)
        
        chosen_activations.append(activation.float().cpu().numpy())

        # Process rejected
        rejected_full = deepcopy(prompt)
        rejected_full.extend(rejected)
        completion_start, completion_end = get_completion_token_indices(tokenizer, rejected_full, prompt_only)
        
        rejected_enc = tokenizer.apply_chat_template(rejected_full, return_tensors="pt").to(model.device)
        
        activations_cache.clear()
        with torch.no_grad():
            model(input_ids=rejected_enc)
        
        hidden_states = activations_cache['output'].squeeze(0)
        
        if position == "completion_mean":
            activation = hidden_states[completion_start:completion_end].mean(dim=0)
        elif position == "last_token":
            activation = hidden_states[-1]
        elif position == "max":
            activation = hidden_states[completion_start:completion_end].max(dim=0).values
        elif position == "last_prompt":
            activation = hidden_states[completion_start - 1]
        else:
            activation = hidden_states.mean(dim=0)
        
        rejected_activations.append(activation.float().cpu().numpy())

    hook.remove()
    
    return np.vstack(chosen_activations), np.vstack(rejected_activations)


def extract_activations_all_layers(
    model,
    tokenizer,
    dataset: List[Dict],
    max_samples: int = 500,
    position: str = "completion_mean"
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Extract activations from all layers in a single forward pass per sample."""
    print(f"\nExtracting activations from all layers (samples: {max_samples}, position: {position})...")

    layers = get_model_layers(model)
    num_layers = len(layers)

    # Register hooks on all layers simultaneously so each sample needs only 2 forward passes
    # (one for chosen, one for rejected) instead of 2 * num_layers passes.
    activations_cache: Dict[int, torch.Tensor] = {}
    hooks = []
    for layer_idx in range(num_layers):
        def make_hook(idx):
            def hook_fn(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                activations_cache[idx] = act.detach()
            return hook_fn
        hooks.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    model.eval()
    samples = dataset[:max_samples]
    chosen_by_layer: Dict[int, list] = {i: [] for i in range(num_layers)}
    rejected_by_layer: Dict[int, list] = {i: [] for i in range(num_layers)}

    def extract_from_cache(completion_start: int, completion_end: int) -> Dict[int, np.ndarray]:
        result = {}
        for idx in range(num_layers):
            hidden = activations_cache[idx].squeeze(0)
            if position == "completion_mean":
                act = hidden[completion_start:completion_end].mean(dim=0)
            elif position == "last_token":
                act = hidden[-1]
            elif position == "max":
                act = hidden[completion_start:completion_end].max(dim=0).values
            elif position == "last_prompt":
                act = hidden[completion_start - 1]
            else:
                act = hidden.mean(dim=0)
            result[idx] = act.float().cpu().numpy()
        return result

    for item in tqdm(samples, desc="Extracting activations"):
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Ensure prompt ends with a user turn before adding generation prompt
        if isinstance(prompt, list):
            # Filter out any empty messages and ensure proper alternation
            cleaned_prompt = []
            last_role = None
            for msg in prompt:
                if msg["role"] != last_role:
                    cleaned_prompt.append(msg)
                    last_role = msg["role"]
            
            # apply_chat_template with add_generation_prompt=True expects last role to be "user"
            if cleaned_prompt and cleaned_prompt[-1]["role"] != "user":
                cleaned_prompt = cleaned_prompt[:-1]  # drop trailing assistant turn
            
            prompt = cleaned_prompt

        # Prompt token length is shared between chosen and rejected — compute once.
        if prompt:
            prompt_len = tokenizer.apply_chat_template(
                prompt, return_tensors="pt", add_generation_prompt=True
            ).shape[1]
        else:
            prompt_len = 0

        # Process chosen
        chosen_enc = tokenizer.apply_chat_template(
            prompt + chosen, return_tensors="pt"
        ).to(model.device)
        activations_cache.clear()
        with torch.no_grad():
            model(input_ids=chosen_enc)
        for idx, act in extract_from_cache(prompt_len, chosen_enc.shape[1]).items():
            chosen_by_layer[idx].append(act)

        # Process rejected
        rejected_enc = tokenizer.apply_chat_template(
            prompt + rejected, return_tensors="pt"
        ).to(model.device)
        activations_cache.clear()
        with torch.no_grad():
            model(input_ids=rejected_enc)
        for idx, act in extract_from_cache(prompt_len, rejected_enc.shape[1]).items():
            rejected_by_layer[idx].append(act)

    for hook in hooks:
        hook.remove()

    chosen_out = {i: np.vstack(chosen_by_layer[i]) for i in range(num_layers)}
    rejected_out = {i: np.vstack(rejected_by_layer[i]) for i in range(num_layers)}
    print(f"Extracted activations shape: {chosen_out[0].shape}")
    return chosen_out, rejected_out


def compute_mean_difference_direction(pos_acts: np.ndarray, neg_acts: np.ndarray) -> np.ndarray:
    """Compute direction as mean(positive) - mean(negative)."""
    direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
    return direction / (np.linalg.norm(direction) + 1e-8)


def compute_logistic_regression_direction(pos_acts: np.ndarray, neg_acts: np.ndarray) -> np.ndarray:
    """Compute direction using logistic regression weights."""
    X = np.vstack([pos_acts, neg_acts])
    y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

    # Project to low-dim PCA space, fit LR, then map weights back to original space.
    pca_n = min(50, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=pca_n, random_state=42)
    X_red = pca.fit_transform(X)

    lr = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X_red, y)

    # Map the low-dim weight vector back to the original activation space.
    direction = pca.components_.T @ lr.coef_[0]
    return direction / (np.linalg.norm(direction) + 1e-8)


def compute_all_directions(pos_acts: np.ndarray, neg_acts: np.ndarray,
                           methods: List[str] = None) -> Dict[str, Optional[np.ndarray]]:
    """Compute directions using the requested methods.

    Methods that do not have a linear direction (knn, kernel_svm) are stored as None.
    They still appear in the dict so that analyze_layer_wise_accuracy can iterate over them.
    """
    result = {}
    if methods is None or 'mean_diff' in methods:
        result['mean_diff'] = compute_mean_difference_direction(pos_acts, neg_acts)
    if methods is None or 'logistic_regression' in methods:
        result['logistic_regression'] = compute_logistic_regression_direction(pos_acts, neg_acts)
    # Non-directional methods — stored as None so they participate in the accuracy loop
    # but skip projection-based metrics (Cohen's d, pos/neg projections).
    if methods is not None and 'knn' in methods:
        result['knn'] = None
    if methods is not None and 'kernel_svm' in methods:
        result['kernel_svm'] = None

    return result


def compute_separation_metrics(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    direction: Optional[np.ndarray],
    use_cross_val: bool = True,
    clf_factory=None,
    pos_acts_train: np.ndarray = None,
    neg_acts_train: np.ndarray = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Compute various metrics quantifying pos-neg separation.

    When pos_acts_train / neg_acts_train are provided (cross-dataset mode), the
    classifier is trained on those activations and evaluated on pos_acts /
    neg_acts.  cross_val is skipped in that case.

    direction may be None for non-directional methods (knn, kernel_svm).  In
    that case Cohen's d and projection statistics are set to NaN.
    """

    metrics = {}

    # 1. Cohen's d (direction-based methods only)
    if direction is not None:
        pos_proj = pos_acts @ direction
        neg_proj = neg_acts @ direction
        mean_diff = pos_proj.mean() - neg_proj.mean()
        pooled_std = np.sqrt((pos_proj.std()**2 + neg_proj.std()**2) / 2)
        metrics['cohens_d'] = mean_diff / (pooled_std + 1e-8)
    else:
        pos_proj = np.array([])
        neg_proj = np.array([])
        metrics['cohens_d'] = float('nan')

    # 2. Classification accuracy with proper evaluation
    X_eval = np.vstack([pos_acts, neg_acts])
    y_eval = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

    if clf_factory is None:
        clf_factory = lambda: LogisticRegression(max_iter=200, random_state=42)

    # Reduce dimensionality before fitting classifiers — the single biggest speedup.
    # Logistic regression cost scales with D²; going from ~4096 → 50 dims is ~6700× cheaper per fit.
    pca_n = min(50, X_eval.shape[1], X_eval.shape[0] - 1)
    pca_pre = PCA(n_components=pca_n, random_state=42)

    cross_dataset_mode = pos_acts_train is not None and neg_acts_train is not None

    if cross_dataset_mode:
        X_train_full = np.vstack([pos_acts_train, neg_acts_train])
        y_train = np.concatenate([np.ones(len(pos_acts_train)), np.zeros(len(neg_acts_train))])
        X_train_red = pca_pre.fit_transform(X_train_full)
        X_eval_red = pca_pre.transform(X_eval)
        clf = clf_factory()
        clf.fit(X_train_red, y_train)
        metrics['classification_accuracy'] = accuracy_score(y_eval, clf.predict(X_eval_red))
        metrics['classification_accuracy_std'] = 0.0
        metrics['classification_accuracy_ci_lower'] = None
        metrics['classification_accuracy_ci_upper'] = None
    elif use_cross_val and len(X_eval) >= 10:
        X_eval_red = pca_pre.fit_transform(X_eval)
        cv_scores = cross_val_score(clf_factory(), X_eval_red, y_eval, cv=5, scoring='accuracy')
        n_folds = len(cv_scores)
        se = cv_scores.std(ddof=1) / np.sqrt(n_folds)
        ci_low, ci_high = t_dist.interval(0.95, df=n_folds - 1, loc=cv_scores.mean(), scale=se)
        metrics['classification_accuracy'] = cv_scores.mean()
        metrics['classification_accuracy_std'] = cv_scores.std()
        metrics['classification_accuracy_ci_lower'] = float(ci_low)
        metrics['classification_accuracy_ci_upper'] = float(ci_high)
    else:
        X_eval_red = pca_pre.fit_transform(X_eval)
        X_tr, X_te, y_tr, y_te = train_test_split(X_eval_red, y_eval, test_size=0.2, random_state=42)
        clf = clf_factory()
        clf.fit(X_tr, y_tr)
        metrics['classification_accuracy'] = accuracy_score(y_te, clf.predict(X_te))
        metrics['classification_accuracy_std'] = 0.0
        metrics['classification_accuracy_ci_lower'] = None
        metrics['classification_accuracy_ci_upper'] = None

    # 3. Euclidean distance between means
    metrics['euclidean_distance'] = np.linalg.norm(pos_acts.mean(axis=0) - neg_acts.mean(axis=0))

    # 4. PC1-based significance test — reuse the already-fitted PCA if it has ≥1 component,
    # otherwise fit a dedicated 1-component PCA.
    if pca_n >= 1 and not cross_dataset_mode:
        # pc1 is just the first column of the already-projected data
        pc1_all = pca_pre.transform(X_eval)[:, 0]
        metrics['pc1_explained_variance'] = float(pca_pre.explained_variance_ratio_[0])
    else:
        pca1 = PCA(n_components=1)
        pc1_all = pca1.fit_transform(X_eval).squeeze()
        metrics['pc1_explained_variance'] = float(pca1.explained_variance_ratio_[0])
    pc1_pos = pc1_all[:len(pos_acts)]
    pc1_neg = pc1_all[len(pos_acts):]

    metrics['pc1_mean_pos'] = float(pc1_pos.mean())
    metrics['pc1_mean_neg'] = float(pc1_neg.mean())
    # pc1_explained_variance already set above

    # check normality of PC1 projections and variance homogeneity before t-test
    metrics['pc1_normality_pos_pval'] = float(shapiro(pc1_pos).pvalue)
    metrics['pc1_normality_neg_pval'] = float(shapiro(pc1_neg).pvalue)
    metrics['pc1_variance_homogeneity_pval'] = float(levene(pc1_pos, pc1_neg).pvalue)

    t_stat, t_pval = ttest_ind(pc1_pos, pc1_neg)
    metrics['pc1_ttest_stat'] = float(t_stat)
    metrics['pc1_ttest_pval'] = float(t_pval)

    u_stat, u_pval = mannwhitneyu(pc1_pos, pc1_neg, alternative='two-sided')
    metrics['pc1_mannwhitney_stat'] = float(u_stat)
    metrics['pc1_mannwhitney_pval'] = float(u_pval)

    metrics['pc1_significant'] = u_pval < 0.05

    return metrics, pos_proj, neg_proj


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix via the kernel trick (avoids an
    O(n^2 * d) broadcast subtraction)."""
    sq_norms = np.sum(X * X, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
    return np.maximum(sq_dists, 0.0)


def compute_kernel_mmd(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    n_permutations: int = 200,
    max_samples: int = 250,
    random_state: int = 42,
) -> Dict[str, float]:
    """Kernel two-sample test (unbiased MMD^2, RBF kernel) on the raw,
    full-dimensional activations, with a permutation-based p-value.

    Unlike Cohen's d / the t-test / Mann-Whitney (which operate on a learned
    direction or a PCA projection), this tests directly whether the two
    full activation distributions differ, using the unbiased MMD^2 estimator
    of Gretton et al. (2012, JMLR) with a median-heuristic bandwidth.
    """
    rng = np.random.RandomState(random_state)

    if len(pos_acts) > max_samples:
        pos_acts = pos_acts[rng.choice(len(pos_acts), max_samples, replace=False)]
    if len(neg_acts) > max_samples:
        neg_acts = neg_acts[rng.choice(len(neg_acts), max_samples, replace=False)]

    n1, n2 = len(pos_acts), len(neg_acts)
    X = np.vstack([pos_acts, neg_acts]).astype(np.float64)

    sq_dists = _pairwise_sq_dists(X)
    off_diag = sq_dists[np.triu_indices(len(X), k=1)]
    median_sq_dist = np.median(off_diag)
    # Median heuristic bandwidth; fall back to 1.0 for degenerate (all-identical) inputs.
    sigma2 = median_sq_dist / 2.0 if median_sq_dist > 0 else 1.0
    K = np.exp(-sq_dists / (2.0 * sigma2))

    def mmd2_stat(n_a: int, n_b: int, kernel: np.ndarray) -> float:
        Kxx = kernel[:n_a, :n_a]
        Kyy = kernel[n_a:, n_a:]
        Kxy = kernel[:n_a, n_a:]
        term_xx = (Kxx.sum() - np.trace(Kxx)) / (n_a * (n_a - 1))
        term_yy = (Kyy.sum() - np.trace(Kyy)) / (n_b * (n_b - 1))
        term_xy = Kxy.sum() / (n_a * n_b)
        return term_xx + term_yy - 2.0 * term_xy

    mmd2_observed = mmd2_stat(n1, n2, K)

    # Permutation test: reshuffle group labels, reusing the precomputed kernel
    # matrix (only the row/column ordering changes between permutations).
    n_total = n1 + n2
    idx = np.arange(n_total)
    perm_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(idx)
        perm_stats[i] = mmd2_stat(n1, n2, K[np.ix_(idx, idx)])

    p_value = (np.sum(perm_stats >= mmd2_observed) + 1) / (n_permutations + 1)

    return {
        'mmd2': float(mmd2_observed),
        'mmd_pvalue': float(p_value),
        'mmd_n_permutations': n_permutations,
        'mmd_n_samples': [n1, n2],
        'mmd_bandwidth_sigma2': float(sigma2),
        'mmd_significant': bool(p_value < 0.05),
    }


def concatenate_layer_activations(
    acts_by_layer: Dict[int, np.ndarray],
    layers: List[int],
) -> np.ndarray:
    """Concatenate activations from multiple layers along the feature axis."""
    return np.concatenate([acts_by_layer[l] for l in layers], axis=1)


def analyze_multilayer_accuracy(
    chosen_by_layer: Dict[int, np.ndarray],
    rejected_by_layer: Dict[int, np.ndarray],
    layers: List[int],
    chosen_eval_by_layer: Dict[int, np.ndarray] = None,
    rejected_eval_by_layer: Dict[int, np.ndarray] = None,
    methods: List[str] = ['mean_diff', 'logistic_regression'],
    cross_dataset: bool = False,
    compute_mmd: bool = True,
    mmd_permutations: int = 200,
    mmd_max_samples: int = 250,
) -> Dict[str, Dict]:
    """Train probes on concatenated activations from multiple layers and measure accuracy."""
    print(f"\nAnalyzing multi-layer probe (layers: {layers})...")

    if chosen_eval_by_layer is None:
        chosen_eval_by_layer = chosen_by_layer
    if rejected_eval_by_layer is None:
        rejected_eval_by_layer = rejected_by_layer

    pos_train = concatenate_layer_activations(chosen_by_layer, layers)
    neg_train = concatenate_layer_activations(rejected_by_layer, layers)
    pos_eval = concatenate_layer_activations(chosen_eval_by_layer, layers)
    neg_eval = concatenate_layer_activations(rejected_eval_by_layer, layers)

    directions = compute_all_directions(pos_train, neg_train, methods)

    results = {}
    for method in methods:
        if method not in directions:
            continue

        clf_factory = None
        if method == 'mlp':
            clf_factory = lambda: MLPClassifier(
                hidden_layer_sizes=(256, 128), max_iter=200, random_state=42
            )
        elif method == 'knn':
            _k = getattr(analyze_layer_wise_accuracy, '_knn_neighbors', 5)
            clf_factory = lambda k=_k: KNeighborsClassifier(n_neighbors=k)
        elif method == 'kernel_svm':
            clf_factory = lambda: SVC(kernel='rbf', random_state=42)

        metrics, _, _ = compute_separation_metrics(
            pos_eval, neg_eval, directions[method],
            use_cross_val=not cross_dataset,
            clf_factory=clf_factory,
            pos_acts_train=pos_train if cross_dataset else None,
            neg_acts_train=neg_train if cross_dataset else None,
        )
        results[method] = {'metrics': metrics, 'direction': directions[method]}

    if compute_mmd:
        results['mmd_test'] = compute_kernel_mmd(
            pos_eval, neg_eval,
            n_permutations=mmd_permutations,
            max_samples=mmd_max_samples,
        )

    return results


def analyze_layer_wise_accuracy(
    chosen_by_layer: Dict[int, np.ndarray],
    rejected_by_layer: Dict[int, np.ndarray],
    chosen_eval_by_layer: Dict[int, np.ndarray] = None,
    rejected_eval_by_layer: Dict[int, np.ndarray] = None,
    methods: List[str] = ['mean_diff', 'logistic_regression'],
    cross_dataset: bool = False,
    compute_mmd: bool = True,
    mmd_permutations: int = 200,
    mmd_max_samples: int = 250,
) -> Dict[int, Dict[str, Dict]]:
    """Train probes on each layer and measure accuracy with proper evaluation.

    Args:
        chosen_by_layer: Training activations for chosen responses
        rejected_by_layer: Training activations for rejected responses
        chosen_eval_by_layer: Evaluation activations for chosen responses (if None, use training data)
        rejected_eval_by_layer: Evaluation activations for rejected responses (if None, use training data)
        methods: List of methods to use for computing directions
        cross_dataset: If True, classifier is trained on training activations and
                       evaluated on eval activations (no cross-val on eval data).
        compute_mmd: If True, also run a kernel-MMD two-sample test (with a
                     permutation p-value) on the full-dimensional activations
                     of each layer, stored under the 'mmd_test' key.
    """
    print("\nAnalyzing layer-wise probe accuracy...")

    # Use training data for eval if not provided
    if chosen_eval_by_layer is None:
        chosen_eval_by_layer = chosen_by_layer
    if rejected_eval_by_layer is None:
        rejected_eval_by_layer = rejected_by_layer

    results_by_layer = {}

    for layer_idx in tqdm(sorted(chosen_by_layer.keys()), desc="Analyzing layers"):
        # Training data (for computing directions and fitting classifiers)
        pos_acts_train = chosen_by_layer[layer_idx]
        neg_acts_train = rejected_by_layer[layer_idx]

        # Evaluation data (for measuring metrics)
        pos_acts_eval = chosen_eval_by_layer[layer_idx]
        neg_acts_eval = rejected_eval_by_layer[layer_idx]

        # Compute directions on training data
        directions = compute_all_directions(pos_acts_train, neg_acts_train, methods)

        layer_results = {}
        for method in methods:
            if method not in directions:
                continue

            clf_factory = None
            if method == 'mlp':
                clf_factory = lambda: MLPClassifier(
                    hidden_layer_sizes=(256, 128), max_iter=200, random_state=42
                )
            elif method == 'knn':
                _k = getattr(analyze_layer_wise_accuracy, '_knn_neighbors', 5)
                clf_factory = lambda k=_k: KNeighborsClassifier(n_neighbors=k)
            elif method == 'kernel_svm':
                clf_factory = lambda: SVC(kernel='rbf', random_state=42)

            # Evaluate on evaluation data.
            # In cross-dataset mode the classifier is trained on train activations
            # so that accuracy reflects genuine generalisation, not in-dataset CV.
            metrics, pos_proj, neg_proj = compute_separation_metrics(
                pos_acts_eval, neg_acts_eval, directions[method],
                use_cross_val=not cross_dataset,
                clf_factory=clf_factory,
                pos_acts_train=pos_acts_train if cross_dataset else None,
                neg_acts_train=neg_acts_train if cross_dataset else None,
            )
            layer_results[method] = {
                'metrics': metrics,
                'direction': directions[method]
            }

        if compute_mmd:
            layer_results['mmd_test'] = compute_kernel_mmd(
                pos_acts_eval, neg_acts_eval,
                n_permutations=mmd_permutations,
                max_samples=mmd_max_samples,
            )

        results_by_layer[layer_idx] = layer_results

    return results_by_layer


# =============================================================================
# CROSS-STAGE ANALYSIS FUNCTIONS WITH IMPROVED PLOTS
# =============================================================================

def compare_layer_wise_across_stages(
    stage_results: Dict[str, Dict[int, Dict]],
    output_dir: str,
    metric_name: str = 'classification_accuracy',
    methods: List[str] = ['mean_diff', 'logistic_regression'],
    show_ci: bool = False,
):
    """Create comprehensive layer-wise comparison across training stages."""
    print(f"\nCreating cross-stage layer-wise comparison for {metric_name}...")

    stages = list(stage_results.keys())
    n_stages = len(stages)

    fig = plt.figure(figsize=(8, 6 * len(methods)), facecolor='white')

    for method_idx, method in enumerate(methods):
        # Extract data for all stages
        stage_data = {}
        for stage in stages:
            if stage not in stage_results:
                continue
            layers = sorted(stage_results[stage].keys())
            values = [stage_results[stage][l][method]['metrics'][metric_name] for l in layers]
            ci_lower = [stage_results[stage][l][method]['metrics'].get('classification_accuracy_ci_lower') for l in layers]
            ci_upper = [stage_results[stage][l][method]['metrics'].get('classification_accuracy_ci_upper') for l in layers]
            stage_data[stage] = (layers, values, ci_lower, ci_upper)

        # Log confidence intervals if requested
        if show_ci and metric_name == 'classification_accuracy':
            print(f"\n  [{method}] 95% Confidence Intervals by layer:")
            for stage in stages:
                if stage not in stage_data:
                    continue
                layers, values, ci_lower, ci_upper = stage_data[stage]
                print(f"    {stage.upper()}:")
                for l, v, lo, hi in zip(layers, values, ci_lower, ci_upper):
                    if lo is not None and hi is not None:
                        print(f"      Layer {l:3d}: {v:.4f}  [{lo:.4f}, {hi:.4f}]")
                    else:
                        print(f"      Layer {l:3d}: {v:.4f}  [CI not available]")

        # Line plot comparison with improved styling
        ax1 = plt.subplot(len(methods), 1, method_idx + 1)
        for stage in stages:
            if stage in stage_data:
                layers, values, ci_lower, ci_upper = stage_data[stage]
                color = COLORS.get(stage, 'gray')
                ax1.plot(layers, values, marker='o', linewidth=3, markersize=8,
                        label=stage.upper(), color=color,
                        alpha=0.85, markeredgewidth=1.5, markeredgecolor='white')
                if show_ci and metric_name == 'classification_accuracy':
                    valid = [(l, lo, hi) for l, lo, hi in zip(layers, ci_lower, ci_upper)
                             if lo is not None and hi is not None]
                    if valid:
                        vl, vlo, vhi = zip(*valid)
                        ax1.fill_between(vl, vlo, vhi, color=color, alpha=0.15)
        
        ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.4, label='Random', zorder=0)
        ax1.set_xlabel('Layer Index', fontsize=13, fontweight='bold')
        ax1.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax1.set_title(f'{method.replace("_", " ").title()}\nEvolution Across Layers', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, frameon=True, shadow=True, fancybox=True)
        ax1.grid(True, alpha=0.2, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_ylim(bottom=0.45)

    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, f'cross_stage_layerwise_{metric_name}.pdf'),
                **PLOT_CONFIG)
    plt.close()


def analyze_direction_transfer(
    stage_activations: Dict[str, Tuple[Dict, Dict]],
    stage_directions: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    best_layer: int
):
    """Test if directions learned on one stage work well on other stages."""
    print("\nAnalyzing direction transfer across stages...")
    
    stages = list(stage_activations.keys())
    methods = list(stage_directions[stages[0]].keys())

    # Compute all transfer matrices once up front so they can be reused for both
    # plotting and the results dict without duplicating the expensive metric calls.
    all_transfer_matrices: Dict[str, np.ndarray] = {}
    # Only direction-based methods can be transferred (knn/kernel_svm have no direction).
    directional_methods = [m for m in methods if stage_directions[stages[0]].get(m) is not None]
    for method in directional_methods:
        matrix = np.zeros((len(stages), len(stages)))
        for i, source_stage in enumerate(stages):
            direction = stage_directions[source_stage][method]
            for j, test_stage in enumerate(stages):
                chosen_acts = stage_activations[test_stage][0][best_layer]
                rejected_acts = stage_activations[test_stage][1][best_layer]
                if direction.shape[0] != chosen_acts.shape[1]:
                    matrix[i, j] = float('nan')
                    continue
                metrics, _, _ = compute_separation_metrics(
                    chosen_acts, rejected_acts, direction, use_cross_val=True
                )
                matrix[i, j] = metrics['classification_accuracy']
        all_transfer_matrices[method] = matrix

    # Create transfer matrix: rows = direction source, cols = test stage
    # Only plot directional methods (knn/kernel_svm have no transferable direction)
    if not directional_methods:
        return {}
    fig, axes = plt.subplots(1, len(directional_methods), figsize=(8*len(directional_methods), 7), facecolor='white')
    if len(directional_methods) == 1:
        axes = [axes]

    for method_idx, method in enumerate(directional_methods):
        transfer_matrix = all_transfer_matrices[method]

        # Plot with improved styling
        ax = axes[method_idx]
        im = ax.imshow(transfer_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, interpolation='nearest')
        
        ax.set_xticks(range(len(stages)))
        ax.set_yticks(range(len(stages)))
        ax.set_xticklabels([s.upper() for s in stages], fontsize=12, fontweight='bold')
        ax.set_yticklabels([s.upper() for s in stages], fontsize=12, fontweight='bold')
        ax.set_xlabel('Test Stage', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Direction Source', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title(f'{method.replace("_", " ").title()}\nDirection Transfer Analysis', 
                    fontsize=15, fontweight='bold', pad=15)
        
        # Add text annotations with better styling
        for i in range(len(stages)):
            for j in range(len(stages)):
                value = transfer_matrix[i, j]
                text_color = 'white' if not np.isnan(value) and value < 0.7 else 'black'
                label = 'N/A' if np.isnan(value) else f'{value:.3f}'
                # Add background circle for better readability
                circle = plt.Circle((j, i), 0.35, color='white', alpha=0.3, zorder=1)
                ax.add_patch(circle)
                ax.text(j, i, label,
                       ha='center', va='center', color=text_color,
                       fontsize=13, fontweight='bold', zorder=2)
        
        # Add colorbar with better styling
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Accuracy', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
        cbar.ax.tick_params(labelsize=11)
        
        # Add grid for better readability
        ax.set_xticks([x - 0.5 for x in range(1, len(stages))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(stages))], minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1.5, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, 'direction_transfer_analysis.pdf'), **PLOT_CONFIG)
    plt.close()
    
    # Build transfer results from pre-computed matrices
    transfer_results = {}
    for method in directional_methods:
        transfer_matrix = all_transfer_matrices[method]
        # Diagonal = same-stage performance
        # Off-diagonal = transfer performance
        transfer_results[method] = {
            'matrix': transfer_matrix.tolist(),
            'mean_transfer': float(np.mean(transfer_matrix[np.triu_indices(len(stages), k=1)])),
            'mean_diagonal': float(np.mean(np.diag(transfer_matrix))),
        }
    
    with open(os.path.join(output_dir, 'direction_transfer_results.json'), 'w') as f:
        json.dump(transfer_results, f, indent=2)
    
    return transfer_results


def compare_feature_importance_across_stages(
    stage_activations: Dict[str, Tuple[Dict, Dict]],
    output_dir: str,
    best_layer: int,
    top_k: int = 50
):
    """Compare which features are important across different training stages."""
    print("\nComparing feature importance across stages...")
    
    stages = list(stage_activations.keys())
    
    # Compute Cohen's d for each stage
    stage_cohens_d = {}
    for stage in stages:
        chosen_acts = stage_activations[stage][0][best_layer]
        rejected_acts = stage_activations[stage][1][best_layer]
        
        chosen_mean = chosen_acts.mean(axis=0)
        rejected_mean = rejected_acts.mean(axis=0)
        mean_diff = chosen_mean - rejected_mean
        
        chosen_std = chosen_acts.std(axis=0)
        rejected_std = rejected_acts.std(axis=0)
        pooled_std = np.sqrt((chosen_std**2 + rejected_std**2) / 2)
        
        cohens_d = mean_diff / (pooled_std + 1e-8)
        stage_cohens_d[stage] = cohens_d
    
    # Find top features for each stage
    stage_top_features = {}
    for stage in stages:
        top_idx = np.argsort(np.abs(stage_cohens_d[stage]))[::-1][:top_k]
        stage_top_features[stage] = set(top_idx)
    
    # Compute overlap between stages
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor='white')
    
    # Plot 1: Feature overlap Venn-style with improved styling
    ax = axes[0]
    
    if len(stages) >= 2:
        # Jaccard similarity matrix
        n = len(stages)
        overlap_matrix = np.zeros((n, n))
        
        for i, s1 in enumerate(stages):
            for j, s2 in enumerate(stages):
                intersection = len(stage_top_features[s1] & stage_top_features[s2])
                union = len(stage_top_features[s1] | stage_top_features[s2])
                overlap_matrix[i, j] = intersection / union if union > 0 else 0
        
        im = ax.imshow(overlap_matrix, cmap='YlOrRd', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([s.upper() for s in stages], fontsize=13, fontweight='bold')
        ax.set_yticklabels([s.upper() for s in stages], fontsize=13, fontweight='bold')
        ax.set_title(f'Top-{top_k} Feature Overlap\n(Jaccard Similarity)', 
                    fontsize=15, fontweight='bold', pad=15)
        
        for i in range(n):
            for j in range(n):
                value = overlap_matrix[i, j]
                text_color = 'white' if value < 0.5 else 'black'
                # Add subtle background
                circle = plt.Circle((j, i), 0.35, color='white', alpha=0.2, zorder=1)
                ax.add_patch(circle)
                ax.text(j, i, f'{value:.2f}',
                       ha='center', va='center', color=text_color, 
                       fontsize=14, fontweight='bold', zorder=2)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Jaccard Similarity', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
        cbar.ax.tick_params(labelsize=11)
        
        # Add grid
        ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, n)], minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1.5, alpha=0.2)
    
    # Plot 2: Correlation of effect sizes across stages with improved styling
    ax = axes[1]
    
    if len(stages) >= 2:
        # Compute Spearman correlation of Cohen's d values
        n = len(stages)
        corr_matrix = np.zeros((n, n))
        
        for i, s1 in enumerate(stages):
            for j, s2 in enumerate(stages):
                corr, _ = spearmanr(stage_cohens_d[s1], stage_cohens_d[s2])
                corr_matrix[i, j] = corr
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([s.upper() for s in stages], fontsize=13, fontweight='bold')
        ax.set_yticklabels([s.upper() for s in stages], fontsize=13, fontweight='bold')
        ax.set_title('Effect Size Correlation\n(Spearman ρ)', 
                    fontsize=15, fontweight='bold', pad=15)
        
        for i in range(n):
            for j in range(n):
                value = corr_matrix[i, j]
                text_color = 'white' if abs(value) < 0.5 else 'black'
                # Add subtle background
                circle = plt.Circle((j, i), 0.35, color='white', alpha=0.2, zorder=1)
                ax.add_patch(circle)
                ax.text(j, i, f'{value:.2f}',
                       ha='center', va='center', color=text_color, 
                       fontsize=14, fontweight='bold', zorder=2)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Spearman ρ', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
        cbar.ax.tick_params(labelsize=11)
        
        # Add grid
        ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, n)], minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1.5, alpha=0.2)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, 'feature_importance_comparison.pdf'), **PLOT_CONFIG)
    plt.close()
    
    # Additional plot: Show top features across stages with improved styling
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
    
    # Get union of all top features
    all_top_features = sorted(set().union(*stage_top_features.values()))[:top_k]
    
    # Create matrix showing effect size for these features across stages
    matrix = np.zeros((len(stages), len(all_top_features)))
    for i, stage in enumerate(stages):
        for j, feat in enumerate(all_top_features):
            matrix[i, j] = stage_cohens_d[stage][feat]
    
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2, interpolation='bilinear')
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([s.upper() for s in stages], fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Index (top features across all stages)', 
                  fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title(f"Effect Size (Cohen's d) for Top-{top_k} Features", 
                fontsize=15, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Cohen's d", fontsize=12, fontweight='bold', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=11)
    
    # Add subtle grid
    ax.set_xticks([x - 0.5 for x in range(1, len(all_top_features))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(stages))], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_heatmap.pdf'), **PLOT_CONFIG)
    plt.close()
    
    # Save overlap statistics
    overlap_stats = {}
    for i, s1 in enumerate(stages):
        for s2 in stages[i+1:]:
            intersection = len(stage_top_features[s1] & stage_top_features[s2])
            union = len(stage_top_features[s1] | stage_top_features[s2])
            overlap_stats[f'{s1}_vs_{s2}'] = {
                'intersection': intersection,
                'union': union,
                'jaccard': intersection / union if union > 0 else 0,
                'overlap_pct': intersection / top_k
            }
    
    with open(os.path.join(output_dir, 'feature_overlap_stats.json'), 'w') as f:
        json.dump(overlap_stats, f, indent=2)


def create_summary_dashboard(
    stage_results: Dict[str, Dict],
    stage_activations: Dict[str, Tuple[Dict, Dict]],
    output_dir: str,
    best_layer: int,
    methods: List[str] = ['mean_diff', 'logistic_regression']
):
    """Create a comprehensive summary dashboard comparing all stages."""
    print("\nCreating summary dashboard...")
    
    stages = list(stage_results.keys())
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(22, 14), facecolor='white')
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # 1. Best layer accuracy comparison with improved styling
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(stages))
    width = 0.35
    
    for method_idx, method in enumerate(methods):
        accuracies = [stage_results[s][best_layer][method]['metrics']['classification_accuracy'] 
                     for s in stages]
        bars = ax1.bar(x_pos + (width * method_idx), accuracies, width, 
                      label=method.replace('_', ' ').title(),
                      alpha=0.85, edgecolor='white', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax1.set_title(f'Best Layer ({best_layer}) Accuracy Comparison', 
                 fontweight='bold', fontsize=13, pad=10)
    ax1.set_xticks(x_pos + width / 2)
    ax1.set_xticklabels([s.upper() for s in stages], fontweight='bold')
    ax1.legend(fontsize=10, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(bottom=0.45)
    
    # 2. Effect size progression with improved styling
    ax2 = fig.add_subplot(gs[0, 1])
    for method in methods:
        cohens_d_values = [stage_results[s][best_layer][method]['metrics']['cohens_d'] 
                          for s in stages]
        ax2.plot(stages, cohens_d_values, marker='o', linewidth=3.5, markersize=12,
                label=method.replace('_', ' ').title(), markeredgewidth=2, 
                markeredgecolor='white', alpha=0.85)
    
    ax2.set_ylabel("Cohen's d", fontweight='bold', fontsize=12)
    ax2.set_title('Effect Size Evolution Across Stages', 
                 fontweight='bold', fontsize=13, pad=10)
    ax2.set_xticklabels([s.upper() for s in stages], fontweight='bold')
    ax2.legend(fontsize=10, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Mean separation distance with improved styling
    ax3 = fig.add_subplot(gs[0, 2])
    for method in methods:
        distances = [stage_results[s][best_layer][method]['metrics']['euclidean_distance'] 
                    for s in stages]
        ax3.plot(stages, distances, marker='s', linewidth=3.5, markersize=12,
                label=method.replace('_', ' ').title(), markeredgewidth=2,
                markeredgecolor='white', alpha=0.85)
    
    ax3.set_ylabel('Euclidean Distance', fontweight='bold', fontsize=12)
    ax3.set_title('Mean Separation Distance', fontweight='bold', fontsize=13, pad=10)
    ax3.set_xticklabels([s.upper() for s in stages], fontweight='bold')
    ax3.legend(fontsize=10, frameon=True, shadow=True)
    ax3.grid(True, alpha=0.2, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4-6. Layer-wise progression for each stage with improved styling
    for idx, stage in enumerate(stages):
        ax = fig.add_subplot(gs[1, idx])
        
        layers = sorted(stage_results[stage].keys())
        for method in methods:
            accuracies = [stage_results[stage][l][method]['metrics']['classification_accuracy'] 
                         for l in layers]
            ax.plot(layers, accuracies, marker='o', linewidth=2.5, markersize=7,
                   label=method.replace('_', ' ').title(), markeredgewidth=1.5,
                   markeredgecolor='white', alpha=0.85)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=2)
        ax.axvline(x=best_layer, color='red', linestyle=':', alpha=0.6, 
                  linewidth=2.5, label='Best Layer')
        ax.set_xlabel('Layer', fontweight='bold', fontsize=11)
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
        ax.set_title(f'{stage.upper()}\nLayer Progression', 
                    fontweight='bold', fontsize=12, pad=10)
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0.45)
    
    # 7. Summary statistics table with improved styling
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    # Create summary table — use first available method for the reference metrics
    _table_method = next(
        (m for m in ['logistic_regression', 'mean_diff', 'knn', 'kernel_svm']
         if m in stage_results[stages[0]][sorted(stage_results[stages[0]].keys())[0]]),
        methods[0]
    )
    table_data = [['Stage', 'Best Accuracy', 'Mean Accuracy', 'Max Cohen\'s d', 'Avg Distance']]

    for stage in stages:
        layers = sorted(stage_results[stage].keys())
        accuracies = [stage_results[stage][layer_idx][_table_method]['metrics']['classification_accuracy']
                     for layer_idx in layers]
        raw_cohens_d = [stage_results[stage][layer_idx][_table_method]['metrics']['cohens_d']
                        for layer_idx in layers]
        cohens_d = [abs(v) for v in raw_cohens_d if not np.isnan(v)]
        distances = [stage_results[stage][layer_idx][_table_method]['metrics']['euclidean_distance']
                    for layer_idx in layers]
        
        table_data.append([
            stage.upper(),
            f"{max(accuracies):.4f}",
            f"{np.mean(accuracies):.4f}",
            f"{max(cohens_d):.4f}" if cohens_d else "N/A",
            f"{np.mean(distances):.2f}"
        ])
    
    table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    # Style header row with gradient
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(weight='bold', color='white', fontsize=13)
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
    
    # Style data rows with stage colors
    for i, stage in enumerate(stages, 1):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if j == 0:
                cell.set_facecolor(COLORS.get(stage, 'lightgray'))
                cell.set_text_props(weight='bold', fontsize=12)
            else:
                cell.set_facecolor('#ecf0f1')
                cell.set_text_props(fontsize=11)
            cell.set_edgecolor('white')
            cell.set_linewidth(1.5)
    
    fig.suptitle('Cross-Stage Probe Analysis Summary Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(output_dir, 'summary_dashboard.pdf'), **PLOT_CONFIG)
    plt.close()


def visualize_2d_separation_multi_stage(
    stage_activations: Dict[str, Tuple[Dict, Dict]],
    stage_directions: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    best_layer: int,
    method: str = 'mean_diff'
):
    """Visualize 2D separation for all stages in one plot."""
    print(f"\nCreating multi-stage 2D separation visualization...")
    
    stages = list(stage_activations.keys())
    n_stages = len(stages)
    
    fig, axes = plt.subplots(1, n_stages, figsize=(8*n_stages, 7), facecolor='white')
    if n_stages == 1:
        axes = [axes]
    
    for idx, stage in enumerate(stages):
        chosen_acts = stage_activations[stage][0][best_layer]
        rejected_acts = stage_activations[stage][1][best_layer]
        direction = stage_directions[stage][method]
        
        all_acts = np.vstack([chosen_acts, rejected_acts])
        pca = PCA(n_components=2)
        acts_2d = pca.fit_transform(all_acts)
        
        chosen_2d = acts_2d[:len(chosen_acts)]
        rejected_2d = acts_2d[len(chosen_acts):]
        
        # Project direction onto 2D
        direction_2d = pca.transform(direction.reshape(1, -1))[0]
        direction_2d = direction_2d / (np.linalg.norm(direction_2d) + 1e-8)
        
        ax = axes[idx]
        
        # Plot rejected with subtle border
        ax.scatter(rejected_2d[:, 0], rejected_2d[:, 1], 
                  c=COLORS['rejected'], alpha=0.45, s=60, 
                  label='Rejected', edgecolors='white', linewidths=0.5)
        
        # Plot chosen with subtle border
        ax.scatter(chosen_2d[:, 0], chosen_2d[:, 1], 
                  c=COLORS['chosen'], alpha=0.65, s=60, 
                  label='Chosen', edgecolors='white', linewidths=0.5)
        
        # Plot direction arrow with better styling
        scale = max(abs(acts_2d[:, 0]).max(), abs(acts_2d[:, 1]).max()) * 0.45
        ax.arrow(0, 0, direction_2d[0]*scale, direction_2d[1]*scale,
                head_width=scale*0.1, head_length=scale*0.06, 
                fc=COLORS['direction'], ec=COLORS['direction'],
                linewidth=4, label='Direction', alpha=0.9, zorder=5)
        
        # Centroids with halos
        # Chosen centroid
        # ax.scatter(*chosen_2d.mean(axis=0), c=COLORS['chosen'], 
        #           marker='*', s=500, alpha=0.3, zorder=8, 
        #           edgecolors='black')  # Halo
        ax.scatter(*chosen_2d.mean(axis=0), c='darkblue',
                  marker='*', s=300,
                  edgecolors='black', linewidths=2, label='Chosen Centre', zorder=10)

        # Rejected centroid
        ax.scatter(*rejected_2d.mean(axis=0), c='darkred',
                  marker='*', s=300, 
                  edgecolors='black', linewidths=2, label='Rejected Centre', zorder=10)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                     fontweight='bold', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', 
                     fontweight='bold', fontsize=12)
        ax.set_title(f'{stage.upper()}\nLayer {best_layer}', 
                    fontsize=14, fontweight='bold', pad=12)
        ax.legend(loc='best', fontsize=10, frameon=True, shadow=True, 
                 fancybox=True, framealpha=0.95)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#fafafa')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, f'2d_separation_all_stages_{method}.pdf'), 
                **PLOT_CONFIG)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Enhanced Probe Analysis for APO")
    
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--dataset-language", type=str, default="amh")
    parser.add_argument("--max-samples", type=int, default=500,
                       help="Number of samples for training (can be less than 500)")
    parser.add_argument("--min-eval-samples", type=int, default=500,
                       help="Minimum number of samples for evaluation (default: 500)")
    parser.add_argument("--eval-dataset", type=str, default=None,
                       help="Dataset to evaluate on (for cross-dataset generalisation). "
                            "If not set, evaluation uses the training dataset.")
    parser.add_argument("--eval-dataset-language", type=str, default=None,
                       help="Language arg for eval dataset (e.g. for AfriSenti). "
                            "Defaults to --dataset-language if not set.")
    parser.add_argument("--eval-max-samples", type=int, default=None,
                       help="Number of eval samples from the eval dataset. "
                            "Defaults to --min-eval-samples.")
    parser.add_argument("--output-dir", type=str, default="./probe_analysis_enhanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-model-path", type=str, default=None)
    parser.add_argument("--po-model-path", type=str, default=None)
    parser.add_argument("--position", type=str, choices=["completion_mean", "last_token", "mean", "max", "last_prompt"],
                       default="completion_mean")
    parser.add_argument("--methods", type=str, nargs='+',
                       default=['logistic_regression'],
                       help="Probe methods: mean_diff, logistic_regression, knn, kernel_svm")
    parser.add_argument("--knn-neighbors", type=int, default=5,
                       help="Number of neighbours for k-NN probe (default: 5)")
    parser.add_argument("--confidence-intervals", action="store_true",
                       help="Log 95%% CI for classification accuracy across all layers "
                            "and shade them in the cross_stage_layerwise_classification_accuracy.pdf plot.")
    parser.add_argument("--layers", type=int, nargs='+', default=None,
                       help="Layer indices to concatenate for multi-layer classification "
                            "(e.g. --layers 8 12 16). When set, runs an additional probe "
                            "trained on the concatenated activations from those layers.")
    parser.add_argument("--disable-mmd", action="store_true",
                       help="Skip the kernel-MMD two-sample test (adds O(n^2) cost per layer).")
    parser.add_argument("--mmd-permutations", type=int, default=200,
                       help="Number of label permutations for the kernel-MMD p-value (default: 200)")
    parser.add_argument("--mmd-max-samples", type=int, default=250,
                       help="Max samples per group for the kernel-MMD test, subsampled for "
                            "O(n^2) tractability on full-dimensional activations (default: 250)")
    parser.add_argument("--shuffle-labels", action="store_true",
                       help="Sanity check (paper Appendix A.1): randomly swap chosen/rejected "
                            "labels before extracting activations. A well-behaved probe should "
                            "collapse to chance and the PCA clusters/centroids should coincide, "
                            "confirming the separation seen with real labels is a genuine "
                            "preference signal rather than a visualisation artefact.")

    args = parser.parse_args()

    # Resolve eval dataset settings
    eval_dataset_name = args.eval_dataset or args.dataset
    eval_dataset_language = args.eval_dataset_language or args.dataset_language
    cross_dataset = args.eval_dataset is not None and args.eval_dataset != args.dataset

    # Ensure eval_samples is at least min_eval_samples
    eval_samples = max(args.max_samples, args.min_eval_samples)
    eval_max_samples = args.eval_max_samples if args.eval_max_samples is not None else eval_samples

    if args.max_samples < args.min_eval_samples:
        print(f"ℹ Info: Training on {args.max_samples} samples, but evaluating on {eval_samples} samples (minimum {args.min_eval_samples})")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Propagate k-NN neighbour count to the analysis function via a function attribute
    analyze_layer_wise_accuracy._knn_neighbors = args.knn_neighbors

    print("="*70)
    print("ENHANCED PROBE ANALYSIS FOR APO")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Train dataset: {args.dataset}")
    if cross_dataset:
        print(f"Eval dataset:  {eval_dataset_name}  (cross-dataset generalisation)")
    else:
        print(f"Eval dataset:  {eval_dataset_name}  (same as train)")
    print(f"Training samples: {args.max_samples}")
    print(f"Evaluation samples: {eval_max_samples}")
    print(f"Position: {args.position}")
    print(f"Output: {args.output_dir}")
    print("="*70)

    # Load training dataset (always needed)
    dataset = load_preference_dataset(args.dataset, max_samples=eval_samples, dataset_args=args.dataset_language)
    if "afrisenti" in args.dataset.lower() and args.dataset_language:
        pos_pending = []
        neg_pending = []
        ds = dataset
        dataset = []
        for point in ds:
            if point["label"] == "positive":
                pos_pending.append(point)
            elif point["label"] == "negative":
                neg_pending.append(point)
            if pos_pending and neg_pending:
                dataset.append({
                    "prompt": [],
                    "chosen": [{"role": "assistant", "content": pos_pending.pop()["tweet"]}],
                    "rejected": [{"role": "assistant", "content": neg_pending.pop()["tweet"]}]
                })

    # Load eval dataset (separate if --eval-dataset provided, else reuse train dataset)
    if cross_dataset:
        print(f"\nLoading eval dataset: {eval_dataset_name} ({eval_max_samples} samples)...")
        eval_dataset = load_preference_dataset(eval_dataset_name, max_samples=eval_max_samples, dataset_args=eval_dataset_language)
    else:
        eval_dataset = dataset

    # Sanity check (paper Appendix A.1): randomly swap chosen/rejected labels so any
    # apparent separation must come from visualisation/statistical artefacts rather
    # than genuine preference structure.
    if args.shuffle_labels:
        print("\n⚠ --shuffle-labels: randomly swapping chosen/rejected labels (sanity check)")
        rng = random.Random(args.seed)

        def swap_chosen_rejected(example, swap_prob=0.5):
            if rng.random() < swap_prob:
                example = dict(example)
                example['chosen'], example['rejected'] = example['rejected'], example['chosen']
            return example

        dataset = [swap_chosen_rejected(example) for example in dataset]
        if cross_dataset:
            eval_dataset = [swap_chosen_rejected(example) for example in eval_dataset]
        else:
            eval_dataset = dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.chat_template = "{{- bos_token }}\n{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Storage for cross-stage analysis
    stage_results = {}
    stage_activations = {}
    stage_directions = {}

    # Analyze pretrained model
    print("\n" + "="*70)
    print("ANALYZING PRETRAINED MODEL")
    print("="*70)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map="auto", 
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # Extract training activations
    print(f"\nExtracting training activations ({args.max_samples} samples)...")
    chosen_by_layer_train, rejected_by_layer_train = extract_activations_all_layers(
        model, tokenizer, dataset, args.max_samples, args.position
    )

    # Extract evaluation activations (from eval_dataset, which may differ from train)
    if cross_dataset:
        print(f"\nExtracting eval activations from eval dataset ({eval_max_samples} samples)...")
        chosen_by_layer_eval, rejected_by_layer_eval = extract_activations_all_layers(
            model, tokenizer, eval_dataset, eval_max_samples, args.position
        )
    elif eval_samples > args.max_samples:
        print(f"\nExtracting evaluation activations ({eval_samples} samples)...")
        chosen_by_layer_eval, rejected_by_layer_eval = extract_activations_all_layers(
            model, tokenizer, dataset, eval_samples, args.position
        )
    else:
        chosen_by_layer_eval = chosen_by_layer_train
        rejected_by_layer_eval = rejected_by_layer_train

    layer_results = analyze_layer_wise_accuracy(
        chosen_by_layer_train, rejected_by_layer_train,
        chosen_by_layer_eval, rejected_by_layer_eval,
        args.methods,
        cross_dataset=cross_dataset,
        compute_mmd=not args.disable_mmd,
        mmd_permutations=args.mmd_permutations,
        mmd_max_samples=args.mmd_max_samples,
    )

    multilayer_results = {}
    if args.layers:
        multilayer_results['pretrained'] = analyze_multilayer_accuracy(
            chosen_by_layer_train, rejected_by_layer_train,
            args.layers,
            chosen_by_layer_eval, rejected_by_layer_eval,
            args.methods,
            cross_dataset=cross_dataset,
            compute_mmd=not args.disable_mmd,
            mmd_permutations=args.mmd_permutations,
            mmd_max_samples=args.mmd_max_samples,
        )
        _ref = next((m for m in ['logistic_regression', 'mean_diff', 'knn', 'kernel_svm']
                     if m in multilayer_results['pretrained']), args.methods[0])
        _acc = multilayer_results['pretrained'][_ref]['metrics']['classification_accuracy']
        print(f"  Multi-layer ({args.layers}) accuracy [{_ref}]: {_acc:.4f}")

    # Pick a reference method for best-layer selection (prefer directional methods for stability)
    _ref_method = next(
        (m for m in ['logistic_regression', 'mean_diff', 'knn', 'kernel_svm'] if m in args.methods),
        args.methods[0]
    )
    best_layer = max(layer_results.keys(),
                     key=lambda layer_idx: layer_results[layer_idx][_ref_method]['metrics']['classification_accuracy'])
    print(f"\n✓ Best layer for pretrained: {best_layer}  (ranked by {_ref_method})")
    _m = layer_results[best_layer][_ref_method]['metrics']
    print(f"  PC1 explained variance: {_m['pc1_explained_variance']:.3f}")
    print(f"  PC1 centroids — chosen: {_m['pc1_mean_pos']:.4f}, rejected: {_m['pc1_mean_neg']:.4f}")
    print(f"  t-test:        t={_m['pc1_ttest_stat']:.3f}, p={_m['pc1_ttest_pval']:.2e}")
    print(f"  Mann-Whitney:  U={_m['pc1_mannwhitney_stat']:.1f}, p={_m['pc1_mannwhitney_pval']:.2e}  {'(significant)' if _m['pc1_significant'] else '(not significant)'}")
    if not args.disable_mmd:
        _mmd = layer_results[best_layer]['mmd_test']
        print(f"  Kernel MMD²:   {_mmd['mmd2']:.4f}  (RBF bandwidth σ²={_mmd['mmd_bandwidth_sigma2']:.2f}, "
              f"n={_mmd['mmd_n_samples']})")
        print(f"  Permutation p: {_mmd['mmd_pvalue']:.4f}  ({_mmd['mmd_n_permutations']} permutations)  "
              f"{'(significant)' if _mmd['mmd_significant'] else '(not significant)'}")

    # Store results (use eval data for cross-stage analysis)
    stage_results['pretrained'] = layer_results
    stage_activations['pretrained'] = (chosen_by_layer_eval, rejected_by_layer_eval)
    
    # Compute and store directions for best layer (use training data)
    pos_acts = chosen_by_layer_train[best_layer]
    neg_acts = rejected_by_layer_train[best_layer]
    directions = compute_all_directions(pos_acts, neg_acts, args.methods)
    stage_directions['pretrained'] = directions

    del model
    torch.cuda.empty_cache()

    # Analyze SFT model if provided
    if args.sft_model_path:
        print("\n" + "="*70)
        print("ANALYZING SFT MODEL")
        print("="*70)
        
        sft_model = AutoModelForCausalLM.from_pretrained(
            args.sft_model_path, device_map="auto", torch_dtype="auto"
        )
        sft_tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
        
        # Extract training activations
        print(f"\nExtracting training activations ({args.max_samples} samples)...")
        sft_chosen_train, sft_rejected_train = extract_activations_all_layers(
            sft_model, sft_tokenizer, dataset, args.max_samples, args.position
        )

        # Extract evaluation activations (from eval_dataset, which may differ from train)
        if cross_dataset:
            print(f"\nExtracting eval activations from eval dataset ({eval_max_samples} samples)...")
            sft_chosen_eval, sft_rejected_eval = extract_activations_all_layers(
                sft_model, sft_tokenizer, eval_dataset, eval_max_samples, args.position
            )
        elif eval_samples > args.max_samples:
            print(f"\nExtracting evaluation activations ({eval_samples} samples)...")
            sft_chosen_eval, sft_rejected_eval = extract_activations_all_layers(
                sft_model, sft_tokenizer, dataset, eval_samples, args.position
            )
        else:
            sft_chosen_eval = sft_chosen_train
            sft_rejected_eval = sft_rejected_train
        
        sft_results = analyze_layer_wise_accuracy(
            sft_chosen_train, sft_rejected_train,
            sft_chosen_eval, sft_rejected_eval,
            args.methods,
            cross_dataset=cross_dataset,
            compute_mmd=not args.disable_mmd,
            mmd_permutations=args.mmd_permutations,
            mmd_max_samples=args.mmd_max_samples,
        )

        if args.layers:
            multilayer_results['sft'] = analyze_multilayer_accuracy(
                sft_chosen_train, sft_rejected_train,
                args.layers,
                sft_chosen_eval, sft_rejected_eval,
                args.methods,
                cross_dataset=cross_dataset,
                compute_mmd=not args.disable_mmd,
                mmd_permutations=args.mmd_permutations,
                mmd_max_samples=args.mmd_max_samples,
            )
            _ref = next((m for m in ['logistic_regression', 'mean_diff', 'knn', 'kernel_svm']
                         if m in multilayer_results['sft']), args.methods[0])
            _acc = multilayer_results['sft'][_ref]['metrics']['classification_accuracy']
            print(f"  Multi-layer ({args.layers}) accuracy [{_ref}]: {_acc:.4f}")

        stage_results['sft'] = sft_results
        stage_activations['sft'] = (sft_chosen_eval, sft_rejected_eval)
        
        sft_pos = sft_chosen_train[best_layer]
        sft_neg = sft_rejected_train[best_layer]
        stage_directions['sft'] = compute_all_directions(sft_pos, sft_neg, args.methods)
        
        del sft_model
        torch.cuda.empty_cache()

    # Analyze PO model if provided
    if args.po_model_path:
        print("\n" + "="*70)
        print("ANALYZING PO MODEL")
        print("="*70)
        
        po_model = AutoModelForCausalLM.from_pretrained(
            args.po_model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
        po_tokenizer = AutoTokenizer.from_pretrained(args.po_model_path)
        
        # Extract training activations
        print(f"\nExtracting training activations ({args.max_samples} samples)...")
        po_chosen_train, po_rejected_train = extract_activations_all_layers(
            po_model, po_tokenizer, dataset, args.max_samples, args.position
        )

        # Extract evaluation activations (from eval_dataset, which may differ from train)
        if cross_dataset:
            print(f"\nExtracting eval activations from eval dataset ({eval_max_samples} samples)...")
            po_chosen_eval, po_rejected_eval = extract_activations_all_layers(
                po_model, po_tokenizer, eval_dataset, eval_max_samples, args.position
            )
        elif eval_samples > args.max_samples:
            print(f"\nExtracting evaluation activations ({eval_samples} samples)...")
            po_chosen_eval, po_rejected_eval = extract_activations_all_layers(
                po_model, po_tokenizer, dataset, eval_samples, args.position
            )
        else:
            po_chosen_eval = po_chosen_train
            po_rejected_eval = po_rejected_train
        
        po_results = analyze_layer_wise_accuracy(
            po_chosen_train, po_rejected_train,
            po_chosen_eval, po_rejected_eval,
            args.methods,
            cross_dataset=cross_dataset,
            compute_mmd=not args.disable_mmd,
            mmd_permutations=args.mmd_permutations,
            mmd_max_samples=args.mmd_max_samples,
        )

        if args.layers:
            multilayer_results['po'] = analyze_multilayer_accuracy(
                po_chosen_train, po_rejected_train,
                args.layers,
                po_chosen_eval, po_rejected_eval,
                args.methods,
                cross_dataset=cross_dataset,
                compute_mmd=not args.disable_mmd,
                mmd_permutations=args.mmd_permutations,
                mmd_max_samples=args.mmd_max_samples,
            )
            _ref = next((m for m in ['logistic_regression', 'mean_diff', 'knn', 'kernel_svm']
                         if m in multilayer_results['po']), args.methods[0])
            _acc = multilayer_results['po'][_ref]['metrics']['classification_accuracy']
            print(f"  Multi-layer ({args.layers}) accuracy [{_ref}]: {_acc:.4f}")

        stage_results['po'] = po_results
        stage_activations['po'] = (po_chosen_eval, po_rejected_eval)
        
        po_pos = po_chosen_train[best_layer]
        po_neg = po_rejected_train[best_layer]
        stage_directions['po'] = compute_all_directions(po_pos, po_neg, args.methods)
        
        del po_model
        torch.cuda.empty_cache()

    # ==================================================================
    # CROSS-STAGE ANALYSIS WITH IMPROVED VISUALIZATIONS
    # ==================================================================

    print("\n" + "="*70)
    print("CROSS-STAGE ANALYSIS")
    print("="*70)

    # Layer-wise comparison (works with any number of stages)
    compare_layer_wise_across_stages(
        stage_results, args.output_dir,
        'classification_accuracy', args.methods,
        show_ci=args.confidence_intervals,
    )

    compare_layer_wise_across_stages(
        stage_results, args.output_dir,
        'cohens_d', args.methods,
    )

    # Summary dashboard (works with any number of stages)
    create_summary_dashboard(
        stage_results, stage_activations, args.output_dir, best_layer, args.methods
    )

    # Multi-stage 2D visualization — only for directional methods
    directional_methods = [
        m for m in args.methods
        if stage_directions['pretrained'].get(m) is not None
    ]
    for method in directional_methods:
        visualize_2d_separation_multi_stage(
            stage_activations, stage_directions, args.output_dir, best_layer, method
        )

    if len(stage_results) > 1:
        # Direction transfer analysis
        transfer_results = analyze_direction_transfer(
            stage_activations, stage_directions, args.output_dir, best_layer
        )

        # Feature importance comparison
        compare_feature_importance_across_stages(
            stage_activations, args.output_dir, best_layer, top_k=50
        )

    # save all metrics for best layer
    def convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return obj

    with open(f"{args.output_dir}/best_layer_metrics.json", "w") as f:
        serializable_data = {
            stage: convert_to_serializable(stage_results[stage][best_layer])
            for stage in stage_results
        }
        json.dump(serializable_data, f, indent=2)

    if multilayer_results:
        with open(f"{args.output_dir}/multilayer_metrics.json", "w") as f:
            json.dump(convert_to_serializable(multilayer_results), f, indent=2)
        print(f"\nMulti-layer results saved to: {args.output_dir}/multilayer_metrics.json")

    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print("="*70)
    print(f"\nTraining samples: {args.max_samples} (dataset: {args.dataset})")
    if cross_dataset:
        print(f"Evaluation samples: {eval_max_samples} (dataset: {eval_dataset_name}  — cross-dataset)")
    else:
        print(f"Evaluation samples: {eval_max_samples}")
    print("\nGenerated visualizations (high-quality PDFs at 150 DPI):")
    print("  • cross_stage_layerwise_*.pdf - Layer progression across stages")
    print("  • direction_transfer_analysis.pdf - Direction transferability")
    print("  • feature_importance_comparison.pdf - Feature overlap analysis")
    print("  • summary_dashboard.pdf - Comprehensive overview")
    print("  • 2d_separation_all_stages_*.pdf - Visual separation comparison")
    print("="*70)


if __name__ == "__main__":
    main()
