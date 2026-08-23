#!/usr/bin/env python3
"""
Run probe_analysis.py on oasst2 for ~10 languages using Qwen3-0.6B
and produce a final multilingual recap plot.
"""

import os
import json
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = "google/gemma-3-4b-it"
DATASET = "oasst2"
MAX_SAMPLES = 500
POSITION = "completion_mean"
OUTPUT_BASE = "./probe_analysis_oasst2_multilingual"

# ~10 languages well-represented in oasst2
LANGUAGES = ["en", "de", "fr", "es", "zh", "ru", "pt", "it", "ja", "ko"]

LANG_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "zh": "Chinese",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
}


def run_language(lang: str):
    output_dir = os.path.join(OUTPUT_BASE, lang)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "probe_analysis.py",
        "--model-name", MODEL,
        "--dataset", DATASET,
        "--dataset-language", lang,
        "--max-samples", str(MAX_SAMPLES),
        "--position", POSITION,
        "--output-dir", output_dir,
        "--methods", "logistic_regression",
    ]

    print(f"\n{'='*65}")
    print(f"Running: {lang} ({LANG_NAMES.get(lang, lang)})")
    print(f"{'='*65}")

    result = subprocess.run(cmd)
    return result.returncode == 0, output_dir


def load_best_layer_metrics(output_dir: str):
    path = os.path.join(output_dir, "best_layer_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_recap(results_by_lang: dict):
    valid_langs = [l for l in LANGUAGES if results_by_lang.get(l) is not None]

    if not valid_langs:
        print("No valid results — skipping recap plot.")
        return

    accuracies, cohens_ds, stds = [], [], []
    for lang in valid_langs:
        lr = (results_by_lang[lang]
              .get("pretrained", {})
              .get("logistic_regression", {})
              .get("metrics", {}))
        accuracies.append(lr.get("classification_accuracy", float("nan")))
        cohens_ds.append(abs(lr.get("cohens_d", float("nan"))))
        stds.append(lr.get("classification_accuracy_std", 0.0) or 0.0)

    labels = [LANG_NAMES.get(l, l) for l in valid_langs]
    x = np.arange(len(valid_langs))
    colors = plt.cm.Set2(np.linspace(0, 1, len(valid_langs)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    # --- Panel 1: classification accuracy ---
    ax = axes[0]
    bars = ax.bar(x, accuracies, color=colors, edgecolor="white",
                  linewidth=1.5, alpha=0.85, yerr=stds, capsize=4,
                  error_kw=dict(elinewidth=1.5, ecolor="gray", capthick=1.5))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5,
               alpha=0.6, label="Chance (0.5)")
    for bar, val in zip(bars, accuracies):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylim(0.40, 1.05)
    ax.set_ylabel("Classification Accuracy (best layer)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Probe Accuracy per Language\n"
        f"({MODEL}, oasst2, {MAX_SAMPLES} samples, {POSITION})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # --- Panel 2: |Cohen's d| ---
    ax = axes[1]
    bars = ax.bar(x, cohens_ds, color=colors, edgecolor="white",
                  linewidth=1.5, alpha=0.85)
    for bar, val in zip(bars, cohens_ds):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("|Cohen's d| (best layer)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Effect Size per Language\n"
        f"({MODEL}, oasst2, {MAX_SAMPLES} samples, {POSITION})",
        fontsize=12, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout(pad=2.0)
    out_path = os.path.join(OUTPUT_BASE, "multilingual_recap.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nRecap plot saved to: {out_path}")
    plt.close()


def print_summary(results_by_lang: dict):
    print("\n" + "="*55)
    print(f"{'Lang':<6} {'Name':<12} {'Accuracy':>10} {'|Cohen d|':>10} {'Std':>8}")
    print("-" * 50)
    for lang in LANGUAGES:
        res = results_by_lang.get(lang)
        if res is None:
            print(f"{lang:<6} {LANG_NAMES.get(lang, lang):<12} {'FAILED':>10}")
            continue
        lr = (res.get("pretrained", {})
               .get("logistic_regression", {})
               .get("metrics", {}))
        acc = lr.get("classification_accuracy", float("nan"))
        cd = abs(lr.get("cohens_d", float("nan")))
        std = lr.get("classification_accuracy_std", float("nan"))
        print(f"{lang:<6} {LANG_NAMES.get(lang, lang):<12} {acc:>10.4f} {cd:>10.4f} {std:>8.4f}")


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    results_by_lang = {}
    for lang in LANGUAGES:
        success, output_dir = run_language(lang)
        if not success:
            print(f"WARNING: run for '{lang}' exited with non-zero code — partial results may exist.")
        results_by_lang[lang] = load_best_layer_metrics(output_dir)

    print("\n\n" + "="*65)
    print("ALL RUNS COMPLETE — generating recap plot")
    print("="*65)

    plot_recap(results_by_lang)
    print_summary(results_by_lang)


if __name__ == "__main__":
    main()
