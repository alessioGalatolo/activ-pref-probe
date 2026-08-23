# Low-Resource Preference Adaptation of LLMs via Activation-Based Label Propagation

Code for the paper *"Low-Resource Preference Adaptation of LLMs via Activation-Based Label Propagation"* (Alessio Galatolo, Meriem Beloucif — Uppsala University).

We show that language models encode preference information (chosen vs. rejected responses) as geometrically separable clusters in their activations, even before any preference optimisation. We exploit this by training a lightweight linear probe on a handful of labelled preference pairs (≤500), using it to label a large unlabelled corpus (50K+), and then running standard preference optimisation (DPO/IPO/CPO/KTO) on the probe-labelled data. This consistently outperforms training directly on the same small labelled set, and remains competitive with baselines trained on 50–100× more labelled data.

```
1. Labelled preference pairs (n ≤ 500)
        │
        ▼
2. Extract activations from an LLM (chosen vs. rejected)
        │
        ▼
3. Train a linear probe on the activation difference
        │
        ▼
4. Label a large unlabelled dataset (50K+) with the probe
        │
        ▼
5. Run preference optimisation (DPO / IPO / CPO / KTO) on the probe-labelled data
```

## Installation

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU for anything beyond the smallest models (the paper's experiments use an A100 80GB for training and an A40 for evaluation). Several models and datasets used in the paper (e.g. Llama 3.2, Llama 2) are gated on Hugging Face — request access and run `huggingface-cli login` (or set `HF_TOKEN`) before using them.

## Quick start

Run the full pipeline (SFT → probe training → label propagation → preference optimisation → LLM-as-a-judge evaluation) on a small model:

```bash
python main.py \
    --model-name Qwen/Qwen3-0.6B \
    --do-sft --sft-dataset tatsu-lab/alpaca --sft-max-samples 1000 \
    --po-dataset Anthropic/hh-rlhf --po-method dpo \
    --probe-subset-size 500 --po-max-samples 5000 \
    --baseline original --eval-samples 100
```

This trains two LoRA-adapted models — one on probe-propagated labels, one on the `--baseline` (original human labels / randomly flipped labels / no preference optimisation) — and reports an LLM-as-a-judge win/tie/loss comparison. See `python main.py --help` for the full set of options (probe layers/type, PO method and hyperparameters, checkpoint-based evaluation, wandb logging, etc.).

## Repository structure

### Core pipeline (used by `main.py`)

| File | Purpose |
|---|---|
| `main.py` | CLI entry point: orchestrates SFT → probe training → label propagation → preference optimisation → evaluation. |
| `config.py` | `APOConfig` dataclass holding all pipeline hyperparameters. |
| `dataset_utils.py` | Loads and formats preference datasets (HH-RLHF, UltraFeedback, Nectar, PRISM, oasst2, AfriSenti, CValues) into a common `{prompt, chosen, rejected}` schema. |
| `model_utils.py` | Locates a model's transformer layer list across different architectures. |
| `activation_extractor.py` | Forward-hook utility that extracts last-token activations from specified layers. |
| `probes.py` | `PreferenceProbe` wrapper around a logistic-regression or small MLP probe. |
| `probe_training.py` | Trains the probe on a labelled subset and uses it to relabel (propagate labels to) the large unlabelled dataset. |
| `training.py` | SFT and preference-optimisation (DPO/IPO/CPO/KTO) training loops, built on TRL + PEFT/LoRA. |
| `callbacks.py` | Saves checkpoints at fixed training-progress intervals, for the time-series evaluation. |
| `evaluation.py` | LLM-as-a-judge win/tie/loss evaluation, ground-truth (classification) evaluation, and checkpoint cross-comparison matrices. |
| `wandb_utils.py` | Weights & Biases init/logging helpers (optional; disable with `--no-wandb`). |

### Standalone analysis & evaluation scripts

These reproduce the paper's other experiments and can be run independently of `main.py` (they generally re-use the modules above).

| File | Purpose |
|---|---|
| `probe_analysis.py` | Produces the paper's preference-geometry results (Section 3): PCA cluster visualisations with centroids, layer-wise probe accuracy, Cohen's d, PC1 t-test/Mann-Whitney significance, kernel-MMD two-sample test, cross-stage (pretrained/SFT/PO) and cross-dataset comparisons, direction-transfer and feature-importance analysis. `--shuffle-labels` reruns everything with chosen/rejected swapped as the sanity check in Appendix A.1. |
| `run_oasst2_multilingual.py` | Driver that runs `probe_analysis.py` across ~10 oasst2 languages and plots a per-language accuracy/effect-size recap (Section 3.4). |
| `probe_gold_eval.py` | Trains a probe and scores it against held-out gold labels: pointwise/pairwise accuracy, ROC/PR curves, calibration (ECE/reliability diagram), error analysis, and a length-confound analysis. |
| `lora_classifier_eval.py` | Baseline: a LoRA-fine-tuned sequence-classification head trained end-to-end on the same labelled budget, scored with the same metrics as `probe_gold_eval.py` (Table 2). |
| `llm_judge_analysis.py` | Baseline: LLM-as-a-judge self-labelling accuracy against a dataset's own chosen/rejected labels (Table 1). |
| `gepa_judge_optimization.py` | Uses [GEPA](https://github.com/gepa-ai/gepa) to evolve the LLM-judge's instructions against label agreement, as an optimised variant of the self-labelling baseline (Section 5.2). |
| `eval.py` | Standalone checkpoint evaluator: runs the win/tie/loss LLM-as-a-judge evaluation (or ground-truth evaluation for AfriSenti) over saved checkpoints from a `main.py` run, without retraining. |
| `eval_prism.py` | PRISM-specific evaluation (Appendix C.2): scores checkpoints by embedding cosine similarity to a target demographic group's ground-truth completions, instead of LLM-as-a-judge. |

## Reproducing the paper's results

- **Section 3 (preference geometry, Figures 2/3/5/6, Tables 9/10)**: `probe_analysis.py`, `run_oasst2_multilingual.py`.
- **Table 1 (LLM-as-judge self-labelling baseline)**: `llm_judge_analysis.py`, optionally optimised with `gepa_judge_optimization.py`.
- **Table 2 (LoRA classifier-head baseline)**: `lora_classifier_eval.py`.
- **Tables 3–8 (main preference-optimisation results)**: `main.py`, followed by `eval.py` for checkpoint-based evaluation.
- **Table 12 / Appendix C.2 (PRISM downstream evaluation)**: `main.py` with `--po-dataset prism`, followed by `eval_prism.py`.
- **Appendix B (probe calibration / error analysis)**: `probe_gold_eval.py`.

Exact hyperparameters for the reported experiments (LoRA rank/alpha, learning rate, batch size, label smoothing, etc.) are listed in Appendix D of the paper. CLI defaults in this repo are reasonable starting points for local experimentation but were not all tuned to match every reported run — pass the paper's values explicitly to reproduce a specific table.

## Responsible use

This method is designed to lower the cost of adapting models to under-served or low-resource preference populations. Like any label-efficient alignment technique, it can amplify whatever is in the seed annotations at scale — including biased or unrepresentative preferences — and, more broadly, could be misused to cheaply tailor a model's outputs to a narrow target audience. We recommend auditing seed annotations, using representative sampling when collecting them, and combining probe-based labelling with standard safety classifiers. See Appendix E of the paper for a fuller discussion, and Section 6.1 for the method's known limitations (in particular, reduced reliability below 1B parameters).

## Citation

A camera-ready citation will be added once available. In the meantime:

```bibtex
@inproceedings{galatolo2026lowresource,
  title     = {Low-Resource Preference Adaptation of {LLM}s via Activation-Based Label Propagation},
  author    = {Galatolo, Alessio and Beloucif, Meriem},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
