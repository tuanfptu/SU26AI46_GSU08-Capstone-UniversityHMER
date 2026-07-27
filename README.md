# University HMER: Handwritten University Calculus Recognition

## Description

This repository contains the clean research code for a final-year project on **Handwritten Mathematical Expression Recognition (HMER)** for university-level calculus expressions captured in real classroom conditions.

The project focuses on handwritten formulas that are more practical and less controlled than standard benchmark images: real paper backgrounds, phone-camera capture, varied lighting, imperfect cropping, and university calculus notation such as fractions, derivatives, logarithms, exponentials, integrals, summations, and nested expressions.

Two complementary model directions are studied:

1. **TAMER-based specialist HMER models**
   - Adapt the original TAMER architecture to university calculus data.
   - Compare adapter variants A0/A1/A2/A3.
   - Fine-tune on a real classroom dataset.
   - Analyze the trade-off between speed, domain adaptation, and generalization.

2. **Uni-MuMER LoRA VLM**
   - Fine-tune `phxember/Uni-MuMER-Qwen3.5-2B` with LoRA on the real classroom dataset.
   - Use the VLM as the main robust model for live demo inputs.

Large datasets, checkpoints, LoRA adapters, logs, and generated outputs are intentionally excluded from this GitHub repository.

## Repository Structure

```text
.
|-- config/              # Selected YAML configs for TAMER, RealFT, Uni-MuMER
|-- assets/fig/          # Figures for README and reports
|-- example_data/        # Placeholder only; real datasets are external
|-- docs/                # Runbooks, requirements, and environment files
|-- eval/                # TAMER evaluation scripts
|-- outputs/             # Placeholder only; checkpoints/adapters are external
|-- preprocess/          # Dataset preprocessing utilities
|-- scripts/             # Training, evaluation, and server scripts
|-- tamer/               # TAMER source code and adapter variants
|-- train/               # TAMER and Uni-MuMER LoRA training entrypoints
|   |-- train.py
|   `-- train_university.py
|-- tests/               # Lightweight tests
```

| Path | Purpose |
|---|---|
| `tamer/` | TAMER architecture, data modules, adapter variants, and training logic. |
| `train/train_university.py` | Main entry point for University12K phase1 and RealFT TAMER experiments. |
| `train/unimumer_lora_train_unsloth.py` | Uni-MuMER LoRA training with Unsloth. |
| `eval/unimumer_eval_manifest.py` | Uni-MuMER zero-shot/LoRA evaluation on manifest splits. |
| `config/` | Reproducible configs for selected final experiments. |
| `scripts/` | Server-oriented run scripts and data preparation utilities. |
| `example_data/` | Dataset placeholder. Actual images/manifests are distributed separately. |
| `outputs/` | Output placeholder. Checkpoints, adapters, metrics, and predictions are stored externally. |

## 📦 Dataset Preparation

Dataset link: [University-HMER-RealClassroom](https://huggingface.co/datasets/tuan3110/University-HMER-RealClassroom)

Expected local dataset layout:

```text
data/
|-- HME100k/
|-- university/
`-- real/
    `-- real_classroom_dataset_manual_removed/
        |-- images/
        |-- real_train.csv
        |-- real_validation.csv
        `-- real_blind_test.csv
```

`University12K` is a curated university-level subset derived from
[MathWriting 2024](https://arxiv.org/abs/2404.10690), a large-scale
handwritten mathematical expression recognition dataset. In this project, it is
used as an intermediate adaptation benchmark before fine-tuning on the real
classroom dataset collected for the final project. A rendered human-written
release is also available on Hugging Face:
[deepcopy/MathWriting-human](https://huggingface.co/datasets/deepcopy/MathWriting-human).

The main real classroom dataset contains:

```text
train:       1103 samples
validation:  259 samples
blind test:  274 samples
total:      1636 images
```

CSV manifests are expected to contain at least:

```text
sample_id,image_path,label
```

Example:

```csv
sample_id,image_path,label
real_0001,images/real_0001.png,\int _ { 0 } ^ { 1 } x ^ { 2 } d x
```

The real classroom dataset is not committed to this repository. It is distributed separately through Hugging Face Datasets:

```text
https://huggingface.co/datasets/tuan3110/University-HMER-RealClassroom
```

## Preprocessing

The preprocessing pipeline is designed to reduce the gap between benchmark-style HMER images and real classroom images.

Main preprocessing steps:

- Normalize labels to the HME100K/TAMER vocabulary when training TAMER.
- Remove or filter invalid/OOV labels for strict TAMER experiments.
- Convert and cache HME100K samples when replay is needed.
- Prepare MathWriting-derived University12K clean splits with canonical-label-disjoint partitioning.
- Prepare real classroom train/validation/blind-test manifests.
- Apply paper/background/lighting/perspective augmentation for TAMER training.

Useful scripts:

```text
scripts/prepare_university_data.py
scripts/prepare_hme_cache.py
scripts/prepare_paper_backgrounds.py
scripts/generate_fixed_splits.py
scripts/preview_augmentation.py
```

Known dataset limitation:

- The real dataset has limited diversity in operator-bound placement.
- Many integral and summation bounds are written diagonally or to the right of the operator.
- Vertically stacked display-style limits are underrepresented.
- This can limit generalization to display-style mathematical layouts.

## 🏃 Inference

### TAMER inference

TAMER checkpoints are stored externally under `outputs/`.

Typical checkpoint paths:

```text
outputs/phase1_a3_dual_seed7/checkpoints/
outputs/real_ft_a3_dual_seed7/checkpoints/
```

Evaluation/inference scripts:

```bash
python eval/evaluate_manifest.py \
  --checkpoint outputs/real_ft_a3_dual_seed7/checkpoints/<checkpoint>.ckpt \
  --dictionary data/HME100k/dictionary.txt \
  --manifest data/real/real_classroom_dataset_manual_removed/real_validation.csv \
  --data-root data/real/real_classroom_dataset_manual_removed \
  --output outputs/eval_tamer_a3_realft
```

### Uni-MuMER inference

Uni-MuMER evaluation supports zero-shot and LoRA-adapter inference:

```bash
python eval/unimumer_eval_manifest.py \
  --config config/unimumer_zero_shot_real.yaml \
  --split validation
```

With LoRA:

```bash
python eval/unimumer_eval_manifest.py \
  --config config/unimumer_lora_real_unsloth.yaml \
  --split validation \
  --lora-path outputs/unimumer_lora_unsloth_real/best_adapter
```

## 📏 Evaluation Metrics

The main metrics are:

| Metric | Description |
|---|---|
| **ExpRate** | Exact expression recognition rate. A prediction is correct only when the full token sequence matches the ground truth. |
| **TER** | Token Error Rate, computed from token-level edit distance. |
| **ValidLaTeX** | Percentage of predictions that can be parsed or rendered as valid LaTeX-like output. |
| **Latency** | Average inference time per image. |
| **Pairwise comparison** | Compares which model is correct or closer on the same samples. |

ExpRate is the primary metric. TER is used to measure how close an incorrect prediction is.

## Model Zoo

Large model files are not committed to GitHub.

| Model | Artifact | Hugging Face | Role |
|---|---|---|---|
| TAMER Original | HME100K pretrained checkpoint | External | Baseline for domain-gap analysis |
| A0 phase1 | `outputs/phase1_a0_control_*` | External | Control adaptation baseline |
| A3 phase1 | `outputs/phase1_a3_dual_seed7/checkpoints/` | External | Dual-adapter TAMER variant |
| A0 RealFT | `outputs/real_ft_a0_control_*` | External | Real-data fine-tuning control |
| A3 RealFT | `outputs/real_ft_a3_dual_seed7/checkpoints/` | [University-HMER-TAMER-A3-RealFT](https://huggingface.co/tuan3110/University-HMER-TAMER-A3-RealFT) | Fast specialist HMER model |
| Uni-MuMER zero-shot | `phxember/Uni-MuMER-Qwen3.5-2B` | [Base model](https://huggingface.co/phxember/Uni-MuMER-Qwen3.5-2B) | VLM zero-shot baseline |
| Uni-MuMER LoRA | `outputs/unimumer_lora_unsloth_real/best_adapter/` | [University-HMER-UniMuMER-LoRA](https://huggingface.co/tuan3110/University-HMER-UniMuMER-LoRA) | Main robust demo model |

Recommended external storage:

- Hugging Face Dataset repo for real classroom data.
- Hugging Face Model repo, GitHub Release, or private cloud storage for checkpoints/adapters.

## Benchmark Results

### Validation

| Model | ExpRate | TER | Latency |
|---|---:|---:|---:|
| TAMER Original | 1.93% | 30.88% | 0.323 s/img |
| A0 phase1 | 5.02% | 18.39% | - |
| A1 phase1 | 3.47% | 18.66% | - |
| A2 phase1 | 3.47% | 17.40% | - |
| A3 phase1 | 5.02% | 16.88% | - |
| A0 RealFT | 53.28% | 5.80% | 0.293 s/img |
| A3 RealFT | 56.37% | 5.45% | 0.299 s/img |
| Uni-MuMER zero-shot | 14.67% | 13.50% | ~2.09 s/img |
| Uni-MuMER LoRA | 64.48% | 4.62% | ~2.60 s/img |

### Blind Test

| Model | ExpRate | TER | Latency |
|---|---:|---:|---:|
| TAMER Original | 4.38% | 22.34% | 0.314 s/img |
| A0 RealFT | 69.34% | 3.05% | 0.301 s/img |
| A3 RealFT | 71.17% | 2.92% | 0.306 s/img |
| Uni-MuMER zero-shot | 23.72% | 7.18% | 1.92 s/img |
| Uni-MuMER LoRA | 74.82% | 3.38% | 2.55 s/img |

### Pairwise A3 RealFT vs Uni-MuMER LoRA on Blind Test

| Category | Count |
|---|---:|
| both_correct | 167 |
| unimumer_lora_only_correct | 38 |
| a3_realft_only_correct | 28 |
| unimumer_lora_closer | 13 |
| a3_realft_closer | 18 |
| same_wrong_distance | 10 |

Summary:

- TAMER-A3 RealFT is fast and strong on the collected classroom distribution.
- Uni-MuMER LoRA is more robust and is selected as the main demo model.
- The two models expose a practical trade-off between speed and generalization.

## Training

This repository keeps a single `requirements.txt` for readability. For full experiments, using separate TAMER and Uni-MuMER environments is still recommended because their CUDA/VLM dependencies can be heavy.

### TAMER environment

```bash
pip install -r requirements.txt
pip install -e .
```

Train selected TAMER configs:

```bash
python train/train_university.py --config config/phase1_a3_dual_rtx3090.yaml
python train/train_university.py --config config/real_ft_a3_dual_rtx3090.yaml
```

Important TAMER configs:

```text
config/hme100k.yaml
config/university_baseline_rtx3090.yaml
config/phase1_a0_control_rtx3090.yaml
config/phase1_a1_encoder_rtx3090.yaml
config/phase1_a2_decoder_rtx3090.yaml
config/phase1_a3_dual_rtx3090.yaml
config/real_ft_a0_control_rtx3090.yaml
config/real_ft_a3_dual_rtx3090.yaml
```

### Uni-MuMER environment

Use a separate environment from TAMER when running full Uni-MuMER LoRA training.

```bash
pip install -r requirements.txt
```

Train Uni-MuMER LoRA with Unsloth:

```bash
python train/unimumer_lora_train_unsloth.py \
  --config config/unimumer_lora_real_unsloth.yaml
```

Evaluate Uni-MuMER LoRA:

```bash
python eval/unimumer_eval_manifest.py \
  --config config/unimumer_lora_real_unsloth.yaml \
  --split validation \
  --lora-path outputs/unimumer_lora_unsloth_real/best_adapter
```

## ✅ TODO

- [ ] Publish the real classroom dataset after privacy and license checks.
- [ ] Upload TAMER checkpoints and Uni-MuMER LoRA adapters to external model storage.
- [ ] Add a small public sample dataset for smoke testing.
- [ ] Add a reproducible demo notebook.
- [ ] Improve robustness to vertically stacked integral/summation bounds.
- [ ] Expand the real dataset with more writers, layouts, and camera conditions.
- [ ] Add richer error-analysis visualizations.
- [ ] Package the Android demo and backend deployment instructions in a separate demo repository.

## 🙏 Acknowledgements

Thanks to the following projects:

- [CoMER](https://github.com/Green-Wood/CoMER)
- [PosFormer](https://github.com/SJTU-DeepVisionLab/PosFormer)
- [TDv2](https://github.com/yqingli123/TDv2)
- [TAMER](https://github.com/qingzhenduyu/TAMER)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [MathNet](https://github.com/felix-schmitt/MathNet)
- [Uni-MuMER](https://github.com/BFlameSwift/Uni-MuMER)
- [Unsloth](https://github.com/unslothai/unsloth)
