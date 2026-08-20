*** BRIDGING THE MODALITY GAP IN MATHEMATICAL
HANDWRITING RECOGNITION USING QWEN3-VL
> Registered capstone title — SU26AI46_GSU08

Research on real-world university-calculus HMER, beginning with a Qwen3-VL-based pipeline and subsequently extending to TAMER domain adaptation and Uni-MuMER Qwen3.5-2B LoRA.

[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/tuan3110/University-HMER-RealClassroom)
[![TAMER-A3](https://img.shields.io/badge/Model-TAMER--A3-blue)](https://huggingface.co/tuan3110/University-HMER-TAMER-A3-RealFT)
[![Uni--MuMER LoRA](https://img.shields.io/badge/Model-Uni--MuMER%20LoRA-purple)](https://huggingface.co/tuan3110/University-HMER-UniMuMER-LoRA)
[![Demo](https://img.shields.io/badge/Demo-Android-green)](https://github.com/Will36237/university-handwritten-math-recognition)

Research code and reproducibility materials for the FPT University capstone project **SU26AI46_GSU08**.

> This public repository contains source code and selected configurations. Large datasets, checkpoints, adapters, predictions, and training logs are stored on Hugging Face or in the university submission package.

## Overview

This project studies handwritten mathematical expression recognition (HMER) for university calculus captured under real classroom conditions. The registered title reflects the project's initial [`Qwen3-VL-HMER`](https://github.com/tuanfptu/Qwen3-VL-HMER) direction. After comparative backbone analysis, the final VLM branch adopted [`phxember/Uni-MuMER-Qwen3.5-2B`](https://huggingface.co/phxember/Uni-MuMER-Qwen3.5-2B). All Uni-MuMER results in this repository therefore refer explicitly to **Uni-MuMER Qwen3.5-2B**, not to the earlier Qwen3-VL prototype.

The study connects three data domains:

```text
HME100K-pretrained TAMER
        ↓
University12K intermediate adaptation
        ↓
RealCalculus-1636 classroom adaptation
```

Two complementary model families are investigated:

- **TAMER-A3 RealFT:** a fast specialist model for the collected classroom distribution.
- **Uni-MuMER LoRA:** a parameter-efficient vision-language model selected for the live application because it is more robust to unconstrained input.

The goal is not to claim one universally superior model, but to quantify the trade-off between **same-domain accuracy, source retention, inference speed, and practical generalization**.

The model-development path is:

```text
Qwen3-VL-HMER initial direction
        ↓
specialist TAMER domain-gap and adapter study
        ↓
Uni-MuMER Qwen3.5-2B backbone selection
        ↓
prompt selection on Real Validation
        ↓
LoRA adaptation and Android deployment
```

## Problem / Motivation

Benchmark HMER images are usually tightly cropped and visually controlled. Classroom photographs introduce paper texture, shadows, perspective distortion, imperfect cropping, long expressions, and writer-dependent two-dimensional layouts.

This creates three research gaps:

1. **Benchmark-to-classroom domain gap.** The HME100K-pretrained TAMER baseline reaches only **1.93% ExpRate** on Real Validation.
2. **Adapter placement in specialist HMER.** There is limited evidence about whether visual, structural, or dual adapters best support TAMER adaptation.
3. **Accuracy-efficiency trade-off.** A specialist decoder is fast but vocabulary-bound; a VLM is more flexible but substantially slower.

The project asks:

- How much can an intermediate calculus domain reduce the benchmark-to-real gap?
- Where should bottleneck adapters be inserted in TAMER?
- How much does real-data fine-tuning improve a frozen classroom Blind Test?
- What is lost through catastrophic forgetting?
- When should a fast specialist be preferred over a LoRA-adapted VLM?

## Key Contributions

1. **University12K:** a MathWriting 2024-derived intermediate calculus domain with canonical-label-disjoint splitting.
2. **RealCalculus-1636:** a privacy-reviewed classroom dataset collected from approximately 180 FPT University HCMC undergraduates.
3. **A0-A3 adapter ablation:** controlled encoder, decoder, and dual-adapter variants for TAMER.
4. **Two-stage specialist adaptation:** HME100K → University12K → RealCalculus-1636, including source-retention evaluation.
5. **Uni-MuMER LoRA adaptation:** only **5,455,872 trainable parameters**, approximately **0.246%** of the base model.
6. **Controlled evaluation:** fixed Validation/Blind splits, prompt selection without Blind Test access, pairwise comparison, category/severity analysis, and Mini-OOD diagnostic evaluation.
7. **Deployment analysis:** an empirical account of the speed versus generalization trade-off between TAMER-A3 RealFT and Uni-MuMER LoRA.

## Method

### TAMER adaptation

TAMER combines a DenseNet visual encoder, image/word positional encoding, a Transformer decoder with coverage attention, and a tree-aware module. The project adds a gated residual bottleneck adapter:

```text
A(x) = x + sigmoid(alpha) × W_up(
           Dropout(GELU(W_down(LayerNorm(x))))
       )
```

The gate initializes the adapter near an identity mapping. The bottleneck limits parameter growth, while dropout regularizes domain adaptation.

| Variant | Encoder adapter | Decoder adapter | Experimental role |
|---|:---:|:---:|---|
| **A0** | No | No | Matched no-adapter control |
| **A1** | Yes | No | Visual-domain adaptation |
| **A2** | No | Yes | Language/structure adaptation |
| **A3** | Yes | Yes | Joint visual and structural adaptation |

**Phase 1** initializes from TAMER v3, adapts to University12K, and uses 40% HME100K replay. **RealFT** initializes A0/A3 from their corresponding phase-1 checkpoints and fine-tunes on RealCalculus-1636 with dynamic augmentation and no replay.

### Uni-MuMER LoRA

#### Why Uni-MuMER Qwen3.5-2B?

The final backbone was selected from the comparative results reported with Uni-MuMER rather than merely because it was newer. The Qwen3.5-2B-based **“This Model”** configuration achieved the highest mean ExpRate (**73.09%**) across eight reported HMER test sets and the best result on five of them. Other variants remained stronger on individual datasets, so the claim is **best average trade-off**, not universal superiority.

| Backbone/configuration | Average ExpRate | Selection relevance |
|---|---:|---|
| Uni-MuMER-3B | 72.19% | Strong specialist VLM baseline |
| **Uni-MuMER Qwen3.5-2B (“This Model”)** | **73.09%** | Highest reported average; selected backbone |
| Qwen3.5-4B | 71.60% | Larger model without a higher average |
| Qwen3-VL-2B | 72.49% | Initial project family and competitive 2B baseline |
| Qwen3-VL-4B | 72.11% | Larger Qwen3-VL comparison |

At 2B scale, the selected backbone also offered a practical accuracy–compute balance for LoRA adaptation on the available 24 GB RTX 3090. The VLM branch adapts [`phxember/Uni-MuMER-Qwen3.5-2B`](https://huggingface.co/phxember/Uni-MuMER-Qwen3.5-2B) using LoRA:

- rank `8`, alpha `16`;
- attention targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`;
- MLP targets: `gate_proj`, `up_proj`, `down_proj`;
- frozen vision tower;
- BF16 and Unsloth gradient checkpointing;
- greedy generation with a fixed prompt.

LoRA changes the language-side attention and MLP projections while preserving the frozen base weights. Uni-MuMER LoRA is used in the live demo; TAMER-A3 remains a specialist research result.

## Dataset

| Dataset | Purpose | Size / split |
|---|---|---:|
| **HME100K** | Source pretraining, phase-1 replay, retention evaluation | Upstream benchmark |
| **University12K** | Intermediate university-calculus adaptation | MathWriting-derived subset |
| **RealCalculus-1636** | Real classroom adaptation and evaluation | 1,103 train / 259 validation / 274 blind |
| **Mini-OOD-20** | Post-selection diagnostic only | 20 images |

RealCalculus-1636 is available at [University-HMER-RealClassroom](https://huggingface.co/datasets/tuan3110/University-HMER-RealClassroom). Personal identifiers are not included in the released metadata.

Preprocessing includes:

- label normalization to the HME100K/TAMER vocabulary;
- strict filtering of invalid or out-of-vocabulary labels for TAMER;
- canonical-label-disjoint University12K splits;
- fixed Real train/validation/blind manifests;
- paper, lighting, perspective, and background augmentation.

The 274-image Blind Test is frozen and is never used for prompt selection, early stopping, or checkpoint selection.

## Experiments & Baselines

### Baselines

| Baseline | Purpose |
|---|---|
| TAMER Original | Measures the HME100K-to-classroom domain gap |
| A0 phase 1 | Controls for adaptation without adapters |
| A1 / A2 / A3 phase 1 | Isolates adapter placement |
| A0 RealFT | Matched real-data control for A3 RealFT |
| Uni-MuMER zero-shot P1 | Measures LoRA gain with the same base model and prompt |

### Training configuration

| Stage | Optimizer / LR | Effective batch | Stopping and selection |
|---|---|---:|---|
| TAMER phase 1 | AdamW + ReduceLROnPlateau, `5e-5` | 32 | max 20 epochs; patience 4; University Validation ExpRate |
| TAMER RealFT | AdamW + ReduceLROnPlateau, `1e-5` | 16 | max 100 epochs; patience 6; Real Validation ExpRate |
| Uni-MuMER LoRA | fused AdamW + cosine schedule, `3e-5` | 8 | max 20 epochs; patience 3; Validation TER |

TAMER uses weight decay `1e-4`, gradient clipping `5.0`, mixed precision, and encoder warm-up. LoRA uses weight decay `0.01`, 5% warm-up, BF16, and gradient accumulation of 8.

### Prompt selection

Three zero-shot prompts were evaluated on the **259-image Validation split only**:

| ID | System prompt | User prompt |
|---|---|---|
| **P1** | `You are a helpful assistant.` | `Convert the mathematical formula in this image to LaTeX format.` |
| **P2** | `You are a mathematical OCR system specialized in handwritten formulas.` | `Recognize the mathematical expression in this image and return its LaTeX representation.` |
| **P3** | `You transcribe handwritten mathematical expressions into LaTeX. Do not explain your answer.` | `Return only the LaTeX expression shown in the image, without Markdown delimiters or additional text.` |

| Prompt | ExpRate | TER | ValidLaTeX | Latency |
|---|---:|---:|---:|---:|
| **P1** | 14.29% | **10.27%** | **99.61%** | **1.667 s/img** |
| P2 | **15.44%** | 13.41% | 99.23% | 1.729 s/img |
| P3 | 15.06% | 13.47% | 99.23% | 1.713 s/img |

P2 yields three more exact matches than P1, but P1 has materially lower TER, higher LaTeX validity, and lower latency. P1 is therefore fixed before the matched zero-shot/LoRA comparison.

### Metrics

- **Exact Match / Expression Recognition Rate (ExpRate):** proportion of predictions whose normalized token sequence exactly matches the reference.
- **Token Error Rate (TER):** corpus-level token edit distance divided by the total number of reference tokens; lower is better.
- **Token Accuracy:** reported as `1 − TER` at corpus level to make token correctness easier to interpret.
- **Valid LaTeX Rate:** proportion of outputs accepted by the project’s LaTeX-validity checker.
- **Category ExpRate:** Exact Match separated by expression category to expose category-specific failure modes.
- **Length-bucket ExpRate:** Exact Match grouped by normalized reference length to measure degradation on longer expressions.
- **Latency:** average end-to-end inference time per image in the stated hardware and decoding environment.
- **95% Wilson interval:** uncertainty interval for binomial proportions such as ExpRate and Valid LaTeX Rate. Wilson intervals are preferred to normal approximations for small or extreme proportions.

Category and length-bucket results are analysis metrics rather than substitutes for the fixed-split aggregate results. Exact Match is intentionally strict and does not treat mathematically equivalent but token-different LaTeX strings as equal.

## Results

### Main benchmark

| Model | Validation ExpRate | Validation TER | Blind ExpRate | Blind TER | Reported latency (blind) |
|---|---:|---:|---:|---:|---:|
| TAMER Original | 1.93% | 30.88% | 4.38% | 22.34% | 0.314 s/img |
| A0 phase 1 | 5.02% | 18.39% | — | — | — |
| A1 phase 1 | 3.47% | 18.66% | — | — | — |
| A2 phase 1 | 3.47% | 17.40% | — | — | — |
| A3 phase 1 | 5.02% | **16.88%** | — | — | — |
| A0 RealFT | 53.28% | 5.80% | 69.34% | 3.05% | 0.301 s/img |
| A3 RealFT | 56.37% | 5.45% | 71.17% | **2.92%** | **0.306 s/img** |
| Uni-MuMER zero-shot P1 | 14.29% | 10.27% | 23.36% | 7.26% | 1.649 s/img |
| **Uni-MuMER LoRA P1** | **64.48%** | **4.62%** | **74.82%** | 3.38% | 2.550 s/img |

### Blind-Test ExpRate uncertainty

The Blind Test contains 274 fixed samples. The intervals below are 95% Wilson score intervals computed from the exact correct-count totals.

| Model | Correct / 274 | Blind ExpRate | 95% Wilson interval |
|---|---:|---:|---:|
| TAMER Original | 12 | 4.38% | [2.52%, 7.50%] |
| A0 RealFT | 190 | 69.34% | [63.65%, 74.50%] |
| A3 RealFT | 195 | 71.17% | [65.54%, 76.21%] |
| Uni-MuMER zero-shot P1 | 64 | 23.36% | [18.74%, 28.71%] |
| **Uni-MuMER LoRA P1** | **205** | **74.82%** | **[69.36%, 79.59%]** |

Because the A3 RealFT and Uni-MuMER LoRA intervals overlap, the observed ExpRate difference alone is not presented as proof of statistical superiority. Model selection also considers TER, validity, pairwise outcomes, latency, and behavior under less-controlled inputs.

### Pairwise Blind-Test comparison

| A3 RealFT vs Uni-MuMER LoRA | Count |
|---|---:|
| Both correct | 167 |
| Uni-MuMER LoRA only correct | 38 |
| A3 RealFT only correct | 28 |
| Uni-MuMER LoRA closer while both wrong | 13 |
| A3 RealFT closer while both wrong | 18 |
| Same wrong distance | 10 |

### Source retention

| Model | HME100K full-test ExpRate | TER |
|---|---:|---:|
| TAMER Original | 69.52% | 3.82% |
| A3 phase 1 | 62.99% | 5.68% |

Using the unrounded logged values, University12K adaptation reduces source ExpRate by **6.54 percentage points**, showing that intermediate adaptation improves the target domain but does not eliminate catastrophic forgetting.

## Analysis / Ablation

### Adapter placement

A3 ties A0 on phase-1 ExpRate but obtains the lowest phase-1 TER. After RealFT, A3 exceeds the matched A0 control by:

- **+3.09 ExpRate points** on Validation;
- **+1.83 ExpRate points** on Blind Test;
- lower Blind TER: **2.92% vs 3.05%**.

This supports dual adapter placement, while the modest margin also shows that most same-domain gain comes from RealFT rather than adapters alone.

### Specialist versus VLM

A3 RealFT is approximately **8.3× faster** than Uni-MuMER LoRA on Blind Test (`2.550 / 0.306`). It is strong on the collected distribution and has the lowest Blind TER. However, its fixed dictionary and limited training-layout diversity restrict unconstrained use.

Uni-MuMER LoRA improves the matched zero-shot baseline by **50.19 ExpRate points on Validation** and **51.46 points on Blind Test**. It obtains the highest ExpRate and 100% ValidLaTeX, supporting its selection for live inference despite higher latency.

### Split interpretation

Blind Test scores exceed Validation scores because the category distributions differ. Validation contains more `log`, `ln`, and nested exponential expressions; Blind Test contains only `mixed_nested` samples and no `e^nested` category. The Blind Test is unseen but remains within the same collection protocol.

### Mini-OOD diagnostic

Uni-MuMER LoRA P1 recognizes **20/20** Mini-OOD samples with **0% TER** and **100% ValidLaTeX**. This set is a post-selection diagnostic only; it is too small to support a general OOD claim.

## Demo

The Android application and FastAPI deployment backend are maintained separately:

**[University HMER Demo Application](https://github.com/Will36237/university-handwritten-math-recognition)**

The workflow is:

```text
Capture / select image
→ manually crop the formula
→ validate the image
→ run Uni-MuMER LoRA
→ display and render LaTeX
```

The live application exposes Uni-MuMER LoRA only. TAMER-A3 remains a research specialist rather than a claim of unconstrained general recognition.

## Installation & Usage

### Install

```bash
pip install -r requirements.txt
pip install -e .
```

Separate TAMER and Uni-MuMER environments are recommended because their CUDA and VLM dependencies differ.

### Train

```bash
# TAMER A3 phase 1
python train/train_university.py --config config/phase1_a3_dual_rtx3090.yaml

# TAMER A3 RealFT
python train/train_university.py --config config/real_ft_a3_dual_rtx3090.yaml

# Uni-MuMER LoRA with Unsloth
python train/unimumer_lora_train_unsloth.py \
  --config config/unimumer_lora_real_unsloth.yaml
```

### Evaluate

```bash
# TAMER
python eval/evaluate_manifest.py \
  --checkpoint outputs/real_ft_a3_dual_seed7/checkpoints/<checkpoint>.ckpt \
  --dictionary data/HME100k/dictionary.txt \
  --manifest data/real/real_classroom_dataset_manual_removed/real_validation.csv \
  --data-root data/real/real_classroom_dataset_manual_removed \
  --output outputs/eval_tamer_a3_realft

# Uni-MuMER LoRA
python eval/unimumer_eval_manifest.py \
  --config config/unimumer_lora_real_unsloth.yaml \
  --split validation \
  --lora-path outputs/unimumer_lora_unsloth_real/best_adapter
```

Model artifacts:

- [TAMER-A3 RealFT](https://huggingface.co/tuan3110/University-HMER-TAMER-A3-RealFT)
- [Uni-MuMER LoRA](https://huggingface.co/tuan3110/University-HMER-UniMuMER-LoRA)
- [RealCalculus-1636 dataset](https://huggingface.co/datasets/tuan3110/University-HMER-RealClassroom)

## Repository Structure

```text
.
|-- config/          # Selected experiment configurations
|-- docs/            # Environment and run documentation
|-- eval/            # TAMER and Uni-MuMER evaluation
|-- example_data/    # Placeholder; full datasets are external
|-- outputs/         # Placeholder; large outputs are external
|-- preprocess/      # Dataset preparation and split utilities
|-- scripts/         # Training/evaluation support scripts
|-- tamer/           # TAMER architecture and adapters
|-- train/           # TAMER and Uni-MuMER training entry points
`-- tests/           # Lightweight regression tests
```

## Limitations / Future Work

### Limitations

- RealCalculus-1636 contains only 1,636 images; writer, device, background, and layout diversity remain limited.
- Integral and summation bounds are frequently placed diagonally or to the right. Vertically stacked display-style bounds are underrepresented.
- Blind Test follows the same collection process as training data and is a same-domain evaluation.
- TAMER uses a fixed HME100K dictionary and cannot reliably handle unseen notation.
- ExpRate treats mathematically equivalent but token-different LaTeX strings as incorrect.
- Validation and Blind Test have different category distributions.
- Mini-OOD-20 is not a statistically representative OOD benchmark.
- The demo rejects invalid files and poor crops, but does not provide a calibrated formula/non-formula classifier or reliable abstention mechanism.

### Future work

- collect more writers, devices, paper types, and camera conditions;
- balance operator-bound layouts, especially vertically stacked integrals and summations;
- add vocabulary expansion or open-vocabulary decoding for specialist HMER;
- evaluate on a larger, independently collected external test set;
- study replay and partial unfreezing to reduce RealFT forgetting;
- add formula relevance classification, confidence calibration, and abstention;
- evaluate mathematical equivalence in addition to exact token matching.

## Citation

If this repository, dataset, or released models support your work, please cite:

```bibtex
@misc{tuan2026universityhmer,
  title        = {Bridging the Modality Gap in Mathematical Handwriting Recognition Using Qwen3-VL},
  author       = {Ha Manh Tuan and Vo Minh Nhat and Lam Gia Thai},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/tuanfptu/SU26AI46_GSU08-Capstone-UniversityHMER}
}
```

Please also cite the relevant upstream works and datasets used in your experiment:

- [TAMER](https://github.com/qingzhenduyu/TAMER)
- [Uni-MuMER](https://github.com/BFlameSwift/Uni-MuMER)
- [MathWriting 2024](https://arxiv.org/abs/2404.10690)
- [Unsloth](https://github.com/unslothai/unsloth)

## Acknowledgements

This work builds on TAMER, Uni-MuMER, MathWriting, Unsloth, CoMER, PosFormer, TDv2, LLaMA-Factory, and MathNet.

## License

See [LICENSE](LICENSE). Datasets, upstream checkpoints, and third-party artifacts remain subject to their respective licenses and terms.
