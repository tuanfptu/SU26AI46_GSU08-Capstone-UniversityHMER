# Mini OOD 20 Evaluation Runbook

Mini OOD 20 is a small sanity-check set for testing whether TAMER checkpoints
still recognize simple handwritten calculus/math expressions outside the main
real-classroom split.

It is not used for training, validation, early stopping, or checkpoint
selection.

## Dataset

```text
data/mini_ood_20/
├── mini_ood_20.csv
└── images/
    ├── OOD-01.png
    └── ...
```

The manifest contains 20 samples with labels that are checked against:

```text
data/HME100k/dictionary.txt
```

## Checkpoints evaluated

```text
TAMER v3:
lightning_logs/version_3/checkpoints/epoch=55-step=175503-val_ExpRate=0.6954.ckpt

A0 phase1:
outputs/phase1_a0_control_seed7/checkpoints/epoch=14-val_university_ExpRate=0.4040.ckpt

A3 phase1:
outputs/phase1_a3_dual_seed7/checkpoints/epoch=13-val_university_ExpRate=0.4040.ckpt

A3 RealFT:
outputs/real_ft_a3_dual_seed7/checkpoints/epoch=56-val_university_ExpRate=0.5637.ckpt
```

## Run on Windows

Activate the TAMER training environment first, then:

```powershell
cd <project-root>
.\scripts\eval_mini_ood_20.ps1 -Python python -Gpus 0
```

If CUDA is available in the active environment:

```powershell
.\scripts\eval_mini_ood_20.ps1 -Python python -Gpus 1
```

## Run on Linux/server

Activate the TAMER training environment first, then:

```bash
cd <project-root>
bash scripts/eval_mini_ood_20.sh
```

To force CPU:

```bash
GPUS=0 bash scripts/eval_mini_ood_20.sh
```

## Outputs

```text
outputs/mini_ood_20_eval/
├── tamer_v3/
├── a0_phase1/
├── a3_phase1/
├── a3_realft/
├── summary.csv
└── failures.csv
```

Read `summary.csv` first. Then inspect `failures.csv` to identify which
expressions each checkpoint misses.
