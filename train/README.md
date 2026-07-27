# Training Entrypoints

This folder contains TAMER training entrypoints.

```text
train.py             # Original TAMER training entry
train_university.py  # Main entry for University12K phase1 and RealFT experiments
```

Typical usage from the repository root:

```bash
python train/train_university.py --config config/phase1_a3_dual_rtx3090.yaml
python train/train_university.py --config config/real_ft_a3_dual_rtx3090.yaml
```
