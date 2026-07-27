# Server Runbook: RealFT + Uni-MuMER

This repo is prepared for the clean real dataset at:

```text
data/real/real_classroom_dataset_manual_removed
```

Expected split sizes:

```text
real_train.csv        1103 samples
real_validation.csv    259 samples
real_blind_test.csv    274 samples
images/               1636 images
```

## 1. Check Data

```bash
cd <project-root>
find data/real/real_classroom_dataset_manual_removed/images -name "*.png" | wc -l
wc -l data/real/real_classroom_dataset_manual_removed/*.csv
```

## 2. TAMER Environment

```bash
cd <project-root>
source <conda-install-dir>/etc/profile.d/conda.sh
conda activate tamer-rtx3090
```

## 3. Evaluate Direct Transfer on Real Validation

This evaluates:

```text
TAMER Original -> Real Validation
A0             -> Real Validation
A1             -> Real Validation
A2             -> Real Validation
A3             -> Real Validation
```

```bash
nohup bash scripts/server_eval_real_validation_tamer.sh > real_clean_validation_tamer.log 2>&1 &
tail -f real_clean_validation_tamer.log
```

Outputs:

```text
outputs/real_clean_eval/*/validation/metrics.json
```

## 4. Train TAMER RealFT

This trains:

```text
A0 + RealFT
A3 + RealFT
```

```bash
nohup bash scripts/server_train_realft.sh > realft_tamer.log 2>&1 &
tail -f realft_tamer.log
```

Configs:

```text
config/real_ft_a0_control_rtx3090.yaml
config/real_ft_a3_dual_rtx3090.yaml
```

Checkpoint selection metric in code:

```text
val_university_ExpRate
```

Report name:

```text
Real Validation ExpRate
```

## 5. Uni-MuMER Environment

Use a separate environment. Do not install Uni-MuMER dependencies into the TAMER env.

```bash
cd <project-root>
source <conda-install-dir>/etc/profile.d/conda.sh
conda create -n unimumer python=3.11 -y
conda activate unimumer
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Optional pre-download:

```bash
huggingface-cli download phxember/Uni-MuMER-Qwen3.5-2B
```

## 6. Uni-MuMER Zero-Shot on Real Validation

```bash
conda activate unimumer
cd <project-root>
nohup bash scripts/server_eval_unimumer_validation.sh > unimumer_zero_shot_validation.log 2>&1 &
tail -f unimumer_zero_shot_validation.log
```

Output:

```text
outputs/unimumer_zero_shot/validation/metrics.json
```

For a quick smoke test:

```bash
python eval/unimumer_eval_manifest.py --config config/unimumer_zero_shot_real.yaml --split validation --limit 5
```

## 7. Uni-MuMER LoRA

Run this after zero-shot works.

```bash
conda activate unimumer
cd <project-root>
nohup bash scripts/server_train_unimumer_lora.sh > unimumer_lora.log 2>&1 &
tail -f unimumer_lora.log
```

Final adapter:

```text
outputs/unimumer_lora_real/final_adapter
```

## 8. Real Test Rule

Do not run `real_blind_test.csv` until the pipeline is locked:

```text
checkpoint
learning rate
epoch
prompt
preprocess
metric
```

Final test targets:

```text
A0 + RealFT          -> Real Test
A3 + RealFT          -> Real Test
Uni-MuMER Zero-shot  -> Real Test
Uni-MuMER + LoRA     -> Real Test
```
