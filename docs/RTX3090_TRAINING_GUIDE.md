# RTX 3090 Linux server training

This is the supported path for fine-tuning University-TAMER on one 24 GB RTX
3090. Run every command from the project directory containing
`train_university.py`.

## 1. Server and bundle check

The server should have an NVIDIA driver, at least 32 GB system RAM and about
20 GB free disk space. CUDA Toolkit is not required because Conda installs the
CUDA 11.8 runtime used by PyTorch.

```bash
nvidia-smi
sha256sum University-TAMER-RTX3090.zip
unzip University-TAMER-RTX3090.zip
cd University-TAMER-RTX3090
```

Compare the result with the companion file
`University-TAMER-RTX3090.zip.sha256`. Do not use the older RTX4060 ZIP: its
HME retention cache predates the train-only, leakage-free split.

## 2. Create the environment

Install Miniconda or Mambaforge, then run:

```bash
conda env create -f environment-rtx3090.yml
conda activate tamer-rtx3090
python -m pip install -e . --no-deps
```

Verify that PyTorch sees the rented card:

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('vram_GiB=', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1) if torch.cuda.is_available() else 0)"
```

Expected: `cuda=True`, an RTX 3090 and approximately 24 GiB VRAM.

## 3. Validate the project before paying for training

```bash
python scripts/validate_project.py \
  --data-root data/university \
  --hme-cache data/hme_cache \
  --dictionary data/HME100k/dictionary.txt \
  --checkpoint 'lightning_logs/version_3/checkpoints/epoch=55-step=175503-val_ExpRate=0.6954.ckpt'
```

Do not continue unless the result contains `"status": "PASS"` and reports:

- University splits: 10,000 train / 1,000 validation / 1,000 test;
- HME cache: 20,000 replay / 3,000 retention validation, both from HME train;
- HME replay/validation overlap: zero; official HME test untouched;
- canonical-label leakage: zero.

## 4. Mandatory smoke test

```bash
python train/train_university.py --config config/smoke_test_rtx3090.yaml
```

This exercises 30 training batches and five batches from each validation
loader using the same batch size as the full run. It must finish without CUDA
OOM, NaN loss, missing checkpoint keys or worker errors. The final line reports
peak CUDA memory.

While it runs, monitor the card in another terminal:

```bash
watch -n 1 nvidia-smi
```

## 5. Full fine-tune

Use `tmux` so an SSH disconnect does not kill the job:

```bash
tmux new -s university-tamer
conda activate tamer-rtx3090
cd /path/to/University-TAMER-RTX3090
mkdir -p outputs
python -u train/train_university.py --config config/university_baseline_rtx3090.yaml 2>&1 | tee outputs/rtx3090-console.log
```

Detach with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t university-tamer
```

The full RTX 3090 configuration is:

```text
micro batch:             16
gradient accumulation:   2
effective batch:         32
precision:               FP16
train mixture:           60% University / 40% HME replay
dynamic augmentation:    enabled for University train only
encoder frozen:          first 2 epochs
maximum epochs:          20
early stopping:          4 validation checks without score improvement
checkpoint selection:    maximum val_retention_score
```

The script first calibrates the original version_3 checkpoint on the fixed HME
validation cache. This is expected and is not an extra training run.

## 6. Resume after interruption

```bash
python -u train/train_university.py \
  --config config/university_baseline_rtx3090.yaml \
  --resume outputs/university_baseline_rtx3090/checkpoints/last.ckpt
```

Use `last.ckpt` only for resuming. For final evaluation, use the path printed
as `Best retention-aware checkpoint`.

## 7. Only if batch 16 is out of memory

An RTX 3090 should normally handle the supplied configuration, but rented
machines can have display processes or unusual limits. Change both RTX 3090
YAML files to:

```yaml
trainer:
  accumulate_grad_batches: 4

data:
  train_batch_size: 8
  eval_batch_size: 2
```

The effective training batch remains 32. Rerun the smoke test before the full
run. Do not lower image resolution or beam size in the final run, because that
would change the agreed experiment.

## Outputs to download before deleting the server

Download the whole directory:

```text
outputs/university_baseline_rtx3090/
```

At minimum preserve `checkpoints/`, `logs/`, the console log and the config
files used for the run. The fixed test set must only be evaluated after the
best checkpoint has been selected from validation.

## Evaluate epoch 9 and build the baseline table

This command does not train or change the model:

```bash
python scripts/evaluate_baseline_suite.py
```

It evaluates `outputs/university_baseline_rtx3090/checkpoints/last.ckpt` on
University realistic, University clean, and the full HME100K test. Existing
`metrics.json` files are skipped, so it is safe to run the command again after
an interruption. Pass `--force` only to recompute completed tests.

Epoch-9 reports are saved in:

```text
outputs/final_evaluation/university_tamer_specialist/
```

The comparison of TAMER v3, epoch 5, and epoch 9 is saved in:

```text
outputs/final_evaluation/baseline_comparison.csv
```

To rebuild only the table after all reports exist:

```bash
python scripts/evaluate_baseline_suite.py --table-only
```
