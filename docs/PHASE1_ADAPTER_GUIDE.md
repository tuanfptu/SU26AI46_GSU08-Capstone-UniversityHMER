# Phase 1: controlled TAMER adapter screening on RTX 3090

This phase compares four runs initialized from the same released TAMER v3
checkpoint. The data, seed, replay ratio, optimizer, scheduler, early stopping,
and checkpoint selection are identical. Only adapter placement changes.

| Run | Encoder adapter | Decoder adapter |
|---|---:|---:|
| A0 control | no | no |
| A1 encoder | yes | no |
| A2 decoder | no | yes |
| A3 dual | yes | yes |

All checkpoints are selected by `val_university_ExpRate`. HME validation is
reported but is not used for stopping, scheduling, or rejection.
The fixed HME validation-cache reference (`0.793666660785675`) is reused from
the calibrated historical run; the official full-test reference remains
`0.6952493192993864` and is used only in final reporting.

## Environment

```bash
conda env create -f environment-rtx3090.yml
conda activate tamer-rtx3090
python -m pip install -e . --no-deps
```

Verify CUDA and project inputs before training:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/validate_project.py \
  --data-root data/university \
  --hme-cache data/hme_cache \
  --dictionary data/HME100k/dictionary.txt \
  --checkpoint 'lightning_logs/version_3/checkpoints/epoch=55-step=175503-val_ExpRate=0.6954.ckpt'
```

## Mandatory smoke test

The dual run exercises both adapter paths and the non-strict legacy checkpoint
load. It trains 30 batches and validates five batches from each loader.

```bash
python scripts/run_phase1.py --smoke
```

Do not start full runs unless the smoke test finishes without missing backbone
keys, NaN, or CUDA OOM.

## Full screening

Run all four sequentially inside `nohup`:

```bash
nohup python scripts/run_phase1.py --runs a0 a1 a2 a3 > phase1_train.log 2>&1 &
echo $!
tail -f phase1_train.log
```

For one run only:

```bash
python scripts/run_phase1.py --runs a2
```

The runner stops immediately if any run fails. Existing output directories are
not deleted; inspect them before restarting. Resume an interrupted individual
run with:

```bash
python train/train_university.py \
  --config config/phase1_a2_decoder_rtx3090.yaml \
  --resume outputs/phase1_a2_decoder_seed7/checkpoints/last.ckpt
```

## Select the architecture using validation only

```bash
python scripts/summarize_phase1_validation.py
```

Choose the highest University-validation ExpRate. Use lower validation TER as
a tie-breaker only after inspecting the logged metrics. Do not use University
test or HME100K test to tune adapter placement or hyperparameters.

After the architecture is fixed, evaluate its selected checkpoint on clean,
realistic, and full HME100K test with the existing evaluation scripts. Compare
the winner against both A0 and the historical epoch-9 result:

```text
Historical epoch 9: realistic 36.2%, clean 37.2%, HME100K 65.11%
```

One command evaluates a selected checkpoint on all three test sets (replace the
checkpoint path with the one printed by the validation summary):

```bash
python scripts/evaluate_phase1_checkpoint.py \
  --checkpoint 'outputs/phase1_a2_decoder_seed7/checkpoints/epoch=XX-val_university_ExpRate=X.XXXX.ckpt' \
  --name a2_decoder_seed7
```

Completed test reports are skipped on rerun. Results are written below
`outputs/final_evaluation/phase1/<name>/`.

## Outputs to preserve

Download the complete directories:

```text
outputs/phase1_a0_control_seed7/
outputs/phase1_a1_encoder_seed7/
outputs/phase1_a2_decoder_seed7/
outputs/phase1_a3_dual_seed7/
phase1_train.log
```
