#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

MANIFEST="data/real/real_classroom_dataset_manual_removed/real_validation.csv"
DATA_ROOT="data/real/real_classroom_dataset_manual_removed"
DICTIONARY="data/HME100k/dictionary.txt"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"

python eval/evaluate_manifest.py \
  --checkpoint lightning_logs/version_3/checkpoints/epoch=55-step=175503-val_ExpRate=0.6954.ckpt \
  --dictionary "${DICTIONARY}" \
  --manifest "${MANIFEST}" \
  --data-root "${DATA_ROOT}" \
  --output outputs/real_clean_eval/tamer_original/validation \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --gpus 1

python eval/evaluate_manifest.py \
  --checkpoint outputs/phase1_a0_control_seed7/checkpoints/epoch=14-val_university_ExpRate=0.4040.ckpt \
  --dictionary "${DICTIONARY}" \
  --manifest "${MANIFEST}" \
  --data-root "${DATA_ROOT}" \
  --output outputs/real_clean_eval/a0_control/validation \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --gpus 1

python eval/evaluate_manifest.py \
  --checkpoint outputs/phase1_a1_encoder_seed7/checkpoints/epoch=16-val_university_ExpRate=0.4110.ckpt \
  --dictionary "${DICTIONARY}" \
  --manifest "${MANIFEST}" \
  --data-root "${DATA_ROOT}" \
  --output outputs/real_clean_eval/a1_encoder/validation \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --gpus 1

python eval/evaluate_manifest.py \
  --checkpoint outputs/phase1_a2_decoder_seed7/checkpoints/epoch=19-val_university_ExpRate=0.4080.ckpt \
  --dictionary "${DICTIONARY}" \
  --manifest "${MANIFEST}" \
  --data-root "${DATA_ROOT}" \
  --output outputs/real_clean_eval/a2_decoder/validation \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --gpus 1

python eval/evaluate_manifest.py \
  --checkpoint outputs/phase1_a3_dual_seed7/checkpoints/epoch=13-val_university_ExpRate=0.4040.ckpt \
  --dictionary "${DICTIONARY}" \
  --manifest "${MANIFEST}" \
  --data-root "${DATA_ROOT}" \
  --output outputs/real_clean_eval/a3_dual/validation \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --gpus 1
