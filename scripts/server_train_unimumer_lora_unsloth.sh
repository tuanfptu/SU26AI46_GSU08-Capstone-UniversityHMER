#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

PYTHONPATH=. python train/unimumer_lora_train_unsloth.py \
  --config config/unimumer_lora_real_unsloth.yaml
