#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

python eval/unimumer_eval_manifest.py \
  --config config/unimumer_zero_shot_real.yaml \
  --split validation
