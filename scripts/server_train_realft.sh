#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

python train/train_university.py --config config/real_ft_a0_control_rtx3090.yaml
python train/train_university.py --config config/real_ft_a3_dual_rtx3090.yaml
