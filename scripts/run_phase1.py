"""Run the controlled A0/A1/A2/A3 adapter screening experiments."""

import argparse
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    "a0": "config/phase1_a0_control_rtx3090.yaml",
    "a1": "config/phase1_a1_encoder_rtx3090.yaml",
    "a2": "config/phase1_a2_decoder_rtx3090.yaml",
    "a3": "config/phase1_a3_dual_rtx3090.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run the mandatory dual-adapter smoke test.")
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=tuple(CONFIGS),
        default=list(CONFIGS),
        help="Full runs to execute sequentially (default: a0 a1 a2 a3).",
    )
    args = parser.parse_args()

    checkpoint = Path(
        "lightning_logs/version_3/checkpoints/epoch=55-step=175503-val_ExpRate=0.6954.ckpt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError("Missing TAMER v3 checkpoint: {}".format(checkpoint))

    configs = ["config/phase1_adapter_smoke_rtx3090.yaml"] if args.smoke else [CONFIGS[x] for x in args.runs]
    for config in configs:
        print("\n=== Running {} ===".format(config), flush=True)
        subprocess.run([sys.executable, "train/train_university.py", "--config", config], check=True)


if __name__ == "__main__":
    main()
