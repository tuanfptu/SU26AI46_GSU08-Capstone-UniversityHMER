# Outputs

Checkpoints, LoRA adapters, prediction files, metrics, and error-analysis artifacts are intentionally not committed to GitHub.

Important expected local outputs:

```text
outputs/
├── phase1_a3_dual_seed7/checkpoints/
├── real_ft_a3_dual_seed7/checkpoints/
└── unimumer_lora_unsloth_real/
    ├── best_adapter/
    └── final_adapter/
```

For reproducibility, keep large artifacts in a release, cloud storage, Hugging Face model repo, or a private backup archive.
