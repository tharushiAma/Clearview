# 04 Evaluation Protocol

To ensure a fair comparison across all candidate checkpoints, the following standardized protocol will be used:

## Evaluation Command Template
```powershell
python src/evaluation/evaluate_and_log.py `
    --val_path data/splits/val.parquet `
    --text_col text_clean `
    --ckpt <CHECKPOINT_PATH> `
    --out_dir outputs/eval/_rerun_all/<MODEL_TAG>/ `
    --msr_strength <0.0 or 0.3> `
    --save_predictions
```

## Parameters
- **Validation Set**: `data/splits/val.parquet` (Consistently used for all final evaluations).
- **Text Column**: `text_clean` (Matches the preprocessing pipeline).
- **MSR Strength**:
    - `0.0` for Baseline models (no MSR resolver).
    - `0.3` for MSR-enabled models (Gold standard for MSR evaluation).

## Candidate Checkpoints to Rerun
Based on the audit, the following checkpoints are prioritized for the final comparison:

1. `outputs/exp_a_baseline_fixed_eval/best_model.pt` (Baseline)
2. `outputs/exp_b_msr_fixed_er/best_model.pt` (MSR Full)
3. `outputs/abl_no_synth_msr/best_model.pt` (Ablation: No Synth)
4. `outputs/abl_no_sampler_msr/best_model.pt` (Ablation: No Sampler)

*Note: Baseline versions of ablations will also be included for completeness if found.*
