# Migration Note

**Critical Update: 4-Class System (NULL Support)**

## Overview

The codebase has been updated to support a 4th class: **NULL** (ID 3), representing aspects that are not mentioned in the text.

- **Phase 6: 2x2 Ablation Matrix**
  - We are comparing 4 configurations: Base, Sampler-Only, Synthetic-Only, and Full-Eagle (Both).
  - For each, we test Baseline (MSR 0.0) vs MSR (0.3).
  - This identifies which component drives the 3.1% Sentiment F1 gain.

### Manual Verification

1. Run `check_conflict.py` to see the logic fix in action.
2. Run `run_final_ablations_4class.ps1` to reproduce the full matrix.

## Checkpoint Compatibility
> [!WARNING]
> **Checkpoints trained prior to this change are INCOMPATIBLE.**

Old checkpoints expect:
-   `num_classes=3`
-   Loss function without weights support
-   Conflict detector with smaller input dimension

**Action Required:** You must retrain your models using the new 4-class system.

## Data Schema
No changes to input parquet files are strictly required, *provided* that "not mentioned" aspects are represented as `None`, `NaN`, `""`, or `"none"`. The data loader will automatically map these to Class 3.

## Usage Changes
-   **Training**: Use the optional `--null_weight` flag (default 0.2) to balance the loss if NULL is dominant.
-   **Evaluation**: Look for `overall_macro_f1_sentiment` for comparable metrics to previous 3-class baselines.
