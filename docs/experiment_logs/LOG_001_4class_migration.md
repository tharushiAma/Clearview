# LOG 001: Migration to 4-Class System (NULL Support)

**Date:** 2026-02-01
**Author:** Antigravity

## Summary

Migrated the ClearView sentiment model from a 3-class system (Negative, Neutral, Positive) to a 4-class system by adding **NULL (Not Mentioned)**. This ensures that missing aspects are explicitly modeled rather than being forced into "Neutral".

## Changes Implemented

### 1. Label Schema

- **Old**: 0=Negative, 1=Neutral, 2=Positive
- **New**: 0=Negative, 1=Neutral, 2=Positive, **3=NULL**
- **Constants**: Centralized in `src/data_layer/_common.py`.

### 2. Model Architecture

- `ImprovedRoBERTaHierarchical` now outputs 4 logits per aspect.
- `ConflictDetector` input dimension updated to match 4 classes.
- **Loss**: Added support for `class_weights` to down-weight the frequent NULL class.

### 3. Data Pipeline

- `RoBERTaDataset` and `EvalDataset` now map `None`, `NaN`, `""`, or explicit `"none"` strings to Class 3 (NULL).

### 4. Evaluation

- Metrics now computed in two modes:
  1. **Overall 4-Class**: Standard Macro F1 over all 4 classes.
  2. **Sentiment-Only**: Macro F1 over {0, 1, 2}, ignoring NULL targets.
- **Advanced Analysis**:
  - **Systemic**: Hamming Loss, Exact Match Ratio.
  - **Class Imbalance**: Balanced Accuracy, G-Mean (geometric mean of recalls).
  - **MSR Reliability**: ROC-AUC, Brier Score, ECE (Calibration).
  - **Statistical Significance**: Wilcoxon Signed-Rank test p-values.
- Added **Presence Detection** metric (Binary F1: Null vs Non-Null).
- Confusion Matrices are now 4x4.

## How to Run

### Training (Baseline)

```bash
python src/models/train_roberta_improved.py \
  --msr_strength 0.0 \
  --out_dir outputs/exp_baseline_4class \
  --null_weight 0.2 \
  --epochs 5
```

### Training (MSR)

```bash
python src/models/train_roberta_improved.py \
  --msr_strength 0.3 \
  --out_dir outputs/exp_msr_4class \
  --null_weight 0.2 \
  --epochs 5
```

### Evaluation

```bash
python src/evaluation/evaluate_and_log.py \
  --ckpt outputs/exp_baseline_4class/best_model.pt \
  --val_path data/splits/val.parquet \
  --out_dir outputs/eval_baseline_4class \
  --msr_strength 0.0
```

## Expected Outputs

- `report.json` will now contain `overall_macro_f1_4class` and `overall_macro_f1_sentiment`.
- `confusion_matrices.csv` will contain flattened 4x4 matrices.

## Conflict Detection Design (Fix #2)

- **Design Choice**: Option B/C (Post-Refinement Conflict).
- **Logic**:
  1. Compute `base_conflict` on initial logits.
  2. Use `base_conflict` as a **gate** for MSR refinement.
  3. Compute `refined_conflict` on the final MSR-adjusted logits.
- **Benefit**: This resolves the "identical probability" issues, allowing the model to explicitly show how MSR *resolves* conflict (probability decreases after refinement).
- **Outputs**: `conflict_score` now returns the **refined** score, while `conflict_base` is available for research logging.
