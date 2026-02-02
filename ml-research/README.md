# Clearview ML Research

This directory contains all the machine learning research code for the Clearview ABSA+MSR project.

## Quick Start

### Training the Full EAGLE Model

```bash
python src/models/train_roberta_improved.py --use_synthetic --use_sampler --msr_strength 0.3 --out_dir outputs/my_model
```

###Running Ablations

```bash
.\run_final_ablations_4class.ps1
```

### Evaluation

```bash
python src/evaluation/evaluate_and_log.py --ckpt outputs/gold_msr_4class/best_model.pt --val_path data/splits/val.parquet --text_col text_clean --msr_strength 0.3
```

### XAI Explanations

```bash
$env:PYTHONPATH="."; python src/xai/Explainable.py --ckpt outputs/gold_msr_4class/best_model.pt --text "Your review text" --aspect all --out outputs/xai/report.json
```

## Project Structure

- **src/** - Source code
  - `models/` - Model architectures and training scripts
  - `data_layer/` - Data loading and preprocessing
  - `evaluation/` - Evaluation metrics and scripts
  - `xai/` - Explainability suite (IG, LIME, SHAP)

- **outputs/** - Results and checkpoints
  - `gold_msr_4class/` - Best MSR model
  - `gold_baseline_4class/` - Baseline model
  - `ablations_4class/` - Ablation study results
  - `eval/` - Evaluation reports and thesis tables
  - `xai/` - Explainability reports

- **data/** - Dataset splits
  - `splits/` - train.parquet, val.parquet, test.parquet
  - `augmented/` - Augmented training data

- **notebooks/** - Jupyter notebooks for analysis

- **configs/** - Model configuration files

- **archive/** - Historical runs and deprecated code

## Key Files

- **METHODOLOGY.md** - Complete technical documentation
- **MIGRATION_NOTE.md** - 4-class system migration notes
- **requirements.txt** - Python dependencies
- **run_final_ablations_4class.ps1** - Automated ablation script

## Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Model Architecture

**EAGLE**: Enhanced Aspect-Gated ABSA with Learning-based MSR Evaluation

- RoBERTa-base encoder (125M parameters)
- Aspect-aware attention mechanism
- Cross-aspect interaction module
- 4-class classification per aspect (Negative/Neutral/Positive/None)
- MSR refinement layers for conflict resolution

## Results Summary

| Model | Macro-F1 | MSR Error Reduction |
|:------|:---------|:--------------------|
| Baseline | 0.6953 | 0 |
| EAGLE+MSR | 0.7241 | 50 |

See `METHODOLOGY.md` for complete results and analysis.

## Citation

If you use this code, please cite: [Your thesis details]
