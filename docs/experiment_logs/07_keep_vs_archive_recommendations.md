# 07 Keep vs Archive Recommendations

Based on the audit and the standardized evaluation rerun, here are the recommendations for preparing a "clean" branch of the project.

## 1. Clean Mainline Set (KEEP)

### Core Checkpoints
- `outputs/exp_a_baseline_fixed_eval/best_model.pt`: The gold standard baseline.
- `outputs/exp_b_msr_fixed_er/best_model.pt`: The gold standard MSR implementation (Eagle v3).

### Essential Source Code
- `src/models/roberta_hierarchical_improved.py`: The definitive model architecture.
- `src/evaluation/evaluate_and_log.py`: The official evaluation pipeline.
- `src/data_layer/`: All scripts in this folder (`01_validate.py` to `04_create_train_aug.py`) are essential for the data pipeline.
- `src/models/train_eagle_v3.py`: The primary training script.

### Essential Data
- `data/splits/*.parquet`: The official train/val/test splits.
- `data/augmented/*.json`: The synthetic data used for augmentation.

---

## 2. Archive (Move to subfolder or keep in experiment branch)

### Legacy/Duplicate Models
- `src/models/roberta_improved_backup.py`: Superseded by `roberta_hierarchical_improved.py`.
- `src/models/msr_resolver_roberta_weighted.py`: Superseded by focal loss version.
- `src/models/train_roberta_improved.py`: Older training script.

### Evaluation Outputs
- `outputs/reports/`: Most individual text files here are legacy artifacts from early runs. Keep only the ones corresponding to the mainline set.
- `outputs/eval/`: Previous evaluation folders (other than `baseline_fixed` and `msr`) can be archived as they are surpassed by the `_rerun_all` results.

---

## 3. Delete (Safe to remove in the future)

### Temporary/Helper Scripts
- `gen_tree.py`, `file_list.txt`, `project_tree.txt`, `full_tree.txt`: Temporary files created during the audit.
- `tools/experiment_helpers/`: Can be removed once the audit is fully finalized and documented.

### Redundant Experiments
- `outputs/eval/msr/`: Duplicate folder of `msr_full`.
- `outputs/eval/baseline_fixed/`: Duplicate of rerun version.

---

## Rationale
The goal is to provide a repository that is easy to audit for a thesis or publication. By keeping ONLY the best baseline and the best MSR model, and the code required to recreate them, we ensure clarity and reproducibility.
