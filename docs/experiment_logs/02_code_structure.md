# 02 Code Structure

| Script Path | Role | CLI Arguments |
| :--- | :--- | :--- |
| src\data_layer\01_validate.py | Data Prep | project_dir |
| src\data_layer\02_Clean.py | Data Prep | project_dir |
| src\data_layer\03_split.py | Data Prep | project_dir |
| src\data_layer\convert_synthetic_robust.py | Data Prep | project_dir |
| src\evaluation\compare_reports.py | Evaluation | out_dir, baseline_report, msr_report |
| src\evaluation\evaluate_and_log.py | Evaluation | max_len, xai_topk, xai_samples, text_col, msr_strength, pred_limit, save_predictions, out_dir, dropout, batch_size, ckpt, val_path, run_xai |
| tools\experiment_helpers\step4_evaluations.py | Evaluation |  |
| src\data_layer\04_create_train_aug.py | Training | project_dir |
| src\models\baseline_distilbert.py | Training | project_dir |
| src\models\baseline_roberta.py | Training | project_dir |
| src\models\explainability.py | Training |  |
| src\models\msr_resolver_roberta_focal_loss.py | Training | eval_only, checkpoint, project_dir, head, train_file, val_file |
| src\models\msr_resolver_roberta_weighted.py | Training | project_dir, head |
| src\models\train_eagle_v3.py | Training | max_len, project_dir, gcn_layers, seed, lr_gcn, patience, gcn_dim, head, batch_size, lr_bert, epochs |
| src\models\train_roberta_hierarchical.py | Training | max_len, output_attentions, project_dir, seed, patience, batch_size, head, lr_bert, epochs, lr_head |
| src\models\train_roberta_improved.py | Training | max_len, text_col, train_aug_path, msr_strength, use_synthetic, lr, seed, out_dir, patience, dropout, batch_size, use_sampler, val_path, epochs, conflict_weight, train_path |
| src\models\train_roberta_improved_backup.py | Training | max_len, text_col, train_aug_path, msr_strength, use_synthetic, lr, seed, out_dir, dropout, batch_size, use_sampler, val_path, epochs, conflict_weight, train_path |
| data\analyze_imbalance.py | Utility |  |
| tools\experiment_helpers\audit_step1_inventory.py | Utility |  |
| tools\experiment_helpers\audit_step2_structure.py | Utility |  |
| tools\experiment_helpers\step1_inventory.py | Utility |  |
| tools\experiment_helpers\step2_entrypoints.py | Utility |  |
| tools\experiment_helpers\step3_checkpoints.py | Utility |  |
| tools\experiment_helpers\step7_compare.py | Utility |  |
| src\xai\Explainable.py | XAI | run_shap, top_k, run_lime, out, text, ckpt, aspect |
