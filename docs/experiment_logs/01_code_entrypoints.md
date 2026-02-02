# 01 Code Entrypoints

| Script Path | Purpose | Key Flags |---
| :--- | :--- | :--- |
| src\models\baseline_distilbert.py | Unknown | project_dir |
| src\models\baseline_roberta.py | Unknown | project_dir |
| src\models\explainability.py | XAI |  |
| src\models\msr_resolver_roberta_focal_loss.py | Unknown | project_dir, head, train_file, val_file, eval_only, checkpoint |
| src\models\msr_resolver_roberta_weighted.py | Unknown | project_dir, head |
| src\models\train_eagle_v3.py | Training | project_dir, head, gcn_dim, gcn_layers, max_len, epochs, patience, batch_size, lr_bert, lr_gcn, seed |
| src\models\train_roberta_hierarchical.py | Training | project_dir, head, max_len, output_attentions, epochs, patience, batch_size, lr_bert, lr_head, seed |
| src\models\train_roberta_improved.py | Training | train_path, train_aug_path, val_path, text_col, use_synthetic, use_sampler, batch_size, max_len, epochs, lr, dropout, msr_strength, conflict_weight, seed, out_dir, patience |
| src\models\train_roberta_improved_backup.py | Training | train_path, train_aug_path, val_path, text_col, use_synthetic, use_sampler, batch_size, max_len, epochs, lr, dropout, msr_strength, conflict_weight, seed, out_dir |
| src\evaluation\compare_reports.py | Unknown | baseline_report, msr_report, out_dir |
| src\evaluation\evaluate_and_log.py | Evaluation | val_path, ckpt, out_dir, text_col, batch_size, max_len, dropout, msr_strength, save_predictions, pred_limit, run_xai, xai_samples, xai_topk |
| src\xai\Explainable.py | XAI | ckpt, text, out, aspect, top_k, run_lime, run_shap |
| src\data_layer\01_validate.py | Preprocessing/Data | project_dir |
| src\data_layer\02_Clean.py | Preprocessing/Data | project_dir |
| src\data_layer\03_split.py | Preprocessing/Data | project_dir |
| src\data_layer\04_create_train_aug.py | Training | project_dir |
| src\data_layer\convert_synthetic_robust.py | Unknown | project_dir |
