# Updated Evaluation Module for EAGLE
# Saves outputs in exact same format as baseline/GCN models for fair comparison

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm
import os


def evaluate_eagle_model(model, dataloader, device, aspects, project_dir, model_name="eagle"):
    """
    Evaluate EAGLE model and save outputs in same format as baseline models.
    
    Saves:
    1. {model_name}_metrics.txt - Classification reports
    2. {model_name}_predictions.csv - Per-aspect predictions with confidence
    3. {model_name}_msr_results.csv - MSR resolution results
    4. {model_name}_confusion_matrices.txt - Confusion matrices
    
    Args:
        model: EAGLE model
        dataloader: validation/test dataloader
        device: cuda or cpu
        aspects: list of aspect names
        project_dir: project directory path
        model_name: name prefix for output files (e.g., "eagle", "eagle_augmented")
    
    Returns:
        results dict with per-aspect metrics
        overall_macro_f1: float
    """
    model.eval()
    
    # Collect predictions and labels
    all_preds = {a: [] for a in aspects}
    all_confs = {a: [] for a in aspects}  # Confidence scores
    all_labels = {a: [] for a in aspects}
    all_msr_outputs = []
    all_texts = []
    
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_name.upper()} MODEL")
    print(f"{'='*80}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            adj_matrix = batch["adj_matrix"].to(device)
            aspect_masks = batch["aspect_masks"].to(device)
            positions = batch["positions"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            aspect_logits, msr_output = model(
                input_ids, attention_mask, adj_matrix, aspect_masks, positions
            )
            
            # Collect aspect-level predictions
            for i, aspect in enumerate(aspects):
                # Get probabilities
                probs = F.softmax(aspect_logits[i], dim=-1)
                
                # Get predictions and confidence
                conf, preds = torch.max(probs, dim=-1)
                
                all_preds[aspect].extend(preds.cpu().tolist())
                all_confs[aspect].extend(conf.cpu().tolist())
                all_labels[aspect].extend(labels[:, i].cpu().tolist())
            
            # Collect MSR outputs
            all_msr_outputs.append({
                'overall_sentiment': msr_output['overall_sentiment'].cpu(),
                'conflict_score': msr_output['conflict_score'].cpu(),
                'aspect_importance': msr_output['aspect_importance'].cpu()
            })
    
    # Get texts from dataset
    all_texts = dataloader.dataset.texts
    
    # ========================================================================
    # 1. SAVE METRICS TEXT FILE (same format as baseline)
    # ========================================================================
    
    metrics_path = f"{project_dir}/outputs/reports/{model_name}_metrics.txt"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    results = {}
    macro_f1_scores = []
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"{model_name.upper()} MODEL - EVALUATION RESULTS\n")
        f.write(f"{'='*80}\n\n")
        
        for aspect in aspects:
            # Filter out -100 labels (missing labels)
            valid = [k for k, y in enumerate(all_labels[aspect]) if y != -100]
            
            if not valid:
                f.write(f"\nAspect: {aspect}\n")
                f.write("No valid samples in validation set.\n")
                f.write("-" * 50 + "\n")
                continue
            
            y_true = [all_labels[aspect][k] for k in valid]
            y_pred = [all_preds[aspect][k] for k in valid]
            
            # Classification report
            report = classification_report(
                y_true, y_pred,
                labels=[0, 1, 2],
                target_names=["negative", "neutral", "positive"],
                zero_division=0
            )
            
            # Compute metrics
            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            results[aspect] = {
                'accuracy': acc,
                'macro_f1': macro_f1,
                'per_class_f1': per_class_f1,
                'y_true': y_true,
                'y_pred': y_pred
            }
            
            macro_f1_scores.append(macro_f1)
            
            # Write to file
            f.write(f"\nAspect: {aspect}\n")
            f.write(report)
            f.write("\n")
            f.write("-" * 50 + "\n")
            
            # Also print to console
            print(f"\n{aspect.upper()}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Macro F1: {macro_f1:.4f}")
            print(f"  Per-class F1: Neg={per_class_f1[0]:.3f}, Neu={per_class_f1[1]:.3f}, Pos={per_class_f1[2]:.3f}")
        
        # Overall metrics
        overall_macro_f1 = np.mean(macro_f1_scores)
        f.write(f"\n{'='*80}\n")
        f.write(f"OVERALL MACRO F1 (Average across aspects): {overall_macro_f1:.4f}\n")
        f.write(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print(f"OVERALL MACRO F1: {overall_macro_f1:.4f}")
    print(f"{'='*80}")
    print(f"Metrics saved to {metrics_path}")
    
    # ========================================================================
    # 2. SAVE CONFUSION MATRICES (additional detail)
    # ========================================================================
    
    cm_path = f"{project_dir}/outputs/reports/{model_name}_confusion_matrices.txt"
    
    with open(cm_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"{model_name.upper()} MODEL - CONFUSION MATRICES\n")
        f.write(f"{'='*80}\n\n")
        
        for aspect in aspects:
            if aspect not in results:
                continue
            
            y_true = results[aspect]['y_true']
            y_pred = results[aspect]['y_pred']
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            
            f.write(f"\nAspect: {aspect}\n")
            f.write(f"Confusion Matrix:\n")
            f.write(f"                Predicted\n")
            f.write(f"              Neg  Neu  Pos\n")
            f.write(f"Actual Neg  [{cm[0,0]:4d} {cm[0,1]:4d} {cm[0,2]:4d}]\n")
            f.write(f"       Neu  [{cm[1,0]:4d} {cm[1,1]:4d} {cm[1,2]:4d}]\n")
            f.write(f"       Pos  [{cm[2,0]:4d} {cm[2,1]:4d} {cm[2,2]:4d}]\n")
            f.write("-" * 50 + "\n")
    
    print(f"Confusion matrices saved to {cm_path}")
    
    # ========================================================================
    # 3. SAVE PREDICTIONS CSV (same format as baseline)
    # ========================================================================
    
    inv_map = {0: "negative", 1: "neutral", 2: "positive", -100: "None"}
    
    out_rows = []
    for idx in range(len(all_texts)):
        row = {"text": all_texts[idx]}
        
        # Add per-aspect predictions, confidence, and true labels
        for aspect in aspects:
            row[f"{aspect}_pred"] = inv_map.get(all_preds[aspect][idx], "None")
            row[f"{aspect}_conf"] = round(all_confs[aspect][idx], 4)
            row[f"{aspect}_true"] = inv_map.get(all_labels[aspect][idx], "None")
        
        out_rows.append(row)
    
    pred_csv = f"{project_dir}/outputs/reports/{model_name}_predictions.csv"
    pd.DataFrame(out_rows).to_csv(pred_csv, index=False, encoding="utf-8")
    print(f"Predictions saved to {pred_csv}")
    
    # ========================================================================
    # 4. SAVE MSR RESULTS CSV (includes MSR resolution)
    # ========================================================================
    
    # Apply MSR resolution to each row
    msr_rows = []
    
    # Flatten MSR outputs
    batch_size = all_msr_outputs[0]['overall_sentiment'].size(0)
    num_batches = len(all_msr_outputs)
    
    overall_sentiments = torch.cat([x['overall_sentiment'] for x in all_msr_outputs], dim=0)
    conflict_scores = torch.cat([x['conflict_score'] for x in all_msr_outputs], dim=0)
    aspect_importance = torch.cat([x['aspect_importance'] for x in all_msr_outputs], dim=0)
    
    for idx in range(len(all_texts)):
        row = {"text": all_texts[idx]}
        
        # Add aspect predictions and confidence
        for aspect in aspects:
            row[f"{aspect}_pred"] = inv_map.get(all_preds[aspect][idx], "None")
            row[f"{aspect}_conf"] = round(all_confs[aspect][idx], 4)
            row[f"{aspect}_true"] = inv_map.get(all_labels[aspect][idx], "None")
        
        # Add MSR outputs
        overall_sent_logits = overall_sentiments[idx]
        overall_sent_pred = torch.argmax(overall_sent_logits).item()
        row["msr_resolution"] = inv_map.get(overall_sent_pred, "None")
        
        conflict = conflict_scores[idx].item()
        row["conflict_score"] = round(conflict, 4)
        
        # Add aspect importance weights
        asp_imp = aspect_importance[idx].cpu().numpy()
        for i, aspect in enumerate(aspects):
            row[f"{aspect}_importance"] = round(asp_imp[i], 4)
        
        msr_rows.append(row)
    
    msr_csv = f"{project_dir}/outputs/reports/{model_name}_msr_results.csv"
    pd.DataFrame(msr_rows).to_csv(msr_csv, index=False, encoding="utf-8")
    print(f"MSR results saved to {msr_csv}")
    
    # ========================================================================
    # 5. SAVE COMPARISON SUMMARY (for easy comparison with baselines)
    # ========================================================================
    
    summary_path = f"{project_dir}/outputs/reports/{model_name}_summary.txt"
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"{model_name.upper()} MODEL - PERFORMANCE SUMMARY\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Overall Macro F1: {overall_macro_f1:.4f}\n\n")
        
        f.write("Per-Aspect Performance:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Aspect':<15} {'Accuracy':>10} {'Macro F1':>10} {'Neg F1':>8} {'Neu F1':>8} {'Pos F1':>8}\n")
        f.write("-" * 80 + "\n")
        
        for aspect in aspects:
            if aspect not in results:
                continue
            
            acc = results[aspect]['accuracy']
            macro_f1 = results[aspect]['macro_f1']
            per_class = results[aspect]['per_class_f1']
            
            f.write(f"{aspect:<15} {acc:10.4f} {macro_f1:10.4f} {per_class[0]:8.3f} {per_class[1]:8.3f} {per_class[2]:8.3f}\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"\nAverage:        {np.mean([r['accuracy'] for r in results.values()]):10.4f} {overall_macro_f1:10.4f}\n")
    
    print(f"Summary saved to {summary_path}")
    
    print(f"\nAll evaluation outputs saved to {project_dir}/outputs/reports/")
    
    return results, overall_macro_f1


def compare_models(project_dir, model_names=['roberta', 'roberta_gcn', 'roberta_gcn_fl', 'eagle']):
    """
    Create a comparison table across all models.
    
    Args:
        project_dir: project directory
        model_names: list of model names to compare
    
    Creates:
        model_comparison.txt - side-by-side comparison
    """
    aspects = ['stayingpower', 'texture', 'smell', 'price', 'colour', 'shipping', 'packing']
    
    # Load metrics from each model
    all_metrics = {}
    
    for model_name in model_names:
        metrics_path = f"{project_dir}/outputs/reports/{model_name}_metrics.txt"
        
        if not os.path.exists(metrics_path):
            print(f"Metrics not found for {model_name}, skipping...")
            continue
        
        # Parse metrics file (simplified - you could make this more robust)
        all_metrics[model_name] = {}
        
        with open(metrics_path, 'r') as f:
            content = f.read()
            
            # Extract overall F1 if available
            if "OVERALL MACRO F1" in content:
                for line in content.split('\n'):
                    if "OVERALL MACRO F1" in line:
                        try:
                            f1 = float(line.split(':')[-1].strip())
                            all_metrics[model_name]['overall'] = f1
                        except:
                            pass
    
    # Create comparison table
    comparison_path = f"{project_dir}/outputs/reports/model_comparison.txt"
    
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*100}\n")
        f.write(f"MODEL COMPARISON - OVERALL MACRO F1\n")
        f.write(f"{'='*100}\n\n")
        
        f.write(f"{'Model':<25} {'Overall Macro F1':>20}\n")
        f.write("-" * 100 + "\n")
        
        for model_name in sorted(all_metrics.keys(), key=lambda x: all_metrics[x].get('overall', 0), reverse=True):
            overall_f1 = all_metrics[model_name].get('overall', 0.0)
            f.write(f"{model_name:<25} {overall_f1:20.4f}\n")
        
        f.write("-" * 100 + "\n")
        
        # Find best model
        if all_metrics:
            best_model = max(all_metrics.keys(), key=lambda x: all_metrics[x].get('overall', 0))
            best_f1 = all_metrics[best_model].get('overall', 0)
            
            f.write(f"\nBEST MODEL: {best_model} (F1 = {best_f1:.4f})\n")
    
    print(f"\nModel comparison saved to {comparison_path}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example of how to use the evaluation functions.
    """
    import argparse
    from eagle_implementation import EAGLE
    from transformers import RobertaTokenizerFast
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", default="eagle", help="Name for output files")
    args = parser.parse_args()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = EAGLE(num_aspects=7, num_classes=3)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    aspects = ['stayingpower', 'texture', 'smell', 'price', 'colour', 'shipping', 'packing']
    
    # Load validation data
    from train_eagle import EAGLEDataset, preprocess_and_cache_adjacency
    import pandas as pd
    
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    val_df = pd.read_parquet(f"{args.project_dir}/data/splits/val.parquet")
    
    # Preprocess adjacency matrices
    cache_dir = f"{args.project_dir}/outputs/cache"
    val_adj = preprocess_and_cache_adjacency(
        val_df, tokenizer, 256,
        f"{cache_dir}/val_adj_eagle.pkl"
    )
    
    val_dataset = EAGLEDataset(val_df, tokenizer, val_adj, aspects, 256)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Evaluate
    results, overall_f1 = evaluate_eagle_model(
        model=model,
        dataloader=val_loader,
        device=device,
        aspects=aspects,
        project_dir=args.project_dir,
        model_name=args.model_name
    )
    
    # Compare with other models
    compare_models(args.project_dir)
