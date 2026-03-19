"""
Evaluation script for trained models
"""

import os
import yaml
import torch
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import create_model
from utils.data_utils import DependencyParser, DependencyParsingDataset, CosmeticReviewDataset, collate_fn_with_dependencies
from utils.metrics import AspectSentimentEvaluator, ErrorAnalyzer, MixedSentimentEvaluator
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded model
        config: Model configuration
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    
    # Create model
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    best_metric = checkpoint.get('best_val_metric', 0.0)
    print(f"Best validation metric: {best_metric:.4f}")
    
    return model, config


def evaluate_model(model, dataloader, config, device, save_dir=None):
    """
    Evaluate model on a dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        config: Configuration dictionary
        device: Device to run evaluation on
        save_dir: Directory to save results (optional)
        
    Returns:
        results: Dictionary of evaluation results
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_aspects = []
    all_texts = []
    all_review_ids = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            aspect_ids = batch['aspect_ids'].to(device)
            labels = batch['labels']
            
            # Prepare edge indices if using GCN
            edge_indices = None
            if config['model'].get('use_dependency_gcn', False):
                edge_indices = [e.to(device) if e is not None else None 
                               for e in batch['edge_indices']]
            
            # Forward pass
            predictions = model(
                input_ids, attention_mask, aspect_ids, edge_indices
            )
            
            # Get predicted classes
            pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
            
            all_predictions.extend(pred_classes)
            all_labels.extend(labels.numpy())
            all_aspects.extend(batch['aspects'])
            all_texts.extend(batch['texts'])
            if 'review_ids' in batch:
                all_review_ids.extend(batch['review_ids'])
            else:
                # Fallback if review_ids not in batch (e.g. older dataloaders)
                all_review_ids.extend([-1] * len(batch['labels']))
    
    # Create evaluator
    evaluator = AspectSentimentEvaluator(config['aspects']['names'])
    
    # Compute metrics for each aspect
    print("\n" + "="*70)
    print("Per-Aspect Results")
    print("="*70)
    
    aspect_metrics = {}
    for aspect in config['aspects']['names']:
        # Filter samples for this aspect
        aspect_mask = np.array([a == aspect for a in all_aspects])
        if aspect_mask.sum() == 0:
            print(f"\n{aspect}: No samples found")
            continue
        
        y_true = np.array(all_labels)[aspect_mask]
        y_pred = np.array(all_predictions)[aspect_mask]
        
        metrics = evaluator.evaluate_aspect(y_true, y_pred, aspect)
        evaluator.print_results(aspect)
        aspect_metrics[aspect] = metrics
    
    # Compute overall metrics
    print("\n" + "="*70)
    print("Overall Results")
    print("="*70)
    
    overall_metrics = evaluator.evaluate_aspect(
        np.array(all_labels),
        np.array(all_predictions),
        'overall'
    )
    evaluator.print_results('overall')
    
    # Compare aspects
    print("\n")
    evaluator.compare_aspects()
    

    
    # Mixed Sentiment Resolution Analysis
    if len(all_review_ids) > 0 and all_review_ids[0] != -1:
        print("\n" + "="*70)
        print("Mixed Sentiment Resolution Analysis")
        print("="*70)
        
        msr_evaluator = MixedSentimentEvaluator(config['aspects']['names'])
        
        # Organize predictions and labels by review_id
        y_true_dict = {}
        y_pred_dict = {}
        
        for i, review_id in enumerate(all_review_ids):
            if review_id not in y_true_dict:
                y_true_dict[review_id] = {}
                y_pred_dict[review_id] = {}
            
            y_true_dict[review_id][all_aspects[i]] = all_labels[i]
            y_pred_dict[review_id][all_aspects[i]] = all_predictions[i]
            
        msr_metrics = msr_evaluator.evaluate_mixed_sentiment_resolution(y_true_dict, y_pred_dict)
        msr_evaluator.print_mixed_sentiment_results(msr_metrics)
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            msr_evaluator.save_mixed_sentiment_analysis(
                msr_metrics, 
                save_dir / 'mixed_sentiment_analysis.json'
            )

    # Error analysis
    print("\n" + "="*70)
    print("Error Analysis")
    print("="*70)
    
    class_names = ['negative', 'neutral', 'positive']
    error_analyzer = ErrorAnalyzer(config['aspects']['names'], class_names)
    
    errors = error_analyzer.analyze_errors(
        all_texts,
        np.array(all_labels),
        np.array(all_predictions),
        all_aspects,
        save_path=save_dir / 'error_analysis.csv' if save_dir else None
    )
    
    # Save results
    if save_dir:
        # Directory already created above if MSR ran, but good to be safe
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        evaluator.save_results(save_dir / 'metrics.json')
        
        # Save confusion matrices
        evaluator.plot_all_confusion_matrices(save_dir)
        
        # Generate LaTeX table
        latex_table = evaluator.generate_latex_table()
        with open(save_dir / 'latex_table.tex', 'w') as f:
            f.write(latex_table)
        print(f"\nLaTeX table saved to {save_dir / 'latex_table.tex'}")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'text': all_texts,
            'aspect': all_aspects,
            'true_label': [class_names[l] for l in all_labels],
            'pred_label': [class_names[p] for p in all_predictions],
            'correct': [l == p for l, p in zip(all_labels, all_predictions)]
        })
        predictions_df.to_csv(save_dir / 'predictions.csv', index=False)
        print(f"Predictions saved to {save_dir / 'predictions.csv'}")
    
    return {
        'overall': overall_metrics,
        'aspects': aspect_metrics,
        'errors': errors
    }


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Multi-Aspect Sentiment Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data CSV file (default: use test set from config)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and config
    model, config = load_model(args.checkpoint, device)
    
    # Create tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config['model']['roberta_model'])
    
    # Initialize dependency parser if needed
    dependency_parser = None
    if config['data'].get('use_dependency_parsing', False):
        print("Initializing dependency parser...")
        dependency_parser = DependencyParser(
            language=config['data'].get('language', 'en')
        )
    
    # Determine data path
    if args.data:
        data_path = args.data
    else:
        data_path = config['data']['test_path']
    
    print(f"Loading data from {data_path}")
    
    # Create dataset
    dataset_class = DependencyParsingDataset if dependency_parser else CosmeticReviewDataset
    dataset = dataset_class(
        data_path=data_path,
        tokenizer=tokenizer,
        config=config,
        aspect_names=config['aspects']['names'],
        dependency_parser=dependency_parser,
        is_train=False
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn_with_dependencies
    )
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        checkpoint_dir = Path(args.checkpoint).parent
        output_dir = checkpoint_dir / 'evaluation_results'
    
    # Evaluate
    results = evaluate_model(model, dataloader, config, device, output_dir)
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Overall Accuracy: {results['overall']['accuracy']:.4f}")
    print(f"  Overall Macro-F1: {results['overall']['macro_f1']:.4f}")
    print(f"  Overall Weighted-F1: {results['overall']['weighted_f1']:.4f}")
    print(f"  Total Errors: {len(results['errors'])}")


if __name__ == "__main__":
    main()
