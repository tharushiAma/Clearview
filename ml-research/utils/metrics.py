"""
Evaluation metrics for multi-aspect sentiment analysis
"""

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, matthews_corrcoef, classification_report
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AspectSentimentEvaluator:
    """
    Comprehensive evaluation for imbalanced multi-aspect sentiment
    """
    def __init__(self, aspect_names):
        """
        Args:
            aspect_names: List of aspect names
        """
        self.aspect_names = aspect_names
        self.results = {}
    
    def evaluate_aspect(self, y_true, y_pred, aspect_name):
        """
        Compute comprehensive metrics for a single aspect
        
        Args:
            y_true: True labels (numpy array)
            y_pred: Predicted labels (numpy array)
            aspect_name: Name of the aspect
            
        Returns:
            Dictionary of metrics
        """
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics (for 3 classes: negative=0, neutral=1, positive=2)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
        )
        
        # Macro metrics (important for imbalanced data)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0, labels=[0, 1, 2]
        )
        
        # Weighted metrics
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0, labels=[0, 1, 2]
        )
        
        # Matthews Correlation Coefficient (good for imbalanced data)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        self.results[aspect_name] = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'mcc': mcc,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'support': support,
            'confusion_matrix': cm
        }
        
        return self.results[aspect_name]
    
    def print_results(self, aspect_name):
        """
        Pretty print results for an aspect
        """
        if aspect_name not in self.results:
            print(f"No results found for {aspect_name}")
            return
        
        results = self.results[aspect_name]
        
        print(f"\n{'='*70}")
        print(f"Results for {aspect_name.upper()}")
        print(f"{'='*70}")
        print(f"Accuracy:      {results['accuracy']:.4f}")
        print(f"Macro F1:      {results['macro_f1']:.4f}")
        print(f"Weighted F1:   {results['weighted_f1']:.4f}")
        print(f"MCC:           {results['mcc']:.4f}")
        
        print(f"\n{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support'}")
        print(f"{'-'*70}")
        
        class_names = ['Negative', 'Neutral', 'Positive']
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<12} {results['per_class_precision'][i]:<12.4f} "
                  f"{results['per_class_recall'][i]:<12.4f} "
                  f"{results['per_class_f1'][i]:<12.4f} {int(results['support'][i])}")
        
        print(f"\nConfusion Matrix:")
        print("              Pred Neg  Pred Neu  Pred Pos")
        print(f"True Neg      {results['confusion_matrix'][0][0]:8d}  "
              f"{results['confusion_matrix'][0][1]:8d}  "
              f"{results['confusion_matrix'][0][2]:8d}")
        print(f"True Neu      {results['confusion_matrix'][1][0]:8d}  "
              f"{results['confusion_matrix'][1][1]:8d}  "
              f"{results['confusion_matrix'][1][2]:8d}")
        print(f"True Pos      {results['confusion_matrix'][2][0]:8d}  "
              f"{results['confusion_matrix'][2][1]:8d}  "
              f"{results['confusion_matrix'][2][2]:8d}")
    
    def plot_confusion_matrix(self, aspect_name, save_path=None):
        """
        Plot confusion matrix for an aspect
        """
        if aspect_name not in self.results:
            print(f"No results found for {aspect_name}")
            return
        
        cm = self.results[aspect_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Neutral', 'Positive'],
                   yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title(f'Confusion Matrix - {aspect_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_confusion_matrices(self, save_dir=None):
        """
        Plot confusion matrices for all aspects
        """
        if not self.results:
            print("No results to plot")
            return
        
        n_aspects = len(self.results)
        n_cols = 3
        n_rows = (n_aspects + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_aspects > 1 else [axes]
        
        for idx, (aspect_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Neg', 'Neu', 'Pos'],
                       yticklabels=['Neg', 'Neu', 'Pos'],
                       ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{aspect_name}\nF1={results["macro_f1"]:.3f}')
            axes[idx].set_ylabel('True')
            axes[idx].set_xlabel('Predicted')
        
        # Hide empty subplots
        for idx in range(n_aspects, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'all_confusion_matrices.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"All confusion matrices saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_aspects(self):
        """
        Compare performance across aspects
        """
        if not self.results:
            print("No results to compare")
            return
        
        print(f"\n{'='*90}")
        print(f"Performance Comparison Across Aspects")
        print(f"{'='*90}")
        print(f"{'Aspect':<15} {'Accuracy':<12} {'Macro-F1':<12} {'Weighted-F1':<12} {'MCC':<12}")
        print(f"{'-'*90}")
        
        for aspect_name in sorted(self.results.keys()):
            results = self.results[aspect_name]
            print(f"{aspect_name:<15} {results['accuracy']:<12.4f} "
                  f"{results['macro_f1']:<12.4f} {results['weighted_f1']:<12.4f} "
                  f"{results['mcc']:<12.4f}")
        
        # Plot comparison
        aspects = list(self.results.keys())
        metrics_to_plot = ['accuracy', 'macro_f1', 'weighted_f1', 'mcc']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [self.results[asp][metric] for asp in aspects]
            axes[idx].bar(aspects, values, color='steelblue', alpha=0.7)
            axes[idx].set_title(metric.replace('_', ' ').title())
            axes[idx].set_ylabel('Score')
            axes[idx].set_ylim([0, 1])
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', 
                             ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def generate_latex_table(self):
        """
        Generate LaTeX table for thesis
        """
        if not self.results:
            return ""
        
        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\begin{tabular}{lcccc}\n"
        latex += "\\hline\n"
        latex += "Aspect & Accuracy & Macro F1 & Weighted F1 & MCC \\\\\n"
        latex += "\\hline\n"
        
        for aspect in sorted(self.results.keys()):
            results = self.results[aspect]
            latex += f"{aspect} & {results['accuracy']:.4f} & "
            latex += f"{results['macro_f1']:.4f} & "
            latex += f"{results['weighted_f1']:.4f} & "
            latex += f"{results['mcc']:.4f} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\caption{Multi-aspect sentiment analysis results}\n"
        latex += "\\label{tab:results}\n"
        latex += "\\end{table}"
        
        return latex
    
    def save_results(self, save_path):
        """
        Save results to file
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for aspect, metrics in self.results.items():
            serializable_results[aspect] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_results[aspect][key] = value.tolist()
                else:
                    serializable_results[aspect][key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {save_path}")


class ErrorAnalyzer:
    """
    Analyze prediction errors for improvement insights
    """
    def __init__(self, aspect_names, class_names):
        self.aspect_names = aspect_names
        self.class_names = class_names  # ['negative', 'neutral', 'positive']
    
    def analyze_errors(self, texts, y_true, y_pred, aspects, save_path=None):
        """
        Analyze misclassified samples
        
        Args:
            texts: List of input texts
            y_true: True labels
            y_pred: Predicted labels
            aspects: Aspect for each sample
            save_path: Path to save error analysis
        """
        # Find misclassified samples
        errors = []
        for i in range(len(texts)):
            if y_true[i] != y_pred[i]:
                errors.append({
                    'text': texts[i],
                    'aspect': aspects[i],
                    'true_label': self.class_names[y_true[i]],
                    'pred_label': self.class_names[y_pred[i]],
                    'error_type': f"{self.class_names[y_true[i]]}->{self.class_names[y_pred[i]]}"
                })
        
        print(f"\nTotal errors: {len(errors)} / {len(texts)} ({len(errors)/len(texts)*100:.2f}%)")
        
        # Error distribution by aspect
        print("\nError distribution by aspect:")
        from collections import Counter
        aspect_errors = Counter([e['aspect'] for e in errors])
        for aspect, count in sorted(aspect_errors.items()):
            total_aspect = sum(1 for a in aspects if a == aspect)
            print(f"  {aspect}: {count} / {total_aspect} ({count/total_aspect*100:.2f}%)")
        
        # Error type distribution
        print("\nError type distribution:")
        error_types = Counter([e['error_type'] for e in errors])
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count} ({count/len(errors)*100:.2f}%)")
        
        # Save detailed errors
        if save_path:
            import pandas as pd
            df = pd.DataFrame(errors)
            df.to_csv(save_path, index=False)
            print(f"\nDetailed error analysis saved to {save_path}")
        
        return errors


class MixedSentimentEvaluator:
    """
    Evaluator for mixed sentiment resolution - a core research contribution
    
    Mixed sentiment = reviews with conflicting sentiments across different aspects
    (e.g., "great color but terrible smell")
    """
    def __init__(self, aspect_names):
        """
        Args:
            aspect_names: List of aspect names
        """
        self.aspect_names = aspect_names
        self.class_names = ['negative', 'neutral', 'positive']
    
    def identify_mixed_sentiment_reviews(self, reviews_data):
        """
        Identify reviews with mixed sentiments (conflicting sentiments across aspects)
        
        Args:
            reviews_data: List of dicts with structure:
                {
                    'review_id': str,
                    'text': str,
                    'aspects': {aspect_name: sentiment_label, ...}
                }
                where sentiment_label is 0=negative, 1=neutral, 2=positive
        
        Returns:
            mixed_reviews: List of review IDs with mixed sentiments
            stats: Dictionary with mixed sentiment statistics
        """
        mixed_reviews = []
        stats = {
            'total_reviews': len(reviews_data),
            'mixed_sentiment_reviews': 0,
            'single_aspect_reviews': 0,
            'multi_aspect_reviews': 0,
            'conflict_types': {
                'positive_negative': 0,  # Has both positive and negative
                'positive_neutral_negative': 0,  # Has all three
                'neutral_with_extremes': 0  # Has neutral with pos or neg
            }
        }
        
        for review in reviews_data:
            # Get active aspects (non-None sentiments)
            active_aspects = {asp: sent for asp, sent in review['aspects'].items() 
                            if sent is not None}
            
            if len(active_aspects) == 0:
                continue
            elif len(active_aspects) == 1:
                stats['single_aspect_reviews'] += 1
                continue
            else:
                stats['multi_aspect_reviews'] += 1
            
            # Get unique sentiment values
            sentiments = set(active_aspects.values())
            
            # Check for mixed sentiment
            has_positive = 2 in sentiments
            has_neutral = 1 in sentiments
            has_negative = 0 in sentiments
            
            is_mixed = False
            
            if has_positive and has_negative:
                is_mixed = True
                if has_neutral:
                    stats['conflict_types']['positive_neutral_negative'] += 1
                else:
                    stats['conflict_types']['positive_negative'] += 1
            elif has_neutral and (has_positive or has_negative):
                # Neutral with any extreme can also be considered mixed
                is_mixed = True
                stats['conflict_types']['neutral_with_extremes'] += 1
            
            if is_mixed:
                mixed_reviews.append(review['review_id'])
                stats['mixed_sentiment_reviews'] += 1
        
        # Calculate percentages
        if stats['multi_aspect_reviews'] > 0:
            stats['mixed_percentage_of_multi'] = (
                stats['mixed_sentiment_reviews'] / stats['multi_aspect_reviews'] * 100
            )
        else:
            stats['mixed_percentage_of_multi'] = 0.0
        
        stats['mixed_percentage_of_total'] = (
            stats['mixed_sentiment_reviews'] / stats['total_reviews'] * 100
        )
        
        return mixed_reviews, stats
    
    def evaluate_mixed_sentiment_resolution(self, y_true_dict, y_pred_dict):
        """
        Evaluate model performance on mixed sentiment reviews
        
        Args:
            y_true_dict: Dict mapping review_id -> {aspect: true_label}
            y_pred_dict: Dict mapping review_id -> {aspect: pred_label}
        
        Returns:
            metrics: Dictionary with mixed sentiment metrics
        """
        # Convert to reviews_data format for mixed review identification
        reviews_data_true = []
        for review_id in y_true_dict:
            reviews_data_true.append({
                'review_id': review_id,
                'text': '',  # Not needed for identification
                'aspects': y_true_dict[review_id]
            })
        
        # Identify mixed sentiment reviews
        mixed_review_ids, mixed_stats = self.identify_mixed_sentiment_reviews(reviews_data_true)
        
        if len(mixed_review_ids) == 0:
            print("Warning: No mixed sentiment reviews found in dataset")
            return {
                'mixed_review_count': 0,
                'mixed_detection_rate': 0.0,
                'mixed_accuracy': 0.0,
                'stats': mixed_stats
            }
        
        # Evaluate predictions on mixed reviews
        correct_mixed = 0
        total_mixed_aspects = 0
        correct_aspects_in_mixed = 0
        
        for review_id in mixed_review_ids:
            if review_id not in y_pred_dict:
                continue
            
            true_aspects = y_true_dict[review_id]
            pred_aspects = y_pred_dict[review_id]
            
            # Check if model correctly predicted ALL aspects for this review
            all_correct = True
            for aspect in true_aspects:
                if aspect in pred_aspects:
                    total_mixed_aspects += 1
                    if true_aspects[aspect] == pred_aspects[aspect]:
                        correct_aspects_in_mixed += 1
                    else:
                        all_correct = False
                else:
                    all_correct = False
            
            if all_correct:
                correct_mixed += 1
        
        # Calculate metrics
        metrics = {
            'mixed_review_count': len(mixed_review_ids),
            'mixed_detection_rate': mixed_stats['mixed_percentage_of_multi'],
            'mixed_review_accuracy': correct_mixed / len(mixed_review_ids) * 100,
            'mixed_aspect_accuracy': correct_aspects_in_mixed / total_mixed_aspects * 100 if total_mixed_aspects > 0 else 0,
            'stats': mixed_stats,
            'total_mixed_aspects': total_mixed_aspects,
            'correct_mixed_aspects': correct_aspects_in_mixed
        }
        
        return metrics
    
    def print_mixed_sentiment_results(self, metrics):
        """
        Print mixed sentiment evaluation results
        """
        print(f"\n{'='*70}")
        print("MIXED SENTIMENT RESOLUTION EVALUATION")
        print(f"{'='*70}")
        
        stats = metrics['stats']
        print(f"\nDataset Statistics:")
        print(f"  Total reviews: {stats['total_reviews']}")
        print(f"  Multi-aspect reviews: {stats['multi_aspect_reviews']}")
        print(f"  Mixed sentiment reviews: {stats['mixed_sentiment_reviews']}")
        print(f"  Mixed % (of multi-aspect): {stats['mixed_percentage_of_multi']:.2f}%")
        print(f"  Mixed % (of total): {stats['mixed_percentage_of_total']:.2f}%")
        
        print(f"\nMixed Sentiment Types:")
        print(f"  Positive + Negative: {stats['conflict_types']['positive_negative']}")
        print(f"  All three sentiments: {stats['conflict_types']['positive_neutral_negative']}")
        print(f"  Neutral with extremes: {stats['conflict_types']['neutral_with_extremes']}")
        
        if metrics['mixed_review_count'] > 0:
            print(f"\nModel Performance on Mixed Sentiment Reviews:")
            print(f"  Total mixed reviews evaluated: {metrics['mixed_review_count']}")
            print(f"  Review-level accuracy: {metrics['mixed_review_accuracy']:.2f}%")
            print(f"    (Reviews where ALL aspects predicted correctly)")
            print(f"  Aspect-level accuracy: {metrics['mixed_aspect_accuracy']:.2f}%")
            print(f"    ({metrics['correct_mixed_aspects']}/{metrics['total_mixed_aspects']} aspects correct)")
        
        print(f"{'='*70}\n")
    
    def save_mixed_sentiment_analysis(self, metrics, save_path):
        """
        Save mixed sentiment analysis to file
        """
        import json
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Mixed sentiment analysis saved to {save_path}")


if __name__ == "__main__":
    # Test the evaluator
    print("Testing AspectSentimentEvaluator...")
    
    # Create dummy data
    y_true = np.array([0, 0, 1, 1, 2, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 2, 1, 0, 1, 2])
    
    aspect_names = ['smell', 'texture', 'price']
    evaluator = AspectSentimentEvaluator(aspect_names)
    
    # Evaluate
    results = evaluator.evaluate_aspect(y_true, y_pred, 'test_aspect')
    
    # Print results
    evaluator.print_results('test_aspect')
    
    # Generate LaTeX table
    latex = evaluator.generate_latex_table()
    print(f"\nLaTeX table:\n{latex}")
    
    print("\nEvaluator test passed!")
