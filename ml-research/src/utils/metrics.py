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
from tqdm import tqdm


class AspectSentimentEvaluator:
    """
    Comprehensive evaluation for imbalanced multi-aspect sentiment
    """
    def __init__(self, aspect_names):
        self.aspect_names = aspect_names
        self.results = {}
        print(f"[Evaluator] AspectSentimentEvaluator ready — tracking {len(aspect_names)} aspects: {aspect_names}")

    def evaluate_aspect(self, y_true, y_pred, aspect_name):
        """
        Compute comprehensive metrics for a single aspect.
        Primary metric is Macro-F1 (weights all classes equally — critical for imbalanced data).
        """
        n = len(y_true)
        print(f"\n[Evaluator] Evaluating aspect: '{aspect_name}'  ({n} samples)")

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0, labels=[0, 1, 2]
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0, labels=[0, 1, 2]
        )
        mcc = matthews_corrcoef(y_true, y_pred)
        cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        self.results[aspect_name] = {
            'accuracy'          : accuracy,
            'macro_precision'   : macro_precision,
            'macro_recall'      : macro_recall,
            'macro_f1'          : macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall'   : weighted_recall,
            'weighted_f1'       : weighted_f1,
            'mcc'               : mcc,
            'per_class_precision': precision,
            'per_class_recall'  : recall,
            'per_class_f1'      : f1,
            'support'           : support,
            'confusion_matrix'  : cm,
        }

        print(f"  Accuracy: {accuracy:.4f}  |  Macro-F1: {macro_f1:.4f}  |  "
              f"Weighted-F1: {weighted_f1:.4f}  |  MCC: {mcc:.4f}")
        print(f"  Per-class F1 — neg: {f1[0]:.4f}  neu: {f1[1]:.4f}  pos: {f1[2]:.4f}")
        print(f"  Support      — neg: {int(support[0])}  neu: {int(support[1])}  pos: {int(support[2])}")

        return self.results[aspect_name]

    def print_results(self, aspect_name):
        """Full formatted result table for one aspect."""
        if aspect_name not in self.results:
            print(f"[Evaluator] No results for '{aspect_name}' — run evaluate_aspect() first")
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
        for i, name in enumerate(['Negative', 'Neutral', 'Positive']):
            print(f"{name:<12} {results['per_class_precision'][i]:<12.4f} "
                  f"{results['per_class_recall'][i]:<12.4f} "
                  f"{results['per_class_f1'][i]:<12.4f} {int(results['support'][i])}")

        print(f"\nConfusion Matrix:")
        print("              Pred Neg  Pred Neu  Pred Pos")
        for row_name, row_idx in [("True Neg", 0), ("True Neu", 1), ("True Pos", 2)]:
            print(f"{row_name}      "
                  f"{results['confusion_matrix'][row_idx][0]:8d}  "
                  f"{results['confusion_matrix'][row_idx][1]:8d}  "
                  f"{results['confusion_matrix'][row_idx][2]:8d}")

    def plot_confusion_matrix(self, aspect_name, save_path=None):
        if aspect_name not in self.results:
            print(f"[Evaluator] No results for '{aspect_name}'")
            return
        cm = self.results[aspect_name]['confusion_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title(f'Confusion Matrix — {aspect_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Evaluator] Confusion matrix saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_all_confusion_matrices(self, save_dir=None):
        if not self.results:
            print("[Evaluator] No results yet — run evaluate_aspect() first")
            return
        n_aspects = len(self.results)
        n_cols    = 3
        n_rows    = (n_aspects + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_aspects > 1 else [axes]

        print(f"[Evaluator] Plotting {n_aspects} confusion matrices...")
        for idx, (aspect_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Neg', 'Neu', 'Pos'],
                        yticklabels=['Neg', 'Neu', 'Pos'],
                        ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{aspect_name}\nF1={results["macro_f1"]:.3f}')
            axes[idx].set_ylabel('True')
            axes[idx].set_xlabel('Predicted')

        for idx in range(n_aspects, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'all_confusion_matrices.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Evaluator] All confusion matrices saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def compare_aspects(self):
        if not self.results:
            print("[Evaluator] No results yet")
            return

        print(f"\n{'='*90}")
        print("Performance Comparison Across Aspects")
        print(f"{'='*90}")
        print(f"{'Aspect':<15} {'Accuracy':<12} {'Macro-F1':<12} {'Weighted-F1':<12} {'MCC':<12}")
        print(f"{'-'*90}")

        for aspect_name in sorted(self.results.keys()):
            r = self.results[aspect_name]
            print(f"{aspect_name:<15} {r['accuracy']:<12.4f} "
                  f"{r['macro_f1']:<12.4f} {r['weighted_f1']:<12.4f} {r['mcc']:<12.4f}")

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
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.show()

    def generate_latex_table(self):
        if not self.results:
            return ""
        latex  = "\\begin{table}[h]\n\\centering\n"
        latex += "\\begin{tabular}{lcccc}\n\\hline\n"
        latex += "Aspect & Accuracy & Macro F1 & Weighted F1 & MCC \\\\\n\\hline\n"
        for aspect in sorted(self.results.keys()):
            r = self.results[aspect]
            latex += (f"{aspect} & {r['accuracy']:.4f} & {r['macro_f1']:.4f} & "
                      f"{r['weighted_f1']:.4f} & {r['mcc']:.4f} \\\\\n")
        latex += "\\hline\n\\end{tabular}\n"
        latex += "\\caption{Multi-aspect sentiment analysis results}\n"
        latex += "\\label{tab:results}\n\\end{table}"
        return latex

    def save_results(self, save_path):
        import json
        serializable = {}
        for aspect, metrics in self.results.items():
            serializable[aspect] = {
                k: v.tolist() if isinstance(v, np.ndarray)
                   else float(v) if isinstance(v, (np.float32, np.float64))
                   else v
                for k, v in metrics.items()
            }
        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"[Evaluator] Results saved to {save_path}")


class ErrorAnalyzer:
    """Analyze prediction errors for improvement insights"""
    def __init__(self, aspect_names, class_names):
        self.aspect_names = aspect_names
        self.class_names  = class_names

    def analyze_errors(self, texts, y_true, y_pred, aspects, save_path=None):
        from collections import Counter
        print(f"\n[ErrorAnalyzer] Analyzing {len(texts)} predictions...")

        errors = [
            {
                'text'      : texts[i],
                'aspect'    : aspects[i],
                'true_label': self.class_names[y_true[i]],
                'pred_label': self.class_names[y_pred[i]],
                'error_type': f"{self.class_names[y_true[i]]}->{self.class_names[y_pred[i]]}",
            }
            for i in range(len(texts)) if y_true[i] != y_pred[i]
        ]

        print(f"[ErrorAnalyzer] Total errors: {len(errors)} / {len(texts)} "
              f"({len(errors)/len(texts)*100:.2f}%)")

        aspect_errors = Counter([e['aspect'] for e in errors])
        print(f"\n[ErrorAnalyzer] Error rate by aspect:")
        for aspect in sorted(aspect_errors):
            total = sum(1 for a in aspects if a == aspect)
            print(f"  {aspect:<16}: {aspect_errors[aspect]:>4} / {total} "
                  f"({aspect_errors[aspect]/total*100:.2f}%)")

        error_types = Counter([e['error_type'] for e in errors])
        print(f"\n[ErrorAnalyzer] Error type distribution:")
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {etype:<25}: {count:>4}  ({count/len(errors)*100:.2f}%)")

        if save_path:
            import pandas as pd
            pd.DataFrame(errors).to_csv(save_path, index=False)
            print(f"\n[ErrorAnalyzer] Detailed errors saved to {save_path}")

        return errors


class MixedSentimentEvaluator:
    """
    Evaluator for mixed sentiment resolution.
    Mixed sentiment = reviews expressing conflicting opinions across different aspects
    (e.g., positive colour but negative smell).
    """
    def __init__(self, aspect_names):
        self.aspect_names = aspect_names
        self.class_names  = ['negative', 'neutral', 'positive']
        print(f"[MSREvaluator] Ready — tracking {len(aspect_names)} aspects")

    def identify_mixed_sentiment_reviews(self, reviews_data):
        print(f"\n[MSREvaluator] Scanning {len(reviews_data)} reviews for mixed sentiment...")
        mixed_reviews: list = []
        stats = {
            'total_reviews'        : len(reviews_data),
            'mixed_sentiment_reviews': 0,
            'single_aspect_reviews': 0,
            'multi_aspect_reviews' : 0,
            'conflict_types'       : {
                'positive_negative'           : 0,
                'positive_neutral_negative'   : 0,
                'neutral_with_extremes'       : 0,
            },
        }

        for review in tqdm(reviews_data, desc="  Scanning reviews"):
            active = {asp: sent for asp, sent in review['aspects'].items() if sent is not None}
            if not active:
                continue
            if len(active) == 1:
                stats['single_aspect_reviews'] += 1  # type: ignore
                continue
            stats['multi_aspect_reviews'] += 1  # type: ignore

            sentiments   = set(active.values())
            has_positive = 2 in sentiments
            has_neutral  = 1 in sentiments
            has_negative = 0 in sentiments
            is_mixed     = False

            if has_positive and has_negative:
                is_mixed = True
                if has_neutral:
                    stats['conflict_types']['positive_neutral_negative'] += 1  # type: ignore
                else:
                    stats['conflict_types']['positive_negative'] += 1  # type: ignore
            elif has_neutral and (has_positive or has_negative):
                is_mixed = True
                stats['conflict_types']['neutral_with_extremes'] += 1  # type: ignore

            if is_mixed:
                mixed_reviews.append(review['review_id'])
                stats['mixed_sentiment_reviews'] += 1  # type: ignore

        multi = stats['multi_aspect_reviews']
        stats['mixed_percentage_of_multi'] = (  # type: ignore[assignment]
            stats['mixed_sentiment_reviews'] / multi * 100 if multi > 0 else 0.0  # type: ignore[operator]
        )
        stats['mixed_percentage_of_total'] = (  # type: ignore[assignment]
            stats['mixed_sentiment_reviews'] / stats['total_reviews'] * 100  # type: ignore[operator]
        )

        print(f"[MSREvaluator] Found {stats['mixed_sentiment_reviews']} mixed reviews "
              f"({stats['mixed_percentage_of_multi']:.1f}% of multi-aspect, "
              f"{stats['mixed_percentage_of_total']:.1f}% of total)")
        return mixed_reviews, stats

    def evaluate_mixed_sentiment_resolution(self, y_true_dict, y_pred_dict):
        print(f"\n[MSREvaluator] Evaluating mixed sentiment resolution on "
              f"{len(y_true_dict)} reviews...")

        reviews_data_true = [
            {'review_id': rid, 'text': '', 'aspects': y_true_dict[rid]}
            for rid in y_true_dict
        ]
        mixed_review_ids, mixed_stats = self.identify_mixed_sentiment_reviews(reviews_data_true)

        if not mixed_review_ids:
            print("[MSREvaluator] WARNING: No mixed sentiment reviews found in dataset")
            return {
                'mixed_review_count'   : 0,
                'mixed_detection_rate' : 0.0,
                'mixed_review_accuracy': 0.0,
                'mixed_aspect_accuracy': 0.0,
                'stats'                : mixed_stats,
            }

        print(f"[MSREvaluator] Scoring predictions on {len(mixed_review_ids)} mixed reviews...")
        correct_reviews: int = 0
        total_aspects: int   = 0
        correct_aspects: int = 0

        for review_id in tqdm(mixed_review_ids, desc="  Scoring mixed reviews"):
            if review_id not in y_pred_dict:
                continue
            true_asp = y_true_dict[review_id]
            pred_asp = y_pred_dict[review_id]
            all_correct = True

            for aspect in true_asp:
                if aspect in pred_asp:
                    total_aspects += 1  # type: ignore
                    if true_asp[aspect] == pred_asp[aspect]:
                        correct_aspects += 1  # type: ignore
                    else:
                        all_correct = False
                else:
                    all_correct = False

            if all_correct:
                correct_reviews += 1  # type: ignore

        review_acc = correct_reviews / len(mixed_review_ids) * 100
        aspect_acc = correct_aspects / total_aspects * 100 if total_aspects > 0 else 0.0

        print(f"[MSREvaluator] Review-level accuracy (all aspects correct): {review_acc:.2f}%")
        print(f"[MSREvaluator] Aspect-level accuracy ({correct_aspects}/{total_aspects}): "
              f"{aspect_acc:.2f}%")

        return {
            'mixed_review_count'   : len(mixed_review_ids),
            'mixed_detection_rate' : mixed_stats['mixed_percentage_of_multi'],  # type: ignore[typeddict-item]
            'mixed_review_accuracy': review_acc,
            'mixed_aspect_accuracy': aspect_acc,
            'stats'                : mixed_stats,
            'total_mixed_aspects'  : total_aspects,
            'correct_mixed_aspects': correct_aspects,
        }

    def print_mixed_sentiment_results(self, metrics):
        print(f"\n{'='*70}")
        print("MIXED SENTIMENT RESOLUTION EVALUATION")
        print(f"{'='*70}")
        stats = metrics['stats']
        print(f"\nDataset stats:")
        print(f"  Total reviews:             {stats['total_reviews']}")
        print(f"  Multi-aspect reviews:      {stats['multi_aspect_reviews']}")
        print(f"  Mixed sentiment reviews:   {stats['mixed_sentiment_reviews']}")
        print(f"  Mixed % of multi-aspect:   {stats['mixed_percentage_of_multi']:.2f}%")
        print(f"  Mixed % of total:          {stats['mixed_percentage_of_total']:.2f}%")
        print(f"\nConflict types:")
        ct = stats['conflict_types']
        print(f"  Positive + Negative:       {ct['positive_negative']}")
        print(f"  All three sentiments:      {ct['positive_neutral_negative']}")
        print(f"  Neutral with extremes:     {ct['neutral_with_extremes']}")
        if metrics['mixed_review_count'] > 0:
            print(f"\nModel performance on mixed reviews:")
            print(f"  Total mixed reviewed:      {metrics['mixed_review_count']}")
            print(f"  Review-level accuracy:     {metrics['mixed_review_accuracy']:.2f}%")
            print(f"    (reviews where ALL aspects correct)")
            print(f"  Aspect-level accuracy:     {metrics['mixed_aspect_accuracy']:.2f}%")
            print(f"    ({metrics['correct_mixed_aspects']}/{metrics['total_mixed_aspects']} aspects correct)")
        print(f"{'='*70}\n")

    def save_mixed_sentiment_analysis(self, metrics, save_path):
        import json
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[MSREvaluator] Analysis saved to {save_path}")


if __name__ == "__main__":
    print("Testing AspectSentimentEvaluator...")
    y_true = np.array([0, 0, 1, 1, 2, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 2, 1, 0, 1, 2])
    evaluator = AspectSentimentEvaluator(['smell', 'texture', 'price'])
    evaluator.evaluate_aspect(y_true, y_pred, 'test_aspect')
    evaluator.print_results('test_aspect')
    print("\nEvaluator test passed!")
