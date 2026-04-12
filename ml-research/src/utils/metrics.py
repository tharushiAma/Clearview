"""
Evaluation metrics for multi-aspect sentiment analysis.

This module contains ONLY the core computational classes:
  - AspectSentimentEvaluator  : per-aspect F1, accuracy, MCC, ROC-AUC
  - ErrorAnalyzer             : misclassification breakdown by type & aspect
  - MixedSentimentEvaluator   : mixed-sentiment resolution (MSR) scoring

Visualisation helpers (confusion-matrix plots, ROC curves, compare_aspects,
generate_latex_table) live in the development notebook
  notebooks/02_model_development/06_metrics.ipynb
and are intentionally kept out of this production module so that training
runs do not depend on matplotlib / seaborn.
"""

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, matthews_corrcoef,
    roc_auc_score, roc_curve
)
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm


class AspectSentimentEvaluator:
    """
    Comprehensive evaluation for imbalanced multi-aspect sentiment.

    Stores per-aspect metrics in self.results and exposes methods used
    by ExperimentTrainer during training / evaluation.

    Primary metric: Macro-F1 (weights all classes equally, which is critical
    for imbalanced data where weighted-F1 can be misleadingly high).
    """
    def __init__(self, aspect_names: list):
        self.aspect_names = aspect_names
        self.results = {}
        print(f"[Evaluator] AspectSentimentEvaluator ready -- tracking {len(aspect_names)} aspects: {aspect_names}")

    def evaluate_aspect(self, y_true, y_pred, aspect_name, y_prob=None):
        """
        Compute comprehensive metrics for a single aspect.

        Parameters
        ----------
        y_true       : array-like of int   true labels (0=neg, 1=neu, 2=pos)
        y_pred       : array-like of int   predicted labels
        aspect_name  : str                 name used as key in self.results
        y_prob       : (n, 3) array or None  softmax probabilities for ROC-AUC

        y_prob is only provided when the full model runs; baselines pass None
        so that ROC-AUC is not computed for them.
        """
        n = len(y_true)
        print(f"\n[Evaluator] Evaluating aspect: '{aspect_name}'  ({n} samples)")

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0, labels=[0, 1, 2]
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0, labels=[0, 1, 2]
        )
        mcc = matthews_corrcoef(y_true, y_pred)
        cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        # ROC AUC (One-vs-Rest) -- only computed when probability estimates are
        # available.  multi_class="ovr" treats each of the 3 sentiment classes
        # as a binary problem in turn; macro average weights all three equally,
        # which is important given the class imbalance in our dataset.
        roc_auc = None
        if y_prob is not None:
            try:
                roc_auc = roc_auc_score(
                    y_true, y_prob, labels=[0, 1, 2],
                    multi_class="ovr", average="macro"
                )
            except Exception as e:
                # Fails when a class has zero samples in a small batch
                print(f"[Evaluator] WARNING: Could not compute ROC AUC for {aspect_name}: {e}")

        # Fixed labels=[0,1,2] so zero-sample classes produce a defined 0.0
        # score rather than shifting array positions.
        self.results[aspect_name] = {
            "accuracy"           : accuracy,
            "macro_precision"    : macro_precision,
            "macro_recall"       : macro_recall,
            "macro_f1"           : macro_f1,           # Primary metric -- weights all 3 classes equally
            "weighted_precision" : weighted_precision,
            "weighted_recall"    : weighted_recall,
            "weighted_f1"        : weighted_f1,         # Secondary -- can be misleadingly high for imbalanced data
            "mcc"                : mcc,                 # Balanced single-number metric; 0=random, +1=perfect
            "per_class_precision": precision,
            "per_class_recall"   : recall,
            "per_class_f1"       : f1,
            "support"            : support,
            "confusion_matrix"   : cm,
            "roc_auc"            : roc_auc,
            "y_true"             : y_true,  # Stored for ROC/AUC curve plotting in notebooks
            "y_prob"             : y_prob,  # Softmax probabilities; None for models without prob output
        }

        print(f"  Accuracy: {accuracy:.4f}  |  Macro-F1: {macro_f1:.4f}  |  "
              f"Weighted-F1: {weighted_f1:.4f}  |  MCC: {mcc:.4f}")
        if roc_auc:
            print(f"  ROC AUC (OvR Macro): {roc_auc:.4f}")
        print(f"  Per-class F1 -- neg: {f1[0]:.4f}  neu: {f1[1]:.4f}  pos: {f1[2]:.4f}")
        print(f"  Support      -- neg: {int(support[0])}  neu: {int(support[1])}  pos: {int(support[2])}")

        return self.results[aspect_name]

    def save_results(self, save_path):
        """Serialise self.results to a JSON file (numpy types converted)."""
        import json
        serializable = {}
        for aspect, metrics in self.results.items():
            serializable[aspect] = {
                k: v.tolist() if isinstance(v, np.ndarray)
                   else float(v) if isinstance(v, (np.float32, np.float64))
                   else v
                for k, v in metrics.items()
            }
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"[Evaluator] Results saved to {save_path}")


class ErrorAnalyzer:
    """
    Analyse prediction errors for model-improvement insights.

    Used only in notebooks / post-hoc analysis; not called by experiment_runner.
    """
    def __init__(self, aspect_names, class_names):
        self.aspect_names = aspect_names
        self.class_names  = class_names

    def analyze_errors(self, texts, y_true, y_pred, aspects, save_path=None):
        print(f"\n[ErrorAnalyzer] Analyzing {len(texts)} predictions...")

        errors = [
            {
                "text"      : texts[i],
                "aspect"    : aspects[i],
                "true_label": self.class_names[y_true[i]],
                "pred_label": self.class_names[y_pred[i]],
                # String form "negative->positive" makes downstream error-type
                # counting and grouping easy without a secondary lookup.
                "error_type": f"{self.class_names[y_true[i]]}->{self.class_names[y_pred[i]]}",
            }
            for i in range(len(texts)) if y_true[i] != y_pred[i]
        ]

        print(f"[ErrorAnalyzer] Total errors: {len(errors)} / {len(texts)} "
              f"({len(errors)/len(texts)*100:.2f}%)")

        aspect_errors = Counter([e["aspect"] for e in errors])
        print(f"\n[ErrorAnalyzer] Error rate by aspect:")
        for aspect in sorted(aspect_errors):
            total = sum(1 for a in aspects if a == aspect)
            print(f"  {aspect:<16}: {aspect_errors[aspect]:>4} / {total} "
                  f"({aspect_errors[aspect]/total*100:.2f}%)")

        error_types = Counter([e["error_type"] for e in errors])
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
    Evaluator for mixed sentiment resolution (MSR).

    Mixed sentiment = reviews expressing conflicting opinions across
    different aspects (e.g., positive colour but negative smell).
    These are the hardest cases for any sentiment model.

    Label encoding: 0 = negative, 1 = neutral, 2 = positive
    """
    def __init__(self, aspect_names):
        self.aspect_names = aspect_names
        self.class_names  = ["negative", "neutral", "positive"]
        print(f"[MSREvaluator] Ready -- tracking {len(aspect_names)} aspects")

    def identify_mixed_sentiment_reviews(self, reviews_data):
        print(f"\n[MSREvaluator] Scanning {len(reviews_data)} reviews for mixed sentiment...")
        mixed_reviews: list = []
        stats = {
            "total_reviews"          : len(reviews_data),
            "mixed_sentiment_reviews": 0,
            "single_aspect_reviews"  : 0,
            "multi_aspect_reviews"   : 0,
            "conflict_types"         : {
                "positive_negative"         : 0,
                "positive_neutral_negative" : 0,
                "neutral_with_extremes"     : 0,
            },
        }

        for review in tqdm(reviews_data, desc="  Scanning reviews"):
            active = {asp: sent for asp, sent in review["aspects"].items() if sent is not None}
            if not active:
                continue
            if len(active) == 1:
                stats["single_aspect_reviews"] += 1
                continue
            stats["multi_aspect_reviews"] += 1

            sentiments   = set(active.values())
            has_positive = 2 in sentiments
            has_neutral  = 1 in sentiments
            has_negative = 0 in sentiments
            is_mixed     = False

            # "positive + negative" is the most interesting conflict: the model
            # must assign opposite polarities to two aspects of the same review.
            if has_positive and has_negative:
                is_mixed = True
                if has_neutral:
                    stats["conflict_types"]["positive_neutral_negative"] += 1  # All three sentiments present
                else:
                    stats["conflict_types"]["positive_negative"] += 1          # Only the two extremes
            elif has_neutral and (has_positive or has_negative):
                # Softer conflict: neutral alongside one polarised aspect
                is_mixed = True
                stats["conflict_types"]["neutral_with_extremes"] += 1

            if is_mixed:
                mixed_reviews.append(review["review_id"])
                stats["mixed_sentiment_reviews"] += 1

        multi = stats["multi_aspect_reviews"]
        stats["mixed_percentage_of_multi"] = (
            stats["mixed_sentiment_reviews"] / multi * 100 if multi > 0 else 0.0
        )
        stats["mixed_percentage_of_total"] = (
            stats["mixed_sentiment_reviews"] / stats["total_reviews"] * 100
        )

        print(f"[MSREvaluator] Found {stats['mixed_sentiment_reviews']} mixed reviews "
              f"({stats['mixed_percentage_of_multi']:.1f}% of multi-aspect, "
              f"{stats['mixed_percentage_of_total']:.1f}% of total)")
        return mixed_reviews, stats

    def evaluate_mixed_sentiment_resolution(self, y_true_dict, y_pred_dict):
        print(f"\n[MSREvaluator] Evaluating mixed sentiment resolution on "
              f"{len(y_true_dict)} reviews...")

        reviews_data_true = [
            {"review_id": rid, "text": "", "aspects": y_true_dict[rid]}
            for rid in y_true_dict
        ]
        mixed_review_ids, mixed_stats = self.identify_mixed_sentiment_reviews(reviews_data_true)

        if not mixed_review_ids:
            print("[MSREvaluator] WARNING: No mixed sentiment reviews found in dataset")
            return {
                "mixed_review_count"   : 0,
                "mixed_prevalence"     : 0.0,
                "mixed_review_accuracy": 0.0,
                "mixed_aspect_accuracy": 0.0,
                "stats"                : mixed_stats,
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
                    total_aspects += 1
                    if true_asp[aspect] == pred_asp[aspect]:
                        correct_aspects += 1
                    else:
                        all_correct = False
                else:
                    all_correct = False

            if all_correct:
                correct_reviews += 1

        review_acc = correct_reviews / len(mixed_review_ids) * 100
        aspect_acc = correct_aspects / total_aspects * 100 if total_aspects > 0 else 0.0

        print(f"[MSREvaluator] Review-level accuracy (all aspects correct): {review_acc:.2f}%")
        print(f"[MSREvaluator] Aspect-level accuracy ({correct_aspects}/{total_aspects}): "
              f"{aspect_acc:.2f}%")

        return {
            "mixed_review_count"   : len(mixed_review_ids),
            "mixed_prevalence"     : mixed_stats["mixed_percentage_of_multi"],  # % of multi-aspect reviews that are mixed
            "mixed_review_accuracy": review_acc,    # % of mixed reviews where ALL aspect predictions were correct
            "mixed_aspect_accuracy": aspect_acc,    # % of individual aspect predictions correct within mixed reviews
            "stats"                : mixed_stats,
            "total_mixed_aspects"  : total_aspects,
            "correct_mixed_aspects": correct_aspects,
        }

    def print_mixed_sentiment_results(self, metrics):
        print(f"\n{'='*70}")
        print("MIXED SENTIMENT RESOLUTION EVALUATION")
        print(f"{'='*70}")
        stats = metrics["stats"]
        print(f"\nDataset stats:")
        print(f"  Total reviews:             {stats['total_reviews']}")
        print(f"  Multi-aspect reviews:      {stats['multi_aspect_reviews']}")
        print(f"  Mixed sentiment reviews:   {stats['mixed_sentiment_reviews']}")
        print(f"  Mixed % of multi-aspect:   {stats['mixed_percentage_of_multi']:.2f}%")
        print(f"  Mixed % of total:          {stats['mixed_percentage_of_total']:.2f}%")
        print(f"\nConflict types:")
        ct = stats["conflict_types"]
        print(f"  Positive + Negative:       {ct['positive_negative']}")
        print(f"  All three sentiments:      {ct['positive_neutral_negative']}")
        print(f"  Neutral with extremes:     {ct['neutral_with_extremes']}")
        if metrics["mixed_review_count"] > 0:
            print(f"\nModel performance on mixed reviews:")
            print(f"  Total mixed reviewed:      {metrics['mixed_review_count']}")
            print(f"  Review-level accuracy:     {metrics['mixed_review_accuracy']:.2f}%")
            print(f"    (reviews where ALL aspects correct)")
            print(f"  Aspect-level accuracy:     {metrics['mixed_aspect_accuracy']:.2f}%")
            print(f"    ({metrics['correct_mixed_aspects']}/{metrics['total_mixed_aspects']} aspects correct)")
        print(f"{'='*70}\n")

    def save_mixed_sentiment_analysis(self, metrics, save_path):
        import json
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[MSREvaluator] Analysis saved to {save_path}")


if __name__ == "__main__":
    print("Testing AspectSentimentEvaluator...")
    y_true = np.array([0, 0, 1, 1, 2, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 2, 1, 0, 1, 2])
    evaluator = AspectSentimentEvaluator(["smell", "texture", "price"])
    evaluator.evaluate_aspect(y_true, y_pred, "test_aspect")
    print("\nEvaluator test passed!")
