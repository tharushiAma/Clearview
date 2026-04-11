"""
baseline_models.py
Baseline model implementations for comparison against the full ClearView model.

Baselines:
  1. PlainRoBERTa         — [CLS] token, single shared head, no aspect awareness
  2. RoBERTaWithCE        — Full architecture but Cross-Entropy loss (no hybrid loss)
  3. BERTBase             — BERT-base-uncased encoder instead of RoBERTa
  4. TFIDFSVMBaseline     — Classical TF-IDF + SVM (per-aspect, per-label)
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel, DistilBertModel

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Plain RoBERTa
# No aspect embeddings, no attention mechanism, no GCN.
# Uses [CLS] token representation with a single shared classification head.
# ─────────────────────────────────────────────────────────────────────────────
class PlainRoBERTa(nn.Module):
    """
    Simplest possible RoBERTa baseline.
    Fine-tunes RoBERTa with a single shared classification head over [CLS].
    Does NOT know which aspect it's predicting — treats all aspects the same.
    """
    def __init__(self, roberta_model='roberta-base', num_classes=3, dropout=0.1):
        super(PlainRoBERTa, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, aspect_id=None, edge_index=None, **kwargs):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            aspect_id: ignored (for interface compatibility with MultiAspectSentimentModel)
        Returns:
            logits: (batch_size, num_classes)
        """
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]  # [CLS] token is the aggregate sentence representation
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: RoBERTa + Aspect Attention (same arch) + Cross-Entropy Loss
# Same as full model but WITHOUT hybrid loss. Used to isolate loss contribution.
# Training is done through experiment_runner.py — this is just the loss config.
# ─────────────────────────────────────────────────────────────────────────────
class CrossEntropyLossWrapper(nn.Module):
    """
    Wraps nn.CrossEntropyLoss with the same interface as AspectSpecificLossManager.
    Used to test full architecture with plain CE loss.
    """
    def __init__(self):
        super(CrossEntropyLossWrapper, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, predictions, targets, aspect_ids, aspect_names):
        """Same interface as AspectSpecificLossManager.compute_loss.

        Returns a loss_details dict with 'ce' and 'total' keys so that
        per-aspect loss logging in the A3 ablation is consistent across all
        loss variants (Hybrid variants log 'focal', 'cb', 'dice', 'total').
        """
        loss = self.criterion(predictions, targets)
        loss_val = loss.item()
        loss_details = {
            'ce':    loss_val,
            'total': loss_val,
        }
        return loss, loss_details


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: DistilBERT-base
# DistilBERT-base-uncased with a single shared [CLS] classification head.
# Lightweight baseline for performance/size trade-off analysis.
# ─────────────────────────────────────────────────────────────────────────────
class DistilBERTBaseline(nn.Module):
    """
    DistilBERT-base-uncased with a single shared [CLS] classification head.
    Aspect-unaware — same as PlainRoBERTa but with DistilBERT encoder.
    """
    def __init__(self, distilbert_model='distilbert-base-uncased', num_classes=3, dropout=0.1):
        super(DistilBERTBaseline, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(distilbert_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, aspect_id=None, edge_index=None, **kwargs):
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 3: BERT-base
# Identical architecture to PlainRoBERTa but using bert-base-uncased.
# Tests whether RoBERTa's pre-training provides meaningful improvements.
# ─────────────────────────────────────────────────────────────────────────────
class BERTBaseline(nn.Module):
    """
    BERT-base-uncased with a single shared [CLS] classification head.
    Aspect-unaware — same as PlainRoBERTa but with BERT encoder.
    """
    def __init__(self, bert_model='bert-base-uncased', num_classes=3, dropout=0.1):
        super(BERTBaseline, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, aspect_id=None, edge_index=None, **kwargs):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 4: TF-IDF + SVM (classical)
# One SVM classifier per aspect per class (one-vs-rest).
# Does not use any deep learning — tests classical NLP baselines.
# ─────────────────────────────────────────────────────────────────────────────
class TFIDFSVMBaseline:
    """
    Classical TF-IDF + LinearSVC baseline.
    Trains one classifier per aspect (3-class: neg/neu/pos).
    No GPU, no transformers — pure sklearn.
    """
    def __init__(self, aspect_names, max_features=50000, ngram_range=(1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline

        self.aspect_names = aspect_names
        self.pipelines = {}

        for aspect in aspect_names:
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,    # Captures both unigrams and bigrams (e.g. 'not good')
                    sublinear_tf=True,          # log(1 + tf) scaling reduces dominance of very frequent terms
                    strip_accents='unicode',
                    analyzer='word',
                    min_df=2,                   # Ignore terms that appear in fewer than 2 documents
                )),
                # CalibratedClassifierCV wraps LinearSVC to produce probability estimates via Platt scaling.
                # LinearSVC alone cannot output probabilities; this is needed for predict_proba().
                ('clf', CalibratedClassifierCV(LinearSVC(
                    class_weight='balanced',    # Automatically adjusts weights inversely proportional to class frequency
                    max_iter=2000,
                    C=1.0,                      # Regularisation strength; 1.0 is the sklearn default
                ))),
            ])
            self.pipelines[aspect] = pipeline

    def fit(self, df, label_map):
        """
        Train one SVM per aspect.
        Args:
            df: DataFrame with 'data' column and aspect columns
            label_map: {'negative': 0, 'neutral': 1, 'positive': 2}
        """
        from sklearn.preprocessing import LabelEncoder
        reverse_map = {v: k for k, v in label_map.items()}

        for aspect in self.aspect_names:
            # Only use rows where the aspect is labelled
            mask = df[aspect].notna()
            if mask.sum() == 0:
                print(f"  Skipping {aspect}: no labelled samples")
                continue

            X = df.loc[mask, 'data'].astype(str).tolist()
            y = df.loc[mask, aspect].map(
                lambda v: label_map.get(str(v).lower(), -1)
            ).tolist()

            # Drop any unmapped labels
            valid = [(x, lbl) for x, lbl in zip(X, y) if lbl != -1]
            if not valid:
                continue
            X_valid, y_valid = zip(*valid)

            print(f"  Training SVM for {aspect}: {len(X_valid)} samples")
            self.pipelines[aspect].fit(X_valid, y_valid)

        print("TF-IDF + SVM training complete.")

    def predict(self, texts, aspect):
        """
        Args:
            texts: List[str]
            aspect: str
        Returns:
            predictions: np.ndarray of int labels
        """
        if aspect not in self.pipelines:
            raise ValueError(f"Unknown aspect: {aspect}")
        return self.pipelines[aspect].predict(texts)

    def predict_proba(self, texts, aspect):
        """Returns probability estimates (neg, neu, pos)."""
        if aspect not in self.pipelines:
            raise ValueError(f"Unknown aspect: {aspect}")
        return self.pipelines[aspect].predict_proba(texts)

    def save(self, save_path):
        """Save all pipelines using joblib."""
        import joblib, os
        os.makedirs(save_path, exist_ok=True)
        for aspect, pipeline in self.pipelines.items():
            path = os.path.join(save_path, f"svm_{aspect}.pkl")
            joblib.dump(pipeline, path)
        print(f"SVM models saved to {save_path}")

    @classmethod
    def load(cls, save_path, aspect_names):
        """Load saved pipelines."""
        import joblib, os
        # Use __new__ to bypass __init__ (which would create new empty pipelines)
        # and directly populate the object with the stored pipelines.
        obj = cls.__new__(cls)
        obj.aspect_names = aspect_names
        obj.pipelines = {}
        for aspect in aspect_names:
            path = os.path.join(save_path, f"svm_{aspect}.pkl")
            if os.path.exists(path):
                obj.pipelines[aspect] = joblib.load(path)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def create_baseline(baseline_name: str, config: dict):
    """
    Factory function for baseline models.
    Args:
        baseline_name: 'plain_roberta' | 'ce_loss' | 'bert_base' | 'tfidf_svm'
        config: full config dict (for compatibility)
    """
    num_classes = config['model']['num_classes']
    dropout     = config['model']['dropout']
    aspects     = config['aspects']['names']

    if baseline_name == 'plain_roberta':
        model = PlainRoBERTa(
            roberta_model=config['model']['roberta_model'],
            num_classes=num_classes,
            dropout=dropout,
        )
        print(f"[Baseline] PlainRoBERTa: {model.get_num_parameters():,} params")
        return model

    elif baseline_name == 'ce_loss':
        # Same model as full system, but loss is replaced in experiment_runner
        from models.model import create_model
        model = create_model(config)
        print("[Baseline] Full architecture + CrossEntropy loss")
        return model

    elif baseline_name == 'distilbert_base':
        model = DistilBERTBaseline(
            distilbert_model='distilbert-base-uncased',
            num_classes=num_classes,
            dropout=dropout,
        )
        print(f"[Baseline] DistilBERT-base: {model.get_num_parameters():,} params")
        return model

    elif baseline_name == 'bert_base':
        model = BERTBaseline(
            bert_model='bert-base-uncased',
            num_classes=num_classes,
            dropout=dropout,
        )
        print(f"[Baseline] BERT-base: {model.get_num_parameters():,} params")
        return model

    elif baseline_name == 'tfidf_svm':
        model = TFIDFSVMBaseline(aspect_names=aspects)
        print("[Baseline] TF-IDF + LinearSVC")
        return model

    else:
        raise ValueError(
            f"Unknown baseline: '{baseline_name}'. "
            "Choose from: plain_roberta, ce_loss, bert_base, tfidf_svm"
        )
