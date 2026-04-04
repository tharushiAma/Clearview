"""
compute_msr.py
==============
Compute Mixed Sentiment Resolution (MSR) metrics for specific experiments
from their existing .pt checkpoints and patch all_results.json.

Usage (run from ml-research root):
    python compute_msr.py

Targets:  A5_shared_head, A2_cls_pooling
"""

import os, sys, json, copy
import numpy as np
import torch
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
SRC_DIR      = os.path.join(PROJECT_ROOT, 'src')
UTILS_DIR    = os.path.join(PROJECT_ROOT, 'utils')

for p in [PROJECT_ROOT, SRC_DIR, UTILS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.model import create_model
from models.losses import AspectSpecificLossManager
from utils.data_utils import create_dataloaders
from utils.metrics import MixedSentimentEvaluator
from transformers import RobertaTokenizer

# ── Settings ──────────────────────────────────────────────────────────────────
RESULTS_DIR  = Path(PROJECT_ROOT) / 'outputs' / 'experiments'
RESULTS_JSON = RESULTS_DIR / 'all_results.json'

EXPERIMENTS = [
    'A2_cls_pooling',
    'A5_shared_head',
]


def run_msr_for_experiment(exp_id: str) -> dict:
    """
    Load checkpoint, rebuild model, run MSR evaluation, return metrics dict.
    """
    ckpt_path = RESULTS_DIR / f'{exp_id}_best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\n{'='*65}")
    print(f"Computing MSR for [{exp_id}]")
    print(f"{'='*65}")

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # ── 2. Rebuild model ──────────────────────────────────────────────────────
    print(f"  Rebuilding model from config …")
    model = create_model(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"  Model loaded OK")

    # ── 3. Build test loader ──────────────────────────────────────────────────
    roberta_name = config['model']['roberta_model']
    tokenizer = RobertaTokenizer.from_pretrained(roberta_name)

    dep_parser = None   # A2_cls_pooling and A5_shared_head don't use dependency parsing
    _, _, test_loader = create_dataloaders(config, tokenizer, dep_parser)

    # ── 4. Inference – collect per-review predictions ─────────────────────────
    print(f"  Running inference on test set …")
    review_true: dict = {}
    review_pred: dict = {}

    use_gcn = config['model'].get('use_dependency_gcn', False)

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            aspect_ids     = batch['aspect_ids'].to(device)

            edge_indices = None
            if use_gcn:
                edge_indices = [
                    e.to(device) if e is not None else None
                    for e in batch['edge_indices']
                ]

            preds = model(input_ids, attention_mask, aspect_ids, edge_indices)
            if isinstance(preds, tuple):
                preds = preds[0]

            pred_classes = torch.argmax(preds, dim=1).cpu().numpy()

            for i in range(len(pred_classes)):
                review_idx  = batch['review_ids'][i]
                aspect_name = batch['aspects'][i]
                true_label  = batch['labels'][i].item()

                if review_idx not in review_true:
                    review_true[review_idx] = {}
                    review_pred[review_idx] = {}

                review_true[review_idx][aspect_name] = true_label
                review_pred[review_idx][aspect_name] = int(pred_classes[i])

    print(f"  Inference done — {len(review_true)} reviews collected")

    # ── 5. MSR evaluation ─────────────────────────────────────────────────────
    mixed_evaluator = MixedSentimentEvaluator(config['aspects']['names'])
    mixed_metrics   = mixed_evaluator.evaluate_mixed_sentiment_resolution(
        review_true, review_pred
    )
    mixed_evaluator.print_mixed_sentiment_results(mixed_metrics)

    # ── 6. Return serialisable dict ───────────────────────────────────────────
    return {
        'mixed_review_count'   : mixed_metrics.get('mixed_review_count', 0),
        'mixed_review_accuracy': mixed_metrics.get('mixed_review_accuracy', 0.0),
        'mixed_aspect_accuracy': mixed_metrics.get('mixed_aspect_accuracy', 0.0),
        'mixed_detection_rate' : mixed_metrics.get('mixed_detection_rate',  0.0),
    }


def main():
    # ── Load existing results ─────────────────────────────────────────────────
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"all_results.json not found at {RESULTS_JSON}")

    with open(RESULTS_JSON, 'r') as f:
        all_results = json.load(f)

    updated = []

    for exp_id in EXPERIMENTS:
        if exp_id not in all_results:
            print(f"\n  WARNING: '{exp_id}' not found in all_results.json — skipping")
            continue

        existing_msr = all_results[exp_id].get('mixed_sentiment', {})
        if existing_msr and existing_msr.get('mixed_review_count', 0) > 0:
            print(f"\n  [{exp_id}] already has MSR metrics — skipping (delete the key to force recompute)")
            continue

        try:
            msr_metrics = run_msr_for_experiment(exp_id)
            all_results[exp_id]['mixed_sentiment'] = msr_metrics
            updated.append(exp_id)
            print(f"\n  [{exp_id}] MSR metrics saved:")
            for k, v in msr_metrics.items():
                print(f"    {k}: {v}")
        except Exception as exc:
            import traceback
            print(f"\n  [{exp_id}] FAILED: {exc}")
            traceback.print_exc()

    # ── Persist updated results ───────────────────────────────────────────────
    if updated:
        with open(RESULTS_JSON, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ all_results.json updated for: {', '.join(updated)}")
    else:
        print("\nNo experiments were updated.")


if __name__ == '__main__':
    main()
