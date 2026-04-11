"""
recompute_eval.py
=================
Evaluation-only script. Loads saved .pt checkpoints and recomputes full
metrics (accuracy, macro-F1, MCC, ROC-AUC, y_prob) WITHOUT any re-training.

Use this when:
  - Checkpoints are available but y_prob / roc_auc were not saved originally.
  - You want to re-evaluate on a different machine / after code changes.
  - You want to patch all_results.json with the corrected/new metrics.

Checkpoint format expected (saved by experiment_runner.py):
  {
    'model_state_dict': ...,
    'config': { ... full experiment config ... }
  }

Usage (run from ml-research root):
    # Re-evaluate ALL experiments that have a .pt checkpoint
    python recompute_eval.py

    # Re-evaluate a specific experiment
    python recompute_eval.py --experiment A1_full_model

    # Point to a custom checkpoint dir
    python recompute_eval.py --results_dir outputs/experiments

    # Dry-run: show what would be evaluated without writing anything
    python recompute_eval.py --dry_run
"""

import os
import sys
import json
import argparse
import copy
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# ── Path bootstrap ────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
SRC_DIR      = os.path.join(PROJECT_ROOT, 'src')
UTILS_DIR    = os.path.join(PROJECT_ROOT, 'utils')
for p in [PROJECT_ROOT, SRC_DIR, UTILS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.model import create_model
from models.losses import AspectSpecificLossManager
from utils.data_utils import create_dataloaders, DependencyParser
from utils.metrics import AspectSentimentEvaluator, MixedSentimentEvaluator
from transformers import RobertaTokenizer, BertTokenizer, DistilBertTokenizer
from experiments.baseline_models import create_baseline


# ── Serialisation helper ──────────────────────────────────────────────────────
def _ser(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16)): return int(obj)
    if isinstance(obj, (np.float64, np.float32)):        return float(obj)
    if isinstance(obj, dict):  return {k: _ser(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_ser(x) for x in obj]
    return obj


# ── Model builder (mirrors experiment_runner logic) ───────────────────────────
def _build_model(exp_id: str, config: dict):
    """Reconstruct the right model class from exp_id prefix and config."""
    if exp_id.startswith('B1_'):
        return create_baseline('plain_roberta', config)
    elif exp_id.startswith('B2_'):
        return create_baseline('distilbert_base', config)
    elif exp_id.startswith('B3_'):
        return create_baseline('bert_base', config)
    else:
        return create_model(config)


def _build_tokenizer(config: dict):
    roberta_name = config['model']['roberta_model']
    if 'distilbert' in roberta_name:
        return DistilBertTokenizer.from_pretrained(roberta_name)
    elif 'bert' in roberta_name and 'roberta' not in roberta_name:
        return BertTokenizer.from_pretrained(roberta_name)
    else:
        return RobertaTokenizer.from_pretrained(roberta_name)


# ── Core evaluation function ──────────────────────────────────────────────────
def evaluate_checkpoint(exp_id: str, ckpt_path: Path, device: torch.device) -> dict:
    """
    Load checkpoint → rebuild model → run full test-set evaluation.

    Returns a result dict matching the all_results.json schema, with
    y_prob and roc_auc properly populated.
    """
    print(f"\n{'='*65}")
    print(f"  Evaluating: [{exp_id}]")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*65}")

    # 1. Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']

    # 2. Rebuild model
    model = _build_model(exp_id, config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"  Model loaded OK → {type(model).__name__}")

    # 3. Build tokenizer + dataloader (test split only)
    tokenizer = _build_tokenizer(config)
    dep_parser = None
    if config.get('data', {}).get('use_dependency_parsing', False) and \
       config.get('model', {}).get('use_dependency_gcn', False):
        dep_parser = DependencyParser(config['data'].get('language', 'en'))
        print(f"  Dependency parser initialised")

    _, _, test_loader = create_dataloaders(config, tokenizer, dep_parser)
    print(f"  Test loader ready — {len(test_loader)} batches")

    # 4. Inference — collect labels, predictions, softmax probabilities
    use_gcn = config['model'].get('use_dependency_gcn', False)
    all_labels, all_aspects = [], []
    all_probs  = []   # softmax probability vectors  [n_samples, 3]

    # For MSR: {review_id: {aspect: label}}
    review_true: dict = {}
    review_pred: dict = {}

    print(f"  Running inference …")
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

            logits = model(input_ids, attention_mask, aspect_ids, edge_indices)
            if isinstance(logits, tuple):
                logits = logits[0]

            # Softmax probabilities
            probs       = torch.softmax(logits, dim=1).cpu().numpy()
            pred_classes = np.argmax(probs, axis=1)

            all_probs.extend(probs)
            all_labels.extend(batch['labels'].numpy())
            all_aspects.extend(batch['aspects'])

            # For MSR
            for i in range(len(pred_classes)):
                rid  = batch['review_ids'][i]
                asp  = batch['aspects'][i]
                true = batch['labels'][i].item()
                if rid not in review_true:
                    review_true[rid] = {}
                    review_pred[rid] = {}
                review_true[rid][asp] = true
                review_pred[rid][asp] = int(pred_classes[i])

    all_probs_arr  = np.array(all_probs)    # shape: (N, 3)
    all_labels_arr = np.array(all_labels)   # shape: (N,)
    all_preds_arr  = np.argmax(all_probs_arr, axis=1)

    print(f"  Inference done — {len(all_labels_arr)} samples")

    # 5. Compute metrics via AspectSentimentEvaluator
    aspect_names = config['aspects']['names']
    evaluator    = AspectSentimentEvaluator(aspect_names)

    per_aspect_metrics = {}
    for aspect in aspect_names:
        mask = np.array([a == aspect for a in all_aspects])
        if mask.sum() == 0:
            continue
        y_true = all_labels_arr[mask]
        y_pred = all_preds_arr[mask]
        y_prob = all_probs_arr[mask]
        per_aspect_metrics[aspect] = evaluator.evaluate_aspect(
            y_true, y_pred, aspect, y_prob=y_prob
        )

    overall_metrics = evaluator.evaluate_aspect(
        all_labels_arr, all_preds_arr, 'overall', y_prob=all_probs_arr
    )

    # 6. MSR evaluation
    mixed_sentiment = {}
    if config.get('experiment', {}).get('evaluate_msr', False) or True:
        # Always attempt MSR — it's informative and cheap
        try:
            mixed_evaluator = MixedSentimentEvaluator(aspect_names)
            mixed_metrics   = mixed_evaluator.evaluate_mixed_sentiment_resolution(
                review_true, review_pred
            )
            mixed_evaluator.print_mixed_sentiment_results(mixed_metrics)
            mixed_sentiment = {
                'mixed_review_count'   : mixed_metrics.get('mixed_review_count', 0),
                'mixed_review_accuracy': mixed_metrics.get('mixed_review_accuracy', 0.0),
                'mixed_aspect_accuracy': mixed_metrics.get('mixed_aspect_accuracy', 0.0),
                'mixed_prevalence' : mixed_metrics.get('mixed_prevalence', 0.0),
            }
        except Exception as msr_exc:
            print(f"  MSR evaluation failed: {msr_exc}")

    # 7. Assemble result (mirrors all_results.json schema)
    def _clean(d):
        """Remove non-serialisable keys (y_true / y_prob arrays are kept)."""
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, (np.int64, np.int32)):
                out[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                out[k] = float(v)
            else:
                out[k] = v
        return out

    per_aspect_out = {
        asp: _clean(metrics)
        for asp, metrics in per_aspect_metrics.items()
    }
    overall_out = _clean(overall_metrics)

    print(f"\n  ── Result summary ────────────────────────────────")
    print(f"  Accuracy:  {overall_out['accuracy']:.4f}")
    print(f"  Macro-F1:  {overall_out['macro_f1']:.4f}")
    print(f"  MCC:       {overall_out['mcc']:.4f}")
    if overall_out.get('roc_auc'):
        print(f"  ROC-AUC:   {overall_out['roc_auc']:.4f}  ✓")
    else:
        print(f"  ROC-AUC:   not computed (check class support)")

    return {
        'experiment_id'  : exp_id,
        'description'    : config.get('experiment', {}).get('name', exp_id),
        'status'         : 'done',
        'error'          : None,
        'reeval_timestamp': datetime.utcnow().isoformat() + 'Z',
        'overall'        : overall_out,
        'per_aspect'     : per_aspect_out,
        'mixed_sentiment': mixed_sentiment,
    }


# ── Discovery: find checkpoints and match to experiment IDs ──────────────────
def discover_checkpoints(results_dir: Path) -> dict:
    """
    Scan results_dir for *_best.pt files and return {exp_id: Path}.
    Falls back to searching the whole outputs/ tree.
    """
    mapping = {}

    # Primary: experiment-named checkpoints  e.g. A1_full_model_best.pt
    for pt in sorted(results_dir.glob('*_best.pt')):
        exp_id = pt.stem.replace('_best', '')
        mapping[exp_id] = pt

    # Secondary: legacy single checkpoint  e.g. outputs/cosmetic_sentiment_v1/best_model.pt
    if not mapping:
        for pt in sorted(Path('outputs').rglob('best_model.pt')):
            # Try to derive exp_id from checkpoint config
            try:
                ckpt   = torch.load(pt, map_location='cpu', weights_only=False)
                exp_id = ckpt.get('config', {}).get('experiment', {}).get('name', pt.parent.name)
                mapping[exp_id] = pt
            except Exception:
                mapping[pt.parent.name] = pt

    return mapping


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Re-run evaluation from saved checkpoints (no training)."
    )
    parser.add_argument('--results_dir', default='outputs/experiments',
                        help='Directory containing *_best.pt checkpoint files')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Specific experiment ID to re-evaluate (e.g. A1_full_model)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Explicit path to a .pt checkpoint file (overrides --experiment lookup)')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Path to write updated results JSON '
                             '(default: <results_dir>/all_results.json)')
    parser.add_argument('--patch', action='store_true', default=True,
                        help='Patch existing all_results.json instead of overwriting (default: True)')
    parser.add_argument('--no_patch', dest='patch', action='store_false',
                        help='Overwrite the existing all_results.json entry completely')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show which checkpoints would be evaluated, then exit')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_json    = Path(args.output_json) if args.output_json \
                  else results_dir / 'all_results.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Results dir: {results_dir}")
    print(f"Output JSON: {out_json}")

    # ── Build experiment → checkpoint map ────────────────────────────────────
    if args.ckpt:
        # User supplied an explicit checkpoint path
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            print(f"ERROR: checkpoint not found: {ckpt_path}")
            sys.exit(1)
        # Derive exp_id
        if args.experiment:
            exp_id = args.experiment
        else:
            ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            exp_id    = ckpt_data.get('config', {}).get('experiment', {}).get('name', ckpt_path.stem)
        checkpoint_map = {exp_id: ckpt_path}
    else:
        checkpoint_map = discover_checkpoints(results_dir)

    if not checkpoint_map:
        print("\nNo checkpoints found. Make sure *_best.pt files are in --results_dir.")
        sys.exit(1)

    # Filter by --experiment if specified
    if args.experiment and not args.ckpt:
        if args.experiment not in checkpoint_map:
            print(f"\nERROR: No checkpoint found for '{args.experiment}'")
            print(f"Available checkpoints: {list(checkpoint_map.keys())}")
            sys.exit(1)
        checkpoint_map = {args.experiment: checkpoint_map[args.experiment]}

    # ── Dry-run ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Checkpoints to evaluate ({len(checkpoint_map)} total):")
    for eid, pt in sorted(checkpoint_map.items()):
        size_mb = pt.stat().st_size / 1024**2
        print(f"    [{eid}]  →  {pt}  ({size_mb:.1f} MB)")
    print(f"{'='*65}")

    if args.dry_run:
        print("\nDry-run mode — exiting without evaluation.")
        return

    # ── Load existing results for patching ────────────────────────────────────
    existing_results = {}
    if args.patch and out_json.exists():
        try:
            with open(out_json) as f:
                existing_results = json.load(f)
            print(f"\nLoaded {len(existing_results)} existing entries from {out_json}")
        except Exception as e:
            print(f"WARNING: could not read {out_json}: {e}")

    # ── Evaluate each checkpoint ──────────────────────────────────────────────
    updated = []
    for exp_id, ckpt_path in sorted(checkpoint_map.items()):
        try:
            result = evaluate_checkpoint(exp_id, ckpt_path, device)

            if args.patch and exp_id in existing_results:
                # Merge: keep existing fields, overwrite with fresh metrics
                merged = copy.deepcopy(existing_results[exp_id])
                merged.update(result)
                existing_results[exp_id] = merged
            else:
                existing_results[exp_id] = result

            updated.append(exp_id)

        except Exception as exc:
            import traceback
            print(f"\n  [{exp_id}] FAILED: {exc}")
            traceback.print_exc()

    # ── Write results ─────────────────────────────────────────────────────────
    if updated:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(existing_results, f, indent=2)
        print(f"\n✓ all_results.json updated for: {', '.join(updated)}")
        print(f"  Saved to: {out_json}")
    else:
        print("\nNo experiments were updated (all failed or none found).")


if __name__ == '__main__':
    main()
