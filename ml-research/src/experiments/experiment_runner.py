"""
experiment_runner.py
Unified runner for all ablation studies and baseline comparisons.

Usage:
    # Run a specific experiment
    python src/experiments/experiment_runner.py --experiment A1_no_gcn

    # Run all ablations
    python src/experiments/experiment_runner.py --group ablations

    # Run all baselines
    python src/experiments/experiment_runner.py --group baselines

    # Run everything
    python src/experiments/experiment_runner.py --group all

    # List all planned experiments
    python src/experiments/experiment_runner.py --list

    # Run from ml-research root directory!
"""

import os
import sys
import yaml
import json
import time
import argparse
import copy
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

# Make sure project root is on sys.path
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
SRC_DIR      = os.path.join(PROJECT_ROOT, 'src')
UTILS_DIR    = os.path.join(PROJECT_ROOT, 'utils')

for p in [PROJECT_ROOT, SRC_DIR, UTILS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ablation_configs import get_all_ablation_specs, get_all_baseline_specs, print_experiment_plan
from baseline_models import create_baseline, CrossEntropyLossWrapper, TFIDFSVMBaseline

from models.model import create_model
from models.losses import AspectSpecificLossManager
from utils.data_utils import create_dataloaders, DependencyParser, compute_class_weights
from utils.metrics import AspectSentimentEvaluator, MixedSentimentEvaluator
from transformers import RobertaTokenizer, BertTokenizer, get_linear_schedule_with_warmup


# ─────────────────────────────────────────────────────────────────────────────
# Shared Experiment Result Structure
# ─────────────────────────────────────────────────────────────────────────────
def empty_result(exp_id: str, desc: str) -> dict:
    return {
        'experiment_id': exp_id,
        'description':   desc,
        'status':        'pending',
        'error':         None,
        'duration_mins': None,
        'overall':       {},
        'per_aspect':    {},
        'mixed_sentiment': {}
    }


# ─────────────────────────────────────────────────────────────────────────────
# Deep Learning Trainer (reusable for full model and baselines)
# ─────────────────────────────────────────────────────────────────────────────
class ExperimentTrainer:
    """
    Lightweight trainer for running ablation / baseline experiments.
    Reuses the same training loop but accepts any nn.Module compatible model.
    """
    def __init__(self, exp_id: str, config: dict, model: torch.nn.Module,
                 loss_manager, tokenizer, results_dir: Path):
        self.exp_id       = exp_id
        self.config       = config
        self.model        = model
        self.loss_manager = loss_manager
        self.results_dir  = results_dir
        self.tokenizer    = tokenizer

        self.device       = torch.device(
            config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

        dep_parser = None
        if config['data'].get('use_dependency_parsing', False):
            dep_parser = DependencyParser(config['data'].get('language', 'en'))

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            config, tokenizer, dep_parser
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
        )
        num_steps = len(self.train_loader) * config['training']['num_epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=num_steps,
        )
        self.use_amp = config['hardware'].get('mixed_precision', False) and torch.cuda.is_available()
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()

        self.evaluator        = AspectSentimentEvaluator(config['aspects']['names'])
        self.best_val_metric  = 0
        self.patience_counter = 0
        self.patience         = config['training']['early_stopping_patience']
        self.global_step      = 0

    def _forward(self, batch):
        """Run forward pass regardless of model type (handles aspect_id arg)."""
        input_ids      = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        aspect_ids     = batch['aspect_ids'].to(self.device)

        edge_indices = None
        if self.config['model'].get('use_dependency_gcn', False):
            edge_indices = [e.to(self.device) if e is not None else None
                           for e in batch['edge_indices']]

        return self.model(input_ids, attention_mask, aspect_ids, edge_indices)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        from tqdm import tqdm

        for batch in tqdm(self.train_loader, desc=f"Training", leave=False):
            aspect_ids = batch['aspect_ids'].to(self.device)
            labels     = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    preds = self._forward(batch)
                    # Handle models returning tuples (aspect-aware model returns (preds, attn, repr))
                    if isinstance(preds, tuple):
                        preds = preds[0]
                    loss, _ = self.loss_manager.compute_loss(
                        preds, labels, aspect_ids, self.config['aspects']['names']
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['training']['max_grad_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self._forward(batch)
                if isinstance(preds, tuple):
                    preds = preds[0]
                loss, _ = self.loss_manager.compute_loss(
                    preds, labels, aspect_ids, self.config['aspects']['names']
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['training']['max_grad_norm']
                )
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()
            self.global_step += 1

        return total_loss / max(len(self.train_loader), 1)

    def evaluate(self, loader) -> dict:
        self.model.eval()
        all_preds, all_labels, all_aspects = [], [], []

        with torch.no_grad():
            from tqdm import tqdm
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                preds = self._forward(batch)
                if isinstance(preds, tuple):
                    preds = preds[0]
                pred_classes = torch.argmax(preds, dim=1).cpu().numpy()
                all_preds.extend(pred_classes)
                all_labels.extend(batch['labels'].numpy())
                all_aspects.extend(batch['aspects'])

        aspect_metrics = {}
        for aspect in self.config['aspects']['names']:
            mask  = np.array([a == aspect for a in all_aspects])
            if mask.sum() == 0:
                continue
            y_true = np.array(all_labels)[mask]
            y_pred = np.array(all_preds)[mask]
            aspect_metrics[aspect] = self.evaluator.evaluate_aspect(y_true, y_pred, aspect)

        overall = self.evaluator.evaluate_aspect(
            np.array(all_labels), np.array(all_preds), 'overall'
        )
        return {'overall': overall, 'aspects': aspect_metrics}

    def train(self) -> dict:
        print(f"\n[{self.exp_id}] Training for {self.config['training']['num_epochs']} epochs")

        t0 = time.time()
        best_ckpt = self.results_dir / f'{self.exp_id}_best.pt'
        
        if best_ckpt.exists():
            print(f"  Checkpoint {best_ckpt} found! Skipping training.")
        else:
            for epoch in range(self.config['training']['num_epochs']):
                train_loss = self.train_epoch()
                val_metrics = self.evaluate(self.val_loader)
                val_f1 = val_metrics['overall']['macro_f1']

                print(f"  Epoch {epoch+1}: loss={train_loss:.4f}  val_macro_f1={val_f1:.4f}  "
                      f"patience={self.patience_counter}/{self.patience}")

                if val_f1 > self.best_val_metric:
                    self.best_val_metric = val_f1
                    self.patience_counter = 0
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'config': self.config,
                    }, best_ckpt)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break

        # Load best and evaluate on test
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])

        test_metrics = self.evaluate(self.test_loader)
        duration_mins = (time.time() - t0) / 60
        print(f"  Done in {duration_mins:.1f} min — test macro_f1={test_metrics['overall']['macro_f1']:.4f}")
        return test_metrics, duration_mins


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF + SVM Runner (classical — no GPU, no epochs)
# ─────────────────────────────────────────────────────────────────────────────
def run_tfidf_svm(exp_id: str, desc: str, config: dict, results_dir: Path) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score,
                                  matthews_corrcoef, precision_recall_fscore_support)

    t0 = time.time()
    label_map = config['aspects']['label_map']
    aspects   = config['aspects']['names']

    train_df = pd.read_csv(config['data']['train_path'])
    test_df  = pd.read_csv(config['data']['test_path'])

    svm = TFIDFSVMBaseline(aspect_names=aspects)
    svm.fit(train_df, label_map)
    svm.save(str(results_dir / exp_id))

    overall_true, overall_pred = [], []
    per_aspect = {}

    for aspect in aspects:
        mask = test_df[aspect].notna()
        if mask.sum() == 0:
            continue
        X_test = test_df.loc[mask, 'data'].astype(str).tolist()
        y_true = test_df.loc[mask, aspect].map(
            lambda v: label_map.get(str(v).lower(), -1)
        ).tolist()
        valid = [(x, y) for x, y in zip(X_test, y_true) if y != -1]
        if not valid:
            continue
        X_valid, y_valid = zip(*valid)

        y_pred = svm.predict(list(X_valid), aspect)
        y_valid = list(y_valid)

        acc = accuracy_score(y_valid, y_pred)
        p, r, f1, sup = precision_recall_fscore_support(
            y_valid, y_pred, average=None, labels=[0, 1, 2], zero_division=0
        )
        macro_f1 = f1_score(y_valid, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_valid, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_valid, y_pred)

        per_aspect[aspect] = {
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'mcc': float(mcc),
            'per_class_f1': [float(x) for x in f1],
            'per_class_precision': [float(x) for x in p],
            'per_class_recall': [float(x) for x in r],
            'support': [int(x) for x in sup],
        }
        overall_true.extend(y_valid)
        overall_pred.extend(y_pred.tolist())

    # Overall
    overall = {
        'accuracy': float(accuracy_score(overall_true, overall_pred)),
        'macro_f1': float(f1_score(overall_true, overall_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(overall_true, overall_pred, average='weighted', zero_division=0)),
        'mcc': float(matthews_corrcoef(overall_true, overall_pred)),
    } if overall_true else {}

    duration_mins = (time.time() - t0) / 60
    print(f"  [{exp_id}] Done in {duration_mins:.1f} min — overall macro_f1={overall.get('macro_f1', 0):.4f}")

    return {
        'experiment_id': exp_id,
        'description':   desc,
        'status':        'done',
        'duration_mins': round(duration_mins, 2),
        'overall':       overall,
        'per_aspect':    per_aspect,
        'mixed_sentiment': {},
        'error':         None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single Deep Learning Experiment Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_dl_experiment(exp_id: str, desc: str, config: dict,
                      results_dir: Path, base_config: dict) -> dict:
    """
    Instantiates the appropriate model and loss, trains, evaluates, and returns results.
    """
    result = empty_result(exp_id, desc)
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])

    try:
        # ── Pick tokenizer ──────────────────────────────────────────────────
        roberta_name = config['model']['roberta_model']
        if 'bert' in roberta_name and 'roberta' not in roberta_name:
            tokenizer = BertTokenizer.from_pretrained(roberta_name)
        else:
            tokenizer = RobertaTokenizer.from_pretrained(roberta_name)

        # ── Build model ─────────────────────────────────────────────────────
        # BUG FIX: Use explicit exp_id prefix check instead of substring
        # matching on the name (which could break if experiment names change).
        if exp_id.startswith('B1_'):
            model = create_baseline('plain_roberta', config)
        elif exp_id.startswith('B3_'):
            model = create_baseline('bert_base', config)
        else:
            # Full model or ablation variants
            model = create_model(config)

        # ── Build loss ─────────────────────────────────────────────────────
        if config['training'].get('use_ce_loss', False):
            loss_manager = CrossEntropyLossWrapper()
        else:
            aspect_class_counts = compute_class_weights(
                config['data']['train_path'],
                config['aspects']['names'],
                config['aspects']['label_map'],
            )
            loss_manager = AspectSpecificLossManager(aspect_class_counts, config['training'])

        # ── Train ──────────────────────────────────────────────────────────
        trainer = ExperimentTrainer(exp_id, config, model, loss_manager,
                                    tokenizer, results_dir)
        test_metrics, duration_mins = trainer.train()

        # Serialise (numpy types → Python)
        def ser(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)): return int(obj)
            if isinstance(obj, (np.float64, np.float32)): return float(obj)
            if isinstance(obj, dict): return {k: ser(v) for k, v in obj.items()}
            if isinstance(obj, list): return [ser(x) for x in obj]
            return obj

        result['status']        = 'done'
        result['duration_mins'] = round(duration_mins, 2)
        result['overall']       = ser(test_metrics['overall'])
        result['per_aspect']    = ser(test_metrics['aspects'])

        # ── A6: MSR Evaluation (Mixed Sentiment Resolution) ─────────────────
        # When the config has evaluate_msr=True (set by ablation_6_mixed_sentiment),
        # run MixedSentimentEvaluator on the test set to capture MSR-specific metrics.
        if config.get('experiment', {}).get('evaluate_msr', False):
            print(f"  [{exp_id}] Running Mixed Sentiment Resolution evaluation...")
            mixed_evaluator = MixedSentimentEvaluator(config['aspects']['names'])
            trainer.model.eval()
            review_true = {}
            review_pred = {}

            with torch.no_grad():
                for batch in trainer.test_loader:
                    input_ids      = batch['input_ids'].to(trainer.device)
                    attention_mask = batch['attention_mask'].to(trainer.device)
                    aspect_ids     = batch['aspect_ids'].to(trainer.device)

                    edge_indices = None
                    if config['model'].get('use_dependency_gcn', False):
                        edge_indices = [e.to(trainer.device) if e is not None else None
                                       for e in batch['edge_indices']]

                    preds = trainer.model(input_ids, attention_mask, aspect_ids, edge_indices)
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

            mixed_metrics = mixed_evaluator.evaluate_mixed_sentiment_resolution(
                review_true, review_pred
            )
            mixed_evaluator.print_mixed_sentiment_results(mixed_metrics)

            # Store key MSR scalars in result (JSON-serialisable)
            result['mixed_sentiment'] = {
                'mixed_review_count':    mixed_metrics.get('mixed_review_count', 0),
                'mixed_review_accuracy': mixed_metrics.get('mixed_review_accuracy', 0.0),
                'mixed_aspect_accuracy': mixed_metrics.get('mixed_aspect_accuracy', 0.0),
                'mixed_detection_rate':  mixed_metrics.get('mixed_detection_rate', 0.0),
            }

    except Exception as exc:
        import traceback
        result['status'] = 'error'
        result['error']  = traceback.format_exc()
        print(f"  [{exp_id}] FAILED: {exc}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_experiments(exp_ids: list, base_config: dict, results_dir: Path) -> dict:
    """Run selected experiments and collect results."""
    all_ablation_specs  = {k: (k, d, c) for k, d, c in get_all_ablation_specs(base_config)}
    all_baseline_specs  = {k: (k, d, c) for k, d, c in get_all_baseline_specs(base_config)}
    all_specs = {**all_ablation_specs, **all_baseline_specs}

    results = {}
    out_path = results_dir / 'all_results.json'
    if out_path.exists():
        try:
            with open(out_path, 'r') as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {out_path}.")
        except Exception as e:
            print(f"Could not load existing {out_path}: {e}")

    for exp_id in exp_ids:
        if exp_id not in all_specs:
            print(f"  Warning: unknown experiment '{exp_id}' — skipping")
            continue

        exp_id, desc, config = all_specs[exp_id]
        print(f"\n{'='*65}")
        print(f"Running: [{exp_id}]  {desc}")
        print(f"{'='*65}")

        if config.get('_baseline_type') == 'tfidf_svm':
            result = run_tfidf_svm(exp_id, desc, config, results_dir)
        else:
            result = run_dl_experiment(exp_id, desc, config, results_dir, base_config)

        results[exp_id] = result

        # Save after each experiment so intermediate results are not lost
        out_path = results_dir / 'all_results.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies and baseline comparisons')
    parser.add_argument('--config',     default='configs/config.yaml',
                        help='Base config file path')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Specific experiment ID to run (e.g. A1_no_gcn)')
    parser.add_argument('--group',      type=str, default=None,
                        choices=['ablations', 'baselines', 'all'],
                        help='Run a group of experiments')
    parser.add_argument('--list',       action='store_true',
                        help='List all available experiments and exit')
    parser.add_argument('--results_dir', default='results/experiments',
                        help='Directory to save results')
    args = parser.parse_args()

    # Load base config
    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    if args.list:
        print_experiment_plan(base_config)
        return

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_ablation_specs = get_all_ablation_specs(base_config)
    all_baseline_specs = get_all_baseline_specs(base_config)

    # Decide which experiments to run
    if args.experiment:
        exp_ids = [args.experiment]
    elif args.group == 'ablations':
        exp_ids = [s[0] for s in all_ablation_specs]
    elif args.group == 'baselines':
        exp_ids = [s[0] for s in all_baseline_specs]
    elif args.group == 'all':
        exp_ids = [s[0] for s in all_baseline_specs] + [s[0] for s in all_ablation_specs]
    else:
        print("Specify --experiment <id> or --group <ablations|baselines|all> or --list")
        parser.print_help()
        return

    results = run_experiments(exp_ids, base_config, results_dir)
    print(f"\nAll results saved to {results_dir / 'all_results.json'}")
    print("Run `python src/experiments/results_analyzer.py` to generate comparison tables.")


if __name__ == '__main__':
    main()
