# train_roberta_improved.py
# Improved Training Script for RoBERTa Hierarchical Model
# Includes: Synthetic data integration + sampler fix + MSR training fix

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizerFast
from sklearn.metrics import f1_score

from roberta_hierarchical_improved import ImprovedRoBERTaHierarchical


# ============================================================================
# DATASET
# ============================================================================

class AspectAwareDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, aspects, text_col="text", max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.aspects = aspects
        self.max_len = max_len

        self.texts = self.df[text_col].astype(str).tolist()
        self.labels = self.df[aspects].astype(str).values.tolist()

        # distribution counts (aspect_idx -> sentiment_id -> count)
        self.aspect_sentiment_counts = defaultdict(lambda: defaultdict(int))
        self._build_counts()

    def _build_counts(self):
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        for row in self.labels:
            for a_idx, l in enumerate(row):
                if l in label_map:
                    self.aspect_sentiment_counts[a_idx][label_map[l]] += 1

    def get_sample_weights(self):
        """
        Weight each sample by rarest (aspect, sentiment) pair in that sample.
        BUG FIX: if a row has no valid labels -> use weight 1.0.
        """
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        weights = []
        for row in self.labels:
            min_count = float("inf")
            for a_idx, l in enumerate(row):
                if l in label_map:
                    sid = label_map[l]
                    count = self.aspect_sentiment_counts[a_idx][sid]
                    min_count = min(min_count, count)

            # BUG FIX: inf -> fallback
            if min_count == float("inf"):
                weight = 1.0
            else:
                weight = 1.0 / (min_count + 1)

            weights.append(weight)
        return weights

    def compute_conflict_label(self, label_ids):
        """
        UNIFIED MSR DEFINITION:
        MIXED = review has ≥2 labeled aspects AND ≥2 different sentiment classes.
        Otherwise → CLEAR.
        
        label_ids: list in {0,1,2} or -100
        0=negative, 1=neutral, 2=positive
        """
        valid = [x for x in label_ids if x != -100]
        if len(valid) < 2:
            return 0  # CLEAR: not enough labeled aspects
        unique_sentiments = len(set(valid))
        return 1 if unique_sentiments >= 2 else 0  # MIXED if ≥2 different sentiments

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        labels = []
        for l in self.labels[idx]:
            labels.append(label_map[l] if l in label_map else -100)

        conflict_label = self.compute_conflict_label(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
            "conflict_label": torch.tensor(conflict_label, dtype=torch.float)
        }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, dataloader, device, aspects):
    model.eval()
    all_preds = []
    all_labels = []

    all_conf_scores = []
    all_conf_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            preds, _, conflict_scores = model.predict(input_ids, attention_mask)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            if "conflict_label" in batch:
                all_conf_scores.append(conflict_scores.cpu())
                all_conf_labels.append(batch["conflict_label"].cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    results = {}
    f1_scores = []

    for i, aspect in enumerate(aspects):
        valid_mask = (all_labels[:, i] != -100)
        if valid_mask.sum() == 0:
            continue

        y_true = all_labels[valid_mask, i]
        y_pred = all_preds[valid_mask, i]

        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        results[aspect] = {"f1_macro": float(f1_macro), "f1_weighted": float(f1_weighted)}
        f1_scores.append(f1_macro)

    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    conflict_metrics = None
    if all_conf_scores and all_conf_labels:
        conf_scores = torch.cat(all_conf_scores, dim=0).numpy()
        conf_true = torch.cat(all_conf_labels, dim=0).numpy().astype(int)
        conf_pred = (conf_scores >= 0.5).astype(int)

        conf_f1_macro = f1_score(conf_true, conf_pred, average="macro", zero_division=0)
        conf_f1_mixed = f1_score(conf_true, conf_pred, pos_label=1, average="binary", zero_division=0)

        conflict_metrics = {
            "conf_f1_macro": float(conf_f1_macro),
            "conf_f1_mixed": float(conf_f1_mixed),
        }

    return results, avg_f1, conflict_metrics


# ============================================================================
# TRAIN
# ============================================================================

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    aspects = ["stayingpower", "texture", "smell", "price", "colour", "shipping", "packing"]

    # -------- Load Data --------
    # Expected columns: text + aspects columns
    if args.use_synthetic:
        train_path = args.train_aug_path
        print(f"Loading augmented training data: {train_path}")
    else:
        train_path = args.train_path
        print(f"Loading original training data: {train_path}")

    val_path = args.val_path
    print(f"Loading validation data: {val_path}")

    train_df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    val_df = pd.read_parquet(val_path) if val_path.endswith(".parquet") else pd.read_csv(val_path)

    # -------- Tokenizer --------
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    # -------- Dataset & Sampler --------
    train_ds = AspectAwareDataset(train_df, tokenizer, aspects, text_col=args.text_col, max_len=args.max_len)
    val_ds = AspectAwareDataset(val_df, tokenizer, aspects, text_col=args.text_col, max_len=args.max_len)

    if args.use_sampler:
        weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # -------- Model --------
    model = ImprovedRoBERTaHierarchical(
        num_aspects=len(aspects),
        num_classes=3,
        aspect_names=aspects,
        hidden_dropout=args.dropout,
        msr_strength=args.msr_strength,
        roberta_name="roberta-base",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            conflict_labels = batch.get("conflict_label", None)
            if conflict_labels is not None:
                conflict_labels = conflict_labels.to(device)

            out = model(input_ids, attention_mask)

            # ✅ MSR training FIX: conflict labels included
            loss, loss_dict = model.compute_loss(
                out,
                labels,
                conflict_labels=conflict_labels,
                loss_weights={"aspect": 1.0, "conflict": args.conflict_weight}
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", conf=f"{loss_dict['conflict']:.4f}")

        avg_train = total_loss / max(len(train_loader), 1)

        # -------- Eval --------
        val_results, val_f1, val_conf = evaluate(model, val_loader, device, aspects)

        print("\n" + "=" * 80)
        print(f"Epoch {epoch} done | TrainLoss={avg_train:.4f} | ValMacroF1={val_f1:.4f}")
        if val_conf is not None:
            print(f"ConflictF1(macro)={val_conf['conf_f1_macro']:.4f} | MIXED_F1={val_conf['conf_f1_mixed']:.4f}")
        print("Per-aspect F1:")
        for asp, m in val_results.items():
            print(f"  {asp:12s} | macro={m['f1_macro']:.4f} | weighted={m['f1_weighted']:.4f}")

        # Save best
        if val_f1 > best_val:
            best_val = val_f1
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to: {save_path}")

    print(f"\nTraining finished. Best ValMacroF1={best_val:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_path", type=str, default="train.parquet")
    ap.add_argument("--train_aug_path", type=str, default="train_aug.parquet")
    ap.add_argument("--val_path", type=str, default="val.parquet")
    ap.add_argument("--text_col", type=str, default="text")

    ap.add_argument("--use_synthetic", action="store_true")
    ap.add_argument("--use_sampler", action="store_true")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--msr_strength", type=float, default=0.3)
    ap.add_argument("--conflict_weight", type=float, default=0.5)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./outputs")

    args = ap.parse_args()
    main(args)
