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

ASPECTS = ["stayingpower", "texture", "smell", "price", "colour", "shipping", "packing"]
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


class AspectAwareDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, aspects, text_col="text", max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.aspects = aspects
        self.max_len = max_len

        self.texts = self.df[text_col].astype(str).tolist()
        self.labels = self.df[aspects].astype(str).values.tolist()

        self.counts = defaultdict(int)
        self._build_counts()

    def _build_counts(self):
        for row in self.labels:
            for a_idx, l in enumerate(row):
                if l in LABEL_MAP:
                    self.counts[(a_idx, LABEL_MAP[l])] += 1

    def compute_conflict_label(self, label_ids):
        valid = [x for x in label_ids if x != -100]
        if len(valid) < 2:
            return 0
        return 1 if len(set(valid)) >= 2 else 0

    def get_sample_weights(self):
        weights = []
        for row in self.labels:
            min_count = float("inf")
            has_any = False
            for a_idx, l in enumerate(row):
                if l in LABEL_MAP:
                    has_any = True
                    c = self.counts[(a_idx, LABEL_MAP[l])]
                    min_count = min(min_count, c)
            weights.append(1.0 if (not has_any or min_count == float("inf")) else 1.0 / (min_count + 1))
        return weights

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

        label_ids = [(LABEL_MAP[l] if l in LABEL_MAP else -100) for l in self.labels[idx]]
        conflict_label = self.compute_conflict_label(label_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "conflict_label": torch.tensor(conflict_label, dtype=torch.float)
        }


def compute_aspect_metrics(preds, labels, aspects):
    results = {}
    f1s = []
    for i, aspect in enumerate(aspects):
        mask = labels[:, i] != -100
        if mask.sum() == 0:
            continue
        y_true = labels[mask, i]
        y_pred = preds[mask, i]
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        results[aspect] = {"f1_macro": float(f1_macro), "f1_weighted": float(f1_weighted)}
        f1s.append(f1_macro)
    return results, float(np.mean(f1s)) if f1s else 0.0


def best_threshold_for_mixed(conf_scores, conf_true):
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (conf_scores >= thr).astype(int)
        f1 = f1_score(conf_true, pred, pos_label=1, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)


def msr_error_reduction(preds_before, preds_after, labels, aspects):
    report = {}
    for i, asp in enumerate(aspects):
        mask = labels[:, i] != -100
        if mask.sum() == 0:
            report[asp] = (0, 0, 0)
            continue
        before_err = int((preds_before[mask, i] != labels[mask, i]).sum())
        after_err = int((preds_after[mask, i] != labels[mask, i]).sum())
        report[asp] = (before_err, after_err, before_err - after_err)
    return report


def evaluate_single_pass(model, dataloader, device, aspects):
    """
    FIXED: single pass ensures preds_before/preds_after/labels are perfectly aligned.
    """
    model.eval()

    all_before, all_after, all_labels = [], [], []
    all_conf_scores, all_conf_true = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            pb, _, _ = model.predict(input_ids, attention_mask, enable_msr=False)
            pa, _, conf = model.predict(input_ids, attention_mask, enable_msr=True)

            all_before.append(pb.cpu())
            all_after.append(pa.cpu())
            all_labels.append(labels.cpu())

            all_conf_scores.append(conf.cpu())
            all_conf_true.append(batch["conflict_label"].cpu())

    preds_before = torch.cat(all_before, dim=0).numpy()
    preds_after = torch.cat(all_after, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()

    conf_scores = torch.cat(all_conf_scores, dim=0).numpy()
    conf_true = torch.cat(all_conf_true, dim=0).numpy().astype(int)

    aspect_results, val_macro_f1 = compute_aspect_metrics(preds_after, labels_np, aspects)

    best_thr, _ = best_threshold_for_mixed(conf_scores, conf_true)
    conf_pred = (conf_scores >= best_thr).astype(int)

    conf_f1_macro = f1_score(conf_true, conf_pred, average="macro", zero_division=0)
    conf_f1_mixed = f1_score(conf_true, conf_pred, pos_label=1, average="binary", zero_division=0)

    clear_mean = float(conf_scores[conf_true == 0].mean()) if (conf_true == 0).any() else 0.0
    mixed_mean = float(conf_scores[conf_true == 1].mean()) if (conf_true == 1).any() else 0.0
    separation = mixed_mean - clear_mean

    conflict_metrics = {
        "conf_f1_macro": float(conf_f1_macro),
        "conf_f1_mixed": float(conf_f1_mixed),
        "best_thr": float(best_thr),
        "clear_mean": clear_mean,
        "mixed_mean": mixed_mean,
        "separation": float(separation),
    }

    return aspect_results, val_macro_f1, conflict_metrics, preds_before, preds_after, labels_np


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_path = args.train_aug_path if args.use_synthetic else args.train_path
    print(f"Loading {'augmented' if args.use_synthetic else 'original'} training data: {train_path}")
    print(f"Loading validation data: {args.val_path}")

    train_df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    val_df = pd.read_parquet(args.val_path) if args.val_path.endswith(".parquet") else pd.read_csv(args.val_path)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    train_ds = AspectAwareDataset(train_df, tokenizer, ASPECTS, text_col=args.text_col, max_len=args.max_len)
    val_ds = AspectAwareDataset(val_df, tokenizer, ASPECTS, text_col=args.text_col, max_len=args.max_len)

    if args.use_sampler:
        weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ImprovedRoBERTaHierarchical(
        num_aspects=len(ASPECTS),
        num_classes=3,
        aspect_names=ASPECTS,
        hidden_dropout=args.dropout,
        msr_strength=args.msr_strength,
        roberta_name="roberta-base"
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = -1.0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            conflict_labels = batch["conflict_label"].to(device)

            out = model(input_ids, attention_mask, enable_msr=True)
            loss, loss_dict = model.compute_loss(
                out, labels,
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

        val_results, val_f1, val_conf, preds_before, preds_after, labels_np = evaluate_single_pass(
            model, val_loader, device, ASPECTS
        )

        er = msr_error_reduction(preds_before, preds_after, labels_np, ASPECTS)

        print("\n" + "=" * 80)
        print(f"Epoch {epoch} done | TrainLoss={avg_train:.4f} | ValMacroF1={val_f1:.4f}")
        print(f"ConflictF1(macro)={val_conf['conf_f1_macro']:.4f} | MIXED_F1={val_conf['conf_f1_mixed']:.4f}")
        print(f"Best Threshold={val_conf['best_thr']:.2f}")
        print(f"CLEAR mean score={val_conf['clear_mean']:.4f} | MIXED mean score={val_conf['mixed_mean']:.4f} | Separation={val_conf['separation']:.4f}")

        if val_conf["separation"] < 0.05:
            print("⚠️  WARNING: Conflict head shows poor separation between CLEAR and MIXED!")

        print("Per-aspect F1:")
        for asp, m in val_results.items():
            print(f"  {asp:12s} | macro={m['f1_macro']:.4f} | weighted={m['f1_weighted']:.4f}")

        print("\nMSR Error Reduction:")
        total_red = 0
        for asp in ASPECTS:
            b, a, r = er[asp]
            total_red += r
            print(f"  {asp:12s} | before={b:3d} | after={a:3d} | reduction={r:+3d}")
        print(f"  {'TOTAL':12s} | reduction={total_red:+d}")

        if val_f1 > best_val:
            best_val = val_f1
            patience_left = args.patience
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to: {save_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping triggered after {epoch} epochs")
                break

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
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--msr_strength", type=float, default=0.3)
    ap.add_argument("--conflict_weight", type=float, default=0.5)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--patience", type=int, default=1)

    args = ap.parse_args()
    main(args)
