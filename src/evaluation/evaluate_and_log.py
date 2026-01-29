# src/evaluation/evaluate_and_log.py
"""
Saves evaluation matrices + metrics for:
1) ABSA (per-aspect confusion matrices, per-aspect macro/weighted F1, overall macro F1)
2) MSR (conflict head metrics, threshold, separation, error reduction BEFORE vs AFTER MSR)
3) XAI metrics (optional): IG faithfulness proxies (comprehensiveness/sufficiency), runtime

Usage examples:
Baseline (no MSR):
  python src/evaluation/evaluate_and_log.py --val_path data/splits/val.parquet --text_col text_clean \
    --ckpt outputs/exp_a_baseline_fixed_eval/best_model.pt --out_dir outputs/eval_baseline --msr_strength 0.0

MSR:
  python src/evaluation/evaluate_and_log.py --val_path data/splits/val.parquet --text_col text_clean \
    --ckpt outputs/exp_b_msr_fixed_er/best_model.pt --out_dir outputs/eval_msr --msr_strength 0.3

MSR + XAI (IG on small sample):
  python src/evaluation/evaluate_and_log.py --val_path data/splits/val.parquet --text_col text_clean \
    --ckpt outputs/exp_b_msr_fixed_er/best_model.pt --out_dir outputs/eval_msr_xai --msr_strength 0.3 \
    --run_xai --xai_samples 40 --xai_topk 10
"""

import os
import sys

# Add project root to sys.path: 2 levels up from src/evaluation/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
import time
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support

# Import your model (make sure PYTHONPATH includes src/models or run from project root)
from src.models.roberta_hierarchical_improved import ImprovedRoBERTaHierarchical

ASPECTS = ["stayingpower", "texture", "smell", "price", "colour", "shipping", "packing"]
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL = {0: "negative", 1: "neutral", 2: "positive"}


# -----------------------------
# Dataset
# -----------------------------
class EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, aspects: List[str], text_col="text", max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.aspects = aspects
        self.max_len = max_len
        self.texts = self.df[text_col].astype(str).tolist()
        self.labels = self.df[aspects].astype(str).values.tolist()

    @staticmethod
    def compute_conflict_label(label_ids: List[int]) -> int:
        valid = [x for x in label_ids if x != -100]
        if len(valid) < 2:
            return 0
        return 1 if len(set(valid)) >= 2 else 0

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
        conflict = self.compute_conflict_label(label_ids)

        return {
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "conflict_label": torch.tensor(conflict, dtype=torch.long),
        }


# -----------------------------
# Helper: conflict threshold
# -----------------------------
def best_threshold_for_mixed(conf_scores: np.ndarray, conf_true: np.ndarray) -> Tuple[float, float]:
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (conf_scores >= thr).astype(int)
        f1 = f1_score(conf_true, pred, pos_label=1, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)


# -----------------------------
# Core evaluation (SINGLE PASS)
# -----------------------------
@torch.no_grad()
def evaluate_single_pass(model, loader, device, msr_on: bool):
    model.eval()

    all_labels = []
    all_conf_true = []
    all_conf_scores = []

    all_preds_before = []
    all_preds_after = []

    all_probs_before = []
    all_probs_after = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        # BEFORE (MSR off)
        pb, prob_b, conf_b = model.predict(input_ids, attn, enable_msr=False)

        # AFTER (MSR on/off based on msr_on)
        pa, prob_a, conf_a = model.predict(input_ids, attn, enable_msr=msr_on)

        all_preds_before.append(pb.cpu().numpy())
        all_preds_after.append(pa.cpu().numpy())
        all_probs_before.append(prob_b.cpu().numpy())
        all_probs_after.append(prob_a.cpu().numpy())

        all_labels.append(labels)

        # conflict score should be taken from AFTER pass (same as reporting mode)
        all_conf_scores.append(conf_a.cpu().numpy())
        all_conf_true.append(batch["conflict_label"].cpu().numpy())

    preds_before = np.concatenate(all_preds_before, axis=0)
    preds_after = np.concatenate(all_preds_after, axis=0)
    probs_before = np.concatenate(all_probs_before, axis=0)  # [N,A,C]
    probs_after = np.concatenate(all_probs_after, axis=0)    # [N,A,C]
    labels_np = np.concatenate(all_labels, axis=0)           # [N,A]
    conf_scores = np.concatenate(all_conf_scores, axis=0)    # [N]
    conf_true = np.concatenate(all_conf_true, axis=0).astype(int)

    return preds_before, preds_after, probs_before, probs_after, labels_np, conf_scores, conf_true


# -----------------------------
# ABSA metrics + confusion matrices
# -----------------------------
def compute_absa_metrics(preds: np.ndarray, labels: np.ndarray, aspects: List[str]) -> Dict:
    per_aspect = {}
    macro_f1s = []

    for i, asp in enumerate(aspects):
        mask = labels[:, i] != -100
        if mask.sum() == 0:
            continue

        y_true = labels[mask, i]
        y_pred = preds[mask, i]

        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        p, r, f1_each, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])

        per_aspect[asp] = {
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "per_class": {
                "negative": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f1_each[0])},
                "neutral":  {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f1_each[1])},
                "positive": {"precision": float(p[2]), "recall": float(r[2]), "f1": float(f1_each[2])},
            },
            "confusion_matrix": cm.tolist(),  # 3x3
            "labels_order": ["negative", "neutral", "positive"]
        }

        macro_f1s.append(f1_macro)

    overall_macro = float(np.mean(macro_f1s)) if macro_f1s else 0.0
    return {"overall_macro_f1": overall_macro, "per_aspect": per_aspect}


# -----------------------------
# MSR error reduction
# -----------------------------
def msr_error_reduction(preds_before, preds_after, labels, aspects):
    report = {}
    total = 0
    for i, asp in enumerate(aspects):
        mask = labels[:, i] != -100
        if mask.sum() == 0:
            report[asp] = {"before": 0, "after": 0, "reduction": 0}
            continue
        before_err = int((preds_before[mask, i] != labels[mask, i]).sum())
        after_err = int((preds_after[mask, i] != labels[mask, i]).sum())
        red = before_err - after_err
        total += red
        report[asp] = {"before": before_err, "after": after_err, "reduction": red}
    return report, total


# -----------------------------
# XAI metrics (Integrated Gradients faithfulness proxy)
# -----------------------------
def run_ig_faithfulness(model, tokenizer, texts: List[str], device, max_len: int, topk: int = 10):
    """
    Comprehensiveness: remove top-k tokens -> prob drop (bigger drop => better explanation)
    Sufficiency: keep only top-k tokens -> prob retained (higher retained prob => better)
    NOTE: This is a practical proxy metric; works without human rationales.
    """
    try:
        from captum.attr import IntegratedGradients
    except Exception as e:
        raise RuntimeError("captum not installed or import failed") from e

    model.eval()

    def forward_func(input_ids, attention_mask, aspect_idx: int, enable_msr: bool):
        out = model(input_ids=input_ids, attention_mask=attention_mask, enable_msr=enable_msr)
        logits = out["aspect_logits"][aspect_idx]  # [B,3]
        probs = torch.softmax(logits, dim=-1)
        return probs

    ig = IntegratedGradients(lambda ids, mask, aidx: forward_func(ids, mask, aidx, enable_msr=True))

    results = []
    t0 = time.time()

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        # Base predictions
        preds_after, probs_after, _ = model.predict(input_ids, attn, enable_msr=True)
        probs_after = probs_after.squeeze(0)  # [A,3]
        preds_after = preds_after.squeeze(0)  # [A]

        for aidx, aspect in enumerate(ASPECTS):
            pred_cls = int(preds_after[aidx].item())
            base_prob = float(probs_after[aidx, pred_cls].item())

            # IG attributions for predicted class
            # Captum expects additional args; we supply aspect index via closure parameter
            attributions = ig.attribute(
                inputs=input_ids,
                additional_forward_args=(attn, aidx),
                target=pred_cls
            )  # [1, T]
            attributions = attributions.squeeze(0)  # [T]

            # Token importance (ignore special/pad)
            token_ids = input_ids.squeeze(0).detach().cpu().numpy().tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            valid_positions = []
            for i, tok in enumerate(tokens):
                if tok in ["<s>", "</s>", "<pad>"]:
                    continue
                if attn.squeeze(0)[i].item() == 0:
                    continue
                valid_positions.append(i)

            if len(valid_positions) == 0:
                continue

            scores = [(i, float(abs(attributions[i].item()))) for i in valid_positions]
            scores.sort(key=lambda x: x[1], reverse=True)
            top_positions = [i for i, _ in scores[:min(topk, len(scores))]]

            # --- Comprehensiveness: remove top-k (mask them to <pad>)
            removed_ids = input_ids.clone()
            for pos in top_positions:
                removed_ids[0, pos] = tokenizer.pad_token_id
            pred_r, prob_r, _ = model.predict(removed_ids, attn, enable_msr=True)
            prob_removed = float(prob_r[0, aidx, pred_cls].item())
            comprehensiveness = base_prob - prob_removed  # bigger is better

            # --- Sufficiency: keep only top-k (mask all others)
            kept_ids = input_ids.clone()
            for pos in valid_positions:
                if pos not in top_positions:
                    kept_ids[0, pos] = tokenizer.pad_token_id
            pred_k, prob_k, _ = model.predict(kept_ids, attn, enable_msr=True)
            prob_kept = float(prob_k[0, aidx, pred_cls].item())
            sufficiency = prob_kept  # higher is better

            results.append({
                "aspect": aspect,
                "pred_class": INV_LABEL[pred_cls],
                "base_prob": base_prob,
                "prob_removed": prob_removed,
                "comprehensiveness_drop": comprehensiveness,
                "prob_kept": prob_kept,
                "sufficiency_prob": sufficiency
            })

    runtime_sec = time.time() - t0

    # Aggregate
    if len(results) == 0:
        return {"runtime_sec": runtime_sec, "num_records": 0}

    comp = [r["comprehensiveness_drop"] for r in results]
    suff = [r["sufficiency_prob"] for r in results]

    return {
        "runtime_sec": float(runtime_sec),
        "num_records": int(len(results)),
        "comprehensiveness_drop_mean": float(np.mean(comp)),
        "comprehensiveness_drop_std": float(np.std(comp)),
        "sufficiency_prob_mean": float(np.mean(suff)),
        "sufficiency_prob_std": float(np.std(suff)),
        "records": results[:200]  # keep first 200 to avoid giant json
    }


# -----------------------------
# Save utilities
# -----------------------------
def save_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_confusion_csv(out_dir: str, absa_metrics: Dict):
    rows = []
    for asp, d in absa_metrics["per_aspect"].items():
        cm = d["confusion_matrix"]  # 3x3
        # Flatten with labels
        rows.append({"aspect": asp, "cm": json.dumps(cm), "labels_order": json.dumps(d["labels_order"])})
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "confusion_matrices.csv"), index=False)


# -----------------------------
# Main
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_parquet(args.val_path) if args.val_path.endswith(".parquet") else pd.read_csv(args.val_path)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    ds = EvalDataset(df, tokenizer, ASPECTS, text_col=args.text_col, max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = ImprovedRoBERTaHierarchical(
        num_aspects=len(ASPECTS),
        num_classes=3,
        aspect_names=ASPECTS,
        hidden_dropout=args.dropout,
        msr_strength=args.msr_strength,
        roberta_name="roberta-base"
    ).to(device)

    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Evaluate (single pass)
    preds_before, preds_after, probs_before, probs_after, labels_np, conf_scores, conf_true = evaluate_single_pass(
        model, loader, device, msr_on=(args.msr_strength > 0)
    )

    # ABSA metrics for AFTER
    absa = compute_absa_metrics(preds_after, labels_np, ASPECTS)

    # Conflict metrics
    best_thr, best_mixed_f1 = best_threshold_for_mixed(conf_scores, conf_true)
    conf_pred = (conf_scores >= best_thr).astype(int)
    conf_f1_macro = f1_score(conf_true, conf_pred, average="macro", zero_division=0)
    conf_mixed_f1 = f1_score(conf_true, conf_pred, pos_label=1, average="binary", zero_division=0)

    clear_mean = float(conf_scores[conf_true == 0].mean()) if (conf_true == 0).any() else 0.0
    mixed_mean = float(conf_scores[conf_true == 1].mean()) if (conf_true == 1).any() else 0.0
    separation = mixed_mean - clear_mean

    conflict = {
        "best_threshold": best_thr,
        "mixed_f1_at_best_threshold": float(best_mixed_f1),
        "conf_f1_macro": float(conf_f1_macro),
        "mixed_f1": float(conf_mixed_f1),
        "clear_mean_score": clear_mean,
        "mixed_mean_score": mixed_mean,
        "separation": float(separation),
        "mixed_rate_true": float(conf_true.mean()),
        "mixed_rate_pred": float(conf_pred.mean()),
    }

    # MSR reduction
    msr_report, total_red = msr_error_reduction(preds_before, preds_after, labels_np, ASPECTS)

    # Save main report
    report = {
        "run": {
            "ckpt": args.ckpt,
            "val_path": args.val_path,
            "text_col": args.text_col,
            "msr_strength": args.msr_strength
        },
        "absa": absa,
        "conflict": conflict,
        "msr_error_reduction": {
            "per_aspect": msr_report,
            "total_reduction": int(total_red)
        }
    }

    os.makedirs(args.out_dir, exist_ok=True)
    save_json(os.path.join(args.out_dir, "report.json"), report)
    save_confusion_csv(args.out_dir, absa)

    # Optional: save per-sample predictions (small)
    if args.save_predictions:
        pred_rows = []
        for i in range(min(len(df), args.pred_limit)):
            row = {"text": ds.texts[i], "conf_score": float(conf_scores[i])}
            for a_idx, asp in enumerate(ASPECTS):
                row[f"{asp}_pred_before"] = INV_LABEL[int(preds_before[i, a_idx])]
                row[f"{asp}_pred_after"]  = INV_LABEL[int(preds_after[i, a_idx])]
                gold = labels_np[i, a_idx]
                row[f"{asp}_gold"] = INV_LABEL[int(gold)] if gold != -100 else "NA"
            pred_rows.append(row)
        pd.DataFrame(pred_rows).to_csv(os.path.join(args.out_dir, "predictions_sample.csv"), index=False)

    # Optional: XAI metrics (IG)
    if args.run_xai:
        sample_texts = ds.texts[:min(args.xai_samples, len(ds))]
        xai = run_ig_faithfulness(
            model=model,
            tokenizer=tokenizer,
            texts=sample_texts,
            device=device,
            max_len=args.max_len,
            topk=args.xai_topk
        )
        save_json(os.path.join(args.out_dir, "xai_ig_metrics.json"), xai)

    print(f"\n✅ Saved evaluation outputs to: {args.out_dir}")
    print(f"ABSA overall macro F1 (AFTER) = {absa['overall_macro_f1']:.4f}")
    print(f"Conflict macro F1 = {conflict['conf_f1_macro']:.4f} | MIXED_F1 = {conflict['mixed_f1']:.4f} | sep={conflict['separation']:.4f}")
    print(f"MSR total error reduction = {total_red:+d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--val_path", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--text_col", type=str, default="text_clean")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--msr_strength", type=float, default=0.0)

    ap.add_argument("--save_predictions", action="store_true")
    ap.add_argument("--pred_limit", type=int, default=200)

    # XAI (optional)
    ap.add_argument("--run_xai", action="store_true")
    ap.add_argument("--xai_samples", type=int, default=40)
    ap.add_argument("--xai_topk", type=int, default=10)

    args = ap.parse_args()
    main(args)
