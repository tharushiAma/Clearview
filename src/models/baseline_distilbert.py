# src/models/baseline_distilbert.py

import argparse
import torch
import torch.nn as nn
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import random
import numpy as np
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------
# Dataset class
# ---------------------------
class ABSADataset(Dataset):
    """
    Custom PyTorch Dataset for multi-aspect sentiment analysis
    """

    def __init__(self, df, tokenizer, aspects, max_len=256):
        self.texts = df["text_clean"].tolist()
        self.labels = df[aspects].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize text for DistilBERT
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Convert labels: negative=0, neutral=1, positive=2, None=-100
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        labels = []
        for l in self.labels[idx]:
            labels.append(label_map[l] if l in label_map else -100)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

# ---------------------------
# Model
# ---------------------------
class MultiAspectDistilBERT(nn.Module):
    """
    DistilBERT with one classification head per aspect
    """

    def __init__(self, num_aspects, num_classes=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)

        # One classifier per aspect
        self.classifiers = nn.ModuleList([
            nn.Linear(self.bert.config.hidden_size, num_classes)
            for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask):
        # BERT forward pass
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)

        # Apply each aspect head
        logits = [clf(pooled) for clf in self.classifiers]
        return logits

# ---------------------------
# Training + Evaluation
# ---------------------------
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0.0
    steps = 0

    # Wrap dataloader with tqdm
    for batch in tqdm(dataloader, desc="Training Batch", leave=False):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)  # shape: [B, num_aspects]

        logits_list = model(input_ids, attention_mask)  # list of [B, 3]

        batch_loss = None

        for aspect_idx, logits in enumerate(logits_list):
            # If all labels are -100 for this aspect in this batch, skip
            if (labels[:, aspect_idx] != -100).any():
                loss_i = loss_fn(logits, labels[:, aspect_idx])
                batch_loss = loss_i if batch_loss is None else (batch_loss + loss_i)

        # If absolutely no aspect labels exist in this batch, skip safely
        if batch_loss is None:
            continue

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        steps += 1

    return total_loss / max(steps, 1)


def evaluate(model, dataloader, device, aspects, project_dir):
    model.eval()
    preds = {a: [] for a in aspects}
    trues = {a: [] for a in aspects}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits_list = model(input_ids, attention_mask)

            for aspect_idx, a in enumerate(aspects):
                p = torch.argmax(logits_list[aspect_idx], dim=1)
                preds[a].extend(p.cpu().tolist())
                trues[a].extend(labels[:, aspect_idx].cpu().tolist())

    metrics_path = f"{project_dir}/outputs/reports/distilbert_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for a in aspects:
            valid = [k for k, y in enumerate(trues[a]) if y != -100]
            if not valid:
                f.write(f"\nAspect: {a}\nNo valid samples in validation set.\n")
                continue

            y_true = [trues[a][k] for k in valid]
            y_pred = [preds[a][k] for k in valid]

            report = classification_report(
                y_true, y_pred,
                labels=[0, 1, 2],
                target_names=["negative", "neutral", "positive"],
                zero_division=0
            )

            print(f"\nAspect: {a}\n{report}")
            f.write(f"\nAspect: {a}\n{report}\n")
            f.write("-" * 50 + "\n")

    print(f"Report saved to {metrics_path}")

    # Save predictions CSV (aligned because val_dl shuffle=False)
    inv_map = {0: "negative", 1: "neutral", 2: "positive", -100: "None"}
    texts = dataloader.dataset.texts

    out_rows = []
    for idx in range(len(texts)):
        row = {"text": texts[idx]}
        for a in aspects:
            row[f"{a}_true"] = inv_map.get(trues[a][idx], "None")
            row[f"{a}_pred"] = inv_map.get(preds[a][idx], "None")
        out_rows.append(row)

    out_csv = f"{project_dir}/outputs/reports/distilbert_predictions.csv"
    pd.DataFrame(out_rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Predictions saved to {out_csv}")

# ---------------------------
# Main
# ---------------------------
def main(project_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    print(f"Using device: {device}")

    aspects = [
        "stayingpower","texture","smell",
        "price","colour","shipping","packing"
    ]

    train_df = pd.read_parquet(f"{project_dir}/data/splits/train.parquet")
    val_df   = pd.read_parquet(f"{project_dir}/data/splits/val.parquet")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Calculate class weights for evidence
    # Calculate class weights for evidence
    print("\nEvidence of Class Imbalance (Training Set):")
    with open(f"{project_dir}/outputs/reports/class_distribution.txt", "w") as f:
        f.write("Class Distribution in Training Set:\n")
        for a in aspects:
            counts = train_df[a].value_counts().to_dict()
            line = f"Aspect '{a}': {counts}"
            print(line)
            f.write(line + "\n")
    print(f"Class distribution saved to {project_dir}/outputs/reports/class_distribution.txt")

    train_ds = ABSADataset(train_df, tokenizer, aspects)
    val_ds   = ABSADataset(val_df, tokenizer, aspects)

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)


    model = MultiAspectDistilBERT(num_aspects=len(aspects)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Train
    for epoch in range(3):
        loss = train_epoch(model, train_dl, optimizer, device)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f}")

    # Save baseline model checkpoint
        torch.save(
            model.state_dict(),
            f"{project_dir}/outputs/checkpoints/baseline_distilbert.pt"
        )
        print("Baseline checkpoint saved to outputs/checkpoints/baseline_distilbert.pt")

    # Evaluate
    evaluate(model, val_dl, device, aspects, project_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", required=True)
    args = parser.parse_args()
    main(args.project_dir)
