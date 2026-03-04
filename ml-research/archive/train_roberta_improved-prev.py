# src/models/train_roberta_improved.py
# Training Script for Improved RoBERTa Hierarchical Model
# Supports MSR and Adaptive Focal Loss

import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from transformers import RobertaTokenizerFast
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import WeightedRandomSampler

# Import the new model
from src.models.roberta_hierarchical_improved import ImprovedRoBERTaHierarchical
from src.data_layer._common import LABEL_MAP, ASPECTS

# ============================================================================
# DATASET CLASS
# ============================================================================

class RoBERTaDataset(Dataset):
    def __init__(self, df, tokenizer, aspects, max_len=256):
        self.texts = df["text_clean"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.aspects = aspects
        self.labels = df[aspects].values.tolist()  # use tolist to handle Mixed types easier
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # LABEL_MAP already imported from _common.py
        labels = []
        for val in self.labels[idx]:
            if val is None:
                labels.append(3) # NULL
            elif val in LABEL_MAP:
                labels.append(LABEL_MAP[val])
            else:
                # Handle cases like logic mismatch or explicit "none"/"null" strings
                s_val = str(val).lower()
                if s_val in ["nan", "none", "null", ""]:
                    labels.append(3)
                else:
                    labels.append(-100)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long)
        }

# ============================================================================
# TRAINING LOGIC
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Compute ground-truth conflict labels (Novelty training)
        # Must exclude Class 3 (NULL) as it's not a conflicting sentiment
        conf_labels = []
        for lab in labels:
            valid = [x.item() for x in lab if x != -100 and x.item() != 3]
            conf_labels.append(1 if len(set(valid)) >= 2 else 0)
        conf_labels = torch.tensor(conf_labels, device=device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, enable_msr=(model.msr_strength > 0))
        
        cw = getattr(model, "class_weights", None)
        loss, loss_dict = model.compute_loss(
            forward_output=out, 
            labels=labels, 
            conflict_labels=conf_labels,
            class_weights=cw
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    aspects = ['stayingpower', 'texture', 'smell', 'price', 'colour', 'shipping', 'packing']
    
    # Load data
    train_path = args.train_aug_path if (args.use_synthetic and args.train_aug_path) else "data/splits/train.parquet"
    print(f"Loading train from: {train_path}")
    train_df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    val_df = pd.read_parquet("data/splits/val.parquet")
    
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    train_ds = RoBERTaDataset(train_df, tokenizer, aspects, args.max_len)
    val_ds = RoBERTaDataset(val_df, tokenizer, aspects, args.max_len)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(not args.use_sampler))
    if args.use_sampler:
        # Multi-label sampler: compute weights based on total occurrence of non-null classes
        # This is a simplification but helps boost rare aspects
        print("Initializing WeightedRandomSampler...")
        label_array = np.array(train_ds.labels) # [N, A]
        
        # Count how many aspects are present (not 'none'/'nan'/null)
        # Class 3 is NULL. We want weights for cases where aspects are present.
        weights = []
        for i in range(len(train_ds)):
            sample_labels = train_ds[i]["labels"].cpu().numpy()
            # If any aspect is 0, 1, or 2 (non-null), give it higher weight
            present = np.any((sample_labels >= 0) & (sample_labels <= 2))
            weights.append(10.0 if present else 1.0) # Boost presence 10x
        
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
        
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    model = ImprovedRoBERTaHierarchical(
        num_aspects=len(aspects),
        num_classes=4,
        aspect_names=aspects,
        msr_strength=args.msr_strength
    ).to(device)

    # Class weights: [1, 1, 1, null_weight]
    model.class_weights = torch.tensor([1.0, 1.0, 1.0, args.null_weight], device=device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.roberta.parameters(), 'lr': args.lr_bert},
        {'params': model.aspect_attention.parameters(), 'lr': args.lr_head},
        {'params': model.cross_aspect.parameters(), 'lr': args.lr_head},
        {'params': model.classifiers.parameters(), 'lr': args.lr_head},
        {'params': model.conflict_detector.parameters(), 'lr': args.lr_head},
        {'params': model.msr_refiners.parameters(), 'lr': args.lr_head}
    ])
    
    best_val_f1 = 0
    os.makedirs(args.out_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch} Loss: {loss:.4f}")
        
        # Simple evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].numpy()
                preds, _, _ = model.predict(ids, mask)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels)
        
        preds_np = np.concatenate(all_preds, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        
        # Macro F1 (Overall)
        f1s = []
        for i in range(len(aspects)):
            mask = labels_np[:, i] != -100
            if mask.sum() > 0:
                f1s.append(f1_score(labels_np[mask, i], preds_np[mask, i], average="macro", zero_division=0))
        val_f1 = np.mean(f1s)
        print(f"Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msr_strength", type=float, default=0.0)
    parser.add_argument("--train_aug_path", type=str, default=None)
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--use_sampler", action="store_true")
    parser.add_argument("--text_col", type=str, default="text_clean")
    parser.add_argument("--out_dir", type=str, default="outputs/tmp_model")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_bert", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--null_weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
