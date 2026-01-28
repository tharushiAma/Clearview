# Training Script for RoBERTa Hierarchical Model
# Simplified training without graph construction

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

# Import the new model
from roberta_hierarchical_implementation import RoBERTaHierarchicalModel
from evaluation_comparison import evaluate_eagle_model


# ============================================================================
# DATASET CLASS (Simplified - No Adjacency Matrices)
# ============================================================================

class RoBERTaDataset(Dataset):
    """
    Dataset for RoBERTa Hierarchical Model.
    No graph/adjacency matrix needed.
    """
    def __init__(self, df, tokenizer, aspects, max_len=256):
        self.texts = df["text_clean"].tolist()
        self.labels = df[aspects].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.aspects = aspects
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Process labels
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        labels = []
        for l in self.labels[idx]:
            labels.append(label_map[l] if l in label_map else -100)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long)
        }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    loss_components = {'aspect': 0, 'msr_sentiment': 0, 'msr_conflict': 0}
    steps = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        optimizer.zero_grad()
        
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        forward_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Compute loss
        loss, loss_dict = model.compute_loss(
            forward_output=forward_output,
            labels=labels
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        for key in loss_components:
            if key + '_loss' in loss_dict:
                loss_components[key] += loss_dict[key + '_loss']
        steps += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / max(steps, 1)
    avg_components = {k: v / max(steps, 1) for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def evaluate(model, dataloader, device, aspects, project_dir, model_name="roberta_hierarchical"):
    """
    Wrapper for evaluation.
    """
    return evaluate_eagle_model(model, dataloader, device, aspects, project_dir, model_name)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(args):
    """
    Main training function.
    """
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Aspects
    aspects = ['stayingpower', 'texture', 'smell', 'price', 'colour', 'shipping', 'packing']
    
    # Load data
    print("\nLoading data (NO SYNTHETIC AUGMENTATION)...")
    train_df = pd.read_parquet(f"{args.project_dir}/data/splits/train.parquet")
    val_df = pd.read_parquet(f"{args.project_dir}/data/splits/val.parquet")
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    
    # Apply Sample Limit if requested
    if args.head is not None:
        print(f"--- TEST MODE: Truncating data to first {args.head} rows ---")
        train_df = train_df.head(args.head)
        val_df = val_df.head(args.head)
    
    # Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    # Create datasets (NO adjacency matrix preprocessing needed)
    print("\n" + "="*80)
    print("STAGE 1: DATA PREPARATION (Simple Tokenization Only)")
    print("="*80)
    
    train_dataset = RoBERTaDataset(train_df, tokenizer, aspects, args.max_len)
    val_dataset = RoBERTaDataset(val_df, tokenizer, aspects, args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    print("\nInitializing RoBERTa Hierarchical Model...")
    model = RoBERTaHierarchicalModel(
        num_aspects=len(aspects),
        num_classes=3,
        aspect_names=aspects,
        hidden_dropout=0.3,
        output_attentions=args.output_attentions
    )
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.roberta.parameters(), 'lr': args.lr_bert},
        {'params': model.classifiers.parameters(), 'lr': args.lr_head},
        {'params': model.msr_module.parameters(), 'lr': args.lr_head}
    ])
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Verify GPU Usage
    print("\n" + "="*80)
    print("STAGE 2: TRAINING PHASE (GPU ACCELERATED)")
    print("="*80)
    
    if device.type == 'cuda':
        print(f"Successfully connected to GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Running on CPU.")
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    patience_limit = args.patience
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_components = train_epoch(model, train_loader, optimizer, device, epoch)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"  Aspect: {train_components['aspect']:.4f}")
        print(f"  MSR Sentiment: {train_components['msr_sentiment']:.4f}")
        
        # Update adaptive focal loss weights
        print("\n[Adaptive Loss] Calculating new class weights based on this epoch...")
        model.update_focal_loss_weights()
        
        # Evaluate
        val_results, val_f1 = evaluate(
            model, val_loader, device, aspects,
            args.project_dir,
            f"roberta_hierarchical_epoch{epoch}"
        )
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            print(f"\nNew best model! Val F1: {val_f1:.4f}")
            
            checkpoint_dir = f"{args.project_dir}/outputs/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_results': val_results
            }, f"{checkpoint_dir}/roberta_hierarchical_best.pt")
        else:
            patience_counter += 1
            print(f"\nNo improvement in Val F1. Patience: {patience_counter}/{patience_limit}")
            
            if patience_counter >= patience_limit:
                print(f"\nEARLY STOPPING TRIGGERED after {epoch} epochs.")
                break
        
        # Step scheduler
        scheduler.step()
    
    print(f"\nTraining complete! Best Val F1: {best_val_f1:.4f}")
    
    # FINAL EVALUATION
    print(f"\nLoading best model for final evaluation...")
    checkpoint = torch.load(f"{args.project_dir}/outputs/checkpoints/roberta_hierarchical_best.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluate(
        model, val_loader, device, aspects,
        args.project_dir,
        "roberta_hierarchical_final"
    )
    
    print("Results saved to: roberta_hierarchical_final_metrics.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoBERTa Hierarchical Model")
    
    parser.add_argument("--project_dir", type=str, required=True,
                       help="Project directory containing data")
    parser.add_argument("--head", type=int, default=None,
                       help="Run on first N rows only (for testing)")
    
    parser.add_argument("--max_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--output_attentions", action="store_true",
                       help="Output RoBERTa self-attention weights for explainability")
    
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr_bert", type=float, default=2e-5, help="Learning rate for BERT")
    parser.add_argument("--lr_head", type=float, default=1e-4, help="Learning rate for classifiers/MSR")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    main(args)
