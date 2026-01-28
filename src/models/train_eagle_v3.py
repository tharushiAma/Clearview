# Training Script for EAGLE V3
# STRICTLY NO DATA AUGMENTATION
# Relies on Adaptive Focal Loss for class imbalance

import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import spacy
import os
import pickle
from tqdm import tqdm
from transformers import RobertaTokenizerFast
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from collections import Counter

# Import EAGLE V3
from eagle_v3_implementation import EAGLE_V3
from evaluation_comparison import evaluate_eagle_model, compare_models

# Load SpaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading SpaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ============================================================================
# PART 1: DEPENDENCY PARSING (Same as V2)
# ============================================================================

def dependency_adj_matrix(text, tokenizer, max_len):
    """
    Create dependency adjacency matrix.
    """
    doc = nlp(text)
    
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True
    )
    
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]
    seq_len = len(input_ids)
    
    adj = np.zeros((seq_len, seq_len), dtype=np.float32)
    
    # Build character to token mapping
    char_to_token_idx = {}
    for i, (start, end) in enumerate(offset_mapping):
        for char_pos in range(start, end):
            char_to_token_idx[char_pos] = i
    
    # Fill adjacency based on dependencies
    for token in doc:
        if token.idx in char_to_token_idx:
            current_idx = char_to_token_idx[token.idx]
            
            if token.head.idx in char_to_token_idx:
                head_idx = char_to_token_idx[token.head.idx]
                adj[current_idx][head_idx] = 1.0
                adj[head_idx][current_idx] = 1.0
    
    # Add self-loops
    for i in range(seq_len):
        adj[i][i] = 1.0
    
    # Normalize
    row_sum = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = np.matmul(np.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    
    return adj_normalized


def preprocess_and_cache_adjacency(df, tokenizer, max_len, cache_path):
    """
    Preprocess and cache adjacency matrices.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached adjacency matrices from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing adjacency matrices for {len(df)} samples...")
    adj_matrices = []
    
    for text in tqdm(df['text_clean'], desc="Building dependency graphs"):
        adj = dependency_adj_matrix(str(text), tokenizer, max_len)
        adj_matrices.append(adj)
    
    print(f"Saving to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(adj_matrices, f)
    
    return adj_matrices


# ============================================================================
# PART 2: DATASET CLASS
# ============================================================================

class EAGLEDataset(Dataset):
    """
    Dataset for EAGLE model.
    """
    def __init__(self, df, tokenizer, adj_matrices, aspects, max_len=256):
        self.texts = df["text_clean"].tolist()
        self.labels = df[aspects].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.aspects = aspects
        self.adj_matrices = adj_matrices
    
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
        
        # Get adjacency matrix
        adj_matrix = torch.tensor(self.adj_matrices[idx], dtype=torch.float32)
        
        # Create aspect masks (simplified placeholder)
        # Real implementation would detect specific aspect tokens
        aspect_masks = torch.zeros(len(self.aspects), self.max_len)
        for i in range(len(self.aspects)):
            aspect_masks[i, :5] = 1.0  # Placeholder: First 5 tokens
        
        # Position indices
        positions = torch.arange(self.max_len)
        
        # Process labels
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        labels = []
        for l in self.labels[idx]:
            labels.append(label_map[l] if l in label_map else -100)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "adj_matrix": adj_matrix,
            "aspect_masks": aspect_masks,
            "positions": positions,
            "labels": torch.tensor(labels, dtype=torch.long)
        }


# ============================================================================
# PART 3: TRAINING LOOP
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
        adj_matrix = batch["adj_matrix"].to(device)
        aspect_masks = batch["aspect_masks"].to(device)
        positions = batch["positions"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        forward_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            syntactic_adj=adj_matrix,
            aspect_masks=aspect_masks,
            positions=positions
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


def evaluate(model, dataloader, device, aspects, project_dir, model_name="eagle_v3"):
    """
    Wrapper for evaluation.
    """
    return evaluate_eagle_model(model, dataloader, device, aspects, project_dir, model_name)


# ============================================================================
# PART 4: MAIN TRAINING FUNCTION
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
    print("\nLoading raw data (NO SYNTHETIC AUGMENTATION)...")
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
    
    # Preprocess adjacency matrices
    cache_dir = f"{args.project_dir}/outputs/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    prefix = f"head{args.head}_" if args.head else ""
    
    print("\n" + "="*80)
    print("STAGE 1: DATA PREPARATION & DEPENDENCY PARSING")
    print("="*80)
    
    # Note: We use distinct cache names for V3 to avoid conflicts
    train_adj = preprocess_and_cache_adjacency(
        train_df, tokenizer, args.max_len,
        f"{cache_dir}/{prefix}train_adj_eagle_v3.pkl"
    )
    
    val_adj = preprocess_and_cache_adjacency(
        val_df, tokenizer, args.max_len,
        f"{cache_dir}/{prefix}val_adj_eagle_v3.pkl"
    )
    
    # Create datasets
    train_dataset = EAGLEDataset(train_df, tokenizer, train_adj, aspects, args.max_len)
    val_dataset = EAGLEDataset(val_df, tokenizer, val_adj, aspects, args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model (EAGLE V3)
    print("\nInitializing EAGLE V3 model...")
    model = EAGLE_V3(
        num_aspects=len(aspects),
        num_classes=3,
        gcn_dim=args.gcn_dim,
        gcn_layers=args.gcn_layers,
        aspect_names=aspects
    )
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.roberta.parameters(), 'lr': args.lr_bert},
        {'params': model.dual_gcn.parameters(), 'lr': args.lr_gcn},
        {'params': model.classifiers.parameters(), 'lr': args.lr_gcn},
        {'params': model.msr_module.parameters(), 'lr': args.lr_gcn}
    ])
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Verify GPU Usage Ready
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
        
        # CRITICAL: Update adaptive focal loss weights
        print("\n[Adaptive Loss] Calculating new class weights based on this epoch...")
        model.update_focal_loss_weights()
        
        # Evaluate
        val_results, val_f1 = evaluate(
            model, val_loader, device, aspects,
            args.project_dir,
            f"eagle_v3_epoch{epoch}"
        )
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0  # Reset counter
            print(f"\nNew best model! Val F1: {val_f1:.4f}")
            
            checkpoint_dir = f"{args.project_dir}/outputs/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_results': val_results
            }, f"{checkpoint_dir}/eagle_v3_best.pt")
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
    checkpoint = torch.load(f"{args.project_dir}/outputs/checkpoints/eagle_v3_best.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluate(
        model, val_loader, device, aspects,
        args.project_dir,
        "eagle_v3_final"
    )
    
    print("Results saved to: eagle_v3_final_metrics.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EAGLE V3 model")
    
    parser.add_argument("--project_dir", type=str, required=True,
                       help="Project directory containing data")
    parser.add_argument("--head", type=int, default=None,
                       help="Run on first N rows only (for testing)")
    
    parser.add_argument("--gcn_dim", type=int, default=300, help="GCN hidden dimension")
    parser.add_argument("--gcn_layers", type=int, default=2, help="Number of GCN layers")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum sequence length")
    
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr_bert", type=float, default=2e-5, help="Learning rate for BERT")
    parser.add_argument("--lr_gcn", type=float, default=1e-4, help="Learning rate for GCN/classifiers")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    main(args)
