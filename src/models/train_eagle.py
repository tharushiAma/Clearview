# Complete Training Script for EAGLE Model
# Includes data augmentation, training loop, and evaluation

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
from sklearn.metrics import classification_report, f1_score, accuracy_score
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from collections import Counter

# Import EAGLE components
from eagle_implementation import EAGLE, AdaptiveFocalLoss

# Load SpaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading SpaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ============================================================================
# PART 1: DATA AUGMENTATION
# ============================================================================

def analyze_class_imbalance(df, aspects):
    """
    Analyze class distribution for each aspect.
    """
    print("\n" + "="*80)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*80)
    
    imbalance_report = []
    
    for aspect in aspects:
        counts = df[aspect].value_counts()
        total = len(df[df[aspect].notna()])
        
        print(f"\n{aspect.upper()}:")
        print(f"  Total samples: {total}")
        
        for label in ['negative', 'neutral', 'positive']:
            count = counts.get(label, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {label:8s}: {count:4d} ({pct:5.1f}%)")
        
        # Calculate imbalance ratio
        if len(counts) > 0:
            max_count = counts.max()
            min_count = counts.min()
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            imbalance_report.append({
                'aspect': aspect,
                'imbalance_ratio': imbalance_ratio,
                'needs_augmentation': imbalance_ratio > 10
            })
            
            print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
            if imbalance_ratio > 10:
                print(f"  ⚠️  SEVERE IMBALANCE - Augmentation recommended")
    
    return imbalance_report


def augment_aspect_data(df, aspect, method='adasyn', target_ratio=0.3):
    """
    Augment data for a single aspect using SMOTE variants.
    
    Args:
        df: dataframe
        aspect: aspect name
        method: 'adasyn', 'borderline_smote', or 'smote_tomek'
        target_ratio: desired minority/majority ratio
    
    Returns:
        augmented dataframe for this aspect
    """
    # Filter rows where aspect is labeled
    df_aspect = df[df[aspect].notna()].copy()
    
    if len(df_aspect) == 0:
        return df_aspect
    
    # Encode text as simple features (bag of words)
    # In practice, you might want to use BERT embeddings
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df_aspect['text_clean']).toarray()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_aspect[aspect])
    
    # Check if augmentation is needed
    class_counts = Counter(y)
    if len(class_counts) < 2:
        print(f"  {aspect}: Only one class present, skipping augmentation")
        return df_aspect
    
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    
    if max_count / min_count < 5:
        print(f"  {aspect}: Balanced enough (ratio {max_count/min_count:.1f}), skipping")
        return df_aspect
    
    try:
        # Apply augmentation
        if method == 'adasyn':
            sampler = ADASYN(
                sampling_strategy=target_ratio,
                random_state=42,
                n_neighbors=min(5, min_count - 1)
            )
        elif method == 'borderline_smote':
            sampler = BorderlineSMOTE(
                sampling_strategy=target_ratio,
                random_state=42,
                k_neighbors=min(5, min_count - 1)
            )
        else:  # smote_tomek
            sampler = SMOTETomek(
                sampling_strategy=target_ratio,
                random_state=42
            )
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Create new text samples (simplified - just duplicate)
        # In practice, you might want to use back-translation or paraphrasing
        n_new = len(X_resampled) - len(X)
        
        if n_new > 0:
            # Find original samples to duplicate
            new_indices = np.random.choice(len(df_aspect), size=n_new, replace=True)
            new_rows = df_aspect.iloc[new_indices].copy()
            
            # Mark as synthetic
            new_rows['text_clean'] = new_rows['text_clean'] + " [AUGMENTED]"
            
            # Decode labels
            y_new = y_resampled[-n_new:]
            new_rows[aspect] = label_encoder.inverse_transform(y_new)
            
            # Combine
            df_augmented = pd.concat([df_aspect, new_rows], ignore_index=True)
            
            print(f"  {aspect}: Added {n_new} synthetic samples")
            print(f"    New distribution: {Counter(df_augmented[aspect])}")
            
            return df_augmented
    
    except Exception as e:
        print(f"  {aspect}: Augmentation failed - {e}")
        return df_aspect
    
    return df_aspect


def augment_training_data(df, aspects):
    """
    Apply data augmentation to all severely imbalanced aspects.
    """
    print("\n" + "="*80)
    print("DATA AUGMENTATION")
    print("="*80)
    
    # Analyze imbalance
    imbalance_report = analyze_class_imbalance(df, aspects)
    
    # Augment severely imbalanced aspects
    augmented_dfs = []
    
    for report in imbalance_report:
        aspect = report['aspect']
        
        if report['needs_augmentation']:
            print(f"\nAugmenting {aspect}...")
            df_aug = augment_aspect_data(df, aspect, method='adasyn')
            augmented_dfs.append(df_aug[[aspect, 'text_clean']])
    
    if augmented_dfs:
        # Merge augmented data with original
        df_final = df.copy()
        
        for df_aug in augmented_dfs:
            # Add new rows
            new_rows = df_aug[~df_aug['text_clean'].isin(df_final['text_clean'])]
            
            if len(new_rows) > 0:
                # Fill missing aspects with None
                for aspect in aspects:
                    if aspect not in new_rows.columns:
                        new_rows[aspect] = None
                
                df_final = pd.concat([df_final, new_rows], ignore_index=True)
        
        print(f"\nOriginal dataset size: {len(df)}")
        print(f"Augmented dataset size: {len(df_final)}")
        print(f"New samples added: {len(df_final) - len(df)}")
        
        return df_final
    
    return df


# ============================================================================
# PART 2: DEPENDENCY PARSING
# ============================================================================

def dependency_adj_matrix(text, tokenizer, max_len):
    """
    Create dependency adjacency matrix.
    Same as your original implementation.
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
# PART 3: DATASET CLASS
# ============================================================================

class EAGLEDataset(Dataset):
    """
    Dataset for EAGLE model with all required inputs.
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
        
        # Create aspect masks (simplified - mark first 3 tokens as aspect)
        # In practice, you should detect actual aspect tokens
        aspect_masks = torch.zeros(len(self.aspects), self.max_len)
        for i in range(len(self.aspects)):
            aspect_masks[i, :3] = 1.0  # Placeholder
        
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
# PART 4: TRAINING LOOP
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
        aspect_logits, msr_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            syntactic_adj=adj_matrix,
            aspect_masks=aspect_masks,
            positions=positions
        )
        
        # Compute loss
        loss, loss_dict = model.compute_loss(
            aspect_logits_list=aspect_logits,
            labels=labels,
            msr_output=msr_output
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
    
    avg_loss = total_loss / steps
    avg_components = {k: v / steps for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def evaluate(model, dataloader, device, aspects):
    """
    Evaluate model on validation/test set.
    """
    model.eval()
    
    all_preds = {a: [] for a in aspects}
    all_labels = {a: [] for a in aspects}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            adj_matrix = batch["adj_matrix"].to(device)
            aspect_masks = batch["aspect_masks"].to(device)
            positions = batch["positions"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward
            aspect_logits, msr_output = model(
                input_ids, attention_mask, adj_matrix, aspect_masks, positions
            )
            
            # Collect predictions
            for i, aspect in enumerate(aspects):
                preds = torch.argmax(aspect_logits[i], dim=-1)
                all_preds[aspect].extend(preds.cpu().tolist())
                all_labels[aspect].extend(labels[:, i].cpu().tolist())
    
    # Compute metrics
    results = {}
    macro_f1_scores = []
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    for aspect in aspects:
        # Filter valid labels
        valid_mask = [i for i, l in enumerate(all_labels[aspect]) if l != -100]
        
        if not valid_mask:
            print(f"\n{aspect}: No valid samples")
            continue
        
        y_true = [all_labels[aspect][i] for i in valid_mask]
        y_pred = [all_preds[aspect][i] for i in valid_mask]
        
        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        results[aspect] = {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'per_class_f1': per_class_f1
        }
        
        macro_f1_scores.append(macro_f1)
        
        # Print results
        print(f"\n{aspect.upper()}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Per-class F1: Neg={per_class_f1[0]:.3f}, Neu={per_class_f1[1]:.3f}, Pos={per_class_f1[2]:.3f}")
    
    # Overall metrics
    overall_macro_f1 = np.mean(macro_f1_scores)
    print(f"\n{'='*80}")
    print(f"OVERALL MACRO F1: {overall_macro_f1:.4f}")
    print(f"{'='*80}\n")
    
    return results, overall_macro_f1


# ============================================================================
# PART 5: MAIN TRAINING FUNCTION
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
    print("\nLoading data...")
    train_df = pd.read_parquet(f"{args.project_dir}/data/splits/train.parquet")
    val_df = pd.read_parquet(f"{args.project_dir}/data/splits/val.parquet")
    
    # Apply data augmentation to training set
    if args.augment:
        train_df = augment_training_data(train_df, aspects)
    
    # Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    # Preprocess adjacency matrices
    cache_dir = f"{args.project_dir}/outputs/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    train_adj = preprocess_and_cache_adjacency(
        train_df, tokenizer, args.max_len,
        f"{cache_dir}/train_adj_eagle.pkl"
    )
    
    val_adj = preprocess_and_cache_adjacency(
        val_df, tokenizer, args.max_len,
        f"{cache_dir}/val_adj_eagle.pkl"
    )
    
    # Create datasets
    train_dataset = EAGLEDataset(train_df, tokenizer, train_adj, aspects, args.max_len)
    val_dataset = EAGLEDataset(val_df, tokenizer, val_adj, aspects, args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    print("\nInitializing EAGLE model...")
    model = EAGLE(
        num_aspects=len(aspects),
        num_classes=3,
        gcn_dim=args.gcn_dim,
        gcn_layers=args.gcn_layers,
        aspect_names=aspects
    )
    model = model.to(device)
    
    # Optimizer with different learning rates
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
    
    # Training loop
    best_val_f1 = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_components = train_epoch(model, train_loader, optimizer, device, epoch)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"  Aspect: {train_components['aspect']:.4f}")
        print(f"  MSR Sentiment: {train_components['msr_sentiment']:.4f}")
        print(f"  MSR Conflict: {train_components['msr_conflict']:.4f}")
        
        # Update adaptive focal loss weights
        print("\nUpdating adaptive focal loss weights...")
        model.update_focal_loss_weights()
        
        # Evaluate
        val_results, val_f1 = evaluate(model, val_loader, device, aspects)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"\n✅ New best model! Val F1: {val_f1:.4f}")
            
            checkpoint_dir = f"{args.project_dir}/outputs/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_results': val_results
            }, f"{checkpoint_dir}/eagle_best.pt")
        
        # Step scheduler
        scheduler.step()
    
    print(f"\nTraining complete! Best Val F1: {best_val_f1:.4f}")


# ============================================================================
# PART 6: ARGUMENT PARSER
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EAGLE model")
    
    # Paths
    parser.add_argument("--project_dir", type=str, required=True,
                       help="Project directory containing data")
    
    # Model hyperparameters
    parser.add_argument("--gcn_dim", type=int, default=300,
                       help="GCN hidden dimension")
    parser.add_argument("--gcn_layers", type=int, default=2,
                       help="Number of GCN layers")
    parser.add_argument("--max_len", type=int, default=256,
                       help="Maximum sequence length")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--lr_bert", type=float, default=2e-5,
                       help="Learning rate for BERT")
    parser.add_argument("--lr_gcn", type=float, default=1e-4,
                       help="Learning rate for GCN and classifiers")
    
    # Data augmentation
    parser.add_argument("--augment", action="store_true",
                       help="Apply data augmentation")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    main(args)
