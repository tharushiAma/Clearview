# EAGLE V2 Training Script with Enhanced Data Augmentation
# Focus: Price negative/neutral, Packing negative, Neutral class overall

import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import spacy
import os
import pickle
from tqdm import tqdm
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import EAGLE V2 model
from eagle_v2_implementation import EAGLE_V2
from evaluation_comparison import evaluate_eagle_model

# Load spacy for dependency parsing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# ==============================================================================
# PART 1: ENHANCED DATA AUGMENTATION FOR MINORITY CLASSES
# ==============================================================================

def analyze_class_distribution(df, aspects):
    """Analyze and print class distribution."""
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    for aspect in aspects:
        counts = df[aspect].value_counts().to_dict()
        total = df[aspect].notna().sum()
        
        print(f"\n{aspect.upper()}:")
        for sentiment in [0, 1, 2]:  # negative, neutral, positive
            count = counts.get(sentiment, 0)
            pct = (count / total * 100) if total > 0 else 0
            sentiment_name = ['negative', 'neutral', 'positive'][sentiment]
            print(f"  {sentiment_name:8s}: {count:5d} ({pct:5.1f}%)")


def augment_price_aspect(df, target_neg=50, target_neu=50):
    """
    CRITICAL: Augment price negative and neutral samples.
    
    Strategy:
    1. Identify existing price-negative/neutral examples
    2. Generate synthetic examples using templates
    3. Back-translate if possible
    
    Args:
        df: original dataframe
        target_neg: target number of negative samples
        target_neu: target number of neutral samples
    
    Returns:
        augmented dataframe
    """
    print("\n" + "="*80)
    print("AUGMENTING PRICE ASPECT")
    print("="*80)
    
    # Price negative templates
    price_neg_templates = [
        "This product is {adj}, not worth the price at all.",
        "Way too {adj} for what you get. Disappointed.",
        "The quality doesn't justify the {adj} price.",
        "I regret buying this, it's {adj}.",
        "{adj} and poor value for money.",
        "Don't waste your money, this is {adj}.",
        "The price is {adj}, I expected better quality.",
        "Not recommended, {adj} for such low quality.",
        "Totally {adj}, you're better off with cheaper alternatives.",
        "{adj} product, looks like a scam honestly."
    ]
    
    price_adj_neg = [
        'overpriced', 'expensive', 'costly', 'pricey', 'too expensive',
        'not affordable', 'a waste of money', 'too much', 'ridiculously expensive',
        'highway robbery'
    ]
    
    # Price neutral templates
    price_neu_templates = [
        "The price is {adj}, nothing special.",
        "{adj} pricing, what you'd expect.",
        "The product is {adj} for the price point.",
        "Price seems {adj}, not cheap but not expensive.",
        "{adj} price, typical for this category.",
        "I'd say the price is {adj}.",
        "{adj} value, could be better could be worse.",
        "The price is {adj} compared to similar products.",
        "It's {adj}, but the quality is just okay.",
        "{adj} but I wouldn't say it's a great deal."
    ]
    
    price_adj_neu = [
        'fair', 'okay', 'reasonable', 'decent', 'average',
        'standard', 'moderate', 'acceptable', 'market-rate', 'typical'
    ]
    
    augmented_rows = []
    
    # Generate price negative samples
    current_neg = (df['price'] == 0).sum()
    needed_neg = max(0, target_neg - current_neg)
    
    print(f"Price Negative: Current={current_neg}, Target={target_neg}, Generating={needed_neg}")
    
    for i in range(needed_neg):
        template = np.random.choice(price_neg_templates)
        adj = np.random.choice(price_adj_neg)
        text = template.format(adj=adj)
        
        # Create new row
        new_row = {
            'text_clean': text,
            'price': 0,  # negative
            # Set other aspects to -100 (ignore)
            'stayingpower': -100,
            'texture': -100,
            'smell': -100,
            'colour': -100,
            'shipping': -100,
            'packing': -100
        }
        augmented_rows.append(new_row)
    
    # Generate price neutral samples
    current_neu = (df['price'] == 1).sum()
    needed_neu = max(0, target_neu - current_neu)
    
    print(f"Price Neutral: Current={current_neu}, Target={target_neu}, Generating={needed_neu}")
    
    for i in range(needed_neu):
        template = np.random.choice(price_neu_templates)
        adj = np.random.choice(price_adj_neu)
        text = template.format(adj=adj)
        
        new_row = {
            'text_clean': text,
            'price': 1,  # neutral
            'stayingpower': -100,
            'texture': -100,
            'smell': -100,
            'colour': -100,
            'shipping': -100,
            'packing': -100
        }
        augmented_rows.append(new_row)
    
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        df = pd.concat([df, aug_df], ignore_index=True)
        print(f"Added {len(augmented_rows)} price samples")
    
    return df


def augment_packing_negative(df, target_neg=30):
    """
    Augment packing negative samples.
    
    Args:
        df: dataframe
        target_neg: target number of negative samples
    
    Returns:
        augmented dataframe
    """
    print("\n" + "="*80)
    print("AUGMENTING PACKING NEGATIVE")
    print("="*80)
    
    packing_neg_templates = [
        "The packaging was {adj}, product arrived damaged.",
        "{adj} packaging, very disappointed.",
        "Poor quality {adj} packaging, not protected at all.",
        "Arrived with {adj} packaging, product was broken.",
        "The box was {adj}, looks like it's been through a war.",
        "{adj} packing job, completely unprofessional.",
        "Packaging is {adj}, product leaked everywhere.",
        "{adj} and cheap packaging, expected better.",
        "The {adj} packaging ruined the whole experience.",
        "Terrible {adj} packaging, would not order again."
    ]
    
    packing_adj_neg = [
        'terrible', 'awful', 'poor', 'damaged', 'broken',
        'torn', 'flimsy', 'inadequate', 'shoddy', 'horrible'
    ]
    
    current_neg = (df['packing'] == 0).sum()
    needed_neg = max(0, target_neg - current_neg)
    
    print(f"Packing Negative: Current={current_neg}, Target={target_neg}, Generating={needed_neg}")
    
    augmented_rows = []
    
    for i in range(needed_neg):
        template = np.random.choice(packing_neg_templates)
        adj = np.random.choice(packing_adj_neg)
        text = template.format(adj=adj)
        
        new_row = {
            'text_clean': text,
            'packing': 0,  # negative
            'stayingpower': -100,
            'texture': -100,
            'smell': -100,
            'price': -100,
            'colour': -100,
            'shipping': -100
        }
        augmented_rows.append(new_row)
    
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        df = pd.concat([df, aug_df], ignore_index=True)
        print(f"Added {len(augmented_rows)} packing negative samples")
    
    return df


def augment_neutral_class(df, aspects, boost_factor=1.5):
    """
    Boost neutral class across all aspects using paraphrasing.
    
    Args:
        df: dataframe
        aspects: list of aspect names
        boost_factor: multiplier for neutral samples
    
    Returns:
        augmented dataframe
    """
    print("\n" + "="*80)
    print("AUGMENTING NEUTRAL CLASS (ALL ASPECTS)")
    print("="*80)
    
    neutral_paraphrase_patterns = [
        lambda text: text.replace(".", ", I guess."),
        lambda text: text.replace(".", ", somewhat."),
        lambda text: text.replace("good", "okay").replace("great", "decent"),
        lambda text: text.replace("bad", "not great").replace("terrible", "not good"),
        lambda text: f"{text} It's alright.",
        lambda text: f"{text} Could be better.",
        lambda text: f"{text} Nothing special.",
    ]
    
    augmented_rows = []
    
    for aspect in aspects:
        # Find neutral samples
        neutral_samples = df[df[aspect] == 1].copy()
        
        if len(neutral_samples) == 0:
            continue
        
        # Sample with replacement
        n_to_generate = int(len(neutral_samples) * (boost_factor - 1))
        
        if n_to_generate == 0:
            continue
        
        print(f"{aspect}: Generating {n_to_generate} neutral samples")
        
        sampled = neutral_samples.sample(n=n_to_generate, replace=True)
        
        for idx, row in sampled.iterrows():
            # Apply random paraphrasing
            text = row['text_clean']
            pattern = np.random.choice(neutral_paraphrase_patterns)
            
            try:
                new_text = pattern(text)
            except:
                new_text = text + " Fairly average."
            
            # Create new row
            new_row = row.to_dict()
            new_row['text_clean'] = new_text
            augmented_rows.append(new_row)
    
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        df = pd.concat([df, aug_df], ignore_index=True)
        print(f"\nTotal neutral augmented samples: {len(augmented_rows)}")
    
    return df


def apply_data_augmentation(df, aspects):
    """
    Apply all data augmentation strategies.
    
    Args:
        df: training dataframe
        aspects: list of aspect names
    
    Returns:
        augmented dataframe
    """
    print("\n" + "="*80)
    print("STARTING DATA AUGMENTATION")
    print("="*80)
    print(f"Original dataset size: {len(df)}")
    
    # 1. Analyze current distribution
    analyze_class_distribution(df, aspects)
    
    # 2. Augment price aspect (CRITICAL)
    df = augment_price_aspect(df, target_neg=50, target_neu=50)
    
    # 3. Augment packing negative
    df = augment_packing_negative(df, target_neg=30)
    
    # 4. Boost neutral class overall
    df = augment_neutral_class(df, aspects, boost_factor=1.5)
    
    # 5. Analyze after augmentation
    print("\n" + "="*80)
    print("AFTER AUGMENTATION")
    print("="*80)
    print(f"Final dataset size: {len(df)}")
    analyze_class_distribution(df, aspects)
    
    return df


# ==============================================================================
# PART 2: DEPENDENCY PARSING (Same as original)
# ==============================================================================

def dependency_adj_matrix(text, tokenizer, max_len):
    """Create dependency adjacency matrix."""
    # Tokenize
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]
    
    tokens = ['<s>'] + tokens + ['</s>']
    
    # Parse
    doc = nlp(text)
    
    # Create adjacency matrix
    n = len(tokens)
    adj = np.zeros((n, n), dtype=np.float32)
    
    # Add self-loops
    np.fill_diagonal(adj, 1.0)
    
    # Map spacy tokens to roberta tokens (simplified)
    for token in doc:
        if token.dep_ != 'ROOT' and token.head != token:
            # Add edge between token and its head
            # This is simplified - in production, need proper alignment
            src_idx = min(token.i + 1, n - 1)
            dst_idx = min(token.head.i + 1, n - 1)
            
            if src_idx < n and dst_idx < n:
                adj[src_idx, dst_idx] = 1.0
                adj[dst_idx, src_idx] = 1.0  # Undirected
    
    # Pad to max_len
    if n < max_len:
        padded = np.zeros((max_len, max_len), dtype=np.float32)
        padded[:n, :n] = adj
        adj = padded
    
    return adj


def preprocess_and_cache_adjacency(df, tokenizer, max_len, cache_path):
    """Preprocess and cache adjacency matrices."""
    if os.path.exists(cache_path):
        print(f"Loading cached adjacency matrices from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("Computing dependency adjacency matrices...")
    adj_matrices = []
    
    for text in tqdm(df['text_clean'].tolist(), desc="Parsing"):
        try:
            adj = dependency_adj_matrix(text, tokenizer, max_len)
        except:
            # Fallback: self-loops only
            adj = np.eye(max_len, dtype=np.float32)
        adj_matrices.append(adj)
    
    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(adj_matrices, f)
    
    print(f"Cached adjacency matrices to {cache_path}")
    return adj_matrices


# ==============================================================================
# PART 3: DATASET CLASS
# ==============================================================================

class EAGLE_V2_Dataset(Dataset):
    """Dataset for EAGLE V2 with all required inputs."""
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
        text = self.texts[idx]
        labels = self.labels[idx]
        adj_matrix = self.adj_matrices[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create aspect masks (simplified: use CLS token)
        # In production, would use actual aspect term positions
        aspect_masks = torch.zeros(len(self.aspects), self.max_len)
        aspect_masks[:, 0] = 1  # Use CLS for all aspects
        
        # Position indices
        positions = torch.arange(self.max_len)
        
        # Convert labels to tensor (-100 for missing values)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        labels_tensor = torch.where(
            torch.isnan(labels_tensor.float()),
            torch.tensor(-100, dtype=torch.long),
            labels_tensor
        )
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'syntactic_adj': torch.tensor(adj_matrix, dtype=torch.float32),
            'aspect_masks': aspect_masks,
            'positions': positions,
            'labels': labels_tensor
        }


# ==============================================================================
# PART 4: TRAINING LOOP
# ==============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        syntactic_adj = batch['syntactic_adj'].to(device)
        aspect_masks = batch['aspect_masks'].to(device)
        positions = batch['positions'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            syntactic_adj=syntactic_adj,
            aspect_masks=aspect_masks,
            positions=positions
        )
        
        # Compute loss
        loss, loss_dict = model.compute_loss(output, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device, aspects, project_dir, model_name="eagle_v2"):
    """Evaluate and save results."""
    # Use existing evaluation function
    return evaluate_eagle_model(
        model=model,
        dataloader=dataloader,
        device=device,
        aspects=aspects,
        project_dir=project_dir,
        model_name=model_name
    )


# ==============================================================================
# PART 5: MAIN TRAINING FUNCTION
# ==============================================================================

def main(args):
    """Main training function."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. Training will be VERY SLOW on CPU.")
        print("   If you have a GPU, ensure PyTorch with CUDA is installed:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n")
    else:
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Paths
    project_dir = args.project_dir
    data_dir = os.path.join(project_dir, 'data', 'splits')
    output_dir = os.path.join(project_dir, 'outputs')
    cache_dir = os.path.join(output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Aspects
    aspects = ['stayingpower', 'texture', 'smell', 'price', 'colour', 'shipping', 'packing']
    
    # Load data - use augmented dataset if available for full training
    print("Loading data...")
    
    # Option to use pre-augmented dataset or augment during training
    train_file = 'train_augmented.parquet' if args.use_preaugmented else 'train.parquet'
    val_file = 'val.parquet'  # Always use original validation set
    
    print(f"Training data: {train_file}")
    print(f"Validation data: {val_file}")
    
    train_df = pd.read_parquet(os.path.join(data_dir, train_file))
    val_df = pd.read_parquet(os.path.join(data_dir, val_file))
    
    # Convert string labels to integers
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    print("Converting string labels to integers...")
    for aspect in aspects:
        # Map strings to integers
        train_df[aspect] = train_df[aspect].map(label_mapping)
        val_df[aspect] = val_df[aspect].map(label_mapping)
        
        # Replace NaN with -100 for ignore_index
        train_df[aspect] = train_df[aspect].fillna(-100).astype(int)
        val_df[aspect] = val_df[aspect].fillna(-100).astype(int)
    
    # Apply data augmentation to training set
    if args.augment:
        train_df = apply_data_augmentation(train_df, aspects)
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Preprocess adjacency matrices
    train_cache = os.path.join(cache_dir, 'train_adj_v2.pkl')
    val_cache = os.path.join(cache_dir, 'val_adj_v2.pkl')
    
    train_adj = preprocess_and_cache_adjacency(train_df, tokenizer, args.max_len, train_cache)
    val_adj = preprocess_and_cache_adjacency(val_df, tokenizer, args.max_len, val_cache)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = EAGLE_V2_Dataset(train_df, tokenizer, train_adj, aspects, args.max_len)
    val_dataset = EAGLE_V2_Dataset(val_df, tokenizer, val_adj, aspects, args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    print("Initializing EAGLE V2 model...")
    model = EAGLE_V2(
        num_aspects=len(aspects),
        num_classes=3,
        gcn_dim=args.gcn_dim,
        gcn_layers=args.gcn_layers,
        aspect_names=aspects,
        use_uncertainty=args.use_uncertainty,
        use_feature_routing=args.use_feature_routing
    )
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1
    )
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    best_val_f1 = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        if epoch % args.eval_every == 0:
            print("\nEvaluating on validation set...")
            val_metrics, avg_f1 = evaluate(
                model, val_loader, device, aspects,
                project_dir, model_name=f"eagle_v2_epoch{epoch}"
            )
            
            # Save best model
            print(f"Validation Macro F1: {avg_f1:.4f}")
            
            if avg_f1 > best_val_f1:
                best_val_f1 = avg_f1
                checkpoint_path = os.path.join(output_dir, 'checkpoints', 'eagle_v2_best.pt')
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1
                }, checkpoint_path)
                
                print(f"Saved best model with F1: {best_val_f1:.4f}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EAGLE V2 Model')
    
    # Paths
    parser.add_argument('--project_dir', type=str, default='c:/Users/lucif/Desktop/Clearview',
                        help='Project directory')
    
    # Model config
    parser.add_argument('--gcn_dim', type=int, default=300, help='GCN hidden dimension')
    parser.add_argument('--gcn_layers', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--max_len', type=int, default=256, help='Max sequence length')
    parser.add_argument('--use_uncertainty', action='store_true', default=True,
                        help='Use uncertainty-aware prediction heads')
    parser.add_argument('--use_feature_routing', action='store_true', default=True,
                        help='Use aspect-specific feature routing')
    
    # Training config
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--eval_every', type=int, default=2, help='Evaluate every N epochs')
    
    # Data augmentation
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation during training (slow)')
    parser.add_argument('--use_preaugmented', action='store_true',
                        help='Use pre-augmented train dataset (faster, recommended)')
    
    args = parser.parse_args()
    
    main(args)
