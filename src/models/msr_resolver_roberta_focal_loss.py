import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import spacy  # Library for NLP tasks (Dependency Parsing) - Used to understand sentence structure ('Good' -> 'Price')
import math
import os
import pickle # Used to save/load complex data structures like lists of matrices
from transformers import RobertaTokenizerFast, RobertaModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import random
from tqdm import tqdm # Progress bar

# ==================================================================================
# 1. WHY SPACY?
# We need to build a "Graph" where words are nodes and grammatical relationships are edges.
# RoBERTa understands "context", but GCN needs explicit "structure" (e.g., Adjective modifies Noun).
# SpaCy provides this structure via Dependency Parsing.
# ==================================================================================

# ---------------------------
# GLOBAL SETUP
# ---------------------------

# Load the English language model from SpaCy.
# We do this once at the top because loading it inside a loop is very slow.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
    exit()

def set_seed(seed=42):
    """
    Sets random seeds for Python, NumPy, and PyTorch.
    This ensures that every time you run the experiment, you get the exact same results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------
# PART 1: PRE-PROCESSING (Adjacency Matrix)
# ---------------------------

def dependency_adj_matrix(text, tokenizer, max_len):
    """
    Creates a mathematical representation (matrix) of the sentence's grammar.
    GCN works by passing information between connected words. 
    If 'Price' depends on 'High', this matrix connects them so they exchange information.
    
    Returns:
        adj_normalized: A scaled matrix ready for GCN multiplication.
    """
    # 1. SpaCy analyzes the sentence structure (Subject, Object, Modifiers, etc.)
    doc = nlp(text)
    
    # 2. RoBERTa breaks text into 'subwords' (e.g., "playing" -> "play" + "ing").
    # We need to match SpaCy's 'Words' to RoBERTa's 'Subwords'.
    encoding = tokenizer(
        text, 
        padding="max_length", 
        truncation=True, 
        max_length=max_len, 
        return_offsets_mapping=True # Crucial: tells us start/end char index of each token
    )
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"] 
    
    seq_len = len(input_ids) # Always 256 due to padding="max_length"
    
    # Initialize an empty N x N matrix with zeros.
    adj = np.zeros((seq_len, seq_len), dtype=np.float32)
    
    # 3. Build a map: Character Position -> RoBERTa Token Index
    # Example: If 'Good' is at char 0-4, map indices 0,1,2,3 -> Token_0
    char_to_token_idx = {}
    for i, (start, end) in enumerate(offset_mapping):
        for char_pos in range(start, end):
            char_to_token_idx[char_pos] = i
            
    # 4. Fill Matrix based on Grammar (Dependency Tree)
    for token in doc:
        # Check if the word exists in our RoBERTa tokens (handling truncation)
        if token.idx in char_to_token_idx:
            current_idx = char_to_token_idx[token.idx] # Index of the child word
            
            # Check if the parent (head) word exists
            if token.head.idx in char_to_token_idx:
                head_idx = char_to_token_idx[token.head.idx] # Index of the parent word
                
                # Connect them! (Undirected graph: A->B and B->A)
                adj[current_idx][head_idx] = 1.0
                adj[head_idx][current_idx] = 1.0
                
    # 5. Add Self-Loops: Every word must connect to itself.
    # Otherwise, the GCN would forget the word's own meaning and only look at neighbors.
    for i in range(seq_len):
        adj[i][i] = 1.0
        
    # 6. Normalize the Matrix (Mathematical stability step)
    # GCN Formula Part: D^(-0.5) * A * D^(-0.5)
    # Why? If a node has many neighbors, summing their features would make the values huge (explode).
    # Normalization scales it down based on the number of neighbors (Degree).
    # This prevents the values from exploding when we multiply matrices many times.
    row_sum = np.sum(adj, axis=1) # Calculate degree of each node
    
    d_inv_sqrt = np.power(row_sum, -0.5) # Inverse square root
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # Handle division by zero (for padding tokens)
    
    d_mat_inv_sqrt = np.diag(d_inv_sqrt) # Convert to diagonal matrix
    
    # Perform the matrix multiplication for normalization
    adj_normalized = np.matmul(np.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    
    return adj_normalized

def preprocess_and_save_adj(df, tokenizer, max_len, save_path):
    """
    Optimization: Checks if we already calculated these matrices.
    If yes, load them. If no, calculate and save them.
    """
    if os.path.exists(save_path):
        print(f"Loading cached adjacency matrices from {save_path}...")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print(f"Computing adjacency matrices for {len(df)} samples...")
    adj_matrices = []
    
    texts = df["text_clean"].tolist()
    # tqdm creates a progress bar so you know how long it will take
    for text in tqdm(texts, desc="Processing Dependencies"):
        adj = dependency_adj_matrix(str(text), tokenizer, max_len)
        adj_matrices.append(adj)
    
    print(f"Saving adjacency matrices to {save_path}...")
    # Save the list of matrices to a file
    with open(save_path, 'wb') as f:
        pickle.dump(adj_matrices, f)
        
    return adj_matrices

# ---------------------------
# PART 2: THE GCN LAYER
# ---------------------------

class GraphConvolution(nn.Module):
    """
    The GCN (Graph Convolutional Network) logic. 
    It doesn't just look at the word itself, but mixes information from its "neighbors" in the sentence graph.
    If 'bad' modifies 'service', this layer ensures 'service' absorbs some 'badness' before classification.
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable Weight Matrix (W)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Bias term (b)
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with small random values
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, text_features, adj):
        """
        Input: RoBERTa Embeddings (text_features) + Dependency Graph (adj)
        Formula: Output = Adj * (Features * W) + Bias
        
        Step 1: Transform features (Features * W) -> Prepare for mixing
        Step 2: Propagate info (Adj * Support) -> Neighbors talk to each other
        """
        # 1. Transform features: H * W
        support = torch.matmul(text_features, self.weight)
        
        # 2. Propagate info across graph: Adj * Support
        # 'bmm' means Batch Matrix Multiplication (processes a whole batch at once)
        output = torch.bmm(adj, support)
        
        # 3. Add bias
        return output + self.bias

# ---------------------------
# PART 2.1: FOCAL LOSS
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Alpha handles class imbalance (like your weights)
        # alpha: A list of weights [weight_neg, weight_neu, weight_pos]
        # If 'Negative' is rare, we give it a high weight (e.g., 10.0) so the model pays 10x attention to it.
        self.alpha = alpha 

    def forward(self, inputs, targets):
        # inputs: [Batch, C], targets: [Batch]
        
        # 1. Calculate 'pt' (probability of target class) WITHOUT weights
        # We need the true model confidence, not skewed by class weights.
        ce_loss_unweighted = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)(inputs, targets)
        pt = torch.exp(-ce_loss_unweighted) 
        
        # 2. Calculate Weighted Cross Entropy (to handle class imbalance)
        ce_loss_weighted = nn.CrossEntropyLoss(weight=self.alpha, reduction='none', ignore_index=-100)(inputs, targets)
        
        # 3. Apply Focal Loss Factor: (1 - pt)^gamma * Weighted_CE
        # (1 - pt)^gamma: The "Focusing Parameter".
        # If model is confident (pt=0.99), (1-pt) is tiny (~0), so Loss is tiny -> Don't worry about easy/correct examples.
        # If model is wrong/unsure (pt=0.10), (1-pt) is large (~0.9), so Loss is big -> "Focus" on this hard example!
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_weighted

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ---------------------------
# PART 3: DATASET
# ---------------------------

class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, adj_matrices, aspects, max_len=256):
        self.texts = df["text_clean"].tolist()
        self.labels = df[aspects].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.aspects = aspects
        self.adj_matrices = adj_matrices # Receive the pre-computed list

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize text for RoBERTa
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # Retrieve the specific adjacency matrix for this sentence
        adj_matrix = self.adj_matrices[idx]
        
        # Process Labels
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        labels = []
        for l in self.labels[idx]:
            # Convert string label to number, use -100 for missing/NaN
            labels.append(label_map[l] if l in label_map else -100)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "adj_matrix": torch.tensor(adj_matrix, dtype=torch.float), # Convert numpy to tensor
            "labels": torch.tensor(labels, dtype=torch.long)
        }

# ---------------------------
# PART 4: THE MODEL (RoBERTa + GCN)
# ---------------------------

class MultiAspectRobertaGCN(nn.Module):
    def __init__(self, num_aspects, num_classes=3, gcn_dim=768):
        super().__init__()
        # Load pre-trained RoBERTa
        self.bert = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.3) # Prevents overfitting
        
        # The GCN Layer
        # It takes RoBERTa's hidden size (768) and outputs gcn_dim (768)
        self.gcn = GraphConvolution(self.bert.config.hidden_size, gcn_dim)
        
        # Classification Heads (One linear layer per aspect)
        self.classifiers = nn.ModuleList([
            nn.Linear(gcn_dim, num_classes)
            for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask, adj_matrix):
        # 1. Get Contextual Embeddings from RoBERTa
        # last_hidden_state shape: [Batch, 256, 768]
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = out.last_hidden_state 
        
        # 2. Refine Embeddings using GCN
        # The model uses the adjacency matrix to mix word meanings
        gcn_output = self.gcn(sequence_output, adj_matrix) 
        gcn_output = torch.relu(gcn_output) # ReLU activation
        
        # 3. Pooling (Extract the summary vector)
        # We take the 0-th token ([CLS]), which now contains info from the whole graph
        pooled_output = gcn_output[:, 0, :] 
        pooled_output = self.dropout(pooled_output)

        # 4. Predict Sentiment for each Aspect
        logits = [clf(pooled_output) for clf in self.classifiers]
        return logits

# ---------------------------
# PART 5: TRAINING & EVALUATION
# ---------------------------

def train_epoch(model, dataloader, optimizer, device, class_weights=None):
    model.train() 
    
    # --- CHANGE: Use Focal Loss instead of CrossEntropy ---
    # gamma=2.0 is the standard paper value. 
    # It reduces the loss of "easy" examples by ~4x-10x.
    loss_fn = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean') 

    total_loss = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="Training Batch", leave=False):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        adj_matrix = batch["adj_matrix"].to(device)
        labels = batch["labels"].to(device)

        logits_list = model(input_ids, attention_mask, adj_matrix)

        batch_loss = None
        for aspect_idx, logits in enumerate(logits_list):
            if (labels[:, aspect_idx] != -100).any():
                # Only calculate loss if we have valid labels (ignore -100)
                loss_i = loss_fn(logits, labels[:, aspect_idx])
                
                # Accumulate loss for all aspects (Multi-Task Learning)
                batch_loss = loss_i if batch_loss is None else (batch_loss + loss_i)

        if batch_loss is None: continue 

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()

        total_loss += batch_loss.item()
        steps += 1

    return total_loss / max(steps, 1)


# ---------------------------
# PART 6: MSR LOGIC (INTERNAL)
# ---------------------------
def resolve_row(row, aspects):
    """
    Decides the final sentiment of a review based on aspect predictions.
    Logic:
    1. Calculate weighted score for Positive and Negative based on confidence.
    2. Check for Conflict (Mixed Sentiment).
    3. Determine final resolution.
    """
    pos = neu = neg = 0
    weighted_pos = 0.0
    weighted_neg = 0.0

    # 1. Aggregate Scores from all Aspects
    for a in aspects:
        pred = row.get(f"{a}_pred") 
        conf = row.get(f"{a}_conf", 1.0)
        
        if pred == "positive":
            pos += 1
            weighted_pos += conf
        elif pred == "negative":
            neg += 1
            weighted_neg += conf
        elif pred == "neutral":
            neu += 1

    # 2. Calculate Conflict Score
    denom = (weighted_pos + weighted_neg + 1e-9)
    conflict_score = (weighted_pos * weighted_neg) / denom
    
    # 3. Determine Resolution
    # If the review has strong Positive AND Negative signals simultaneously -> It's MIXED.
    tau = 0.25 # Conflict Threshold (if score > 0.25, we flag it as conflict)
    
    if (pos > 0 and neg > 0) and (conflict_score >= tau):
        resolution = "MIXED"
    elif weighted_pos > weighted_neg:
        resolution = "Overall Positive"
    elif weighted_neg > weighted_pos:
        resolution = "Overall Negative"
    elif neu > 0:
        resolution = "Neutral"
    else:
        resolution = "No Aspect Detected"
        
    return resolution, round(conflict_score, 4)

def evaluate(model, dataloader, device, aspects, project_dir):
    model.eval() # Set to eval mode (disable dropout)
    preds = {a: [] for a in aspects}
    confs = {a: [] for a in aspects}
    trues = {a: [] for a in aspects}

    with torch.no_grad(): # No need to track gradients during validation
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            adj_matrix = batch["adj_matrix"].to(device)
            labels = batch["labels"].to(device)

            logits_list = model(input_ids, attention_mask, adj_matrix)

            for aspect_idx, a in enumerate(aspects):
                # Calculate Probabilities
                probs = torch.softmax(logits_list[aspect_idx], dim=1)
                
                # Get max probability (Confidence) and index (Prediction)
                conf, p = torch.max(probs, dim=1)
                
                preds[a].extend(p.cpu().tolist())
                confs[a].extend(conf.cpu().tolist())
                trues[a].extend(labels[:, aspect_idx].cpu().tolist())

    # --- Save Metrics ---
    metrics_path = f"{project_dir}/outputs/reports/roberta_gcn_fl_new_metrics.txt"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        for a in aspects:
            # Filter out -100 labels
            valid = [k for k, y in enumerate(trues[a]) if y != -100]
            if not valid: continue
            
            y_true = [trues[a][k] for k in valid]
            y_pred = [preds[a][k] for k in valid]
            
            report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=["negative", "neutral", "positive"], zero_division=0)
            print(f"\nAspect: {a}\n{report}")
            f.write(f"\nAspect: {a}\n{report}\n")

    # --- Save Predictions CSV ---
    print("Saving predictions & MSR Resolution...")
    inv_map = {0: "negative", 1: "neutral", 2: "positive", -100: "None"}
    texts = dataloader.dataset.texts
    
    out_rows = []
    for idx in range(len(texts)):
        row = {"text": texts[idx]}
        for a in aspects:
            row[f"{a}_pred"] = inv_map.get(preds[a][idx], "None")
            row[f"{a}_conf"] = round(confs[a][idx], 4) # Save confidence score
            row[f"{a}_true"] = inv_map.get(trues[a][idx], "None")
        
        # --- APPLY MSR RESOLUTION ---
        resolution, conflict_score = resolve_row(row, aspects)
        row["msr_resolution"] = resolution
        row["conflict_score"] = conflict_score
            
        out_rows.append(row)

    out_csv = f"{project_dir}/outputs/reports/roberta_msr_fl_new_results.csv"
    pd.DataFrame(out_rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Final MSR results saved to {out_csv}")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main(project_dir, head=None, train_file=None, val_file=None, eval_only=False, checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- STRICT GPU ENFORCEMENT ---
    if device.type != "cuda":
        raise RuntimeError(
            "CRITICAL: GPU NOT FOUND!\n"
            "This script is configured to run ONLY on a GPU.\n"
            "Please ensure you have an NVIDIA GPU and PyTorch with CUDA support installed.\n"
            "Install command: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    # ------------------------------

    set_seed(42)
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    aspects = ["stayingpower","texture","smell","price","colour","shipping","packing"]

    # Load Dataframes
    train_path = f"{project_dir}/data/splits/{train_file}" if train_file else f"{project_dir}/data/splits/train.parquet"
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_parquet(train_path)
    val_path = f"{project_dir}/data/splits/{val_file}" if val_file else f"{project_dir}/data/splits/val.parquet"
    print(f"Loading validation data from: {val_path}")
    val_df = pd.read_parquet(val_path)

    # Apply Sample Limit if requested
    if head is not None:
        print(f"--- TEST MODE: Truncating data to first {head} rows ---")
        train_df = train_df.head(head)
        val_df = val_df.head(head)

    # --- Calculate Class Weights ---
    print("Calculating class weights...")
    label_counts = { "positive": 0, "negative": 0, "neutral": 0 }
    total_labels = 0
    
    for a in aspects:
        counts = train_df[a].value_counts()
        for label in ["positive", "negative", "neutral"]:
            c = counts.get(label, 0)
            label_counts[label] += c
            total_labels += c
            
    # Order: negative=0, neutral=1, positive=2 (Same as label_map)
    weights = []
    for label in ["negative", "neutral", "positive"]: 
        count = label_counts[label]
        if count > 0:
            w = total_labels / (3.0 * count)
        else:
            w = 1.0 
        weights.append(w)
        
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Class Counts: {label_counts}")
    print(f"Class Weights (Neg, Neu, Pos): {weights}")
    # -------------------------------

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    max_len = 256
    
    # Setup Cache Directory
    cache_dir = f"{project_dir}/outputs/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Adjacency matrices
    prefix = f"head{head}_" if head else ""
    train_cache_name = f"{prefix}{train_file.replace('.parquet', '')}_adj.pkl" if train_file else f"{prefix}train_adj.pkl"
    print("Preparing Training Data...")
    train_adj = preprocess_and_save_adj(train_df, tokenizer, max_len, f"{cache_dir}/{train_cache_name}")
    
    print("Preparing Validation Data...")
    val_cache_name = f"{prefix}{val_file.replace('.parquet', '')}_adj.pkl" if val_file else f"{prefix}val_adj.pkl"
    val_adj = preprocess_and_save_adj(val_df, tokenizer, max_len, f"{cache_dir}/{val_cache_name}")

    # Initialize Datasets
    train_ds = ABSADataset(train_df, tokenizer, train_adj, aspects, max_len=max_len)
    val_ds   = ABSADataset(val_df, tokenizer, val_adj, aspects, max_len=max_len)

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Initialize Model
    model = MultiAspectRobertaGCN(num_aspects=len(aspects)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    if eval_only:
        if checkpoint is None:
            raise ValueError("Must provide --checkpoint when using --eval_only")
        print(f"Loading checkpoint from {checkpoint}...")
        model.load_state_dict(torch.load(checkpoint))
    else:
        print("Starting training with DepGCN...")
        for epoch in range(1 if head else 3): # Train only 1 epoch if in test mode
            loss = train_epoch(model, train_dl, optimizer, device, class_weights)
            print(f"Epoch {epoch+1} | Loss: {loss:.4f}")

            # Save Checkpoint
            os.makedirs(f"{project_dir}/outputs/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"{project_dir}/outputs/checkpoints/roberta_gcn_epoch{epoch}.pt")

    evaluate(model, val_dl, device, aspects, project_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--head", type=int, default=None, help="Run on first N rows only (for testing)")
    parser.add_argument("--train_file", type=str, default=None, help="Filename of the training data (e.g., train_augmented.parquet)")
    parser.add_argument("--val_file", type=str, default=None, help="Filename of the validation data (e.g., val_augmented.parquet)")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run evaluation only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load for evaluation")
    args = parser.parse_args()
    main(args.project_dir, args.head, args.train_file, args.val_file, args.eval_only, args.checkpoint)
