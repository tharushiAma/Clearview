
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast
import sys
import os

# Mock the project structure
sys.path.append(os.getcwd())
sys.path.append(r"c:\Users\TharushiAmasha\Downloads\outputs\src\models")

try:
    from msr_resolver_roberta import MultiAspectRobertaGCN, dependency_adj_matrix, ABSADataset
except ImportError:
    print("Could not import msr_resolver_roberta. Please run this script from the correct context.")
    exit(1)

def test_model():
    print("Testing Model with Dummy Data...")
    
    # 1. Mock Data
    text = "The colors are great but the smell is terrible."
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    max_len = 32
    
    # 2. Test Adjacency Matrix Construction and Normalization
    print("Testing Adjacency Matrix...")
    try:
        adj = dependency_adj_matrix(text, tokenizer, max_len)
        print(f"Adjacency Matrix Shape: {adj.shape}")
        assert adj.shape == (max_len, max_len)
        
        # Check Normalization: Row sums should roughly be related to degrees, not just integers
        # A simple check is that it's not all 0s or 1s
        print(f"Matrix Sample (first 5x5): \n{adj[:5,:5]}")
        print("Adjacency Matrix Construction SUCCESS")
    except Exception as e:
        print(f"ADJACENCY MATRIX FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Test Model Forward Pass
    print("Testing Model Forward Pass...")
    num_aspects = 3
    model = MultiAspectRobertaGCN(num_aspects=num_aspects, num_classes=3)
    
    # Create dummy batch
    input_ids = tokenizer(text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
    attention_mask = tokenizer(text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")["attention_mask"]
    adj_matrix = torch.tensor(adj).unsqueeze(0) # Batch size 1
    
    try:
        logits = model(input_ids, attention_mask, adj_matrix)
        print(f"Logits output list length: {len(logits)}")
        print(f"Logits shape for aspect 0: {logits[0].shape}")
        assert len(logits) == num_aspects
        assert logits[0].shape == (1, 3) # [Batch, Num_Classes]
        print("Model Forward Pass SUCCESS")
    except Exception as e:
        print(f"MODEL FORWARD PASS FAILED: {e}")
        import traceback
        traceback.print_exc()

    # 4. Test Dataset Class (New Signature)
    print("Testing Dataset Class...")
    try:
        df = pd.DataFrame({"text_clean": [text], "aspect1": ["positive"], "aspect2": ["negative"], "aspect3": ["neutral"]})
        aspects = ["aspect1", "aspect2", "aspect3"]
        adj_list = [adj] # Mock pre-computed list
        
        ds = ABSADataset(df, tokenizer, adj_list, aspects, max_len=max_len)
        item = ds[0]
        assert "adj_matrix" in item
        assert item["adj_matrix"].shape == (max_len, max_len)
        print("Dataset Logic SUCCESS")
    except Exception as e:
        print(f"DATASET FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
