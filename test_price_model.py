import torch
import numpy as np
import spacy
from transformers import RobertaTokenizerFast
from src.models.msr_resolver_roberta_focal_loss import MultiAspectRobertaGCN, dependency_adj_matrix

# Load Spacy
nlp = spacy.load("en_core_web_sm")

def test_custom_sentences(model_path, sentences, device):
    print(f"Loading model from {model_path}...")
    
    # 1. Setup
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    aspects = ["stayingpower","texture","smell","price","colour","shipping","packing"]
    max_len = 256
    
    # 2. Load Model
    model = MultiAspectRobertaGCN(num_aspects=len(aspects)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("\n--- RUNNING INFERENCE ---")
    
    correct = 0
    
    for text in sentences:
        # Preprocess
        enc = tokenizer(text, padding="max_length", truncation=True, max_length=max_len, return_offsets_mapping=True)
        input_ids = torch.tensor(enc["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(enc["attention_mask"]).unsqueeze(0).to(device)
        
        # Compute Adjacency Matrix (On the fly)
        # Note: We need to use the imported function and wrap logic slightly if needed, 
        # but here we can just call the logic directly or reuse the function if it handles single inputs well.
        # The function in src.models... takes (text, tokenizer, max_len)
        adj = dependency_adj_matrix(text, tokenizer, max_len)
        adj_matrix = torch.tensor(adj, dtype=torch.float).unsqueeze(0).to(device)
        
        # Prediction
        with torch.no_grad():
            logits_list = model(input_ids, attention_mask, adj_matrix)
            
            # Price is index 3 in the aspects list ["stayingpower","texture","smell","price",...]
            price_logits = logits_list[3] 
            probs = torch.softmax(price_logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            pred_label = label_map[pred_idx.item()]
            
            # Print Result
            is_neg = (pred_label == "negative")
            symbol = "✅" if is_neg else "❌"
            if is_neg: correct += 1
            
            print(f"{symbol} Text: '{text}'")
            print(f"   Pred: {pred_label.upper()} (Conf: {conf.item():.4f})")
            print("-" * 50)

    print(f"\nSummary: Detected {correct}/{len(sentences)} Negative Price Reviews.")

if __name__ == "__main__":
    # Test sentences that were NOT in the training set (Verification Set)
    test_sentences = [
        "I honestly think it is too expensive for a drugstore brand.",
        "The color is nice but $35 is robbery.",
        "I would buy it again if it wasn't so pricey.",
        "Overpriced compared to simpler products.",
        "It costs way too much.",
        "Great lipstick, terrible price.",
        "Not worth the high cost.",
        "Spending 40 bucks on this was a mistake.",
        "The value simply isn't there.",
        "Way too much money for such a tiny tube."
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Using the last checkpoint (Epoch 2 -> 3rd epoch)
    model_path = "outputs/checkpoints/roberta_gcn_epoch2.pt" 
    
    test_custom_sentences(model_path, test_sentences, device)
