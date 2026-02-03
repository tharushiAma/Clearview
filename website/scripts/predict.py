#!/usr/bin/env python3
import sys
import os
import json
import torch

# Add ml-research to path using hardcoded absolute path
ML_RESEARCH_PATH = r"c:\Users\lucif\Desktop\Clearview\ml-research"
if ML_RESEARCH_PATH not in sys.path:
    sys.path.append(ML_RESEARCH_PATH)

from src.models.roberta_hierarchical_improved import ImprovedRoBERTaHierarchical
from src.data_layer._common import ASPECTS, INV_LABEL
from transformers import RobertaTokenizerFast

# Load model (cached in memory for subsequent calls in same process)
_MODEL_CACHE = {}

def get_model(ckpt_path, msr_strength):
    key = (ckpt_path, msr_strength)
    if key not in _MODEL_CACHE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        
        model = ImprovedRoBERTaHierarchical(
            num_aspects=len(ASPECTS),
            num_classes=4,
            aspect_names=ASPECTS,
            hidden_dropout=0.3,
            msr_strength=msr_strength,
            roberta_name="roberta-base"
        ).to(device)
        
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        
        _MODEL_CACHE[key] = (model, tokenizer, device)
    
    return _MODEL_CACHE[key]

def main():
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    
    text = input_data["text"]
    msr_enabled = input_data.get("msr_enabled", True)
    msr_strength = input_data.get("msr_strength", 0.3)
    
    # Default checkpoint with hardcoded absolute path
    default_ckpt = r"c:\Users\lucif\Desktop\Clearview\ml-research\outputs\gold_msr_4class\best_model.pt"
    ckpt_path = input_data.get("ckpt_path", default_ckpt)
    
    model, tokenizer, device = get_model(ckpt_path, msr_strength)
    
    # Encode
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    
    # Predict before and after
    with torch.no_grad():
        preds_b, probs_b, conf_b = model.predict(input_ids, attn, enable_msr=False)
        preds_a, probs_a, conf_a = model.predict(input_ids, attn, enable_msr=msr_enabled)
    
    aspects_res = []
    for i, asp in enumerate(ASPECTS):
        pb_cls = int(preds_b[0, i].item())
        pa_cls = int(preds_a[0, i].item())
        
        prob_b_vec = probs_b[0, i].detach().cpu().numpy().tolist()
        prob_a_vec = probs_a[0, i].detach().cpu().numpy().tolist()
        
        changed = (pb_cls != pa_cls)
        
        aspects_res.append({
            "name": asp,
            "label": INV_LABEL[pa_cls],
            "confidence": prob_a_vec[pa_cls],
            "probs": prob_a_vec,
            "before": {"label": INV_LABEL[pb_cls], "confidence": prob_b_vec[pb_cls]},
            "after": {"label": INV_LABEL[pa_cls], "confidence": prob_a_vec[pa_cls]},
            "changed_by_msr": changed
        })
    
    result = {
        "aspects": aspects_res,
        "conflict_prob": float(conf_a[0].item()),
        "timings": {"total_ms": 0}
    }
    
    # Write output to stdout
    print(json.dumps(result))

if __name__ == "__main__":
    main()
