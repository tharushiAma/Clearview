#!/usr/bin/env python3
import sys
import os
import json
import torch

# Add ml-research to path using hardcoded absolute path
ML_RESEARCH_PATH = r"c:\Users\lucif\Desktop\Clearview\ml-research"
if ML_RESEARCH_PATH not in sys.path:
    sys.path.append(ML_RESEARCH_PATH)

from src.xai.Explainable import ClearViewExplainer
from src.data_layer._common import ASPECTS

_EXPLAINER_CACHE = {}

def get_explainer(ckpt_path, msr_strength):
    key = (ckpt_path, msr_strength)
    if key not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[key] = ClearViewExplainer(ckpt_path, msr_strength=msr_strength)
    return _EXPLAINER_CACHE[key]

def main():
    input_data = json.loads(sys.stdin.read())
    
    text = input_data["text"]
    aspect = input_data.get("aspect", "all")
    methods = input_data.get("methods", ["ig"])
    msr_enabled = input_data.get("msr_enabled", True)
    msr_strength = input_data.get("msr_strength", 0.3)
    
    # Default checkpoint with hardcoded absolute path
    default_ckpt = r"c:\Users\lucif\Desktop\Clearview\ml-research\outputs\gold_msr_4class\best_model.pt"
    ckpt_path = input_data.get("ckpt_path", default_ckpt)
    
    ex = get_explainer(ckpt_path, msr_strength)
    
    bundle = {
        "text": text,
        "requested_aspect": aspect,
        "ig_conflict": ex.explain_ig_conflict(text, enable_msr=True, top_k=10),
        "aspects": {}
    }
    
    aspect_list = ASPECTS if aspect == "all" else [aspect]
    
    for asp in aspect_list:
        if asp not in ASPECTS:
            continue
        
        asp_data = {
            "ig_aspect": ex.explain_ig_aspect(text, asp, enable_msr=True, top_k=10),
            "msr_delta": ex.explain_msr_delta(text, asp, top_k=10)
        }
        
        bundle["aspects"][asp] = asp_data
    
    print(json.dumps(bundle))

if __name__ == "__main__":
    main()
