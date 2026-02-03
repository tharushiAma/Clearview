#!/usr/bin/env python3
"""
ClearView FastAPI Backend Server
Loads ML models once at startup and keeps them in memory for instant predictions.
"""

import sys
import os
import json
import time
from typing import Optional, List, Dict, Any
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add ml-research to path
ML_RESEARCH_PATH = r"c:\Users\lucif\Desktop\Clearview\ml-research"
if ML_RESEARCH_PATH not in sys.path:
    sys.path.append(ML_RESEARCH_PATH)

from src.models.roberta_hierarchical_improved import ImprovedRoBERTaHierarchical
from src.data_layer._common import ASPECTS, INV_LABEL
from src.xai.Explainable import ClearViewExplainer
from transformers import RobertaTokenizerFast

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class PredictRequest(BaseModel):
    text: str
    msr_enabled: bool = True
    msr_strength: float = 0.3
    ckpt_path: Optional[str] = None

class ExplainRequest(BaseModel):
    text: str
    aspect: str = "all"
    methods: List[str] = ["ig"]
    msr_enabled: bool = True
    msr_strength: float = 0.3
    ckpt_path: Optional[str] = None

# ============================================================================
# Global Model Cache
# ============================================================================

# Model cache: key = (ckpt_path, msr_strength), value = (model, tokenizer, device)
_MODEL_CACHE: Dict[tuple, tuple] = {}
_EXPLAINER_CACHE: Dict[tuple, Any] = {}

# Default checkpoint path
DEFAULT_CKPT = r"c:\Users\lucif\Desktop\Clearview\ml-research\outputs\gold_msr_4class\best_model.pt"

print("=" * 80)
print("Starting ClearView Backend Server")
print("=" * 80)

def get_model(ckpt_path: str, msr_strength: float):
    """Load model or retrieve from cache. This is called at startup."""
    key = (ckpt_path, msr_strength)
    
    if key not in _MODEL_CACHE:
        print(f"\n🔄 Loading model from: {ckpt_path}")
        print(f"   MSR Strength: {msr_strength}")
        start_time = time.time()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {device}")
        
        # Load tokenizer
        print("   Loading RoBERTa tokenizer...")
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        
        # Create model
        print("   Creating model architecture...")
        model = ImprovedRoBERTaHierarchical(
            num_aspects=len(ASPECTS),
            num_classes=4,
            aspect_names=ASPECTS,
            hidden_dropout=0.3,
            msr_strength=msr_strength,
            roberta_name="roberta-base"
        ).to(device)
        
        # Load checkpoint
        print("   Loading checkpoint weights...")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        
        elapsed = time.time() - start_time
        print(f"✅ Model loaded successfully in {elapsed:.2f} seconds")
        
        _MODEL_CACHE[key] = (model, tokenizer, device)
    else:
        print(f"♻️  Using cached model for {ckpt_path}")
    
    return _MODEL_CACHE[key]

def get_explainer(ckpt_path: str, msr_strength: float):
    """Load explainer or retrieve from cache."""
    key = (ckpt_path, msr_strength)
    
    if key not in _EXPLAINER_CACHE:
        print(f"\n🔄 Loading XAI explainer...")
        start_time = time.time()
        
        explainer = ClearViewExplainer(ckpt_path, msr_strength=msr_strength)
        
        elapsed = time.time() - start_time
        print(f"✅ Explainer loaded in {elapsed:.2f} seconds")
        
        _EXPLAINER_CACHE[key] = explainer
    else:
        print(f"♻️  Using cached explainer")
    
    return _EXPLAINER_CACHE[key]

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="ClearView ML Backend",
    description="Persistent ML model server for aspect-based sentiment analysis",
    version="1.0.0"
)

# Add CORS middleware to allow Next.js to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Startup Event - Load Models
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models into memory at startup."""
    print("\n" + "=" * 80)
    print("🚀 STARTUP: Pre-loading models into memory...")
    print("=" * 80)
    
    try:
        # Pre-load the default model
        get_model(DEFAULT_CKPT, 0.3)
        
        # Pre-load the explainer
        get_explainer(DEFAULT_CKPT, 0.3)
        
        print("\n" + "=" * 80)
        print("✅ All models loaded successfully!")
        print("🎯 Server is ready to handle requests instantly")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"\n❌ ERROR loading models: {e}")
        print("Server will start but predictions will fail until models are loaded.")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "ClearView ML Backend",
        "models_loaded": len(_MODEL_CACHE),
        "explainers_loaded": len(_EXPLAINER_CACHE)
    }

@app.post("/predict")
def predict(request: PredictRequest):
    """Run prediction with MSR."""
    try:
        start_time = time.time()
        
        # Get model
        ckpt_path = request.ckpt_path or DEFAULT_CKPT
        model, tokenizer, device = get_model(ckpt_path, request.msr_strength)
        
        # Encode input
        enc = tokenizer(
            request.text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        
        # Predict before and after MSR
        with torch.no_grad():
            preds_b, probs_b, conf_b = model.predict(input_ids, attn, enable_msr=False)
            preds_a, probs_a, conf_a = model.predict(input_ids, attn, enable_msr=request.msr_enabled)
        
        # Format results
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
        
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "aspects": aspects_res,
            "conflict_prob": float(conf_a[0].item()),
            "timings": {"total_ms": elapsed}
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain(request: ExplainRequest):
    """Generate XAI explanations."""
    try:
        start_time = time.time()
        
        print("\n" + "="*60)
        print(f"🔍 Starting XAI Analysis")
        print("="*60)
        
        # Get explainer
        ckpt_path = request.ckpt_path or DEFAULT_CKPT
        print(f"📊 Step 1/3: Loading explainer...")
        ex = get_explainer(ckpt_path, request.msr_strength)
        
        # Clear the internal IG cache for this new request
        if hasattr(ex, "_ig_cache"):
            ex._ig_cache.clear()
            
        print(f"✅ Explainer loaded")
        
        # Generate conflict explanation
        print(f"📊 Step 2/3: Computing conflict explanation...")
        bundle = {
            "text": request.text,
            "requested_aspect": request.aspect,
            "ig_conflict": ex.explain_ig_conflict(request.text, enable_msr=True, top_k=10),
            "aspects": {},
            "progress": []  # Track progress for frontend
        }
        print(f"✅ Conflict explanation complete")
        
        aspect_list = ASPECTS if request.aspect == "all" else [request.aspect]
        total_aspects = len(aspect_list)
        
        print(f"📊 Step 3/3: Analyzing {total_aspects} aspect(s)...")
        
        for idx, asp in enumerate(aspect_list, 1):
            if asp not in ASPECTS:
                continue
            
            print(f"  🔍 [{idx}/{total_aspects}] Analyzing '{asp}' aspect...")
            
            asp_data = {
                "ig_aspect": ex.explain_ig_aspect(request.text, asp, enable_msr=True, top_k=10),
                "msr_delta": ex.explain_msr_delta(request.text, asp, top_k=10)
            }
            
            bundle["aspects"][asp] = asp_data
            bundle["progress"].append(f"Completed {asp}")
            print(f"  ✅ '{asp}' complete")
        
        elapsed = (time.time() - start_time) * 1000
        bundle["timings"] = {"total_ms": elapsed}
        
        print(f"\n✅ XAI Analysis Complete!")
        print(f"⏱️  Total time: {elapsed/1000:.1f} seconds")
        print("="*60 + "\n")
        
        return bundle
        
    except Exception as e:
        print(f"❌ Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 Starting ClearView Backend Server")
    print("=" * 80)
    print(f"📍 Server will run on: http://localhost:8000")
    print(f"📍 Health check: http://localhost:8000/")
    print(f"📍 Docs: http://localhost:8000/docs")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
