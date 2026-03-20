"""
backend/model_cache.py
Model loading, adapter creation, and in-memory caching.
Imports from config.py so models know where to find checkpoints.
"""
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch

from .config import (
    ML_RESEARCH_DIR,
    TRAINED_CKPT,
    DEFAULT_CKPT,
)

# ── Try importing the trained model adapter ──────────────────────────────────
TrainedModelAdapter: type = None  # type: ignore[assignment]
_trained_adapter_available = False
try:
    from inference_bridge.trained_model_adapter import TrainedModelAdapter  # type: ignore[no-redef]
    _trained_adapter_available = True
    print("[OK] Trained model adapter available")
except ImportError as e:
    print(f"[WARN] Trained model adapter not available: {e}")

# ── Try importing the trained model XAI bridge ───────────────────────────────
TrainedModelXAI: type = None  # type: ignore[assignment]
_trained_xai_available = False
try:
    from inference_bridge.trained_model_xai import TrainedModelXAI  # type: ignore[no-redef]
    _trained_xai_available = True
    print("[OK] Trained model XAI bridge available")
except ImportError as e:
    print(f"[WARN] Trained model XAI bridge not available: {e}")

# ── Try importing legacy ml-research model ───────────────────────────────────
ImprovedRoBERTaHierarchical: type = None  # type: ignore[assignment]
ASPECTS: list = []
INV_LABEL: dict = {}
ClearViewExplainer: type = None  # type: ignore[assignment]
RobertaTokenizerFast: type = None  # type: ignore[assignment]
_legacy_model_available = False
if ML_RESEARCH_DIR:
    try:
        from src.models.roberta_hierarchical_improved import ImprovedRoBERTaHierarchical  # type: ignore[no-redef]
        from src.data_layer._common import ASPECTS, INV_LABEL  # type: ignore[no-redef]
        from src.xai.Explainable import ClearViewExplainer  # type: ignore[no-redef]
        from transformers import RobertaTokenizerFast  # type: ignore[no-redef]
        _legacy_model_available = True
        print("[OK] Legacy ml-research model available")
    except ImportError as e:
        print(f"[WARN] Legacy model not available: {e}")
else:
    print("[WARN] ml-research directory not found — legacy model unavailable")

# ── Decide which model path to use ───────────────────────────────────────────
import os as _os
USE_TRAINED_MODEL: bool = _trained_adapter_available and _os.path.exists(TRAINED_CKPT)
if USE_TRAINED_MODEL:
    print(f"[MODEL] Using NEWLY TRAINED model: {TRAINED_CKPT}")
else:
    print(f"[LOAD]  Using LEGACY model: {DEFAULT_CKPT}")

# ── In-memory caches ─────────────────────────────────────────────────────────
_MODEL_CACHE: Dict[Tuple, Tuple] = {}
_EXPLAINER_CACHE: Dict[Tuple, Any] = {}
_TRAINED_ADAPTER: Optional[Any] = None
_TRAINED_XAI: Optional[Any] = None


def get_trained_adapter() -> Any:
    """Return (and cache) the trained model adapter."""
    global _TRAINED_ADAPTER
    if _TRAINED_ADAPTER is None:
        print(f"\n[LOAD] Loading trained model adapter from: {TRAINED_CKPT}")
        t = time.time()
        _TRAINED_ADAPTER = TrainedModelAdapter(TRAINED_CKPT)
        print(f"[OK]   Trained adapter loaded in {time.time() - t:.2f}s")
    return _TRAINED_ADAPTER


def get_trained_xai() -> Any:
    """Return (and cache) the trained model XAI bridge."""
    global _TRAINED_XAI
    if _TRAINED_XAI is None:
        print(f"\n[LOAD] Loading trained model XAI bridge from: {TRAINED_CKPT}")
        t = time.time()
        _TRAINED_XAI = TrainedModelXAI(TRAINED_CKPT)
        print(f"[OK]   Trained XAI bridge loaded in {time.time() - t:.2f}s")
    return _TRAINED_XAI


def get_model(ckpt_path: str, msr_strength: float) -> Tuple:
    """Return (model, tokenizer, device), loading + caching on first call."""
    if USE_TRAINED_MODEL:
        return None, None, None  # handled by get_trained_adapter()

    if not _legacy_model_available:
        raise RuntimeError(
            "No model available: trained adapter failed and legacy model not found."
        )

    key = (ckpt_path, msr_strength)
    if key not in _MODEL_CACHE:
        print(f"\n[LOAD] Loading legacy model from: {ckpt_path}")
        t = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")  # type: ignore[attr-defined]
        model = ImprovedRoBERTaHierarchical(
            num_aspects=len(ASPECTS),
            num_classes=4,
            aspect_names=ASPECTS,
            hidden_dropout=0.3,
            msr_strength=msr_strength,
            roberta_name="roberta-base",
        ).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        print(f"[OK]   Legacy model loaded in {time.time() - t:.2f}s")
        _MODEL_CACHE[key] = (model, tokenizer, device)
    else:
        print("[CACHE] Using cached legacy model")
    return _MODEL_CACHE[key]


def get_explainer(ckpt_path: str, msr_strength: float) -> Any:
    """Return (and cache) a ClearViewExplainer for the legacy model."""
    key = (ckpt_path, msr_strength)
    if key not in _EXPLAINER_CACHE:
        print("\n[LOAD] Loading XAI explainer...")
        t = time.time()
        _EXPLAINER_CACHE[key] = ClearViewExplainer(ckpt_path, msr_strength=msr_strength)
        print(f"[OK]   Explainer loaded in {time.time() - t:.2f}s")
    else:
        print("[CACHE] Using cached explainer")
    return _EXPLAINER_CACHE[key]


def preload_models() -> None:
    """Pre-load all models into memory. Called from the FastAPI lifespan."""
    print("\n" + "=" * 60)
    print("[START] Pre-loading models into memory...")
    print("=" * 60)
    try:
        if USE_TRAINED_MODEL:
            get_trained_adapter()
            if _trained_xai_available:
                get_trained_xai()
            print("\n[OK] Newly trained model + XAI bridge ready!")
        else:
            get_model(DEFAULT_CKPT, 0.3)
            get_explainer(DEFAULT_CKPT, 0.3)
            print("\n[OK] Legacy model loaded and ready!")
        print("[MODEL] Server is ready to handle requests")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n[ERR] ERROR loading models: {e}")
        print("Server will start but predictions will fail until models load.")
