"""
backend/model_cache.py
Model loading, adapter creation, and in-memory caching.
Imports from config.py so models know where to find checkpoints.
"""
import sys
import time
from typing import Any, Optional

from .config import TRAINED_CKPT

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

# ── In-memory caches ─────────────────────────────────────────────────────────
_TRAINED_ADAPTER: Optional[Any] = None
_TRAINED_XAI: Optional[Any] = None


def get_trained_adapter() -> Any:
    """Return (and cache) the trained model adapter."""
    global _TRAINED_ADAPTER
    if _TRAINED_ADAPTER is None:
        if not _trained_adapter_available:
            raise RuntimeError(
                "Trained model adapter is not available. "
                "Check that inference_bridge/trained_model_adapter.py can be imported."
            )
        print(f"\n[LOAD] Loading trained model adapter from: {TRAINED_CKPT}")
        t = time.time()
        _TRAINED_ADAPTER = TrainedModelAdapter(TRAINED_CKPT)
        print(f"[OK]   Trained adapter loaded in {time.time() - t:.2f}s")
    return _TRAINED_ADAPTER


def get_trained_xai() -> Any:
    """Return (and cache) the trained model XAI bridge."""
    global _TRAINED_XAI
    if _TRAINED_XAI is None:
        if not _trained_xai_available:
            raise RuntimeError(
                "Trained model XAI bridge is not available. "
                "Check that inference_bridge/trained_model_xai.py can be imported."
            )
        print(f"\n[LOAD] Loading trained model XAI bridge from: {TRAINED_CKPT}")
        t = time.time()
        _TRAINED_XAI = TrainedModelXAI(TRAINED_CKPT)
        print(f"[OK]   Trained XAI bridge loaded in {time.time() - t:.2f}s")
    return _TRAINED_XAI


def preload_models() -> None:
    """Pre-load all models into memory. Called from the FastAPI lifespan."""
    print("\n" + "=" * 60)
    print("[START] Pre-loading models into memory...")
    print("=" * 60)
    try:
        get_trained_adapter()
        if _trained_xai_available:
            get_trained_xai()
        print("\n[OK] Trained model + XAI bridge ready!")
        print("[MODEL] Server is ready to handle requests")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n[ERR] ERROR loading models: {e}")
        print("Server will start but predictions will fail until models load.")
