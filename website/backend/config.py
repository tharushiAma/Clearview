"""
backend/config.py
All environment-sourced configuration and derived path constants.
"""
import os
import sys

# ── Project paths ────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
# backend/ lives inside website/, so project_root is two levels up
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# ── ml-research path resolution ──────────────────────────────────────────────
# Add ml-research/ to sys.path so inference_bridge imports work correctly
_ml_research_dir = os.path.join(project_root, "ml-research")
if os.path.exists(_ml_research_dir) and _ml_research_dir not in sys.path:
    sys.path.insert(0, _ml_research_dir)

# ── Checkpoint path (env-var overridable) ────────────────────────────────────
_default_trained = os.path.join(
    project_root, "ml-research", "outputs", "cosmetic_sentiment_v1", "best_model.pt"
)
TRAINED_CKPT: str = os.environ.get("CKPT_PATH", _default_trained)

# ── CORS (env-var overridable) ───────────────────────────────────────────────
_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _origins_env.split(",") if o.strip()]
    if _origins_env
    else ["http://localhost:3000", "http://127.0.0.1:3000"]
)
