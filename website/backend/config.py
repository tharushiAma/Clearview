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
ML_RESEARCH_DIR: str = ""
for candidate in [
    os.path.join(project_root, "ml-research"),
    os.path.join(current_dir, "..", "ml-research"),
]:
    if os.path.exists(candidate):
        ML_RESEARCH_DIR = candidate
        break

if ML_RESEARCH_DIR and ML_RESEARCH_DIR not in sys.path:
    sys.path.insert(0, ML_RESEARCH_DIR)

# ── Checkpoint paths (env-var overridable) ───────────────────────────────────
_default_trained = os.path.join(
    project_root, "ml-research", "outputs", "cosmetic_sentiment_v1", "best_model.pt"
)
TRAINED_CKPT: str = os.environ.get("CKPT_PATH", _default_trained)

_default_legacy = os.path.join(
    project_root, "ml-research", "outputs", "gold_msr_4class", "best_model.pt"
)
DEFAULT_CKPT: str = os.environ.get("LEGACY_CKPT_PATH", _default_legacy)
if not os.path.exists(DEFAULT_CKPT):
    DEFAULT_CKPT = TRAINED_CKPT

# ── CORS (env-var overridable) ───────────────────────────────────────────────
_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _origins_env.split(",") if o.strip()]
    if _origins_env
    else ["http://localhost:3000", "http://127.0.0.1:3000"]
)
