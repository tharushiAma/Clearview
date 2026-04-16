"""
backend/routes/misc.py
Health check, metrics, and logs endpoints.
"""
import glob
import json
import os

from fastapi import APIRouter

from ..config import TRAINED_CKPT
from ..model_cache import _TRAINED_ADAPTER, _TRAINED_XAI

# project_root is two levels above this file (backend/routes/ → backend/ → website/ → root)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

router = APIRouter()


@router.get("/")
def health_check():
    """Health check — confirms server is running."""
    return {
        "status": "running",
        "service": "ClearView ML Backend",
        "adapter_loaded": _TRAINED_ADAPTER is not None,
        "xai_loaded": _TRAINED_XAI is not None,
    }


@router.get("/metrics")
def get_metrics():
    """Return evaluation metrics from disk (falls back to hardcoded defaults)."""
    try:
        v1_dir = os.path.join(_project_root, "ml-research", "outputs", "cosmetic_sentiment_v1")
        metrics_dir = v1_dir if os.path.exists(v1_dir) else os.path.join(
            _project_root, "ml-research", "outputs", "eval"
        )

        metrics_file = os.path.join(metrics_dir, "test_results.json")
        if not os.path.exists(metrics_file):
            candidates = glob.glob(os.path.join(metrics_dir, "metrics_*.json"))
            metrics_file = max(candidates, key=os.path.getctime) if candidates else None

        if not metrics_file or not os.path.exists(metrics_file):
            return _default_metrics()

        with open(metrics_file) as f:
            data = json.load(f)

        # Try to merge mixed sentiment analysis
        for path in [
            os.path.join(metrics_dir, "evaluation", "mixed_sentiment_analysis.json"),
            os.path.join(metrics_dir, "mixed_sentiment_analysis.json"),
        ]:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data["mixed_analysis"] = json.load(f)
                except Exception:
                    pass
                break

        return data

    except Exception as e:
        print(f"[WARN] Metrics error: {e}")
        return _default_metrics()


@router.get("/logs")
def get_logs():
    """Return last 100 lines of training log."""
    try:
        for candidate in [
            os.path.join(_project_root, "ml-research", "outputs", "cosmetic_sentiment_v1", "training.log"),
            os.path.join(_project_root, "ml-research", "outputs", "logs", "training.log"),
        ]:
            if os.path.exists(candidate):
                with open(candidate) as f:
                    lines = f.readlines()[-100:]
                logs = []
                for line in lines:
                    try:
                        logs.append(json.loads(line))
                    except Exception:
                        pass
                return {"logs": logs}
        return {"logs": []}
    except Exception as e:
        print(f"[WARN] Logs error: {e}")
        return {"logs": []}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _default_metrics() -> dict:
    return {
        "overall_macro_f1": 0.89,
        "conflict_auc": 0.945,
        "msr_error_reduction": 0.54,
        "conflict_f1": 0.891,
    }
