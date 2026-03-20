"""
backend/routes/explain.py
/explain endpoint — XAI attribution analysis.
"""
import time
import traceback
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import DEFAULT_CKPT
from ..model_cache import (
    USE_TRAINED_MODEL,
    _trained_xai_available,
    _legacy_model_available,
    get_trained_xai,
    get_explainer,
)

try:
    from src.data_layer._common import ASPECTS
except ImportError:
    ASPECTS = []

router = APIRouter()


class ExplainRequest(BaseModel):
    text: str
    aspect: str = "all"
    methods: List[str] = ["ig"]
    msr_enabled: bool = True
    msr_strength: float = 0.3
    ckpt_path: str | None = None


@router.post("/explain")
def explain(request: ExplainRequest):
    """Generate XAI explanations (Integrated Gradients, LIME, SHAP)."""
    try:
        start = time.time()
        print("\n" + "=" * 60)
        print("[XAI] Starting analysis")
        print("=" * 60)

        # ── Trained model XAI (preferred) ────────────────────────────────
        if USE_TRAINED_MODEL and _trained_xai_available:
            print("[XAI] Using trained model XAI bridge (attention-based)")
            ex = get_trained_xai()
            aspect_list = ex.aspect_names if request.aspect == "all" else [request.aspect]

            print("[XAI] Step 1/2: Computing conflict explanation...")
            ig_conflict = ex.explain_ig_conflict(request.text, enable_msr=True, top_k=10)
            print("[OK]  Conflict explanation complete")

            bundle = {
                "text": request.text,
                "requested_aspect": request.aspect,
                "ig_conflict": ig_conflict,
                "aspects": {},
                "progress": [],
            }

            total = len(aspect_list)
            print(f"[XAI] Step 2/2: Analysing {total} aspect(s)...")
            for idx, asp in enumerate(aspect_list, 1):
                if asp not in ex.aspect_names:
                    continue
                print(f"  [{idx}/{total}] Analysing '{asp}'...")
                asp_data: dict = {}

                if "ig" in request.methods:
                    asp_data["ig_aspect"] = ex.explain_ig_aspect(
                        request.text, asp, enable_msr=True, top_k=10
                    )
                if "lime" in request.methods:
                    asp_data["lime_aspect"] = ex.explain_lime_aspect(
                        request.text, asp, top_k=10
                    )
                if "shap" in request.methods:
                    asp_data["shap_aspect"] = ex.explain_shap_aspect(
                        request.text, asp, top_k=10
                    )

                bundle["aspects"][asp] = asp_data
                bundle["progress"].append(f"Completed {asp}")
                print(f"  [OK] '{asp}' done")

            elapsed = (time.time() - start) * 1000
            bundle["timings"] = {"total_ms": elapsed}
            print(f"\n[OK] XAI complete — {elapsed / 1000:.1f}s")
            print("=" * 60 + "\n")
            return bundle

        # ── Legacy ClearViewExplainer fallback ────────────────────────────
        if not _legacy_model_available:
            raise HTTPException(
                status_code=503,
                detail="No XAI backend available. Trained model XAI bridge failed to load.",
            )

        ckpt_path = request.ckpt_path or DEFAULT_CKPT
        print("[XAI] Step 1/3: Loading legacy explainer...")
        ex = get_explainer(ckpt_path, request.msr_strength)
        if hasattr(ex, "_ig_cache"):
            ex._ig_cache.clear()

        print("[XAI] Step 2/3: Computing conflict explanation...")
        bundle = {
            "text": request.text,
            "requested_aspect": request.aspect,
            "ig_conflict": ex.explain_ig_conflict(request.text, enable_msr=True, top_k=10),
            "aspects": {},
            "progress": [],
        }

        aspect_list = ASPECTS if request.aspect == "all" else [request.aspect]
        total = len(aspect_list)
        print(f"[XAI] Step 3/3: Analysing {total} aspect(s)...")
        for idx, asp in enumerate(aspect_list, 1):
            if asp not in ASPECTS:
                continue
            print(f"  [{idx}/{total}] Analysing '{asp}'...")
            bundle["aspects"][asp] = {
                "ig_aspect": ex.explain_ig_aspect(
                    request.text, asp, enable_msr=True, top_k=10
                ),
                "msr_delta": ex.explain_msr_delta(request.text, asp, top_k=10),
            }
            bundle["progress"].append(f"Completed {asp}")
            print(f"  [OK] '{asp}' done")

        elapsed = (time.time() - start) * 1000
        bundle["timings"] = {"total_ms": elapsed}
        print(f"\n[OK] XAI complete — {elapsed / 1000:.1f}s")
        print("=" * 60 + "\n")
        return bundle

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERR] Explanation error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
