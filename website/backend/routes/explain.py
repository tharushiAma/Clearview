"""
backend/routes/explain.py
/explain endpoint — XAI attribution analysis.
"""
import time
import traceback
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..model_cache import (
    _trained_xai_available,
    get_trained_xai,
)

router = APIRouter()


class ExplainRequest(BaseModel):
    text: str
    aspect: str = "all"
    methods: List[str] = ["ig"]
    msr_enabled: bool = True


@router.post("/explain")
def explain(request: ExplainRequest):
    """Generate XAI explanations (Integrated Gradients, LIME, SHAP)."""
    try:
        start = time.time()
        print("\n" + "=" * 60)
        print("[XAI] Starting analysis")
        print("=" * 60)

        if not _trained_xai_available:
            raise HTTPException(
                status_code=503,
                detail="XAI bridge is not available. Check that inference_bridge/trained_model_xai.py can be imported.",
            )

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

        from inference_bridge.inference import is_mentioned
        
        total = len(aspect_list)
        print(f"[XAI] Step 2/2: Analysing {total} aspect(s)...")
        for idx, asp in enumerate(aspect_list, 1):
            if asp not in ex.aspect_names:
                continue
            
            # If requesting 'all' aspects, skip aspects completely absent from the text
            # to massively speed up XAI requests.
            if request.aspect == "all" and not is_mentioned(request.text, asp):
                print(f"  [{idx}/{total}] Skipping '{asp}' (not mentioned)")
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

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERR] Explanation error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
