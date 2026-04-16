"""
backend/routes/predict.py
/predict and /predict-bulk endpoints.
"""
import time
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..model_cache import get_trained_adapter

router = APIRouter()


class PredictRequest(BaseModel):
    text: str
    msr_enabled: bool = True


class BulkPredictRequest(BaseModel):
    reviews: List[str]
    msr_enabled: bool = True


@router.post("/predict")
def predict(request: PredictRequest):
    """Run single-review prediction."""
    try:
        start = time.time()
        adapter = get_trained_adapter()
        result = adapter.predict(request.text, enable_msr=request.msr_enabled)
        result["timings"] = {"total_ms": (time.time() - start) * 1000}
        return result

    except Exception as e:
        print(f"[ERR] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-bulk")
def predict_bulk(request: BulkPredictRequest):
    """Run predictions on a list of reviews and return aggregated dashboard data."""
    try:
        start = time.time()

        if not request.reviews:
            raise HTTPException(status_code=400, detail="No reviews provided")

        adapter = get_trained_adapter()
        rows = []

        for i, text in enumerate(request.reviews):
            try:
                res = adapter.predict(text, enable_msr=request.msr_enabled)
                rows.append(
                    {
                        "review_index": i,
                        "text": text,
                        "aspects": res.get("aspects", []),
                        "conflict_prob": res.get("conflict_prob", 0.0),
                    }
                )
            except Exception as e:
                print(f"[WARN] Error processing review {i}: {e}")
                rows.append(
                    {
                        "review_index": i,
                        "text": text,
                        "aspects": [],
                        "conflict_prob": 0.0,
                        "error": str(e),
                    }
                )

        # ── Aggregate statistics ────────────────────────────────────────────
        aspect_names_list: List[str] = []
        if rows and rows[0].get("aspects"):
            aspect_names_list = [a["name"] for a in rows[0]["aspects"]]

        aspect_summary = {
            asp: {"POS": 0, "NEG": 0, "NEU": 0, "NULL": 0}
            for asp in aspect_names_list
        }
        aspect_confidence: dict = {asp: [] for asp in aspect_names_list}
        mixed_count = 0
        total_processed = 0

        _LABEL_NORM = {
            "positive": "POS",
            "negative": "NEG",
            "neutral": "NEU",
            "not_mentioned": "NULL",
        }

        for row in rows:
            if row.get("error") or not row.get("aspects"):
                continue
            total_processed += 1
            labels_for_row = []
            for asp_data in row["aspects"]:
                asp_name = asp_data.get("name", "")
                label = _LABEL_NORM.get(asp_data.get("label", "NULL"), asp_data.get("label", "NULL"))
                conf = asp_data.get("confidence", 0.0)
                if asp_name in aspect_summary:
                    aspect_summary[asp_name][label] = (
                        aspect_summary[asp_name].get(label, 0) + 1
                    )
                    aspect_confidence[asp_name].append(conf)
                labels_for_row.append(label)

            non_null = [l for l in labels_for_row if l != "NULL"]
            if "POS" in non_null and "NEG" in non_null:
                mixed_count += 1

        avg_confidence = {
            asp: (sum(vals) / len(vals) if vals else 0.0)
            for asp, vals in aspect_confidence.items()
        }

        overall_counts = {"POS": 0, "NEG": 0, "NEU": 0, "NULL": 0}
        for counts in aspect_summary.values():
            for label, cnt in counts.items():
                overall_counts[label] = overall_counts.get(label, 0) + cnt

        return {
            "total_reviews": len(request.reviews),
            "total_processed": total_processed,
            "mixed_count": mixed_count,
            "overall_counts": overall_counts,
            "aspect_summary": aspect_summary,
            "avg_confidence": avg_confidence,
            "rows": rows,
            "timings": {"total_ms": (time.time() - start) * 1000},
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERR] Bulk prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
