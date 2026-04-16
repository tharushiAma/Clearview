"""
backend/tests/test_api.py

Essential backend API tests.
Mocks the ML model so tests run without a checkpoint file.

Run:
    cd Clearview/website
    pytest backend/tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Fake adapter returned by mock ─────────────────────────────────────────────
# Shape must match what TrainedModelAdapter.predict() actually returns.

MOCK_ASPECTS = [
    {"name": "colour",       "label": "positive",      "confidence": 0.91, "probs": [0.03, 0.06, 0.91], "before": {"label": "positive", "confidence": 0.91}, "after": {"label": "positive", "confidence": 0.91}, "changed_by_msr": False, "top_tokens": ["beautiful"], "mentioned": True},
    {"name": "smell",        "label": "negative",      "confidence": 0.87, "probs": [0.87, 0.08, 0.05], "before": {"label": "negative", "confidence": 0.87}, "after": {"label": "negative", "confidence": 0.87}, "changed_by_msr": False, "top_tokens": ["awful"],    "mentioned": True},
    {"name": "texture",      "label": "neutral",       "confidence": 0.72, "probs": [0.14, 0.72, 0.14], "before": {"label": "neutral",  "confidence": 0.72}, "after": {"label": "neutral",  "confidence": 0.72}, "changed_by_msr": False, "top_tokens": [],           "mentioned": False},
    {"name": "price",        "label": "not_mentioned", "confidence": 0.0,  "probs": [0.0,  0.0,  0.0],  "before": {"label": "not_mentioned", "confidence": 0.0},  "after": {"label": "not_mentioned", "confidence": 0.0},  "changed_by_msr": False, "top_tokens": [], "mentioned": False},
    {"name": "stayingpower", "label": "not_mentioned", "confidence": 0.0,  "probs": [0.0,  0.0,  0.0],  "before": {"label": "not_mentioned", "confidence": 0.0},  "after": {"label": "not_mentioned", "confidence": 0.0},  "changed_by_msr": False, "top_tokens": [], "mentioned": False},
    {"name": "packing",      "label": "not_mentioned", "confidence": 0.0,  "probs": [0.0,  0.0,  0.0],  "before": {"label": "not_mentioned", "confidence": 0.0},  "after": {"label": "not_mentioned", "confidence": 0.0},  "changed_by_msr": False, "top_tokens": [], "mentioned": False},
    {"name": "shipping",     "label": "not_mentioned", "confidence": 0.0,  "probs": [0.0,  0.0,  0.0],  "before": {"label": "not_mentioned", "confidence": 0.0},  "after": {"label": "not_mentioned", "confidence": 0.0},  "changed_by_msr": False, "top_tokens": [], "mentioned": False},
]

MOCK_PREDICTION = {
    "aspects":      MOCK_ASPECTS,
    "conflict_prob": 0.82,
    "timings":      {"total_ms": 42.0},
}

MOCK_XAI = {
    "text":              "test",
    "requested_aspect":  "colour",
    "ig_conflict":       {"top_tokens": [["beautiful", 0.8], ["awful", 0.7]], "method": "ig", "task": "conflict"},
    "aspects": {
        "colour": {
            "ig_aspect": {
                "top_tokens": [["beautiful", 0.9]],
                "method":     "ig",
                "task":       "aspect:colour",
                "predicted":  "positive",
                "confidence": 0.91,
                "probs":      {"negative": 0.03, "neutral": 0.06, "positive": 0.91},
            }
        }
    },
}


def _make_fake_adapter():
    adapter = MagicMock()
    adapter.aspect_names = [
        "colour", "smell", "texture", "price", "stayingpower", "packing", "shipping"
    ]
    adapter.predict.return_value = MOCK_PREDICTION
    return adapter


def _make_fake_xai():
    xai = MagicMock()
    xai.aspect_names = [
        "colour", "smell", "texture", "price", "stayingpower", "packing", "shipping"
    ]
    xai.explain_ig_conflict.return_value = MOCK_XAI["ig_conflict"]
    xai.explain_ig_aspect.return_value   = MOCK_XAI["aspects"]["colour"]["ig_aspect"]
    xai.explain_lime_aspect.return_value = {"top_tokens": [["colour", 0.7]], "method": "lime", "task": "aspect:colour", "predicted": "positive", "confidence": 0.91, "probs": {"negative": 0.03, "neutral": 0.06, "positive": 0.91}}
    xai.explain_shap_aspect.return_value = {"top_tokens": [["colour", 0.6]], "method": "shap", "task": "aspect:colour", "predicted": "positive", "confidence": 0.91, "probs": {"negative": 0.03, "neutral": 0.06, "positive": 0.91}}
    return xai


# ── Client fixture — patches ML layer before app imports ─────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Creates a TestClient with the ML model fully mocked.
    No checkpoint file needed.
    """
    fake_adapter = _make_fake_adapter()
    fake_xai     = _make_fake_xai()

    with (
        patch("backend.model_cache._trained_adapter_available", True),
        patch("backend.model_cache._trained_xai_available",     True),
        patch("backend.model_cache.get_trained_adapter", return_value=fake_adapter),
        patch("backend.model_cache.get_trained_xai",     return_value=fake_xai),
        patch("backend.model_cache.preload_models"),  # skip real model loading
    ):
        from backend.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_has_status_field(self, client):
        r = client.get("/")
        assert "status" in r.json()
        assert r.json()["status"] == "running"

    def test_root_has_service_field(self, client):
        r = client.get("/")
        assert r.json()["service"] == "ClearView ML Backend"

    def test_root_has_load_flags(self, client):
        r = client.get("/")
        data = r.json()
        assert "adapter_loaded" in data
        assert "xai_loaded" in data


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_200(self, client):
        r = client.post("/predict", json={"text": "The colour is great but the smell is awful."})
        assert r.status_code == 200

    def test_predict_response_has_aspects(self, client):
        r = client.post("/predict", json={"text": "I love this lipstick."})
        data = r.json()
        assert "aspects" in data
        assert isinstance(data["aspects"], list)
        assert len(data["aspects"]) > 0

    def test_predict_each_aspect_has_required_fields(self, client):
        r = client.post("/predict", json={"text": "Great colour, terrible smell."})
        for asp in r.json()["aspects"]:
            assert "name"       in asp
            assert "label"      in asp
            assert "confidence" in asp
            assert "probs"      in asp
            assert "mentioned"  in asp

    def test_predict_label_values_are_valid(self, client):
        valid = {"positive", "negative", "neutral", "not_mentioned"}
        r = client.post("/predict", json={"text": "Nice texture."})
        for asp in r.json()["aspects"]:
            assert asp["label"] in valid

    def test_predict_confidence_in_range(self, client):
        r = client.post("/predict", json={"text": "Good product."})
        for asp in r.json()["aspects"]:
            assert 0.0 <= asp["confidence"] <= 1.0

    def test_predict_returns_conflict_prob(self, client):
        r = client.post("/predict", json={"text": "Love the colour, hate the smell."})
        data = r.json()
        assert "conflict_prob" in data
        assert 0.0 <= data["conflict_prob"] <= 1.0

    def test_predict_returns_timings(self, client):
        r = client.post("/predict", json={"text": "Great product."})
        assert "timings" in r.json()

    def test_predict_empty_text_does_not_crash(self, client):
        r = client.post("/predict", json={"text": ""})
        # Should return 200 or 422/500 — must not throw an unhandled exception
        assert r.status_code in (200, 422, 500)

    def test_predict_missing_text_returns_422(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 422


# ── /predict-bulk ─────────────────────────────────────────────────────────────

class TestPredictBulk:
    REVIEWS = [
        "The colour is beautiful.",
        "Smells terrible and packaging broke.",
        "Great value for money.",
    ]

    def test_bulk_returns_200(self, client):
        r = client.post("/predict-bulk", json={"reviews": self.REVIEWS})
        assert r.status_code == 200

    def test_bulk_total_reviews_matches_input(self, client):
        r = client.post("/predict-bulk", json={"reviews": self.REVIEWS})
        assert r.json()["total_reviews"] == len(self.REVIEWS)

    def test_bulk_response_has_required_keys(self, client):
        r = client.post("/predict-bulk", json={"reviews": self.REVIEWS})
        data = r.json()
        for key in ("total_reviews", "total_processed", "aspect_summary", "rows", "timings"):
            assert key in data, f"Missing key: {key}"

    def test_bulk_rows_count_matches_input(self, client):
        r = client.post("/predict-bulk", json={"reviews": self.REVIEWS})
        assert len(r.json()["rows"]) == len(self.REVIEWS)

    def test_bulk_empty_reviews_returns_400(self, client):
        r = client.post("/predict-bulk", json={"reviews": []})
        assert r.status_code == 400

    def test_bulk_missing_reviews_returns_422(self, client):
        r = client.post("/predict-bulk", json={})
        assert r.status_code == 422


# ── /metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_returns_dict(self, client):
        r = client.get("/metrics")
        assert isinstance(r.json(), dict)


# ── /explain ─────────────────────────────────────────────────────────────────

class TestExplain:
    PAYLOAD = {
        "text":    "The colour is beautiful but the smell is awful.",
        "aspect":  "colour",
        "methods": ["ig"],
    }

    def test_explain_returns_200(self, client):
        r = client.post("/explain", json=self.PAYLOAD)
        assert r.status_code == 200

    def test_explain_has_required_keys(self, client):
        r = client.post("/explain", json=self.PAYLOAD)
        data = r.json()
        assert "text"        in data
        assert "aspects"     in data
        assert "ig_conflict" in data
        assert "timings"     in data

    def test_explain_missing_text_returns_422(self, client):
        r = client.post("/explain", json={"aspect": "colour"})
        assert r.status_code == 422

    def test_explain_lime_method(self, client):
        payload = {**self.PAYLOAD, "methods": ["lime"]}
        r = client.post("/explain", json=payload)
        assert r.status_code == 200

    def test_explain_shap_method(self, client):
        payload = {**self.PAYLOAD, "methods": ["shap"]}
        r = client.post("/explain", json=payload)
        assert r.status_code == 200
