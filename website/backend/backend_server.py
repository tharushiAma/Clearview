#!/usr/bin/env python3
"""
website/backend/backend_server.py
Entry point for the ClearView FastAPI backend.

All application logic is in this package:
  config.py         — env vars & path constants
  model_cache.py    — model loading & caching
  routes/predict.py — /predict, /predict-bulk
  routes/explain.py — /explain
  routes/misc.py    — /, /metrics, /logs
  main.py           — FastAPI app, CORS, lifespan

Usage (from the website/ directory):
  python backend/backend_server.py
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

# This file lives at website/backend/backend_server.py
# To import `from backend.main import app`, we need website/ on sys.path
_this_file = os.path.abspath(__file__)
_backend_dir = os.path.dirname(_this_file)
_website_dir = os.path.dirname(_backend_dir)

if _website_dir not in sys.path:
    sys.path.insert(0, _website_dir)

try:
    from backend.main import app
except ImportError:
    # Fallback if already in website/ directory context
    from main import app  # type: ignore

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("ClearView Backend  →  http://localhost:8000")
    print("API docs           →  http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
