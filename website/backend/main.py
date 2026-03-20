"""
backend/main.py
FastAPI application factory — thin wiring layer.
Creates the app, registers CORS, mounts all routers, and defines the lifespan hook.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ALLOWED_ORIGINS
from .model_cache import preload_models
from .routes import predict, explain, misc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load ML models into memory before accepting requests."""
    preload_models()
    yield  # Server runs here


app = FastAPI(
    title="ClearView ML Backend",
    description="Aspect-based sentiment analysis with XAI explanations",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
print(f"[CORS] Allowed origins: {ALLOWED_ORIGINS}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(misc.router)
app.include_router(predict.router)
app.include_router(explain.router)
