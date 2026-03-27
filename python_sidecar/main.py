"""
LettuceCache embedding sidecar.

Exposes:
  POST /embed           { "text": str }               -> { "embedding": list[float] }
  POST /embed_batch     { "texts": list[str] }         -> { "embeddings": list[list[float]] }
  GET  /health                                         -> { "status": "ok" }

Environment variables:
  MODEL_NAME  (default: all-MiniLM-L6-v2)
  WORKERS     (uvicorn worker count, default: 1 — handled by Dockerfile CMD)
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("embedding_sidecar")

MODEL_NAME: str = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")

# Module-level model reference populated at startup
_model: Optional[SentenceTransformer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    log.info("Loading sentence-transformer model: %s", MODEL_NAME)
    _model = SentenceTransformer(MODEL_NAME)
    # Warm-up pass so first real request is not penalised
    _ = _model.encode(["warmup"], convert_to_numpy=True)
    log.info("Model loaded. Embedding dimension: %d", _model.get_sentence_embedding_dimension())
    yield
    log.info("Shutting down embedding sidecar")


app = FastAPI(title="LettuceCache Embedding Sidecar", version="1.0.0", lifespan=lifespan)


# ── Request/Response schemas ──────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: List[float]
    model: str
    dimension: int


class BatchEmbedRequest(BaseModel):
    texts: List[str]


class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int
    count: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dimension": _model.get_sentence_embedding_dimension(),
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    vec: np.ndarray = _model.encode(req.text, convert_to_numpy=True, normalize_embeddings=True)
    return EmbedResponse(
        embedding=vec.tolist(),
        model=MODEL_NAME,
        dimension=int(vec.shape[0]),
    )


@app.post("/embed_batch", response_model=BatchEmbedResponse)
def embed_batch(req: BatchEmbedRequest) -> BatchEmbedResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list must not be empty")
    if len(req.texts) > 256:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 256")

    vecs: np.ndarray = _model.encode(
        req.texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32
    )
    return BatchEmbedResponse(
        embeddings=vecs.tolist(),
        model=MODEL_NAME,
        dimension=int(vecs.shape[1]),
        count=len(req.texts),
    )
