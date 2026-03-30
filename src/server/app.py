import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.optimizations.quantization import GGUFEngine
from src.optimizations.kv_cache import KVCache
from src.benchmarks.metrics import record, start_metrics_server
from src.config import settings
import structlog

log = structlog.get_logger()

GGUF_Q4_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# ── Global state ──────────────────────────────────────────
engine = GGUFEngine(model_path=GGUF_Q4_PATH)
cache  = KVCache(max_size=50, policy="lru")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — loading model...")
    engine.load_model()
    start_metrics_server(8001)
    log.info("Ready")
    yield
    log.info("Shutting down")


app = FastAPI(
    title="LLM Inference Optimization API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=200, ge=1, le=500)
    category: str = Field(default="general")
    use_cache: bool = Field(default=True)


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    tokens_generated: int
    ttft_seconds: float
    tps: float
    cache_hit: bool
    model: str


# ── Endpoints ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": engine.is_loaded,
        "cache_stats": cache.stats(),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    request_id = str(uuid.uuid4())[:8]
    log.info("Request received", request_id=request_id, prompt=req.prompt[:40])

    cache_hit = False

    if req.use_cache:
        cached = cache.get(req.prompt)
        if cached:
            cache_hit = True
            result = cached
            log.info("Cache hit", request_id=request_id)
        else:
            result = engine.generate(req.prompt, req.max_new_tokens)
            cache.put(req.prompt, result)
    else:
        result = engine.generate(req.prompt, req.max_new_tokens)

    record(result, category=req.category)

    return GenerateResponse(
        request_id=request_id,
        text=result["text"],
        tokens_generated=result["tokens_generated"],
        ttft_seconds=result["ttft_seconds"],
        tps=result["tps"],
        cache_hit=cache_hit,
        model=GGUF_Q4_PATH,
    )


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.get("/metrics/summary")
def metrics_summary():
    return {
        "cache": cache.stats(),
        "model": GGUF_Q4_PATH,
        "quantization": "Q4_K_M",
    }