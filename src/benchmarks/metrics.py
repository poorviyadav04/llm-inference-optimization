from prometheus_client import Histogram, Gauge, Counter, start_http_server
import structlog

log = structlog.get_logger()

# ── Metrics definitions ──────────────────────────────────
TTFT = Histogram(
    "llm_ttft_seconds",
    "Time to first token in seconds",
    buckets=[1, 2, 5, 10, 20, 50, 100, 200],
    labelnames=["prompt_category"],
)

TPS = Gauge(
    "llm_tokens_per_second",
    "Tokens generated per second",
    labelnames=["prompt_category"],
)

TOKENS_TOTAL = Counter(
    "llm_tokens_generated_total",
    "Total tokens generated since startup",
)

REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total inference requests processed",
    labelnames=["prompt_category"],
)

MEMORY_BYTES = Gauge(
    "llm_memory_bytes",
    "Current process memory usage in bytes",
)


def record(result: dict, category: str = "unknown"):
    """Call this after every engine.generate() to push metrics."""
    TTFT.labels(prompt_category=category).observe(result["ttft_seconds"])
    TPS.labels(prompt_category=category).set(result["tps"])
    TOKENS_TOTAL.inc(result["tokens_generated"])
    REQUESTS_TOTAL.labels(prompt_category=category).inc()


def start_metrics_server(port: int = 8001):
    start_http_server(port)
    log.info("Prometheus metrics server started", port=port)