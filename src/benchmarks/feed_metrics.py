import csv
import time
from pathlib import Path
from prometheus_client import Gauge, Counter, Histogram, start_http_server
import structlog

log = structlog.get_logger()

# ── Metrics ───────────────────────────────────────────────
TTFT_GAUGE = Gauge(
    "llm_ttft_seconds_gauge",
    "TTFT per optimization mode",
    labelnames=["mode", "category"],
)
TPS_GAUGE = Gauge(
    "llm_tps_gauge",
    "Tokens per second per mode",
    labelnames=["mode"],
)
SPEEDUP_GAUGE = Gauge(
    "llm_speedup",
    "Speedup vs float32 baseline",
    labelnames=["mode"],
)
CACHE_HIT_RATE = Gauge(
    "llm_cache_hit_rate",
    "KV cache hit rate per policy",
    labelnames=["policy"],
)
ACCEPTANCE_RATE = Gauge(
    "llm_speculative_acceptance_rate",
    "Speculative decoding acceptance rate",
)
TOKENS_COUNTER = Counter(
    "llm_total_tokens",
    "Total tokens generated across all benchmarks",
    labelnames=["mode"],
)


def load_csv(filename: str) -> list[dict]:
    path = Path("data") / filename
    if not path.exists():
        log.warning("File not found", path=str(path))
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def feed_quantization():
    rows = load_csv("quantization_comparison.csv")
    baselines = {}
    for row in rows:
        mode     = row["mode"]
        category = row.get("category", "general")
        ttft     = float(row["ttft_seconds"])
        tps      = float(row["tps"])
        tokens   = int(row["tokens_generated"])
        TTFT_GAUGE.labels(mode=mode, category=category).set(ttft)
        TPS_GAUGE.labels(mode=mode).set(tps)
        TOKENS_COUNTER.labels(mode=mode).inc(tokens)
        if mode not in baselines:
            baselines[mode] = ttft
    # speedup relative to float32
    f32_ttft = next((v for k,v in baselines.items() if k=="float32"), None)
    if f32_ttft:
        for mode, ttft in baselines.items():
            SPEEDUP_GAUGE.labels(mode=mode).set(round(f32_ttft / ttft, 2))
    log.info("Fed quantization metrics", rows=len(rows))


def feed_kv_cache():
    rows = load_csv("kv_cache_comparison.csv")
    policy_hits   = {}
    policy_total  = {}
    for row in rows:
        policy   = row["policy"]
        hit      = row.get("cache_hit", "False") == "True"
        policy_total[policy] = policy_total.get(policy, 0) + 1
        if hit:
            policy_hits[policy] = policy_hits.get(policy, 0) + 1
    for policy, total in policy_total.items():
        hits = policy_hits.get(policy, 0)
        rate = hits / total if total > 0 else 0
        CACHE_HIT_RATE.labels(policy=policy).set(round(rate, 4))
    log.info("Fed KV cache metrics", policies=list(policy_total.keys()))


def feed_speculative():
    rows = load_csv("speculative_comparison.csv")
    spec_rows = [r for r in rows if r["mode"] == "speculative"]
    if spec_rows:
        rates = [float(r["acceptance_rate"]) for r in spec_rows if r["acceptance_rate"]]
        if rates:
            avg_rate = sum(rates) / len(rates)
            ACCEPTANCE_RATE.set(avg_rate)

        base_rows = [r for r in rows if r["mode"] == "baseline"]
        if base_rows and spec_rows:
            base_ttft = sum(float(r["ttft_seconds"]) for r in base_rows) / len(base_rows)
            spec_ttft = sum(float(r["ttft_seconds"]) for r in spec_rows) / len(spec_rows)
            if spec_ttft > 0:
                SPEEDUP_GAUGE.labels(mode="speculative").set(round(base_ttft / spec_ttft, 2))
    log.info("Fed speculative metrics", rows=len(spec_rows))


if __name__ == "__main__":
    start_http_server(8002)
    log.info("Metrics server started", port=8002)

    feed_quantization()
    feed_kv_cache()
    feed_speculative()

    log.info("All metrics fed — keeping server alive for Prometheus scraping")
    log.info("Open Grafana at http://localhost:3000")

    while True:
        feed_quantization()
        feed_kv_cache()
        feed_speculative()
        time.sleep(15)