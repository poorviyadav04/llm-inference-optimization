import csv
import statistics
from pathlib import Path
from src.optimizations.kv_cache import KVCache
from src.optimizations.quantization import GGUFEngine
import structlog

log = structlog.get_logger()

GGUF_Q4_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Mix of repeated and unique prompts to test cache behaviour
PROMPTS = [
    ("What is Python?",                    "medium"),
    ("What is 2+2?",                       "short"),
    ("What is Python?",                    "medium"),  # repeat — should be cache hit
    ("Explain neural networks briefly.",   "medium"),
    ("What is 2+2?",                       "short"),   # repeat — should be cache hit
    ("What is machine learning?",          "medium"),
    ("What is Python?",                    "medium"),  # repeat again
    ("What is the capital of France?",     "short"),
    ("Explain neural networks briefly.",   "medium"),  # repeat
    ("What is machine learning?",          "medium"),  # repeat
]

def run_with_cache(policy: str, engine: GGUFEngine) -> dict:
    cache = KVCache(max_size=10, policy=policy)
    results = []

    for prompt, category in PROMPTS:
        cached = cache.get(prompt)
        if cached:
            result = cached.copy()
            result["cache_hit"] = True
            result["ttft_seconds"] = 0.0   # instant
        else:
            result = engine.generate(prompt)
            result["cache_hit"] = False
            cache.put(prompt, result)

        result["category"] = category
        result["policy"] = policy
        results.append(result)
        log.info("Done",
                 policy=policy,
                 cache_hit=result["cache_hit"],
                 ttft=result["ttft_seconds"])

    stats = cache.stats()
    return {
        "results":   results,
        "hit_rate":  stats["hit_rate"],
        "hits":      stats["hits"],
        "misses":    stats["misses"],
    }

def run_without_cache(engine: GGUFEngine) -> list[dict]:
    results = []
    for prompt, category in PROMPTS:
        result = engine.generate(prompt)
        result["cache_hit"] = False
        result["category"] = category
        result["policy"] = "no_cache"
        results.append(result)
        log.info("Done (no cache)", ttft=result["ttft_seconds"])
    return results

def save_results(all_results: list[dict], filename: str = "kv_cache_comparison.csv"):
    path = Path("data") / filename
    keys = ["policy", "category", "cache_hit", "ttft_seconds", "tps", "tokens_generated"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in keys})
    log.info("Saved", path=str(path))

def print_summary(no_cache_results, cache_data: dict):
    no_cache_ttfts = [r["ttft_seconds"] for r in no_cache_results]
    print("\n══ KV CACHE COMPARISON ══")
    print(f"\n  NO CACHE")
    print(f"    TTFT mean : {statistics.mean(no_cache_ttfts):.2f}s")
    print(f"    Total time: {sum(no_cache_ttfts):.2f}s")

    for policy, data in cache_data.items():
        ttfts = [r["ttft_seconds"] for r in data["results"]]
        print(f"\n  {policy.upper()}")
        print(f"    TTFT mean : {statistics.mean(ttfts):.2f}s")
        print(f"    Total time: {sum(ttfts):.2f}s")
        print(f"    Hit rate  : {data['hit_rate']*100:.1f}%")
        print(f"    Hits/Miss : {data['hits']}/{data['misses']}")
        reduction = (1 - statistics.mean(ttfts) / statistics.mean(no_cache_ttfts)) * 100
        print(f"    TTFT reduction: {reduction:.1f}%")

if __name__ == "__main__":
    engine = GGUFEngine(model_path=GGUF_Q4_PATH)
    engine.load_model()

    log.info("Running without cache...")
    no_cache = run_without_cache(engine)

    cache_data = {}
    for policy in ["fifo", "lru", "sliding_window"]:
        log.info("Running with cache", policy=policy)
        cache_data[policy] = run_with_cache(policy, engine)

    all_results = no_cache.copy()
    for data in cache_data.values():
        all_results += data["results"]

    save_results(all_results)
    print_summary(no_cache, cache_data)