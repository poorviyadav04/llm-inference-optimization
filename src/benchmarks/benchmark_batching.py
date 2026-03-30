import time
import csv
import statistics
from pathlib import Path
from src.optimizations.batching import (
    Request, NoBatchScheduler,
    StaticBatchScheduler, ContinuousBatchScheduler
)
from src.optimizations.quantization import GGUFEngine
import structlog

log = structlog.get_logger()

GGUF_Q4_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

PROMPTS = [
    ("What is 2+2?",                         1),
    ("What is the capital of France?",        1),
    ("Explain what Python is.",               2),
    ("What is machine learning?",             2),
    ("What is a neural network?",             2),
    ("What is the speed of light?",           1),
    ("Explain what an API is.",               2),
    ("What is artificial intelligence?",      2),
]

def make_requests() -> list[Request]:
    return [
        Request(
            request_id=f"req_{i}",
            prompt=prompt,
            priority=priority,
            max_new_tokens=100,
        )
        for i, (prompt, priority) in enumerate(PROMPTS)
    ]

def run_scheduler(scheduler, label: str) -> dict:
    requests = make_requests()
    start = time.time()
    completed = scheduler.process(requests)
    total_time = time.time() - start

    wait_times = [r.wait_time for r in completed]
    ttfts = [r.result["ttft_seconds"] for r in completed]
    tps_list = [r.result["tps"] for r in completed]
    throughput = len(completed) / total_time

    return {
        "label":       label,
        "total_time":  round(total_time, 2),
        "throughput":  round(throughput, 3),
        "wait_mean":   round(statistics.mean(wait_times), 2),
        "wait_max":    round(max(wait_times), 2),
        "ttft_mean":   round(statistics.mean(ttfts), 2),
        "tps_mean":    round(statistics.mean(tps_list), 2),
        "requests":    len(completed),
    }

def save_results(all_stats: list[dict]):
    path = Path("data") / "batching_comparison.csv"
    keys = ["label", "total_time", "throughput", "wait_mean",
            "wait_max", "ttft_mean", "tps_mean", "requests"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_stats)
    log.info("Saved", path=str(path))

def print_summary(all_stats: list[dict]):
    print("\n══ BATCHING COMPARISON ══")
    baseline = all_stats[0]["throughput"]
    for s in all_stats:
        speedup = s["throughput"] / baseline
        print(f"\n  {s['label'].upper()}")
        print(f"    Total time  : {s['total_time']}s")
        print(f"    Throughput  : {s['throughput']} req/s")
        print(f"    Wait mean   : {s['wait_mean']}s")
        print(f"    Wait max    : {s['wait_max']}s")
        print(f"    TTFT mean   : {s['ttft_mean']}s")
        print(f"    Speedup     : {speedup:.2f}x vs no_batch")

if __name__ == "__main__":
    engine = GGUFEngine(model_path=GGUF_Q4_PATH)
    engine.load_model()

    all_stats = []

    log.info("Running no_batch...")
    all_stats.append(run_scheduler(
        NoBatchScheduler(engine), "no_batch"
    ))

    log.info("Running static_batch...")
    all_stats.append(run_scheduler(
        StaticBatchScheduler(engine, batch_size=4), "static_batch"
    ))

    log.info("Running continuous_batch...")
    all_stats.append(run_scheduler(
        ContinuousBatchScheduler(engine, max_concurrent=4), "continuous_batch"
    ))

    save_results(all_stats)
    print_summary(all_stats)