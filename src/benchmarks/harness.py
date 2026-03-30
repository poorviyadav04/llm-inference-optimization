import csv
import time
import statistics
from pathlib import Path
from src.server.inference import engine
import structlog
from src.benchmarks.metrics import record, start_metrics_server

log = structlog.get_logger()

SHORT_PROMPTS = [
    "What is 2+2?",
    "Name the capital of France.",
    "What colour is the sky?",
]

MEDIUM_PROMPTS = [
    "Explain what a neural network is in simple terms.",
    "What are the main differences between Python and JavaScript?",
    "Describe how the internet works in a few sentences.",
]

LONG_PROMPTS = [
    "Write a detailed explanation of how transformers work in machine learning, covering attention mechanisms, positional encoding, and why they replaced RNNs.",
    "Explain the entire process of how a web request travels from a browser to a server and back, including DNS, TCP, HTTP, and rendering.",
]

def run_benchmark(prompts: list[str], label: str) -> list[dict]:
    results = []
    for i, prompt in enumerate(prompts):
        log.info("Running prompt", label=label, index=i+1, total=len(prompts))
        result = engine.generate(prompt)
        record(result, category=label)
        result["prompt"] = prompt[:50]
        result["label"] = label
        results.append(result)
        log.info("Done", ttft=result["ttft_seconds"], tps=result["tps"])
    return results

def save_results(results: list[dict], filename: str = "baseline_results.csv"):
    path = Path("data") / filename
    keys = ["label", "prompt", "tokens_generated", "ttft_seconds", "tps", "text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in keys})
    log.info("Results saved", path=str(path))

def print_summary(results: list[dict]):
    for label in ["short", "medium", "long"]:
        group = [r for r in results if r["label"] == label]
        if not group:
            continue
        ttfts = [r["ttft_seconds"] for r in group]
        tps_list = [r["tps"] for r in group]
        print(f"\n── {label.upper()} PROMPTS ──")
        print(f"  TTFT  mean={statistics.mean(ttfts):.2f}s  max={max(ttfts):.2f}s")
        print(f"  TPS   mean={statistics.mean(tps_list):.2f}  max={max(tps_list):.2f}")

if __name__ == "__main__":
    engine.load_model()
    all_results = []
    all_results += run_benchmark(SHORT_PROMPTS, "short")
    all_results += run_benchmark(MEDIUM_PROMPTS, "medium")
    all_results += run_benchmark(LONG_PROMPTS, "long")
    save_results(all_results)
    print_summary(all_results)