import csv
import statistics
from pathlib import Path
from src.optimizations.quantization import QuantizedEngine, GGUFEngine
import structlog

log = structlog.get_logger()

TEST_PROMPTS = {
    "short":  ["What is 2+2?", "Name the capital of France."],
    "medium": ["Explain what a neural network is in simple terms."],
    "long":   ["Explain how transformers work in machine learning, covering attention mechanisms and why they replaced RNNs."],
}

GGUF_Q4_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

def run_hf_mode(mode: str) -> list[dict]:
    engine = QuantizedEngine(mode=mode)
    engine.load_model()
    results = []
    for category, prompts in TEST_PROMPTS.items():
        for prompt in prompts:
            log.info("Running", mode=mode, category=category)
            result = engine.generate(prompt)
            result["category"] = category
            results.append(result)
            log.info("Done", ttft=result["ttft_seconds"], tps=result["tps"])
    del engine
    return results

def run_gguf_mode(model_path: str, label: str) -> list[dict]:
    engine = GGUFEngine(model_path=model_path)
    engine.load_model()
    results = []
    for category, prompts in TEST_PROMPTS.items():
        for prompt in prompts:
            log.info("Running", mode=label, category=category)
            result = engine.generate(prompt)
            result["category"] = category
            result["mode"] = label
            results.append(result)
            log.info("Done", ttft=result["ttft_seconds"], tps=result["tps"])
    del engine
    return results

def save_results(results: list[dict]):
    path = Path("data") / "quantization_comparison.csv"
    keys = ["mode", "category", "tokens_generated", "ttft_seconds", "tps", "text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in keys})
    log.info("Saved", path=str(path))

def print_comparison(all_results: list[dict]):
    print("\n══ QUANTIZATION COMPARISON ══")
    for mode in ["float32", "Q4_K_M"]:
        results = [r for r in all_results if r["mode"] == mode]
        if not results:
            continue
        ttfts = [r["ttft_seconds"] for r in results]
        tps_list = [r["tps"] for r in results]
        print(f"\n  {mode.upper()}")
        print(f"    TTFT mean : {statistics.mean(ttfts):.2f}s")
        print(f"    TTFT max  : {max(ttfts):.2f}s")
        print(f"    TPS  mean : {statistics.mean(tps_list):.2f}")

    # speedup calculation
    f32 = [r for r in all_results if r["mode"] == "float32"]
    q4  = [r for r in all_results if r["mode"] == "Q4_K_M"]
    if f32 and q4:
        speedup = statistics.mean([r["ttft_seconds"] for r in f32]) / \
                  statistics.mean([r["ttft_seconds"] for r in q4])
        print(f"\n  SPEEDUP (float32 → Q4_K_M): {speedup:.2f}x faster")

if __name__ == "__main__":
    all_results = []
    all_results += run_hf_mode("float32")
    all_results += run_gguf_mode(GGUF_Q4_PATH, "Q4_K_M")
    save_results(all_results)
    print_comparison(all_results)