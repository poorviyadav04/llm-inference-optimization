import csv
import statistics
from pathlib import Path
from src.optimizations.speculative import SpeculativeDecoder
from src.optimizations.quantization import GGUFEngine
import structlog

log = structlog.get_logger()

GGUF_Q4_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

TEST_PROMPTS = [
    "What is 2+2?",
    "What is the capital of France?",
    "What is Python programming language?",
    "What is machine learning?",
    "What is a neural network?",
]

def run_baseline(prompts: list[str]) -> list[dict]:
    engine = GGUFEngine(model_path=GGUF_Q4_PATH)
    engine.load_model()
    results = []
    for prompt in prompts:
        log.info("Baseline generating", prompt=prompt[:40])
        result = engine.generate(prompt, max_new_tokens=100)
        result["mode"] = "baseline"
        results.append(result)
        log.info("Done", ttft=result["ttft_seconds"], tps=result["tps"])
    return results

def run_speculative(prompts: list[str]) -> list[dict]:
    decoder = SpeculativeDecoder(
        draft_model_path=GGUF_Q4_PATH,
        target_model_path=GGUF_Q4_PATH,
        num_speculative_tokens=5,
    )
    decoder.load_model()
    results = []
    for prompt in prompts:
        log.info("Speculative generating", prompt=prompt[:40])
        result = decoder.generate(prompt, max_new_tokens=100)
        results.append(result)
        log.info("Done",
                 ttft=result["ttft_seconds"],
                 tps=result["tps"],
                 acceptance_rate=result["acceptance_rate"])
    return results

def save_results(all_results: list[dict]):
    path = Path("data") / "speculative_comparison.csv"
    keys = ["mode", "tokens_generated", "ttft_seconds", "tps", "acceptance_rate", "text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in keys})
    log.info("Saved", path=str(path))

def print_summary(baseline: list[dict], speculative: list[dict]):
    base_ttfts = [r["ttft_seconds"] for r in baseline]
    spec_ttfts = [r["ttft_seconds"] for r in speculative]
    base_tps   = [r["tps"] for r in baseline]
    spec_tps   = [r["tps"] for r in speculative]
    acc_rates  = [r["acceptance_rate"] for r in speculative]

    speedup = statistics.mean(base_ttfts) / statistics.mean(spec_ttfts) \
              if statistics.mean(spec_ttfts) > 0 else 0

    print("\n══ SPECULATIVE DECODING COMPARISON ══")
    print(f"\n  BASELINE (normal generation)")
    print(f"    TTFT mean : {statistics.mean(base_ttfts):.2f}s")
    print(f"    TPS  mean : {statistics.mean(base_tps):.2f}")

    print(f"\n  SPECULATIVE DECODING (K=5)")
    print(f"    TTFT mean        : {statistics.mean(spec_ttfts):.2f}s")
    print(f"    TPS  mean        : {statistics.mean(spec_tps):.2f}")
    print(f"    Acceptance rate  : {statistics.mean(acc_rates):.2%}")
    print(f"\n  SPEEDUP : {speedup:.2f}x faster")

if __name__ == "__main__":
    baseline    = run_baseline(TEST_PROMPTS)
    speculative = run_speculative(TEST_PROMPTS)
    save_results(baseline + speculative)
    print_summary(baseline, speculative)