import csv
import statistics
from pathlib import Path
from rouge_score import rouge_scorer
from src.optimizations.quantization import QuantizedEngine, GGUFEngine
import structlog

log = structlog.get_logger()

GGUF_Q4_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Fixed prompts with reference answers for ROUGE scoring
EVAL_PROMPTS = [
    {
        "prompt": "What is 2+2?",
        "reference": "2 + 2 equals 4",
    },
    {
        "prompt": "What is the capital of France?",
        "reference": "The capital of France is Paris",
    },
    {
        "prompt": "What is Python programming language?",
        "reference": "Python is a high-level interpreted programming language known for its simple readable syntax",
    },
    {
        "prompt": "What is machine learning?",
        "reference": "Machine learning is a subset of artificial intelligence where systems learn from data to make predictions",
    },
    {
        "prompt": "What is a neural network?",
        "reference": "A neural network is a computational model inspired by the human brain consisting of layers of interconnected nodes",
    },
]

def evaluate_model(engine, label: str) -> list[dict]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []
    for item in EVAL_PROMPTS:
        log.info("Evaluating", mode=label, prompt=item["prompt"][:40])
        result = engine.generate(item["prompt"], max_new_tokens=100)
        scores = scorer.score(item["reference"], result["text"])
        rouge_l = round(scores["rougeL"].fmeasure, 4)
        results.append({
            "mode":       label,
            "prompt":     item["prompt"],
            "reference":  item["reference"],
            "generated":  result["text"],
            "rouge_l":    rouge_l,
            "ttft":       result["ttft_seconds"],
            "tps":        result["tps"],
        })
        log.info("Done", rouge_l=rouge_l, ttft=result["ttft_seconds"])
    return results

def save_results(results: list[dict]):
    path = Path("data") / "quality_comparison.csv"
    keys = ["mode", "prompt", "reference", "generated", "rouge_l", "ttft", "tps"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    log.info("Saved", path=str(path))

def print_summary(all_results: list[dict]):
    print("\n══ QUALITY vs SPEED TRADEOFF ══")
    for mode in ["float32", "Q4_K_M"]:
        group = [r for r in all_results if r["mode"] == mode]
        if not group:
            continue
        rouge_scores = [r["rouge_l"] for r in group]
        ttfts = [r["ttft"] for r in group]
        print(f"\n  {mode}")
        print(f"    ROUGE-L mean : {statistics.mean(rouge_scores):.4f}")
        print(f"    TTFT mean    : {statistics.mean(ttfts):.2f}s")
        print(f"    TPS mean     : {statistics.mean([r['tps'] for r in group]):.2f}")

    f32 = [r for r in all_results if r["mode"] == "float32"]
    q4  = [r for r in all_results if r["mode"] == "Q4_K_M"]
    if f32 and q4:
        speedup     = statistics.mean([r["ttft"] for r in f32]) / statistics.mean([r["ttft"] for r in q4])
        quality_drop = (statistics.mean([r["rouge_l"] for r in f32]) - statistics.mean([r["rouge_l"] for r in q4])) * 100
        print(f"\n  SPEEDUP      : {speedup:.2f}x faster")
        print(f"  QUALITY DROP : {quality_drop:.1f}% ROUGE-L reduction")

if __name__ == "__main__":
    all_results = []

    f32_engine = QuantizedEngine(mode="float32")
    f32_engine.load_model()
    all_results += evaluate_model(f32_engine, "float32")
    del f32_engine

    q4_engine = GGUFEngine(model_path=GGUF_Q4_PATH)
    q4_engine.load_model()
    all_results += evaluate_model(q4_engine, "Q4_K_M")
    del q4_engine

    save_results(all_results)
    print_summary(all_results)