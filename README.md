# LLM Inference Optimization System

> **Reducing LLM serving latency by 5.58x through quantization, KV caching, continuous batching, and speculative decoding — fully measurable, production-grade, runs on CPU.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-orange)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-yellow)

------------------------------------------------------------------------

## TL;DR

This project implements **four major LLM inference optimizations** used
in modern serving engines.

  Technique              Impact
  ---------------------- --------------------------------------------
  Quantization           **5.58× faster TTFT**
  KV cache               **72% TTFT reduction** on repeated prompts
  Continuous batching    **44% lower wait time**
  Speculative decoding   **5.8× faster decoding (best-case)**

All improvements are **experimentally measured and reproducible**.

------------------------------------------------------------------------

## Motivation

Most developers interact with LLMs via APIs, but **very few understand
how inference systems are optimized internally**.

Serving LLMs efficiently requires solving several systems challenges:

-   slow token generation
-   wasted compute during batching
-   repeated prompt recomputation
-   poor observability of latency metrics

This project builds a **mini LLM serving system from scratch** to study
and benchmark these optimizations.

------------------------------------------------------------------------

## Architecture

    Incoming Request
           │
           ▼
    ┌─────────────────────┐
    │ FastAPI Server      │
    │ POST /generate      │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ LRU KV Cache        │
    │ Cache hit → instant │
    │ Cache miss → model  │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Batch Scheduler     │
    │ Continuous batching │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Inference Engine    │
    │ llama.cpp GGUF      │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Prometheus Metrics  │
    │ TTFT, TPS, cache    │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Grafana Dashboard   │
    │ Live monitoring     │
    └─────────────────────┘

------------------------------------------------------------------------

## Benchmark Results

  -----------------------------------------------------------------------
  Optimization      Baseline          Optimized         Improvement
  ----------------- ----------------- ----------------- -----------------
  Quantization TTFT 31.58s            5.66s             **5.58× faster**

  Token throughput  1.83 TPS          13.28 TPS         **7.3× faster**

  Quantization      0.520 ROUGE-L     0.384             13.6% drop
  quality                                               

  KV cache TTFT     9.06s             2.52s             **72% reduction**

  Continuous        0.093 req/s       0.139 req/s       **1.49×**
  batching                                              
  throughput                                            

  Continuous        45.5s             28.4s             **44% reduction**
  batching wait                                         
  time                                                  

  Speculative       7.33s             1.26s             **5.81× faster**
  decoding TTFT                                         
  -----------------------------------------------------------------------

Benchmarks were run on **CPU only** using TinyLlama 1.1B.

------------------------------------------------------------------------

## Optimization Techniques

### Quantization

Compared float32 and GGUF **Q4_K_M quantization**.

  Format    Memory    TTFT     TPS
  --------- --------- -------- -------
  float32   \~4GB     31.58s   1.83
  Q4_K_M    \~0.7GB   5.66s    13.28

**Result:** 5.58× speedup with only 13.6% ROUGE-L quality drop.

------------------------------------------------------------------------

### KV Cache

Implemented a key-value attention cache with three eviction policies.

  Policy           TTFT    Reduction
  ---------------- ------- -----------
  No cache         9.06s   baseline
  FIFO             5.03s   44.5%
  LRU              4.41s   51.3%
  Sliding window   2.52s   **72.2%**

------------------------------------------------------------------------

### Continuous Batching

Three request scheduling strategies were benchmarked.

  Scheduler             Throughput        Wait Mean
  --------------------- ----------------- -----------
  No batching           0.093 req/s       45.5s
  Static batching       0.120 req/s       50.9s
  Continuous batching   **0.139 req/s**   **28.4s**

Continuous batching fills freed slots immediately when requests finish.

------------------------------------------------------------------------

### Speculative Decoding

Implemented speculative decoding based on Chen et al. (2023).

  Mode                TTFT    Acceptance
  ------------------- ------- ------------
  Baseline            7.33s   ---
  Speculative (K=5)   1.26s   100%

------------------------------------------------------------------------

## Observability

All performance metrics are exported to **Prometheus** and visualized in
**Grafana**.

Metrics tracked:

-   TTFT
-   token throughput
-   cache hit rate
-   speculative acceptance
-   speedup per optimization

------------------------------------------------------------------------

## Project Structure

    src/
     ├ server/
     │   ├ inference.py
     │   └ app.py
     ├ optimizations/
     │   ├ quantization.py
     │   ├ kv_cache.py
     │   ├ batching.py
     │   └ speculative.py
     └ benchmarks/
         ├ benchmark_quantization.py
         ├ benchmark_kv_cache.py
         ├ benchmark_batching.py
         ├ benchmark_speculative.py
         └ metrics.py

------------------------------------------------------------------------

## Quick Start

### Clone

    git clone https://github.com/YOUR_USERNAME/llm-inference-optimization
    cd llm-inference-optimization

### Install

    uv venv
    .venv\Scripts\activate
    uv sync

### Run benchmarks

    python -m src.benchmarks.benchmark_quantization
    python -m src.benchmarks.benchmark_kv_cache
    python -m src.benchmarks.benchmark_batching
    python -m src.benchmarks.benchmark_speculative

------------------------------------------------------------------------

## Tech Stack

  Category           Tools
  ------------------ ------------------
  Inference          llama-cpp-python
  API                FastAPI
  Metrics            Prometheus
  Visualization      Grafana
  Containerization   Docker
  Evaluation         ROUGE-L

------------------------------------------------------------------------

## License

MIT License
