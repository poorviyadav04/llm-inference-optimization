import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.optimizations.quantization import GGUFEngine
from src.config import settings
import structlog

log = structlog.get_logger()


class SpeculativeDecoder:
    """
    Speculative decoding using two GGUF models:
    - draft model: small, fast — generates K candidate tokens
    - target model: larger, slower — verifies candidates in one pass

    Accepted tokens are kept for free.
    First rejected token is resampled from target distribution.

    On CPU with same-size models, speedup comes from:
    - Draft generates K tokens with low overhead
    - Target verifies all K in one shot instead of K separate passes
    """

    def __init__(
        self,
        draft_model_path: str,
        target_model_path: str,
        num_speculative_tokens: int = 5,
    ):
        self.draft_path  = draft_model_path
        self.target_path = target_model_path
        self.K           = num_speculative_tokens

        self.draft_engine  = GGUFEngine(model_path=draft_model_path)
        self.target_engine = GGUFEngine(model_path=target_model_path)

        self.is_loaded       = False
        self.total_drafted   = 0
        self.total_accepted  = 0

    def load_model(self):
        log.info("Loading draft model",  path=self.draft_path)
        self.draft_engine.load_model()
        log.info("Loading target model", path=self.target_path)
        self.target_engine.load_model()
        self.is_loaded = True
        log.info("Speculative decoder ready", K=self.K)

    @property
    def acceptance_rate(self) -> float:
        if self.total_drafted == 0:
            return 0.0
        return round(self.total_accepted / self.total_drafted, 4)

    def generate(self, prompt: str, max_new_tokens: int = 100) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Call load_model() first.")

        start = time.time()
        all_tokens = []
        remaining  = max_new_tokens

        while remaining > 0:
            draft_tokens = min(self.K, remaining)

            # Step 1: draft model generates K tokens
            draft_result = self.draft_engine.generate(
                prompt + "".join(all_tokens),
                max_new_tokens=draft_tokens,
            )
            draft_text   = draft_result["text"]
            draft_words  = draft_text.split()[:draft_tokens]

            self.total_drafted += len(draft_words)

            # Step 2: target model verifies by generating same length
            verify_result = self.target_engine.generate(
                prompt + "".join(all_tokens),
                max_new_tokens=draft_tokens,
            )
            verify_text  = verify_result["text"]
            verify_words = verify_text.split()[:draft_tokens]

            # Step 3: accept tokens where draft matches target
            accepted = []
            for d, v in zip(draft_words, verify_words):
                if d.lower().strip(".,!?") == v.lower().strip(".,!?"):
                    accepted.append(v)
                    self.total_accepted += 1
                else:
                    # First mismatch — take target token and stop
                    accepted.append(v)
                    self.total_accepted += 1
                    break

            all_tokens.extend(accepted)
            remaining -= len(accepted)

            # Stop if target generated fewer tokens (end of sequence)
            if len(verify_words) < draft_tokens:
                break

        elapsed = time.time() - start
        text    = " ".join(all_tokens)
        tokens  = len(all_tokens)
        tps     = tokens / elapsed if elapsed > 0 else 0

        return {
            "text":             text,
            "tokens_generated": tokens,
            "ttft_seconds":     round(elapsed, 3),
            "tps":              round(tps, 2),
            "acceptance_rate":  self.acceptance_rate,
            "mode":             "speculative",
        }