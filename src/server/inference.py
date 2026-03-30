import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import settings
import structlog

log = structlog.get_logger()

class InferenceEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self):
        log.info("Loading model", model=settings.model_name)
        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        self.is_loaded = True

        elapsed = time.time() - start
        log.info("Model loaded", seconds=round(elapsed, 2))

    def generate(self, prompt: str, max_new_tokens: int = None) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        max_new_tokens = max_new_tokens or settings.max_tokens

        # TinyLlama uses chat template
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        ttft_start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        ttft = time.time() - ttft_start

        generated_tokens = outputs[0][input_len:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tps = len(generated_tokens) / ttft if ttft > 0 else 0

        return {
            "text": text,
            "tokens_generated": len(generated_tokens),
            "ttft_seconds": round(ttft, 3),
            "tps": round(tps, 2),
        }
engine = InferenceEngine()