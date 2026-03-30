import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import settings
import structlog

log = structlog.get_logger()

class QuantizedEngine:
    """
    Loads the same model at different quantization levels
    and exposes the same generate() interface as InferenceEngine
    so the benchmark harness can swap them transparently.
    """

    def __init__(self, mode: str = "float32"):
        """
        mode: "float32" | "int8"
        For INT4 we use llama-cpp-python with GGUF — separate class below.
        """
        assert mode in ("float32", "int8"), f"Unknown mode: {mode}"
        self.mode = mode
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self):
        log.info("Loading quantized model", mode=self.mode, model=settings.model_name)
        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

        if self.mode == "float32":
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                dtype=torch.float32,
            )
        elif self.mode == "int8":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )

        self.model.eval()
        self.is_loaded = True
        log.info("Quantized model loaded", mode=self.mode, seconds=round(time.time()-start, 2))

    def generate(self, prompt: str, max_new_tokens: int = None) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Call load_model() first.")

        max_new_tokens = max_new_tokens or settings.max_tokens
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - start

        generated_tokens = outputs[0][input_len:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tps = len(generated_tokens) / elapsed if elapsed > 0 else 0

        return {
            "text": text,
            "tokens_generated": len(generated_tokens),
            "ttft_seconds": round(elapsed, 3),
            "tps": round(tps, 2),
            "mode": self.mode,
        }

class GGUFEngine:
    """
    Runs inference using llama-cpp-python with GGUF quantized models.
    Supports Q4_K_M and Q8_0 formats — runs entirely on CPU.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.llm = None
        self.is_loaded = False

    def load_model(self):
        from llama_cpp import Llama
        log.info("Loading GGUF model", path=self.model_path)
        start = time.time()
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            verbose=False,
        )
        self.is_loaded = True
        log.info("GGUF model loaded", seconds=round(time.time()-start, 2))

    def generate(self, prompt: str, max_new_tokens: int = 200) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Call load_model() first.")

        formatted = f"<|system|>You are a helpful assistant.</s><|user|>{prompt}</s><|assistant|>"

        start = time.time()
        output = self.llm(
            formatted,
            max_tokens=max_new_tokens,
            echo=False,
        )
        elapsed = time.time() - start

        text = output["choices"][0]["text"].strip()
        tokens_generated = output["usage"]["completion_tokens"]
        tps = tokens_generated / elapsed if elapsed > 0 else 0

        return {
            "text": text,
            "tokens_generated": tokens_generated,
            "ttft_seconds": round(elapsed, 3),
            "tps": round(tps, 2),
            "mode": self.model_path.split(".")[-2],
        }