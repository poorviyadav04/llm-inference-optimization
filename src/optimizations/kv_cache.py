import time
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
import structlog

log = structlog.get_logger()


@dataclass
class CacheEntry:
    key: str
    prompt_hash: str
    result: dict
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


class KVCache:
    """
    Simulates a KV cache with three eviction policies:
    - fifo: evict the oldest entry
    - lru:  evict the least recently used entry
    - sliding_window: keep only the last N entries
    """

    def __init__(self, max_size: int = 20, policy: str = "lru"):
        assert policy in ("fifo", "lru", "sliding_window")
        self.max_size = max_size
        self.policy = policy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _make_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.strip().lower().encode()).hexdigest()

    def get(self, prompt: str) -> Optional[dict]:
        key = self._make_key(prompt)
        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]
        entry.last_accessed = time.time()
        entry.access_count += 1
        self.hits += 1

        if self.policy == "lru":
            self.cache.move_to_end(key)

        log.info("Cache HIT", policy=self.policy, key=key[:8])
        return entry.result

    def put(self, prompt: str, result: dict):
        key = self._make_key(prompt)

        if key in self.cache:
            self.cache[key].result = result
            self.cache[key].last_accessed = time.time()
            return

        if len(self.cache) >= self.max_size:
            self._evict()

        self.cache[key] = CacheEntry(
            key=key,
            prompt_hash=key,
            result=result,
        )

        if self.policy == "lru":
            self.cache.move_to_end(key)

    def _evict(self):
        if self.policy == "fifo":
            evicted = next(iter(self.cache))
            del self.cache[evicted]
            log.info("Cache EVICT (FIFO)", key=evicted[:8])

        elif self.policy == "lru":
            evicted = next(iter(self.cache))
            del self.cache[evicted]
            log.info("Cache EVICT (LRU)", key=evicted[:8])

        elif self.policy == "sliding_window":
            evicted = next(iter(self.cache))
            del self.cache[evicted]
            log.info("Cache EVICT (sliding_window)", key=evicted[:8])

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self.cache)

    def stats(self) -> dict:
        return {
            "policy":   self.policy,
            "size":     self.size,
            "max_size": self.max_size,
            "hits":     self.hits,
            "misses":   self.misses,
            "hit_rate": self.hit_rate,
        }