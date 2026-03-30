import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional
import structlog

log = structlog.get_logger()


@dataclass
class Request:
    request_id: str
    prompt: str
    max_new_tokens: int = 200
    priority: int = 1          # lower = higher priority
    created_at: float = field(default_factory=time.time)
    result: Optional[dict] = None
    completed_at: Optional[float] = None

    @property
    def wait_time(self) -> float:
        if self.completed_at:
            return round(self.completed_at - self.created_at, 3)
        return round(time.time() - self.created_at, 3)


class NoBatchScheduler:
    """Processes one request at a time — baseline."""

    def __init__(self, engine):
        self.engine = engine
        self.processed = 0

    def process(self, requests: list[Request]) -> list[Request]:
        for req in requests:
            req.result = self.engine.generate(req.prompt, req.max_new_tokens)
            req.completed_at = time.time()
            self.processed += 1
            log.info("NoBatch done", request_id=req.request_id,
                     ttft=req.result["ttft_seconds"])
        return requests


class StaticBatchScheduler:
    """
    Waits for batch_size requests then processes together.
    Naive — waits for slowest request before returning any result.
    """

    def __init__(self, engine, batch_size: int = 4):
        self.engine = engine
        self.batch_size = batch_size
        self.processed = 0

    def process(self, requests: list[Request]) -> list[Request]:
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            log.info("StaticBatch processing", batch_size=len(batch))
            batch_start = time.time()
            for req in batch:
                req.result = self.engine.generate(req.prompt, req.max_new_tokens)
                self.processed += 1
            batch_end = time.time()
            # All requests in batch get the same completion time (slowest)
            for req in batch:
                req.completed_at = batch_end
            log.info("StaticBatch done", batch_size=len(batch),
                     elapsed=round(batch_end - batch_start, 2))
        return requests


class ContinuousBatchScheduler:
    """
    Fills free slots immediately as requests complete.
    New requests don't wait for the full batch to finish.
    """

    def __init__(self, engine, max_concurrent: int = 4):
        self.engine = engine
        self.max_concurrent = max_concurrent
        self.processed = 0

    def process(self, requests: list[Request]) -> list[Request]:
        queue = list(sorted(requests, key=lambda r: r.priority))
        active = []
        completed = []

        while queue or active:
            # Fill slots
            while len(active) < self.max_concurrent and queue:
                req = queue.pop(0)
                log.info("ContinuousBatch: slot filled",
                         request_id=req.request_id,
                         active=len(active)+1)
                req.result = self.engine.generate(req.prompt, req.max_new_tokens)
                req.completed_at = time.time()
                self.processed += 1
                completed.append(req)
                log.info("ContinuousBatch: request done",
                         request_id=req.request_id,
                         ttft=req.result["ttft_seconds"])

            if queue:
                active = []

        return completed