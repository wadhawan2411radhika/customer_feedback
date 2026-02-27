"""Cost and latency tracking for OpenAI streaming calls."""

from dataclasses import dataclass
from typing import Optional

# USD per 1M tokens
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}


@dataclass
class CostRecord:
    model: str
    version: str = ""           # "baseline" or "enhanced"
    query: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    time_to_first_token: Optional[float] = None
    total_time: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> Optional[float]:
        pricing = MODEL_PRICING.get(self.model)
        if not pricing:
            return None
        return (
            self.input_tokens  / 1_000_000 * pricing["input"] +
            self.output_tokens / 1_000_000 * pricing["output"]
        )

    def to_dict(self, output: str = "") -> dict:
        return {
            "query": self.query,
            "version": self.version,
            "output": output,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6) if self.cost_usd is not None else None,
            "ttft_s": round(self.time_to_first_token, 3) if self.time_to_first_token else None,
            "total_time_s": round(self.total_time, 3) if self.total_time else None,
        }

    def __str__(self) -> str:
        cost  = f"${self.cost_usd:.6f}" if self.cost_usd is not None else "N/A"
        ttft  = f"{self.time_to_first_token:.3f}s" if self.time_to_first_token is not None else "N/A"
        total = f"{self.total_time:.3f}s" if self.total_time is not None else "N/A"
        return (
            f"input={self.input_tokens} tok  "
            f"output={self.output_tokens} tok  "
            f"total={self.total_tokens} tok  "
            f"cost={cost}  "
            f"ttft={ttft}  "
            f"total_time={total}"
        )