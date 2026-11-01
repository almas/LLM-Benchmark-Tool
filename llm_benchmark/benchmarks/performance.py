"""Performance benchmark - measures latency and token throughput."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_benchmark.logger import logger
from llm_benchmark.models import BenchmarkTask, BenchmarkType

# Prompt profiles control how much work each request does. ``heavy`` is the
# historical default: long outputs make the decode phase dominate the wall
# clock, which is what you want when measuring tokens/second.
PROMPT_PROFILES: Dict[str, Dict[str, Any]] = {
    "short": {
        "system_prompt": "Answer in one short sentence.",
        "max_tokens": 128,
    },
    "medium": {
        "system_prompt": (
            "Answer clearly and completely, but stay concise. Aim for a short "
            "paragraph plus a small code example."
        ),
        "max_tokens": 512,
    },
    "heavy": {
        "system_prompt": (
            "You are an expert software engineer. Produce a complete developer-grade answer "
            "that includes architecture rationale, code examples, implementation details, "
            "and test strategy. The response should be long-form, target at least 1200 tokens, "
            "and should feel like a real code development task solution."
        ),
        "max_tokens": 2048,
    },
}


class PerformanceBenchmark:
    """Performance benchmark - measures response time and throughput."""

    name = BenchmarkType.PERFORMANCE
    # Resolved relative to the package so the tool works from any directory.
    DATA_FILE = str(Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "performance.json")
    DEFAULT_PROFILE = "heavy"

    def __init__(
        self,
        provider: Any = None,
        config: Any = None,
        num_trials: int = 10,
        prompt_profile: Optional[str] = None,
    ):
        self.provider = provider
        self.config = config
        self.num_trials = int(num_trials or 10)
        profile = prompt_profile or getattr(config, "prompt_profile", None) or self.DEFAULT_PROFILE
        if profile not in PROMPT_PROFILES:
            logger.warning(
                "Unknown prompt_profile %r (available: %s) - using %r.",
                profile, ", ".join(PROMPT_PROFILES), self.DEFAULT_PROFILE,
            )
            profile = self.DEFAULT_PROFILE
        self.prompt_profile = profile

    def get_tasks(self) -> List[BenchmarkTask]:
        return PerformanceBenchmark.generate_tasks(self.num_trials, self.prompt_profile)

    @staticmethod
    def generate_tasks(num_tasks: int = 10, prompt_profile: str = DEFAULT_PROFILE) -> List[BenchmarkTask]:
        if not os.path.exists(PerformanceBenchmark.DATA_FILE):
            raise FileNotFoundError(f"Benchmark data file not found: {PerformanceBenchmark.DATA_FILE}")

        with open(PerformanceBenchmark.DATA_FILE, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        if not prompts:
            raise ValueError(f"Benchmark data file is empty: {PerformanceBenchmark.DATA_FILE}")

        profile = PROMPT_PROFILES.get(prompt_profile) or PROMPT_PROFILES[PerformanceBenchmark.DEFAULT_PROFILE]

        tasks: List[BenchmarkTask] = []
        for i in range(num_tasks):
            tasks.append(
                BenchmarkTask(
                    id=f"perf_{i}",
                    benchmark_type=BenchmarkType.PERFORMANCE,
                    prompt=prompts[i % len(prompts)],
                    system_prompt=profile["system_prompt"],
                    # The runner only applies this when max_tokens was not set
                    # explicitly on the CLI or in config.yaml.
                    metadata={"max_tokens": profile["max_tokens"], "prompt_profile": prompt_profile},
                )
            )
        return tasks

    # ------------------------------------------------------------- reporting
    def performance_summary(self, result: Any) -> Dict[str, Any]:
        """Condensed, engine-comparable view of one performance benchmark."""
        decode = result.decode_tps_stats
        ttft = result.ttft_stats
        e2e = result.e2e_tps_stats
        itl = result.itl_stats

        return {
            "benchmark": result.benchmark_type.value,
            "model": result.model_name,
            "provider": getattr(result.provider, "value", result.provider),
            "measurement_mode": "streaming" if result.streaming_measured else "non-streaming",
            "prompt_profile": self.prompt_profile,
            "trials": len(result.trials),
            "usable_trials": len(result.successful_trials),
            "failed_trials": len(result.failed_trials),
            "duration": result.duration,
            "ttft_seconds": ttft,
            "decode_tokens_per_second": decode,
            "e2e_tokens_per_second": e2e,
            "inter_token_latency_seconds": itl,
            "aggregate_output_tokens_per_second": result.aggregate_throughput,
            "total_output_tokens": result.total_output_tokens,
            "total_input_tokens": result.total_input_tokens,
            "token_sources": sorted({t.token_source for t in result.successful_trials if t.token_source}),
        }
