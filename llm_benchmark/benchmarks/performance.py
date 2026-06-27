import json
import os
from typing import List
from llm_benchmark.models import BenchmarkTask, BenchmarkType

class PerformanceBenchmark:
    """Performance benchmark - measures response time and throughput."""

    name = BenchmarkType.PERFORMANCE
    DATA_FILE = "llm_benchmark/data/benchmarks/performance.json"

    @staticmethod
    def generate_tasks(num_tasks: int = 10) -> List[BenchmarkTask]:
        if not os.path.exists(PerformanceBenchmark.DATA_FILE):
            raise FileNotFoundError(f"Benchmark data file not found: {PerformanceBenchmark.DATA_FILE}")
            
        with open(PerformanceBenchmark.DATA_FILE, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            
        tasks = []
        for i in range(num_tasks):
            prompt = prompts[i % len(prompts)]
            tasks.append(
                BenchmarkTask(
                    id=f"perf_{i}",
                    benchmark_type=BenchmarkType.PERFORMANCE,
                    prompt=prompt,
                    system_prompt=(
                        "You are an expert software engineer. Produce a complete developer-grade answer "
                        "that includes architecture rationale, code examples, implementation details, "
                        "and test strategy. The response should be long-form, target at least 1200 tokens, "
                        "and should feel like a real code development task solution."
                    ),
                    metadata={"max_tokens": 2048},
                )
            )
        return tasks