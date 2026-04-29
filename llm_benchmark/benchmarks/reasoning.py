import json
import os
from typing import List
from llm_benchmark.models import BenchmarkTask, BenchmarkType

class ReasoningBenchmark:
    """Reasoning benchmark - measures logical deduction and problem solving."""

    name = BenchmarkType.REASONING
    DATA_FILE = "llm_benchmark/data/benchmarks/reasoning.json"

    @staticmethod
    def generate_tasks(num_tasks: int = 10) -> List[BenchmarkTask]:
        if not os.path.exists(ReasoningBenchmark.DATA_FILE):
            raise FileNotFoundError(f"Benchmark data file not found: {ReasoningBenchmark.DATA_FILE}")
            
        with open(ReasoningBenchmark.DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tasks = []
        for i in range(num_tasks):
            item = data[i % len(data)]
            tasks.append(
                BenchmarkTask(
                    id=f"reas_{i}",
                    benchmark_type=BenchmarkType.REASONING,
                    prompt=item["prompt"],
                    system_prompt=item.get("system_prompt", "Think step-by-step and explain your reasoning before providing the final answer."),
                    expected_answer=item.get("expected_answer") or item.get("answer")
                )
            )
        return tasks
