import json
import os
from typing import List
from llm_benchmark.models import BenchmarkTask, BenchmarkType

class AccuracyBenchmark:
    """Accuracy benchmark - measures correctness based on gold standard answers."""

    name = BenchmarkType.ACCURACY
    DATA_FILE = "llm_benchmark/data/benchmarks/accuracy.json"

    @staticmethod
    def generate_tasks(num_tasks: int = 10) -> List[BenchmarkTask]:
        if not os.path.exists(AccuracyBenchmark.DATA_FILE):
            raise FileNotFoundError(f"Benchmark data file not found: {AccuracyBenchmark.DATA_FILE}")
            
        with open(AccuracyBenchmark.DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tasks = []
        # Data is expected to be a list of {"prompt": "...", "expected_answer": "..."}
        for i in range(num_tasks):
            item = data[i % len(data)]
            tasks.append(
                BenchmarkTask(
                    id=f"acc_{i}",
                    benchmark_type=BenchmarkType.ACCURACY,
                    prompt=item["prompt"],
                    system_prompt="Provide a concise and accurate answer.",
                    expected_answer=item.get("expected_answer") or item.get("answer")
                )
            )
        return tasks
