import json
import os
from typing import List
from llm_benchmark.models import BenchmarkTask, BenchmarkType

class MultilingualBenchmark:
    """Multilingual benchmark - measures translation and cross-lingual capabilities."""

    name = BenchmarkType.MULTILINGUAL
    DATA_FILE = "llm_benchmark/data/benchmarks/multilingual.json"

    @staticmethod
    def generate_tasks(num_tasks: int = 10) -> List[BenchmarkTask]:
        if not os.path.exists(MultilingualBenchmark.DATA_FILE):
            raise FileNotFoundError(f"Benchmark data file not found: {MultilingualBenchmark.DATA_FILE}")
            
        with open(MultilingualBenchmark.DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tasks = []
        for i in range(num_tasks):
            item = data[i % len(data)]
            tasks.append(
                BenchmarkTask(
                    id=f"multi_{i}",
                    benchmark_type=BenchmarkType.MULTILINGUAL,
                    prompt=item["prompt"],
                    system_prompt="You are a professional translator and linguist. Provide accurate and culturally appropriate translations or responses.",
                    expected_answer=item.get("expected_answer") or item.get("answer")
                )
            )
        return tasks
