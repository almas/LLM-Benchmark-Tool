import json
import os
from typing import List
from llm_benchmark.models import BenchmarkTask, BenchmarkType

class CodeGenerationBenchmark:
    """Code generation benchmark - measures programming capability."""

    name = BenchmarkType.CODE_GENERATION
    DATA_FILE = "llm_benchmark/data/benchmarks/code_generation.json"

    @staticmethod
    def generate_tasks(num_tasks: int = 10) -> List[BenchmarkTask]:
        if not os.path.exists(CodeGenerationBenchmark.DATA_FILE):
            raise FileNotFoundError(f"Benchmark data file not found: {CodeGenerationBenchmark.DATA_FILE}")
            
        with open(CodeGenerationBenchmark.DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tasks = []
        for i in range(num_tasks):
            item = data[i % len(data)]
            tasks.append(
                BenchmarkTask(
                    id=f"code_{i}",
                    benchmark_type=BenchmarkType.CODE_GENERATION,
                    prompt=item["prompt"],
                    system_prompt="You are an expert software engineer. Provide clean, efficient, and well-documented code.",
                    expected_answer=item.get("expected_answer") or item.get("answer")
                )
            )
        return tasks
