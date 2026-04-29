from typing import List, Type
from llm_benchmark.models import BenchmarkTask, BenchmarkType
from llm_benchmark.benchmarks.performance import PerformanceBenchmark
from llm_benchmark.benchmarks.accuracy import AccuracyBenchmark
from llm_benchmark.benchmarks.reasoning import ReasoningBenchmark
from llm_benchmark.benchmarks.code_generation import CodeGenerationBenchmark
from llm_benchmark.benchmarks.multilingual import MultilingualBenchmark
from llm_benchmark.benchmarks.runner import BenchmarkRunner

# Map benchmark types to their respective classes
BENCHMARK_MAP: dict[BenchmarkType, Type] = {
    BenchmarkType.PERFORMANCE: PerformanceBenchmark,
    BenchmarkType.ACCURACY: AccuracyBenchmark,
    BenchmarkType.REASONING: ReasoningBenchmark,
    BenchmarkType.CODE_GENERATION: CodeGenerationBenchmark,
    BenchmarkType.MULTILINGUAL: MultilingualBenchmark,
}

def generate_benchmark_tasks(benchmark_type: BenchmarkType, num_tasks: int = 10) -> List[BenchmarkTask]:
    """
    Factory function to generate tasks for a specific benchmark type.
    """
    benchmark_class = BENCHMARK_MAP.get(benchmark_type)
    if not benchmark_class:
        raise ValueError(f"Unsupported benchmark type: {benchmark_type}")
    
    return benchmark_class.generate_tasks(num_tasks)