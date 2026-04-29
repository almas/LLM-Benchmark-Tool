"""LLM Benchmark Tool - A comprehensive tool for evaluating Large Language Models."""

__version__ = "1.0.0"
__author__ = "LLM Benchmark Team"

from llm_benchmark.models import BenchmarkResult, BenchmarkConfig
from llm_benchmark.cli import main

__all__ = ["BenchmarkResult", "BenchmarkConfig", "main"]