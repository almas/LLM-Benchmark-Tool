"""CLI interface for the LLM Benchmark Tool."""

import argparse
import asyncio
import sys
import os
from typing import List, Optional
from llm_benchmark.logger import setup_logger, logger

from llm_benchmark.models import (
    ProviderType,
    BenchmarkType,
    OutputFormat,
    BenchmarkConfig,
)
from llm_benchmark.providers import get_provider
from llm_benchmark.benchmarks import BenchmarkRunner
from llm_benchmark.reports import ReportGenerator


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="llm-benchmark",
        description="LLM Benchmark Tool - Evaluate Large Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks with dummy provider
  python main.py --provider dummy --model test-model --benchmarks all

  # Run specific benchmarks
  python main.py --provider dummy --model test-model --benchmarks performance accuracy

  # Run with verbose output
  python main.py --provider dummy --model test-model --benchmarks all --verbose

  # Output JSON report
  python main.py --provider dummy --model test-model --benchmarks all --output report.json

  # Output HTML report
  python main.py --provider dummy --model test-model --benchmarks all --output report.html --format html
        """,
    )

    # Provider options
    provider_group = parser.add_argument_group("Provider Options")
    provider_group.add_argument(
        "--provider", "-p",
        choices=[pt.value for pt in ProviderType],
        default="dummy",
        help="LLM provider to use (default: dummy)",
    )
    provider_group.add_argument(
        "--model", "-m",
        default=None,
        help="Model name to benchmark (required for local provider if not using default model)",
    )
    provider_group.add_argument(
        "--api-key",
        default=None,
        help="API key for the provider",
    )
    provider_group.add_argument(
        "--base-url",
        default=None,
        help="Base URL for the API (for local providers or custom endpoints)",
    )
    provider_group.add_argument(
        "--azure-endpoint",
        default=None,
        help="Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com)",
    )
    provider_group.add_argument(
        "--api-version",
        default="2024-02-01",
        help="API version for Azure (default: 2024-02-01)",
    )

    # Benchmark options
    benchmark_group = parser.add_argument_group("Benchmark Options")
    benchmark_group.add_argument(
        "--benchmarks", "-b",
        nargs="+",
        choices=[bt.value for bt in BenchmarkType] + ["all"],
        default=["all"],
        help="Benchmark types to run (default: all)",
    )
    benchmark_group.add_argument(
        "--num-trials", "-n",
        type=int,
        default=10,
        help="Number of trials per benchmark (default: 10)",
    )
    benchmark_group.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Number of questions for accuracy benchmark (default: 20)",
    )
    benchmark_group.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens in response (default: 500)",
    )
    benchmark_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)",
    )
    benchmark_group.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (format determined by extension: .json, .html, or .txt)",
    )
    output_group.add_argument(
        "--format", "-f",
        choices=[of.value for of in OutputFormat],
        default=None,
        help="Output format (default: text for stdout, determined by --output extension if specified)",
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for each trial",
    )

    return parser.parse_args(args)


def resolve_benchmark_types(benchmark_names: List[str]) -> List[BenchmarkType]:
    """Resolve benchmark name strings to BenchmarkType enums."""
    if "all" in benchmark_names:
        return list(BenchmarkType)
    return [BenchmarkType(name) for name in benchmark_names]


def resolve_output_format(args) -> OutputFormat:
    """Resolve output format from arguments."""
    if args.format:
        return OutputFormat(args.format)

    if args.output:
        ext = os.path.splitext(args.output)[1].lower()
        format_map = {
            ".json": OutputFormat.JSON,
            ".html": OutputFormat.HTML,
            ".htm": OutputFormat.HTML,
            ".txt": OutputFormat.TEXT,
        }
        return format_map.get(ext, OutputFormat.TEXT)

    return OutputFormat.TEXT


async def run_benchmark(args) -> int:
    """Run the benchmark and generate report."""
    # Initialize logger
    setup_logger(verbose=args.verbose)
    
    # Resolve benchmark types
    benchmark_types = resolve_benchmark_types(args.benchmarks)

    # Resolve output format
    output_format = resolve_output_format(args)

    # Build provider config
    provider_config = {
        "name": args.provider,
        "model": args.model,
    }

    if args.api_key:
        provider_config["api_key"] = args.api_key
    if args.base_url:
        provider_config["base_url"] = args.base_url
    if args.azure_endpoint:
        provider_config["azure_endpoint"] = args.azure_endpoint
    if args.api_version:
        provider_config["api_version"] = args.api_version

    # Create config
    config = BenchmarkConfig(
        provider=provider_config,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        num_trials=args.num_trials,
        num_questions=args.num_questions,
        output_format=output_format,
        output_file=args.output,
        verbose=args.verbose,
    )

    # Get provider
    try:
        provider_type = ProviderType(args.provider)
        provider = get_provider(provider_type, **provider_config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    logger.info(f"LLM Benchmark Tool v1.0.0")
    model_str = args.model if args.model else "default-loaded-model"
    logger.info(f"Provider: {args.provider} | Model: {model_str}")
    logger.info(f"Benchmarks: {', '.join(bt.value for bt in benchmark_types)}")
    print("=" * 50)

    # Create runner and run benchmarks
    runner = BenchmarkRunner(provider, config)

    try:
        results = await runner.run_all(benchmark_types)
    except Exception as e:
        logger.exception(f"Error running benchmarks: {e}")
        return 1

    # Generate report
    report_gen = ReportGenerator(results)
    output_file = args.output

    if output_file:
        content = report_gen.generate(output_format, output_file)
        print(f"\nReport saved to: {output_file}")
    else:
        content = report_gen.generate(output_format)
        print("\n" + content)

    return 0


def main(args=None):
    """Main entry point for the CLI."""
    parsed_args = parse_args(args)
    return asyncio.run(run_benchmark(parsed_args))


if __name__ == "__main__":
    main()