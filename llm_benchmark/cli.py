"""CLI interface for the LLM Benchmark Tool.

Effective settings are merged from several sources, highest precedence first:

    1. command line flags
    2. this tool's environment variables (``LLM_BENCH_*``)
    3. ``config.yaml`` (``--config``, ``$LLM_BENCH_CONFIG``, ``./config.yaml``,
       ``llm_benchmark/config.yaml``, ``~/.config/llm-benchmark/config.yaml``)
    4. generic SDK variables (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``,
       ``OPENAI_API_BASE``, ``AZURE_OPENAI_*``), used only when ``config.yaml``
       does not carry the setting
    5. built-in defaults
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from llm_benchmark.config import load_config, resolve_measurement, resolve_provider, summarise
from llm_benchmark.logger import setup_logger, logger
from llm_benchmark.models import (
    BenchmarkConfig,
    BenchmarkType,
    OutputFormat,
    ProviderType,
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
Configuration:
  Settings are read from config.yaml, then overridden by LLM_BENCH_* variables
  and finally by the flags below. Generic SDK variables (OPENAI_API_KEY,
  OPENAI_BASE_URL, OPENAI_API_BASE, AZURE_OPENAI_*) are only used when
  config.yaml omits that setting, so an ambient OPENAI_API_BASE cannot redirect
  the run away from the endpoint in your config file. Copy
  llm_benchmark/config.yaml.template to llm_benchmark/config.yaml (or
  ./config.yaml) and edit it. Inside the file, ${VAR} and ${VAR:default} are
  expanded from the environment, so API keys can stay out of the file.

Examples:
  # Run everything against a local vLLM/SGLang server using config.yaml
  python main.py

  # Single-stream decode speed (the default) with 20 trials
  python main.py --provider local --base-url http://127.0.0.1:8000/v1 \\
      --model Qwen/Qwen2.5-7B-Instruct --benchmarks performance -n 20

  # Aggregate throughput of the same server at 8 concurrent streams
  python main.py --benchmarks performance --concurrency 8

  # Fixed-length outputs so every trial is comparable (vLLM/SGLang)
  python main.py --benchmarks performance --ignore-eos --max-tokens 512

  # JSON report
  python main.py --benchmarks all --output report.json
        """,
    )

    # Config file
    config_group = parser.add_argument_group("Configuration Options")
    config_group.add_argument(
        "--config", "-c",
        default=None,
        metavar="PATH",
        help="Path to a YAML config file (overrides the discovery order)",
    )

    # Provider options
    provider_group = parser.add_argument_group("Provider Options")
    provider_group.add_argument(
        "--provider", "-p",
        choices=[pt.value for pt in ProviderType],
        default=None,
        help="LLM provider to use (default: from config.yaml, else dummy)",
    )
    provider_group.add_argument(
        "--model", "-m",
        default=None,
        help="Model name to benchmark (omit for local servers to use the served model)",
    )
    provider_group.add_argument(
        "--api-key",
        default=None,
        help="API key for the provider (prefer provider.api_key: ${OPENAI_API_KEY})",
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
        default=None,
        help="API version for Azure",
    )
    provider_group.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-request timeout in seconds",
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
        default=None,
        help="Number of performance trials",
    )
    benchmark_group.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions for the accuracy benchmark",
    )
    benchmark_group.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks for the reasoning/code/multilingual benchmarks",
    )

    # Measurement options - these decide *what* is being measured.
    measurement_group = parser.add_argument_group("Measurement Options")
    measurement_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens in the response",
    )
    measurement_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.0 for reproducible timings)",
    )
    measurement_group.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "Requests in flight. Keep 1 (the default) to measure the single-stream "
            "decode speed that engine dashboards report; higher values measure "
            "aggregate throughput."
        ),
    )
    measurement_group.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Unmeasured requests sent before timing starts (default: 1)",
    )
    stream_group = measurement_group.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=None,
        help="Stream tokens so TTFT and decode speed can be separated (default)",
    )
    stream_group.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming; tokens/s then includes prefill and queueing",
    )
    usage_group = measurement_group.add_mutually_exclusive_group()
    usage_group.add_argument(
        "--include-usage",
        dest="include_usage",
        action="store_true",
        default=None,
        help="Ask the server for exact token counts on the final stream chunk (default)",
    )
    usage_group.add_argument(
        "--no-include-usage",
        dest="include_usage",
        action="store_false",
        help="Do not request usage; output token counts are then estimated locally",
    )
    measurement_group.add_argument(
        "--token-source",
        choices=["auto", "usage", "estimate"],
        default=None,
        help="Where output token counts come from (default: auto)",
    )
    measurement_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed sent to the server (default: 1234)",
    )
    measurement_group.add_argument(
        "--no-seed",
        dest="no_seed",
        action="store_true",
        default=False,
        help="Do not send a seed at all (some servers reject it)",
    )
    measurement_group.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Runner-level retries for failed requests (SDK retries are always off)",
    )
    measurement_group.add_argument(
        "--cooldown",
        type=float,
        default=None,
        help="Seconds to wait between trials (thermal/power recovery)",
    )
    measurement_group.add_argument(
        "--prompt-profile",
        choices=["short", "medium", "heavy"],
        default=None,
        help="Prompt shape used by the performance benchmark (default: heavy)",
    )
    measurement_group.add_argument(
        "--ignore-eos",
        action="store_true",
        default=None,
        help="Force exactly max_tokens output tokens (vLLM/SGLang servers)",
    )
    measurement_group.add_argument(
        "--extra-body",
        default=None,
        metavar="JSON",
        help="Extra JSON body sent with every request, e.g. '{\"enable_thinking\": false}'",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path",
    )
    output_group.add_argument(
        "--format", "-f",
        choices=[fmt.value for fmt in OutputFormat],
        default=None,
        help="Output format (default: inferred from --output extension, else text)",
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose per-trial output",
    )

    return parser.parse_args(args)


# CLI attribute names understood by llm_benchmark.config.resolve_*().
CLI_OVERRIDE_KEYS = (
    "provider", "model", "api_key", "base_url", "azure_endpoint", "api_version",
    "timeout", "num_trials", "num_questions", "num_tasks", "max_tokens",
    "temperature", "concurrency", "warmup", "stream", "include_usage",
    "token_source", "seed", "retries", "cooldown", "prompt_profile", "ignore_eos",
)


def collect_cli_overrides(args) -> Dict[str, Any]:
    """Turn parsed flags into the override mapping used by the config loader."""
    overrides: Dict[str, Any] = {key: getattr(args, key, None) for key in CLI_OVERRIDE_KEYS}

    extra_body = getattr(args, "extra_body", None)
    if extra_body:
        try:
            parsed = json.loads(extra_body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--extra-body is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError('--extra-body must be a JSON object, e.g. \'{"k": 1}\'')
        overrides["extra_body"] = parsed

    return overrides


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


def build_benchmark_config(args, provider_cfg: Dict[str, Any], measurement: Dict[str, Any]) -> BenchmarkConfig:
    """Assemble the effective :class:`BenchmarkConfig`."""
    provider = dict(provider_cfg)
    provider.pop("_provenance", None)
    if provider.get("name") == ProviderType.AZURE.value and provider.get("azure_endpoint"):
        # Azure routes requests through the endpoint instead of a generic base_url.
        provider["base_url"] = provider.pop("azure_endpoint")
    provider.pop("azure_endpoint", None)

    settings = {k: v for k, v in measurement.items() if not k.startswith("_")}
    if getattr(args, "no_seed", False):
        settings["seed"] = None

    return BenchmarkConfig(
        provider=provider,
        output_format=resolve_output_format(args),
        output_file=args.output,
        verbose=bool(args.verbose),
        **settings,
    )


async def run_benchmark(args) -> int:
    """Run the benchmark and generate the report."""
    setup_logger(verbose=args.verbose)

    try:
        raw_config, config_path = load_config(args.config)
        provider_cfg = resolve_provider(raw_config, collect_cli_overrides(args))
        measurement = resolve_measurement(raw_config, collect_cli_overrides(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    benchmark_types = resolve_benchmark_types(args.benchmarks)
    output_format = resolve_output_format(args)
    config = build_benchmark_config(args, provider_cfg, measurement)

    summary = summarise(provider_cfg, measurement)
    logger.info(
        f"Effective configuration: provider={provider_cfg.get('name')} "
        f"model={provider_cfg.get('model') or '<engine default>'} "
        f"concurrency={measurement.get('concurrency')} "
        f"warmup={measurement.get('warmup_requests')} "
        f"streaming={measurement.get('stream')} "
        f"output={args.output or 'stdout'}/{output_format.value}"
    )
    print("=" * 64)
    print("LLM Benchmark Tool v1.0.0")
    print(f"config file   : {config_path if config_path else '<none - defaults/env only>'}")
    print(summary)
    print(f"benchmarks    : {', '.join(bt.value for bt in benchmark_types)}")
    print(f"output        : {args.output or '<stdout>'} ({output_format.value})")
    print("=" * 64)

    # "name" selects the provider class; it is not a constructor argument.
    provider_kwargs = {
        k: v for k, v in config.provider.items() if v is not None and k != "name"
    }
    try:
        provider = get_provider(config.provider.get("name"), **provider_kwargs)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    logger.info(
        f"Provider: {config.provider.get('name')} | "
        f"Model: {config.provider.get('model') or '<engine default>'}"
    )

    runner = BenchmarkRunner(provider, config)
    try:
        results = await runner.run_all_benchmarks(benchmark_types)
    except Exception as exc:
        logger.exception(f"Error running benchmarks: {exc}")
        return 1
    finally:
        await provider.close()

    if not results:
        print("No benchmark results produced - see the errors above.", file=sys.stderr)
        return 1

    report_gen = ReportGenerator(results)
    if args.output:
        report_gen.generate(output_format, args.output)
        print(f"\nReport saved to: {args.output}")
    else:
        print("\n" + report_gen.generate(output_format))

    return 0


def main(args=None):
    """Main entry point for the CLI."""
    parsed_args = parse_args(args)
    return asyncio.run(run_benchmark(parsed_args))


if __name__ == "__main__":
    main()

