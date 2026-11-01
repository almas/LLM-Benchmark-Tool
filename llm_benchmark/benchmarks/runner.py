"""Benchmark runner for coordinating evaluations."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from llm_benchmark.logger import logger
from llm_benchmark.models import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkTask,
    BenchmarkType,
    BenchmarkTrial,
    estimate_tokens,
)
from llm_benchmark.providers import BaseProvider


class BenchmarkRunner:
    """Orchestrates benchmark execution."""

    def __init__(self, provider: BaseProvider, config: BenchmarkConfig):
        self.provider = provider
        self.config = config
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._cooldown = float(getattr(config, "cooldown", 0.0) or 0.0)

    # ------------------------------------------------------------ public API
    async def run_benchmark(
        self,
        benchmark_type: BenchmarkType,
        tasks: Optional[List[BenchmarkTask]] = None,
        num_tasks: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run a complete benchmark of the given type."""
        if isinstance(benchmark_type, str):
            benchmark_type = BenchmarkType(benchmark_type)

        settings = self.config.settings_for(benchmark_type)

        if tasks is None:
            tasks = self._get_tasks(benchmark_type)
        if num_tasks is not None:
            tasks = tasks[:num_tasks]

        concurrency = max(1, int(getattr(self.config, "concurrency", 1) or 1))
        started_at = time.time()

        if not tasks:
            logger.warning(f"No tasks available for benchmark: {benchmark_type.value}")
            return BenchmarkResult(
                benchmark_type=benchmark_type,
                model_name=self._model_name(),
                provider=self._provider_name(),
                trials=[],
                start_time=started_at,
                end_time=time.time(),
                config={"error": "No tasks available"},
            )

        mode = "streaming" if settings.get("stream") else "non-streaming"
        logger.info(
            f"Running {benchmark_type.value}: {len(tasks)} task(s), "
            f"concurrency={concurrency}, mode={mode}, max_tokens={settings.get('max_tokens')}"
        )

        await self._warmup(benchmark_type, settings)
        trials = await self._run_tasks(tasks, settings)

        result = BenchmarkResult(
            benchmark_type=benchmark_type,
            model_name=self._model_name(),
            provider=self._provider_name(),
            trials=trials,
            start_time=started_at,
            end_time=time.time(),
            config={
                "concurrency": concurrency,
                "mode": mode,
                "warmup_requests": int(getattr(self.config, "warmup_requests", 0) or 0),
                "token_source": settings.get("token_source"),
                "settings": settings,
            },
        )

        actual_source = result.measured_token_source
        if actual_source != "unknown":
            # What actually happened beats what was requested: a server that
            # sends no usage field means the numbers are estimates.
            result.config["token_source"] = actual_source

        decode = result.decode_tps_stats
        ttft = result.ttft_stats
        e2e = result.e2e_tps_stats
        logger.info(
            f"{benchmark_type.value}: {len(result.successful_trials)}/{len(result.trials)} usable | "
            f"decode mean {_num(decode['mean'])} / median {_num(decode['median'])} / "
            f"p95 {_num(decode['p95'])} tok/s | e2e median {_num(e2e['median'])} tok/s | "
            f"TTFT median {_fmt(ttft['median'])}s / p95 {_fmt(ttft['p95'])}s | "
            f"aggregate {_num(result.aggregate_throughput)} tok/s"
        )
        return result

    async def run_all_benchmarks(self, benchmark_types: List[BenchmarkType]) -> List[BenchmarkResult]:
        """Run multiple benchmarks sequentially."""
        results: List[BenchmarkResult] = []
        for benchmark_type in benchmark_types:
            try:
                results.append(await self.run_benchmark(benchmark_type))
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(f"Error running {benchmark_type.value}: {exc}")
        return results

    # Backwards-compatible alias.
    run_all = run_all_benchmarks

    async def run_performance_benchmark(self, num_trials: Optional[int] = None) -> Dict[str, Any]:
        """Run the performance benchmark and return its summary."""
        from llm_benchmark.benchmarks.performance import PerformanceBenchmark

        if num_trials is None:
            per = self.config.per_benchmark.get(BenchmarkType.PERFORMANCE.value) or {}
            num_trials = int(per.get("num_trials", self.config.num_trials))

        benchmark = PerformanceBenchmark(self.provider, self.config, num_trials=num_trials)
        result = await self.run_benchmark(BenchmarkType.PERFORMANCE, benchmark.get_tasks())
        return benchmark.performance_summary(result)

    # --------------------------------------------------------------- helpers
    async def _warmup(self, benchmark_type: BenchmarkType, settings: Dict[str, Any]) -> None:
        """Send unmeasured requests so the engine is warm before timing.

        The first request to a cold engine pays for KV-cache allocation, CUDA
        graph capture and lazy weight loading. Counting it drags the reported
        tokens/s down.
        """
        count = int(getattr(self.config, "warmup_requests", 0) or 0)
        if count <= 0:
            return

        task = BenchmarkTask(
            id="warmup",
            benchmark_type=benchmark_type,
            prompt="Warm-up request: briefly describe what a token is in a large language model.",
        )
        warmup_tokens = min(64, int(settings.get("max_tokens", 64) or 64))

        for index in range(count):
            trial = await self.provider.warmup(task, max_tokens=warmup_tokens)
            if trial.error:
                logger.warning(f"Warm-up {index + 1}/{count} failed: {trial.error}")
            elif self.config.verbose and trial.ttft is not None:
                logger.debug(
                    f"Warm-up {index + 1}/{count}: TTFT {trial.ttft:.3f}s, "
                    f"decode {trial.decode_tokens_per_second or 0.0:.1f} tok/s"
                )
        await asyncio.sleep(0.5)

    async def _run_tasks(self, tasks: List[BenchmarkTask], settings: Dict[str, Any]) -> List[BenchmarkTrial]:
        """Run every task under a semaphore-limited event loop."""
        concurrency = max(1, int(getattr(self.config, "concurrency", 1) or 1))
        self._semaphore = asyncio.Semaphore(concurrency)

        outcomes = await asyncio.gather(
            *(self._run_task(task, settings) for task in tasks),
            return_exceptions=True,
        )

        trials: List[BenchmarkTrial] = []
        for task, outcome in zip(tasks, outcomes):
            if isinstance(outcome, BenchmarkTrial):
                trials.append(outcome)
            elif isinstance(outcome, Exception):
                logger.error(f"Task {task.id} raised: {outcome}")
                trials.append(BenchmarkTrial(task_id=task.id, prompt=task.prompt, error=str(outcome)))
        return trials

    async def _run_task(self, task: BenchmarkTask, settings: Dict[str, Any]) -> BenchmarkTrial:
        """Run one task; failures are retried and never enter the statistics."""
        assert self._semaphore is not None
        retries = max(0, int(getattr(self.config, "retries", 0) or 0))

        task_settings = dict(settings)
        task_max_tokens = (task.metadata or {}).get("max_tokens")
        if task_max_tokens and not task_settings.get("max_tokens_explicit"):
            # Benchmark definitions may ask for longer outputs; an explicit
            # max_tokens from the CLI or config.yaml always wins.
            task_settings["max_tokens"] = int(task_max_tokens)

        async with self._semaphore:
            attempt = 0
            while True:
                try:
                    trial = await self.provider.run_benchmark(task, task_settings)
                except Exception as exc:  # provider produced no usable trial
                    trial = BenchmarkTrial(
                        task_id=task.id,
                        prompt=task.prompt,
                        error=f"{type(exc).__name__}: {exc}",
                    )

                trial.task_id = task.id
                trial.prompt = task.prompt
                trial.benchmark_type = task.benchmark_type.value
                trial.attempt = attempt + 1
                if trial.expected_match is None and task.expected_answer:
                    # Carry the gold answer onto the trial so the accuracy
                    # score can be recomputed from recorded trials.
                    trial.expected_match = task.expected_answer
                if not trial.error:
                    self._ensure_token_counts(trial)

                if not trial.error or attempt >= retries:
                    break

                attempt += 1
                logger.warning(f"Task {task.id} failed ({trial.error}); retry {attempt}/{retries}.")
                await asyncio.sleep(min(2 ** attempt, 10))

            if trial.error:
                logger.error(
                    f"Task {task.id} failed after {trial.attempt} attempt(s): {trial.error}"
                )
            elif self.config.verbose:
                logger.debug(
                    f"Task {task.id}: output_tokens={trial.output_tokens}, "
                    f"ttft={_fmt(trial.ttft)}s, decode={_fmt(trial.decode_tokens_per_second)} tok/s, "
                    f"e2e={_fmt(trial.e2e_tokens_per_second)} tok/s, tokens={trial.token_source}"
                )

            if self._cooldown > 0:
                await asyncio.sleep(self._cooldown)
            return trial

    def _ensure_token_counts(self, trial: BenchmarkTrial) -> BenchmarkTrial:
        """Fill in missing token counts so no measured trial is discarded.

        Providers normally read the counts from the server's usage field. When
        the endpoint does not send one (older or non-conforming servers), the
        runner estimates locally instead of dropping the sample, and labels the
        benchmark's token source as "estimate".
        """
        token_source = str(getattr(trial, "token_source", "auto") or "auto").lower()

        if trial.output_tokens is None:
            response = trial.response or ""
            chunks = int(getattr(trial, "chunks", 0) or 0)
            trial.output_tokens = estimate_tokens(response) if response else (chunks or None)
            if trial.output_tokens and token_source == "auto":
                trial.token_source = "estimate"

        if trial.input_tokens is None:
            estimated_prompt = estimate_tokens(trial.prompt or "")
            if estimated_prompt:
                trial.input_tokens = estimated_prompt
                if token_source == "auto":
                    trial.token_source = "estimate"

        return trial

    def _get_tasks(self, benchmark_type: BenchmarkType) -> List[BenchmarkTask]:
        """Build the task list for one benchmark, honouring per-benchmark counts."""
        # Imported lazily: llm_benchmark.benchmarks imports this module.
        from llm_benchmark.benchmarks import generate_benchmark_tasks

        per = self.config.per_benchmark.get(benchmark_type.value) or {}
        if benchmark_type == BenchmarkType.ACCURACY:
            count = int(per.get("num_questions", self.config.num_questions))
        elif benchmark_type == BenchmarkType.PERFORMANCE:
            count = int(per.get("num_trials", self.config.num_trials))
        else:
            count = int(per.get("num_tasks", self.config.num_tasks))
        return generate_benchmark_tasks(benchmark_type, count)

    def _model_name(self) -> str:
        return self.provider.get_provider_info().get("model") or "unknown"

    def _provider_name(self) -> str:
        info = self.provider.get_provider_info()
        provider = info.get("type", "unknown")
        try:
            from llm_benchmark.models import ProviderType
            return ProviderType(provider)
        except (ValueError, KeyError):
            return provider


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _num(value: Optional[float], digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"

