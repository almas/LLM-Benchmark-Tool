import asyncio
import time
from typing import List, Dict, Any
from llm_benchmark.models import BenchmarkTask, BenchmarkTrial, BenchmarkType, BenchmarkConfig, BenchmarkResult
from llm_benchmark.providers import BaseProvider
# Removed circular import from llm_benchmark.benchmarks
from llm_benchmark.logger import logger

class BenchmarkRunner:
    """Coordinates the execution of LLM benchmarks across different tasks and models."""

    def __init__(self, provider: BaseProvider, config: BenchmarkConfig):
        self.provider = provider
        self.config = config

    async def run_all(self, benchmark_types: List[BenchmarkType]) -> List[BenchmarkResult]:
        """
        Run all specified benchmarks.
        
        Args:
            benchmark_types: List of benchmark types to execute.
            
        Returns:
            A list of BenchmarkResult objects.
        """
        all_results = []
        
        for b_type in benchmark_types:
            logger.info(f"Starting benchmark: {b_type.value}")
            
            # Import inside the method to avoid circular dependency with __init__.py
            from llm_benchmark.benchmarks import generate_benchmark_tasks
            tasks = generate_benchmark_tasks(b_type, self.config.num_trials)
            
            # Execute trials concurrently with a limit on concurrency to avoid rate limits
            trials = await self.run_benchmark(tasks)
            
            # Create a BenchmarkResult object for this specific benchmark
            result = BenchmarkResult(
                model_name=self.provider.get_provider_info().get("model", "unknown"),
                provider=self.provider.get_provider_info().get("type"), # This is likely a string, but should be ProviderType
                benchmark_type=b_type,
                trials=trials,
                start_time=time.time(),
                end_time=time.time(),
                config=self.config.__dict__
            )
            
            # Fix the provider type if it's just a string
            from llm_benchmark.models import ProviderType
            try:
                result.provider = ProviderType(result.provider)
            except (ValueError, AttributeError):
                pass

            all_results.append(result)
            
            logger.info(f"Completed benchmark: {b_type.value} with {len(trials)} trials")
            
        return all_results

    async def run_benchmark(self, tasks: List[BenchmarkTask]) -> List[BenchmarkTrial]:
        """
        Run a set of benchmark tasks concurrently.
        
        Args:
            tasks: List of tasks to execute.
            
        Returns:
            List of benchmark trials.
        """
        # Limit concurrency to 5 to be safe with most API providers
        semaphore = asyncio.Semaphore(5)
        
        async def wrapped_run(task: BenchmarkTask) -> BenchmarkTrial:
            async with semaphore:
                return await self._run_single_trial(task)

        # Create a list of coroutines
        coros = [wrapped_run(task) for task in tasks]
        
        # Run all tasks concurrently and wait for completion
        return await asyncio.gather(*coros)

    async def _run_single_trial(self, task: BenchmarkTask) -> BenchmarkTrial:
        """
        Execute a single trial with basic retry logic.
        """
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if self.config.verbose:
                    logger.debug(f"Running task {task.id} (Attempt {attempt + 1})")
                
                # The provider handles the actual API call
                result = await self.provider.run_benchmark(task, self.config.__dict__)
                
                if result.error:
                    raise Exception(result.error)
                
                # Copy expected_answer from task to trial for accuracy scoring
                if result.expected_match is None and task.expected_answer:
                    result.expected_match = task.expected_answer
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Trial {task.id} failed (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
        
        # If all retries fail, return a failed trial result
        logger.error(f"Trial {task.id} failed after {max_retries} attempts.")
        return BenchmarkTrial(
            task_id=task.id,
            prompt=task.prompt,
            response="",
            error=f"Max retries exceeded. Last error: {str(last_error)}",
            response_time=0.0
        )
