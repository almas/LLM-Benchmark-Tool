"""LLM Provider interface and implementations.

All API providers measure timing with ``time.perf_counter()`` around a *streamed*
completion so that the three phases of a request stay separable:

    t0 ─── queue + prefill ───► first token ─── decode ───► last token ─► stream end
          (TTFT)                        (decode_time / ITL)          (tail/usage chunk)

``response_time`` is the end-to-end wall clock, ``ttft`` the time to the first
generated token and ``decode_time`` the pure generation window. The tokens/s
reported by the engine dashboard corresponds to ``decode_time``, **not** to
``response_time``.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from llm_benchmark.logger import logger
from llm_benchmark.models import (
    BenchmarkTask,
    BenchmarkTrial,
    Message,
    ProviderType,
    estimate_tokens,
)


def build_messages(task: BenchmarkTask) -> List[Message]:
    """Turn a benchmark task into a chat message list."""
    messages: List[Message] = []
    if task.system_prompt:
        messages.append(Message(role="system", content=task.system_prompt))
    messages.append(Message(role="user", content=task.prompt))
    return messages


def prompt_text(messages: List[Message]) -> str:
    return "\n".join(m.content for m in messages if m.content)


@dataclass
class StreamTiming:
    """Raw timings collected while consuming one SSE completion stream."""

    request_started: float
    stream_ended: float
    first_token_at: Optional[float] = None
    last_token_at: Optional[float] = None
    chunks: int = 0
    content: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None

    @property
    def total_time(self) -> float:
        return self.stream_ended - self.request_started

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.request_started

    @property
    def decode_time(self) -> Optional[float]:
        if self.first_token_at is None or self.last_token_at is None:
            return None
        if self.last_token_at <= self.first_token_at:
            return None
        return self.last_token_at - self.first_token_at


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    provider_type: ProviderType = ProviderType.DUMMY

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        **kwargs,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.options = kwargs

    # ------------------------------------------------------------------ API
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Message],
        max_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs,
    ) -> BenchmarkTrial:
        """Generate a chat completion and return a measured trial result."""
        raise NotImplementedError("Subclasses must implement chat_completion")

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Return provider information."""
        raise NotImplementedError("Subclasses must implement get_provider_info")

    async def close(self) -> None:
        """Release network resources (no-op by default)."""
        return None

    # -------------------------------------------------------- shared driver
    async def run_benchmark(self, task: BenchmarkTask, config: Dict[str, Any]) -> BenchmarkTrial:
        """Run a single benchmark task and return the measured result."""
        config = config or {}
        return await self.chat_completion(
            build_messages(task),
            max_tokens=int(config.get("max_tokens", 500)),
            temperature=float(config.get("temperature", 0.0)),
            stream=bool(config.get("stream", True)),
            include_usage=bool(config.get("include_usage", True)),
            token_source=str(config.get("token_source", "auto")),
            seed=config.get("seed"),
            ignore_eos=bool(config.get("ignore_eos", False)),
            extra_body=dict(config.get("extra_body") or {}),
        )

    async def warmup(
        self,
        task: BenchmarkTask,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> BenchmarkTrial:
        """Fire a throw-away request so the engine is warm before measuring."""
        trial = await self.chat_completion(
            build_messages(task),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            include_usage=False,
        )
        trial.warmup = True
        return trial


async def consume_stream(stream: AsyncIterator[Any], request_started: float) -> StreamTiming:
    """Consume an SSE completion stream, timing every generated chunk.

    Role-only and heartbeat chunks are ignored on purpose: they carry no
    generated token and would otherwise falsify the time-to-first-token.
    """
    parts: List[str] = []
    timing = StreamTiming(request_started=request_started, stream_ended=request_started)

    async for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            if prompt_tokens is not None:
                timing.prompt_tokens = prompt_tokens
            if completion_tokens is not None:
                timing.completion_tokens = completion_tokens

        for choice in (getattr(chunk, "choices", None) or []):
            delta = getattr(choice, "delta", None)
            if delta is not None:
                piece = getattr(delta, "content", None) or ""
                # Reasoning models stream their chain of thought in a separate
                # field; those tokens are generated by the engine as well.
                reasoning = getattr(delta, "reasoning_content", None) or ""
                text = piece + reasoning
                if text:
                    now = time.perf_counter()
                    if timing.first_token_at is None:
                        timing.first_token_at = now
                    timing.last_token_at = now
                    timing.chunks += 1
                    parts.append(text)
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason:
                timing.finish_reason = finish_reason

    timing.stream_ended = time.perf_counter()
    timing.content = "".join(parts)
    return timing


def is_unsupported_param_error(exc: Exception, param: str) -> bool:
    """True when the server rejected a request because of one unsupported key."""
    status = getattr(exc, "status_code", None)
    if status not in (400, 404, 422):
        return False
    return param.lower() in str(exc).lower()


class SDKProvider(BaseProvider):
    """Shared implementation for OpenAI-compatible chat-completion APIs.

    The OpenAI SDK is configured with ``max_retries=0``: silent SDK retries
    would re-send the request *inside* the measured window and inflate every
    timing. Retries are handled by the benchmark runner instead, which keeps
    the failed attempt out of the statistics.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        **kwargs,
    ):
        super().__init__(api_key=api_key, model=model, timeout=timeout, **kwargs)
        self.base_url = base_url
        self._client: Any = None

    # --------------------------------------------------------------- client
    def _new_client(self) -> Any:
        from openai import AsyncOpenAI

        kwargs: Dict[str, Any] = {
            "api_key": self.api_key or "not-needed",
            "timeout": self.timeout,
            "max_retries": 0,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncOpenAI(**kwargs)

    def _client_or_create(self) -> Any:
        if self._client is None:
            self._client = self._new_client()
        return self._client

    async def _ensure_ready(self) -> None:
        """Hook for subclasses (e.g. resolving the default model id)."""
        return None

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            self._client = None

    # --------------------------------------------------------------- params
    def _build_params(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if options.get("seed") is not None:
            params["seed"] = options["seed"]

        extra_body = dict(options.get("extra_body") or {})
        if options.get("ignore_eos"):
            # Fixed output length keeps every trial the same size, so decode
            # speed is comparable trial to trial (vLLM / SGLang extension).
            extra_body["ignore_eos"] = True
        if extra_body:
            params["extra_body"] = extra_body

        if options.get("stream", True):
            params["stream"] = True
            if options.get("include_usage", True):
                # Ask the server for exact token counts on the final chunk.
                params["stream_options"] = {"include_usage": True}
        return params

    async def _create_stream(self, params: Dict[str, Any]) -> Tuple[Any, float]:
        """Open a streamed completion; timing starts at the successful call."""
        client = self._client_or_create()
        attempt = dict(params)

        while True:
            request_started = time.perf_counter()
            try:
                stream = await client.chat.completions.create(**attempt)
                return stream, request_started
            except Exception as exc:
                if "stream_options" in attempt and is_unsupported_param_error(exc, "stream_options"):
                    logger.debug("Server rejected stream_options - retrying without it.")
                    attempt = {k: v for k, v in attempt.items() if k != "stream_options"}
                    continue
                if "max_tokens" in attempt and is_unsupported_param_error(exc, "max_tokens"):
                    logger.debug("Server rejected max_tokens - using max_completion_tokens.")
                    attempt = {
                        ("max_completion_tokens" if k == "max_tokens" else k): v
                        for k, v in attempt.items()
                    }
                    continue
                raise

    async def _create_completion(self, params: Dict[str, Any]) -> Tuple[Any, float]:
        client = self._client_or_create()
        attempt = dict(params)

        while True:
            request_started = time.perf_counter()
            try:
                response = await client.chat.completions.create(**attempt)
                return response, request_started
            except Exception as exc:
                if "max_tokens" in attempt and is_unsupported_param_error(exc, "max_tokens"):
                    attempt = {
                        ("max_completion_tokens" if k == "max_tokens" else k): v
                        for k, v in attempt.items()
                    }
                    continue
                raise

    # --------------------------------------------------------------- trials

    async def chat_completion(
        self,
        messages: List[Message],
        max_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs,
    ) -> BenchmarkTrial:
        """Run one completion and return it with full phase timings."""
        await self._ensure_ready()
        options = kwargs
        params = self._build_params(messages, max_tokens, temperature, options)
        prompt = prompt_text(messages)
        request_started = time.perf_counter()

        try:
            if options.get("stream", True):
                stream, request_started = await self._create_stream(params)
                timing = await consume_stream(stream, request_started)
                return self._trial_from_timing(timing, messages, prompt, options)

            response, request_started = await self._create_completion(params)
            ended = time.perf_counter()
            usage = getattr(response, "usage", None)
            content = ""
            choices = getattr(response, "choices", None) or []
            if choices:
                content = getattr(choices[0].message, "content", None) or ""
            timing = StreamTiming(
                request_started=request_started,
                stream_ended=ended,
                first_token_at=None,     # not observable without streaming
                last_token_at=None,
                chunks=1 if content else 0,
                content=content,
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                finish_reason=getattr(choices[0], "finish_reason", None) if choices else None,
            )
            return self._trial_from_timing(timing, messages, prompt, options)

        except Exception as exc:
            return self._error_trial(messages, prompt, request_started, exc)

    def _trial_from_timing(
        self,
        timing: StreamTiming,
        messages: List[Message],
        prompt: str,
        options: Dict[str, Any],
    ) -> BenchmarkTrial:
        """Translate raw stream timings into a :class:`BenchmarkTrial`."""
        token_source = str(options.get("token_source", "auto"))
        output_tokens: Optional[int] = None
        source: Optional[str] = None

        if timing.completion_tokens is not None and token_source != "estimate":
            output_tokens, source = timing.completion_tokens, "usage"
        elif timing.content:
            output_tokens, source = estimate_tokens(timing.content), "estimate"
            if not getattr(self, "_warned_no_usage", False):
                self._warned_no_usage = True
                logger.warning(
                    "the server returned no usage counts - token counts and tokens/s "
                    "are estimated locally (~4 chars per token) and will differ from "
                    "the engine's own figures."
                )

        input_tokens = timing.prompt_tokens
        if input_tokens is None:
            input_tokens = estimate_tokens(prompt)

        response_time = timing.total_time
        if response_time <= 0:
            response_time = None

        return BenchmarkTrial(
            task_id=f"{self.provider_type.value}-trial",
            prompt=messages[-1].content if messages else "",
            response=timing.content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_time=response_time,
            ttft=timing.ttft,
            # Without streaming the decode window cannot be isolated from
            # prefill, so it is left as None instead of pretending it is equal.
            decode_time=timing.decode_time,
            total_time=timing.total_time,
            chunks=timing.chunks,
            finish_reason=timing.finish_reason,
            token_source=source,
            metadata={
                "streamed": bool(options.get("stream", True)),
                "estimated_input_tokens": timing.prompt_tokens is None,
            },
        )

    def _error_trial(
        self,
        messages: List[Message],
        prompt: str,
        request_started: Any,
        exc: Exception,
    ) -> BenchmarkTrial:
        """Record a failed request without polluting the timing statistics."""
        started = request_started if isinstance(request_started, float) else time.perf_counter()
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("Request failed: %s", error)
        return BenchmarkTrial(
            task_id=f"{self.provider_type.value}-error",
            prompt=messages[-1].content if messages else "",
            response="",
            response_time=time.perf_counter() - started,
            error=error,
        )

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "type": self.provider_type.value,
            "model": self.model or "<engine default>",
            "base_url": self.base_url,
            "timeout": self.timeout,
            "api_key_set": bool(self.api_key),
        }




class OpenAIProvider(SDKProvider):
    """OpenAI API provider."""

    provider_type = ProviderType.OPENAI

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        timeout: int = 120,
        organization_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(api_key=api_key, model=model, base_url=base_url, timeout=timeout, **kwargs)
        self.organization_id = organization_id

    def _new_client(self) -> Any:
        from openai import AsyncOpenAI

        kwargs: Dict[str, Any] = {
            "api_key": self.api_key or "",
            "timeout": self.timeout,
            "max_retries": 0,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.organization_id:
            kwargs["organization"] = self.organization_id
        return AsyncOpenAI(**kwargs)


class AzureOpenAIProvider(SDKProvider):
    """Azure OpenAI API provider."""

    provider_type = ProviderType.AZURE

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        timeout: int = 120,
        api_version: str = "2023-05-15",
        deployment: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(api_key=api_key, model=model, base_url=base_url, timeout=timeout, **kwargs)
        self.api_version = api_version
        # Azure routes the model through the deployment name.
        self.deployment = deployment or model

    def _new_client(self) -> Any:
        from openai import AsyncAzureOpenAI

        return AsyncAzureOpenAI(
            api_key=self.api_key or "",
            api_version=self.api_version,
            azure_endpoint=self.base_url or "",
            timeout=self.timeout,
            max_retries=0,
        )

    def _build_params(self, messages, max_tokens, temperature, options) -> Dict[str, Any]:
        params = super()._build_params(messages, max_tokens, temperature, options)
        params["model"] = self.deployment
        return params

    def get_provider_info(self) -> Dict[str, Any]:
        info = super().get_provider_info()
        info["deployment"] = self.deployment
        info["api_version"] = self.api_version
        return info


class LocalProvider(SDKProvider):
    """Local inference server exposing an OpenAI-compatible API.

    Works with vLLM, SGLang, Ollama, LM Studio, LocalAI, TGI and similar
    engines. This is the provider whose numbers are comparable with the
    engine's own ``tokens/s`` counters when run with ``concurrency: 1``.
    """

    provider_type = ProviderType.LOCAL

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "http://localhost:8000/v1",
        timeout: int = 120,
        **kwargs,
    ):
        super().__init__(api_key=api_key or "not-needed", model=model, base_url=base_url, timeout=timeout, **kwargs)
        self._default_model: Optional[str] = None

    async def _ensure_ready(self) -> None:
        """Resolve the served model when the config leaves ``model`` unset."""
        if self.model:
            return
        client = self._client_or_create()
        try:
            listing = await client.models.list()
        except Exception as exc:
            logger.warning(
                "Could not list models from %s (%s) - set 'provider.model' in config.yaml.",
                self.base_url,
                exc,
            )
            return
        served = [getattr(item, "id", None) for item in (getattr(listing, "data", None) or [])]
        served = [m for m in served if m]
        if not served:
            return
        self._default_model = served[0]
        self.model = self._default_model
        logger.info(f"Using model served by the engine: {self.model}")

    def get_provider_info(self) -> Dict[str, Any]:
        info = super().get_provider_info()
        info["served_models"] = [self.model] if self.model else []
        return info



class DummyProvider(BaseProvider):
    """Simulated provider used for smoke tests and offline demos.

    It emits a realistic token stream (first-token latency plus a steady
    decode rate) so the streaming metrics and the report layout can be
    validated without a GPU.
    """

    provider_type = ProviderType.DUMMY

    def __init__(self, api_key: Optional[str] = None, model: str = "dummy-model", **kwargs):
        super().__init__(api_key=api_key or "dummy-key", model=model, **kwargs)

    async def chat_completion(
        self,
        messages: List[Message],
        max_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs,
    ) -> BenchmarkTrial:
        prompt = messages[-1].content if messages else ""
        response = f"Generated response for: {prompt[:100]} " * 10

        words = response.split()[: max(1, min(max_tokens, 100))]
        stream = bool(kwargs.get("stream", True))
        request_started = time.perf_counter()
        await asyncio.sleep(0.05)  # simulated prefill
        first_token_at = time.perf_counter() if stream else None
        for _ in words:
            await asyncio.sleep(0.02)  # simulated ~50 tokens/s decode
        last_token_at = time.perf_counter() if stream else None
        await asyncio.sleep(0.01)  # simulated tail
        ended = time.perf_counter()

        content = " ".join(words)
        timing = StreamTiming(
            request_started=request_started,
            stream_ended=ended,
            first_token_at=first_token_at,
            last_token_at=last_token_at,
            chunks=len(words) if stream else 0,
            content=content,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=len(words),
            finish_reason="stop",
        )
        return BenchmarkTrial(
            task_id="dummy-trial",
            prompt=prompt,
            response=timing.content,
            input_tokens=timing.prompt_tokens,
            output_tokens=timing.completion_tokens,
            response_time=timing.total_time,
            ttft=timing.ttft,
            decode_time=timing.decode_time,
            total_time=timing.total_time,
            chunks=timing.chunks,
            finish_reason=timing.finish_reason,
            token_source="usage",
        )

    def get_provider_info(self) -> Dict[str, Any]:
        return {"type": self.provider_type.value, "model": self.model, "simulated": True}


_PROVIDER_CLASSES = {
    ProviderType.OPENAI: OpenAIProvider,
    ProviderType.AZURE: AzureOpenAIProvider,
    ProviderType.LOCAL: LocalProvider,
    ProviderType.DUMMY: DummyProvider,
}

# Backwards-compatible alias.
AzureProvider = AzureOpenAIProvider


def get_provider(provider_type: Any, **kwargs) -> BaseProvider:
    """Instantiate the provider for ``provider_type``."""
    if isinstance(provider_type, str):
        try:
            provider_type = ProviderType(provider_type.lower())
        except ValueError:
            available = ", ".join(p.value for p in ProviderType)
            raise ValueError(f"Unknown provider type: {provider_type}. Available: {available}")

    provider_class = _PROVIDER_CLASSES.get(provider_type)
    if provider_class is None:
        available = ", ".join(p.value for p in ProviderType)
        raise ValueError(f"Unknown provider type: {provider_type}. Available: {available}")

    kwargs.pop("name", None)
    kwargs.pop("_provenance", None)
    return provider_class(**kwargs)


__all__ = [
    "BaseProvider",
    "SDKProvider",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "AzureProvider",
    "LocalProvider",
    "DummyProvider",
    "StreamTiming",
    "build_messages",
    "consume_stream",
    "get_provider",
    "is_unsupported_param_error",
]

