"""Data models for the LLM Benchmark Tool."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import time


class ProviderType(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    AZURE = "azure"
    LOCAL = "local"
    DUMMY = "dummy"


class BenchmarkType(str, Enum):
    """Available benchmark types."""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    MULTILINGUAL = "multilingual"


class OutputFormat(str, Enum):
    """Output report formats."""
    JSON = "json"
    HTML = "html"
    TEXT = "text"


def percentile(sorted_values: List[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile of an already sorted list."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * max(0.0, min(1.0, q))
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


_CJK_RANGES = (
    (0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF), (0xF900, 0xFAFF), (0xFF00, 0xFF60),
)


def estimate_tokens(text: str) -> int:
    """Rough token count used when the server does not report ``usage``.

    Most sub-word tokenizers average ~4 characters per token for Latin text,
    while CJK glyphs are close to one token per character. This is only a
    fallback: the report labels such numbers as ``estimated``.
    """
    if not text:
        return 0
    cjk = 0
    other_chars = 0
    for char in text:
        code = ord(char)
        if any(lo <= code <= hi for lo, hi in _CJK_RANGES):
            cjk += 1
        else:
            other_chars += 1
    words = max(0, len(text.split()) - cjk)
    return max(1, cjk + int(round(other_chars / 4.0)) + int(round(words * 0.15)))


def redact_secret(secret: Optional[str]) -> str:
    """Printable, non-reversible representation of a secret."""
    if not secret:
        return "<unset>"
    text = str(secret)
    if len(text) <= 10:
        return f"{text[:2]}***"
    return f"{text[:6]}...{text[-4:]} ({len(text)} chars)"


@dataclass
class Message:
    """A chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    id: str
    benchmark_type: BenchmarkType
    prompt: str
    expected_answer: Optional[str] = None
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkTrial:
    """A single trial within a benchmark."""
    task_id: str
    prompt: str
    response: str = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    response_time: Optional[float] = None  # seconds, end-to-end (wall clock)
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None
    expected_match: Optional[str] = None  # For accuracy comparison
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- Streaming / decode-phase timing (seconds, measured with perf_counter) ---
    ttft: Optional[float] = None            # time to first generated token (queue + prefill + net)
    decode_time: Optional[float] = None     # last token - first token (pure generation)
    total_time: Optional[float] = None      # request start -> stream closed
    chunks: Optional[int] = None            # SSE chunks carrying generated content
    finish_reason: Optional[str] = None
    token_source: Optional[str] = None      # "usage" (server reported) | "estimated"
    warmup: bool = False                    # True for discarded warm-up requests
    benchmark_type: Optional[str] = None    # owning benchmark (set by the runner)
    attempt: int = 1                        # >1 when the runner retried the request

    @property
    def ok(self) -> bool:
        """A trial only counts when it produced tokens without an error."""
        return self.error is None and (self.output_tokens or 0) > 0

    @property
    def decode_tokens_per_second(self) -> Optional[float]:
        """Pure generation speed: decode-phase tokens / decode-phase time."""
        if not self.output_tokens or not self.decode_time or self.decode_time <= 0:
            return None
        generated = max(1, self.output_tokens - 1)
        return generated / self.decode_time

    @property
    def e2e_tokens_per_second(self) -> Optional[float]:
        """End-to-end speed including connect, queueing and prefill."""
        if not self.output_tokens or not self.response_time or self.response_time <= 0:
            return None
        return self.output_tokens / self.response_time

    @property
    def mean_itl(self) -> Optional[float]:
        """Mean inter-token latency (seconds per generated token)."""
        if not self.decode_time or not self.output_tokens or self.output_tokens < 2:
            return None
        return self.decode_time / (self.output_tokens - 1)

    @property
    def prefill_tokens_per_second(self) -> Optional[float]:
        """Approximate prefill rate (input tokens / TTFT; includes queue + network)."""
        if not self.input_tokens or not self.ttft or self.ttft <= 0:
            return None
        return self.input_tokens / self.ttft


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model_name: str
    provider: ProviderType
    benchmark_type: BenchmarkType
    trials: List[BenchmarkTrial] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Total benchmark duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def successful_trials(self) -> List[BenchmarkTrial]:
        """Trials that actually produced tokens (errors/empty are excluded)."""
        return [t for t in self.trials if t.ok]

    @property
    def failed_trials(self) -> List[BenchmarkTrial]:
        """Trials that errored or produced nothing."""
        return [t for t in self.trials if not t.ok]

    @property
    def avg_response_time(self) -> Optional[float]:
        """Average end-to-end response time across successful trials.

        Failed trials used to be recorded with ``response_time=0.0`` and were
        averaged in, which pulled the mean down and made runs look faster.
        """
        times = [t.response_time for t in self.successful_trials if t.response_time]
        if not times:
            return None
        return sum(times) / len(times)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens or 0 for t in self.trials)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens or 0 for t in self.trials)

    @property
    def total_tokens(self) -> int:
        """Total tokens processed."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_tokens_per_second(self) -> Optional[float]:
        """Decode-phase tokens per second (the number comparable to engine logs).

        Computed as sum(generated tokens) / sum(decode time) over the trials
        that have streaming data. Falls back to the end-to-end rate when the
        provider could not stream (in which case prefill is included and the
        number is labelled accordingly by the report).
        """
        pairs = [
            (max(1, t.output_tokens - 1), t.decode_time)
            for t in self.successful_trials
            if t.output_tokens and t.decode_time and t.decode_time > 0
        ]
        if pairs:
            total_tokens = sum(p[0] for p in pairs)
            total_time = sum(p[1] for p in pairs)
            return total_tokens / total_time if total_time > 0 else None
        return self.avg_e2e_tokens_per_second

    @property
    def avg_e2e_tokens_per_second(self) -> Optional[float]:
        """End-to-end tokens per second (prefill + queue + network included)."""
        pairs = [
            (t.output_tokens, t.response_time)
            for t in self.successful_trials
            if t.output_tokens and t.response_time and t.response_time > 0
        ]
        if not pairs:
            return None
        total_tokens = sum(p[0] for p in pairs)
        total_time = sum(p[1] for p in pairs)
        return total_tokens / total_time if total_time > 0 else None

    @property
    def aggregate_throughput(self) -> Optional[float]:
        """Output tokens per second of wall clock for the whole benchmark.

        Only meaningful with concurrency > 1: it is the system throughput,
        while ``avg_tokens_per_second`` stays the per-stream decode speed.
        """
        duration = self.duration
        tokens = self.total_output_tokens
        if not duration or duration <= 0 or tokens <= 0:
            return None
        return tokens / duration

    @property
    def streaming_measured(self) -> bool:
        """True when at least one trial has real decode timing."""
        return any(t.decode_time for t in self.successful_trials)

    def _values(self, attr: str) -> List[float]:
        return [
            getattr(t, attr)
            for t in self.successful_trials
            if getattr(t, attr) is not None
        ]

    def _stats(self, attr: str) -> Dict[str, Optional[float]]:
        """mean / median / p95 / min / max of a per-trial metric."""
        values = sorted(self._values(attr))
        if not values:
            return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
        return {
            "mean": sum(values) / len(values),
            "median": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "min": values[0],
            "max": values[-1],
        }

    @property
    def ttft_stats(self) -> Dict[str, Optional[float]]:
        return self._stats("ttft")

    @property
    def decode_tps_stats(self) -> Dict[str, Optional[float]]:
        return self._stats("decode_tokens_per_second")

    @property
    def e2e_tps_stats(self) -> Dict[str, Optional[float]]:
        return self._stats("e2e_tokens_per_second")

    @property
    def itl_stats(self) -> Dict[str, Optional[float]]:
        return self._stats("mean_itl")

    @property
    def measured_token_source(self) -> str:
        """How the recorded token counts were obtained: usage / estimate / mixed.

        Reported so a benchmark whose token counts were guessed locally cannot be
        read as though the numbers came from the server.
        """
        counts = {"usage": 0, "estimate": 0}
        for trial in self.successful_trials:
            source = str(getattr(trial, "token_source", "") or "").lower()
            if source == "usage":
                counts["usage"] += 1
            elif source.startswith("estimat"):
                counts["estimate"] += 1
        if counts["usage"] and counts["estimate"]:
            return "mixed"
        if counts["usage"]:
            return "usage"
        if counts["estimate"]:
            return "estimate"
        return "unknown"

    @property
    def avg_ttft(self) -> Optional[float]:
        return self.ttft_stats["mean"]

    @property
    def avg_decode_tokens_per_second(self) -> Optional[float]:
        return self.decode_tps_stats["mean"]

    @property
    def avg_mean_itl(self) -> Optional[float]:
        return self.itl_stats["mean"]

    def accuracy_score(self) -> Optional[float]:
        """Calculate accuracy score using keyword matching if expected answers are available.
        
        Uses keyword overlap scoring: a response is considered correct if it contains
        sufficient keyword overlap with the expected answer.
        """
        trials_with_expected = [t for t in self.trials if t.expected_match]
        if not trials_with_expected:
            return None
        
        correct = 0
        for t in trials_with_expected:
            if self._is_answer_correct(t.response, t.expected_match):
                correct += 1
        
        return correct / len(trials_with_expected)
    
    def _is_answer_correct(self, response: str, expected: str) -> bool:
        """Check if response matches expected answer using keyword-based matching.
        
        Scoring:
        - Exact match (case-insensitive): 100%
        - Response contains expected answer as substring: 100%
        - Expected answer contains response as substring: 100%
        - Keyword overlap >= 50%: partial match (counted as correct for questions)
        """
        resp = response.strip().lower()
        exp = expected.strip().lower()
        
        # Exact match
        if resp == exp:
            return True
        
        # Response contains expected answer
        if exp in resp:
            return True
        
        # Expected answer contains response (for short answers)
        if len(exp) >= len(resp) * 0.5 and resp in exp:
            return True
        
        # Keyword overlap scoring
        # Extract meaningful words (excluding common stop words)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
            'before', 'after', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
            'both', 'either', 'neither', 'each', 'every', 'all', 'any', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
            'than', 'too', 'very', 'just', 'that', 'this', 'it', 'its', 'i', 'me',
            'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'when', 'where',
            'why', 'how', 'if', 'then', 'else', 'all', 'ok', 'yes', 'no'
        }
        
        # Get meaningful words
        resp_words = set(w.lower() for w in resp.split() if w.lower() not in stop_words and len(w) > 1)
        exp_words = set(w.lower() for w in exp.split() if w.lower() not in stop_words and len(w) > 1)
        
        # If no meaningful words found, can't score
        if not exp_words:
            return False
        
        # Calculate keyword overlap
        if resp_words:
            overlap = len(resp_words & exp_words) / len(exp_words)
            # 50% or more keyword overlap counts as correct
            return overlap >= 0.5
        else:
            return False

    @property
    def has_expected_answers(self) -> bool:
        """Check if this benchmark has any trials with expected answers."""
        return any(t.expected_match for t in self.trials)
    
    @property
    def applicable_accuracy(self) -> Optional[float]:
        """Return accuracy score only for benchmark types that support it.
        
        Returns None for benchmark types that don't use accuracy scoring
        (PERFORMANCE, CODE_GENERATION, MULTILINGUAL).
        """
        if self.benchmark_type not in (BenchmarkType.ACCURACY, BenchmarkType.REASONING):
            return None
        return self.accuracy_score()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = {
            "model_name": self.model_name,
            "provider": self.provider.value if hasattr(self.provider, "value") else self.provider,
            "benchmark_type": self.benchmark_type.value,
            "duration": self.duration,
            "num_trials": len(self.trials),
            "num_failed_trials": len(self.failed_trials),
            "measurement_mode": "streaming" if self.streaming_measured else "non-streaming",
            "token_source": self.measured_token_source,
            "measurement_settings": self.config,
            "avg_response_time": self.avg_response_time,
            "avg_ttft": self.avg_ttft,
            "avg_decode_tokens_per_second": self.avg_decode_tokens_per_second,
            "avg_e2e_tokens_per_second": self.avg_e2e_tokens_per_second,
            "avg_mean_itl": self.avg_mean_itl,
            "aggregate_throughput": self.aggregate_throughput,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            # Kept for backwards compatibility: decode rate when available.
            "avg_tokens_per_second": self.avg_tokens_per_second,
            "ttft_stats": self.ttft_stats,
            "decode_tokens_per_second_stats": self.decode_tps_stats,
            "e2e_tokens_per_second_stats": self.e2e_tps_stats,
            "inter_token_latency_stats": self.itl_stats,
            "trials": [
                {
                    "task_id": t.task_id,
                    "prompt": t.prompt,
                    "response": t.response,
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "token_source": t.token_source,
                    "response_time": t.response_time,
                    "ttft": t.ttft,
                    "decode_time": t.decode_time,
                    "decode_tokens_per_second": t.decode_tokens_per_second,
                    "e2e_tokens_per_second": t.e2e_tokens_per_second,
                    "mean_itl": t.mean_itl,
                    "finish_reason": t.finish_reason,
                    "error": t.error,
                }
                for t in self.trials
            ],
        }
        
        # Only include accuracy for applicable benchmark types
        acc = self.applicable_accuracy
        if acc is not None:
            data["accuracy_score"] = acc
        
        return data


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    provider: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 500
    temperature: float = 0.0
    timeout: int = 120
    num_trials: int = 10
    num_questions: int = 20
    num_tasks: int = 10
    output_format: OutputFormat = OutputFormat.TEXT
    output_file: Optional[str] = None
    verbose: bool = False

    # --- Measurement methodology (see README "Measuring tokens/second") ---
    # concurrency=1 keeps the run single-stream, which is what engine-side
    # token/s counters measure. Higher values measure aggregate throughput and
    # will show a lower per-stream decode rate.
    concurrency: int = 1
    warmup_requests: int = 1
    stream: bool = True
    include_usage: bool = True
    token_source: str = "auto"      # auto | usage | estimate
    seed: Optional[int] = 1234
    retries: int = 2                # runner-level retries (SDK retries disabled)
    cooldown: float = 0.0           # optional pause between trials
    prompt_profile: str = "heavy"   # performance benchmark prompt profile
    ignore_eos: bool = False        # fixed output length (vLLM/SGLang servers)
    extra_body: Dict[str, Any] = field(default_factory=dict)
    per_benchmark: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    max_tokens_explicit: bool = False

    def __post_init__(self):
        if not self.provider:
            self.provider = {
                "name": ProviderType.DUMMY.value,
                "model": "dummy-model",
            }

    def settings_for(self, benchmark_type: "BenchmarkType") -> Dict[str, Any]:
        """Global settings overridden by the per-benchmark YAML section."""
        settings: Dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
            "include_usage": self.include_usage,
            "token_source": self.token_source,
            "seed": self.seed,
            "ignore_eos": self.ignore_eos,
            "extra_body": dict(self.extra_body or {}),
            "max_tokens_explicit": self.max_tokens_explicit,
        }
        for key, value in (self.per_benchmark.get(benchmark_type.value) or {}).items():
            if key in ("num_trials", "num_questions", "num_tasks", "prompt_profile"):
                continue
            settings[key] = value
        return settings

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the configuration (secrets redacted)."""
        provider = {
            k: (redact_secret(v) if ("key" in k.lower() or "token" in k.lower()) else v)
            for k, v in (self.provider or {}).items()
            if not str(k).startswith("_")
        }
        return {
            "provider": provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "num_trials": self.num_trials,
            "num_questions": self.num_questions,
            "num_tasks": self.num_tasks,
            "concurrency": self.concurrency,
            "warmup_requests": self.warmup_requests,
            "stream": self.stream,
            "include_usage": self.include_usage,
            "token_source": self.token_source,
            "seed": self.seed,
            "retries": self.retries,
            "cooldown": self.cooldown,
            "prompt_profile": self.prompt_profile,
            "ignore_eos": self.ignore_eos,
            "extra_body": self.extra_body,
            "per_benchmark": self.per_benchmark,
        }
