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
    response: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    response_time: Optional[float] = None  # seconds
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None
    expected_match: Optional[str] = None  # For accuracy comparison
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    def avg_response_time(self) -> Optional[float]:
        """Average response time across trials."""
        times = [t.response_time for t in self.trials if t.response_time is not None]
        if not times:
            return None
        return sum(times) / len(times)

    @property
    def total_tokens(self) -> int:
        """Total tokens processed."""
        total = 0
        for t in self.trials:
            if t.input_tokens:
                total += t.input_tokens
            if t.output_tokens:
                total += t.output_tokens
        return total

    @property
    def avg_tokens_per_second(self) -> Optional[float]:
        """Average tokens per second."""
        token_times = [
            (t.output_tokens, t.response_time)
            for t in self.trials
            if t.output_tokens and t.response_time and t.response_time > 0
        ]
        if not token_times:
            return None
        total_tokens = sum(t[0] for t in token_times)
        total_time = sum(t[1] for t in token_times)
        if total_time == 0:
            return None
        return total_tokens / total_time

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
            "provider": self.provider.value,
            "benchmark_type": self.benchmark_type.value,
            "duration": self.duration,
            "num_trials": len(self.trials),
            "avg_response_time": self.avg_response_time,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_second": self.avg_tokens_per_second,
            "trials": [
                {
                    "task_id": t.task_id,
                    "prompt": t.prompt,
                    "response": t.response,
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "response_time": t.response_time,
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
    temperature: float = 0.7
    timeout: int = 120
    num_trials: int = 10
    num_questions: int = 20
    num_tasks: int = 10
    output_format: OutputFormat = OutputFormat.TEXT
    output_file: Optional[str] = None
    verbose: bool = False

    def __post_init__(self):
        if not self.provider:
            self.provider = {
                "name": ProviderType.DUMMY.value,
                "model": "dummy-model",
            }