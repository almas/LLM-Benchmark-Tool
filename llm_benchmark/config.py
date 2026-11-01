"""Configuration loading for the LLM Benchmark Tool.

The tool is configured from three sources, merged with a strict precedence
order (highest wins):

    1. command line flags
    2. environment variables  (LLM_BENCH_*, OPENAI_API_KEY, ...)
    3. the YAML config file   (config.yaml)
    4. built-in defaults

The config file is searched for in this order:

    $LLM_BENCH_CONFIG
    ./config.yaml, ./config.yml
    ./llm_benchmark/config.yaml            (the file shipped with the tool)
    ~/.config/llm-benchmark/config.yaml

Values inside the YAML file may reference environment variables with the
``${VAR}`` or ``${VAR:default}`` syntax, which is the recommended way to keep
secrets out of the file itself.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from llm_benchmark.logger import logger

ENV_PREFIX = "LLM_BENCH_"

# Built-in fallbacks used when neither CLI, env nor config.yaml provide a value.
DEFAULTS: Dict[str, Any] = {
    "provider.name": "dummy",
    "provider.base_url": None,
    "provider.model": None,
    "provider.timeout": 120,
    "provider.api_version": "2024-02-01",
    "provider.azure_endpoint": None,
    "measurement.num_trials": 10,
    "measurement.num_questions": 20,
    "measurement.num_tasks": 10,
    "measurement.max_tokens": 500,
    "measurement.temperature": 0.0,
    "measurement.concurrency": 1,
    "measurement.warmup_requests": 1,
    "measurement.stream": True,
    "measurement.include_usage": True,
    "measurement.token_source": "auto",
    "measurement.seed": 1234,
    "measurement.retries": 2,
    "measurement.cooldown": 0.0,
    "measurement.prompt_profile": "heavy",
    "measurement.ignore_eos": False,
    "measurement.extra_body": {},
}

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

# Keys that hold secrets and must never be printed in clear text.
SECRET_KEYS = ("api_key", "api-key", "apikey", "token", "secret")


def redact(secret: Optional[str]) -> str:
    """Return a printable, non-reversible representation of a secret."""
    if not secret:
        return "<unset>"
    if len(secret) <= 10:
        return f"{secret[:2]}***"
    return f"{secret[:6]}...{secret[-4:]} ({len(secret)} chars)"


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:default}`` placeholders."""
    if isinstance(value, str):
        def _replace(match):
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default if default is not None else "")

        return _ENV_PATTERN.sub(_replace, value).strip()
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def default_config_path() -> Path:
    """Path of the ``config.yaml`` shipped inside the package."""
    return Path(__file__).resolve().parent / "config.yaml"


def candidate_config_paths(explicit: Optional[str] = None) -> List[Path]:
    """All config locations that will be probed, in precedence order."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get(f"{ENV_PREFIX}CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            Path.cwd() / "config.yaml",
            Path.cwd() / "config.yml",
            default_config_path(),
            Path.home() / ".config" / "llm-benchmark" / "config.yaml",
        ]
    )
    seen: set = set()
    unique: List[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def find_config_file(explicit: Optional[str] = None) -> Optional[Path]:
    for path in candidate_config_paths(explicit):
        if path.is_file():
            return path
    if explicit:
        raise FileNotFoundError(
            f"Config file not found: {explicit} (check --config / LLM_BENCH_CONFIG)"
        )
    return None


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping, tolerating an empty file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "PyYAML is required to read config.yaml - run: pip install PyYAML"
        ) from exc

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Top level of {path} must be a mapping, got {type(data).__name__}"
        )
    return expand_env(data)


def load_config(explicit: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Return ``(raw_config, path)`` for the first config file that exists."""
    path = find_config_file(explicit)
    if path is None:
        logger.debug("No config.yaml found - using defaults/env only.")
        return {}, None
    config = load_yaml(path)
    logger.info(f"Loaded config file: {path}")
    return config, path


def _dig(data: Dict[str, Any], *keys: str) -> Any:
    """Safe nested lookup: ``_dig(cfg, "provider", "api_key")``."""
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _cast(value: Any, caster: Any) -> Any:
    if value is None or value == "":
        return None
    if caster is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(value)
    try:
        return caster(value)
    except (TypeError, ValueError):
        logger.warning(
            f"Could not parse value {value!r} as {getattr(caster, '__name__', caster)} - ignoring."
        )
        return None


class Resolved:
    """A resolved setting together with the source it came from."""

    __slots__ = ("value", "source")

    def __init__(self, value: Any, source: str):
        self.value = value
        self.source = source


def resolve(
    key: str,
    cli_value: Any,
    config: Dict[str, Any],
    yaml_keys: Iterable[str],
    env_names: Iterable[str] = (),
    caster: Any = None,
) -> Resolved:
    """Resolve one setting using CLI > env > YAML > default precedence."""
    if cli_value is not None and cli_value != "":
        value = caster(cli_value) if caster else cli_value
        return Resolved(value, "cli")

    for name in env_names:
        raw = _env(name)
        if raw is not None:
            value = _cast(raw, caster) if caster else raw
            if value is not None:
                return Resolved(value, f"env:{name}")

    for path in yaml_keys:
        raw = _dig(config, *path.split("."))
        if raw is not None and raw != "":
            value = _cast(raw, caster) if caster else raw
            if value is not None:
                return Resolved(value, f"config.yaml:{path}")

    return Resolved(DEFAULTS.get(key), "default")



def resolve_provider(config: Dict[str, Any], cli: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CLI/env/YAML into a single provider configuration dict."""
    name = resolve(
        "provider.name",
        cli.get("provider"),
        config,
        ("provider.name", "provider.type", "name"),
        (f"{ENV_PREFIX}PROVIDER",),
    ).value

    api_key = resolve(
        "provider.api_key",
        cli.get("api_key"),
        config,
        ("provider.api_key", "api_key"),
        (f"{ENV_PREFIX}API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"),
    )

    provider_cfg: Dict[str, Any] = {"name": name}

    model = resolve(
        "provider.model",
        cli.get("model"),
        config,
        ("provider.model", "model"),
        (f"{ENV_PREFIX}MODEL",),
    )
    if model.value:
        provider_cfg["model"] = model.value

    base_url = resolve(
        "provider.base_url",
        cli.get("base_url"),
        config,
        ("provider.base_url", "base_url"),
        (f"{ENV_PREFIX}BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"),
    )
    if base_url.value:
        provider_cfg["base_url"] = base_url.value

    azure_endpoint = resolve(
        "provider.azure_endpoint",
        cli.get("azure_endpoint"),
        config,
        ("provider.azure_endpoint", "azure_endpoint"),
        (f"{ENV_PREFIX}AZURE_ENDPOINT", "AZURE_OPENAI_ENDPOINT"),
    )
    if azure_endpoint.value:
        provider_cfg["azure_endpoint"] = azure_endpoint.value

    api_version = resolve(
        "provider.api_version",
        cli.get("api_version"),
        config,
        ("provider.api_version", "api_version"),
        (f"{ENV_PREFIX}API_VERSION", "AZURE_OPENAI_API_VERSION"),
    )
    if api_version.value:
        provider_cfg["api_version"] = api_version.value

    timeout = resolve(
        "provider.timeout",
        cli.get("timeout"),
        config,
        ("provider.timeout", "timeout"),
        (f"{ENV_PREFIX}TIMEOUT",),
        caster=int,
    )
    provider_cfg["timeout"] = timeout.value

    if api_key.value:
        provider_cfg["api_key"] = api_key.value

    provider_cfg["_provenance"] = {
        "name": "cli" if cli.get("provider") else "config.yaml/default",
        "model": model.source,
        "base_url": base_url.source,
        "timeout": timeout.source,
        "api_key": api_key.source if api_key.value else "unset",
    }

    _validate_provider(provider_cfg)
    return provider_cfg


def _validate_provider(provider_cfg: Dict[str, Any]) -> None:
    """Warn about provider settings that will almost certainly fail at runtime."""
    from llm_benchmark.models import ProviderType

    name = provider_cfg.get("name")
    api_key = provider_cfg.get("api_key")

    if name == ProviderType.OPENAI.value and not api_key:
        logger.warning(
            "Provider 'openai' selected but no api_key was found "
            "(config.yaml provider.api_key, --api-key or OPENAI_API_KEY)."
        )
    if name == ProviderType.AZURE.value and not provider_cfg.get("azure_endpoint"):
        logger.warning("Provider 'azure' selected but no azure_endpoint is configured.")
    if name == ProviderType.LOCAL.value and not provider_cfg.get("base_url"):
        logger.warning(
            "Provider 'local' selected but no base_url is configured - "
            "set provider.base_url in config.yaml or pass --base-url."
        )
    if api_key:
        logger.info(
            "API key loaded from %s: %s",
            provider_cfg["_provenance"]["api_key"],
            redact(api_key),
        )


def summarise(provider_cfg: Dict[str, Any], settings: Dict[str, Any]) -> str:
    """Human readable summary of the effective configuration."""
    lines = [
        f"  provider      : {provider_cfg.get('name')}",
        f"  base_url      : {provider_cfg.get('base_url') or '<default>'}",
        f"  model         : {provider_cfg.get('model') or '<engine default>'}",
        f"  api_key       : {redact(provider_cfg.get('api_key'))}",
        f"  timeout       : {provider_cfg.get('timeout')}s",
        f"  concurrency   : {settings.get('concurrency')}",
        f"  warmup        : {settings.get('warmup_requests')} request(s)",
        f"  streaming     : {settings.get('stream')} "
        f"(usage={settings.get('include_usage')}, tokens={settings.get('token_source')})",
        f"  max_tokens    : {settings.get('max_tokens')} "
        f"@ temperature={settings.get('temperature')}",
    ]
    return "\n".join(lines)



# (settings key, CLI attr, config.yaml paths, env vars, caster)
MEASUREMENT_SPEC = [
    ("num_trials", "num_trials",
     ("num_trials", "benchmarks.num_trials", "measurement.num_trials",
      "benchmarks.performance.num_trials"), ("LLM_BENCH_NUM_TRIALS",), int),
    ("num_questions", "num_questions",
     ("num_questions", "benchmarks.num_questions", "measurement.num_questions",
      "benchmarks.accuracy.num_questions"), ("LLM_BENCH_NUM_QUESTIONS",), int),
    ("num_tasks", "num_tasks",
     ("num_tasks", "benchmarks.num_tasks", "measurement.num_tasks"),
     ("LLM_BENCH_NUM_TASKS",), int),
    ("max_tokens", "max_tokens",
     ("max_tokens", "benchmarks.max_tokens", "measurement.max_tokens"),
     ("LLM_BENCH_MAX_TOKENS",), int),
    ("temperature", "temperature",
     ("temperature", "benchmarks.temperature", "measurement.temperature"),
     ("LLM_BENCH_TEMPERATURE",), float),
    ("concurrency", "concurrency",
     ("concurrency", "measurement.concurrency", "benchmarks.concurrency"),
     ("LLM_BENCH_CONCURRENCY",), int),
    ("warmup_requests", "warmup",
     ("warmup_requests", "warmup", "measurement.warmup_requests"),
     ("LLM_BENCH_WARMUP",), int),
    ("stream", "stream", ("stream", "measurement.stream"), ("LLM_BENCH_STREAM",), bool),
    ("include_usage", "include_usage",
     ("include_usage", "measurement.include_usage"), ("LLM_BENCH_INCLUDE_USAGE",), bool),
    ("token_source", "token_source",
     ("token_source", "measurement.token_source"), ("LLM_BENCH_TOKEN_SOURCE",), str),
    ("seed", "seed", ("seed", "measurement.seed"), ("LLM_BENCH_SEED",), int),
    ("retries", "retries",
     ("retries", "measurement.retries", "benchmarks.retries"), ("LLM_BENCH_RETRIES",), int),
    ("cooldown", "cooldown", ("cooldown", "measurement.cooldown"),
     ("LLM_BENCH_COOLDOWN",), float),
    ("prompt_profile", "prompt_profile",
     ("prompt_profile", "benchmarks.performance.prompt_profile",
      "measurement.prompt_profile"), ("LLM_BENCH_PROMPT_PROFILE",), str),
    ("ignore_eos", "ignore_eos",
     ("ignore_eos", "benchmarks.performance.ignore_eos", "measurement.ignore_eos"),
     ("LLM_BENCH_IGNORE_EOS",), bool),
    ("extra_body", "extra_body", ("extra_body", "measurement.extra_body"), (), None),
]

# Where a max_tokens value counts as an explicit user choice (it then wins over
# the per-task metadata that benchmark definitions may carry).
EXPLICIT_MAX_TOKENS_SOURCES = (
    "cli",
    "config.yaml:max_tokens",
    "config.yaml:benchmarks.max_tokens",
    "config.yaml:measurement.max_tokens",
    "config.yaml:benchmarks.performance.max_tokens",
    "env:LLM_BENCH_MAX_TOKENS",
)


def resolve_measurement(config: Dict[str, Any], cli: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve measurement/execution settings using the documented precedence."""
    settings: Dict[str, Any] = {}
    for key, cli_key, yaml_keys, env_names, caster in MEASUREMENT_SPEC:
        # DEFAULTS are keyed by "measurement.<key>", so resolve against that name.
        resolved = resolve(
            f"measurement.{key}", cli.get(cli_key), config, yaml_keys, env_names, caster
        )
        settings[key] = resolved.value
        settings[f"_{key}_source"] = resolved.source

    settings["max_tokens_explicit"] = (
        settings["_max_tokens_source"] in EXPLICIT_MAX_TOKENS_SOURCES
    )

    per_benchmark: Dict[str, Dict[str, Any]] = {}
    yaml_benchmarks = config.get("benchmarks") or {}
    if isinstance(yaml_benchmarks, dict):
        for bench_name, values in yaml_benchmarks.items():
            if isinstance(values, dict):
                per_benchmark[str(bench_name)] = dict(values)

    # A count given on the command line (or via LLM_BENCH_*) must beat a
    # per-benchmark value from config.yaml, otherwise `-n 5` silently does
    # nothing when benchmarks.performance.num_trials is set.
    for bench_name, count_key in (
        ("performance", "num_trials"),
        ("accuracy", "num_questions"),
        ("reasoning", "num_tasks"),
        ("code_generation", "num_tasks"),
        ("multilingual", "num_tasks"),
    ):
        if settings.get(f"_{count_key}_source") in ("cli",) and settings.get(count_key):
            per_benchmark.setdefault(bench_name, {})[count_key] = settings[count_key]

    settings["per_benchmark"] = per_benchmark

    _validate_measurement(settings)
    return settings


def _validate_measurement(settings: Dict[str, Any]) -> None:
    concurrency = settings.get("concurrency") or 1
    if concurrency > 1:
        logger.warning(
            "concurrency=%s: %s requests share one engine, so the per-stream "
            "decode rate will be far below the single-user rate shown by the "
            "engine dashboard. Use concurrency 1 to measure per-token speed.",
            concurrency, concurrency,
        )
    if not settings.get("stream", True):
        logger.warning(
            "streaming disabled: time-to-first-token and decode speed cannot be "
            "separated from prefill, tokens/s becomes an end-to-end average."
        )
    if settings.get("token_source") not in ("auto", "usage", "estimate"):
        logger.warning(
            "Unknown token_source %r - falling back to 'auto'.",
            settings.get("token_source"),
        )
        settings["token_source"] = "auto"
    if int(settings.get("warmup_requests") or 0) <= 0:
        logger.warning(
            "warmup_requests=0: the first trial includes cold-start costs "
            "(KV-cache allocation, paging, kernel warm-up) and will look slow."
        )

