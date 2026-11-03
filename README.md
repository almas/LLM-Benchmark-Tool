# LLM Benchmark Tool

An easy python benchmark tool for evaluating Large Language Models (LLMs) across multiple dimensions. This tool is automatically written by the Qwen3.6-35B local LLM with the Cline.

## Features

- **Performance Metrics**: Measure response time, tokens per second, and throughput
- **Accuracy Tests**: Evaluate model accuracy on question answering tasks
- **Code Generation**: Benchmark code generation capabilities
- **Reasoning Tasks**: Test logical reasoning and problem-solving abilities
- **Multilingual Support**: Evaluate performance across multiple languages
- **Custom Benchmarks**: Create and run custom benchmark tests
- **Detailed Reports**: Generate reports with comparisons

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python main.py --provider openai --model gpt-3.5-turlo --benchmarks all
```

### Available Providers

- `openai` - OpenAI APIs (GPT-3.5, GPT-4, etc.)
- `azure` - Azure OpenAI Service
- `local` - Local models via OpenAI-compatible APIs
- `dummy` - Dummy provider for testing (returns mock responses)

### Running Benchmarks

```bash
# Run all benchmarks
python main.py --provider local --model test-model --base-url http://127.0.0.1:8000/v1 --benchmarks all

# Use default model
python main.py --provider local --base-url http://127.0.0.1:8000/v1 --benchmarks all

# Run specific benchmarks
python main.py --provider local --model test-model --base-url http://127.0.0.1:8000/v1 --benchmarks performance --num-trials 3
## For example:
python main.py --provider local --model gemma-4-31B-it-AWQ-8bit --base-url http://127.0.0.1:8000/v1 --benchmarks performance --num-trials 5

# Run with custom configuration
python main.py --provider local --model test-model --base-url http://127.0.0.1:8000/v1 --benchmarks accuracy --num-questions 20
```

### Generating Reports

```bash
# Generate JSON report
python main.py --provider local --model test-model --base-url http://127.0.0.1:8000/v1 --benchmarks all --output report.json

# Generate HTML report
python main.py --provider local --model test-model --base-url http://127.0.0.1:8000/v1 --benchmarks all --output report.html --format html
```

## Configuration

Settings are merged from these levels, highest precedence first:

```
CLI flags  >  LLM_BENCH_*  >  config.yaml  >  generic SDK env vars  >  defaults
                               (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_API_BASE, AZURE_OPENAI_*)
```

Generic SDK variables are only consulted when `config.yaml` does not carry that setting, so an
`OPENAI_API_BASE` exported for some other program can no longer silently move a benchmark to a
different endpoint. The summary printed before the first request shows where every provider
setting came from, e.g. `base_url : http://127.0.0.1:8001/v1  (from config.yaml:provider.base_url)`.

`config.yaml` is searched in this order: `--config`, `$LLM_BENCH_CONFIG`, `./config.yaml`,
`./config.yml`, `llm_benchmark/config.yaml`, `~/.config/llm-benchmark/config.yaml`.
Copy `llm_benchmark/config.yaml.template` to one of those paths. Inside the file,
`${VAR}` and `${VAR:default}` are expanded from the environment, so API keys can stay
out of the file. The effective settings (with the API key redacted) are printed before
the first request.

```yaml
provider:
  name: local                 # openai | azure | local | dummy
  model: Qwen/Qwen2.5-7B-Instruct
  base_url: http://127.0.0.1:8000/v1
  api_key: ${OPENAI_API_KEY}  # optional for local servers
  timeout: 120

measurement:
  concurrency: 1              # 1 = single-stream decode speed
  warmup_requests: 1          # unmeasured request before timing starts
  stream: true                # separates TTFT/prefill from decode
  include_usage: true         # ask the server for exact token counts
  token_source: auto          # auto | usage | estimate
  seed: 1234
  retries: 2                  # runner-level retries; SDK retries stay off
  cooldown: 0.0
  prompt_profile: heavy       # short | medium | heavy
  ignore_eos: false           # force exactly max_tokens (vLLM/SGLang)

benchmarks:
  num_trials: 20
  num_questions: 20
  num_tasks: 10
  max_tokens: 512
  temperature: 0.0
  performance:
    num_trials: 20
    max_tokens: 512
```

## How performance is measured

The number most people want from a local engine is the **single-stream decode rate** -
the tokens/second a user sees while reading the answer. That is what this tool reports
by default, and it is measured the same way vLLM/SGLang report it:

- Requests are **streamed**, so prefill and generation are timed separately:
  - `TTFT` - request sent -> first generated token (queueing + prefill + network).
  - `Decode tokens/sec` - generated tokens / (last token - first token). Prefill excluded.
  - `End-to-end tok/sec` - tokens / whole request wall clock; always lower.
  - `Inter-token latency` - mean gap between streamed tokens.
  - `Aggregate output` - output tokens / benchmark wall clock (system throughput).
- `concurrency: 1` keeps one request in flight. Raising it multiplies *aggregate*
  throughput while the per-stream decode rate drops - both numbers are reported so the
  difference is visible instead of accidental.
- `warmup_requests` run before timing so cold-start costs (KV-cache allocation, CUDA
  graph capture, paging) do not drag the first trial down.
- The SDK's own retries are disabled; only runner-level retries happen, and retried or
  failed requests never enter the statistics (`Trials: 20 (19 usable, 1 failed)`).
- Output token counts come from the server's `usage` field when available
  (`stream_options.include_usage`); otherwise they are estimated and labelled as such
  in the report.
- `ignore_eos: true` makes vLLM/SGLang generate exactly `max_tokens` tokens so every
  trial is comparable - the standard trick in engine benchmarks.
- `prompt_profile` picks the workload shape: `short` (chat), `medium`, or `heavy`
  (long input + long output, closest to engine dashboards).

Use `--no-stream` only to compare against a non-streaming client; the report then says
plainly that tokens/s includes prefill.

## Benchmark Types

| Benchmark | Description |
|-----------|-------------|
| `performance` | Measures response time, tokens per second, throughput |
| `accuracy` | Evaluates answer accuracy on QA tasks |
| `code_generation` | Tests code generation capabilities |
| `reasoning` | Tests logical reasoning abilities |
| `multilingual` | Evaluates cross-language performance |
| `all` | Runs all benchmarks |

## Output Formats

- `json` - JSON format with detailed metrics
- `html` - HTML report with charts and visualizations
- `text` - Plain text output for terminal display

## License

MIT License
