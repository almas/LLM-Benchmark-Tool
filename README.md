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

Create a `config.yaml` file:

```yaml
provider:
  name: openai
  api_key: your-api-key
  base_url: https://api.openai.com/v1
  timeout: 120

benchmarks:
  performance:
    max_tokens: 500
    num_trials: 10
  accuracy:
    num_questions: 20
  code_generation:
    num_tasks: 10
  reasoning:
    num_tasks: 10
```

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
