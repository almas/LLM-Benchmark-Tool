"""Report generators for benchmark results."""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from llm_benchmark.models import BenchmarkResult, BenchmarkType, OutputFormat


class ReportGenerator:
    """Generates reports from benchmark results."""

    def __init__(self, results: List[BenchmarkResult]):
        self.results = results

    def generate(self, output_format: OutputFormat = OutputFormat.TEXT, output_file: Optional[str] = None) -> str:
        """Generate a report in the specified format."""
        if output_format == OutputFormat.JSON:
            content = self._generate_json()
        elif output_format == OutputFormat.HTML:
            content = self._generate_html()
        else:
            content = self._generate_text()

        if output_file:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

        return content

    def _generate_json(self) -> str:
        """Generate JSON report."""
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "tool": "LLM Benchmark Tool",
                "version": "1.0.0",
                "total_benchmarks": len(self.results),
            },
            "benchmarks": [],
        }

        for result in self.results:
            benchmark_data = result.to_dict()
            report_data["benchmarks"].append(benchmark_data)

        return json.dumps(report_data, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _num(value: Optional[float], digits: int = 1) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    @classmethod
    def _stats_line(
        cls,
        label: str,
        stats: Dict[str, Optional[float]],
        digits: int = 1,
        unit: str = "",
    ) -> str:
        """One aligned line with mean / median / p95 / min / max of a metric."""

        def show(value: Optional[float]) -> str:
            return cls._num(value, digits) + (unit if value is not None else "")

        return (
            f"  {label:<22}{show(stats.get('mean'))}"
            f" | median {show(stats.get('median'))}"
            f" | p95 {show(stats.get('p95'))}"
            f" | min {show(stats.get('min'))}"
            f" | max {show(stats.get('max'))}"
        )

    @staticmethod
    def _token_sources(result: BenchmarkResult) -> str:
        """How output token counts were obtained (server usage vs estimate)."""
        counts: Dict[str, int] = {}
        for trial in result.successful_trials:
            key = trial.token_source or "unknown"
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return "n/a"
        return ", ".join(f"{name} ({n})" for name, n in sorted(counts.items()))

    def _generate_text(self) -> str:
        """Generate plain text report."""
        lines = []
        lines.append("=" * 70)
        lines.append("LLM BENCHMARK REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Benchmarks Run: {len(self.results)}")
        lines.append("")

        for result in self.results:
            lines.append("-" * 70)
            # Handle cases where result might be BenchmarkType or BenchmarkResult
            b_type = result.benchmark_type if hasattr(result, 'benchmark_type') else result
            b_type_val = b_type.value if hasattr(b_type, 'value') else str(b_type)
            lines.append(f"Benchmark: {b_type_val.upper()}")
            lines.append(f"Model: {result.model_name}")
            lines.append(f"Provider: {result.provider.value}")
            lines.append("-" * 70)

            # How the numbers were produced - always show this, otherwise a
            # decode rate and an end-to-end rate look interchangeable.
            run_config = getattr(result, "config", None) or {}
            mode = "streaming" if result.streaming_measured else "non-streaming"
            lines.append(
                f"  Measurement:         {mode}, concurrency {run_config.get('concurrency', 'n/a')}, "
                f"warm-up {run_config.get('warmup_requests', 'n/a')} request(s)"
            )
            lines.append(f"  Output token counts: {self._token_sources(result)}")

            lines.append(
                f"  Trials:              {len(result.trials)} "
                f"({len(result.successful_trials)} usable, {len(result.failed_trials)} failed)"
            )

            if result.duration is not None:
                lines.append(f"  Total Duration:      {result.duration:.2f}s")

            lines.append(self._stats_line("Decode tokens/sec:", result.decode_tps_stats))
            lines.append(self._stats_line("End-to-end tok/sec:", result.e2e_tps_stats))
            lines.append(self._stats_line("TTFT:", result.ttft_stats, digits=3, unit="s"))
            lines.append(self._stats_line("Inter-token lat:", result.itl_stats, digits=4, unit="s"))

            if result.aggregate_throughput is not None:
                lines.append(f"  Aggregate output:    {result.aggregate_throughput:.1f} tok/s (wall clock)")

            if result.avg_response_time is not None:
                lines.append(f"  Avg Response Time:   {result.avg_response_time:.3f}s")

            lines.append(
                f"  Total Tokens:        {result.total_tokens} "
                f"(in {result.total_input_tokens} / out {result.total_output_tokens})"
            )

            if not result.streaming_measured:
                lines.append(
                    "  Note: no decode-phase timing - tokens/s includes prefill, "
                    "queueing and network (enable streaming to separate them)."
                )

            # Accuracy score if available (only for applicable benchmark types)
            accuracy = result.applicable_accuracy
            if accuracy is not None:
                lines.append(f"  Accuracy Score:      {accuracy:.2%}")
            elif result.has_expected_answers:
                # Show N/A for benchmarks with expected answers but no applicable accuracy
                lines.append(f"  Accuracy Score:      N/A")

            # Error count
            errors = [t for t in result.trials if t.error]
            if errors:
                lines.append(f"  Errors:              {len(errors)}")

            lines.append("")

            # Individual trial details (abbreviated)
            for trial in result.trials[:5]:  # Show first 5 trials
                status = "ERROR" if trial.error else "OK"
                response_time = f"{trial.response_time:.2f}s" if trial.response_time is not None else "n/a"
                ttft = f"{trial.ttft:.3f}s" if trial.ttft is not None else "n/a"
                decode = (
                    f"{trial.decode_tokens_per_second:.1f} tok/s"
                    if trial.decode_tokens_per_second is not None
                    else "n/a"
                )
                lines.append(
                    f"  [{status}] {trial.task_id}: {response_time}, {trial.output_tokens or 0} tokens, "
                    f"TTFT {ttft}, decode {decode}"
                )
                if trial.error:
                    lines.append(f"         Error: {trial.error}")

            if len(result.trials) > 5:
                lines.append(f"  ... and {len(result.trials) - 5} more trials")

            lines.append("")

        # Overall summary
        lines.append("=" * 70)
        lines.append("OVERALL SUMMARY")
        lines.append("=" * 70)

        avg_times = [r.avg_response_time for r in self.results if r.avg_response_time is not None]
        if avg_times:
            lines.append(f"  Average Response Time (all benchmarks): {sum(avg_times) / len(avg_times):.3f}s")

        decode_rates = [
            r.avg_decode_tokens_per_second
            for r in self.results
            if r.avg_decode_tokens_per_second is not None
        ]
        if decode_rates:
            lines.append(
                f"  Decode tokens/sec (all benchmarks):         "
                f"{sum(decode_rates) / len(decode_rates):.1f}"
            )

        e2e_rates = [
            r.avg_e2e_tokens_per_second
            for r in self.results
            if r.avg_e2e_tokens_per_second is not None
        ]
        if e2e_rates:
            lines.append(
                f"  End-to-end tokens/sec (all benchmarks):     "
                f"{sum(e2e_rates) / len(e2e_rates):.1f}"
            )

        total_trials = sum(len(r.trials) for r in self.results)
        lines.append(f"  Total Trials: {total_trials}")

        total_errors = sum(len([t for t in r.trials if t.error]) for r in self.results)
        if total_errors > 0:
            lines.append(f"  Total Errors: {total_errors}")

        lines.append("=" * 70)

        return "\n".join(lines)

    def _generate_html(self) -> str:
        """Generate HTML report with charts and visualizations."""
        benchmark_rows = []
        accuracy_bars = []

        for result in self.results:
            avg_time = result.avg_response_time or 0
            decode_stats = result.decode_tps_stats
            tps = decode_stats["median"] or result.avg_tokens_per_second or 0
            ttft_median = result.ttft_stats["median"]
            e2e_median = result.e2e_tps_stats["median"]
            aggregate = result.aggregate_throughput
            meas = result.config or {}
            mode = "streaming" if result.streaming_measured else "non-streaming"
            concurrency = meas.get("concurrency", 1)
            token_source = meas.get("token_source") or "unknown"
            
            # Only show accuracy for applicable benchmark types
            accuracy = result.applicable_accuracy if hasattr(result, 'applicable_accuracy') else None
            accuracy_pct = (accuracy * 100) if accuracy is not None else None

            # Progress bar for metrics
            time_bar = self._create_progress_bar(avg_time * 100, max_width=50, color="#3498db")
            tps_bar = self._create_progress_bar(tps, max_width=50, max_value=100, color="#2ecc71")

            acc_html = ""
            if accuracy_pct is not None:
                acc_color = "#2ecc71" if accuracy_pct >= 80 else "#f39c12" if accuracy_pct >= 50 else "#e74c3c"
                acc_html = f"""
                    <tr>
                        <td>Accuracy</td>
                        <td>{accuracy_pct:.1f}%</td>
                        <td>
                            <div style="background:#ecf0f1;border-radius:4px;height:20px;width:100%;">
                                <div style="background:{acc_color};width:{accuracy_pct}%;height:100%;border-radius:4px;"></div>
                            </div>
                        </td>
                    </tr>
                """
            elif result.has_expected_answers:
                # Show N/A for benchmarks with expected answers but not applicable accuracy
                acc_html = """
                    <tr>
                        <td>Accuracy</td>
                        <td>N/A</td>
                        <td><span style="color:#95a5a6;">—</span></td>
                    </tr>
                """

            benchmark_rows.append(f"""
            <tr>
                <td><strong>{result.benchmark_type.value.upper() if hasattr(result, 'benchmark_type') else result.value.upper()}</strong></td>
                <td>{result.model_name}</td>
                <td>{result.provider.value}</td>
                <td>{len(result.successful_trials)}/{len(result.trials)}</td>
                <td>{avg_time:.3f}s</td>
                <td>{tps:.1f}</td>
                <td>{self._num(e2e_median, 1)}</td>
                <td>{self._num(aggregate, 1)}</td>
                <td>{mode} / conc {concurrency} / tokens {token_source}</td>
                <td>{self._num(ttft_median, 3) + ('s' if ttft_median is not None else '')}</td>
                <td>{result.total_tokens}</td>
            </tr>
            {acc_html}
            """)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f6fa;
            color: #2c3e50;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .report-header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            margin: 10px;
            display: inline-block;
            min-width: 200px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #3498db;
            color: white;
            padding: 12px 15px;
            text-align: left;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .trial-details {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}
        .trial-item {{
            padding: 10px;
            border-left: 3px solid #3498db;
            margin: 10px 0;
            background: #f8f9fa;
        }}
        .trial-item.error {{
            border-left-color: #e74c3c;
            background: #fdf0f0;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="report-header">
        <h1>LLM Benchmark Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Benchmarks: {len(self.results)}</p>
    </div>

    <h2>Key Metrics</h2>
    <div>
"""

        # Key metric cards
        avg_times = [r.avg_response_time for r in self.results if r.avg_response_time is not None]
        avg_tps = [
            r.avg_decode_tokens_per_second or r.avg_tokens_per_second
            for r in self.results
            if (r.avg_decode_tokens_per_second or r.avg_tokens_per_second) is not None
        ]
        total_trials = sum(len(r.trials) for r in self.results)

        if avg_times:
            overall_avg = sum(avg_times) / len(avg_times)
            html += f'''        <div class="metric-card">
            <div class="metric-value">{overall_avg:.3f}s</div>
            <div class="metric-label">Avg Response Time</div>
        </div>
'''

        if avg_tps:
            overall_tps = sum(avg_tps) / len(avg_tps)
            html += f'''        <div class="metric-card">
            <div class="metric-value">{overall_tps:.1f}</div>
            <div class="metric-label">Decode tokens/sec</div>
        </div>
'''

        html += f'''        <div class="metric-card">
            <div class="metric-value">{total_trials}</div>
            <div class="metric-label">Total Trials</div>
        </div>
    </div>

    <h2>Benchmark Results</h2>
    <table>
        <thead>
            <tr>
                <th>Benchmark</th>
                <th>Model</th>
                <th>Provider</th>
                <th>Trials (usable)</th>
                <th>Avg Response</th>
                <th>Decode tok/s</th>
                <th>E2E tok/s</th>
                <th>Aggregate tok/s</th>
                <th>Measurement</th>
                <th>TTFT median</th>
                <th>Total Tokens</th>
            </tr>
        </thead>
        <tbody>
'''
        html += "\n".join(benchmark_rows)
        html += """        </tbody>
    </table>

    <h2>Detailed Trial Results</h2>
"""

        for result in self.results:
            html += f'''    <div class="trial-details">
                <h3>{result.benchmark_type.value.upper() if hasattr(result, 'benchmark_type') else result.value.upper()}</h3>
'''
            for trial in result.trials:
                error_class = "error" if trial.error else ""
                badge_class = "badge-danger" if trial.error else "badge-success"
                badge_text = "ERROR" if trial.error else "OK"
                response_time = f"{trial.response_time:.2f}s" if trial.response_time is not None else "n/a"
                decode_rate = (
                    f"{trial.decode_tokens_per_second:.1f} tok/s"
                    if trial.decode_tokens_per_second is not None
                    else "n/a"
                )
                ttft = f"{trial.ttft:.3f}s" if trial.ttft is not None else "n/a"
                html += f'''        <div class="trial-item {error_class}">
            <span class="badge {badge_class}">{badge_text}</span>
            <strong>{trial.task_id}</strong>
            - {response_time}, {trial.output_tokens or 0} tokens, TTFT {ttft}, decode {decode_rate}
            {'<br><small style="color:#e74c3c">Error: ' + trial.error + '</small>' if trial.error else ''}
        </div>
'''
            html += "    </div>\n"

        html += """
    <footer>
        <p>Generated by LLM Benchmark Tool v1.0.0</p>
    </footer>
</body>
</html>"""

        return html

    def _create_progress_bar(
        self,
        value: float,
        max_width: int = 50,
        max_value: float = None,
        color: str = "#3498db",
    ) -> str:
        """Create an HTML progress bar."""
        if max_value is None:
            max_value = value * 1.2 if value > 0 else 1
        pct = min(100, (value / max_value) * 100) if max_value > 0 else 0
        width = max(1, int(pct / 100 * max_width))
        return f'<div style="background:#ecf0f1;border-radius:4px;height:20px;width:{max_width}px;">' \
               f'<div style="background:{color};width:{width}px;height:100%;border-radius:4px;"></div></div>'