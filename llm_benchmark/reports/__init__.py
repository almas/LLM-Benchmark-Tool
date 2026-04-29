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

            # Summary statistics
            lines.append(f"  Trials:              {len(result.trials)}")

            if result.duration is not None:
                lines.append(f"  Total Duration:      {result.duration:.2f}s")

            if result.avg_response_time is not None:
                lines.append(f"  Avg Response Time:   {result.avg_response_time:.3f}s")

            if result.avg_tokens_per_second is not None:
                lines.append(f"  Avg Tokens/Second:   {result.avg_tokens_per_second:.1f}")

            lines.append(f"  Total Tokens:        {result.total_tokens}")

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
                lines.append(f"  [{status}] {trial.task_id}: {trial.response_time:.2f}s, "
                           f"{trial.output_tokens or 0} tokens")
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

        avg_tps = [r.avg_tokens_per_second for r in self.results if r.avg_tokens_per_second is not None]
        if avg_tps:
            lines.append(f"  Average Tokens/Second (all benchmarks): {sum(avg_tps) / len(avg_tps):.1f}")

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
            tps = result.avg_tokens_per_second or 0
            
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
                <td>{len(result.trials)}</td>
                <td>{avg_time:.3f}s</td>
                <td>{tps:.1f}</td>
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
        avg_tps = [r.avg_tokens_per_second for r in self.results if r.avg_tokens_per_second is not None]
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
            <div class="metric-label">Avg Tokens/Second</div>
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
                <th>Trials</th>
                <th>Avg Response</th>
                <th>Tokens/Sec</th>
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
                html += f'''        <div class="trial-item {error_class}">
            <span class="badge {badge_class}">{badge_text}</span>
            <strong>{trial.task_id}</strong>
            - {trial.response_time:.2f}s, {trial.output_tokens or 0} tokens
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