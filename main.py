#!/usr/bin/env python3
"""LLM Benchmark Tool - Main entry point."""

import sys
import os

# Ensure the current project directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_benchmark.cli import main

if __name__ == "__main__":
    sys.exit(main())