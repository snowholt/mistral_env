#!/usr/bin/env python3
"""
BeautyAI Independent Benchmarking & Testing System
================================================

This is an independent system for benchmarking and testing BeautyAI models
for content filtering and performance evaluation.

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start BeautyAI API** (in the main project):
   ```bash
   cd /path/to/beautyai
   python -m beautyai_inference.api.app
   ```

3. **Start Web UI (NEW!):**
   ```bash
   cd src/web_ui
   python start_ui.py
   # Open http://localhost:5000
   ```

4. **Run Quick Demo:**
   ```bash
   cd src/benchmarking
   python demo_benchmarking.py
   ```

5. **Run Full Benchmark:**
   ```bash
   cd src/benchmarking
   python enhanced_benchmarking.py --model all --sample-size 100
   ```

6. **Analyze Results:**
   ```bash
   cd src/benchmarking
   python analyze_results.py benchmark_results_*.json --generate-report
   ```

7. **Run All Tests:**
   ```bash
   python run_tests.py
   ```

## Directory Structure

```
benchmark_and_test/
├── README.md                           # This file
├── requirements.txt                    # Dependencies
├── run_tests.py                       # Test runner
├── ENHANCED_BENCHMARKING.md           # Benchmarking documentation
├── CONTENT_FILTERING.md               # Content filtering documentation
├── endpoints.md                       # API endpoints documentation
├── src/                              # Source code
│   ├── web_ui/                       # BeautyAI Web UI Assistant
│   │   ├── app.py                    # Flask web application
│   │   ├── start_ui.py               # UI launcher script
│   │   ├── templates/index.html      # Main UI template
│   │   └── README.md                 # Web UI documentation
│   ├── benchmarking/                 # Benchmarking tools
│   │   ├── enhanced_benchmarking.py  # Main benchmarking script
│   │   ├── analyze_results.py        # Results analyzer
│   │   └── demo_benchmarking.py      # Demo script
│   ├── content_filtering/            # Content filtering tools
│   │   ├── content_filter_cli.py     # CLI tool
│   │   └── content_filter_demo.py    # Demo script
│   ├── tests/                       # Test suite
│   │   ├── test_enhanced_benchmarking.py  # Benchmarking tests
│   │   └── test_content_filter.py    # Content filter tests
│   └── data/                        # Reference data
│       ├── 2000QAToR.csv            # Cosmetic procedure questions
│       └── models_info.md           # Model information
```

## Features

### ✅ **BeautyAI Web UI Assistant**
- Beautiful animated fractal background interface
- Real-time chat with BeautyAI models
- Advanced parameter controls and optimization presets
- Thinking mode visualization
- Content filtering controls
- Performance metrics display

### ✅ **Enhanced Benchmarking System**
- Tests all 2000 cosmetic procedure questions
- Measures content filtering effectiveness 
- Performance benchmarking (latency, throughput)
- Multi-model comparison
- Export to JSON and CSV formats
- Comprehensive analysis and reporting

### ✅ **Content Filtering Tools**
- CLI tools for testing content filters
- Demo scripts for validation
- Integration with BeautyAI API

### ✅ **Comprehensive Test Suite**
- Unit tests for all components
- Benchmark validation tests
- Content filter tests
- Categorized test runner

## Requirements

- Python 3.10+
- BeautyAI API running on localhost:8000
- Dependencies: aiohttp, pandas, matplotlib, seaborn, pytest

## Independence

This system is designed to run completely independently from the main BeautyAI project.
It only requires the BeautyAI API to be running to perform the actual benchmarking.

## Usage Examples

See ENHANCED_BENCHMARKING.md and CONTENT_FILTERING.md for detailed usage examples
and comprehensive documentation.






