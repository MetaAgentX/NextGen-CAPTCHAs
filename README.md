<h1 align="center">Next-Gen CAPTCHAs</h1>

<p align="center">
  <a href="https://github.com/MetaAgentX/NextGen-CAPTCHAs">
    <img src="./assets/logo.png" width="400" alt="Next-Gen CAPTCHAs" />
  </a>
</p>

<div align="center">

[![Webpage](https://img.shields.io/badge/Webpage-Live-2ea44f)](https://greenoso.github.io/NextGen-CAPTCHAs_webpage/)
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2602.09012)
[![Open In Spaces](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue)](https://huggingface.co/spaces/zcahjl3/NextGen-CAPTCHAs)
[![Dataset](https://img.shields.io/badge/%F0%9F%93%A6-dataset-orange)](https://huggingface.co/datasets/YaxinLuo/NextGen-CAPTCHAs)
![License](https://img.shields.io/badge/license-MIT-orange)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![GitHub stars](https://img.shields.io/github/stars/MetaAgentX/NextGen-CAPTCHAs?style=social)](https://github.com/MetaAgentX/NextGen-CAPTCHAs/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MetaAgentX/NextGen-CAPTCHAs?style=social)](https://github.com/MetaAgentX/NextGen-CAPTCHAs/network/members)

</div>

A defense framework against MLLM-based web GUI agents, with an accompanying benchmark snapshot of 519 puzzles across 27 CAPTCHA families. This repository provides both the generative CAPTCHA system and tools for evaluating agent resistance.

## Introduction

Current CAPTCHA types are no longer safe to MLLMs backed GUI agents. Browser-Use agent backed by frontier models like GPT-5.2, Gemini-3-Pro, and Claude-Opus4.5 achieve considerable pass rates on existing logic CAPTCHAs (Bingo, etc.), effectively collapsing the security barrier.

<p align="center">
  <img src="./assets/case_example.png" alt="Case Analysis" width="75%">
</p>

Next-Gen CAPTCHAs exploit the **Cognitive Gap** — the persistent asymmetry between human intuition and the over-segmented, step-by-step reasoning of GUI agents. We design interactive tasks that are solvable for humans without domain knowledge but systematically hard for agents due to bottlenecks in spatial grounding, temporal integration, and perception-to-action alignment.

We target five cognitive gap categories:
- **G1 — Scene-Structure Inference**: observation interpretation and grounding under partial observability
- **G2 — Temporal Integration**: multi-step evidence accumulation from motion and sequential reveals
- **G3 — Numerosity & Invariants**: decision-boundary sensitivity to discrete quantities and counts
- **G4 — Latent-State Tracking**: effective working-memory management across interaction steps
- **G5 — Perception-to-Action**: robust low-level execution of correct browser interactions

This repository provides:
1. **Defense Framework** — procedural generation of unlimited CAPTCHA instances (`captcha_generation/`)
2. **Benchmark Snapshot** — 519 puzzles across 27 families for reproducible evaluation (`captcha_data/`)
3. **Agent Evaluation Tools** — CLI integrations for Browser-Use and CrewAI (`agent_frameworks/`)

## Table of Contents

- [Introduction](#introduction)
- [News](#news)
- [Quick Start](#quick-start)
- [Key Results](#key-results)
- [CAPTCHA Families](#captcha-families)
- [Architecture: Defense Framework + Benchmark](#architecture-defense-framework--benchmark)
- [Benchmark Dataset](#benchmark-dataset)
- [Installation](#installation)
- [Download Dataset from Hugging Face](#download-dataset-from-hugging-face)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Running Benchmarks](#running-benchmarks)
- [Agent Frameworks](#agent-frameworks)
- [Project Structure](#project-structure)
- [License](#license)

## News

- [2026-02-09] Interactive demo is live on Hugging Face Spaces: https://huggingface.co/spaces/zcahjl3/NextGen-CAPTCHAs
- [2026-02-09] Project webpage is live: https://greenoso.github.io/NextGen-CAPTCHAs_webpage/

## Quick Start

```bash
# Install
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/MetaAgentX/NextGen-CAPTCHAs.git
cd NextGen-CAPTCHAs && uv sync

# Launch the CAPTCHA server
uv run app.py
# Open http://127.0.0.1:7860

# Run a benchmark (example with Gemini)
./test_benchmark.sh --llm google --model gemini-3-pro-preview \
    --max-output-tokens 32768 --max-steps 50 --port 7860 \
    --puzzles 'Dice_Roll_Path:5' --isolate-puzzles --seed 0 --headless
```

## Key Results

### Pipeline Overview

Our CAPTCHA generation pipeline identifies the human-agent cognitive gap and uses it to design tasks that are easy for humans but hard for AI agents.

<p align="center">
  <img src="./assets/nextgencaptcha_pipeline2.png" alt="NextGen-CAPTCHA Pipeline" width="45%">
</p>

### CAPTCHA Examples

Examples of the 27 CAPTCHA families spanning diverse cognitive challenges including 3D reasoning, spatial understanding, pattern recognition, and more.

<p align="center">
  <img src="./assets/nextgencaptchaexampleplot2.png" alt="CAPTCHA Examples" width="125%">
</p>

###  Advanced Model Performance on Next-Gen CAPTCHAs

Frontier AI models still lag significantly behind human performance on our benchmark, with a **92.90% accuracy gap**.

<p align="center">
  <img src="./assets/figure1_model_performance.png" alt="Model Performance Comparison" width="60%">
</p>

### Cost-Effectiveness Analysis

The best models achieve only single-digit accuracy while incurring high costs and long response times.

- **Full 519-puzzle evaluation**: Gemini-3-Pro-High, Gemini-3-Flash-High, Doubao-Seed-1.8-Thinking-HighEffort, Qwen3-VL-Plus-ThinkingHigh
- **135-puzzle lite subset** (due to latency/cost): Claude-Opus4.5-Extended-ThinkingHigh, GPT-5.2-xHigh — accuracy reflects the 135-puzzle lite subset only; costs in the figure are linearly extrapolated to the full 519-puzzle set for comparability

<p align="center">
  <img src="./assets/figure_cost_effectiveness.png" alt="Cost-Effectiveness Analysis" width="60%">
</p>

### Success–Trajectory Correlation

Success–trajectory correlation differs between current and Next-Gen CAPTCHAs: current CAPTCHAs show non-trivial correlations with interaction metrics (more steps/reasoning can help), while Next-Gen correlations stay near zero — more effort or reasoning budget rarely improves agent success.

<p align="center">
  <img src="./assets/captcha_correlation1.png" alt="Failure Case Correlation" width="115%">
</p>


## CAPTCHA Families

27 puzzle types targeting the five cognitive gap categories (G1–G5):

| CAPTCHA Type | Targeted Gaps |
|--------------|---------------|
| 3D Viewpoint | G1, G4, G5 |
| Backmost Layer | G1 |
| Box Folding | G1, G4 |
| Color Counting | G3 |
| Dice Roll Path | G3, G4, G5 |
| Dynamic Jigsaw | G2, G4, G5 |
| Hole Counting | G1, G3 |
| Illusory Ribbons | G1, G3 |
| Layered Stack | G1, G3 |
| Mirror | G1 |
| Multi Script | G1 |
| Occluded Pattern Counting | G1, G3 |
| Red Dot | G5 |
| Rotation Match | G1, G4 |
| Shadow Direction | G1 |
| Shadow Plausible | G1 |
| Spooky Circle | G2 |
| Spooky Circle Grid | G2, G3 |
| Spooky Jigsaw | G2, G4, G5 |
| Spooky Shape Grid | G2 |
| Spooky Size | G2, G5 |
| Spooky Text | G2 |
| Static Jigsaw | G4, G5 |
| Structure From Motion | G2 |
| Subway Paths | G3, G4 |
| Temporal Object Continuity | G2, G4 |
| Trajectory Recovery | G2, G4 |

## Architecture: Defense Framework + Benchmark

This repository contains two complementary components:

1. **Defense Framework** (generative system): Unbounded CAPTCHA generation with automatic generation code script (`captcha_generation/`)
2. **Benchmark Snapshot** (static dataset): Fixed set of puzzles for reproducible evaluation (`captcha_data/`)

The benchmark is a **snapshot** from the defense framework, not an exhaustive test set.

## Benchmark Dataset

- **Full set**: 519 puzzles — 20 per type (except Mirror: 11, Shadow_Plausible: 8)
- **Lite subset**: 135 puzzles — 5 per type, for cost-effective evaluation under limited query budgets

For the full lite subset index table, see [BENCHMARKING.md — Benchmark Dataset Details](BENCHMARKING.md#benchmark-dataset-details).

## Installation

**Requirements:** Python 3.11+ (required for browser-use integration). See `requirements.txt` or `pyproject.toml` for dependencies.

### Using uv (Highly Recommended! Faster! Easier & more reproducible!)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the application
# export BENCHMARK_RESULTS_FILE="results_{provider}_{model}_{timestamp}.json" (customize result filename; {timestamp} is server start time)
uv run app.py

# Use uv to run code so you do not have to activate virtual env everytime.
# Test the Browser-Use framework default agents (Their in house model BU1.0)
uv run agent_frameworks/browseruse_cli.py --url http://127.0.0.1:7860 --llm browser-use 

# Test with other models (e.x. openai gpt5 here)
uv run agent_frameworks/browseruse_cli.py --url http://127.0.0.1:7860 --llm openai --model gpt-5-2025-08-07
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Download Dataset from Hugging Face

Browse at https://huggingface.co/datasets/YaxinLuo/NextGen-CAPTCHAs or download via CLI:

```bash
# Make sure the hf CLI is installed
uv tool install hf

# Download the dataset
hf download YaxinLuo/NextGen-CAPTCHAs --repo-type=dataset

# download in custom path
hf download YaxinLuo/NextGen-CAPTCHAs --repo-type=dataset --local-dir ./NextGen-CAPTCHAs
```

## Usage

### Development Mode

```bash
export DEVELOPMENT=1
python app.py
```

The application will run in debug mode on `http://127.0.0.1:7860`

### Production Mode

```bash
# Using gunicorn (recommended for production)
gunicorn -w 4 -b 127.0.0.1:7860 app:app

# Or run directly
python app.py
```

The application will run on `http://127.0.0.1:7860`

Note: To accept connections from other machines, use `0.0.0.0` instead of `127.0.0.1` in the gunicorn command.

### Docker

```bash
# Build the image
docker build -t captcha-benchmark .

# Run the container
docker run -p 7860:7860 captcha-benchmark
```

## Configuration

The application supports various configuration options through environment variables and the puzzle data files. Each puzzle type has its own `ground_truth.json` file containing puzzle definitions and correct answers.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port number | `7860` |
| `DEVELOPMENT` | Enable debug mode | (unset) |
| `BENCHMARK_RESULTS_FILE` | Custom results filename pattern (`{timestamp}` uses server start so one file per run) | `benchmark_results_{provider}_{model}_{timestamp}.json` |

### Server Options

```bash
uv run app.py [OPTIONS]

Options:
  --port PORT              Server port (default: 7860 or PORT env var)
  --host HOST              Host to bind to (default: 127.0.0.1)
  --results-file FILE      Override benchmark results filename
  --fixed-sequential       Load puzzles in fixed order (0,1,2,3...) instead of random
  --human-test             Save all results to benchmark_results_human.json
```

**Examples:**

```bash
# Custom port
uv run app.py --port 7865

# Fixed sequential mode for human testing (puzzles load in order)
uv run app.py --fixed-sequential

# Human test mode (saves all results to single file)
uv run app.py --fixed-sequential --human-test

# Combined
uv run app.py --port 7865 --fixed-sequential --human-test
```

**Fixed Sequential Mode:** When enabled, puzzles within each type load in natural order (`dice_roll_path_0000`, `dice_roll_path_0001`, ...) instead of randomly. Useful for human testing where different testers handle different puzzle ranges.

**Human Test Mode:** When enabled, all benchmark results are saved to a single `benchmark_results_human.json` file instead of timestamped files. This makes it easy to collect and analyze human test results separately from agent benchmarks.

### Reproducible Puzzle Generation

For dynamically generated puzzles (Dynamic_Jigsaw, Red_Dot, Spooky_Circle, etc.), you can pass a `seed` parameter to get reproducible results:

```bash
# Get the same puzzle every time with seed=0
curl "http://127.0.0.1:7860/api/get_puzzle?type=Dynamic_Jigsaw&seed=0"
```

This is useful for:
- Debugging specific puzzle instances
- Fair benchmarking (same puzzles across different models)
- Reproducing test failures

## API Endpoints

- `GET /` - Main web interface
- `GET /api/get_puzzle` - Get a new puzzle
- `POST /api/check_answer` - Submit an answer for verification
- `GET /api/puzzle_types` - Get list of available puzzle types

## Running Benchmarks

```bash
# Run with Gemini 3 Pro
./test_benchmark.sh --llm google --model gemini-3-pro-preview \
    --max-output-tokens 32768 --max-steps 50 --port 7860 \
    --puzzles 'Dice_Roll_Path:10' --isolate-puzzles --seed 0 --headless

# Run with GPT-5.2
./test_benchmark.sh --llm openai --model gpt-5.2 \
    --reasoning-effort xhigh --max-steps 50 --port 7860 \
    --puzzles 'Mirror:10' --isolate-puzzles --seed 0 --headless

# Run with Claude Sonnet 4
./test_benchmark.sh --llm anthropic --model claude-sonnet-4-20250514 \
    --max-steps 50 --port 7860 \
    --puzzles 'Box_Folding:5' --isolate-puzzles --seed 0 --headless
```

See [BENCHMARKING.md](BENCHMARKING.md) for full CLI reference, parallel execution, paper reproduction settings, output format, and result classification.

### Paper Reproducibility

For reproducing the paper's main results (two-tier evaluation protocol, seed settings, latency caveats), see [BENCHMARKING.md — Reproducing Paper Results](BENCHMARKING.md#reproducing-paper-results).

## Agent Frameworks

We provide two agent framework integrations for evaluating GUI agents against Next-Gen CAPTCHAs:

**Browser-Use** (default, used in paper):
```bash
uv run agent_frameworks/browseruse_cli.py --llm openai --model gpt-4o
```

**CrewAI** (alternative framework):
```bash
uv run agent_frameworks/crewai_cli.py \
    --url http://127.0.0.1:7860 \
    --llm openai --model gpt-4o
```

See [AGENT_FRAMEWORKS.md](AGENT_FRAMEWORKS.md) for the POMDP formulation, extended CLI options, CrewAI details, and framework comparison.

## Project Structure

```
NextGen-CAPTCHAs/
├── app.py                 # Main Flask application
├── main.py                # Alternative entry point
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Modern Python project configuration
├── Dockerfile             # Docker configuration
├── captcha_generation/    # Procedural CAPTCHA generators (27 types)
├── captcha_data/          # Benchmark puzzle data (27 type directories)
├── agent_frameworks/      # Agent CLI integrations (Browser-Use, CrewAI)
├── static/                # Static assets (CSS, JS)
├── templates/             # Jinja2 templates
├── BENCHMARKING.md        # Detailed benchmarking reference
└── AGENT_FRAMEWORKS.md    # Agent framework & POMDP details
```

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{nextgencaptchas2026,
  title={Next-Gen CAPTCHAs: A Defense Framework Against MLLM-Based Web GUI Agents},
  author={TODO: Add authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MetaAgentX/NextGen-CAPTCHAs&type=Date)](https://star-history.com/#MetaAgentX/NextGen-CAPTCHAs&Date)

## License

This project is licensed under the MIT License. See `LICENSE` for details.
