# Agent Frameworks Reference

Detailed documentation for the agent frameworks used to evaluate Next-Gen CAPTCHAs. For a quick start, see the [README](README.md#agent-frameworks).

## Table of Contents

- [POMDP Model](#pomdp-model)
  - [POMDP Components Mapping](#pomdp-components-mapping)
- [Browser-Use CLI (`browseruse_cli.py`)](#browser-use-cli-browseruse_clipy)
- [Extended CLI (`browseruse_extended_cli.py`)](#extended-cli-browseruse_extended_clipy)
  - [Dependencies for Extended Features](#dependencies-for-extended-features)
- [CrewAI CLI (`crewai_cli.py`)](#crewai-cli-crewai_clipy)
- [Framework Comparison: Browser-Use vs CrewAI](#framework-comparison-browser-use-vs-crewai)

## POMDP Model

We model the VLM web agent as an extended POMDP (Partially Observable Markov Decision Process):

$$\mathcal{W} = (S, O, X, A_{\text{web}}, A_{\text{think}}, Z, T_{\text{env}}, U, R, \kappa)$$

> **Note:** The extended CLI supports additional components (`A_api`, separate `U_mem`/`U_think`) not in the paper's core formulation.

Below is how each theoretical component maps to our browser-use implementation, with corresponding code variable names.

### POMDP Components Mapping

#### Observation Space ($o_t$)

| Theory | Description | Code Variable |
|--------|-------------|---------------|
| $I_t$ | Screenshot | `browser_state.screenshot` |
| $D_t$ | DOM tree | `page.evaluate()` extracts `window.currentPuzzle` |
| $\text{meta}_t$ | URL, viewport | `args.url`, `browser_kwargs['viewport']` |

#### Internal State ($x_t$)

| Theory | Description | Code Variable |
|--------|-------------|---------------|
| $m_t$ | Procedural memory | `MemoryConfig(memory_interval=N)`, `enable_memory` |
| $\tau_t$ | Thought trace | `agent_output` in step callback, `use_thinking=True` |

#### Action Spaces

| Theory | Description | Code Variable |
|--------|-------------|---------------|
| $A_{\text{web}}$ | Browser actions | Default browser-use actions: click, scroll, type, drag |
| $A_{\text{api}}$ | External API calls (optional, not in paper) | Extended CLI only: `http_get`, `http_post` |
| $A_{\text{think}}$ | Planner reasoning | `planner_llm`, `planner_interval`, `is_planner_reasoning` |

#### Dynamics & Learning

| Theory | Description | Code Variable |
|--------|-------------|---------------|
| $\pi$ | Policy (action selection) | `llm` param to `Agent()`, created by `llm_factories[llm_name](args)` |
| $T_{\text{env}}$ | World transition | Playwright browser via `Browser(**browser_kwargs)` |
| $U$ | State update | `memory_interval` in `MemoryConfig`, agent state updates |
| $R$ | Reward signal | `detect_result_from_page()` returns `is_correct` |
| $\kappa$ | Deliberation cost | `usage` dict: `input_tokens`, `output_tokens`, `reasoning_tokens` |

#### Feature Availability

| Component | Basic CLI | Extended CLI | Dependencies |
|-----------|-----------|--------------|--------------|
| Observations ($o_t$) | Screenshot + DOM | Same | - |
| Memory ($m_t$) | Default | Custom interval | mem0 |
| Thought ($\tau_t$) | Built-in | Planner LLM | - |
| Web Actions ($A_{\text{web}}$) | All | Same | - |
| API Actions ($A_{\text{api}}$) | Extensible | Optional extension | aiohttp |
| Cost Tracking ($\kappa$) | Token logging | Same | - |

## Browser-Use CLI (`browseruse_cli.py`)

Standard browser-use agent with:
- Screenshot + DOM observations
- Web actions (click, type, scroll, drag)
- LLM-based policy
- Token cost tracking

```bash
# Run with OpenAI
uv run agent_frameworks/browseruse_cli.py --llm openai --model gpt-4o

# Run with Anthropic
uv run agent_frameworks/browseruse_cli.py --llm anthropic --model claude-sonnet-4-20250514

# Run with Google
uv run agent_frameworks/browseruse_cli.py --llm google --model gemini-2.0-flash
```

## Extended CLI (`browseruse_extended_cli.py`)

Adds POMDP-aligned features:

| Feature | Flag | POMDP Component |
|---------|------|-----------------|
| Procedural Memory | `--procedural-memory-interval N` | U_mem (memory consolidation) |
| Disable Memory | `--disable-procedural-memory` | Disable m_t updates |
| Planner LLM | `--enable-planner` | A_think (separate reasoning) |
| Planner Model | `--planner-model MODEL` | Different model for planning |
| Planner Interval | `--planner-interval N` | Plan every N steps |
| Planner Reasoning | `--planner-reasoning` | Extended reasoning format |
| API Actions | `--enable-api-actions` | A_api (external HTTP calls) |
| API Timeout | `--api-timeout N` | Timeout for API calls |
| API Domains | `--api-allowed-domains` | Security: restrict allowed domains |

```bash
# With procedural memory (consolidates every 5 steps)
uv run agent_frameworks/browseruse_extended_cli.py \
    --llm openai --model gpt-4o \
    --procedural-memory-interval 5

# With separate planner (uses cheaper model for planning)
uv run agent_frameworks/browseruse_extended_cli.py \
    --llm openai --model gpt-4o \
    --enable-planner --planner-model gpt-4o-mini

# With API actions enabled (allows agent to call external HTTP APIs)
uv run agent_frameworks/browseruse_extended_cli.py \
    --llm openai --model gpt-4o \
    --enable-api-actions

# With API actions and domain restrictions (security)
uv run agent_frameworks/browseruse_extended_cli.py \
    --llm openai --model gpt-4o \
    --enable-api-actions --api-allowed-domains "api.example.com,api.openai.com"

# Full configuration with all features
uv run agent_frameworks/browseruse_extended_cli.py \
    --llm openai --model gpt-4o \
    --procedural-memory-interval 5 \
    --enable-planner --planner-model gpt-4o-mini --planner-interval 3 \
    --enable-api-actions --api-timeout 60
```

### Dependencies for Extended Features

```bash
# Memory features require mem0 and sentence-transformers
pip install mem0 sentence-transformers

# API actions require aiohttp
pip install aiohttp

# Or with uv
uv add mem0 sentence-transformers aiohttp
```

If not installed, features are gracefully disabled with a warning.

## CrewAI CLI (`crewai_cli.py`)

An alternative agent framework using CrewAI for benchmarking comparisons. This implementation mirrors browseruse_cli.py functionality for fair performance testing.

**Key Differences from Browser-Use:**

| Feature | browseruse_cli.py | crewai_cli.py |
|---------|-------------------|---------------|
| Browser Control | Direct Playwright via browser-use Agent | CrewAI's BrowserTool abstraction |
| Agent Architecture | Single agent with built-in reasoning | CrewAI Agent + Crew orchestration |
| Step Callbacks | Custom step hooks with screenshots | CrewAI's built-in step_callback |
| Token Capture | HTTP interception for all providers | Same HTTP interception approach |

**Supported LLM Providers:**
- `openai` - GPT-4o, GPT-5, etc.
- `anthropic` - Claude models
- `google` - Gemini models
- `groq` - Groq-hosted models
- `azure-openai` - Azure OpenAI Service
- `vllm` - Self-hosted vLLM inference
- `qwen` - Alibaba Qwen models (via DashScope)
- `doubao` - ByteDance Doubao models

**Basic Usage:**

```bash
# Run with OpenAI
uv run agent_frameworks/crewai_cli.py \
    --url http://127.0.0.1:7860 \
    --llm openai --model gpt-4o

# Run with Anthropic Claude
uv run agent_frameworks/crewai_cli.py \
    --url http://127.0.0.1:7860 \
    --llm anthropic --model claude-sonnet-4-20250514

# Run with Google Gemini (thinking model)
uv run agent_frameworks/crewai_cli.py \
    --url http://127.0.0.1:7860 \
    --llm google --model gemini-2.5-pro-preview-05-06 \
    --max-output-tokens 32768 --thinking-budget 16384

# Run with vLLM (self-hosted)
uv run agent_frameworks/crewai_cli.py \
    --url http://127.0.0.1:7860 \
    --llm vllm --model Qwen/Qwen3-VL-8B-Thinking \
    --base-url http://localhost:8000/v1 \
    --max-output-tokens 65536

# Run with Qwen (thinking mode enabled)
uv run agent_frameworks/crewai_cli.py \
    --url http://127.0.0.1:7860 \
    --llm qwen --model qwen-vl-max-latest
```

**CLI Arguments:**

```bash
uv run agent_frameworks/crewai_cli.py [OPTIONS]

Required:
  --url URL               Target URL (e.g., http://127.0.0.1:7860)

LLM Provider:
  --llm PROVIDER          openai, anthropic, google, groq, azure-openai, vllm, qwen, doubao
  --model MODEL           Model name (provider-specific)
  --base-url URL          API base URL (vLLM, Azure)
  --api-key KEY           API key override

Model Parameters:
  --temperature FLOAT     Sampling temperature (default: provider-specific)
  --reasoning-effort LVL  OpenAI: none, low, medium, high, xhigh
  --max-output-tokens N   Max output tokens (Gemini, vLLM)
  --thinking-budget N     Gemini 2.5 thinking budget
  --thinking-level LVL    Gemini 3 thinking: minimal, low, medium, high
  --disable-thinking      Disable thinking for vLLM Qwen models

Execution:
  --max-steps N           Max steps per puzzle (default: 1000)
  --max-actions-per-step  Max actions per step (default: 10)
  --max-failures N        Max consecutive failures (default: 5)
  --llm-timeout SECS      LLM request timeout (default: 1800)
  --step-timeout SECS     Step execution timeout (default: 1800)
  --isolate-puzzles       Fresh agent per puzzle (no memory)
  --headless              Run browser headless

Logging:
  --no-log-llm            Disable LLM logging
  --llm-log-dir DIR       Log directory (default: llm_logs)
  --run-id ID             Shared run ID for logs
  --debug-vllm            Debug output for vLLM
```

## Framework Comparison: Browser-Use vs CrewAI

Use this guide to choose the right framework for your benchmarking needs:

| Use Case | Recommended Framework | Reason |
|----------|----------------------|--------|
| Production benchmarking | browseruse_cli.py | More mature, direct Playwright control |
| CrewAI performance testing | crewai_cli.py | Direct comparison with browseruse |
| Custom step callbacks | browseruse_cli.py | Better step-level hooks |
| Multi-agent scenarios | crewai_cli.py | CrewAI's orchestration features |
| Token cost analysis | Both | Same HTTP interception approach |

**Running Comparative Benchmarks:**

```bash
# Test the same puzzle with both frameworks

# Browser-Use framework
./test_benchmark.sh --llm openai --model gpt-4o \
    --puzzles 'Dice_Roll_Path:5' --isolate-puzzles --seed 0 --headless

# CrewAI framework (run directly)
uv run agent_frameworks/crewai_cli.py \
    --url http://127.0.0.1:7860 \
    --llm openai --model gpt-4o \
    --isolate-puzzles --headless
```

Both frameworks produce compatible output:
- `benchmark_results_*.json` - Results compatible with analysis tools
- `llm_logs/` - Per-puzzle logs with token usage
