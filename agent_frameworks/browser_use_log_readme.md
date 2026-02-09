# Browser-Use LLM Logging System

This document describes the LLM logging system used in the CAPTCHA benchmark framework. The system captures detailed LLM interactions for debugging, analysis, and fine-tuning.

## Overview

**Main Components:**
- `llm_logger.py` - `EnhancedLLMLogger` class for structured logging
- `browseruse_cli.py` - Provider-specific capture clients and factories

**Purpose:**
- Capture complete LLM input/output pairs
- Track token usage (input, output, reasoning)
- Save screenshots for visual debugging
- Support multiple LLM providers with native field names

## Directory Structure

```
llm_logs/{run_id}/
├── run_config.json           # CLI args, LLM info, environment
├── summary.json              # Aggregated stats across all puzzles
├── benchmark_results.json    # Per-puzzle results
└── puzzles/
    ├── {puzzle_id}/
    │   ├── log.jsonl         # Detailed events (debugging)
    │   ├── steps.jsonl       # Clean input/output pairs (fine-tuning)
    │   └── screenshots/
    │       ├── step_001.jpg
    │       ├── step_002.jpg
    │       └── ...
    └── ...
```

### File Descriptions

| File | Purpose |
|------|---------|
| `run_config.json` | CLI arguments, LLM configuration, environment info |
| `summary.json` | Total tokens, accuracy, per-puzzle stats |
| `benchmark_results.json` | Final benchmark results for analysis |
| `log.jsonl` | Detailed event stream (puzzle_start, puzzle_end, errors) |
| `steps.jsonl` | Clean LLM call pairs for fine-tuning datasets |
| `screenshots/` | Browser screenshots at each agent step |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      browseruse_cli.py                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ CapturingAsyncClient (OpenAI)                           │   │
│  │ └─> Intercepts HTTP responses                           │   │
│  │ └─> Stores in RawResponseCapture singleton              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ CapturingVLLMAsyncClient (vLLM/Qwen)                    │   │
│  │ └─> Parses </think> tags from response                  │   │
│  │ └─> Extracts JSON for browser-use agent                 │   │
│  │ └─> Stores reasoning in RawVLLMResponseCapture          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ RawGeminiResponseCapture (Gemini)                       │   │
│  │ └─> Captures usage_metadata via monkey-patch            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       llm_logger.py                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  EnhancedLLMLogger                                              │
│  ├─ log_step() ─────────► Routes to provider-specific method   │
│  │   ├─ _log_step_openai()     # OpenAI native format          │
│  │   ├─ _log_step_gemini()     # Gemini native format          │
│  │   ├─ _log_step_anthropic()  # Anthropic native format       │
│  │   └─ _log_step_vllm()       # vLLM native format            │
│  │                                                              │
│  ├─ start_puzzle(puzzle_type, puzzle_id)                        │
│  ├─ end_puzzle(answer, is_correct)                              │
│  ├─ save_screenshot(step, screenshot_data)                      │
│  └─ write_summary()                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Provider-Specific Formats

Each provider uses its **native API field names** for clarity and debugging.

### OpenAI (GPT-5.2, o1, o3)

```json
{
  "call_id": 1,
  "puzzle_step": 1,
  "provider": "openai",
  "model": "gpt-5.2",
  "timing": {
    "start_time": "2026-01-06T14:00:00",
    "end_time": "2026-01-06T14:00:05",
    "duration_ms": 5000
  },
  "request_params": {
    "reasoning_effort": "xhigh",
    "temperature": 0.5,
    "max_completion_tokens": 65536
  },
  "input": {"messages": [...]},
  "output": {...},
  "usage": {
    "prompt_tokens": 1000,
    "completion_tokens": 500,
    "total_tokens": 1500,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 200}
  }
}
```

**Key Fields:**
- `reasoning_effort` - Controls reasoning depth: `none`, `low`, `medium`, `high`, `xhigh`
- `reasoning_tokens` in `completion_tokens_details` - Internal thinking tokens (separate from output)
- `cached_tokens` in `prompt_tokens_details` - Cached input tokens

**Token Counting:**
- `completion_tokens` = visible output tokens only
- `reasoning_tokens` = internal thinking tokens (NOT included in completion_tokens)
- Total output = completion_tokens + reasoning_tokens

### Google Gemini (Gemini 3, 2.5)

```json
{
  "call_id": 1,
  "puzzle_step": 1,
  "provider": "google",
  "model": "gemini-3-flash-preview",
  "timing": {...},
  "request_params": {
    "thinking_budget": 24576,
    "temperature": 0.5,
    "max_output_tokens": 65536
  },
  "input": {"messages": [...]},
  "output": {...},
  "usage": {
    "prompt_token_count": 1000,
    "candidates_token_count": 500,
    "thoughts_token_count": 200,
    "total_token_count": 1700,
    "cached_content_token_count": 0
  }
}
```

**Key Fields:**
- `thoughts_token_count` - Direct field for thinking/reasoning tokens
- `candidates_token_count` - Output tokens (Gemini naming)

### Anthropic (Claude 4.5, Claude 4)

```json
{
  "call_id": 1,
  "puzzle_step": 1,
  "provider": "anthropic",
  "model": "claude-opus-4-5-20251101",
  "timing": {...},
  "request_params": {
    "temperature": 0.5,
    "max_tokens": 65536
  },
  "input": {"messages": [...]},
  "output": {...},
  "usage": {
    "input_tokens": 1000,
    "output_tokens": 700,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  },
  "thinking_content": "Let me analyze this puzzle..."
}
```

**Key Fields:**
- `thinking_content` - Extracted from content blocks with `type: "thinking"`
- Reasoning tokens counted via `tiktoken` (not provided by API)
- **Note:** Anthropic includes thinking tokens IN `output_tokens`

### vLLM/Qwen (Qwen3-VL-8B-Thinking)

```json
{
  "call_id": 1,
  "puzzle_step": 1,
  "provider": "vllm",
  "model": "Qwen/Qwen3-VL-8B-Thinking",
  "timing": {...},
  "request_params": {
    "enable_thinking": true,
    "max_output_tokens": 40960
  },
  "input": {"messages": [...]},
  "output": {...},
  "usage": {
    "prompt_tokens": 8000,
    "completion_tokens": 3500,
    "total_tokens": 11500
  },
  "reasoning_content": "Let me analyze this CAPTCHA puzzle...",
  "reasoning_tokens": 1200
}
```

**Key Fields:**
- `reasoning_content` - Full thinking text (parsed from `</think>` tag)
- `reasoning_tokens` - Counted using Qwen3 VL tokenizer
- **Note:** vLLM outputs `reasoning text</think>{"json": ...}`

## Capture Singletons

The system uses singleton classes to capture raw API responses that would otherwise be hidden by browser-use's wrapper classes.

### RawResponseCapture (OpenAI)

```python
class RawResponseCapture:
    """Thread-safe singleton to store raw OpenAI API responses."""
    _instance = None
    _lock = threading.Lock()

    def store(self, response_json: dict): ...
    def consume(self) -> dict: ...  # Get and clear
```

### RawGeminiResponseCapture (Gemini)

```python
class RawGeminiResponseCapture:
    """Singleton to store raw Gemini usage metadata."""

    def store(self, usage_dict: dict): ...
    def consume(self) -> dict: ...
    def peek(self) -> dict: ...  # Get without clearing
```

### RawVLLMResponseCapture (vLLM/Qwen)

```python
class RawVLLMResponseCapture:
    """Singleton to store vLLM usage and reasoning content."""

    def store(self, usage_dict: dict, reasoning_content: str = None): ...
    def consume(self) -> tuple[dict, str]: ...  # Returns (usage, reasoning)
    def peek(self) -> tuple[dict, str]: ...
```

## Wiring (How Captures Connect to Logger)

In `browseruse_cli.py`, captures are injected into the logger:

```python
# After creating llm_logger instance
llm_logger._raw_openai_capture = RawResponseCapture.get_instance()
llm_logger._raw_gemini_capture = RawGeminiResponseCapture.get_instance()
llm_logger._raw_vllm_capture = RawVLLMResponseCapture.get_instance()
llm_logger._count_tokens_qwen = count_tokens_qwen  # For vLLM token counting
llm_logger._llm_instance = llm  # For accessing request params
```

## Token Counting

### OpenAI
Tokens provided directly by API in `usage` field.

### Gemini
Tokens provided directly by API in `usage_metadata` field.

### Anthropic
- Input/output tokens from API
- **Thinking tokens:** Counted using `tiktoken` (cl100k_base encoding) from `thinking` content blocks
- **Note:** Thinking tokens are INCLUDED in `output_tokens` (not separate)

### vLLM/Qwen
- Input/output tokens from API
- **Reasoning tokens:** Counted using actual Qwen3 VL tokenizer from `transformers` library:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-VL-8B-Thinking",
    trust_remote_code=True
)
token_count = len(tokenizer.encode(reasoning_text))
```

## Qwen Thinking Model Parsing

Qwen3-VL-8B-Thinking outputs in a special format:

```
reasoning text here...
</think>

{"thinking": "...", "action": [...]}
```

**Note:** There's often NO opening `<think>` tag, only a closing `</think>`.

The `CapturingVLLMAsyncClient` handles this:

1. Intercepts HTTP response
2. Splits content on `</think>`
3. Stores reasoning content for logging
4. Modifies response to contain only JSON
5. Passes clean JSON to browser-use agent

## Usage Examples

### Running with Different Providers

```bash
# OpenAI GPT-5.2 (standard)
./test_benchmark.sh --llm openai --model gpt-5.2 --puzzles 'Rotation_Match:3'

# OpenAI GPT-5.2 with xhigh reasoning (IMPORTANT: increase timeouts and max tokens!)
# xhigh can take 30+ minutes per response and generate very long outputs
./test_benchmark.sh --llm openai --model gpt-5.2 \
  --reasoning-effort xhigh \
  --max-output-tokens 65536 \
  --llm-timeout 3600 \
  --step-timeout 3600 \
  --puzzles 'Rotation_Match:3'

# Gemini
./test_benchmark.sh --llm google --model gemini-3-flash-preview --puzzles 'Rotation_Match:3'

# Anthropic
./test_benchmark.sh --llm anthropic --model claude-opus-4-5-20251101 --puzzles 'Rotation_Match:3'

# vLLM/Qwen (thinking enabled by default)
./test_benchmark.sh --llm vllm --model Qwen/Qwen3-VL-8B-Thinking \
  --base-url http://10.127.105.39:8000/v1 --puzzles 'Rotation_Match:3'

# vLLM/Qwen (thinking disabled)
./test_benchmark.sh --llm vllm --model Qwen/Qwen3-VL-8B-Thinking \
  --base-url http://10.127.105.39:8000/v1 --disable-thinking --puzzles 'Rotation_Match:3'
```

### Timeout Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--llm-timeout` | 1800s (30 min) | HTTP client timeout for LLM API calls |
| `--step-timeout` | 1800s (30 min) | Browser-use agent step timeout |

**Important for GPT-5.2 xhigh:** The `xhigh` reasoning effort can take 30+ minutes per response.
Always set both timeouts to at least 3600 seconds (60 minutes) when using xhigh:

```bash
--llm-timeout 3600 --step-timeout 3600
```

### Reasoning Effort (OpenAI GPT-5.2+)

```bash
--reasoning-effort <level>
```

| Level | Description | Typical Response Time |
|-------|-------------|----------------------|
| `none` | No reasoning | Fast |
| `low` | Minimal reasoning | Fast |
| `medium` | Moderate reasoning | ~1-5 minutes |
| `high` | Thorough reasoning | ~5-15 minutes |
| `xhigh` | Maximum reasoning | **30+ minutes** |

### Verifying Request Parameters

When running with `--reasoning-effort`, the CLI logs whether the parameter is being sent:

```
[RAW REQUEST] ✅ reasoning_effort FOUND in request!
[RAW REQUEST] Value: xhigh
```

If you see this warning, the experiments may be invalid:
```
[RAW REQUEST] ❌ reasoning_effort NOT in request!
[RAW REQUEST] ⚠️  YOUR EXPERIMENTS MAY BE INVALID!
```

### Viewing Logs

```bash
# List runs
ls llm_logs/

# View run config
cat llm_logs/2026-01-06_14-05-20_vllm_Qwen/Qwen3-VL-8B-Thinking/run_config.json

# View summary
cat llm_logs/2026-01-06_14-05-20_vllm_Qwen/Qwen3-VL-8B-Thinking/summary.json

# View steps for a puzzle (one JSON per line)
cat llm_logs/2026-01-06_14-05-20_vllm_Qwen/Qwen3-VL-8B-Thinking/puzzles/rotation_match_0/steps.jsonl

# Pretty print a single step
head -1 llm_logs/.../puzzles/rotation_match_0/steps.jsonl | python3 -m json.tool
```

### Analyzing Token Usage

```python
import json

# Load steps
with open('llm_logs/.../puzzles/rotation_match_0/steps.jsonl') as f:
    steps = [json.loads(line) for line in f]

# Analyze by provider
for step in steps:
    provider = step.get('provider')
    usage = step.get('usage', {})

    if provider == 'google':
        print(f"Input: {usage.get('prompt_token_count')}")
        print(f"Output: {usage.get('candidates_token_count')}")
        print(f"Thinking: {usage.get('thoughts_token_count')}")
    elif provider == 'vllm':
        print(f"Input: {usage.get('prompt_tokens')}")
        print(f"Output: {usage.get('completion_tokens')}")
        print(f"Reasoning: {step.get('reasoning_tokens')}")
```

## Summary.json Format

```json
{
  "run_id": "2026-01-06_14-05-20_vllm_Qwen/Qwen3-VL-8B-Thinking",
  "total_puzzles": 3,
  "correct_puzzles": 2,
  "accuracy": 0.667,
  "total_llm_calls": 15,
  "total_input_tokens": 25000,
  "total_output_tokens": 10000,
  "total_reasoning_tokens": 5000,
  "puzzles": [
    {
      "puzzle_number": 1,
      "puzzle_type": "Rotation_Match",
      "puzzle_id": "rotation_match_0",
      "is_correct": true,
      "duration_seconds": 45.2,
      "steps": 5,
      "input_tokens": 8000,
      "output_tokens": 3500,
      "reasoning_tokens": 1500
    }
  ]
}
```

## Early Call Buffering

LLM calls that occur before puzzle detection are buffered and flushed when `start_puzzle()` is called:

- `_early_call_buffer` - Buffered log.jsonl entries
- `_early_step_buffer` - Buffered steps.jsonl entries
- `_early_screenshot_buffer` - Buffered screenshots

This ensures all data is correctly associated with puzzles even when the agent makes initial calls before the puzzle is detected.
