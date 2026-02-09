# Benchmarking Reference

Full CLI reference and detailed documentation for running benchmarks on Next-Gen CAPTCHAs. For a quick start, see the [README](README.md#running-benchmarks).

## Table of Contents

- [CLI Reference](#cli-reference)
- [Provider-Specific Examples](#provider-specific-examples)
- [Parallel Benchmarks](#parallel-benchmarks)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Output Files](#output-files)
- [Result Classification](#result-classification)
- [Benchmark Dataset Details](#benchmark-dataset-details)

## CLI Reference

```bash
./test_benchmark.sh [OPTIONS]

LLM Provider Options:
  --llm <provider>        openai, google, anthropic, vllm (required)
  --model <name>          Model name (provider-specific defaults apply)
  --reasoning-effort <l>  OpenAI GPT-5.2+ (none, low, medium, high, xhigh)
  --max-output-tokens <n> Gemini/vLLM thinking models (32768 recommended)
  --thinking-budget <n>   Gemini 2.5 thinking budget (0-32768 tokens, -1 for dynamic)
  --thinking-level <l>    Gemini 3 thinking level (minimal, low, medium, high)

vLLM-Specific Options:
  --base-url <url>        vLLM API endpoint (required for vLLM)
  --api-key <key>         API key (use 'EMPTY' for local vLLM)
  --debug-vllm            Enable debug output for vLLM/Qwen thinking models

Qwen3-VL-8B-Thinking Inference Parameters (Applied Automatically):
  When using --model Qwen/Qwen3-VL-8B-Thinking, official defaults are applied:
  - temperature: 0.6          (Official default for Qwen3 THINKING models)
  - top_p: 0.95               (Official default)
  - top_k: 20                 (Official default, via extra_body)
  - repetition_penalty: 1.0   (Official default, via extra_body)
  - max_tokens: <value>       (Uses max_tokens instead of max_completion_tokens for vLLM)
  Reference: https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html

Puzzle Options:
  --puzzles <config>      Puzzle selection format:
                          - Simple count: type:N (first N puzzles, indices 0 to N-1)
                          - Range syntax: type:[start:end]+[start:end]+...
                          - Mixed: type1:20,type2:[0:5]+[15:20]
                          Examples:
                            'Dice_Roll_Path:20'              # First 20 puzzles (0-19)
                            'Mirror:[5:10]'              # Puzzles 5-9 only
                            'Box_Folding:[0:3]+[10:15]'  # Puzzles 0-2 and 10-14
  --isolate-puzzles       Fresh agent per puzzle (recommended for fair benchmarks)
                          Alias: --no-memory
  --persistent-agent      Agent keeps context across puzzles within each type
                          Alias: --with-memory
  --seed <n>              Random seed for reproducibility (default: 0)

Other Options:
  --max-steps <n>         Max steps per puzzle (default: 200)
  --headless              Run browser headless
  --port <n>              Server port (default: 7860)
```

## Provider-Specific Examples

```bash
# Recommended: Use --isolate-puzzles for fair benchmarking (fresh agent per puzzle)
# --seed 0: Seed for puzzle generation (reproducibility)
# --max-steps 50, --port 7860
# Each agent uses recommended parameters from its LLM provider to test ability limits

# Run with Gemini 3 Pro (thinking model - requires max-output-tokens)
./test_benchmark.sh --llm google --model gemini-3-pro-preview \
    --max-output-tokens 32768 --max-steps 50 --port 7860 \
    --puzzles 'Dice_Roll_Path:10' --isolate-puzzles --seed 0 --headless

# Run with GPT-5.2 (very slow - use longer timeouts)
./test_benchmark.sh --llm openai --model gpt-5.2 \
    --reasoning-effort xhigh --max-steps 50 --port 7860 \
    --llm-timeout 3600 --step-timeout 3600 \
    --puzzles 'Mirror:10' --isolate-puzzles --seed 0 --headless

# Run with Claude Sonnet 4
./test_benchmark.sh --llm anthropic --model claude-sonnet-4-20250514 \
    --max-steps 50 --port 7860 \
    --puzzles 'Box_Folding:5' --isolate-puzzles --seed 0 --headless

# Run with vLLM (Qwen3-VL-8B-Thinking)
./test_benchmark.sh \
    --llm vllm \
    --model Qwen/Qwen3-VL-8B-Thinking \
    --base-url http://YOUR_VLLM_SERVER:8000/v1 \
    --max-output-tokens 65536 --max-steps 50 --port 7860 \
    --puzzles 'Rotation_Match:3' \
    --isolate-puzzles --seed 0 \
    --headless

# Run specific puzzle ranges (e.g., puzzles 0-2 and 10-14)
./test_benchmark.sh --llm google --model gemini-3-flash-preview \
    --max-output-tokens 65536 --max-steps 50 --port 7860 \
    --puzzles 'Box_Folding:[0:3]+[10:15]' --isolate-puzzles --seed 0 --headless
```

## Parallel Benchmarks

Run multiple benchmarks simultaneously using different ports:

```bash
# Terminal 1: Test Gemini 3 on Dice_Roll_Path puzzles
./test_benchmark.sh --llm google --model gemini-3-pro-preview \
    --max-output-tokens 32768 \
    --port 7860 --puzzles 'Dice_Roll_Path:10' --isolate-puzzles --seed 0 --headless

# Terminal 2: Test GPT-5 on Mirror puzzles
./test_benchmark.sh --llm openai --model gpt-5-2025-08-07 \
    --reasoning-effort xhigh \
    --port 7861 --puzzles 'Mirror:10' --isolate-puzzles --seed 0 --headless

# Terminal 3: Test Claude on Box_Folding puzzles
./test_benchmark.sh --llm anthropic --model claude-sonnet-4-20250514 \
    --port 7862 --puzzles 'Box_Folding:10' --isolate-puzzles --seed 0 --headless
```

Each parallel run:
- Starts its own server on the specified port
- Writes to a port-specific results file (`benchmark_results_port7861.json`, etc.)
- Has isolated LLM logs in `llm_logs/` directory
- Automatically stops its server when complete

## Reproducing Paper Results

For closest reproduction of the paper's main results:

- **Default agent framework**: use Browser-Use as the main evaluation framework.
- **Episode protocol**: run one puzzle per episode with state reset between puzzles (`--isolate-puzzles`).
- **Two-tier evaluation (paper protocol)**:
  - **Full 519 puzzles**: `Gemini-3-Pro-High`, `Gemini-3-Flash-High`, `Qwen3-VL-Plus-ThinkingHigh`, `Doubao-Seed-1.8-Thinking-HighEffort`
  - **135-puzzle subset**: `GPT-5.2-xHigh`, `Claude-Opus4.5-Extended-ThinkingHigh`
- **Seed/reproducibility**: use fixed seed (`--seed 0`) for consistent puzzle generation where applicable.
- **Latency/API caveat**: for highest-reasoning settings we use large max-token budgets (no intentional truncation pressure from our side), but remote provider APIs can still intermittently fail. In particular, GPT-5.2-xHigh may occasionally return provider-side HTTP 500/non-truncation failures under long, high-token runs; this is treated as upstream service instability rather than local token-budget exhaustion.

## Output Files

After a benchmark run:
- `benchmark_results_TIMESTAMP.json` - Per-puzzle results (JSONL format)
- `test_log_TIMESTAMP.txt` - Console output log
- `llm_logs/TIMESTAMP_provider_model/` - Detailed LLM interaction logs
  - `summary.json` - Aggregated statistics
  - `puzzles/` - Per-puzzle step logs with screenshots

## Result Classification

Each trial result has two fields:
- `is_correct`: `true` only when a correct answer is submitted; `false` for all other cases
- `user_answer`: The submitted value or an error code indicating why no correct answer was submitted

**When `is_correct: true`**
- Agent submitted the correct answer
- `user_answer` contains the submitted value (e.g., `[6, 7]`, `"yes"`, `3`)

**When `is_correct: false`**

| Reason | `user_answer` |
|--------|---------------|
| Wrong answer submitted | The actual submitted value (e.g., `[1, 2]`, `"no"`, `5`) |
| 3+ consecutive invalid element indices | `NO_SUBMIT_INVALID_INDEX` |
| 3+ consecutive non-clickable elements | `NO_SUBMIT_NOT_CLICKABLE` |
| 3+ consecutive LLM parsing errors | `NO_SUBMIT_FORMAT_ERROR` |
| 3+ consecutive other action errors | `NO_SUBMIT_OTHER_ERROR` |
| Max steps reached, no errors | `NO_SUBMISSION_MAX_STEPS` |
| Agent navigated away from puzzle | `NAVIGATION_FAILURE` |
| Time limit exceeded | `ERROR_TIMEOUT` |
| Config error (missing API key) | `ERROR_VALUE_ERROR` |
| Other exceptions | `ERROR_<TYPE>` |

## Benchmark Dataset Details

### Full Test Set (519 puzzles)

Each CAPTCHA type includes the first 20 puzzles (indices 0-19), except:
- **Mirror**: 11 puzzles (indices 0-10)
- **Shadow_Plausible**: 8 puzzles (indices 0-7)

### Lite Subset (135 puzzles)

A curated subset of 5 puzzles per type for quick evaluation:

| Type | Selected Indices |
|------|------------------|
| 3D_Viewpoint | 2, 4, 7, 14, 17 |
| Backmost_Layer | 2, 3, 10, 15, 16 |
| Box_Folding | 2, 6, 7, 10, 19 |
| Color_Counting | 6, 10, 11, 13, 15 |
| Dice_Roll_Path | 9, 11, 12, 15, 19 |
| Dynamic_Jigsaw | 3, 9, 10, 17, 19 |
| Hole_Counting | 6, 9, 14, 17, 19 |
| Illusory_Ribbons | 2, 7, 9, 10, 12 |
| Layered_Stack | 3, 4, 8, 9, 17 |
| Mirror | 0, 4, 6, 7, 10 |
| Multi_Script | 0, 1, 2, 3, 11 |
| Occluded_Pattern_Counting | 1, 5, 6, 8, 19 |
| Red_Dot | 2, 3, 10, 15, 19 |
| Rotation_Match | 2, 4, 15, 16, 18 |
| Shadow_Direction | 1, 2, 8, 12, 17 |
| Shadow_Plausible | 1, 2, 3, 4, 5 |
| Spooky_Circle | 0, 1, 8, 14, 16 |
| Spooky_Circle_Grid | 6, 7, 8, 13, 16 |
| Spooky_Jigsaw | 2, 10, 11, 14, 15 |
| Spooky_Shape_Grid | 3, 6, 10, 15, 19 |
| Spooky_Size | 0, 2, 10, 12, 15 |
| Spooky_Text | 0, 3, 7, 8, 19 |
| Static_Jigsaw | 1, 5, 10, 11, 13 |
| Structure_From_Motion | 1, 2, 3, 4, 7 |
| Subway_Paths | 3, 5, 6, 15, 19 |
| Temporal_Object_Continuity | 0, 1, 3, 13, 17 |
| Trajectory_Recovery | 0, 3, 6, 12, 18 |
