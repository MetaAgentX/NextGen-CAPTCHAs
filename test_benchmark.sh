#!/bin/bash

# Unified CAPTCHA Benchmark Test Script
# Supports: OpenAI (GPT-5, GPT-4o), Google (Gemini), Anthropic (Claude), Qwen, Doubao
#
# To run while MacBook sleeps:
#   caffeinate -s ./test_benchmark.sh --headless
#
# To run in background (survives terminal close):
#   caffeinate -s nohup ./test_benchmark.sh --headless > test_output.log 2>&1 &
#   tail -f test_output.log  # to monitor progress

set -e

# Load .env file if it exists (source API keys)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    # Export variables from .env, skipping comments and empty lines
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Default values
LLM_PROVIDER=""
MODEL=""
REASONING_EFFORT=""  # For GPT-5.2+ xhigh reasoning
MAX_OUTPUT_TOKENS=""  # For Gemini thinking models
THINKING_BUDGET=""  # For Gemini thinking models (e.g., 24576 for max thinking)
ANTHROPIC_THINKING_BUDGET=""  # For Claude extended thinking (min 1024)
ANTHROPIC_EFFORT=""  # For Claude Opus 4.5 effort control (low/medium/high)
LLM_TIMEOUT=""  # LLM call timeout (default 1800s in CLI)
STEP_TIMEOUT=""  # Step timeout (default 1800s in CLI)
BASE_URL=""  # Custom API base URL (for vllm provider)
API_KEY=""  # API key (for vllm provider)
DEBUG_VLLM=""  # Debug flag for vLLM/Qwen thinking models
SEED=""  # Random seed for reproducibility (default: auto-generated from puzzle type and index)
MAX_STEPS=50
MAX_FAILURES=1  # One-shot: stop after 1 failure per puzzle
MAX_RETRIES=3   # Retry puzzle up to 3 times if API error (no result recorded)
HEADLESS=""
VERBOSE="--verbose"
SERVER_PORT=7860  # Server port (use different ports for parallel runs)
SERVER_HOST="127.0.0.1"
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
# Results file - will be timestamped to avoid overwriting
RESULTS_FILE=""  # Set after TIMESTAMP is generated
CROSS_PUZZLE_MEMORY="false"  # Default: isolate puzzles for fair benchmarking
SEQUENTIAL_MODE="false"
START_SERVER="true"  # Whether to start server (set to false if already running)

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Provider configurations (bash 3.2 compatible)

declare -a PUZZLE_CONFIG=()

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "LLM Provider Options:"
    echo "  --llm <provider>        openai, google, anthropic, qwen, doubao, vllm"
    echo "  --model <name>          Model name (defaults vary by provider)"
    echo "  --base-url <url>        Custom API base URL (required for vllm, optional for qwen/doubao)"
    echo "  --api-key <key>         API key (use 'EMPTY' for local vLLM)"
    echo "  --reasoning-effort <l>  Reasoning effort for GPT-5.2+ (none, low, medium, high, xhigh)"
    echo "  --max-output-tokens <n> Max output tokens for Gemini thinking models (e.g., 65536)"
    echo "  --thinking-budget <n>   Gemini 2.5 thinking budget (0-32768 tokens, -1 for dynamic)"
    echo "  --thinking-level <l>    Gemini 3 thinking level (minimal, low, medium, high)"
    echo "  --anthropic-thinking-budget <n>  Claude extended thinking budget (min 1024, e.g., 10000)"
    echo "  --anthropic-effort <l>  Claude Opus 4.5 effort level (low, medium, high)"
    echo "  --llm-timeout <sec>     LLM call timeout in seconds (default: 1800)"
    echo "  --step-timeout <sec>    Step timeout in seconds (default: 1800)"
    echo "  --debug-vllm            Enable debug output for vLLM/Qwen thinking models"
    echo ""
    echo "Puzzle Options:"
    echo "  --puzzles <config>      Format: type1:count1,type2:count2,..."
    echo "                          Example: --puzzles 'Dice_Count:10,Mirror:5'"
    echo "                          Default: all types with 20 puzzles each"
    echo ""
    echo "  --sequential <limit>    Use mode=sequential: single agent, server cycles through types"
    echo "                          <limit> = total puzzles. Example: --sequential 300"
    echo ""
    echo "Memory Options:"
    echo "  --isolate-puzzles       Fresh agent for EACH puzzle (no cross-puzzle memory, fair benchmarking)"
    echo "                          Agent still has full memory within each puzzle attempt"
    echo "  --persistent-agent      Agent persists and keeps context across puzzles within each type"
    echo "                          Alias: --with-memory"
    echo "  --seed <n>              Random seed for reproducibility (default: auto-generated)"
    echo ""
    echo "Other Options:"
    echo "  --max-steps <n>         Max steps per puzzle (default: 50)"
    echo "  --max-failures <n>      Max failures before stop (default: 1)"
    echo "  --headless              Run browser headless"
    echo "  --quiet                 Disable verbose output"
    echo "  --port <n>              Server port (default: 7860, use different ports for parallel runs)"
    echo "  --no-server             Don't start server (use if server already running)"
    echo "  --help                  Show this help"
    echo ""
    echo "Parallel Testing:"
    echo "  Run multiple benchmarks in parallel using different ports:"
    echo "  Terminal 1: $0 --llm google --port 7860 --puzzles 'Dice_Count:10'"
    echo "  Terminal 2: $0 --llm openai --port 7861 --puzzles 'Mirror:10'"
    echo ""
    echo "Examples:"
    echo "  $0 --llm openai                                    # GPT-5, all puzzles"
    echo "  $0 --llm google --model gemini-2.5-flash           # Gemini Flash"
    echo "  $0 --llm anthropic --model claude-opus-4-5-20251101 --anthropic-thinking-budget 10000 --anthropic-effort high --max-output-tokens 16000"
    echo "  $0 --llm vllm --model Qwen/Qwen3-VL-2B-Instruct --base-url http://10.127.105.39:8000/v1"
    echo "  $0 --puzzles 'Dice_Count:10,Mirror:5' --isolate-puzzles"
    echo "  $0 --sequential 300 --llm anthropic"
}

ALL_PUZZLE_TYPES=(
    "3D_Viewpoint" "Audio_Match" "Audio_Video_Alignment" "Backmost_Layer"
    "Box_Folding" "Color_Cipher" "Color_Counting" "Dice_Count"
    "Dice_Roll_Path" "Dynamic_Jigsaw" "Global_Phase_Drift" "Hole_Counting"
    "Illusion_Count" "Illusion_Grid" "Illusion_Order" "Illusion_Type"
    "Illusory_Ribbons" "Layered_Stack" "Map_Parity" "Mirror"
    "Multi_Script" "Occluded_Pattern_Counting" "Red_Dot" "Rhythm"
    "Rotation_Match" "Set_Game" "Shadow_Direction" "Shadow_Plausible"
    "Spooky_Circle" "Spooky_Circle_Grid" "Spooky_Jigsaw" "Spooky_Shape_Grid"
    "Spooky_Size" "Spooky_Text" "Squiggle" "Static_Jigsaw"
    "Storyboard_Logic" "Structure_From_Motion" "Subway_Paths"
    "Temporal_Object_Continuity" "Trajectory_Recovery" "Transform_Pipeline"
)

# Parse puzzle config into indices - outputs space-separated indices to stdout
# Supports: "20" (first 20), "[1:5]+[10:15]" (ranges joined by +), "[0:5]" (single range)
# Usage: puzzle_indices=($(parse_puzzle_indices "$pspec"))
parse_puzzle_indices() {
    local config="$1"
    local result=""

    # Check if it's a simple number (e.g., "20")
    if [[ "$config" =~ ^[0-9]+$ ]]; then
        for ((i=0; i<config; i++)); do
            result="$result $i"
        done
        echo $result
        return
    fi

    # Parse range format: [start:end]+[start:end]+...
    # Remove brackets, split by +
    local ranges="${config//[\[\]]/}"
    local OLD_IFS="$IFS"
    IFS='+' read -ra range_parts <<< "$ranges"
    IFS="$OLD_IFS"

    for range in "${range_parts[@]}"; do
        if [[ "$range" =~ ^([0-9]+):([0-9]+)$ ]]; then
            local start="${BASH_REMATCH[1]}"
            local end="${BASH_REMATCH[2]}"
            for ((i=start; i<end; i++)); do
                result="$result $i"
            done
        fi
    done
    echo $result
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --puzzles)
            IFS=',' read -ra PUZZLE_CONFIG <<< "$2"
            shift 2
            ;;
        --sequential)
            SEQUENTIAL_MODE="true"
            SEQUENTIAL_LIMIT="$2"
            shift 2
            ;;
        --isolate-puzzles)
            CROSS_PUZZLE_MEMORY="false"
            shift
            ;;
        --with-memory|--persistent-agent)
            CROSS_PUZZLE_MEMORY="true"
            shift
            ;;
        --llm)
            LLM_PROVIDER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --reasoning-effort)
            REASONING_EFFORT="$2"
            shift 2
            ;;
        --max-output-tokens)
            MAX_OUTPUT_TOKENS="$2"
            shift 2
            ;;
        --thinking-budget)
            THINKING_BUDGET="$2"
            shift 2
            ;;
        --thinking-level)
            THINKING_LEVEL="$2"
            shift 2
            ;;
        --anthropic-thinking-budget)
            ANTHROPIC_THINKING_BUDGET="$2"
            shift 2
            ;;
        --anthropic-effort)
            ANTHROPIC_EFFORT="$2"
            shift 2
            ;;
        --llm-timeout)
            LLM_TIMEOUT="$2"
            shift 2
            ;;
        --step-timeout)
            STEP_TIMEOUT="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --debug-vllm)
            DEBUG_VLLM="--debug-vllm"
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --max-failures)
            MAX_FAILURES="$2"
            shift 2
            ;;
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        --quiet)
            VERBOSE=""
            shift
            ;;
        --port)
            SERVER_PORT="$2"
            SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
            shift 2
            ;;
        --no-server)
            START_SERVER="false"
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Require --llm to be specified
if [ -z "$LLM_PROVIDER" ]; then
    echo -e "${RED}ERROR: --llm is required${NC}"
    echo ""
    echo "Usage: $0 --llm <provider> [OPTIONS]"
    echo ""
    echo "Providers: openai, google, anthropic, qwen, doubao, vllm"
    echo ""
    echo "Examples:"
    echo "  $0 --llm openai"
    echo "  $0 --llm google --model gemini-2.5-flash"
    echo "  $0 --llm anthropic --puzzles 'Dice_Count:10'"
    echo "  $0 --llm qwen --model qwen3-vl-plus-2025-12-19"
    echo "  $0 --llm doubao --model doubao-seed-1-8-251228"
    echo "  $0 --llm vllm --model Qwen/Qwen3-VL-2B-Instruct --base-url http://10.127.105.39:8000/v1"
    exit 1
fi

# Validate provider and check API key (bash 3.2 compatible)
case "$LLM_PROVIDER" in
    openai)
        if [ -z "$OPENAI_API_KEY" ]; then
            echo -e "${RED}ERROR: OPENAI_API_KEY not set${NC}"
            echo -e "Set your API key: export OPENAI_API_KEY='your-key'"
            exit 1
        fi
        [ -z "$MODEL" ] && MODEL="gpt-5-2025-08-07"
        ;;
    google)
        if [ -z "$GOOGLE_API_KEY" ]; then
            echo -e "${RED}ERROR: GOOGLE_API_KEY not set${NC}"
            echo -e "Set your API key: export GOOGLE_API_KEY='your-key'"
            exit 1
        fi
        [ -z "$MODEL" ] && MODEL="gemini-2.5-pro"
        ;;
    anthropic)
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo -e "${RED}ERROR: ANTHROPIC_API_KEY not set${NC}"
            echo -e "Set your API key: export ANTHROPIC_API_KEY='your-key'"
            exit 1
        fi
        [ -z "$MODEL" ] && MODEL="claude-sonnet-4-20250514"
        ;;
    vllm)
        if [ -z "$MODEL" ]; then
            echo -e "${RED}ERROR: --model is required for vllm provider${NC}"
            echo -e "Example: --model Qwen/Qwen3-VL-2B-Instruct"
            exit 1
        fi
        if [ -z "$BASE_URL" ]; then
            echo -e "${RED}ERROR: --base-url is required for vllm provider${NC}"
            echo -e "Example: --base-url http://10.127.105.39:8000/v1"
            exit 1
        fi
        [ -z "$API_KEY" ] && API_KEY="EMPTY"
        ;;
    qwen)
        if [ -z "$DASHSCOPE_API_KEY" ]; then
            echo -e "${RED}ERROR: DASHSCOPE_API_KEY not set${NC}"
            echo -e "Set your API key: export DASHSCOPE_API_KEY='your-key'"
            echo -e "Get your key from: https://dashscope.console.aliyun.com/"
            exit 1
        fi
        [ -z "$MODEL" ] && MODEL="qwen3-vl-plus-2025-12-19"
        ;;
    doubao)
        if [ -z "$ARK_API_KEY" ]; then
            echo -e "${RED}ERROR: ARK_API_KEY not set${NC}"
            echo -e "Set your API key: export ARK_API_KEY='your-key'"
            echo -e "Get your key from: https://console.volcengine.com/ark/"
            exit 1
        fi
        [ -z "$MODEL" ] && MODEL="doubao-seed-1-8-251228"
        ;;
    *)
        echo -e "${RED}ERROR: Unknown provider '$LLM_PROVIDER'${NC}"
        echo "Supported: openai, google, anthropic, qwen, doubao, vllm"
        exit 1
        ;;
esac

# Skip puzzle config if using sequential mode
if [ "$SEQUENTIAL_MODE" = "false" ]; then
    # Default: all types with 20 each
    if [ ${#PUZZLE_CONFIG[@]} -eq 0 ]; then
        for ptype in "${ALL_PUZZLE_TYPES[@]}"; do
            PUZZLE_CONFIG+=("${ptype}:20")
        done
    fi

    # Validate config
    TOTAL_PUZZLES=0
    TOTAL_TYPES=0

    for config in "${PUZZLE_CONFIG[@]}"; do
        IFS=':' read -r ptype pcount <<< "$config"

        valid=false
        for valid_type in "${ALL_PUZZLE_TYPES[@]}"; do
            if [ "$ptype" == "$valid_type" ]; then
                valid=true
                break
            fi
        done

        if [ "$valid" = false ]; then
            echo -e "${RED}Error: Unknown puzzle type '$ptype'${NC}"
            exit 1
        fi

        # Validate puzzle specification (simple count or range syntax)
        if ! [[ "$pcount" =~ ^[0-9]+$ ]] && ! [[ "$pcount" =~ ^\[.*\]$ ]]; then
            echo -e "${RED}Error: Invalid puzzle specification '$pcount' for '$ptype'${NC}"
            echo -e "${RED}Use either a number (e.g., '20') or range syntax (e.g., '[0:5]+[10:15]')${NC}"
            exit 1
        fi

        # Count puzzles from specification
        puzzle_count_arr=($(parse_puzzle_indices "$pcount"))
        TOTAL_PUZZLES=$((TOTAL_PUZZLES + ${#puzzle_count_arr[@]}))
        TOTAL_TYPES=$((TOTAL_TYPES + 1))
    done
fi

# Header
echo "=========================================="
echo -e "${BLUE}CAPTCHA Benchmark - Unified Test${NC}"
echo "=========================================="
echo ""
echo -e "Provider:      ${GREEN}$LLM_PROVIDER${NC}"
echo -e "Model:         ${GREEN}$MODEL${NC}"

if [ "$SEQUENTIAL_MODE" = "true" ]; then
    echo -e "Mode:          ${GREEN}SEQUENTIAL${NC}"
    echo -e "Total Puzzles: ${GREEN}$SEQUENTIAL_LIMIT${NC}"
else
    echo -e "Memory Mode:   ${GREEN}$([ "$CROSS_PUZZLE_MEMORY" = "true" ] && echo "WITH memory" || echo "NO memory (one-shot)")${NC}"
    echo -e "Total Types:   ${GREEN}$TOTAL_TYPES${NC}"
    echo -e "Total Puzzles: ${GREEN}$TOTAL_PUZZLES${NC}"
    echo ""
    echo "Puzzle Configuration:"
    for config in "${PUZZLE_CONFIG[@]}"; do
        IFS=':' read -r ptype pcount <<< "$config"
        echo "  - $ptype: $pcount"
    done
fi
echo ""

# Generate timestamp first (needed for results file naming)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
START_TIME=$(date +%s)

# Results file handling - use port-specific names for parallel runs
# This allows multiple parallel runs on different ports without file conflicts
if [ "$SERVER_PORT" != "7860" ]; then
    INTERNAL_RESULTS_FILE="benchmark_results_port${SERVER_PORT}.json"
else
    INTERNAL_RESULTS_FILE="benchmark_results.json"
fi
RESULTS_FILE="benchmark_results_${TIMESTAMP}.json"

# Start or check server
SERVER_PID=""
if [ "$START_SERVER" = "true" ]; then
    # Kill any existing server on this port to ensure correct --results-file is used
    if curl -s "$SERVER_URL" > /dev/null 2>&1; then
        echo -e "${YELLOW}Stopping existing server on port $SERVER_PORT to ensure correct results file...${NC}"
        # Find and kill the process using this port
        EXISTING_PID=$(lsof -ti :$SERVER_PORT 2>/dev/null || true)
        if [ -n "$EXISTING_PID" ]; then
            kill $EXISTING_PID 2>/dev/null || true
            sleep 2
        fi
    fi

    echo -e "${BLUE}Starting server on port $SERVER_PORT...${NC}"
    uv run app.py --port "$SERVER_PORT" --results-file "$INTERNAL_RESULTS_FILE" &
    SERVER_PID=$!
    # Wait for server to start
    for i in {1..30}; do
        if curl -s "$SERVER_URL" > /dev/null 2>&1; then
            echo -e "${GREEN}Server started (PID: $SERVER_PID)${NC}"
            break
        fi
        sleep 1
    done
    if ! curl -s "$SERVER_URL" > /dev/null 2>&1; then
        echo -e "${RED}ERROR: Server failed to start on port $SERVER_PORT${NC}"
        exit 1
    fi
else
    # --no-server mode: just check if server is running
    if ! curl -s "$SERVER_URL" > /dev/null 2>&1; then
        echo -e "${RED}ERROR: Server not running at $SERVER_URL${NC}"
        echo "Start with: uv run app.py --port $SERVER_PORT"
        exit 1
    fi
    echo -e "${GREEN}Using existing server at $SERVER_URL${NC}"
fi
echo ""

# Backup any existing results file before starting new run (preserves interrupted runs)
if [ -f "$INTERNAL_RESULTS_FILE" ]; then
    BACKUP_FILE="${INTERNAL_RESULTS_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
    mv "$INTERNAL_RESULTS_FILE" "$BACKUP_FILE"
    echo -e "${YELLOW}Backed up previous incomplete results to: $BACKUP_FILE${NC}"
fi
touch "$INTERNAL_RESULTS_FILE"
echo -e "${BLUE}Results will be saved to: llm_logs/${TIMESTAMP}_${LLM_PROVIDER}_${MODEL}/benchmark_results.json${NC}"
echo -e "${BLUE}Server port: $SERVER_PORT${NC}"

echo "Starting at $(date)"
echo "=========================================="
echo ""

# Build reasoning effort flag if set
REASONING_EFFORT_FLAG=""
if [ -n "$REASONING_EFFORT" ]; then
    REASONING_EFFORT_FLAG="--reasoning-effort $REASONING_EFFORT"
    echo -e "${BLUE}Reasoning effort: $REASONING_EFFORT${NC}"
fi

# Build max output tokens flag if set (for Gemini thinking models)
MAX_OUTPUT_TOKENS_FLAG=""
if [ -n "$MAX_OUTPUT_TOKENS" ]; then
    MAX_OUTPUT_TOKENS_FLAG="--max-output-tokens $MAX_OUTPUT_TOKENS"
    echo -e "${BLUE}Max output tokens: $MAX_OUTPUT_TOKENS${NC}"
fi

# Build thinking budget flag if set (Gemini 2.5)
THINKING_BUDGET_FLAG=""
if [ -n "$THINKING_BUDGET" ]; then
    THINKING_BUDGET_FLAG="--thinking-budget $THINKING_BUDGET"
    echo -e "${BLUE}Thinking budget: $THINKING_BUDGET${NC}"
fi

# Build thinking level flag if set (Gemini 3)
THINKING_LEVEL_FLAG=""
if [ -n "$THINKING_LEVEL" ]; then
    THINKING_LEVEL_FLAG="--thinking-level $THINKING_LEVEL"
    echo -e "${BLUE}Thinking level: $THINKING_LEVEL${NC}"
fi

# Build Anthropic thinking budget flag if set (Claude 4.5)
ANTHROPIC_THINKING_BUDGET_FLAG=""
if [ -n "$ANTHROPIC_THINKING_BUDGET" ]; then
    ANTHROPIC_THINKING_BUDGET_FLAG="--anthropic-thinking-budget $ANTHROPIC_THINKING_BUDGET"
    echo -e "${BLUE}Anthropic thinking budget: $ANTHROPIC_THINKING_BUDGET${NC}"
fi

# Build Anthropic effort flag if set (Claude Opus 4.5 only)
ANTHROPIC_EFFORT_FLAG=""
if [ -n "$ANTHROPIC_EFFORT" ]; then
    ANTHROPIC_EFFORT_FLAG="--anthropic-effort $ANTHROPIC_EFFORT"
    echo -e "${BLUE}Anthropic effort: $ANTHROPIC_EFFORT${NC}"
fi

# Build LLM timeout flag if set
LLM_TIMEOUT_FLAG=""
if [ -n "$LLM_TIMEOUT" ]; then
    LLM_TIMEOUT_FLAG="--llm-timeout $LLM_TIMEOUT"
    echo -e "${BLUE}LLM timeout: ${LLM_TIMEOUT}s${NC}"
fi

# Build step timeout flag if set
STEP_TIMEOUT_FLAG=""
if [ -n "$STEP_TIMEOUT" ]; then
    STEP_TIMEOUT_FLAG="--step-timeout $STEP_TIMEOUT"
    echo -e "${BLUE}Step timeout: ${STEP_TIMEOUT}s${NC}"
fi

# Build base URL flag if set (for vllm provider)
BASE_URL_FLAG=""
if [ -n "$BASE_URL" ]; then
    BASE_URL_FLAG="--base-url $BASE_URL"
    echo -e "${BLUE}Base URL: $BASE_URL${NC}"
fi

# Build API key flag if set (for vllm provider)
API_KEY_FLAG=""
if [ -n "$API_KEY" ]; then
    API_KEY_FLAG="--api-key $API_KEY"
fi

# Function to run a puzzle type with memory (batch mode)
run_puzzle_type() {
    local ptype=$1
    local pcount=$2
    local session_id=$3

    echo -e "${YELLOW}>>> Testing: $ptype ($pcount puzzles)${NC}"

    local url="${SERVER_URL}/?type=${ptype}&continue_active=true&session_id=${session_id}"

    uv run agent_frameworks/browseruse_cli.py \
        --url "$url" \
        --llm "$LLM_PROVIDER" \
        --model "$MODEL" \
        --limit "$pcount" \
        --max-steps "$MAX_STEPS" \
        --max-failures "$MAX_FAILURES" \
        --run-id "$TIMESTAMP" \
        $REASONING_EFFORT_FLAG \
        $MAX_OUTPUT_TOKENS_FLAG \
        $THINKING_BUDGET_FLAG \
        $THINKING_LEVEL_FLAG \
        $ANTHROPIC_THINKING_BUDGET_FLAG \
        $ANTHROPIC_EFFORT_FLAG \
        $LLM_TIMEOUT_FLAG \
        $STEP_TIMEOUT_FLAG \
        $BASE_URL_FLAG \
        $API_KEY_FLAG \
        $DEBUG_VLLM \
        $HEADLESS \
        $VERBOSE 2>&1 | tee -a "test_log_${TIMESTAMP}.txt" || true

    echo ""
}

# Function to run a single puzzle (for isolate-puzzles mode)
run_single_puzzle() {
    local ptype=$1
    local puzzle_idx=$2
    local total=$3
    local session_id=$4
    local progress=${5:-$((puzzle_idx + 1))}  # Default for backwards compatibility

    echo -e "  ${YELLOW}[$progress/$total]${NC} $ptype puzzle $puzzle_idx"

    # Use provided seed or generate deterministic seed based on puzzle type and index
    local seed
    if [ -n "$SEED" ]; then
        seed="$SEED"
    else
        local seed_str="${ptype}_${puzzle_idx}"
        seed=$(echo -n "$seed_str" | md5sum | tr -d -c '0-9' | head -c 9)
    fi

    local url="${SERVER_URL}/?type=${ptype}&puzzle_index=${puzzle_idx}&session_id=${session_id}&seed=${seed}"

    uv run agent_frameworks/browseruse_cli.py \
        --url "$url" \
        --llm "$LLM_PROVIDER" \
        --model "$MODEL" \
        --limit 1 \
        --max-steps "$MAX_STEPS" \
        --max-failures "$MAX_FAILURES" \
        --isolate-puzzles \
        --run-id "$TIMESTAMP" \
        $REASONING_EFFORT_FLAG \
        $MAX_OUTPUT_TOKENS_FLAG \
        $THINKING_BUDGET_FLAG \
        $THINKING_LEVEL_FLAG \
        $ANTHROPIC_THINKING_BUDGET_FLAG \
        $ANTHROPIC_EFFORT_FLAG \
        $LLM_TIMEOUT_FLAG \
        $STEP_TIMEOUT_FLAG \
        $BASE_URL_FLAG \
        $API_KEY_FLAG \
        $DEBUG_VLLM \
        $HEADLESS \
        $VERBOSE 2>&1 | tee -a "test_log_${TIMESTAMP}.txt" || true
}

# Main execution
if [ "$SEQUENTIAL_MODE" = "true" ]; then
    echo -e "${BLUE}Mode: SEQUENTIAL${NC}"
    echo "Single agent run, server cycles through all types"
    echo "Total puzzles: $SEQUENTIAL_LIMIT"
    echo ""

    uv run agent_frameworks/browseruse_cli.py \
        --url "${SERVER_URL}/?mode=sequential" \
        --llm "$LLM_PROVIDER" \
        --model "$MODEL" \
        --limit "$SEQUENTIAL_LIMIT" \
        --max-steps "$MAX_STEPS" \
        --max-failures "$MAX_FAILURES" \
        --run-id "$TIMESTAMP" \
        $REASONING_EFFORT_FLAG \
        $MAX_OUTPUT_TOKENS_FLAG \
        $THINKING_BUDGET_FLAG \
        $THINKING_LEVEL_FLAG \
        $ANTHROPIC_THINKING_BUDGET_FLAG \
        $ANTHROPIC_EFFORT_FLAG \
        $LLM_TIMEOUT_FLAG \
        $STEP_TIMEOUT_FLAG \
        $BASE_URL_FLAG \
        $API_KEY_FLAG \
        $DEBUG_VLLM \
        $HEADLESS \
        $VERBOSE 2>&1 | tee -a "test_log_${TIMESTAMP}.txt" || true

elif [ "$CROSS_PUZZLE_MEMORY" = "true" ]; then
    echo -e "${BLUE}Mode: WITH cross-puzzle memory${NC}"
    echo "Agent keeps context across puzzles within each type"
    echo ""
    SESSION_ID="session_${TIMESTAMP}"

    for config in "${PUZZLE_CONFIG[@]}"; do
        IFS=':' read -r ptype pcount <<< "$config"
        run_puzzle_type "$ptype" "$pcount" "$SESSION_ID"
    done
else
    echo -e "${BLUE}Mode: NO memory (fresh agent per puzzle)${NC}"
    echo "One-shot testing for fair benchmarking"
    echo ""

    for config in "${PUZZLE_CONFIG[@]}"; do
        IFS=':' read -r ptype pspec <<< "$config"
        SESSION_ID="session_${ptype}_${TIMESTAMP}"

        # Parse indices from specification (supports "20" or "[0:5]+[10:15]")
        puzzle_indices=($(parse_puzzle_indices "$pspec"))
        total_puzzles=${#puzzle_indices[@]}

        echo -e "${YELLOW}>>> Testing: $ptype ($total_puzzles puzzles, fresh agent each)${NC}"

        progress=0
        for idx in "${puzzle_indices[@]}"; do
            ((progress++))
            result_recorded=false

            for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
                count_before=$(wc -l < "$INTERNAL_RESULTS_FILE" 2>/dev/null | tr -d ' ' || echo 0)

                run_single_puzzle "$ptype" "$idx" "$total_puzzles" "$SESSION_ID" "$progress"

                count_after=$(wc -l < "$INTERNAL_RESULTS_FILE" 2>/dev/null | tr -d ' ' || echo 0)

                if [ "$count_after" -gt "$count_before" ]; then
                    result_recorded=true
                    break
                fi

                if [ "$attempt" -lt "$MAX_RETRIES" ]; then
                    echo -e "  ${RED}⚠ No result recorded (API error?). Retry $attempt/$MAX_RETRIES for puzzle $idx${NC}"
                    sleep 2
                fi
            done

            if [ "$result_recorded" = false ]; then
                echo -e "  ${RED}✗ WARNING: Puzzle $idx skipped after $MAX_RETRIES attempts${NC}"
            fi
        done

        echo ""
    done
fi

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo -e "${GREEN}Benchmark Complete!${NC}"
echo "=========================================="
echo ""

# Move results file to LLM logs directory (single location only)
if [ -f "$INTERNAL_RESULTS_FILE" ] && [ -s "$INTERNAL_RESULTS_FILE" ]; then
    LLM_LOG_DIR="llm_logs/${TIMESTAMP}_${LLM_PROVIDER}_${MODEL}"
    if [ -d "$LLM_LOG_DIR" ]; then
        RESULTS_FILE="$LLM_LOG_DIR/benchmark_results.json"
        mv "$INTERNAL_RESULTS_FILE" "$RESULTS_FILE"
        echo -e "${GREEN}Results saved to: $RESULTS_FILE${NC}"
    else
        # Fallback: find the most recent matching log directory
        FALLBACK_DIR=$(ls -td llm_logs/*_${LLM_PROVIDER}_${MODEL} 2>/dev/null | head -1)
        if [ -n "$FALLBACK_DIR" ] && [ -d "$FALLBACK_DIR" ]; then
            RESULTS_FILE="$FALLBACK_DIR/benchmark_results.json"
            mv "$INTERNAL_RESULTS_FILE" "$RESULTS_FILE"
            echo -e "${GREEN}Results saved to: $RESULTS_FILE${NC}"
        else
            # Last resort: keep timestamped file in current directory
            mv "$INTERNAL_RESULTS_FILE" "$RESULTS_FILE"
            echo -e "${YELLOW}Results saved to: $RESULTS_FILE (no LLM log directory found)${NC}"
        fi
    fi
elif [ -f "$INTERNAL_RESULTS_FILE" ]; then
    echo -e "${YELLOW}Warning: Results file exists but is empty${NC}"
    rm -f "$INTERNAL_RESULTS_FILE"
else
    echo -e "${YELLOW}Warning: No results file found${NC}"
fi
echo ""

# Statistics
echo "=========================================="
echo "           STATISTICS SUMMARY"
echo "=========================================="
echo ""
echo "Run Info:"
echo "  Provider: $LLM_PROVIDER"
echo "  Model: $MODEL"
echo "  Memory: $([ "$CROSS_PUZZLE_MEMORY" = "true" ] && echo "With" || echo "Without")"
echo "  Duration: ${TOTAL_TIME}s ($(printf '%02d:%02d:%02d' $((TOTAL_TIME/3600)) $((TOTAL_TIME%3600/60)) $((TOTAL_TIME%60))))"
echo ""

if [ -f "$RESULTS_FILE" ]; then
    echo "Per-Type Results:"
    echo "-----------------------------------------"
    printf "%-28s %7s %7s %9s\n" "Type" "Correct" "Total" "Accuracy"
    echo "-----------------------------------------"

    OVERALL_CORRECT=0
    OVERALL_TOTAL=0

    for config in "${PUZZLE_CONFIG[@]}"; do
        IFS=':' read -r ptype pcount <<< "$config"

        type_total=$(grep -c "\"puzzle_type\": \"$ptype\"" "$RESULTS_FILE" 2>/dev/null | head -1 || echo "0")
        type_correct=$(grep "\"puzzle_type\": \"$ptype\"" "$RESULTS_FILE" 2>/dev/null | grep -c "\"correct\": true" | head -1 || echo "0")
        type_total=${type_total:-0}
        type_correct=${type_correct:-0}

        if [ "$type_total" -gt 0 ] 2>/dev/null; then
            accuracy=$(echo "scale=1; $type_correct * 100 / $type_total" | bc)
            printf "%-28s %7d %7d %8.1f%%\n" "$ptype" "$type_correct" "$type_total" "$accuracy"
            OVERALL_CORRECT=$((OVERALL_CORRECT + type_correct))
            OVERALL_TOTAL=$((OVERALL_TOTAL + type_total))
        else
            printf "%-28s %7s %7s %9s\n" "$ptype" "-" "-" "-"
        fi
    done

    echo "-----------------------------------------"
    if [ "$OVERALL_TOTAL" -gt 0 ]; then
        OVERALL_ACCURACY=$(echo "scale=1; $OVERALL_CORRECT * 100 / $OVERALL_TOTAL" | bc)
        printf "%-28s %7d %7d %8.1f%%\n" "TOTAL" "$OVERALL_CORRECT" "$OVERALL_TOTAL" "$OVERALL_ACCURACY"
    fi
    echo ""
    echo "Summary: $OVERALL_CORRECT / $OVERALL_TOTAL correct (${OVERALL_ACCURACY:-0}%)"

    # Per-puzzle breakdown
    echo ""
    echo "=========================================="
    echo "         PER-PUZZLE BREAKDOWN"
    echo "=========================================="
    echo ""
    printf "%-20s %-30s %8s\n" "Type" "Puzzle ID" "Result"
    echo "-----------------------------------------------------------"

    while IFS= read -r line; do
        # Use head -1 to get only the first match (top-level, not from action_sequence)
        ptype=$(echo "$line" | grep -o '"puzzle_type": "[^"]*"' | head -1 | cut -d'"' -f4)
        pid=$(echo "$line" | grep -o '"puzzle_id": "[^"]*"' | head -1 | cut -d'"' -f4)
        correct=$(echo "$line" | grep -o '"correct": [^,}]*' | head -1 | cut -d' ' -f2)

        if [ -n "$ptype" ] && [ -n "$pid" ]; then
            if [ "$correct" = "true" ]; then
                result="${GREEN}PASS${NC}"
            else
                result="${RED}FAIL${NC}"
            fi
            printf "%-20s %-30s " "$ptype" "$pid"
            echo -e "$result"
        fi
    done < "$RESULTS_FILE"

    echo "-----------------------------------------------------------"

    # Failed puzzles summary
    echo ""
    echo "Failed Puzzles (for difficulty analysis):"
    echo "-----------------------------------------"
    failed_count=0
    while IFS= read -r line; do
        # Use head -1 to get only the first match (top-level, not from action_sequence)
        ptype=$(echo "$line" | grep -o '"puzzle_type": "[^"]*"' | head -1 | cut -d'"' -f4)
        pid=$(echo "$line" | grep -o '"puzzle_id": "[^"]*"' | head -1 | cut -d'"' -f4)
        correct=$(echo "$line" | grep -o '"correct": [^,}]*' | head -1 | cut -d' ' -f2)

        if [ "$correct" = "false" ] && [ -n "$ptype" ] && [ -n "$pid" ]; then
            echo "  - $ptype / $pid"
            failed_count=$((failed_count + 1))
        fi
    done < "$RESULTS_FILE"

    if [ "$failed_count" -eq 0 ]; then
        echo "  (none - all puzzles passed!)"
    fi

else
    echo -e "${YELLOW}Results file not found: $RESULTS_FILE${NC}"
fi

echo ""
echo "Files:"
echo "  Results: $RESULTS_FILE"
echo "  Log: test_log_${TIMESTAMP}.txt"
echo "  LLM Logs: llm_logs/${TIMESTAMP}_${LLM_PROVIDER}_${MODEL}/"
echo "=========================================="

# Cleanup: stop server if we started it
if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo -e "${BLUE}Stopping server (PID: $SERVER_PID)...${NC}"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo -e "${GREEN}Server stopped${NC}"
fi
