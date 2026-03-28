"""
Generate one fair, type-level oracle strategy per CAPTCHA family using Gemini.

Design goals:
- Use full task generation code with no truncation.
- Use full related evaluation/runtime code with no truncation.
- Remove instance-level leakage such as ground-truth samples or prompt variants.
- Ask for a single general strategy that matches the real browser-use benchmark.
- Force JSON output with a single `strategy` field.
- Reject strategies that mention hidden internals instead of visible webpage behavior.

Usage:
    python draft/generate_oracle_strategies.py --api-key <GEMINI_API_KEY>
    python draft/generate_oracle_strategies.py --api-key <KEY> --force
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from typing import Iterable

from google import genai
from google.genai import types


TYPE_TO_GENERATOR = {
    "3D_Viewpoint": "generate_3d_viewpoint_matching.py",
    "Backmost_Layer": "generate_backmost_layer.py",
    "Box_Folding": "generate_box_folding.py",
    "Color_Counting": "generate_color_counting_puzzles.py",
    "Dice_Roll_Path": "generate_dice_roll_path.py",
    "Dynamic_Jigsaw": "generate_dynamic_jigsaw.py",
    "Hole_Counting": "generate_hole_counting.py",
    "Illusory_Ribbons": "generate_illusory_ribbons.py",
    "Layered_Stack": "generate_layered_stack.py",
    "Mirror": None,
    "Multi_Script": "generate_multi_script.py",
    "Occluded_Pattern_Counting": "generate_occluded_pattern_counting.py",
    "Red_Dot": None,
    "Rotation_Match": "generate_rotation_match.py",
    "Shadow_Direction": "generate_shadow_direction.py",
    "Shadow_Plausible": "generate_sketches.py",
    "Spooky_Circle": "generate_spooky_circles.py",
    "Spooky_Circle_Grid": "generate_spooky_circle_grid.py",
    "Spooky_Jigsaw": "generate_spooky_jigsaw.py",
    "Spooky_Shape_Grid": "generate_spooky_shape_grid.py",
    "Spooky_Size": "generate_spooky_size_comparison.py",
    "Spooky_Text": "generate_spooky_text.py",
    "Static_Jigsaw": None,
    "Structure_From_Motion": "generate_structure_from_motion.py",
    "Subway_Paths": "generate_subway_paths.py",
    "Temporal_Object_Continuity": "generate_temporal_object_continuity.py",
    "Trajectory_Recovery": "generate_trajectory_recovery.py",
}

TYPE_TO_INPUT_TYPE = {
    "3D_Viewpoint": "viewpoint_select",
    "Backmost_Layer": "backmost_layer_select",
    "Box_Folding": "box_folding_select",
    "Color_Counting": "color_counting_select",
    "Dice_Roll_Path": "number",
    "Dynamic_Jigsaw": "jigsaw_puzzle",
    "Hole_Counting": "hole_counting_select",
    "Illusory_Ribbons": "illusory_ribbons_select",
    "Layered_Stack": "layered_stack_select",
    "Mirror": "mirror_select",
    "Multi_Script": "multi_script_select",
    "Occluded_Pattern_Counting": "dual_number",
    "Red_Dot": "red_dot_click",
    "Rotation_Match": "rotation_match_select",
    "Shadow_Direction": "shadow_direction_select",
    "Shadow_Plausible": "shadow_plausible",
    "Spooky_Circle": "number",
    "Spooky_Circle_Grid": "circle_grid_select",
    "Spooky_Jigsaw": "jigsaw_puzzle",
    "Spooky_Shape_Grid": "shape_grid_select",
    "Spooky_Size": "spooky_size_click",
    "Spooky_Text": "text",
    "Static_Jigsaw": "jigsaw_puzzle",
    "Structure_From_Motion": "structure_from_motion_select",
    "Subway_Paths": "subway_paths_select",
    "Temporal_Object_Continuity": "temporal_continuity_select",
    "Trajectory_Recovery": "trajectory_recovery_select",
}

GRID_INPUT_TYPES = {
    "structure_from_motion_select",
    "circle_grid_select",
    "circle_grid_direction_select",
    "shape_grid_select",
    "color_counting_select",
    "hole_counting_select",
    "rotation_match_select",
    "rhythm_select",
    "backmost_layer_select",
    "shadow_direction_select",
    "global_phase_drift_select",
    "temporal_continuity_select",
    "layered_stack_select",
    "illusory_ribbons_select",
    "subway_paths_select",
    "trajectory_recovery_select",
    "set_game_select",
    "audio_match_select",
    "viewpoint_select",
    "box_folding_select",
    "illusion_grid_select",
    "multi_script_select",
}

INPUT_TYPE_TO_FRONTEND_FUNCTIONS = {
    "number": ["configureNumberPuzzle"],
    "text": ["configureTextPuzzle"],
    "dual_number": ["configureDualNumberPuzzle"],
    "shadow_plausible": ["setupShadowPlausibleGrid"],
    "mirror_select": ["setupMirrorSelect"],
    "squiggle_select": ["setupSquiggleSelect"],
    "color_cipher": ["setupColorCipher"],
    "red_dot_click": ["setupRedDotClick", "finalizeRedDotAttempt", "submitRedDotAttempt"],
    "spooky_size_click": ["setupSpookySizeClick"],
    "storyboard_logic": ["setupStoryboardLogic"],
    "jigsaw_puzzle": ["setupJigsawPuzzle"],
    "transform_pipeline_select": ["setupTransformPipelineSelect"],
    "illusion_order": ["setupIllusionOrder"],
    "illusion_count": ["setupIllusionCount"],
    "map_parity_select": ["setupMapParitySelect"],
}

for input_type in GRID_INPUT_TYPES:
    INPUT_TYPE_TO_FRONTEND_FUNCTIONS[input_type] = ["setupGridSelection"]

INPUT_TYPE_TO_INTERACTION_SUMMARY = {
    "number": "Read the visible prompt and rendered content, then type a single number into the webpage input and submit once.",
    "text": "Read the visible prompt and rendered content, then type the answer into the webpage input and submit once.",
    "dual_number": "Read the visible prompt and rendered content, fill two number inputs on the page, and submit once.",
    "shadow_plausible": "Inspect a grid of images, click every matching image in the webpage UI, then submit once.",
    "mirror_select": "Inspect the reference image and option images, click every matching option in the webpage UI, then submit once.",
    "red_dot_click": "Act directly in the webpage canvas area; click the visible target as it appears. The task auto-submits/continues through the runtime flow instead of using the normal submit button.",
    "spooky_size_click": "Click the target location directly in the webpage canvas area.",
    "jigsaw_puzzle": "Drag puzzle pieces within the webpage and submit the completed arrangement once.",
}

for input_type in GRID_INPUT_TYPES:
    INPUT_TYPE_TO_INTERACTION_SUMMARY[input_type] = (
        "Read the visible prompt, inspect the rendered reference/animation/grid if present, click the matching cells in the webpage UI, then submit once."
    )

PROMPT_VERSION = "v2_full_code_runtime_only_json"
DEFAULT_MODEL = "gemini-3-pro-preview"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.path.join(BASE_DIR, "captcha_generation")
APP_PY = os.path.join(BASE_DIR, "app.py")
BROWSERUSE_CLI = os.path.join(BASE_DIR, "agent_frameworks", "browseruse_cli.py")
SCRIPT_JS = os.path.join(BASE_DIR, "static", "js", "script.js")

JSON_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {
            "type": "string",
            "description": "One concise, general, UI-only solving strategy for this CAPTCHA family.",
        }
    },
    "required": ["strategy"],
    "additionalProperties": False,
}

SPECIAL_GENERATION_FUNCTIONS = {
    "Red_Dot": ["generate_red_dot"],
    "Static_Jigsaw": ["generate_jigsaw_image", "generate_jigsaw_puzzle"],
}

SPECIAL_SERVER_HELPERS = {
    "Dynamic_Jigsaw": [
        "build_served_jigsaw_positions",
        "register_jigsaw_validation",
        "format_jigsaw_correct_positions",
    ],
    "Spooky_Jigsaw": [
        "build_served_jigsaw_positions",
        "register_jigsaw_validation",
        "format_jigsaw_correct_positions",
    ],
    "Static_Jigsaw": [
        "build_served_jigsaw_positions",
        "register_jigsaw_validation",
        "format_jigsaw_correct_positions",
    ],
}

FORBIDDEN_STRATEGY_PATTERNS = [
    (r"\bground[_ -]?truth\b", "ground-truth references"),
    (r"\bshape_data\b", "internal frontend data structures"),
    (r"\breference_shape\b", "internal variable names"),
    (r"\bvalidation_token\b", "internal validation fields"),
    (r"\bpuzzle_id\b", "internal identifiers"),
    (r"\bwindow\.currentPuzzle\b", "hidden page internals"),
    (r"\b0-?indexed\b", "hidden answer formatting"),
    (r"\brow-major\b", "hidden answer formatting"),
    (r"\bJSON array\b", "hidden answer formatting"),
    (r"\bDOM\b", "source-level DOM guidance instead of visible UI behavior"),
    (r"\bsource code\b", "source-code references"),
    (r"\bgenerator code\b", "generator-code references"),
    (r"\bserver\b", "server/internal references"),
    (r"\bfilename\b", "filename references"),
    (r"\bURL\b", "URL references"),
    (r"\bAPI\b", "API references"),
]


def read_file_full(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_python_function_source(path: str, function_name: str) -> str:
    source = read_file_full(path)
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            start = node.lineno - 1
            end = node.end_lineno
            return "\n".join(lines[start:end]).rstrip()
    return f"[FUNCTION {function_name} NOT FOUND IN {os.path.basename(path)}]"


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _extract_python_branch(lines: list[str], start_idx: int) -> str:
    base_indent = _indent_width(lines[start_idx])
    block = [lines[start_idx]]
    idx = start_idx + 1
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if stripped and _indent_width(line) <= base_indent:
            break
        block.append(line)
        idx += 1
    return "\n".join(block).rstrip()


def extract_python_function_branches(path: str, function_name: str, puzzle_type: str) -> str:
    function_source = extract_python_function_source(path, function_name)
    lines = function_source.splitlines()
    pattern = re.compile(
        rf'^\s*(if|elif)\s+puzzle_type\s*==\s*["\']{re.escape(puzzle_type)}["\']\s*:'
    )
    blocks = []
    for idx, line in enumerate(lines):
        if pattern.match(line):
            blocks.append(_extract_python_branch(lines, idx))
    if not blocks:
        return f"[NO {function_name} BRANCHES FOUND FOR {puzzle_type}]"
    return "\n\n".join(unique_preserve_order(blocks))


def extract_javascript_function_source(path: str, function_name: str) -> str:
    source = read_file_full(path)
    match = re.search(rf'function\s+{re.escape(function_name)}\s*\(', source)
    if not match:
        return f"[FUNCTION {function_name} NOT FOUND IN {os.path.basename(path)}]"

    start = match.start()
    brace_start = source.find("{", match.end())
    if brace_start == -1:
        return f"[FUNCTION {function_name} HAS NO BODY IN {os.path.basename(path)}]"

    depth = 0
    for idx in range(brace_start, len(source)):
        char = source[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:idx + 1].rstrip()

    return f"[FUNCTION {function_name} BODY NOT CLOSED IN {os.path.basename(path)}]"


def extract_generation_context(puzzle_type: str) -> str:
    sections = []
    generator_file = TYPE_TO_GENERATOR.get(puzzle_type)
    if generator_file:
        generator_path = os.path.join(GEN_DIR, generator_file)
        sections.append(
            f"### {generator_file}\n```python\n{read_file_full(generator_path)}\n```"
        )

    for helper_name in SPECIAL_GENERATION_FUNCTIONS.get(puzzle_type, []):
        helper_source = extract_python_function_source(APP_PY, helper_name)
        sections.append(f"### app.py::{helper_name}\n```python\n{helper_source}\n```")

    if not sections:
        sections.append(
            "### Generation Note\n"
            "This puzzle type has no standalone generator script mapped in this repository. "
            "Use the runtime/evaluation code below to infer the family-level mechanics."
        )

    return "\n\n".join(sections)


def extract_server_runtime_context(puzzle_type: str) -> str:
    sections = [
        "### app.py::get_puzzle relevant branches",
        f"```python\n{extract_python_function_branches(APP_PY, 'get_puzzle', puzzle_type)}\n```",
        "### app.py::check_answer relevant branches",
        f"```python\n{extract_python_function_branches(APP_PY, 'check_answer', puzzle_type)}\n```",
    ]

    for helper_name in SPECIAL_SERVER_HELPERS.get(puzzle_type, []):
        helper_source = extract_python_function_source(APP_PY, helper_name)
        sections.append(f"### app.py::{helper_name}\n```python\n{helper_source}\n```")

    return "\n\n".join(sections)


def extract_frontend_runtime_context(puzzle_type: str) -> str:
    input_type = TYPE_TO_INPUT_TYPE[puzzle_type]
    function_names = list(INPUT_TYPE_TO_FRONTEND_FUNCTIONS.get(input_type, []))
    function_names.append("submitAnswer")
    function_names = unique_preserve_order(function_names)

    sections = []
    for function_name in function_names:
        source = extract_javascript_function_source(SCRIPT_JS, function_name)
        sections.append(f"### script.js::{function_name}\n```javascript\n{source}\n```")
    return "\n\n".join(sections)


def summarize_visible_features(puzzle_type: str) -> str:
    input_type = TYPE_TO_INPUT_TYPE[puzzle_type]
    get_puzzle_context = extract_python_function_branches(APP_PY, "get_puzzle", puzzle_type)

    feature_lines = [
        f"- Frontend input mode: `{input_type}`.",
        f"- Legitimate interaction pattern: {INPUT_TYPE_TO_INTERACTION_SUMMARY.get(input_type, 'Use the visible webpage controls only.')}",
        "- Instance-specific targets vary per puzzle and must be read from the on-screen prompt and rendered content at runtime.",
        "- Use the code only to infer family-level mechanics, visible UI structure, and real edge cases.",
        "- Never turn hidden code-only names, server fields, or answer formats into instructions.",
    ]

    if '"reference_image"' in get_puzzle_context:
        feature_lines.append("- A visible reference image or reference panel may appear in the webpage UI.")
    if '"option_images"' in get_puzzle_context:
        feature_lines.append("- The webpage may render multiple selectable option images or grid cells.")
    if '"movement_gif"' in get_puzzle_context or '"media_type": \'gif\'' in get_puzzle_context or "media_type = \"gif\"" in get_puzzle_context:
        feature_lines.append("- Animated media may need to be watched over time instead of judged from a single frame.")
    if '"audio_path"' in get_puzzle_context:
        feature_lines.append("- Audio may be part of the rendered task interface.")
    if '"timeout_ms"' in get_puzzle_context:
        feature_lines.append("- This task has time-sensitive runtime behavior.")
    if '"required_hits"' in get_puzzle_context:
        feature_lines.append("- The task may require multiple sequential successful interactions in one run.")
    if '"grid_size"' in get_puzzle_context:
        feature_lines.append("- The webpage may display the task in a grid layout.")

    return "\n".join(feature_lines)


def build_prompt_for_type(puzzle_type: str) -> str:
    browseruse_prompt_code = extract_python_function_source(BROWSERUSE_CLI, "_build_task_prompt")
    generation_context = extract_generation_context(puzzle_type)
    server_runtime_context = extract_server_runtime_context(puzzle_type)
    frontend_runtime_context = extract_frontend_runtime_context(puzzle_type)
    type_summary = summarize_visible_features(puzzle_type)

    return f"""You are writing one fair, general strategy hint for a browser GUI agent in an academic CAPTCHA benchmark.

Task:
- Produce exactly one strategy for the CAPTCHA family "{puzzle_type}".
- The strategy must be general to the puzzle family, not overfit to any single instance.
- The strategy must help an agent solve the puzzle the same legitimate way a strong human would: by reading the on-screen prompt, inspecting the rendered webpage, and interacting with the visible UI.

Hard constraints:
- The agent can see webpage screenshots and rendered UI, and it can click, type, drag, and scroll.
- The agent is evaluated under the real browser-use benchmark prompt and one-shot submission rules shown below.
- Use the code only to infer family-level mechanics, visible UI structure, and real edge cases.
- Do not mention source code, generator code, ground truth, DOM access, server fields, filenames, URLs, JSON answer formats, internal variable names, or hidden identifiers.
- Do not hardcode any instance-specific values such as exact counts, shapes, directions, words, or targets. Say to read the current target from the on-screen prompt.
- Write only in terms of what the agent can visibly read or do on the webpage.

Output contract:
- Return valid JSON only.
- The JSON must have exactly one key: "strategy".
- The value of "strategy" must be plain English text.
- Write a single concise strategy, roughly 180-350 words.

## Type-Level Summary
{type_summary}

## Shared Browser-Use Benchmark Prompt Code
```python
{browseruse_prompt_code}
```

## Full Task Generation Code
{generation_context}

## Full Related Server Evaluation / Runtime Code
{server_runtime_context}

## Full Related Frontend Runtime Code
{frontend_runtime_context}
"""


def extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_strategy_text(strategy: str) -> str:
    if not isinstance(strategy, str):
        raise ValueError("Strategy must be a string.")

    cleaned = strategy.strip()
    if not cleaned:
        raise ValueError("Strategy is empty.")

    word_count = len(cleaned.split())
    if word_count < 60:
        raise ValueError(f"Strategy is too short ({word_count} words).")

    for pattern, reason in FORBIDDEN_STRATEGY_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            raise ValueError(f"Strategy contains forbidden content: {reason}.")

    return cleaned


def parse_strategy_response(response_text: str) -> str:
    payload = extract_json_object(response_text)
    strategy = payload.get("strategy")
    return validate_strategy_text(strategy)


def generate_strategy_for_type(client: genai.Client, model: str, puzzle_type: str) -> str:
    prompt = build_prompt_for_type(puzzle_type)
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_json_schema=JSON_RESPONSE_SCHEMA,
    )

    repair_error = None
    current_prompt = prompt

    for attempt_index in range(2):
        response = client.models.generate_content(
            model=model,
            contents=current_prompt,
            config=config,
        )

        raw_text = (response.text or "").strip()
        try:
            return parse_strategy_response(raw_text)
        except Exception as exc:
            repair_error = exc
            if attempt_index == 1:
                break
            current_prompt = (
                f"{prompt}\n\n"
                "Your previous response violated the output contract.\n"
                f"Validation error: {exc}\n"
                "Return corrected JSON only, with exactly one key: \"strategy\".\n"
                "Do not mention hidden internals or answer formats.\n"
                f"Previous response:\n{raw_text}"
            )

    raise ValueError(f"Failed to obtain a valid strategy: {repair_error}")


def normalize_strategy_record(record: dict, model: str) -> dict:
    if not isinstance(record, dict):
        return {
            "strategy": str(record).strip(),
            "model_used": model,
            "prompt_version": "legacy",
        }

    return {
        "strategy": str(record.get("strategy", "")).strip(),
        "model_used": str(record.get("model_used", model)),
        "prompt_version": str(record.get("prompt_version", "legacy")),
    }


def load_existing_strategies(output_path: str, model: str) -> tuple[dict, bool]:
    if not os.path.exists(output_path):
        return {}, False

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    normalized = {}
    changed = False
    for puzzle_type, record in data.items():
        normalized_record = normalize_strategy_record(record, model)
        normalized[puzzle_type] = normalized_record
        if normalized_record != record:
            changed = True
    return normalized, changed


def save_strategies(output_path: str, strategies: dict) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(strategies, f, indent=2, ensure_ascii=False)


def should_skip(existing_record: dict, model: str, force: bool) -> bool:
    if force or not existing_record:
        return False
    if not existing_record.get("strategy"):
        return False
    if existing_record.get("strategy", "").startswith("[ERROR:"):
        return False
    return (
        existing_record.get("prompt_version") == PROMPT_VERSION
        and existing_record.get("model_used") == model
    )


def generate_strategies(
    api_key: str,
    model: str = DEFAULT_MODEL,
    force: bool = False,
) -> None:
    client = genai.Client(api_key=api_key)
    output_path = os.path.join(BASE_DIR, "draft", "oracle_strategies.json")

    strategies, normalized_changed = load_existing_strategies(output_path, model)
    if strategies:
        print(f"Loaded {len(strategies)} existing strategies from {output_path}")
    if force:
        print("Force mode: regenerating all strategies")

    puzzle_types = sorted(TYPE_TO_GENERATOR.keys())

    for index, puzzle_type in enumerate(puzzle_types, start=1):
        existing = strategies.get(puzzle_type)
        if should_skip(existing, model, force):
            print(f"[{index}/{len(puzzle_types)}] {puzzle_type}: up to date, skipping")
            continue

        print(f"[{index}/{len(puzzle_types)}] {puzzle_type}: generating strategy...")

        try:
            strategy = generate_strategy_for_type(client, model, puzzle_type)
            strategies[puzzle_type] = {
                "strategy": strategy,
                "model_used": model,
                "prompt_version": PROMPT_VERSION,
            }
            print(f"  -> Done ({len(strategy)} chars)")
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            strategies[puzzle_type] = {
                "strategy": f"[ERROR: {exc}]",
                "model_used": model,
                "prompt_version": PROMPT_VERSION,
            }

        save_strategies(output_path, strategies)
        time.sleep(1)

    if normalized_changed:
        save_strategies(output_path, strategies)

    print(f"\nDone! Strategies saved to {output_path}")
    print(f"Total: {len(strategies)} types")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fair oracle solving strategies via Gemini"
    )
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model to use (default: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all strategies even if current prompt_version entries exist",
    )
    args = parser.parse_args()

    generate_strategies(args.api_key, args.model, args.force)
