"""Generate Spooky Circle Grid Direction puzzles using the cell library."""

import json
import random
from pathlib import Path


def load_library(library_dir):
    """Load the cell library index."""
    library_path = Path(library_dir) / "library_index.json"
    with open(library_path, 'r') as f:
        return json.load(f)


def generate_puzzle(library, puzzle_id, target_direction=None, num_targets=None):
    """Generate a puzzle asking users to click circles moving in a specific direction."""
    directions = library['directions']
    num_variations = library['num_variations']

    if target_direction is None:
        target_direction = random.choice(directions)
    if num_targets is None:
        num_targets = random.randint(2, 4)

    # Decide total number of circles
    total_circles = random.randint(num_targets + 1, min(9, num_targets + 5))

    # Select cells for circles
    all_cell_indices = list(range(9))
    circle_indices = random.sample(all_cell_indices, total_circles)

    # Assign target direction to some circles
    target_indices = sorted(random.sample(circle_indices, num_targets))

    # Build cell options
    cell_options = []
    cell_config = {}

    for cell_idx in range(9):
        if cell_idx not in circle_indices:
            # Empty cell
            var_id = random.randint(0, num_variations - 1)
            cell_key = f"empty_{var_id}"
            cell_options.append(library['library'][cell_key])
            cell_config[cell_idx] = {
                "direction": None,
                "is_target": False,
                "is_empty": True
            }
        elif cell_idx in target_indices:
            # Target circle
            var_id = random.randint(0, num_variations - 1)
            cell_key = f"{target_direction}_{var_id}"
            cell_options.append(library['library'][cell_key])
            cell_config[cell_idx] = {
                "direction": target_direction,
                "is_target": True,
                "is_empty": False
            }
        else:
            # Distractor circle (different direction)
            other_directions = [d for d in directions if d != target_direction]
            distractor_dir = random.choice(other_directions)
            var_id = random.randint(0, num_variations - 1)
            cell_key = f"{distractor_dir}_{var_id}"
            cell_options.append(library['library'][cell_key])
            cell_config[cell_idx] = {
                "direction": distractor_dir,
                "is_target": False,
                "is_empty": False
            }

    target_cells = [(idx // 3, idx % 3) for idx in target_indices]

    direction_prompts = {
        'clockwise': 'Click all circles rotating CLOCKWISE',
        'counterclockwise': 'Click all circles rotating COUNTERCLOCKWISE',
        'up': 'Click all circles moving UP',
        'down': 'Click all circles moving DOWN',
        'left': 'Click all circles moving LEFT',
        'right': 'Click all circles moving RIGHT'
    }

    return {
        "puzzle_id": puzzle_id,
        "options": cell_options,
        "answer": target_indices,
        "grid_size": [3, 3],
        "target_direction": target_direction,
        "target_cells": target_cells,
        "num_targets": num_targets,
        "total_circles": total_circles,
        "cell_config": cell_config,
        "prompt": direction_prompts.get(target_direction, f"Click all circles moving {target_direction}"),
        "description": f"Grid with {total_circles} circles, {num_targets} moving {target_direction}"
    }


def generate_dataset(library_dir, num_puzzles=20):
    """Generate a dataset of puzzles."""
    library = load_library(library_dir)
    ground_truth = {}

    for i in range(num_puzzles):
        puzzle_id = f"spooky_circle_grid_direction_{i:04d}"
        print(f"Generating {puzzle_id}...")

        puzzle = generate_puzzle(library, puzzle_id)

        ground_truth[puzzle_id] = {
            "answer": puzzle["answer"],
            "prompt": puzzle["prompt"],
            "description": puzzle["description"],
            "options": puzzle["options"],
            "grid_size": puzzle["grid_size"],
            "difficulty": 5,
            "media_type": "gif",
            "target_direction": puzzle["target_direction"],
            "target_cells": puzzle["target_cells"],
            "num_targets": puzzle["num_targets"],
            "total_circles": puzzle["total_circles"],
            "cell_config": puzzle["cell_config"]
        }

        print(f"  → {puzzle['prompt']}")
        print(f"  → Answer: {puzzle['answer']}")

    output_path = Path(library_dir) / "ground_truth.json"
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGenerated {num_puzzles} puzzles!")
    print(f"Ground truth: {output_path}")


if __name__ == "__main__":
    library_dir = str(Path(__file__).parent.parent / "captcha_data" / "Spooky_Circle_Grid_Direction")
    generate_dataset(library_dir, num_puzzles=20)

    print("\n" + "="*70)
    print("🎯 Spooky Circle Grid Direction Puzzles Generated!")
    print("Click all circles moving in the specified direction")
    print("="*70)
