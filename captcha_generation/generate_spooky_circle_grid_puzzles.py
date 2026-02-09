"""Generate Spooky Circle Grid puzzles using the cell library."""

import json
import random
from pathlib import Path


def load_library(library_dir):
    """Load the cell library index."""
    library_path = Path(library_dir) / "library_index.json"
    with open(library_path, 'r') as f:
        return json.load(f)


def generate_puzzle(library, puzzle_id, num_circles=None):
    """Generate a puzzle asking users to click all cells with circles."""
    num_variations = library['num_variations']

    if num_circles is None:
        num_circles = random.randint(2, 5)

    # Select which cells have circles
    all_cell_indices = list(range(9))
    circle_indices = sorted(random.sample(all_cell_indices, num_circles))

    # Build cell options
    cell_options = []
    for cell_idx in range(9):
        if cell_idx in circle_indices:
            var_id = random.randint(0, num_variations - 1)
            cell_key = f"circle_{var_id}"
            cell_options.append(library['library'][cell_key])
        else:
            var_id = random.randint(0, num_variations - 1)
            cell_key = f"empty_{var_id}"
            cell_options.append(library['library'][cell_key])

    circle_cells = [(idx // 3, idx % 3) for idx in circle_indices]

    return {
        "puzzle_id": puzzle_id,
        "options": cell_options,
        "answer": circle_indices,
        "grid_size": [3, 3],
        "circle_cells": circle_cells,
        "num_circles": num_circles,
        "prompt": "Click all cells containing circles",
        "description": f"Grid with {num_circles} cell(s) containing motion-contrast circles"
    }


def generate_dataset(library_dir, num_puzzles=20):
    """Generate a dataset of puzzles."""
    library = load_library(library_dir)
    ground_truth = {}

    for i in range(num_puzzles):
        puzzle_id = f"spooky_circle_grid_{i:04d}"
        print(f"Generating {puzzle_id}...")

        puzzle = generate_puzzle(library, puzzle_id)

        ground_truth[puzzle_id] = {
            "answer": puzzle["answer"],
            "prompt": puzzle["prompt"],
            "description": puzzle["description"],
            "options": puzzle["options"],
            "grid_size": puzzle["grid_size"],
            "difficulty": 4,
            "media_type": "gif",
            "circle_cells": puzzle["circle_cells"],
            "num_circles": puzzle["num_circles"]
        }

        print(f"  → {puzzle['description']}")
        print(f"  → Answer: {puzzle['answer']}")

    output_path = Path(library_dir) / "ground_truth.json"
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGenerated {num_puzzles} puzzles!")
    print(f"Ground truth: {output_path}")


if __name__ == "__main__":
    library_dir = str(Path(__file__).parent.parent / "captcha_data" / "Spooky_Circle_Grid")
    generate_dataset(library_dir, num_puzzles=20)

    print("\n" + "="*70)
    print("🎯 Spooky Circle Grid Puzzles Generated!")
    print("Click all cells containing circles")
    print("="*70)
