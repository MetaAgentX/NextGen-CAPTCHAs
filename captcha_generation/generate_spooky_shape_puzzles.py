"""
Generate Spooky Shape Grid puzzles using the pre-generated cell library.

Each puzzle selects 9 cells from the library to create a 3x3 grid.
"""

import json
import random
import os
from collections import Counter
from pathlib import Path


def validate_library(library, library_dir):
    """Ensure the library index is self-consistent and files exist."""
    required_fields = {"shapes", "directions", "num_variations", "library"}
    missing_fields = required_fields - set(library.keys())
    if missing_fields:
        raise ValueError(f"Library index missing required fields: {sorted(missing_fields)}")

    lib_map = library["library"]
    shapes = library["shapes"]
    directions = library["directions"]
    num_variations = library["num_variations"]

    expected_keys = [
        f"{shape}_{direction}_{var_id}"
        for shape in shapes
        for direction in directions
        for var_id in range(num_variations)
    ]

    missing_keys = [k for k in expected_keys if k not in lib_map]
    if missing_keys:
        raise ValueError(f"Library index missing expected entries: {missing_keys}")

    if "empty_cell" not in lib_map:
        raise ValueError("Library index missing 'empty_cell' entry for blank cells")

    collisions = [fname for fname, count in Counter(lib_map.values()).items() if count > 1]
    if collisions:
        raise ValueError(f"Library filenames reused across keys: {collisions}")

    base_dir = Path(library_dir)
    missing_files = [fname for fname in lib_map.values() if not (base_dir / fname).exists()]
    if missing_files:
        raise FileNotFoundError(f"Library references missing GIF files: {missing_files}")


def load_library(library_dir):
    """Load the cell library index."""
    library_path = Path(library_dir) / "library_index.json"
    with open(library_path, 'r') as f:
        library = json.load(f)
    validate_library(library, library_dir)
    return library


def generate_puzzle(library, puzzle_id, num_targets=None, target_shape=None, target_direction=None):
    """
    Generate a single puzzle using library cells.

    Args:
        library: Library index dictionary
        puzzle_id: Unique puzzle identifier
        num_targets: Number of target cells (None = random 2-4)
        target_shape: Target shape to find (None = random)
        target_direction: Target direction (None = random)

    Returns:
        Puzzle dictionary with cell references
    """
    shapes = library['shapes']
    directions = library['directions']
    num_variations = library['num_variations']
    lib_map = library['library']
    empty_cell_filename = lib_map["empty_cell"]

    # Choose target
    if target_shape is None:
        target_shape = random.choice(shapes)
    if target_direction is None:
        target_direction = random.choice(directions)
    if num_targets is None:
        num_targets = random.randint(2, 4)

    # Decide total number of shapes (6-9 for a 3x3 grid)
    total_shapes = random.randint(max(num_targets + 2, 6), 9)

    # Select which cells will have shapes
    all_cell_indices = list(range(9))
    shape_cell_indices = random.sample(all_cell_indices, total_shapes)

    # Assign target to random subset
    target_indices = random.sample(shape_cell_indices, num_targets)

    # Build cell configuration
    cell_options = []  # List of cell GIF filenames (9 total, one per grid position)
    cell_config = {}

    for cell_idx in range(9):
        if cell_idx not in shape_cell_indices:
            # Empty cell - use the dedicated empty cell asset
            cell_options.append(empty_cell_filename)
            cell_config[cell_idx] = {
                "shape": None,
                "direction": None,
                "is_target": False,
                "is_empty": True
            }
        elif cell_idx in target_indices:
            # Target cell
            var_id = random.randint(0, num_variations - 1)
            cell_key = f"{target_shape}_{target_direction}_{var_id}"
            if cell_key not in lib_map:
                raise KeyError(f"Missing library entry for target key {cell_key}")
            cell_options.append(lib_map[cell_key])
            cell_config[cell_idx] = {
                "shape": target_shape,
                "direction": target_direction,
                "is_target": True,
                "is_empty": False
            }
        else:
            # Distractor cell - different shape OR different direction
            if random.random() < 0.5:
                # Different shape
                other_shapes = [s for s in shapes if s != target_shape]
                cell_shape = random.choice(other_shapes)
                cell_dir = random.choice(directions)
            else:
                # Same shape, opposite direction
                cell_shape = target_shape
                cell_dir = 'clockwise' if target_direction == 'counterclockwise' else 'counterclockwise'

            var_id = random.randint(0, num_variations - 1)
            cell_key = f"{cell_shape}_{cell_dir}_{var_id}"
            if cell_key not in lib_map:
                raise KeyError(f"Missing library entry for distractor key {cell_key}")
            cell_options.append(lib_map[cell_key])
            cell_config[cell_idx] = {
                "shape": cell_shape,
                "direction": cell_dir,
                "is_target": False,
                "is_empty": False
            }

    # Get answer
    answer = sorted(target_indices)
    target_cells = [(idx // 3, idx % 3) for idx in answer]

    shape_emoji = {
        'circle': '⭕',
        'square': '⬜',
        'triangle': '🔺'
    }

    return {
        "puzzle_id": puzzle_id,
        "options": cell_options,
        "answer": answer,
        "grid_size": [3, 3],
        "target_shape": target_shape,
        "target_direction": target_direction,
        "target_cells": target_cells,
        "cell_config": cell_config,
        "prompt": f"Click all {shape_emoji.get(target_shape, '')} {target_shape}s rotating {target_direction}",
        "description": f"Grid with {total_shapes} shapes, {num_targets} {target_shape}s rotating {target_direction}"
    }


def generate_dataset(library_dir, num_puzzles=20):
    """Generate a dataset of puzzles."""
    library = load_library(library_dir)

    ground_truth = {}

    for i in range(num_puzzles):
        puzzle_id = f"spooky_shape_grid_{i:04d}"

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
            "target_shape": puzzle["target_shape"],
            "target_direction": puzzle["target_direction"],
            "target_cells": puzzle["target_cells"],
            "cell_config": puzzle["cell_config"]
        }

        print(f"  → {puzzle['prompt']}")
        print(f"  → Answer: {puzzle['answer']}")

    # Save ground truth
    output_path = Path(library_dir) / "ground_truth.json"
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGenerated {num_puzzles} puzzles!")
    print(f"Ground truth: {output_path}")


if __name__ == "__main__":
    library_dir = str(Path(__file__).parent.parent / "captcha_data" / "Spooky_Shape_Grid")

    generate_dataset(library_dir, num_puzzles=20)

    print("\n" + "="*70)
    print("🎯 Spooky Shape Grid Puzzles Generated!")
    print("="*70)
    print("\n💡 Using Pre-generated Cell Library:")
    print("  • Each puzzle references 9 cells from the library")
    print("  • No redundant cell generation")
    print("  • Maximum space efficiency!")
