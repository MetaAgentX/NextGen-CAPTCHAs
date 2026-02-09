#!/usr/bin/env python3
"""
Temporal Object Continuity CAPTCHA

CONCEPT:
- 4x4 grid of cells
- Each cell shows an object (circle, square, triangle) moving horizontally
- Objects pass behind vertical occluders (black bars)
- ONE cell: The object CHANGES IDENTITY while behind the occluder
  (e.g., red circle enters, blue square exits)

HUMAN PERCEPTION:
- Amodal completion: We mentally track objects behind occluders
- Object permanence: We expect the same object to emerge
- Identity violation is INSTANTLY obvious ("that's not the same thing!")

LLM WEAKNESS:
- No temporal object tracking across occlusion
- Analyzes frames independently
- Cannot build persistent object representation
- Sees: "red circle on left, blue square on right" - no violation detected
"""

import numpy as np
from PIL import Image, ImageDraw
import json
import random
import math
from pathlib import Path

# Configuration
GRID_SIZE = 4
CELL_SIZE = 150
NUM_FRAMES = 100  # 100 frames for full animation
FRAME_DURATION = 40  # 40ms = 25fps

OUTPUT_DIR = Path(__file__).parent.parent / "captcha_data" / "Temporal_Object_Continuity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_PUZZLES = 20

# Animation parameters
OBJECT_SIZE = 15  # Size of moving objects (smaller)
OCCLUDER_SIZE = 55  # Size of occluder (width/height) - just big enough to contain objects
OCCLUDER_POSITIONS = [45, 105]  # X positions of occluders in cell
MOTION_SPEED = 1.5  # Pixels per frame (slower for smoother animation)

# Object types
SHAPES = ['circle', 'square', 'triangle', 'star', 'hexagon']
COLORS = [
    (220, 80, 80),   # Red
    (80, 150, 220),  # Blue
    (100, 200, 100), # Green
    (220, 180, 70),  # Yellow
    (200, 100, 200), # Purple
    (255, 140, 0),   # Orange
]

# Occluder shapes (varied per cell)
OCCLUDER_SHAPES = ['circle', 'square', 'diamond', 'hexagon']


def draw_shape(draw, x, y, size, shape, color, outline_width=2):
    """Draw a shape at position (x, y)."""
    if shape == 'circle':
        draw.ellipse([x - size, y - size, x + size, y + size],
                    fill=color, outline=(0, 0, 0), width=outline_width)
    elif shape == 'square':
        draw.rectangle([x - size, y - size, x + size, y + size],
                      fill=color, outline=(0, 0, 0), width=outline_width)
    elif shape == 'triangle':
        points = [
            (x, y - size),           # Top
            (x - size, y + size),    # Bottom left
            (x + size, y + size)     # Bottom right
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=outline_width)
    elif shape == 'star':
        # 5-pointed star
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            radius = size if i % 2 == 0 else size * 0.4
            px = x + radius * math.cos(angle)
            py = y - radius * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=outline_width)
    elif shape == 'hexagon':
        points = []
        for i in range(6):
            angle = i * math.pi / 3
            px = x + size * math.cos(angle)
            py = y + size * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=outline_width)
    elif shape == 'diamond':
        points = [
            (x, y - size),      # Top
            (x + size, y),      # Right
            (x, y + size),      # Bottom
            (x - size, y)       # Left
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=outline_width)


def is_point_in_shape(px, py, cx, cy, size, shape):
    """Check if point (px, py) is inside a shape centered at (cx, cy)."""
    if shape == 'circle':
        dist_sq = (px - cx) ** 2 + (py - cy) ** 2
        return dist_sq <= size ** 2
    elif shape == 'square':
        return abs(px - cx) <= size and abs(py - cy) <= size
    elif shape == 'diamond':
        # Diamond is rotated square - Manhattan distance
        return abs(px - cx) + abs(py - cy) <= size * 1.4
    elif shape == 'hexagon':
        # Simplified hexagon check
        dist_sq = (px - cx) ** 2 + (py - cy) ** 2
        return dist_sq <= size ** 2
    return False


def is_object_fully_occluded(obj_x, obj_y, obj_size, occ_x, occ_y, occ_size, occ_shape):
    """
    Check if the ENTIRE object is inside the occluder (fully hidden).
    We need to check if all corners/edges of the object are inside the occluder.
    """
    # Check key points around the object boundary
    test_points = [
        (obj_x - obj_size, obj_y - obj_size),  # Top-left
        (obj_x + obj_size, obj_y - obj_size),  # Top-right
        (obj_x - obj_size, obj_y + obj_size),  # Bottom-left
        (obj_x + obj_size, obj_y + obj_size),  # Bottom-right
        (obj_x, obj_y - obj_size),             # Top
        (obj_x, obj_y + obj_size),             # Bottom
        (obj_x - obj_size, obj_y),             # Left
        (obj_x + obj_size, obj_y),             # Right
    ]

    # ALL points must be inside the occluder
    for px, py in test_points:
        if not is_point_in_shape(px, py, occ_x, occ_y, occ_size, occ_shape):
            return False
    return True


def create_object_continuity_gif(cell_idx, initial_shape, initial_color,
                                 final_shape=None, final_color=None,
                                 change_behind_occluder=0, occluder_shapes=None):
    """
    Create a GIF showing an object moving across the cell behind occluders.
    Object changes identity ONLY when FULLY hidden behind an occluder.

    Args:
        cell_idx: Cell index for random seed
        initial_shape: Shape that enters from left
        initial_color: Color that enters from left
        final_shape: Shape that exits on right (None = same as initial)
        final_color: Color that exits on right (None = same as initial)
        change_behind_occluder: Index of occluder where change happens (-1 = no change)
        occluder_shapes: List of shapes for each occluder

    Returns:
        List of PIL Image frames
    """
    frames = []

    # If no change specified, object stays the same
    if final_shape is None:
        final_shape = initial_shape
    if final_color is None:
        final_color = initial_color

    # Default occluder shapes if not provided
    if occluder_shapes is None:
        occluder_shapes = ['square'] * len(OCCLUDER_POSITIONS)

    # Object travels from left to right
    start_x = -OBJECT_SIZE * 2
    end_x = CELL_SIZE + OBJECT_SIZE * 2
    total_distance = end_x - start_x

    # Track if we've made the change yet
    has_changed = False

    for frame in range(NUM_FRAMES):
        # Create base image
        img = Image.new('RGB', (CELL_SIZE, CELL_SIZE), (245, 245, 245))
        draw = ImageDraw.Draw(img)

        # Draw border
        draw.rectangle([0, 0, CELL_SIZE - 1, CELL_SIZE - 1],
                      outline=(180, 180, 180), width=2)

        # Calculate object position
        progress = frame / NUM_FRAMES
        obj_x = int(start_x + progress * total_distance)
        obj_y = CELL_SIZE // 2

        # Check if object is FULLY occluded behind the target occluder
        if change_behind_occluder >= 0 and not has_changed:
            occ_idx = change_behind_occluder
            occ_x = OCCLUDER_POSITIONS[occ_idx]
            occ_y = CELL_SIZE // 2
            occ_shape = occluder_shapes[occ_idx]

            if is_object_fully_occluded(obj_x, obj_y, OBJECT_SIZE,
                                       occ_x, occ_y, OCCLUDER_SIZE // 2, occ_shape):
                # Object is COMPLETELY hidden - make the change NOW
                has_changed = True

        # Determine current shape and color
        if has_changed:
            current_shape = final_shape
            current_color = final_color
        else:
            current_shape = initial_shape
            current_color = initial_color

        # Draw object FIRST (background)
        draw_shape(draw, obj_x, obj_y, OBJECT_SIZE, current_shape, current_color)

        # Draw occluders ON TOP (so they actually occlude the object!)
        for i, occ_x in enumerate(OCCLUDER_POSITIONS):
            occ_y = CELL_SIZE // 2
            occ_shape = occluder_shapes[i]
            # Occluders are dark gray - drawn ON TOP to hide the object
            draw_shape(draw, occ_x, occ_y, OCCLUDER_SIZE // 2, occ_shape, (60, 60, 60), outline_width=3)

        frames.append(img)

    return frames


def save_gif(frames, path):
    """Save frames as looping GIF."""
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION,
        loop=0,
        optimize=False
    )


def generate_puzzle(puzzle_idx):
    """Generate one Temporal Object Continuity puzzle."""
    puzzle_dir = OUTPUT_DIR / f"puzzle_{puzzle_idx}"
    puzzle_dir.mkdir(exist_ok=True)

    # Choose MULTIPLE violating cells (2-4 cells)
    num_violating = random.randint(2, 4)
    all_cells = list(range(GRID_SIZE * GRID_SIZE))
    violating_cells = random.sample(all_cells, num_violating)
    violating_cells.sort()  # Keep them sorted for consistency

    print(f"  Puzzle {puzzle_idx}: {num_violating} violating cells: {violating_cells}")

    cell_files = []

    # For each cell, choose random shape and color
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell_idx = row * GRID_SIZE + col

            # Random initial properties
            initial_shape = random.choice(SHAPES)
            initial_color = random.choice(COLORS)

            # Random occluder shapes for THIS cell
            occluder_shapes = [random.choice(OCCLUDER_SHAPES) for _ in range(len(OCCLUDER_POSITIONS))]

            if cell_idx in violating_cells:
                # VIOLATING CELL: Object changes behind occluder
                # Choose different shape AND color
                final_shape = random.choice([s for s in SHAPES if s != initial_shape])
                final_color = random.choice([c for c in COLORS if c != initial_color])

                # Change happens behind first occluder (index 0)
                change_occluder = 0

                print(f"    Cell {cell_idx}: {initial_shape} -> {final_shape}, occluders: {occluder_shapes}")
            else:
                # NORMAL CELL: Object stays the same
                final_shape = initial_shape
                final_color = initial_color
                change_occluder = -1  # No change

            # Generate GIF
            frames = create_object_continuity_gif(
                cell_idx, initial_shape, initial_color,
                final_shape, final_color, change_occluder, occluder_shapes
            )

            # Save
            filename = f"cell_{row}_{col}.gif"
            save_gif(frames, puzzle_dir / filename)
            cell_files.append(filename)

    # Convert violating cells to positions
    violating_positions = [[cell // GRID_SIZE, cell % GRID_SIZE] for cell in violating_cells]

    return {
        "puzzle_id": f"temporal_continuity_{puzzle_idx}",
        "type": "grid_select",
        "violating_cells": violating_cells,
        "violating_positions": violating_positions,
        "answer": violating_cells,
        "grid_size": [GRID_SIZE, GRID_SIZE],
        "puzzle_dir": f"puzzle_{puzzle_idx}",
        "cell_files": cell_files,
        "mechanism": "object_identity_change_behind_occluder"
    }


def main():
    """Generate all puzzles."""
    print("=" * 80)
    print("Temporal Object Continuity CAPTCHA")
    print("=" * 80)
    print(f"\nGenerating {NUM_PUZZLES} puzzles...")
    print(f"  • 4x4 grid of moving objects")
    print(f"  • Objects pass behind {len(OCCLUDER_POSITIONS)} vertical occluders")
    print(f"  • ONE object illegally changes identity while occluded")
    print(f"  • Humans spot instantly via object permanence")
    print(f"  • LLMs lack temporal object tracking across occlusion")
    print("=" * 80 + "\n")

    ground_truth = {}

    for i in range(NUM_PUZZLES):
        puzzle_data = generate_puzzle(i)
        ground_truth[puzzle_data["puzzle_id"]] = puzzle_data

    # Save ground truth
    with open(OUTPUT_DIR / "ground_truth.json", 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n" + "=" * 80)
    print(f"✓ Generated {NUM_PUZZLES} puzzles")
    print(f"✓ Saved to {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
