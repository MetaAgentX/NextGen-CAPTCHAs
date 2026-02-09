"""
Layered Stack CAPTCHA Generator

Each cell shows a vertical stack of shapes (2-4 shapes stacked on top of each other).
The task is to select cells based on counting relationships, e.g.:
"Select cells where there are 2 circles under a square"

This exploits VLM weaknesses in:
- Depth reasoning (which shape is on top/bottom)
- Counting with spatial constraints (counting shapes in specific positions)
- Understanding occlusion patterns in stacks
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image, ImageDraw

# -----------------------------
# Configuration
# -----------------------------

OUTPUT_DIR = Path('../captcha_data/Layered_Stack/')
NUM_CELLS_IN_POOL = 60  # Total cells to generate in pool
NUM_PUZZLES = 20  # Number of puzzles to generate
IMAGE_SIZE = 600
CELL_SIZE = 300  # Size of each cell in the grid (higher resolution)
GRID_SIZE = (4, 4)  # 4x4 grid (16 cells per puzzle)

# Stack configuration
MIN_SHAPES_IN_STACK = 5
MAX_SHAPES_IN_STACK = 5

# Shape size within each cell
SHAPE_SIZE_RATIO = 0.25  # Shape takes 25% of cell width

# Color palette (distinct colors for easy identification)
COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "red":      (255, 30, 30),
    "blue":     (30, 60, 255),
    "green":    (30, 220, 30),
    "yellow":   (255, 220, 0),
    "orange":   (255, 140, 0),
    "purple":   (180, 30, 200),
}

SHAPE_TYPES = ["circle", "square", "triangle", "star"]

# Question templates
QUESTION_TEMPLATES = [
    "Select all cells where a {top_shape} is on top (highest layer) with exactly {count} {target_shape_plural} in ALL layers below it",
    "Find cells where the top shape is a {top_shape} and there are exactly {count} {target_shape_plural} total in all lower layers",
    "Click cells where a {top_shape} sits at the highest position with exactly {count} {target_shape_plural} anywhere beneath it (count all layers below)",
]

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ShapeInStack:
    """A shape in a vertical stack"""
    shape_type: str
    color_name: str
    color_rgb: Tuple[int, int, int]
    position: int  # 0=bottom, higher=on top
    fill_pattern: str  # "solid", "stripes", "dots"

@dataclass
class CellStack:
    """A vertical stack of shapes in one cell"""
    shapes: List[ShapeInStack]

@dataclass
class CaptchaSample:
    image: Image.Image
    question: str
    answer: List[int]  # Indices of correct cells
    puzzle_type: str
    metadata: Dict

# -----------------------------
# Shape rendering
# -----------------------------

def draw_shape(draw: ImageDraw.Draw, shape_type: str, center_x: float, center_y: float,
               size: float, color: Tuple[int, int, int], fill_pattern: str = "solid"):
    """Draw a single shape at given position with fill pattern"""
    from PIL import Image as PILImage

    # Stars are 20% larger
    if shape_type == "star":
        size = size * 1.2

    half_size = size / 2
    x_min = center_x - half_size
    y_min = center_y - half_size
    x_max = center_x + half_size
    y_max = center_y + half_size

    # Create a mask for the shape
    mask = PILImage.new('L', (int(size), int(size)), 0)
    mask_draw = ImageDraw.Draw(mask)

    if shape_type == "circle":
        mask_draw.ellipse([0, 0, size, size], fill=255)
    elif shape_type == "square":
        mask_draw.rectangle([0, 0, size, size], fill=255)
    elif shape_type == "triangle":
        points = [
            (size/2, 0),           # Top
            (0, size),             # Bottom left
            (size, size)           # Bottom right
        ]
        mask_draw.polygon(points, fill=255)
    elif shape_type == "star":
        outer_r = size / 2
        inner_r = outer_r * 0.4
        points = []
        for i in range(10):
            angle = (i * 36 - 90) * np.pi / 180
            r = outer_r if i % 2 == 0 else inner_r
            points.append((size/2 + r * np.cos(angle), size/2 + r * np.sin(angle)))
        mask_draw.polygon(points, fill=255)

    # Apply fill pattern
    pattern_img = PILImage.new('RGB', (int(size), int(size)), color)
    pattern_draw = ImageDraw.Draw(pattern_img)

    if fill_pattern == "stripes":
        # Diagonal stripes
        stripe_spacing = max(6, int(size * 0.12))
        stripe_width = max(2, int(size * 0.04))
        for i in range(-int(size), int(size*2), stripe_spacing):
            pattern_draw.line([(i, 0), (i + size, size)], fill=(255, 255, 255), width=stripe_width)
    elif fill_pattern == "crosshatch":
        # Crosshatch pattern
        spacing = max(8, int(size * 0.15))
        line_width = max(1, int(size * 0.02))
        # Horizontal lines
        for y in range(0, int(size), spacing):
            pattern_draw.line([(0, y), (size, y)], fill=(255, 255, 255), width=line_width)
        # Vertical lines
        for x in range(0, int(size), spacing):
            pattern_draw.line([(x, 0), (x, size)], fill=(255, 255, 255), width=line_width)
    elif fill_pattern == "squares":
        # Small squares pattern
        square_spacing = max(12, int(size * 0.2))
        square_size = max(4, int(size * 0.08))
        for dx in range(square_spacing//2, int(size), square_spacing):
            for dy in range(square_spacing//2, int(size), square_spacing):
                pattern_draw.rectangle([dx-square_size//2, dy-square_size//2,
                                       dx+square_size//2, dy+square_size//2],
                                      fill=(255, 255, 255))

    # Paste the pattern onto the image using the mask
    # Convert pattern to match draw's image mode
    img = draw._image
    img.paste(pattern_img, (int(x_min), int(y_min)), mask)

    # Draw outline
    if shape_type == "circle":
        draw.ellipse([x_min, y_min, x_max, y_max], outline=(0, 0, 0), width=3)
    elif shape_type == "square":
        draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 0, 0), width=3)
    elif shape_type == "triangle":
        points = [
            (center_x, y_min),
            (x_min, y_max),
            (x_max, y_max)
        ]
        draw.polygon(points, outline=(0, 0, 0), width=3)
    elif shape_type == "star":
        outer_r = size / 2
        inner_r = outer_r * 0.4
        points = []
        for i in range(10):
            angle = (i * 36 - 90) * np.pi / 180
            r = outer_r if i % 2 == 0 else inner_r
            points.append((center_x + r * np.cos(angle), center_y + r * np.sin(angle)))
        draw.polygon(points, outline=(0, 0, 0), width=3)

def render_cell_stack(stack: CellStack, cell_size: int) -> Image.Image:
    """Render a single cell with shapes guaranteed to show layering relationships"""
    img = Image.new('RGB', (cell_size, cell_size), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Draw border
    draw.rectangle([0, 0, cell_size - 1, cell_size - 1], outline=(200, 200, 200), width=2)

    num_shapes = len(stack.shapes)
    shape_size = cell_size * SHAPE_SIZE_RATIO

    # Margin from edges
    margin = shape_size / 2 + 5

    # Generate positions with guaranteed overlap for consecutive layers
    positions = []

    # Sort shapes by position (bottom to top) for proper rendering
    sorted_shapes = sorted(stack.shapes, key=lambda s: s.position)

    for i, shape in enumerate(sorted_shapes):
        if i == 0:
            # First shape (bottom layer): place in one of the four corners
            corner = random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
            corner_offset = shape_size * 0.7  # Distance from corner

            if corner == 'top_left':
                center_x = margin + corner_offset
                center_y = margin + corner_offset
                base_direction = np.array([1.0, 1.0])  # Chain toward bottom-right
            elif corner == 'top_right':
                center_x = cell_size - margin - corner_offset
                center_y = margin + corner_offset
                base_direction = np.array([-1.0, 1.0])  # Chain toward bottom-left
            elif corner == 'bottom_left':
                center_x = margin + corner_offset
                center_y = cell_size - margin - corner_offset
                base_direction = np.array([1.0, -1.0])  # Chain toward top-right
            else:  # bottom_right
                center_x = cell_size - margin - corner_offset
                center_y = cell_size - margin - corner_offset
                base_direction = np.array([-1.0, -1.0])  # Chain toward top-left

            # Normalize base direction
            base_direction = base_direction / np.linalg.norm(base_direction)

            positions.append((center_x, center_y, shape.position))
        else:
            # For shapes at layer N > 0: Continue chain toward opposite corner
            # Use base_direction with angle variation (±15-30°)
            prev_layer_positions = [
                (px, py) for px, py, layer in positions if layer == shape.position - 1
            ]

            # Get positions from 2 layers below (to avoid total overlap)
            two_layers_below = [
                (px, py) for px, py, layer in positions if layer == shape.position - 2
            ]

            # Try to find a position that overlaps with previous layer but ensures visibility
            max_attempts = 100
            found = False
            for attempt in range(max_attempts):
                if prev_layer_positions:
                    # Pick a random shape from previous layer to overlap with
                    target_x, target_y = random.choice(prev_layer_positions)

                    # Get the previous layer shape type to check if both are stars
                    prev_shape = sorted_shapes[i-1]
                    curr_shape_is_star = shape.shape_type == "star"
                    prev_shape_is_star = prev_shape.shape_type == "star"

                    # Offset distance ensures at least 30% overlap
                    # Distance should be 35-65% of shape size for good overlap
                    if curr_shape_is_star and prev_shape_is_star:
                        # Stars are closer but still visible
                        offset_dist = random.uniform(shape_size * 0.35, shape_size * 0.55)
                    else:
                        # Normal distance: ensures good overlap and visibility
                        offset_dist = random.uniform(shape_size * 0.40, shape_size * 0.65)

                    # Chain arrangement: continue in base_direction with small angle variation
                    # Add random angle deviation (±10°)
                    angle_deviation = random.uniform(-np.pi/6, np.pi/6)  # ±10°
                    base_angle = np.arctan2(base_direction[1], base_direction[0])
                    angle = base_angle + angle_deviation

                    center_x = target_x + offset_dist * np.cos(angle)
                    center_y = target_y + offset_dist * np.sin(angle)

                    # Check bounds
                    if margin <= center_x <= cell_size - margin and margin <= center_y <= cell_size - margin:
                        # Verify overlap with previous layer (not too close, not too far)
                        has_good_overlap = False
                        for px, py in prev_layer_positions:
                            dist = ((center_x - px)**2 + (center_y - py)**2)**0.5
                            # Require partial overlap: distance should be 30-70% of shape size
                            if shape_size * 0.3 < dist < shape_size * 0.75:
                                has_good_overlap = True
                                break

                        # Check that we don't fully overlap with N-2 layer
                        avoids_full_overlap = True
                        if two_layers_below:
                            for px, py in two_layers_below:
                                dist = ((center_x - px)**2 + (center_y - py)**2)**0.5
                                # If too close to N-2 layer (almost centered), reject
                                if dist < shape_size * 0.25:
                                    avoids_full_overlap = False
                                    break

                        # NEW: Check minimum visible area constraint
                        # Count how many shapes from LOWER layers will be heavily occluded by this position
                        # We want to ensure lower shapes remain at least 30% visible
                        lower_shapes_ok = True
                        for lower_x, lower_y, lower_layer in positions:
                            if lower_layer < shape.position:
                                # Check if this new position would heavily occlude the lower shape
                                dist_to_lower = ((center_x - lower_x)**2 + (center_y - lower_y)**2)**0.5
                                # If we're placing almost directly on top of a lower shape, reject
                                if dist_to_lower < shape_size * 0.35:
                                    lower_shapes_ok = False
                                    break

                        if has_good_overlap and avoids_full_overlap and lower_shapes_ok:
                            positions.append((center_x, center_y, shape.position))
                            found = True
                            break

            if not found:
                # Fallback: force overlap with first previous layer shape, avoiding N-2
                if prev_layer_positions:
                    target_x, target_y = prev_layer_positions[0]
                    # Try different angles to avoid N-2
                    for angle_attempt in range(8):
                        angle = angle_attempt * np.pi / 4  # Try 8 directions
                        offset_dist = shape_size * 0.5
                        center_x = target_x + offset_dist * np.cos(angle)
                        center_y = target_y + offset_dist * np.sin(angle)

                        # Check bounds
                        if margin <= center_x <= cell_size - margin and margin <= center_y <= cell_size - margin:
                            # Check distance from N-2
                            good_position = True
                            if two_layers_below:
                                for px, py in two_layers_below:
                                    dist = ((center_x - px)**2 + (center_y - py)**2)**0.5
                                    if dist < shape_size * 0.25:
                                        good_position = False
                                        break

                            if good_position:
                                positions.append((np.clip(center_x, margin, cell_size - margin),
                                               np.clip(center_y, margin, cell_size - margin),
                                               shape.position))
                                found = True
                                break

                    if not found:
                        # Last resort: just place it with clipping
                        offset_dist = shape_size * 0.5
                        angle = random.uniform(0, 2 * np.pi)
                        center_x = np.clip(target_x + offset_dist * np.cos(angle), margin, cell_size - margin)
                        center_y = np.clip(target_y + offset_dist * np.sin(angle), margin, cell_size - margin)
                        positions.append((center_x, center_y, shape.position))
                else:
                    # Last resort: random position
                    center_x = random.uniform(margin, cell_size - margin)
                    center_y = random.uniform(margin, cell_size - margin)
                    positions.append((center_x, center_y, shape.position))

    # Draw shapes from bottom to top
    for (center_x, center_y, _), shape in zip(positions, sorted_shapes):
        draw_shape(draw, shape.shape_type, center_x, center_y, shape_size, shape.color_rgb, shape.fill_pattern)

    return img

# -----------------------------
# Stack generation
# -----------------------------

def generate_random_stack() -> CellStack:
    """Generate a random vertical stack of shapes with fill patterns"""
    num_shapes = random.randint(MIN_SHAPES_IN_STACK, MAX_SHAPES_IN_STACK)

    shapes = []
    fill_patterns = ["solid", "stripes", "crosshatch", "squares"]

    for position in range(num_shapes):
        shape_type = random.choice(SHAPE_TYPES)
        color_name = random.choice(list(COLOR_PALETTE.keys()))
        color_rgb = COLOR_PALETTE[color_name]
        fill_pattern = random.choice(fill_patterns)

        shapes.append(ShapeInStack(
            shape_type=shape_type,
            color_name=color_name,
            color_rgb=color_rgb,
            position=position,
            fill_pattern=fill_pattern
        ))

    return CellStack(shapes=shapes)

def count_shapes_under_top_shape(stack: CellStack, target_shape: str, top_shape: str) -> int:
    """
    Count how many target_shape instances are under (below) top_shape in the stack.

    Returns -1 if top_shape is not on top (position == max position).
    """
    # Find the topmost shape
    max_position = max(s.position for s in stack.shapes)
    top_shapes = [s for s in stack.shapes if s.position == max_position]

    # Check if any of the top shapes match the required top_shape
    if not any(s.shape_type == top_shape for s in top_shapes):
        return -1  # Top shape doesn't match

    # Count target shapes below the top
    count = sum(1 for s in stack.shapes
               if s.shape_type == target_shape and s.position < max_position)

    return count

# -----------------------------
# Puzzle generation
# -----------------------------

def generate_question_and_answer(stacks: List[CellStack]) -> Tuple[str, List[int], Dict]:
    """Generate a question and find matching cells"""

    # Choose target and top shapes
    target_shape = random.choice(SHAPE_TYPES)
    top_shape = random.choice([s for s in SHAPE_TYPES if s != target_shape])

    # Count target shapes under top shape for each cell
    counts = []
    for stack in stacks:
        count = count_shapes_under_top_shape(stack, target_shape, top_shape)
        counts.append(count)

    # Find the most common non-negative count (that appears 2-8 times)
    from collections import Counter
    valid_counts = [c for c in counts if c >= 0]

    if not valid_counts:
        # Fallback: no valid cells
        return None, [], {}

    count_freq = Counter(valid_counts)
    # Filter counts that appear 2-8 times (not too rare, not too common)
    # IMPORTANT:
    # 1. Exclude count=0 to avoid confusing questions like "0 shapes below X"
    # 2. Require at least 2 cells matching to ensure meaningful question
    good_counts = [(cnt, freq) for cnt, freq in count_freq.items() if 2 <= freq <= 8 and cnt > 0]

    if not good_counts:
        # Fallback: pick any count >= 1 that appears at least 2 times
        good_counts = [(cnt, freq) for cnt, freq in count_freq.items() if cnt > 0 and freq >= 2]

    if not good_counts:
        # No valid questions possible (need at least 2 matching cells)
        return None, [], {}

    # Choose the count with best frequency
    target_count, _ = random.choice(good_counts)

    # Find cells that match
    answer_indices = [i for i, c in enumerate(counts) if c == target_count]

    # Generate question
    target_shape_plural = target_shape + "s" if target_shape != "star" else "stars"
    template = random.choice(QUESTION_TEMPLATES)
    question = template.format(
        count=target_count,
        target_shape_plural=target_shape_plural,
        top_shape=top_shape
    )

    metadata = {
        "target_shape": target_shape,
        "top_shape": top_shape,
        "target_count": target_count,
        "num_matching_cells": len(answer_indices)
    }

    return question, answer_indices, metadata

def generate_cell_pool() -> Dict:
    """Generate pool of cell images with their metadata

    Returns:
        cell_pool: dict mapping cell_id -> {filename, stack_metadata}
    """
    print(f"Generating cell pool of {NUM_CELLS_IN_POOL} images...")
    cell_pool = {}

    for i in range(NUM_CELLS_IN_POOL):
        cell_id = f"cell_{i:03d}"
        filename = f"{cell_id}.png"

        # Generate random stack
        stack = generate_random_stack()

        # Render and save
        cell_img = render_cell_stack(stack, CELL_SIZE)
        cell_img.save(OUTPUT_DIR / filename)

        # Store metadata about the stack
        stack_info = {
            "shapes": [
                {
                    "shape_type": s.shape_type,
                    "color_name": s.color_name,
                    "position": s.position,
                    "fill_pattern": s.fill_pattern
                }
                for s in stack.shapes
            ]
        }

        cell_pool[cell_id] = {
            "filename": filename,
            "stack": stack_info
        }

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{NUM_CELLS_IN_POOL} cells...")

    return cell_pool

def reconstruct_stack_from_metadata(stack_info: Dict) -> CellStack:
    """Reconstruct CellStack from metadata"""
    shapes = []
    for shape_data in stack_info["shapes"]:
        shapes.append(ShapeInStack(
            shape_type=shape_data["shape_type"],
            color_name=shape_data["color_name"],
            color_rgb=COLOR_PALETTE[shape_data["color_name"]],
            position=shape_data["position"],
            fill_pattern=shape_data["fill_pattern"]
        ))
    return CellStack(shapes=shapes)

def generate_one_puzzle(puzzle_id: int, cell_pool: Dict) -> Tuple[str, List[int], Dict, List[str]]:
    """Generate one puzzle by selecting cells from the pool

    Returns:
        question, answer, metadata, cell_ids
    """
    rows, cols = GRID_SIZE
    num_cells = rows * cols

    # Try to find a good question
    max_attempts = 100
    for attempt in range(max_attempts):
        # Randomly select cells from pool
        selected_cell_ids = random.sample(list(cell_pool.keys()), num_cells)

        # Reconstruct stacks from metadata
        stacks = [
            reconstruct_stack_from_metadata(cell_pool[cell_id]["stack"])
            for cell_id in selected_cell_ids
        ]

        # Try to generate a valid question
        question, answer, metadata = generate_question_and_answer(stacks)

        if question and len(answer) >= 2:  # Need at least 2 matching cells
            return question, answer, metadata, selected_cell_ids

    # Fallback: use first 16 cells with dummy question
    selected_cell_ids = list(cell_pool.keys())[:num_cells]
    question = "Click all cells where a square is stacked on top of exactly 2 circles (look for circles in lower layers)"
    answer = [0, 5, 10]
    metadata = {"fallback": True}

    return question, answer, metadata, selected_cell_ids

# -----------------------------
# Main generation
# -----------------------------

def generate_full_dataset():
    """Generate complete dataset with cell pool approach"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating Layered Stack CAPTCHA dataset...")
    print(f"  - Cell pool size: {NUM_CELLS_IN_POOL}")
    print(f"  - Number of puzzles: {NUM_PUZZLES}")
    print()

    # Step 1: Generate cell pool
    cell_pool = generate_cell_pool()

    print(f"\n✓ Cell pool generated: {len(cell_pool)} cells")
    print(f"\nGenerating {NUM_PUZZLES} puzzles from cell pool...")

    # Step 2: Generate puzzles by sampling from pool
    ground_truth = {}

    for i in range(NUM_PUZZLES):
        question, answer, metadata, cell_ids = generate_one_puzzle(i, cell_pool)

        puzzle_id = f"layered_stack_{i:04d}"

        # Store ground truth
        ground_truth[puzzle_id] = {
            "prompt": question,
            "answer": answer,
            "cells": cell_ids,
            "grid_size": list(GRID_SIZE),
            "metadata": metadata
        }

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{NUM_PUZZLES} puzzles...")

    # Save JSON files
    with open(OUTPUT_DIR / 'cell_pool.json', 'w') as f:
        json.dump(cell_pool, f, indent=2)

    with open(OUTPUT_DIR / 'ground_truth.json', 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Dataset generation complete!")
    print(f"  - Generated {NUM_CELLS_IN_POOL} cell images")
    print(f"  - Created {NUM_PUZZLES} puzzles")
    print(f"  - Saved cell_pool.json ({len(cell_pool)} entries)")
    print(f"  - Saved ground_truth.json ({len(ground_truth)} entries)")
    print(f"\nOutput location: {OUTPUT_DIR.absolute()}")

    # Print example questions
    print("\nExample questions:")
    for i, (puzzle_id, data) in enumerate(list(ground_truth.items())[:5]):
        answer_str = str(data['answer'])
        print(f"  {i+1}. {data['prompt']}")
        print(f"     Answer: {answer_str} ({len(data['answer'])} cells)")

if __name__ == "__main__":
    generate_full_dataset()
