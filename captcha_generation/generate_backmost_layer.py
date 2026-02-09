#!/usr/bin/env python3
"""
Backmost Layer CAPTCHA Generator

Generates a CAPTCHA where users must identify which shape is in the back.
- Creates three overlapping shapes in each cell
- Each shape has different color and position
- Users select cells where the BACKMOST (furthest back) shape matches the reference
- Requires depth perception and occlusion reasoning
- Pool-based architecture: generate shape variants, compose puzzles from pool
"""

import numpy as np
from PIL import Image, ImageDraw
import json
import random
from pathlib import Path

# Configuration
CELL_SIZE = 240  # Each cell is 240x240 pixels
NUM_PUZZLES = 20
GRID_SIZE = (4, 4)  # 4x4 grid shown to user
TOTAL_CELLS = GRID_SIZE[0] * GRID_SIZE[1]

# Pool configuration
NUM_REFERENCE_SHAPES = 8  # 8 different reference shapes to identify
VARIANTS_PER_SHAPE = 12  # 12 different layering configurations per reference shape
TOTAL_POOL_SIZE = NUM_REFERENCE_SHAPES * VARIANTS_PER_SHAPE  # 96 images total

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "captcha_data" / "Backmost_Layer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth.json"


def generate_shape_mask(shape_type, size=80):
    """Generate a binary mask for a shape."""
    mask = np.zeros((size, size), dtype=bool)
    center = size // 2

    if shape_type == 'circle':
        y, x = np.ogrid[:size, :size]
        circle_mask = (x - center)**2 + (y - center)**2 <= (size // 2 - 5)**2
        mask = circle_mask

    elif shape_type == 'square':
        margin = size // 6
        mask[margin:size-margin, margin:size-margin] = True

    elif shape_type == 'triangle':
        points = np.array([
            [center, margin := size // 6],
            [margin, size - margin],
            [size - margin, size - margin]
        ])
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(p) for p in points], fill=255)
        mask = np.array(img) > 128

    elif shape_type == 'rhombus':
        points = np.array([
            [center, size // 8],
            [size // 8, center],
            [center, size * 7 // 8],
            [size * 7 // 8, center]
        ])
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(p) for p in points], fill=255)
        mask = np.array(img) > 128

    elif shape_type == 'star':
        # 5-pointed star
        outer_r = size // 2 - 8
        inner_r = outer_r // 2
        points = []
        for i in range(10):
            angle = (i * 36 - 90) * np.pi / 180
            r = outer_r if i % 2 == 0 else inner_r
            x = int(center + r * np.cos(angle))
            y = int(center + r * np.sin(angle))
            points.append([x, y])
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(p) for p in points], fill=255)
        mask = np.array(img) > 128

    elif shape_type == 'hexagon':
        outer_r = size // 2 - 8
        points = []
        for i in range(6):
            angle = (i * 60 - 90) * np.pi / 180
            x = int(center + outer_r * np.cos(angle))
            y = int(center + outer_r * np.sin(angle))
            points.append([x, y])
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(p) for p in points], fill=255)
        mask = np.array(img) > 128

    elif shape_type == 'cross':
        bar_width = size // 3
        margin = (size - bar_width) // 2
        # Vertical bar
        mask[margin // 2:size - margin // 2, margin:margin + bar_width] = True
        # Horizontal bar
        mask[margin:margin + bar_width, margin // 2:size - margin // 2] = True

    elif shape_type == 'heart':
        # Simple heart shape
        y, x = np.ogrid[:size, :size]
        # Two circles for top lobes
        left_circle = ((x - size * 0.35)**2 + (y - size * 0.35)**2) <= (size * 0.2)**2
        right_circle = ((x - size * 0.65)**2 + (y - size * 0.35)**2) <= (size * 0.2)**2
        # Triangle for bottom
        points = np.array([
            [size * 0.2, size * 0.45],
            [center, size * 0.85],
            [size * 0.8, size * 0.45]
        ])
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(p) for p in points], fill=255)
        triangle_mask = np.array(img) > 128
        mask = left_circle | right_circle | triangle_mask

    return mask


def random_color():
    """Generate a distinct color."""
    colors = [
        (220, 60, 60),    # Red
        (60, 120, 220),   # Blue
        (60, 180, 60),    # Green
        (220, 180, 60),   # Yellow
        (200, 60, 200),   # Magenta
        (60, 180, 200),   # Cyan
        (220, 140, 60),   # Orange
        (140, 80, 200),   # Purple
    ]
    return random.choice(colors)


def apply_pattern_fill(mask, base_color, pattern_type):
    """
    Apply a pattern fill to a shape mask.
    Returns RGB array with pattern applied.
    """
    h, w = mask.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    if pattern_type == 'solid':
        # Solid fill
        result[mask] = base_color

    elif pattern_type == 'horizontal_stripes':
        # Horizontal stripes
        stripe_mask = np.zeros((h, w), dtype=bool)
        for i in range(0, h, 8):
            stripe_mask[i:i+4, :] = True
        combined_mask = mask & stripe_mask
        result[combined_mask] = base_color
        result[mask & ~stripe_mask] = tuple(max(0, c - 60) for c in base_color)

    elif pattern_type == 'vertical_stripes':
        # Vertical stripes
        stripe_mask = np.zeros((h, w), dtype=bool)
        for i in range(0, w, 8):
            stripe_mask[:, i:i+4] = True
        combined_mask = mask & stripe_mask
        result[combined_mask] = base_color
        result[mask & ~stripe_mask] = tuple(max(0, c - 60) for c in base_color)

    elif pattern_type == 'dots':
        # Dotted pattern
        dot_mask = np.zeros((h, w), dtype=bool)
        for i in range(4, h, 10):
            for j in range(4, w, 10):
                if i < h and j < w:
                    dot_mask[max(0, i-2):min(h, i+3), max(0, j-2):min(w, j+3)] = True
        combined_mask = mask & dot_mask
        result[combined_mask] = base_color
        result[mask & ~dot_mask] = tuple(max(0, c - 80) for c in base_color)

    elif pattern_type == 'diagonal_stripes':
        # Diagonal stripes
        stripe_mask = np.zeros((h, w), dtype=bool)
        for i in range(h):
            for j in range(w):
                if (i + j) % 12 < 6:
                    stripe_mask[i, j] = True
        combined_mask = mask & stripe_mask
        result[combined_mask] = base_color
        result[mask & ~stripe_mask] = tuple(max(0, c - 60) for c in base_color)

    elif pattern_type == 'checkered':
        # Checkered pattern
        checker_mask = np.zeros((h, w), dtype=bool)
        for i in range(0, h, 12):
            for j in range(0, w, 12):
                checker_mask[i:i+6, j:j+6] = True
                if i+6 < h and j+6 < w:
                    checker_mask[i+6:min(h, i+12), j+6:min(w, j+12)] = True
        combined_mask = mask & checker_mask
        result[combined_mask] = base_color
        result[mask & ~checker_mask] = tuple(max(0, c - 70) for c in base_color)

    return result


def create_layered_image(backmost_shape, middle_shape, frontmost_shape,
                          backmost_color, middle_color, frontmost_color,
                          backmost_pattern, middle_pattern, frontmost_pattern,
                          rng=None):
    """
    Create an image with three overlapping shapes.
    The backmost shape is what we're testing for.
    Shapes MUST overlap significantly to make depth ordering meaningful.
    Uses diverse positioning patterns while ensuring clear occlusion cues.
    """
    if rng is None:
        rng = np.random.RandomState()

    img = Image.new('RGB', (CELL_SIZE, CELL_SIZE), (240, 240, 240))

    # Generate masks for all three shapes - make them LARGER for better overlap
    shape_size = 100
    backmost_mask = generate_shape_mask(backmost_shape, shape_size)
    middle_mask = generate_shape_mask(middle_shape, shape_size)
    frontmost_mask = generate_shape_mask(frontmost_shape, shape_size)

    # CRITICAL: Each shape must have a UNIQUELY VISIBLE portion
    # Not just the backmost - ALL THREE shapes need visible parts
    # Otherwise humans cannot tell which hidden shape is backmost vs middle

    # Strategy: Use a "fan" or "cascade" arrangement where each layer
    # is offset in a consistent direction, ensuring all have visible portions

    center = CELL_SIZE // 2

    # Choose a random cascade direction (8 directions)
    cascade_patterns = [
        # (dx_back_to_mid, dy_back_to_mid, dx_mid_to_front, dy_mid_to_front)
        (30, 30, 30, 30),      # Lower-right cascade
        (-30, -30, -30, -30),  # Upper-left cascade
        (30, -30, 30, -30),    # Upper-right cascade
        (-30, 30, -30, 30),    # Lower-left cascade
        (35, 0, 35, 0),        # Right cascade
        (-35, 0, -35, 0),      # Left cascade
        (0, 35, 0, 35),        # Down cascade
        (0, -35, 0, -35),      # Up cascade
        (30, 18, 18, 30),      # Right-down L
        (-30, -18, -18, -30),  # Left-up L
        (18, 30, 30, 18),      # Down-right L
        (-18, -30, -30, -18),  # Up-left L
    ]

    pattern_idx = rng.randint(0, len(cascade_patterns))
    base_dx1, base_dy1, base_dx2, base_dy2 = cascade_patterns[pattern_idx]

    # Ensure all positions are within bounds
    def clamp_position(offset, size, canvas_size):
        """Ensure shape stays within canvas bounds"""
        y, x = offset
        y = max(5, min(y, canvas_size - size - 5))
        x = max(5, min(x, canvas_size - size - 5))
        return (y, x)

    # Target visibility range: 40-50% for BOTH backmost and middle
    MIN_VIS, MAX_VIS = 0.40, 0.50

    # Try many random configurations until we find one where BOTH layers are 40-50% visible
    best_config = None
    best_score = float('inf')

    for attempt in range(50):
        # Try different scales
        scale = 0.6 + (attempt % 10) * 0.1  # 0.6, 0.7, 0.8, ... 1.5

        # Pick random cascade pattern for each attempt
        pattern_idx = rng.randint(0, len(cascade_patterns))
        base_dx1, base_dy1, base_dx2, base_dy2 = cascade_patterns[pattern_idx]

        dx1 = int(base_dx1 * scale) + rng.randint(-8, 8)
        dy1 = int(base_dy1 * scale) + rng.randint(-8, 8)
        dx2 = int(base_dx2 * scale) + rng.randint(-8, 8)
        dy2 = int(base_dy2 * scale) + rng.randint(-8, 8)

        back_start_x = rng.randint(-20, 20)
        back_start_y = rng.randint(-20, 20)

        back_offset = (center + back_start_y - shape_size // 2,
                       center + back_start_x - shape_size // 2)
        middle_offset = (center + back_start_y + dy1 - shape_size // 2,
                         center + back_start_x + dx1 - shape_size // 2)
        front_offset = (center + back_start_y + dy1 + dy2 - shape_size // 2,
                        center + back_start_x + dx1 + dx2 - shape_size // 2)

        back_offset = clamp_position(back_offset, shape_size, CELL_SIZE)
        middle_offset = clamp_position(middle_offset, shape_size, CELL_SIZE)
        front_offset = clamp_position(front_offset, shape_size, CELL_SIZE)

        # Create full-size masks
        full_backmost = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
        full_middle = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
        full_frontmost = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)

        by, bx = back_offset
        full_backmost[by:by+shape_size, bx:bx+shape_size] = backmost_mask
        my, mx = middle_offset
        full_middle[my:my+shape_size, mx:mx+shape_size] = middle_mask
        fy, fx = front_offset
        full_frontmost[fy:fy+shape_size, fx:fx+shape_size] = frontmost_mask

        # Calculate visibility for BOTH layers
        backmost_total = np.sum(full_backmost)
        backmost_occluded = np.sum(full_backmost & (full_middle | full_frontmost))
        backmost_vis = (backmost_total - backmost_occluded) / backmost_total if backmost_total > 0 else 0

        middle_total = np.sum(full_middle)
        middle_occluded = np.sum(full_middle & full_frontmost)
        middle_vis = (middle_total - middle_occluded) / middle_total if middle_total > 0 else 0

        # Calculate overlap regions - these should be visually distinct
        # overlap_12: where backmost is covered by middle (shows 1-2 layer relationship)
        # overlap_23: where middle is covered by front (shows 2-3 layer relationship)
        overlap_12 = full_backmost & full_middle  # Layer 1-2 overlap region
        overlap_23 = full_middle & full_frontmost  # Layer 2-3 overlap region
        triple_overlap = full_backmost & full_middle & full_frontmost  # All 3 stacked

        # We want the triple overlap to be small relative to each pairwise overlap
        # This ensures the 1-2 and 2-3 relationships are visible in distinct areas
        overlap_12_area = np.sum(overlap_12)
        overlap_23_area = np.sum(overlap_23)
        triple_area = np.sum(triple_overlap)

        # Calculate how much of each pairwise overlap is "clean" (not triple-stacked)
        # We want at least 50% of each overlap region to NOT be triple-stacked
        MAX_TRIPLE_RATIO = 0.50
        overlap_12_clean = (overlap_12_area - triple_area) / overlap_12_area if overlap_12_area > 0 else 1
        overlap_23_clean = (overlap_23_area - triple_area) / overlap_23_area if overlap_23_area > 0 else 1

        # Check if BOTH visibility and overlap separation are good
        vis_ok = MIN_VIS <= backmost_vis <= MAX_VIS and MIN_VIS <= middle_vis <= MAX_VIS
        overlap_ok = overlap_12_clean >= (1 - MAX_TRIPLE_RATIO) and overlap_23_clean >= (1 - MAX_TRIPLE_RATIO)

        if vis_ok and overlap_ok:
            break

        # Track best attempt (closest to target range + good overlap separation)
        target = (MIN_VIS + MAX_VIS) / 2  # 0.45
        vis_score = abs(backmost_vis - target) + abs(middle_vis - target)
        overlap_score = max(0, MAX_TRIPLE_RATIO - overlap_12_clean) + max(0, MAX_TRIPLE_RATIO - overlap_23_clean)
        score = vis_score + overlap_score * 0.5  # Weight visibility more
        if score < best_score:
            best_score = score
            best_config = (back_offset, middle_offset, front_offset, by, bx, my, mx, fy, fx,
                          full_backmost.copy(), full_middle.copy(), full_frontmost.copy(),
                          backmost_vis, middle_vis)

    # If no perfect match found, use best attempt
    if not (vis_ok and overlap_ok):
        if best_config:
            (back_offset, middle_offset, front_offset, by, bx, my, mx, fy, fx,
             full_backmost, full_middle, full_frontmost, backmost_vis, middle_vis) = best_config

    # Convert to numpy array for manipulation
    img_array = np.array(img)

    # Apply pattern fills to each shape
    backmost_filled = apply_pattern_fill(backmost_mask, backmost_color, backmost_pattern)
    middle_filled = apply_pattern_fill(middle_mask, middle_color, middle_pattern)
    frontmost_filled = apply_pattern_fill(frontmost_mask, frontmost_color, frontmost_pattern)

    # Draw layers from back to front with patterns
    # 1. Backmost layer
    img_array[by:by+shape_size, bx:bx+shape_size][backmost_mask] = \
        backmost_filled[backmost_mask]

    # 2. Middle layer (overwrites backmost where they overlap)
    img_array[my:my+shape_size, mx:mx+shape_size][middle_mask] = \
        middle_filled[middle_mask]

    # 3. Frontmost layer (overwrites everything where it overlaps)
    img_array[fy:fy+shape_size, fx:fx+shape_size][frontmost_mask] = \
        frontmost_filled[frontmost_mask]

    # Add thin black outlines for clarity
    from scipy.ndimage import binary_dilation

    # Outline for each shape
    for mask, offset in [(full_backmost, back_offset),
                          (full_middle, middle_offset),
                          (full_frontmost, front_offset)]:
        outline = binary_dilation(mask) & ~mask
        img_array[outline] = (0, 0, 0)  # Black outline

    return Image.fromarray(img_array), backmost_vis, middle_vis


def generate_cell_pool():
    """
    Generate pool of layered shape images.
    Each image has 3 overlapping shapes, and we track which is backmost.
    """
    print("Generating pool of backmost layer images...")

    all_shapes = ['circle', 'square', 'triangle', 'rhombus', 'star', 'hexagon', 'cross', 'heart']

    cell_pool = {}
    cell_id_counter = 0

    # For each reference shape (the one we'll ask about)
    for ref_shape_idx, reference_shape in enumerate(all_shapes[:NUM_REFERENCE_SHAPES]):
        print(f"\n  Generating variations with {reference_shape} as backmost...")

        # Create multiple variants with this shape as backmost
        for variant_idx in range(VARIANTS_PER_SHAPE):
            # Reference shape is always backmost
            backmost_shape = reference_shape

            # Choose two different shapes for middle and front
            other_shapes = [s for s in all_shapes if s != reference_shape]
            middle_shape, frontmost_shape = random.sample(other_shapes, 2)

            # Assign random distinct colors
            colors = random.sample([
                (220, 60, 60), (60, 120, 220), (60, 180, 60),
                (220, 180, 60), (200, 60, 200), (60, 180, 200),
                (220, 140, 60), (140, 80, 200)
            ], 3)
            backmost_color, middle_color, frontmost_color = colors

            # Assign random patterns to make it harder for LLMs
            patterns = ['solid', 'horizontal_stripes', 'vertical_stripes', 'dots',
                       'diagonal_stripes', 'checkered']
            backmost_pattern, middle_pattern, frontmost_pattern = random.sample(patterns, 3)

            # Create RNG for this variant to ensure consistent randomization
            seed = ref_shape_idx * 1000 + variant_idx
            variant_rng = np.random.RandomState(seed)

            # Generate the layered image with diverse positioning
            img, backmost_vis, middle_vis = create_layered_image(
                backmost_shape, middle_shape, frontmost_shape,
                backmost_color, middle_color, frontmost_color,
                backmost_pattern, middle_pattern, frontmost_pattern,
                rng=variant_rng
            )

            # Save image
            cell_id = f"backmost_{reference_shape}_{variant_idx}"
            img_filename = f"{cell_id}.png"
            img_path = OUTPUT_DIR / img_filename
            img.save(img_path)

            # Store metadata with visibility info
            cell_pool[cell_id] = {
                "filename": img_filename,
                "backmost_shape": backmost_shape,
                "middle_shape": middle_shape,
                "frontmost_shape": frontmost_shape,
                "backmost_color": backmost_color,
                "middle_color": middle_color,
                "frontmost_color": frontmost_color,
                "backmost_pattern": backmost_pattern,
                "middle_pattern": middle_pattern,
                "frontmost_pattern": frontmost_pattern,
                "backmost_visibility": round(backmost_vis, 3),
                "middle_visibility": round(middle_vis, 3),
            }

            cell_id_counter += 1

            # Print actual visibility for each variant
            print(f"      {cell_id}: back_vis={backmost_vis:.1%}, mid_vis={middle_vis:.1%}")

    print(f"\n✓ Generated {len(cell_pool)} layered images")

    # Save pool metadata
    pool_file = OUTPUT_DIR / "cell_pool.json"
    with open(pool_file, 'w') as f:
        json.dump(cell_pool, f, indent=2)

    print(f"✓ Cell pool metadata saved to {pool_file}")

    return cell_pool


def generate_puzzles(cell_pool):
    """
    Generate puzzles by selecting images from the pool.
    Each puzzle: pick a reference shape, select 4-6 cells where it's backmost, fill rest with other shapes.
    """
    print(f"\nGenerating {NUM_PUZZLES} puzzles...")

    all_shapes = ['circle', 'square', 'triangle', 'rhombus', 'star', 'hexagon', 'cross', 'heart']
    reference_shapes = all_shapes[:NUM_REFERENCE_SHAPES]

    # Define confusing shape pairs - when asking about one, exclude the other
    confusing_pairs = {
        'square': 'rhombus',
        'rhombus': 'square',
    }

    # Group cells by backmost shape
    cells_by_backmost = {}
    for cell_id, data in cell_pool.items():
        backmost = data["backmost_shape"]
        if backmost not in cells_by_backmost:
            cells_by_backmost[backmost] = []
        cells_by_backmost[backmost].append(cell_id)

    ground_truth = {}

    for puzzle_idx in range(NUM_PUZZLES):
        puzzle_id = f"backmost_layer_{puzzle_idx:04d}"

        # Pick a random reference shape
        reference_shape = random.choice(reference_shapes)

        # Get the confusing shape to exclude (if any)
        excluded_shape = confusing_pairs.get(reference_shape, None)

        # Get cells with this shape as backmost (target cells)
        target_cells = cells_by_backmost[reference_shape]

        # Get distractor cells (other shapes as backmost, excluding confusing shapes)
        distractor_cells = []
        for shape in reference_shapes:
            if shape != reference_shape and shape != excluded_shape:
                distractor_cells.extend(cells_by_backmost[shape])

        # Decide how many targets (4-6 out of 16)
        num_targets = random.randint(4, 6)
        num_distractors = TOTAL_CELLS - num_targets

        # Select cells
        selected_targets = random.sample(target_cells, min(num_targets, len(target_cells)))
        selected_distractors = random.sample(distractor_cells, min(num_distractors, len(distractor_cells)))

        # Combine and shuffle
        all_cells = selected_targets + selected_distractors
        random.shuffle(all_cells)

        # Ensure exactly 16 cells
        all_cells = all_cells[:TOTAL_CELLS]

        # Find correct indices
        correct_indices = [i for i, cell_id in enumerate(all_cells)
                          if cell_pool[cell_id]["backmost_shape"] == reference_shape]

        # Pick one target cell as reference
        reference_cell = random.choice(selected_targets)

        # Create puzzle
        prompt = f"Click all cells where the BACKMOST (furthest back) shape is a {reference_shape}. The backmost shape is partially hidden behind the other shapes."

        puzzle_data = {
            "prompt": prompt,
            "description": f"Identify cells with {reference_shape} as backmost layer",
            "reference_cell": reference_cell,
            "reference_image": cell_pool[reference_cell]["filename"],
            "reference_shape": reference_shape,
            "cells": all_cells,
            "answer": sorted(correct_indices),
            "input_type": "grid_select",
            "grid_size": GRID_SIZE,
        }

        ground_truth[puzzle_id] = puzzle_data

        print(f"✓ {puzzle_id}: Reference={reference_shape}, {len(correct_indices)} matches")

    # Save ground truth
    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Successfully generated {NUM_PUZZLES} Backmost Layer puzzles")
    print(f"✓ Ground truth saved to {GROUND_TRUTH_FILE}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("Backmost Layer CAPTCHA Generator")
    print("=" * 60)

    # Generate cell pool
    cell_pool = generate_cell_pool()

    # Generate puzzles
    generate_puzzles(cell_pool)


if __name__ == "__main__":
    main()
