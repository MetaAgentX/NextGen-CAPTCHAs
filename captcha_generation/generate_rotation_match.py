#!/usr/bin/env python3
"""
Rotation Match CAPTCHA Generator

Generates a CAPTCHA where users must identify shapes that are the same but rotated.
- Creates strange irregular shapes
- Rotates them by random angles (0-360 degrees)
- Users select all tiles showing the same shape (rotation-invariant)
- 4x4 grid with one target shape appearing multiple times at different rotations
- Pool-based architecture: generate shape variants, compose puzzles from pool
"""

import numpy as np
from PIL import Image
import json
import random
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import rotate as scipy_rotate

# Configuration
MASK_SIZE = 60  # Generate shape masks at 60x60 pixels (larger for better human recognition)
OUTPUT_SIZE = 240  # Final output images at 240x240 pixels for fine-grained textures
NUM_PUZZLES = 20
GRID_SIZE = (4, 4)  # 4x4 grid shown to user
TOTAL_CELLS = GRID_SIZE[0] * GRID_SIZE[1]

# Pool configuration
NUM_UNIQUE_SHAPES = 30  # 30 different unique shapes
ROTATIONS_PER_SHAPE = 6  # 6 different rotations per shape
TOTAL_POOL_SIZE = NUM_UNIQUE_SHAPES * ROTATIONS_PER_SHAPE  # 180 images total

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "captcha_data" / "Rotation_Match"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth.json"


def random_color():
    """Generate a high-contrast RGB color."""
    high_contrast_colors = [
        (220, 20, 20),    # Bright red
        (20, 20, 220),    # Bright blue
        (20, 200, 20),    # Bright green
        (220, 180, 20),   # Yellow/gold
        (220, 20, 220),   # Magenta
        (20, 200, 220),   # Cyan
        (220, 100, 20),   # Orange
        (140, 20, 220),   # Purple
        (20, 140, 100),   # Teal
        (220, 20, 100),   # Pink/rose
        (100, 220, 20),   # Lime green
        (220, 140, 180),  # Light pink
        (100, 180, 220),  # Sky blue
        (180, 100, 20),   # Brown/rust
        (20, 100, 180),   # Deep blue
    ]
    return random.choice(high_contrast_colors)


# Discrete rotation angles (45° increments) - easier for humans to mentally rotate
DISCRETE_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


def generate_paper_texture_background(size):
    """
    Generate a realistic paper/parchment texture background.
    Has fiber-like patterns and speckles that humans easily ignore,
    but adds high-frequency noise that disrupts VLM edge detection.

    Args:
        size: Size of the background (size x size)

    Returns:
        RGB numpy array with paper textured background
    """
    # Base paper color - warm cream/off-white
    base_r, base_g, base_b = 245, 240, 232

    # Layer 1: Large-scale color variation (warm/cool shifts)
    large_noise = np.random.randn(size, size)
    large_noise = ndimage.gaussian_filter(large_noise, sigma=size/4)
    large_noise = (large_noise - large_noise.min()) / (large_noise.max() - large_noise.min())
    large_variation = (large_noise - 0.5) * 15  # +/- 7.5 intensity

    # Layer 2: Medium-scale fiber texture (directional noise)
    # Create horizontal fiber-like patterns
    fiber_h = np.random.randn(size, size)
    fiber_h = ndimage.gaussian_filter1d(fiber_h, sigma=1.5, axis=1)  # Blur horizontally
    fiber_h = ndimage.gaussian_filter1d(fiber_h, sigma=0.5, axis=0)  # Less blur vertically

    # Create vertical fiber-like patterns
    fiber_v = np.random.randn(size, size)
    fiber_v = ndimage.gaussian_filter1d(fiber_v, sigma=0.5, axis=1)
    fiber_v = ndimage.gaussian_filter1d(fiber_v, sigma=1.5, axis=0)

    # Combine fibers with slight angle variation
    fiber_texture = (fiber_h * 0.6 + fiber_v * 0.4) * 8  # Subtle fiber effect

    # Layer 3: Fine grain noise (high frequency detail that confuses VLMs)
    fine_noise = np.random.randn(size, size) * 6
    fine_noise = ndimage.gaussian_filter(fine_noise, sigma=0.5)

    # Layer 4: Random speckles (small dark spots like paper imperfections)
    speckles = np.random.random((size, size))
    speckle_mask = speckles < 0.003  # ~0.3% of pixels are speckles
    speckle_intensity = np.zeros((size, size))
    speckle_intensity[speckle_mask] = np.random.uniform(-30, -15, np.sum(speckle_mask))
    # Blur speckles slightly so they're not single pixels
    speckle_intensity = ndimage.gaussian_filter(speckle_intensity, sigma=0.8)

    # Combine all layers
    combined = large_variation + fiber_texture + fine_noise + speckle_intensity

    # Create RGB channels with slight color variation
    # Paper has slightly more warmth in shadows, cooler in highlights
    r_channel = np.clip(base_r + combined + large_variation * 0.3, 215, 255).astype(np.uint8)
    g_channel = np.clip(base_g + combined, 210, 250).astype(np.uint8)
    b_channel = np.clip(base_b + combined - large_variation * 0.2, 200, 245).astype(np.uint8)

    background = np.stack([r_channel, g_channel, b_channel], axis=-1)

    return background


def scale_mask(mask, scale_factor=4):
    """Scale up a binary mask using nearest neighbor interpolation."""
    from scipy.ndimage import zoom
    return zoom(mask.astype(float), scale_factor, order=0) > 0.5


def apply_pattern_fill(mask, base_color):
    """Apply a pattern/texture fill to a mask region."""
    h, w = mask.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    # Choose pattern type - use clear, bold patterns
    pattern_type = random.choice([
        'solid',
        'horizontal',
        'vertical',
        'diagonal',
        'checkerboard',
        'dots',
    ])

    r, g, b = base_color

    if pattern_type == 'solid':
        result[mask] = base_color

    elif pattern_type == 'horizontal':
        stripe_width = random.randint(5, 8)
        for y in range(h):
            if (y // stripe_width) % 2 == 0:
                result[y, :] = base_color
            else:
                result[y, :] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'vertical':
        stripe_width = random.randint(5, 8)
        for x in range(w):
            if (x // stripe_width) % 2 == 0:
                result[:, x] = base_color
            else:
                result[:, x] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'diagonal':
        stripe_width = random.randint(6, 10)
        for y in range(h):
            for x in range(w):
                if ((x + y) // stripe_width) % 2 == 0:
                    result[y, x] = base_color
                else:
                    result[y, x] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'checkerboard':
        square_size = random.randint(8, 12)
        for y in range(h):
            for x in range(w):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    result[y, x] = base_color
                else:
                    result[y, x] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'dots':
        result[mask] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        dot_spacing = random.randint(10, 14)
        dot_radius = random.randint(3, 5)
        for y in range(0, h, dot_spacing):
            for x in range(0, w, dot_spacing):
                for dy in range(-dot_radius, dot_radius + 1):
                    for dx in range(-dot_radius, dot_radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and dx*dx + dy*dy <= dot_radius*dot_radius:
                            result[ny, nx] = base_color
        result[~mask] = (0, 0, 0)

    return result


def generate_strange_shape_mask(size=60):
    """
    Generate a very distinctive, irregular shape using noise-based generation.
    The shape should be easily distinguishable and recognizable even when rotated.
    Ensures the shape stays within a safe inner region to avoid exceeding
    boundaries after rotation.

    Returns:
        Binary mask (numpy array) where shape pixels are True
    """
    # Define safe inner region (leave margin for rotation)
    # For a square that can be rotated, inscribed circle has radius = size/(2*sqrt(2))
    # We use a margin to ensure shapes don't exceed boundaries when rotated
    margin = int(size * 0.15)  # 15% margin on each side
    inner_size = size - 2 * margin

    # Use multiple layers of noise for complex, distinguishable shapes
    noise1 = np.random.randn(inner_size, inner_size)
    noise2 = np.random.randn(inner_size, inner_size)
    noise3 = np.random.randn(inner_size, inner_size)

    # Combine noise at different scales for more distinctive features
    # Use varied sigma values for more unique shapes
    sigma1 = random.uniform(inner_size/5, inner_size/4)
    sigma2 = random.uniform(inner_size/10, inner_size/8)
    sigma3 = random.uniform(inner_size/15, inner_size/12)

    blurred1 = ndimage.gaussian_filter(noise1, sigma=sigma1)
    blurred2 = ndimage.gaussian_filter(noise2, sigma=sigma2)
    blurred3 = ndimage.gaussian_filter(noise3, sigma=sigma3)

    # Combine with varied weights for more diversity
    weight1 = random.uniform(0.4, 0.6)
    weight2 = random.uniform(0.2, 0.4)
    weight3 = 1.0 - weight1 - weight2
    combined = weight1 * blurred1 + weight2 * blurred2 + weight3 * blurred3

    # More varied threshold for different shape sizes
    threshold = np.percentile(combined, random.randint(30, 55))
    inner_mask = combined > threshold

    # Apply stronger morphological operations for more distinctive shapes
    morph_choice = random.choice(['erode', 'dilate', 'open', 'close', 'erode_dilate', 'dilate_erode'])
    if morph_choice == 'erode':
        inner_mask = ndimage.binary_erosion(inner_mask, iterations=random.randint(2, 3))
    elif morph_choice == 'dilate':
        inner_mask = ndimage.binary_dilation(inner_mask, iterations=random.randint(2, 3))
    elif morph_choice == 'open':
        inner_mask = ndimage.binary_opening(inner_mask, iterations=random.randint(1, 2))
    elif morph_choice == 'close':
        inner_mask = ndimage.binary_closing(inner_mask, iterations=random.randint(2, 3))
    elif morph_choice == 'erode_dilate':
        inner_mask = ndimage.binary_erosion(inner_mask, iterations=2)
        inner_mask = ndimage.binary_dilation(inner_mask, iterations=3)
    elif morph_choice == 'dilate_erode':
        inner_mask = ndimage.binary_dilation(inner_mask, iterations=3)
        inner_mask = ndimage.binary_erosion(inner_mask, iterations=2)

    # Ensure single connected component
    labeled, num_features = ndimage.label(inner_mask)
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())[1:]
        if len(component_sizes) > 0:
            largest_component = np.argmax(component_sizes) + 1
            inner_mask = (labeled == largest_component)

    # Ensure reasonable size and not too simple
    shape_area = np.sum(inner_mask)
    total_area = inner_size * inner_size
    if shape_area < total_area * 0.15 or shape_area > total_area * 0.70:  # Too small or too large
        return generate_strange_shape_mask(size)

    # Place the inner shape in the center of the full-size mask with margin
    mask = np.zeros((size, size), dtype=bool)
    mask[margin:margin+inner_size, margin:margin+inner_size] = inner_mask

    return mask


def rotate_shape_mask(mask, angle):
    """
    Rotate a shape mask by a given angle.

    Args:
        mask: Binary mask (numpy array)
        angle: Rotation angle in degrees (0-360)

    Returns:
        Rotated binary mask
    """
    # Rotate using scipy with order=0 (nearest neighbor) to keep binary
    rotated = scipy_rotate(mask.astype(float), angle, reshape=False, order=0, mode='constant', cval=0)
    return rotated > 0.5


def generate_shape_pool():
    """
    Generate pool of shape variants (unique shapes at different rotations).

    Returns:
        Dictionary mapping cell_id to shape metadata
    """
    print("Generating pool of shape variants...")

    cell_pool = {}

    for shape_idx in range(NUM_UNIQUE_SHAPES):
        # Generate one unique shape
        shape_mask = generate_strange_shape_mask(MASK_SIZE)

        # Create multiple rotated versions of this shape
        for rot_idx in range(ROTATIONS_PER_SHAPE):
            cell_id = f"shape_{shape_idx}_rot_{rot_idx}"

            # Use discrete rotation angles (45° increments) - easier for humans
            angle = random.choice(DISCRETE_ANGLES)

            # Rotate the mask
            rotated_mask = rotate_shape_mask(shape_mask, angle)

            # Use paper texture background - fiber patterns and speckles
            # Humans ignore it easily, but high-frequency noise disrupts VLM edge detection
            shape_color = random_color()

            # Scale up to output size
            mask_scaled = scale_mask(rotated_mask, OUTPUT_SIZE // MASK_SIZE)

            # Create image with paper textured background
            img_array = generate_paper_texture_background(OUTPUT_SIZE)

            # Apply shape pattern on top of textured background
            shape_pattern = apply_pattern_fill(mask_scaled, shape_color)
            img_array[mask_scaled] = shape_pattern[mask_scaled]

            img = Image.fromarray(img_array)

            # Save image
            filename = f"{cell_id}.png"
            img_path = OUTPUT_DIR / filename
            img.save(img_path)

            # Store in pool
            cell_pool[cell_id] = {
                "filename": filename,
                "shape_id": shape_idx,  # Which unique shape this is
                "rotation_angle": angle,
            }

        if (shape_idx + 1) % 10 == 0:
            print(f"  Generated shapes: {shape_idx + 1}/{NUM_UNIQUE_SHAPES}")

    print(f"✓ Generated {len(cell_pool)} shape variants")

    # Save pool metadata
    pool_file = OUTPUT_DIR / "cell_pool.json"
    with open(pool_file, 'w') as f:
        json.dump(cell_pool, f, indent=2)

    print(f"✓ Cell pool metadata saved to {pool_file}")

    return cell_pool


def generate_puzzles(cell_pool):
    """
    Generate puzzles by selecting shapes from the pool.
    Each puzzle has one target shape appearing most frequently (4-7 times),
    and distractor shapes appearing less frequently (1-3 times each).
    User must click all instances of the MOST COMMON shape.
    """
    print(f"\nGenerating {NUM_PUZZLES} puzzles...")

    # Group cells by shape_id
    shapes_by_id = {}
    for cell_id, data in cell_pool.items():
        shape_id = data["shape_id"]
        if shape_id not in shapes_by_id:
            shapes_by_id[shape_id] = []
        shapes_by_id[shape_id].append(cell_id)

    ground_truth = {}

    puzzle_idx = 0
    while puzzle_idx < NUM_PUZZLES:
        puzzle_id = f"rotation_match_{puzzle_idx:04d}"

        # Pick a random target shape (the one that will appear most)
        target_shape_id = random.randint(0, NUM_UNIQUE_SHAPES - 1)

        # Get all rotations of target shape
        target_cells = shapes_by_id[target_shape_id]

        # Decide how many target instances (4-7 out of 16) - this will be the MOST
        num_targets = random.randint(4, 7)

        # Select target cells
        selected_targets = random.sample(target_cells, min(num_targets, len(target_cells)))

        # For remaining slots, add 4-5 other shapes with fewer instances each
        # More distractors but each very different - easier for humans to distinguish
        remaining_slots = TOTAL_CELLS - num_targets
        num_distractor_shapes = random.randint(4, 5)

        # Pick distractor shape IDs - ensure they're visually different by picking
        # shapes that are spread out in the shape_id space
        available_distractors = [sid for sid in range(NUM_UNIQUE_SHAPES) if sid != target_shape_id]
        distractor_shape_ids = random.sample(
            available_distractors,
            min(num_distractor_shapes, len(available_distractors))
        )

        # Distribute remaining slots among distractors, ensuring each has LESS than target
        selected_distractors = []
        slots_per_distractor = remaining_slots // num_distractor_shapes
        max_per_distractor = min(slots_per_distractor, num_targets - 1)  # Must be less than target

        for sid in distractor_shape_ids:
            # Each distractor gets 1 to max_per_distractor instances (typically 1-2 each)
            num_instances = random.randint(1, max(1, max_per_distractor))
            distractor_cells = shapes_by_id[sid]
            selected = random.sample(distractor_cells, min(num_instances, len(distractor_cells)))
            selected_distractors.extend(selected)

        # Combine all cells
        all_cells = selected_targets + selected_distractors

        # Pad to 16 if needed (unlikely but just in case)
        while len(all_cells) < TOTAL_CELLS:
            filler_shape_id = random.choice([sid for sid in range(NUM_UNIQUE_SHAPES) if sid != target_shape_id])
            all_cells.append(random.choice(shapes_by_id[filler_shape_id]))

        all_cells = all_cells[:TOTAL_CELLS]

        # Shuffle the grid
        random.shuffle(all_cells)

        # Find correct indices (all instances of target shape - the most common one)
        correct_indices = [i for i, cell_id in enumerate(all_cells)
                          if cell_pool[cell_id]["shape_id"] == target_shape_id]

        # Verify target is actually the most common
        shape_counts = {}
        for cell_id in all_cells:
            sid = cell_pool[cell_id]["shape_id"]
            shape_counts[sid] = shape_counts.get(sid, 0) + 1

        max_count = max(shape_counts.values())
        most_common_shapes = [sid for sid, count in shape_counts.items() if count == max_count]

        # If there's a tie, regenerate this puzzle
        if len(most_common_shapes) > 1 or most_common_shapes[0] != target_shape_id:
            print(f"  Regenerating {puzzle_id} due to count tie...")
            continue

        # Create ground truth
        prompt = f"Click all tiles showing the shape that appears MOST FREQUENTLY in the grid (same shape even if rotated or styled differently)."

        puzzle_data = {
            "prompt": prompt,
            "description": f"Identify all instances of the most common shape (rotation and color/texture invariant)",
            "target_shape_id": target_shape_id,
            "cells": all_cells,
            "answer": sorted(correct_indices),
            "input_type": "grid_select",
            "grid_size": GRID_SIZE,
        }

        ground_truth[puzzle_id] = puzzle_data

        print(f"✓ {puzzle_id}: Most common shape appears {len(correct_indices)} times")

        puzzle_idx += 1

    # Save ground truth
    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Successfully generated {len(ground_truth)} Rotation Match puzzles")
    print(f"✓ Ground truth saved to {GROUND_TRUTH_FILE}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("Rotation Match CAPTCHA Generator")
    print("=" * 60)

    # Generate shape pool
    cell_pool = generate_shape_pool()

    # Generate puzzles
    generate_puzzles(cell_pool)


if __name__ == "__main__":
    main()
