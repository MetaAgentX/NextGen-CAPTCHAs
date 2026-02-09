#!/usr/bin/env python3
"""
Spooky CAPTCHA Sample Visualization Generator

Generates static PNG images showing the hidden content boundaries
for all spooky captcha puzzles using red edge overlays on noise.

Creates one visualization per puzzle (20 per type, 120 total).
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.ndimage import binary_dilation
import json
import os
import random
import math
from pathlib import Path


# === Configuration ===
BASE_DIR = Path(__file__).parent.parent / "captcha_data"
SAMPLES_PER_TYPE = 20


# === Core Utility Functions ===

def generate_mid_frequency_noise(height, width, sigma=3.0):
    """Generate mid-spatial frequency noise."""
    noise = np.random.randn(height, width)
    filtered_noise = ndimage.gaussian_filter(noise, sigma=sigma)
    filtered_noise = (filtered_noise - filtered_noise.min()) / (filtered_noise.max() - filtered_noise.min())
    return filtered_noise


def generate_visualization_background(height, width, seed=42):
    """Generate a single-frame noise background."""
    np.random.seed(seed)
    noise = generate_mid_frequency_noise(height, width, sigma=3.0)

    base_luminance = 128.0
    noise_amplitude = 70.0
    background = base_luminance + noise_amplitude * (noise - 0.5) * 2.0
    background = np.clip(background, 0, 255).astype(np.uint8)

    return background


def compute_edges(mask, threshold=0.5):
    """
    Compute thin 1-pixel edges using binary erosion.
    """
    from scipy.ndimage import binary_erosion
    # Create binary mask
    binary = (mask > threshold).astype(np.uint8)
    # Erode to get inner boundary
    eroded = binary_erosion(binary, iterations=3)
    # Edge is the difference (3 pixels wide)
    edges = binary - eroded
    return edges.astype(np.float32)


def create_visualization(mask, width, height, output_path):
    """
    Create visualization PNG with noise background and red edge overlay.
    """
    background = generate_visualization_background(height, width)
    edges = compute_edges(mask, threshold=0.1)

    img_array = np.stack([background, background, background], axis=-1)

    edge_mask = edges > 0
    img_array[edge_mask, 0] = 255  # Red channel
    img_array[edge_mask, 1] = 0    # Green channel
    img_array[edge_mask, 2] = 0    # Blue channel

    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(output_path, 'PNG')


# === Shape Mask Functions ===

def create_text_mask(text, width, height, font_size=160):
    """Create a binary mask for text."""
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    font_paths = [
        '/System/Library/Fonts/Helvetica.ttc',
        '/Library/Fonts/Arial Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        'C:\\Windows\\Fonts\\arialbd.ttf',
    ]

    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill=255, font=font)

    mask_array = np.array(mask_img).astype(np.float32) / 255.0
    mask_array = ndimage.gaussian_filter(mask_array, sigma=2.0)

    return mask_array


def create_circle_mask(cx, cy, radius, width, height, edge_falloff=10.0):
    """Create a soft-edged circle mask."""
    y_coords, x_coords = np.ogrid[:height, :width]
    distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    mask = np.clip((radius - distance) / edge_falloff, 0, 1)
    return mask


def create_shape_mask(shape_type, cx, cy, radius, width, height):
    """Create a mask for a single shape (circle, square, triangle)."""
    if shape_type == 'circle':
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        mask = (distance <= radius).astype(float)
    elif shape_type == 'square':
        y_coords, x_coords = np.ogrid[:height, :width]
        mask = ((np.abs(x_coords - cx) <= radius) & (np.abs(y_coords - cy) <= radius)).astype(float)
    elif shape_type == 'triangle':
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        points = [
            (cx, cy - radius),
            (cx - radius, cy + radius),
            (cx + radius, cy + radius)
        ]
        draw.polygon(points, fill=255)
        mask = np.array(img).astype(float) / 255.0
    elif shape_type == 'pentagon':
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        points = []
        for i in range(5):
            angle = math.pi * 2 * i / 5 - math.pi / 2
            px = cx + radius * math.cos(angle)
            py = cy + radius * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=255)
        mask = np.array(img).astype(float) / 255.0
    elif shape_type == 'star':
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        points = []
        for i in range(10):
            angle = math.pi * 2 * i / 10 - math.pi / 2
            r = radius if i % 2 == 0 else radius * 0.4
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=255)
        mask = np.array(img).astype(float) / 255.0
    else:
        mask = np.zeros((height, width), dtype=float)

    # Smooth the mask edges
    mask = ndimage.gaussian_filter(mask, sigma=2.0)
    mask = np.clip(mask, 0, 1)

    return mask


def create_shape_pattern(width, height, seed):
    """
    Create a pattern of shapes for jigsaw content.
    Replicates the logic from generate_spooky_jigsaw.py.
    """
    np.random.seed(seed)
    mask = np.zeros((height, width), dtype=float)
    shape_types = ['circle', 'square', 'triangle', 'pentagon', 'star']

    grid_rows = 3
    grid_cols = 3
    cell_width = width // grid_cols
    cell_height = height // grid_rows

    for row in range(grid_rows):
        for col in range(grid_cols):
            cell_left = col * cell_width
            cell_top = row * cell_height
            cell_right = cell_left + cell_width
            cell_bottom = cell_top + cell_height

            margin = 20
            shape_type = np.random.choice(shape_types)
            cx = np.random.randint(cell_left + margin, cell_right - margin)
            cy = np.random.randint(cell_top + margin, cell_bottom - margin)
            size = np.random.randint(40, 70)

            shape_mask = create_shape_mask(shape_type, cx, cy, size, width, height)
            mask = np.maximum(mask, shape_mask)

    num_extra = np.random.randint(3, 7)
    for _ in range(num_extra):
        shape_type = np.random.choice(shape_types)
        cx = np.random.randint(40, width - 40)
        cy = np.random.randint(40, height - 40)
        size = np.random.randint(35, 65)

        shape_mask = create_shape_mask(shape_type, cx, cy, size, width, height)
        mask = np.maximum(mask, shape_mask)

    mask = ndimage.gaussian_filter(mask, sigma=2.0)
    mask = np.clip(mask, 0, 1)

    return mask


# === Per-Type Mask Reconstruction ===

def reconstruct_spooky_text_mask(puzzle_idx, ground_truth):
    """Reconstruct text mask for visualization."""
    filename = f"spooky_text_{puzzle_idx:04d}.gif"
    text = ground_truth[filename]["answer"].upper()

    width, height = 600, 250
    font_size = min(160, int(height * 0.8))

    text_mask = create_text_mask(text, width, height, font_size)
    return text_mask, width, height


def reconstruct_spooky_circle_mask(puzzle_idx, ground_truth):
    """Reconstruct circle masks for visualization."""
    output_path = f"spooky_{puzzle_idx:04d}.gif"
    seed = hash(output_path) % 2**32
    np.random.seed(seed)

    key = f"spooky_circle_{puzzle_idx:04d}"
    num_circles = ground_truth[key]["answer"]

    width, height = 400, 400

    def circles_overlap(c1_center, c1_radius, c2_center, c2_radius, min_spacing=20):
        cx1, cy1 = c1_center
        cx2, cy2 = c2_center
        distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        return distance < (c1_radius + c2_radius + min_spacing)

    circles = []
    max_attempts = 100

    for i in range(num_circles):
        placed = False
        for attempt in range(max_attempts):
            cx = np.random.randint(50, width - 50)
            cy = np.random.randint(50, height - 50)
            radius = np.random.randint(30, 50)

            overlaps = False
            for existing_circle in circles:
                if circles_overlap(
                    (cx, cy), radius,
                    existing_circle['center'], existing_circle['radius'],
                    min_spacing=20
                ):
                    overlaps = True
                    break

            if not overlaps:
                circles.append({
                    'center': (cx, cy),
                    'radius': radius
                })
                placed = True
                break

    combined_mask = np.zeros((height, width), dtype=np.float32)
    for circle in circles:
        cx, cy = circle['center']
        radius = circle['radius']
        mask = create_circle_mask(cx, cy, radius, width, height, edge_falloff=10.0)
        combined_mask = np.maximum(combined_mask, mask)

    return combined_mask, width, height


def reconstruct_spooky_circle_grid_mask(puzzle_idx, ground_truth):
    """Reconstruct circle grid mask from ground truth positions."""
    key = f"spooky_circle_grid_{puzzle_idx:04d}"
    puzzle_data = ground_truth[key]

    circle_cells = puzzle_data["circle_cells"]

    cell_size = 120
    width = height = 360
    radius = int(cell_size * 0.35)  # ~42 pixels

    combined_mask = np.zeros((height, width), dtype=np.float32)

    for row, col in circle_cells:
        center_y = row * cell_size + cell_size // 2
        center_x = col * cell_size + cell_size // 2

        mask = create_circle_mask(center_x, center_y, radius, width, height, edge_falloff=10.0)
        combined_mask = np.maximum(combined_mask, mask)

    return combined_mask, width, height


def reconstruct_spooky_shape_grid_mask(puzzle_idx, ground_truth):
    """Reconstruct shape grid mask from ground truth cell_config."""
    key = f"spooky_shape_grid_{puzzle_idx:04d}"
    puzzle_data = ground_truth[key]
    cell_config = puzzle_data["cell_config"]

    cell_size = 120
    width = height = 360
    radius = int(cell_size * 0.35)  # ~42 pixels

    combined_mask = np.zeros((height, width), dtype=np.float32)

    for cell_idx_str, config in cell_config.items():
        cell_idx = int(cell_idx_str)
        if config["is_empty"]:
            continue

        row = cell_idx // 3
        col = cell_idx % 3
        center_y = row * cell_size + cell_size // 2
        center_x = col * cell_size + cell_size // 2
        shape_type = config["shape"]

        mask = create_shape_mask(shape_type, center_x, center_y, radius, width, height)
        combined_mask = np.maximum(combined_mask, mask)

    return combined_mask, width, height


def reconstruct_spooky_size_mask(puzzle_idx, ground_truth):
    """Reconstruct size comparison mask (all shapes)."""
    key = f"spooky_size_{puzzle_idx:04d}"
    puzzle_data = ground_truth[key]

    width, height = 600, 400

    combined_mask = np.zeros((height, width), dtype=np.float32)

    # Check if all_shapes is available (new format)
    if "all_shapes" in puzzle_data:
        for shape_data in puzzle_data["all_shapes"]:
            shape_type = shape_data["shape"]
            center_x = shape_data["x"]
            center_y = shape_data["y"]
            radius = shape_data["radius"]

            mask = create_shape_mask(shape_type, center_x, center_y, radius, width, height)
            combined_mask = np.maximum(combined_mask, mask)
    else:
        # Fallback to target only (old format)
        target = puzzle_data["target_position"]
        target_shape = puzzle_data["target_shape"]
        diameter = target["diameter"]
        center_x = target["x"] + diameter // 2
        center_y = target["y"] + diameter // 2
        radius = diameter // 2
        combined_mask = create_shape_mask(target_shape, center_x, center_y, radius, width, height)

    return combined_mask, width, height


def reconstruct_spooky_jigsaw_mask(puzzle_idx):
    """Reconstruct jigsaw content mask using deterministic seed."""
    width = height = 450
    seed = puzzle_idx * 12345

    content_mask = create_shape_pattern(width, height, seed)

    return content_mask, width, height


# === Main Generation Functions ===

def generate_spooky_text_samples():
    """Generate samples for Spooky_Text."""
    type_dir = BASE_DIR / "Spooky_Text"
    samples_dir = type_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    with open(type_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    for i in range(SAMPLES_PER_TYPE):
        mask, width, height = reconstruct_spooky_text_mask(i, ground_truth)
        output_path = samples_dir / f"spooky_text_{i:04d}_viz.png"
        create_visualization(mask, width, height, output_path)
        print(f"  Generated: {output_path.name}")


def generate_spooky_circle_samples():
    """Generate samples for Spooky_Circle."""
    type_dir = BASE_DIR / "Spooky_Circle"
    samples_dir = type_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    with open(type_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    for i in range(SAMPLES_PER_TYPE):
        mask, width, height = reconstruct_spooky_circle_mask(i, ground_truth)
        output_path = samples_dir / f"spooky_{i:04d}_viz.png"
        create_visualization(mask, width, height, output_path)
        print(f"  Generated: {output_path.name}")


def generate_spooky_circle_grid_samples():
    """Generate samples for Spooky_Circle_Grid."""
    type_dir = BASE_DIR / "Spooky_Circle_Grid"
    samples_dir = type_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    with open(type_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    for i in range(SAMPLES_PER_TYPE):
        mask, width, height = reconstruct_spooky_circle_grid_mask(i, ground_truth)
        output_path = samples_dir / f"spooky_circle_grid_{i:04d}_viz.png"
        create_visualization(mask, width, height, output_path)
        print(f"  Generated: {output_path.name}")


def generate_spooky_shape_grid_samples():
    """Generate samples for Spooky_Shape_Grid."""
    type_dir = BASE_DIR / "Spooky_Shape_Grid"
    samples_dir = type_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    with open(type_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    for i in range(SAMPLES_PER_TYPE):
        mask, width, height = reconstruct_spooky_shape_grid_mask(i, ground_truth)
        output_path = samples_dir / f"spooky_shape_grid_{i:04d}_viz.png"
        create_visualization(mask, width, height, output_path)
        print(f"  Generated: {output_path.name}")


def generate_spooky_size_samples():
    """Generate samples for Spooky_Size (all shapes)."""
    type_dir = BASE_DIR / "Spooky_Size"
    samples_dir = type_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    with open(type_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    for i in range(SAMPLES_PER_TYPE):
        mask, width, height = reconstruct_spooky_size_mask(i, ground_truth)
        output_path = samples_dir / f"spooky_size_{i:04d}_viz.png"
        create_visualization(mask, width, height, output_path)
        print(f"  Generated: {output_path.name}")


def generate_spooky_jigsaw_samples():
    """Generate samples for Spooky_Jigsaw."""
    type_dir = BASE_DIR / "Spooky_Jigsaw"
    samples_dir = type_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    for i in range(SAMPLES_PER_TYPE):
        mask, width, height = reconstruct_spooky_jigsaw_mask(i)
        output_path = samples_dir / f"spooky_jigsaw_{i:04d}_viz.png"
        create_visualization(mask, width, height, output_path)
        print(f"  Generated: {output_path.name}")


def main():
    """Generate all spooky captcha sample visualizations."""
    print("=" * 60)
    print("Spooky CAPTCHA Sample Visualization Generator")
    print("=" * 60)

    print("\n[1/6] Generating Spooky_Text samples...")
    generate_spooky_text_samples()

    print("\n[2/6] Generating Spooky_Circle samples...")
    generate_spooky_circle_samples()

    print("\n[3/6] Generating Spooky_Circle_Grid samples...")
    generate_spooky_circle_grid_samples()

    print("\n[4/6] Generating Spooky_Shape_Grid samples...")
    generate_spooky_shape_grid_samples()

    print("\n[5/6] Generating Spooky_Size samples...")
    generate_spooky_size_samples()

    print("\n[6/6] Generating Spooky_Jigsaw samples...")
    generate_spooky_jigsaw_samples()

    total = 6 * SAMPLES_PER_TYPE
    print("\n" + "=" * 60)
    print(f"Complete! Generated {total} visualizations.")
    print("=" * 60)
    print("\nOutput locations:")
    for puzzle_type in ["Spooky_Text", "Spooky_Circle", "Spooky_Circle_Grid",
                        "Spooky_Shape_Grid", "Spooky_Size", "Spooky_Jigsaw"]:
        print(f"  {BASE_DIR / puzzle_type / 'samples'}/")


if __name__ == "__main__":
    main()
