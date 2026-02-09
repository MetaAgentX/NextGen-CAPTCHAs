#!/usr/bin/env python3
"""
Spooky CAPTCHA Animated Visualization GIF Generator

Generates animated GIF visualizations with red edges on scrolling noise
for spooky captcha puzzles. These are used in debug mode on the web terminal.

For grid types: generates visualization GIFs for the cell library
For single-GIF types: generates per-puzzle visualizations (3 examples each)
For jigsaw: generates visualization for pieces from 3 puzzles
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.ndimage import binary_erosion
import json
import os
import math
from pathlib import Path


# === Configuration ===
BASE_DIR = Path(__file__).parent.parent / "captcha_data"
EXAMPLES_PER_TYPE = 3  # Only generate 3 examples per single-GIF type

# Animation parameters
NUM_FRAMES = 30
FPS = 12
CELL_SIZE = 120
SHAPE_RADIUS = int(CELL_SIZE * 0.35)  # ~42 pixels


# === Core Utility Functions ===

def generate_mid_frequency_noise(height, width, sigma=3.0):
    """Generate mid-spatial frequency noise."""
    noise = np.random.randn(height, width)
    filtered_noise = ndimage.gaussian_filter(noise, sigma=sigma)
    filtered_noise = (filtered_noise - filtered_noise.min()) / (filtered_noise.max() - filtered_noise.min())
    return filtered_noise


def compute_edges(mask, threshold=0.5, edge_width=3):
    """Compute edges using binary erosion."""
    binary = (mask > threshold).astype(np.uint8)
    eroded = binary_erosion(binary, iterations=edge_width)
    edges = binary - eroded
    return edges.astype(np.float32)


def create_animated_visualization(mask, width, height, output_path, num_frames=NUM_FRAMES, fps=FPS, seed=42):
    """
    Create animated GIF with scrolling noise + red edges.

    Args:
        mask: Binary mask showing shape boundaries
        width, height: Dimensions
        output_path: Where to save the GIF
        num_frames: Number of animation frames
        fps: Frames per second
        seed: Random seed for reproducible noise
    """
    np.random.seed(seed)

    frames = []
    edges = compute_edges(mask, threshold=0.1)

    # Generate oversized noise for scrolling
    scroll_total = num_frames + 10
    bg_noise = generate_mid_frequency_noise(height + scroll_total, width, sigma=3.0)

    base_luminance = 128.0
    noise_amplitude = 70.0

    for frame_idx in range(num_frames):
        # Slice background (scrolling vertically)
        bg_frame = bg_noise[frame_idx:frame_idx + height, :width]

        # Convert to grayscale values
        background = base_luminance + noise_amplitude * (bg_frame - 0.5) * 2.0
        background = np.clip(background, 0, 255).astype(np.uint8)

        # Create RGB image
        img_array = np.stack([background, background, background], axis=-1)

        # Apply red edges
        edge_mask = edges > 0
        img_array[edge_mask, 0] = 255  # Red
        img_array[edge_mask, 1] = 0    # Green
        img_array[edge_mask, 2] = 0    # Blue

        frames.append(Image.fromarray(img_array))

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )


# === Shape Mask Functions ===

def create_circle_mask(cx, cy, radius, width, height):
    """Create a binary circle mask."""
    y_coords, x_coords = np.ogrid[:height, :width]
    distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    mask = (distance <= radius).astype(float)
    return mask


def create_shape_mask(shape_type, cx, cy, radius, width, height, rotation_angle=0):
    """Create a binary mask for a shape (circle, square, triangle)."""
    if shape_type == 'circle':
        return create_circle_mask(cx, cy, radius, width, height)

    elif shape_type == 'square':
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        # Rotate square points
        half_diagonal = radius
        points = []
        for i in range(4):
            base_angle = rotation_angle + 45 + i * 90
            angle_rad = np.radians(base_angle)
            x = cx + half_diagonal * np.cos(angle_rad)
            y = cy + half_diagonal * np.sin(angle_rad)
            points.append((x, y))
        draw.polygon(points, fill=255)
        mask = np.array(img).astype(float) / 255.0
        return mask

    elif shape_type == 'triangle':
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        points = []
        for i in range(3):
            base_angle = rotation_angle - 90 + i * 120
            angle_rad = np.radians(base_angle)
            x = cx + radius * np.cos(angle_rad)
            y = cy + radius * np.sin(angle_rad)
            points.append((x, y))
        draw.polygon(points, fill=255)
        mask = np.array(img).astype(float) / 255.0
        return mask

    elif shape_type == 'pentagon':
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        points = []
        for i in range(5):
            angle = math.pi * 2 * i / 5 - math.pi / 2 + np.radians(rotation_angle)
            px = cx + radius * math.cos(angle)
            py = cy + radius * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=255)
        mask = np.array(img).astype(float) / 255.0
        return mask

    elif shape_type == 'star':
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        points = []
        for i in range(10):
            angle = math.pi * 2 * i / 10 - math.pi / 2 + np.radians(rotation_angle)
            r = radius if i % 2 == 0 else radius * 0.4
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=255)
        mask = np.array(img).astype(float) / 255.0
        return mask

    else:
        return np.zeros((height, width), dtype=float)


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
    return mask_array


# === Cell Library Visualization Generators ===

def generate_circle_grid_cell_viz():
    """Generate visualization GIFs for Spooky_Circle_Grid cell library."""
    print("  Generating Circle_Grid cell visualizations...")

    type_dir = BASE_DIR / "Spooky_Circle_Grid"
    samples_dir = type_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    cell_size = CELL_SIZE
    center = cell_size // 2
    radius = SHAPE_RADIUS

    # Circle cells (10 variations)
    for i in range(10):
        mask = create_circle_mask(center, center, radius, cell_size, cell_size)
        output_path = samples_dir / f"cell_circle_{i}_viz.gif"
        create_animated_visualization(mask, cell_size, cell_size, output_path, seed=i*100)
        print(f"    Created: {output_path.name}")

    # Empty cells (10 variations) - just noise, no edges
    for i in range(10):
        mask = np.zeros((cell_size, cell_size), dtype=float)
        output_path = samples_dir / f"cell_empty_{i}_viz.gif"
        create_animated_visualization(mask, cell_size, cell_size, output_path, seed=i*100+1000)
        print(f"    Created: {output_path.name}")


def generate_shape_grid_cell_viz():
    """Generate visualization GIFs for Spooky_Shape_Grid cell library."""
    print("  Generating Shape_Grid cell visualizations...")

    type_dir = BASE_DIR / "Spooky_Shape_Grid"
    samples_dir = type_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    cell_size = CELL_SIZE
    center = cell_size // 2
    radius = SHAPE_RADIUS

    # Library structure: shape_direction_variation
    shapes = ['circle', 'square', 'triangle']
    directions = ['clockwise', 'counterclockwise']
    num_variations = 5

    cell_idx = 0
    for shape in shapes:
        for direction in directions:
            for variation in range(num_variations):
                mask = create_shape_mask(shape, center, center, radius, cell_size, cell_size)
                output_path = samples_dir / f"cell_{cell_idx:02d}_viz.gif"
                create_animated_visualization(mask, cell_size, cell_size, output_path, seed=cell_idx*100)
                print(f"    Created: {output_path.name} ({shape}_{direction}_{variation})")
                cell_idx += 1

    # Empty cell
    mask = np.zeros((cell_size, cell_size), dtype=float)
    output_path = samples_dir / f"cell_empty_viz.gif"
    create_animated_visualization(mask, cell_size, cell_size, output_path, seed=9999)
    print(f"    Created: {output_path.name}")


# === Single-GIF Type Visualizations ===

def generate_spooky_text_viz():
    """Generate visualization GIFs for Spooky_Text (3 examples)."""
    print("  Generating Spooky_Text visualizations...")

    type_dir = BASE_DIR / "Spooky_Text"
    samples_dir = type_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    with open(type_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    width, height = 600, 250
    font_size = min(160, int(height * 0.8))

    for i in range(EXAMPLES_PER_TYPE):
        filename = f"spooky_text_{i:04d}.gif"
        text = ground_truth[filename]["answer"].upper()

        mask = create_text_mask(text, width, height, font_size)
        output_path = samples_dir / f"spooky_text_{i:04d}_viz.gif"
        create_animated_visualization(mask, width, height, output_path, seed=i*12345)
        print(f"    Created: {output_path.name} (text: {text})")


def generate_spooky_circle_viz():
    """Generate visualization GIFs for Spooky_Circle (3 examples)."""
    print("  Generating Spooky_Circle visualizations...")

    type_dir = BASE_DIR / "Spooky_Circle"
    samples_dir = type_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    with open(type_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    width, height = 400, 400

    for puzzle_idx in range(EXAMPLES_PER_TYPE):
        # Reconstruct circle positions using same seed as original generator
        output_path_orig = f"spooky_{puzzle_idx:04d}.gif"
        seed = hash(output_path_orig) % 2**32
        np.random.seed(seed)

        key = f"spooky_circle_{puzzle_idx:04d}"
        num_circles = ground_truth[key]["answer"]

        # Replicate circle placement logic
        def circles_overlap(c1_center, c1_radius, c2_center, c2_radius, min_spacing=20):
            cx1, cy1 = c1_center
            cx2, cy2 = c2_center
            distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
            return distance < (c1_radius + c2_radius + min_spacing)

        circles = []
        max_attempts = 100

        for _ in range(num_circles):
            for attempt in range(max_attempts):
                cx = np.random.randint(50, width - 50)
                cy = np.random.randint(50, height - 50)
                radius = np.random.randint(30, 50)

                overlaps = False
                for existing in circles:
                    if circles_overlap((cx, cy), radius, existing['center'], existing['radius']):
                        overlaps = True
                        break

                if not overlaps:
                    circles.append({'center': (cx, cy), 'radius': radius})
                    break

        # Create combined mask
        combined_mask = np.zeros((height, width), dtype=np.float32)
        for circle in circles:
            cx, cy = circle['center']
            radius = circle['radius']
            mask = create_circle_mask(cx, cy, radius, width, height)
            combined_mask = np.maximum(combined_mask, mask)

        output_path = samples_dir / f"spooky_{puzzle_idx:04d}_viz.gif"
        create_animated_visualization(combined_mask, width, height, output_path, seed=puzzle_idx*54321)
        print(f"    Created: {output_path.name} ({num_circles} circles)")


def generate_spooky_size_viz():
    """Generate visualization GIFs for Spooky_Size (3 examples)."""
    print("  Generating Spooky_Size visualizations...")

    type_dir = BASE_DIR / "Spooky_Size"
    samples_dir = type_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    with open(type_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    width, height = 600, 400

    for puzzle_idx in range(EXAMPLES_PER_TYPE):
        key = f"spooky_size_{puzzle_idx:04d}"
        puzzle_data = ground_truth[key]

        combined_mask = np.zeros((height, width), dtype=np.float32)

        # Use all_shapes if available
        if "all_shapes" in puzzle_data:
            for shape_data in puzzle_data["all_shapes"]:
                shape_type = shape_data["shape"]
                center_x = shape_data["x"]
                center_y = shape_data["y"]
                radius = shape_data["radius"]

                mask = create_shape_mask(shape_type, center_x, center_y, radius, width, height)
                combined_mask = np.maximum(combined_mask, mask)
        else:
            # Fallback to target only
            target = puzzle_data["target_position"]
            target_shape = puzzle_data["target_shape"]
            diameter = target["diameter"]
            center_x = target["x"] + diameter // 2
            center_y = target["y"] + diameter // 2
            radius = diameter // 2
            combined_mask = create_shape_mask(target_shape, center_x, center_y, radius, width, height)

        output_path = samples_dir / f"spooky_size_{puzzle_idx:04d}_viz.gif"
        create_animated_visualization(combined_mask, width, height, output_path, seed=puzzle_idx*67890)

        num_shapes = len(puzzle_data.get("all_shapes", [])) or 1
        print(f"    Created: {output_path.name} ({num_shapes} shapes)")


# === Jigsaw Piece Visualizations ===

def create_jigsaw_shape_pattern(width, height, seed):
    """Create a pattern of shapes for jigsaw content."""
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

    # Add extra shapes
    num_extra = np.random.randint(3, 7)
    for _ in range(num_extra):
        shape_type = np.random.choice(shape_types)
        cx = np.random.randint(40, width - 40)
        cy = np.random.randint(40, height - 40)
        size = np.random.randint(35, 65)

        shape_mask = create_shape_mask(shape_type, cx, cy, size, width, height)
        mask = np.maximum(mask, shape_mask)

    return mask


def generate_spooky_jigsaw_viz():
    """Generate visualization GIFs for Spooky_Jigsaw pieces and reference images (3 puzzles)."""
    print("  Generating Spooky_Jigsaw visualizations...")

    type_dir = BASE_DIR / "Spooky_Jigsaw"
    samples_dir = type_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    with open(type_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    canvas_size = 450
    piece_size = 150
    grid_size = 3

    for puzzle_idx in range(EXAMPLES_PER_TYPE):
        key = f"spooky_jigsaw_{puzzle_idx:04d}"
        puzzle_data = ground_truth[key]
        pieces = puzzle_data["pieces"]

        # Reconstruct full pattern mask using deterministic seed
        seed = puzzle_idx * 12345
        full_mask = create_jigsaw_shape_pattern(canvas_size, canvas_size, seed)

        # Generate visualization GIF for the full reference image
        ref_viz_filename = f"spooky_jigsaw_{puzzle_idx:04d}_viz.gif"
        ref_output_path = samples_dir / ref_viz_filename
        create_animated_visualization(
            full_mask, canvas_size, canvas_size, ref_output_path,
            seed=puzzle_idx * 99999
        )
        print(f"    Created: {ref_viz_filename} (reference)")

        # Generate visualization for each piece
        for piece_idx, piece_filename in enumerate(pieces):
            # Calculate piece position in grid
            row = piece_idx // grid_size
            col = piece_idx % grid_size

            y_start = row * piece_size
            y_end = (row + 1) * piece_size
            x_start = col * piece_size
            x_end = (col + 1) * piece_size

            # Extract piece mask
            piece_mask = full_mask[y_start:y_end, x_start:x_end]

            # Generate visualization GIF for this piece
            viz_filename = piece_filename.replace('.gif', '_viz.gif')
            output_path = samples_dir / viz_filename
            create_animated_visualization(
                piece_mask, piece_size, piece_size, output_path,
                seed=puzzle_idx * 1000 + piece_idx
            )
            print(f"    Created: {viz_filename}")


# === Main ===

def main():
    """Generate all spooky CAPTCHA visualization GIFs."""
    print("=" * 60)
    print("Spooky CAPTCHA Animated Visualization GIF Generator")
    print("=" * 60)

    print("\n[1/6] Generating Spooky_Circle_Grid cell library...")
    generate_circle_grid_cell_viz()

    print("\n[2/6] Generating Spooky_Shape_Grid cell library...")
    generate_shape_grid_cell_viz()

    print("\n[3/6] Generating Spooky_Text visualizations...")
    generate_spooky_text_viz()

    print("\n[4/6] Generating Spooky_Circle visualizations...")
    generate_spooky_circle_viz()

    print("\n[5/6] Generating Spooky_Size visualizations...")
    generate_spooky_size_viz()

    print("\n[6/6] Generating Spooky_Jigsaw visualizations...")
    generate_spooky_jigsaw_viz()

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print("\nGenerated visualization GIFs:")
    print("  - Circle_Grid cell library: 20 GIFs")
    print("  - Shape_Grid cell library: 31 GIFs")
    print("  - Spooky_Text: 3 GIFs")
    print("  - Spooky_Circle: 3 GIFs")
    print("  - Spooky_Size: 3 GIFs")
    print(f"  - Spooky_Jigsaw: {EXAMPLES_PER_TYPE * 9} piece GIFs")
    print("\nUsage: Add ?show_viz=true to URL to see visualizations in debug mode")


if __name__ == "__main__":
    main()
