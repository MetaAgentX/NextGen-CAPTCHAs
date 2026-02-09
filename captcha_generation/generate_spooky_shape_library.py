"""
Spooky Shape Grid CAPTCHA - Library Generator

Creates a library of cell animations that can be reused across multiple puzzles.

Library Structure:
- 3 shapes: circle, square, triangle
- 2 rotations: clockwise, counterclockwise
- 5 noise variations per (shape, rotation) pair
- Total: 3 × 2 × 5 = 30 cell GIF files

Then puzzles reference these library cells instead of generating unique ones.
"""

import numpy as np
from PIL import Image, ImageDraw
import json
import os
import random
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import shift


def rotate_noise(noise_field, angle_degrees):
    """Rotate a noise field by a given angle."""
    return ndimage.rotate(noise_field, angle_degrees, reshape=False, order=1, mode='wrap')


def generate_mid_frequency_noise(height, width, sigma=3.0):
    """Generate mid-spatial frequency noise using scipy."""
    noise = np.random.randn(height, width)
    filtered_noise = ndimage.gaussian_filter(noise, sigma=sigma)
    filtered_noise = (filtered_noise - filtered_noise.min()) / (filtered_noise.max() - filtered_noise.min())
    return filtered_noise


def create_shape_mask_on_canvas(shape_type, center_x, center_y, radius, width, height, rotation_angle=0):
    """Create a binary mask for a ROTATING shape on a full canvas."""
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if shape_type == 'circle':
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        draw.ellipse(bbox, fill=255)
    elif shape_type == 'square':
        half_diagonal = radius
        points = []
        for i in range(4):
            base_angle = rotation_angle + 45 + i * 90
            angle_rad = np.radians(base_angle)
            x = center_x + half_diagonal * np.cos(angle_rad)
            y = center_y + half_diagonal * np.sin(angle_rad)
            points.append((x, y))
        draw.polygon(points, fill=255)
    elif shape_type == 'triangle':
        points = []
        for i in range(3):
            angle = rotation_angle + i * 120 - 90
            angle_rad = np.radians(angle)
            x = center_x + radius * np.cos(angle_rad)
            y = center_y + radius * np.sin(angle_rad)
            points.append((x, y))
        draw.polygon(points, fill=255)

    mask_array = np.array(mask_img).astype(np.float32) / 255.0
    return mask_array


def generate_cell_gif(
    shape_type,
    direction,
    noise_seed,
    output_path,
    cell_size=120,
    num_frames=24,
    fps=12
):
    """
    Generate a single cell GIF with a rotating shape.

    Args:
        shape_type: 'circle', 'square', or 'triangle'
        direction: 'clockwise' or 'counterclockwise'
        noise_seed: Seed for noise generation
        output_path: Where to save the GIF
        cell_size: Size of the cell in pixels
        num_frames: Number of frames
        fps: Frames per second

    Returns:
        Filename of the generated GIF
    """
    np.random.seed(noise_seed)
    random.seed(noise_seed)

    width = height = cell_size
    center_x = center_y = cell_size // 2
    radius = int(cell_size * 0.35)

    # Rotation parameters
    rotation_speed = 5  # Degrees per frame

    # Visual parameters
    base_luminance = 128.0
    noise_amplitude = 70.0

    # Generate base noise for this cell
    region_size = int(radius * 3)
    base_noise = generate_mid_frequency_noise(region_size, region_size, sigma=2.5)
    base_noise = (base_noise - 0.5) * 2.0

    # Place noise on canvas
    shape_noise_canvas = np.zeros((height, width))
    half_region = region_size // 2
    y_start = max(0, center_y - half_region)
    y_end = min(height, center_y + half_region)
    x_start = max(0, center_x - half_region)
    x_end = min(width, center_x + half_region)

    noise_y_start = half_region - (center_y - y_start)
    noise_y_end = noise_y_start + (y_end - y_start)
    noise_x_start = half_region - (center_x - x_start)
    noise_x_end = noise_x_start + (x_end - x_start)

    shape_noise_canvas[y_start:y_end, x_start:x_end] = base_noise[noise_y_start:noise_y_end, noise_x_start:noise_x_end]

    # Create unrotated shape mask
    shape_mask_unrotated = create_shape_mask_on_canvas(shape_type, center_x, center_y, radius, width, height, rotation_angle=0)

    # Generate background noise - large enough to avoid wrapping artifacts
    scroll_total = num_frames * 1  # 1 pixel per frame scroll
    pad = scroll_total + 10  # Extra padding for safety
    bg_noise = generate_mid_frequency_noise(height + pad, width, sigma=3.0)
    bg_noise = (bg_noise - 0.5) * 2.0

    # Generate frames
    frames = []
    for frame_idx in range(num_frames):
        # Calculate rotation angle
        if direction == 'clockwise':
            rotation_angle = -rotation_speed * frame_idx
        else:  # counterclockwise
            rotation_angle = rotation_speed * frame_idx

        # Background - slice without wrapping
        bg_start = frame_idx * 1
        bg_frame = bg_noise[bg_start:bg_start+height, 0:width]

        # Rotate noise and mask around cell center
        canvas_center_y = height // 2
        canvas_center_x = width // 2
        shift_y = canvas_center_y - center_y
        shift_x = canvas_center_x - center_x

        shifted_noise = shift(shape_noise_canvas, (shift_y, shift_x), order=1, mode='constant', cval=0)
        shifted_mask = shift(shape_mask_unrotated, (shift_y, shift_x), order=1, mode='constant', cval=0)

        rotated_shifted_noise = rotate_noise(shifted_noise, rotation_angle)
        rotated_shifted_mask = rotate_noise(shifted_mask, rotation_angle)

        rotated_noise_canvas = shift(rotated_shifted_noise, (-shift_y, -shift_x), order=1, mode='constant', cval=0)
        shape_mask_rotated = shift(rotated_shifted_mask, (-shift_y, -shift_x), order=1, mode='constant', cval=0)

        shape_mask_rotated = ndimage.gaussian_filter(shape_mask_rotated, sigma=3.0)

        # Composite
        img_array = base_luminance + noise_amplitude * bg_frame
        shape_signal = base_luminance + noise_amplitude * rotated_noise_canvas
        img_array = img_array * (1 - shape_mask_rotated) + shape_signal * shape_mask_rotated

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img_rgb = np.stack([img_array, img_array, img_array], axis=-1)

        frame = Image.fromarray(img_rgb)
        frames.append(frame)

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000/fps),
        loop=0
    )

    return os.path.basename(output_path)


def generate_empty_cell_gif(output_path, cell_size=120, num_frames=24, fps=12, seed=0):
    """
    Generate a background-only GIF (no shape) for empty grid cells.
    """
    np.random.seed(seed)
    random.seed(seed)

    width = height = cell_size
    base_luminance = 128.0
    noise_amplitude = 70.0

    # Generate background noise - large enough to avoid wrapping artifacts
    scroll_total = num_frames * 1  # 1 pixel per frame scroll
    pad = scroll_total + 10
    bg_noise = generate_mid_frequency_noise(height + pad, width, sigma=3.0)
    bg_noise = (bg_noise - 0.5) * 2.0

    frames = []
    for frame_idx in range(num_frames):
        bg_start = frame_idx * 1
        bg_frame = bg_noise[bg_start:bg_start+height, 0:width]
        img_array = base_luminance + noise_amplitude * bg_frame
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img_rgb = np.stack([img_array, img_array, img_array], axis=-1)
        frames.append(Image.fromarray(img_rgb))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000/fps),
        loop=0
    )

    return os.path.basename(output_path)


def generate_library(output_dir, num_variations=5):
    """
    Generate a library of cell GIFs.

    Args:
        output_dir: Directory to save the library
        num_variations: Number of noise variations per (shape, direction) pair

    Returns:
        Dictionary mapping (shape, direction, variation_id) to filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = ['circle', 'square', 'triangle']
    directions = ['clockwise', 'counterclockwise']

    library = {}
    cell_idx = 0
    for shape in shapes:
        for direction in directions:
            for var_id in range(num_variations):
                # Generate filename
                filename = f"cell_{cell_idx:02d}.gif"
                output_path = output_dir / filename

                # Generate unique seed
                seed = hash(f"{shape}_{direction}_{var_id}") % 2**32

                print(f"Generating {filename}...")

                # Generate the cell GIF
                generate_cell_gif(
                    shape_type=shape,
                    direction=direction,
                    noise_seed=seed,
                    output_path=str(output_path),
                    cell_size=120,
                    num_frames=36,
                    fps=12
                )

                # Store in library
                library[(shape, direction, var_id)] = filename
                cell_idx += 1

    # Generate a dedicated empty cell (background-only)
    empty_filename = "cell_empty.gif"
    empty_path = output_dir / empty_filename
    generate_empty_cell_gif(
        output_path=str(empty_path),
        cell_size=120,
        num_frames=36,
        fps=12,
        seed=12345
    )

    # Save library index
    library_index_path = output_dir / "library_index.json"
    library_serializable = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in library.items()}
    library_serializable["empty_cell"] = empty_filename

    with open(library_index_path, 'w') as f:
        json.dump({
            "shapes": shapes,
            "directions": directions,
            "num_variations": num_variations,
            "total_cells": len(library),
            "library": library_serializable
        }, f, indent=2)

    print(f"\nLibrary generated: {len(library)} cell GIFs")
    print(f"Library index: {library_index_path}")

    return library


if __name__ == "__main__":
    output_dir = str(Path(__file__).parent.parent / "captcha_data" / "Spooky_Shape_Grid")

    generate_library(output_dir, num_variations=5)

    print("\n" + "="*70)
    print("🎯 Spooky Shape Grid Cell Library Generated!")
    print("="*70)
    print("\n📚 Library Structure:")
    print("  ✓ 3 shapes: circle, square, triangle")
    print("  ✓ 2 rotations: clockwise, counterclockwise")
    print("  ✓ 5 noise variations per combination")
    print("  ✓ Total: 30 cell GIF files")
    print("\n💾 Space Efficiency:")
    print("  • Each puzzle reuses these 30 cells")
    print("  • No need to generate cells per puzzle")
    print("  • Significant storage savings!")
