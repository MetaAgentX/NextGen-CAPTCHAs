"""
Spooky Circle Grid - Library Generator

Creates a library of cells:
- Circle cells (visible circles with motion contrast)
- Empty cells (just noise, no circle)
- Multiple noise variations for each type
"""

import numpy as np
from PIL import Image, ImageDraw
import json
import os
from pathlib import Path
from scipy import ndimage


def generate_mid_frequency_noise(height, width, sigma=3.0):
    """Generate mid-spatial frequency noise."""
    noise = np.random.randn(height, width)
    filtered_noise = ndimage.gaussian_filter(noise, sigma=sigma)
    filtered_noise = (filtered_noise - filtered_noise.min()) / (filtered_noise.max() - filtered_noise.min())
    return filtered_noise


def scroll_noise(noise_field, offset_y=0, offset_x=0):
    """Scroll a noise field."""
    result = noise_field
    if offset_y != 0:
        result = np.roll(result, offset_y, axis=0)
    if offset_x != 0:
        result = np.roll(result, offset_x, axis=1)
    return result


def generate_circle_cell(has_circle, noise_seed, output_path, cell_size=120, num_frames=30, fps=15):
    """Generate a single cell GIF (with or without circle)."""
    np.random.seed(noise_seed)

    width = height = cell_size
    center_y = center_x = cell_size // 2
    radius = int(cell_size * 0.35)

    scroll_speed = 2
    base_luminance = 128.0
    noise_amplitude = 70.0

    # Generate noise fields
    pad = scroll_speed * num_frames
    large_size = cell_size + 2 * pad

    bg_noise_field = generate_mid_frequency_noise(large_size, large_size, sigma=3.0)
    bg_noise_field = (bg_noise_field - 0.5) * 2.0

    if has_circle:
        circle_noise_field = generate_mid_frequency_noise(large_size, large_size, sigma=3.0)
        circle_noise_field = (circle_noise_field - 0.5) * 2.0

        # Create circle mask
        y_coords, x_coords = np.ogrid[:cell_size, :cell_size]
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        mask = np.clip((radius - distance) / 10.0, 0, 1)

    frames = []
    for frame_idx in range(num_frames):
        # Background scrolls up
        bg_offset = -frame_idx * scroll_speed
        bg_scrolled = scroll_noise(bg_noise_field, offset_y=bg_offset)
        bg_frame = bg_scrolled[pad:pad+height, pad:pad+width]

        img_array = base_luminance + noise_amplitude * bg_frame

        if has_circle:
            # Circle scrolls down (opposite direction)
            circle_offset = frame_idx * scroll_speed
            circle_scrolled = scroll_noise(circle_noise_field, offset_y=circle_offset)
            circle_frame = circle_scrolled[pad:pad+height, pad:pad+width]

            circle_signal = base_luminance + noise_amplitude * circle_frame
            img_array = img_array * (1 - mask) + circle_signal * mask

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img_rgb = np.stack([img_array, img_array, img_array], axis=-1)

        frame = Image.fromarray(img_rgb)
        frames.append(frame)

    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0)
    return os.path.basename(output_path)


def generate_library(output_dir, num_variations=10):
    """Generate library of circle/empty cells."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    library = {}

    # Generate circle cells
    for var_id in range(num_variations):
        filename = f"cell_circle_{var_id}.gif"
        output_path = output_dir / filename
        seed = hash(f"circle_{var_id}") % 2**32

        print(f"Generating {filename}...")
        generate_circle_cell(True, seed, str(output_path))
        library[('circle', var_id)] = filename

    # Generate empty cells
    for var_id in range(num_variations):
        filename = f"cell_empty_{var_id}.gif"
        output_path = output_dir / filename
        seed = hash(f"empty_{var_id}") % 2**32

        print(f"Generating {filename}...")
        generate_circle_cell(False, seed, str(output_path))
        library[('empty', var_id)] = filename

    # Save library index
    library_index = {
        "types": ["circle", "empty"],
        "num_variations": num_variations,
        "total_cells": len(library),
        "library": {f"{k[0]}_{k[1]}": v for k, v in library.items()}
    }

    library_path = output_dir / "library_index.json"
    with open(library_path, 'w') as f:
        json.dump(library_index, f, indent=2)

    print(f"\nLibrary generated: {len(library)} cell GIFs")
    print(f"Library index: {library_path}")

    return library


if __name__ == "__main__":
    output_dir = str(Path(__file__).parent.parent / "captcha_data" / "Spooky_Circle_Grid")
    generate_library(output_dir, num_variations=10)

    print("\n" + "="*70)
    print("🎯 Spooky Circle Grid Cell Library Generated!")
    print("="*70)
    print("\n📚 Library: 2 types × 10 variations = 20 cells")
