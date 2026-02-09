"""
Spooky Circle Grid Direction - Library Generator

Creates a library of cells with circles moving in different directions:
- Rotation: clockwise, counterclockwise
- Translation: up, down, left, right
- Empty cells
"""

import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
from scipy import ndimage


def rotate_noise(noise_field, angle_degrees):
    """Rotate a noise field."""
    return ndimage.rotate(noise_field, angle_degrees, reshape=False, order=1, mode='wrap')


def scroll_noise(noise_field, offset_y=0, offset_x=0):
    """Scroll a noise field."""
    result = noise_field
    if offset_y != 0:
        result = np.roll(result, offset_y, axis=0)
    if offset_x != 0:
        result = np.roll(result, offset_x, axis=1)
    return result


def generate_mid_frequency_noise(height, width, sigma=3.0):
    """Generate mid-spatial frequency noise."""
    noise = np.random.randn(height, width)
    filtered_noise = ndimage.gaussian_filter(noise, sigma=sigma)
    filtered_noise = (filtered_noise - filtered_noise.min()) / (filtered_noise.max() - filtered_noise.min())
    return filtered_noise


def generate_directional_circle_cell(direction, noise_seed, output_path, cell_size=120, num_frames=30, fps=15):
    """Generate a cell with a circle moving in a specific direction."""
    np.random.seed(noise_seed)

    width = height = cell_size
    center_y = center_x = cell_size // 2
    radius = int(cell_size * 0.35)

    rotation_speed = 4
    scroll_speed = 2
    base_luminance = 128.0
    noise_amplitude = 70.0

    # Create circle mask
    y_coords, x_coords = np.ogrid[:cell_size, :cell_size]
    distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    mask = np.clip((radius - distance) / 10.0, 0, 1)

    # Generate noise
    region_size = int(radius * 3)
    base_noise = generate_mid_frequency_noise(region_size, region_size, sigma=2.5)
    base_noise = (base_noise - 0.5) * 2.0

    bg_noise = generate_mid_frequency_noise(height + 100, width + 100, sigma=3.0)
    bg_noise = (bg_noise - 0.5) * 2.0

    frames = []
    for frame_idx in range(num_frames):
        # Background (slight upward movement)
        bg_offset = -frame_idx * 1
        bg_frame = scroll_noise(bg_noise, offset_y=bg_offset)[50:50+height, 50:50+width]

        img_array = base_luminance + noise_amplitude * bg_frame

        # Apply movement to circle noise based on direction
        if direction == 'clockwise':
            angle = -rotation_speed * frame_idx
            moved_noise = rotate_noise(base_noise, angle)
        elif direction == 'counterclockwise':
            angle = rotation_speed * frame_idx
            moved_noise = rotate_noise(base_noise, angle)
        elif direction == 'up':
            offset = -frame_idx * scroll_speed
            moved_noise = scroll_noise(base_noise, offset_y=offset)
        elif direction == 'down':
            offset = frame_idx * scroll_speed
            moved_noise = scroll_noise(base_noise, offset_y=offset)
        elif direction == 'left':
            offset = -frame_idx * scroll_speed
            moved_noise = scroll_noise(base_noise, offset_x=offset)
        elif direction == 'right':
            offset = frame_idx * scroll_speed
            moved_noise = scroll_noise(base_noise, offset_x=offset)
        else:  # empty
            moved_noise = base_noise

        # Place noise in circle region
        circle_noise_full = np.zeros((height, width))
        half_region = region_size // 2
        y_start = max(0, center_y - half_region)
        y_end = min(height, center_y + half_region)
        x_start = max(0, center_x - half_region)
        x_end = min(width, center_x + half_region)

        noise_y_start = half_region - (center_y - y_start)
        noise_y_end = noise_y_start + (y_end - y_start)
        noise_x_start = half_region - (center_x - x_start)
        noise_x_end = noise_x_start + (x_end - x_start)

        circle_noise_full[y_start:y_end, x_start:x_end] = moved_noise[noise_y_start:noise_y_end, noise_x_start:noise_x_end]

        circle_signal = base_luminance + noise_amplitude * circle_noise_full
        img_array = img_array * (1 - mask) + circle_signal * mask

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img_rgb = np.stack([img_array, img_array, img_array], axis=-1)

        frame = Image.fromarray(img_rgb)
        frames.append(frame)

    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0)
    return os.path.basename(output_path)


def generate_empty_cell(noise_seed, output_path, cell_size=120, num_frames=30, fps=15):
    """Generate an empty cell (just background noise)."""
    np.random.seed(noise_seed)

    width = height = cell_size
    base_luminance = 128.0
    noise_amplitude = 70.0

    bg_noise = generate_mid_frequency_noise(height + 100, width + 100, sigma=3.0)
    bg_noise = (bg_noise - 0.5) * 2.0

    frames = []
    for frame_idx in range(num_frames):
        bg_offset = -frame_idx * 1
        bg_frame = scroll_noise(bg_noise, offset_y=bg_offset)[50:50+height, 50:50+width]
        img_array = base_luminance + noise_amplitude * bg_frame
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img_rgb = np.stack([img_array, img_array, img_array], axis=-1)
        frame = Image.fromarray(img_rgb)
        frames.append(frame)

    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0)
    return os.path.basename(output_path)


def generate_library(output_dir, num_variations=5):
    """Generate library of directional circle cells."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    directions = ['clockwise', 'counterclockwise', 'up', 'down', 'left', 'right']
    library = {}

    # Generate directional circles
    for direction in directions:
        for var_id in range(num_variations):
            filename = f"cell_{direction}_{var_id}.gif"
            output_path = output_dir / filename
            seed = hash(f"{direction}_{var_id}") % 2**32

            print(f"Generating {filename}...")
            generate_directional_circle_cell(direction, seed, str(output_path))
            library[(direction, var_id)] = filename

    # Generate empty cells
    for var_id in range(num_variations):
        filename = f"cell_empty_{var_id}.gif"
        output_path = output_dir / filename
        seed = hash(f"empty_{var_id}") % 2**32

        print(f"Generating {filename}...")
        generate_empty_cell(seed, str(output_path))
        library[('empty', var_id)] = filename

    library_index = {
        "directions": directions,
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
    output_dir = str(Path(__file__).parent.parent / "captcha_data" / "Spooky_Circle_Grid_Direction")
    generate_library(output_dir, num_variations=5)

    print("\n" + "="*70)
    print("🎯 Spooky Circle Grid Direction Cell Library Generated!")
    print("="*70)
    print("\n📚 Library: 7 types × 5 variations = 35 cells")
