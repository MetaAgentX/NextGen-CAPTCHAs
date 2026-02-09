"""
Spooky Circle Grid Direction CAPTCHA Generator

Generates animated GIFs with circles in a GRID layout with DIRECTIONAL movement.
Users must identify cells with circles moving in a SPECIFIC DIRECTION.

Movement Types:
- Rotation: Clockwise or Counterclockwise (noise pattern rotates)
- Translation: Up, Down, Left, or Right (noise pattern scrolls)

Challenge Types:
- "Which cells have CLOCKWISE circles?"
- "Which cells have COUNTERCLOCKWISE circles?"
- "Which cells have circles moving UP?"
- "Which cells have circles moving DOWN?"
- "Which cells have circles moving LEFT?"
- "Which cells have circles moving RIGHT?"

Key Features:
- 3×3 grid layout
- Random subset of cells contain circles
- Each circle has a specific movement direction
- Background has random/opposite movement
- User must identify cells matching the target direction
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import json
import os
from pathlib import Path
from scipy import ndimage


def rotate_noise(noise_field, angle_degrees):
    """
    Rotate a noise field by a given angle.

    Args:
        noise_field: 2D numpy array
        angle_degrees: Rotation angle in degrees (positive = counterclockwise)

    Returns:
        Rotated noise field
    """
    return ndimage.rotate(noise_field, angle_degrees, reshape=False, order=1, mode='wrap')


def scroll_noise(noise_field, offset_y=0, offset_x=0):
    """
    Scroll a noise field by given offsets.

    Args:
        noise_field: 2D numpy array
        offset_y: Vertical offset (positive = down)
        offset_x: Horizontal offset (positive = right)

    Returns:
        Scrolled noise field
    """
    result = noise_field
    if offset_y != 0:
        result = np.roll(result, offset_y, axis=0)
    if offset_x != 0:
        result = np.roll(result, offset_x, axis=1)
    return result


def generate_mid_frequency_noise(height, width, sigma=3.0):
    """
    Generate mid-spatial frequency noise using scipy.

    Args:
        height, width: Dimensions
        sigma: Blur amount for mid-frequency filtering

    Returns:
        Grayscale noise array with values 0-1
    """
    # Start with white noise
    noise = np.random.randn(height, width)
    # Apply Gaussian filter to get mid-frequency noise
    filtered_noise = ndimage.gaussian_filter(noise, sigma=sigma)
    # Normalize to 0-1 range
    filtered_noise = (filtered_noise - filtered_noise.min()) / (filtered_noise.max() - filtered_noise.min())
    return filtered_noise


def generate_spooky_circle_grid_direction_gif(
    grid_size=(3, 3),
    target_direction=None,
    num_target_circles=None,
    output_path="spooky_circle_grid_direction_0.gif",
    cell_size=120,
    num_frames=30,
    fps=15,
    seed=None
):
    """
    Generate a direction-based grid CAPTCHA.

    Args:
        grid_size: Tuple (rows, cols) for grid layout
        target_direction: Direction to ask about ('clockwise', 'counterclockwise', 'up', 'down', 'left', 'right')
        num_target_circles: Number of circles with target direction (None = random 1-3)
        output_path: Where to save the GIF
        cell_size: Size of each grid cell in pixels
        num_frames: Number of frames in animation
        fps: Frames per second
        seed: Random seed for deterministic generation

    Returns:
        Dictionary with metadata
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(hash(output_path) % 2**32)

    rows, cols = grid_size
    total_cells = rows * cols

    # Available directions
    all_directions = ['clockwise', 'counterclockwise', 'up', 'down', 'left', 'right']

    # Select target direction
    if target_direction is None:
        target_direction = np.random.choice(all_directions)

    # Determine number of circles with target direction
    if num_target_circles is None:
        num_target_circles = np.random.randint(1, 4)  # 1 to 3

    # Decide how many total circles to place (target + distractors)
    total_circles = np.random.randint(num_target_circles + 1, min(total_cells, num_target_circles + 5))

    # Select cells for circles
    all_cell_indices = list(range(total_cells))
    circle_cell_indices = sorted(np.random.choice(all_cell_indices, size=total_circles, replace=False).tolist())

    # Assign target direction to random subset
    target_indices = sorted(np.random.choice(circle_cell_indices, size=num_target_circles, replace=False).tolist())

    # Assign random non-target directions to remaining circles
    other_directions = [d for d in all_directions if d != target_direction]

    # Create direction mapping for each circle cell
    cell_directions = {}
    for idx in circle_cell_indices:
        if idx in target_indices:
            cell_directions[idx] = target_direction
        else:
            cell_directions[idx] = np.random.choice(other_directions)

    # Convert to (row, col) coordinates
    circle_cells = [(idx // cols, idx % cols) for idx in circle_cell_indices]
    target_cells = [(idx // cols, idx % cols) for idx in target_indices]

    # Image dimensions
    width = cols * cell_size
    height = rows * cell_size

    # Movement parameters
    rotation_speed = 4  # Degrees per frame (reduced for subtler rotation)
    scroll_speed = 2  # Pixels per frame (slightly reduced for consistency)

    # Visual parameters
    base_luminance = 128.0
    noise_amplitude = 70.0

    # Create masks and noise fields for each circle cell
    circle_data = []

    for idx in circle_cell_indices:
        row = idx // cols
        col = idx % cols
        direction = cell_directions[idx]

        # Calculate cell bounds
        y_start = row * cell_size
        y_end = (row + 1) * cell_size
        x_start = col * cell_size
        x_end = (col + 1) * cell_size

        # Create circle centered in this cell
        cell_center_y = (y_start + y_end) // 2
        cell_center_x = (x_start + x_end) // 2
        radius = int(cell_size * 0.35)

        # Create soft-edged circle mask
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((x_coords - cell_center_x)**2 + (y_coords - cell_center_y)**2)
        mask = np.clip((radius - distance) / 10.0, 0, 1)

        # Generate base noise for this circle region only
        # Make it slightly larger than the circle for rotation without edge artifacts
        circle_region_size = int(radius * 3)  # 3x radius for safe rotation
        base_noise = generate_mid_frequency_noise(circle_region_size, circle_region_size, sigma=2.5)
        base_noise = (base_noise - 0.5) * 2.0

        circle_data.append({
            'mask': mask,
            'base_noise': base_noise,
            'direction': direction,
            'row': row,
            'col': col,
            'cell_index': idx,
            'center_x': cell_center_x,
            'center_y': cell_center_y,
            'radius': radius,
            'region_size': circle_region_size
        })

    # Generate background noise (static or slightly moving)
    bg_noise = generate_mid_frequency_noise(height + 100, width + 100, sigma=3.0)
    bg_noise = (bg_noise - 0.5) * 2.0

    # Generate frames
    frames = []
    for frame_idx in range(num_frames):
        # Background (slight upward movement)
        bg_offset = -frame_idx * 1
        bg_frame = scroll_noise(bg_noise, offset_y=bg_offset)[50:50+height, 50:50+width]

        # Start with background
        img_array = base_luminance + noise_amplitude * bg_frame

        # Add circles with directional movement
        for data in circle_data:
            mask = data['mask']
            base_noise = data['base_noise']
            direction = data['direction']
            center_x = data['center_x']
            center_y = data['center_y']
            radius = data['radius']
            region_size = data['region_size']

            # Apply movement based on direction (only to the small circle region)
            if direction == 'clockwise':
                # Rotate clockwise around center (negative angle)
                angle = -rotation_speed * frame_idx
                moved_noise = rotate_noise(base_noise, angle)
            elif direction == 'counterclockwise':
                # Rotate counterclockwise around center (positive angle)
                angle = rotation_speed * frame_idx
                moved_noise = rotate_noise(base_noise, angle)
            elif direction == 'up':
                # Scroll upward within the region
                offset = -frame_idx * scroll_speed
                moved_noise = scroll_noise(base_noise, offset_y=offset)
            elif direction == 'down':
                # Scroll downward within the region
                offset = frame_idx * scroll_speed
                moved_noise = scroll_noise(base_noise, offset_y=offset)
            elif direction == 'left':
                # Scroll left within the region
                offset = -frame_idx * scroll_speed
                moved_noise = scroll_noise(base_noise, offset_x=offset)
            elif direction == 'right':
                # Scroll right within the region
                offset = frame_idx * scroll_speed
                moved_noise = scroll_noise(base_noise, offset_x=offset)

            # Create a full-size noise array for this circle
            circle_noise_full = np.zeros((height, width))

            # Place the moved circle region at the correct location
            half_region = region_size // 2
            y_start = max(0, center_y - half_region)
            y_end = min(height, center_y + half_region)
            x_start = max(0, center_x - half_region)
            x_end = min(width, center_x + half_region)

            # Calculate corresponding indices in the moved_noise
            noise_y_start = half_region - (center_y - y_start)
            noise_y_end = noise_y_start + (y_end - y_start)
            noise_x_start = half_region - (center_x - x_start)
            noise_x_end = noise_x_start + (x_end - x_start)

            # Place the noise
            circle_noise_full[y_start:y_end, x_start:x_end] = moved_noise[noise_y_start:noise_y_end, noise_x_start:noise_x_end]

            # Apply circle signal
            circle_signal = base_luminance + noise_amplitude * circle_noise_full
            img_array = img_array * (1 - mask) + circle_signal * mask

        # Clip to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Convert to RGB
        img_rgb = np.stack([img_array, img_array, img_array], axis=-1)

        # Draw grid lines
        frame_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(frame_img)
        grid_color = (80, 80, 80)

        # Vertical lines
        for i in range(1, cols):
            x = i * cell_size
            draw.line([(x, 0), (x, height)], fill=grid_color, width=2)

        # Horizontal lines
        for i in range(1, rows):
            y = i * cell_size
            draw.line([(0, y), (width, y)], fill=grid_color, width=2)

        # Border
        draw.rectangle([(0, 0), (width-1, height-1)], outline=grid_color, width=2)

        frames.append(frame_img)

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000/fps),
        loop=0
    )

    # Create metadata
    metadata = {
        "grid_size": grid_size,
        "total_cells": total_cells,
        "target_direction": target_direction,
        "total_circles": total_circles,
        "target_cells": target_cells,
        "target_cell_indices": target_indices,
        "answer": num_target_circles,
        "all_circle_cells": circle_cells,
        "all_circle_indices": circle_cell_indices,
        "cell_directions": {str(k): v for k, v in cell_directions.items()},
        "description": f"Grid with {total_circles} circles, {num_target_circles} moving {target_direction}"
    }

    return metadata


def generate_dataset(output_dir, num_samples=20, grid_size=(3, 3)):
    """
    Generate a dataset of spooky circle grid direction CAPTCHAs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = {}

    for i in range(num_samples):
        filename = f"spooky_circle_grid_direction_{i}.gif"
        output_path = output_dir / filename

        print(f"Generating {filename}...")
        metadata = generate_spooky_circle_grid_direction_gif(
            grid_size=grid_size,
            target_direction=None,  # Random
            num_target_circles=None,  # Random
            output_path=str(output_path),
            cell_size=120,
            num_frames=30,
            fps=15,
            seed=None
        )

        # Create user-friendly prompt
        direction_prompts = {
            'clockwise': 'Which cells contain circles rotating CLOCKWISE?',
            'counterclockwise': 'Which cells contain circles rotating COUNTERCLOCKWISE?',
            'up': 'Which cells contain circles moving UP?',
            'down': 'Which cells contain circles moving DOWN?',
            'left': 'Which cells contain circles moving LEFT?',
            'right': 'Which cells contain circles moving RIGHT?'
        }

        prompt = direction_prompts.get(metadata['target_direction'],
                                       f"Which cells have circles moving {metadata['target_direction']}?")

        # Store ground truth
        ground_truth[filename] = {
            "answer": metadata["answer"],
            "prompt": prompt,
            "description": metadata["description"],
            "media_path": f"captcha_data/Spooky_Circle_Grid_Direction/{filename}",
            "media_type": "gif",
            "difficulty": 5,
            "grid_size": metadata["grid_size"],
            "total_cells": metadata["total_cells"],
            "target_direction": metadata["target_direction"],
            "target_cells": metadata["target_cells"],
            "target_cell_indices": metadata["target_cell_indices"],
            "all_circle_cells": metadata["all_circle_cells"],
            "cell_directions": metadata["cell_directions"]
        }

        print(f"  → Direction: {metadata['target_direction']}, Target cells: {metadata['target_cells']}")

    # Save ground truth
    ground_truth_path = output_dir / "ground_truth.json"
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGenerated {num_samples} samples!")
    print(f"Output directory: {output_dir}")
    print(f"Ground truth: {ground_truth_path}")


if __name__ == "__main__":
    # Create output directory
    output_dir = str(Path(__file__).parent.parent / "captcha_data" / "Spooky_Circle_Grid_Direction")

    # Generate dataset
    generate_dataset(output_dir, num_samples=20, grid_size=(3, 3))

    print("\n" + "="*70)
    print("🎯 Spooky Circle Grid Direction CAPTCHA Dataset Generated!")
    print("Directional Motion Detection in Grid Layout")
    print("="*70)
    print("\n🔬 Technical Implementation:")
    print("  ✓ 6 Movement Types: clockwise, counterclockwise, up, down, left, right")
    print("  ✓ Rotation: 4°/frame for clockwise/counterclockwise (subtle)")
    print("  ✓ Translation: 2 pixels/frame for up/down/left/right")
    print("  ✓ Mixed Directions: Each circle has different movement")
    print("  ✓ Target Identification: User identifies cells with specific direction")
    print("\n📊 Challenge Structure:")
    print("  • 3×3 grid with 2-8 circles total")
    print("  • 1-3 circles move in the TARGET direction")
    print("  • Others move in DISTRACTOR directions")
    print("  • User must identify only target direction cells")
    print("\n🧠 Why Humans Can See It:")
    print("  • Motion perception distinguishes rotation vs translation")
    print("  • Directional selectivity in visual cortex (V5/MT)")
    print("  • Can track clockwise vs counterclockwise rotation")
    print("  • Can distinguish up/down/left/right motion")
    print("  • After 1-2 seconds: target cells become obvious")
    print("\n🤖 Why LLMs/Vision Models Fail:")
    print("  ✗ Need sophisticated motion analysis")
    print("  ✗ Must detect rotation direction (not just motion)")
    print("  ✗ Must distinguish translation direction")
    print("  ✗ Must segment multiple simultaneous motions")
    print("  ✗ Most models lack directional motion selectivity")
    print("\n🏆 Advanced Directional Motion CAPTCHA!")
    print("    Humans: Easy - natural motion perception")
    print("    LLMs: Very hard - requires advanced motion analysis")
    print("    Perfect for testing directional motion understanding!")
