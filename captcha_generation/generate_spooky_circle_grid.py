"""
Spooky Circle Grid CAPTCHA Generator

Generates animated GIFs with circles in a GRID layout.
Circles are only visible through OPPOSITE MOTION (temporal coherence).
Individual frames look like uniform noise with no spatial features.

Key Features:
- Grid layout (e.g., 3×3 grid of cells)
- Random subset of cells contain motion-contrast circles
- Each cell: either has a circle (opposite motion) or pure noise
- Equal-variance masking: all cells look identical in single frames
- Humans count how many cells contain circles after ~1-2 seconds

Motion Contrast Technique:
- Background noise: Scrolls in one direction (e.g., upward)
- Circle regions: Scroll in OPPOSITE direction (e.g., downward)
- Per-frame: uniform noise everywhere
- Over time: circles emerge through motion detection
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import json
import os
from pathlib import Path


def scroll_noise(noise_field, offset, direction='vertical'):
    """
    Scroll a noise field by a given offset.

    Args:
        noise_field: 2D numpy array
        offset: Number of pixels to scroll
        direction: 'vertical' (up/down) or 'horizontal' (left/right)

    Returns:
        Scrolled noise field
    """
    if direction == 'vertical':
        return np.roll(noise_field, offset, axis=0)
    else:  # horizontal
        return np.roll(noise_field, offset, axis=1)


def generate_mid_frequency_noise_pil(height, width, sigma=3.0):
    """
    Generate mid-spatial frequency noise using PIL (no scipy needed).

    Args:
        height, width: Dimensions
        sigma: Blur amount for mid-frequency filtering

    Returns:
        Grayscale noise array with values 0-1
    """
    # Start with white noise
    noise = np.random.randn(height, width)
    # Normalize to 0-255
    noise_norm = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype(np.uint8)

    # Apply Gaussian blur using PIL
    pil_img = Image.fromarray(noise_norm)
    for _ in range(int(sigma)):
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1.5))

    filtered_noise = np.array(pil_img).astype(np.float32)
    # Normalize to 0-1 range
    filtered_noise = (filtered_noise - filtered_noise.min()) / (filtered_noise.max() - filtered_noise.min())
    return filtered_noise


def generate_spooky_circle_grid_gif(
    grid_size=(3, 3),
    num_circles=None,
    output_path="spooky_circle_grid_0.gif",
    cell_size=120,
    num_frames=30,
    fps=15,
    seed=None
):
    """
    Generate a grid-based CAPTCHA where circles appear through opposite motion.

    Args:
        grid_size: Tuple (rows, cols) for grid layout
        num_circles: Number of cells with circles (None = random 1 to half of cells)
        output_path: Where to save the GIF
        cell_size: Size of each grid cell in pixels
        num_frames: Number of frames in animation
        fps: Frames per second
        seed: Random seed for deterministic generation

    Returns:
        Dictionary with metadata (grid_size, circle_cells, answer)
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(hash(output_path) % 2**32)

    rows, cols = grid_size
    total_cells = rows * cols

    # Determine how many circles to place
    if num_circles is None:
        # Random between 1 and half of total cells
        num_circles = np.random.randint(1, max(2, total_cells // 2 + 1))

    # Select random cells to contain circles
    all_cell_indices = list(range(total_cells))
    circle_cell_indices = sorted(np.random.choice(all_cell_indices, size=num_circles, replace=False).tolist())

    # Convert to (row, col) coordinates
    circle_cells = [(idx // cols, idx % cols) for idx in circle_cell_indices]

    # Image dimensions
    width = cols * cell_size
    height = rows * cell_size

    # Motion parameters
    scroll_speed = 2  # Pixels per frame
    direction = 'vertical'

    # Visual parameters
    base_luminance = 128.0
    noise_amplitude = 70.0

    # Create circle masks for cells that have circles
    circle_masks = np.zeros((rows, cols, height, width), dtype=np.float32)

    for row, col in circle_cells:
        # Calculate cell bounds
        y_start = row * cell_size
        y_end = (row + 1) * cell_size
        x_start = col * cell_size
        x_end = (col + 1) * cell_size

        # Create circle centered in this cell
        cell_center_y = (y_start + y_end) // 2
        cell_center_x = (x_start + x_end) // 2

        # Circle radius (fits within cell with margin)
        radius = int(cell_size * 0.35)  # 35% of cell size

        # Create soft-edged circle mask
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((x_coords - cell_center_x)**2 + (y_coords - cell_center_y)**2)

        # Soft-edged mask with gradual falloff
        mask = np.clip((radius - distance) / 10.0, 0, 1)
        circle_masks[row, col] = mask

    # Generate large noise fields for scrolling (to avoid edge artifacts)
    pad = scroll_speed * num_frames
    large_height = height + 2 * pad
    large_width = width + 2 * pad

    # Background noise field (scrolls one direction)
    bg_noise_field = generate_mid_frequency_noise_pil(large_height, large_width, sigma=3.0)
    bg_noise_field = (bg_noise_field - 0.5) * 2.0

    # Circle noise field (scrolls OPPOSITE direction)
    circle_noise_field = generate_mid_frequency_noise_pil(large_height, large_width, sigma=3.0)
    circle_noise_field = (circle_noise_field - 0.5) * 2.0

    # Generate frames with opposite motion
    frames = []
    for frame_idx in range(num_frames):
        # Calculate scroll offsets
        # Background scrolls UP (negative offset)
        bg_offset = -frame_idx * scroll_speed
        # Circles scroll DOWN (positive offset)
        circle_offset = frame_idx * scroll_speed

        # Extract current frame from scrolling background noise
        bg_scrolled = scroll_noise(bg_noise_field, bg_offset, direction)
        bg_frame = bg_scrolled[pad:pad+height, pad:pad+width]

        # Extract current frame from scrolling circle noise
        circle_scrolled = scroll_noise(circle_noise_field, circle_offset, direction)
        circle_frame = circle_scrolled[pad:pad+height, pad:pad+width]

        # Start with background noise
        img_array = base_luminance + noise_amplitude * bg_frame

        # Composite circles using masks
        for row, col in circle_cells:
            mask = circle_masks[row, col]
            circle_signal = base_luminance + noise_amplitude * circle_frame
            img_array = img_array * (1 - mask) + circle_signal * mask

        # Clip to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Convert grayscale to RGB
        img_rgb = np.stack([img_array, img_array, img_array], axis=-1)

        # Draw visible grid lines
        frame_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(frame_img)

        # More visible grid lines (darker, thicker)
        grid_color = (80, 80, 80)  # Dark gray, fully opaque

        # Vertical lines
        for i in range(1, cols):
            x = i * cell_size
            draw.line([(x, 0), (x, height)], fill=grid_color, width=2)

        # Horizontal lines
        for i in range(1, rows):
            y = i * cell_size
            draw.line([(0, y), (width, y)], fill=grid_color, width=2)

        # Draw border around entire grid
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

    # Return metadata
    metadata = {
        "grid_size": grid_size,
        "total_cells": total_cells,
        "circle_cells": circle_cells,
        "circle_cell_indices": circle_cell_indices,
        "answer": num_circles,
        "description": f"Grid with {num_circles} cell(s) containing motion-contrast circles"
    }

    return metadata


def generate_dataset(output_dir, num_samples=20, grid_size=(3, 3)):
    """
    Generate a dataset of spooky circle grid CAPTCHAs.

    Args:
        output_dir: Directory to save the generated GIFs
        num_samples: Number of samples to generate
        grid_size: Tuple (rows, cols) for grid layout
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = {}

    for i in range(num_samples):
        # Generate filename
        filename = f"spooky_circle_grid_{i}.gif"
        output_path = output_dir / filename

        # Generate the GIF
        print(f"Generating {filename}...")
        metadata = generate_spooky_circle_grid_gif(
            grid_size=grid_size,
            num_circles=None,  # Random
            output_path=str(output_path),
            cell_size=120,
            num_frames=30,
            fps=15,
            seed=None
        )

        # Store ground truth
        ground_truth[filename] = {
            "answer": metadata["answer"],
            "prompt": f"How many cells contain circles in this {grid_size[0]}×{grid_size[1]} grid?",
            "description": metadata["description"],
            "media_path": f"captcha_data/Spooky_Circle_Grid/{filename}",
            "media_type": "gif",
            "difficulty": 4,
            "grid_size": metadata["grid_size"],
            "total_cells": metadata["total_cells"],
            "circle_cells": metadata["circle_cells"],
            "circle_cell_indices": metadata["circle_cell_indices"]
        }

        print(f"  → {metadata['answer']} circles in cells: {metadata['circle_cells']}")

    # Save ground truth
    ground_truth_path = output_dir / "ground_truth.json"
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGenerated {num_samples} samples!")
    print(f"Output directory: {output_dir}")
    print(f"Ground truth: {ground_truth_path}")


if __name__ == "__main__":
    # Generate dataset
    output_dir = str(Path(__file__).parent.parent / "captcha_data" / "Spooky_Circle_Grid")
    generate_dataset(output_dir, num_samples=20, grid_size=(3, 3))

    print("\n" + "="*70)
    print("🎯 Spooky Circle Grid CAPTCHA Dataset Generated!")
    print("Motion Contrast CAPTCHA in Grid Layout")
    print("="*70)
    print("\n🔬 Technical Implementation:")
    print("  ✓ Grid Layout: 3×3 cells (9 total cells)")
    print("  ✓ Random Subset: 1 to 4 cells contain circles")
    print("  ✓ Motion Contrast: Circles scroll opposite to background")
    print("  ✓ Background scrolls UP, circles scroll DOWN")
    print("  ✓ Subtle grid lines for visual reference")
    print("  ✓ Equal variance: All cells look identical per frame")
    print("\n📊 Per-Frame Analysis:")
    print("  • Single frame: All cells show uniform noise")
    print("  • No way to distinguish circle cells from empty cells")
    print("  • Identical noise statistics across all cells")
    print("  • Grid structure visible but content indistinguishable")
    print("\n🧠 Why Humans Can See It:")
    print("  • Watch animation for 1-2 seconds")
    print("  • Motion detection reveals opposite-moving regions")
    print("  • Circles 'pop out' in specific grid cells")
    print("  • Easy to count which cells contain circles")
    print("  • Grid structure helps organize the count")
    print("\n🤖 Why LLMs/Vision Models Fail:")
    print("  ✗ Single frame: Just grid of identical noise")
    print("  ✗ No spatial features to identify circle cells")
    print("  ✗ Temporal mean/std: FLAT across all cells")
    print("  ✗ Would need motion segmentation + optical flow analysis")
    print("  ✗ Most vision models lack motion processing")
    print("\n🏆 Grid-based Temporal Coherence CAPTCHA!")
    print("    Humans: Easy counting task after brief observation")
    print("    LLMs: Blind without sophisticated motion analysis")
    print("    Perfect for benchmarking temporal perception!")
