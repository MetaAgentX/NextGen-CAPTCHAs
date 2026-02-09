"""
Spooky Shape Grid CAPTCHA Generator

Generates animated GIFs with a 3x3 grid where shapes (triangle, square, circle)
rotate clockwise or counterclockwise. Each shape rotates with the same radius
to ensure fair comparison.

Challenge: Users must identify all cells containing a specific shape rotating
in a specific direction (e.g., "Click all triangles rotating clockwise").

Key Features:
- Equal rotation radius for all shapes (ensures fair visual difficulty)
- Mid-frequency noise background to obscure static frames
- Shapes only visible through motion coherence
- Multiple shapes and rotation directions in one grid
"""

import numpy as np
from PIL import Image, ImageDraw
import json
import os
import random
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


def create_shape_mask_on_canvas(shape_type, center_x, center_y, radius, width, height, rotation_angle=0):
    """
    Create a binary mask for a ROTATING shape on a full canvas.

    Args:
        shape_type: 'circle', 'square', or 'triangle'
        center_x, center_y: Center position on canvas
        radius: Radius of the shape (same for all shapes - distance to furthest point)
        width, height: Canvas dimensions
        rotation_angle: Current rotation angle in degrees

    Returns:
        Binary mask array where rotated shape is rendered
    """
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if shape_type == 'circle':
        # Circle doesn't change with rotation
        bbox = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius
        ]
        draw.ellipse(bbox, fill=255)

    elif shape_type == 'square':
        # Square inscribed in circle - all 4 corners touch the circle at radius distance
        # For a square inscribed in a circle, the side length = radius * sqrt(2)
        # but we want the corners at radius distance from center
        half_diagonal = radius  # Distance from center to corner

        # Create 4 corners at 45° intervals, then rotate by rotation_angle
        points = []
        for i in range(4):
            base_angle = rotation_angle + 45 + i * 90  # Start at 45° to make it diamond-like initially
            angle_rad = np.radians(base_angle)
            x = center_x + half_diagonal * np.cos(angle_rad)
            y = center_y + half_diagonal * np.sin(angle_rad)
            points.append((x, y))
        draw.polygon(points, fill=255)

    elif shape_type == 'triangle':
        # Equilateral triangle inscribed in circle - all 3 corners at radius distance
        points = []
        for i in range(3):
            angle = rotation_angle + i * 120 - 90  # 120° apart, -90 to point upward initially
            angle_rad = np.radians(angle)
            x = center_x + radius * np.cos(angle_rad)
            y = center_y + radius * np.sin(angle_rad)
            points.append((x, y))
        draw.polygon(points, fill=255)

    # Convert to numpy array and normalize
    mask_array = np.array(mask_img).astype(np.float32) / 255.0

    return mask_array


def generate_spooky_shape_grid_gif(
    output_path="spooky_shape_grid_0.gif",
    grid_size=(3, 3),
    cell_size=120,
    num_frames=24,
    fps=12,
    seed=None,
    target_shape=None,
    target_direction=None,
    num_targets=None
):
    """
    Generate a 3x3 grid GIF with rotating shapes.

    Args:
        output_path: Where to save the GIF
        grid_size: Grid dimensions (rows, cols)
        cell_size: Size of each cell in pixels
        num_frames: Number of frames in animation
        fps: Frames per second
        seed: Random seed for deterministic generation
        target_shape: Shape to identify ('circle', 'square', 'triangle')
        target_direction: Direction to identify ('clockwise', 'counterclockwise')
        num_targets: Number of target shapes (None = random)

    Returns:
        Dictionary with puzzle metadata
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        base_seed = seed
    else:
        base_seed = hash(output_path) % 2**32
        np.random.seed(base_seed)
        random.seed(base_seed)

    rows, cols = grid_size
    total_cells = rows * cols

    # Shape choices
    all_shapes = ['circle', 'square', 'triangle']
    all_directions = ['clockwise', 'counterclockwise']

    # Choose target shape and direction
    if target_shape is None:
        target_shape = np.random.choice(all_shapes)
    if target_direction is None:
        target_direction = np.random.choice(all_directions)

    # Determine number of target shapes
    if num_targets is None:
        num_targets = np.random.randint(2, 5)  # 2 to 4 targets

    # Decide total number of shapes (targets + distractors)
    total_shapes = np.random.randint(num_targets + 2, min(total_cells, num_targets + 6))

    # Select cells for shapes
    all_cell_indices = list(range(total_cells))
    shape_cell_indices = sorted(np.random.choice(all_cell_indices, size=total_shapes, replace=False).tolist())

    # Assign target to random subset
    target_indices = sorted(np.random.choice(shape_cell_indices, size=num_targets, replace=False).tolist())

    # Create shape/direction mapping for each shape cell
    cell_config = {}
    for idx in shape_cell_indices:
        if idx in target_indices:
            cell_config[idx] = {
                'shape': target_shape,
                'direction': target_direction,
                'is_target': True
            }
        else:
            # Distractor - different shape OR different direction
            if np.random.random() < 0.5:
                # Different shape, any direction
                other_shapes = [s for s in all_shapes if s != target_shape]
                distractor_shape = np.random.choice(other_shapes)
                distractor_direction = np.random.choice(all_directions)
            else:
                # Same shape, opposite direction
                distractor_shape = target_shape
                distractor_direction = 'clockwise' if target_direction == 'counterclockwise' else 'counterclockwise'

            cell_config[idx] = {
                'shape': distractor_shape,
                'direction': distractor_direction,
                'is_target': False
            }

    # Image dimensions
    width = cols * cell_size
    height = rows * cell_size

    # Rotation parameter
    rotation_speed = 5  # Degrees per frame

    # Visual parameters
    base_luminance = 128.0
    noise_amplitude = 70.0

    # Store shape configuration for frame generation
    shape_data = []

    for idx in shape_cell_indices:
        row = idx // cols
        col = idx % cols
        config = cell_config[idx]

        # Calculate cell bounds
        y_start = row * cell_size
        y_end = (row + 1) * cell_size
        x_start = col * cell_size
        x_end = (col + 1) * cell_size

        # Center of this cell
        cell_center_y = (y_start + y_end) // 2
        cell_center_x = (x_start + x_end) // 2
        radius = int(cell_size * 0.35)  # Same radius for all shapes

        # Generate base noise for this shape (static pattern)
        region_size = int(radius * 3)
        np.random.seed(base_seed + idx * 100)
        base_noise = generate_mid_frequency_noise(region_size, region_size, sigma=2.5)
        base_noise = (base_noise - 0.5) * 2.0

        shape_data.append({
            'base_noise': base_noise,
            'direction': config['direction'],
            'shape': config['shape'],
            'is_target': config['is_target'],
            'center_x': cell_center_x,
            'center_y': cell_center_y,
            'radius': radius,
            'region_size': region_size
        })

    # Generate background noise (static or slight movement)
    bg_noise = generate_mid_frequency_noise(height + 100, width + 100, sigma=3.0)
    bg_noise = (bg_noise - 0.5) * 2.0

    # Generate frames
    frames = []
    for frame_idx in range(num_frames):
        # Background
        bg_offset = -frame_idx * 1
        bg_frame = np.roll(bg_noise, bg_offset, axis=0)[50:50+height, 50:50+width]

        # Start with background
        img_array = base_luminance + noise_amplitude * bg_frame

        # Add rotating shapes
        for data in shape_data:
            base_noise = data['base_noise']
            direction = data['direction']
            shape_type = data['shape']
            center_x = data['center_x']
            center_y = data['center_y']
            radius = data['radius']
            region_size = data['region_size']

            # Calculate rotation angle for this frame
            if direction == 'clockwise':
                rotation_angle = -rotation_speed * frame_idx  # Negative for clockwise
            else:  # counterclockwise
                rotation_angle = rotation_speed * frame_idx  # Positive for counterclockwise

            # STEP 1: Place the UNROTATED noise on a canvas centered at the shape position
            # This creates a noise field that is "painted" on the shape
            shape_noise_canvas = np.zeros((height, width))
            half_region = region_size // 2
            y_start = max(0, center_y - half_region)
            y_end = min(height, center_y + half_region)
            x_start = max(0, center_x - half_region)
            x_end = min(width, center_x + half_region)

            # Corresponding indices in base_noise
            noise_y_start = half_region - (center_y - y_start)
            noise_y_end = noise_y_start + (y_end - y_start)
            noise_x_start = half_region - (center_x - x_start)
            noise_x_end = noise_x_start + (x_end - x_start)

            # Place the UNROTATED noise
            shape_noise_canvas[y_start:y_end, x_start:x_end] = base_noise[noise_y_start:noise_y_end, noise_x_start:noise_x_end]

            # STEP 2: Create UNROTATED shape mask on canvas
            shape_mask_unrotated = create_shape_mask_on_canvas(
                shape_type,
                center_x,
                center_y,
                radius,
                width,
                height,
                rotation_angle=0  # No rotation initially
            )

            # STEP 3: Rotate the entire canvas (with noise and mask) around the grid cell center
            # Create a composite of noise and mask, rotate it, then extract the rotated versions

            # Use scipy's rotation with offset to rotate around a specific point
            # We'll rotate the entire canvas, which rotates everything around the canvas center
            # To rotate around (center_x, center_y), we need to:
            # 1. Shift so (center_x, center_y) is at canvas center
            # 2. Rotate
            # 3. Shift back

            from scipy.ndimage import shift

            # Calculate shift to move grid cell center to canvas center
            canvas_center_y = height // 2
            canvas_center_x = width // 2
            shift_y = canvas_center_y - center_y
            shift_x = canvas_center_x - center_x

            # Shift noise and mask so grid center is at canvas center
            shifted_noise = shift(shape_noise_canvas, (shift_y, shift_x), order=1, mode='constant', cval=0)
            shifted_mask = shift(shape_mask_unrotated, (shift_y, shift_x), order=1, mode='constant', cval=0)

            # Rotate around canvas center (which is now the grid cell center)
            rotated_shifted_noise = rotate_noise(shifted_noise, rotation_angle)
            rotated_shifted_mask = rotate_noise(shifted_mask, rotation_angle)

            # Shift back to original position
            rotated_noise_canvas = shift(rotated_shifted_noise, (-shift_y, -shift_x), order=1, mode='constant', cval=0)
            shape_mask_rotated = shift(rotated_shifted_mask, (-shift_y, -shift_x), order=1, mode='constant', cval=0)

            # Apply soft blur to mask for smoother edges
            shape_mask_rotated = ndimage.gaussian_filter(shape_mask_rotated, sigma=3.0)

            # Apply shape signal (rotated noise within the rotating shape boundary)
            shape_signal = base_luminance + noise_amplitude * rotated_noise_canvas
            img_array = img_array * (1 - shape_mask_rotated) + shape_signal * shape_mask_rotated

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

    # Instead of saving as one GIF, save each cell as a separate GIF
    # This allows users to click on individual cells like the Mirror CAPTCHA

    # Extract each cell into separate GIF files
    cell_gifs = []
    for cell_idx in range(total_cells):
        row = cell_idx // cols
        col = cell_idx % cols

        # Extract this cell from all frames
        cell_frames = []
        for frame in frames:
            # Crop this cell from the frame
            y_start = row * cell_size
            y_end = (row + 1) * cell_size
            x_start = col * cell_size
            x_end = (col + 1) * cell_size

            cell_frame = frame.crop((x_start, y_start, x_end, y_end))
            cell_frames.append(cell_frame)

        # Save this cell as a separate GIF
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        cell_filename = f"{base_name}_cell_{cell_idx}.gif"
        cell_path = os.path.join(os.path.dirname(output_path), cell_filename)

        cell_frames[0].save(
            cell_path,
            save_all=True,
            append_images=cell_frames[1:],
            duration=int(1000/fps),
            loop=0
        )

        cell_gifs.append(cell_filename)

    # Also save the full grid for reference/debugging
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000/fps),
        loop=0
    )

    # Get answer (indices of target cells)
    answer = sorted([idx for idx in shape_cell_indices if cell_config[idx]['is_target']])
    target_cells = [(idx // cols, idx % cols) for idx in answer]

    shape_emoji = {
        'circle': '⭕',
        'square': '⬜',
        'triangle': '🔺'
    }

    return {
        'grid_size': grid_size,
        'total_cells': total_cells,
        'target_shape': target_shape,
        'target_direction': target_direction,
        'total_shapes': total_shapes,
        'num_targets': num_targets,
        'target_cells': target_cells,
        'target_cell_indices': answer,
        'answer': answer,
        'all_shape_indices': shape_cell_indices,
        'cell_config': {str(k): v for k, v in cell_config.items()},
        'prompt': f"Click all {shape_emoji.get(target_shape, '')} {target_shape}s rotating {target_direction}",
        'description': f"Grid with {total_shapes} shapes, {num_targets} {target_shape}s rotating {target_direction}",
        'cell_gifs': cell_gifs  # List of individual cell GIF filenames
    }


def generate_dataset(output_dir, num_samples=20, grid_size=(3, 3)):
    """
    Generate a dataset of Spooky Shape Grid CAPTCHAs.

    Args:
        output_dir: Directory to save the generated GIFs
        num_samples: Number of samples to generate
        grid_size: Grid dimensions (rows, cols)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = {}

    for i in range(num_samples):
        filename = f"spooky_shape_grid_{i:04d}.gif"
        output_path = output_dir / filename

        print(f"Generating {filename}...")

        metadata = generate_spooky_shape_grid_gif(
            output_path=str(output_path),
            grid_size=grid_size,
            cell_size=120,
            num_frames=24,
            fps=12,
            seed=None,
            target_shape=None,
            target_direction=None,
            num_targets=None
        )

        # Create user-friendly prompt
        shape_emoji = {
            'circle': '⭕',
            'square': '⬜',
            'triangle': '🔺'
        }

        prompt = f"Click all {shape_emoji.get(metadata['target_shape'], '')} {metadata['target_shape']}s rotating {metadata['target_direction']}"

        # Store ground truth in Mirror CAPTCHA format
        # Use a puzzle_id (without .gif extension) as the key
        puzzle_id = os.path.splitext(filename)[0]

        ground_truth[puzzle_id] = {
            "answer": metadata["answer"],
            "prompt": prompt,
            "description": metadata["description"],
            "options": metadata["cell_gifs"],  # List of cell GIF filenames
            "grid_size": metadata["grid_size"],
            "difficulty": 5,
            "media_type": "gif",
            "target_shape": metadata["target_shape"],
            "target_direction": metadata["target_direction"],
            "target_cells": metadata["target_cells"],
            "target_cell_indices": metadata["target_cell_indices"],
            "all_shape_indices": metadata["all_shape_indices"],
            "cell_config": metadata["cell_config"]
        }

        print(f"  → Shape: {metadata['target_shape']}, Direction: {metadata['target_direction']}")
        print(f"  → Target cells: {metadata['target_cells']}")

    # Save ground truth
    ground_truth_path = output_dir / "ground_truth.json"
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGenerated {num_samples} samples!")
    print(f"Output directory: {output_dir}")
    print(f"Ground truth: {ground_truth_path}")


if __name__ == "__main__":
    output_dir = str(Path(__file__).parent.parent / "captcha_data" / "Spooky_Shape_Grid")

    generate_dataset(output_dir, num_samples=20, grid_size=(3, 3))

    print("\n" + "="*70)
    print("🎯 Spooky Shape Grid CAPTCHA Dataset Generated!")
    print("Shape + Direction Motion Detection")
    print("="*70)
    print("\n🔬 Technical Implementation:")
    print("  ✓ 3 Shape Types: Circle, Square, Triangle")
    print("  ✓ 2 Rotation Types: Clockwise, Counterclockwise")
    print("  ✓ Equal rotation radius for all shapes (fair comparison)")
    print("  ✓ Rotation: 5°/frame (smooth motion)")
    print("  ✓ Grid: 3×3 with mixed shapes and directions")
    print("  ✓ Targets: 2-4 shapes matching specific criteria")
    print("\n📊 Challenge Structure:")
    print("  • 3×3 grid with 4-10 shapes total")
    print("  • 2-4 shapes match TARGET (shape + direction)")
    print("  • Others are DISTRACTORS (different shape OR direction)")
    print("  • User must identify only matching cells")
    print("\n🧠 Why Humans Can See It:")
    print("  • Shape recognition: Circle, square, triangle easily distinguished")
    print("  • Motion perception: Clockwise vs counterclockwise rotation")
    print("  • Compound task: Must match BOTH shape AND direction")
    print("  • After 2-3 seconds: Target cells become clear")
    print("\n🤖 Why LLMs/Vision Models Fail:")
    print("  ✗ Need to segment multiple simultaneous rotations")
    print("  ✗ Must recognize 3 different shape types in noise")
    print("  ✗ Must detect rotation direction for each shape")
    print("  ✗ Must perform compound matching (shape AND direction)")
    print("  ✗ Most models lack multi-object motion tracking")
    print("\n🏆 Advanced Compound Motion CAPTCHA!")
    print("    Humans: Moderate - requires attention to detail")
    print("    LLMs: Very hard - requires shape + motion analysis")
    print("    Perfect for testing compound visual reasoning!")
