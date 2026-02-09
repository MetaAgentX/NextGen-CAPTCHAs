"""
Spooky Size Comparison CAPTCHA Generator

Generates animated GIFs where users must click the largest or smallest shape
(triangle, square, or circle) among multiple rotating shapes. The shapes are
only visible through motion contrast - static frames show noise, but rotation
reveals the shapes.
"""

import numpy as np
from PIL import Image, ImageDraw
import json
import random
from pathlib import Path
from scipy import ndimage

# Configuration
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 400
NUM_FRAMES = 36
FPS = 12
NUM_PUZZLES = 20

# Shape parameters
SHAPES = ['circle', 'square', 'triangle']
SIZE_SMALL = (70, 80)
SIZE_MEDIUM = (100, 110)
SIZE_LARGE = (130, 140)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "captcha_data" / "Spooky_Size"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth.json"


def generate_mid_frequency_noise(height, width, sigma=3.0):
    """Generate mid-spatial frequency noise."""
    noise = np.random.randn(height, width)
    filtered_noise = ndimage.gaussian_filter(noise, sigma=sigma)
    filtered_noise = (filtered_noise - filtered_noise.min()) / (filtered_noise.max() - filtered_noise.min())
    return filtered_noise


def rotate_noise(noise_field, angle_degrees):
    """Rotate noise field by given angle."""
    return ndimage.rotate(noise_field, angle_degrees, reshape=False, order=1, mode='wrap')


def create_shape_mask_on_canvas(shape_type, center_x, center_y, radius, width, height, rotation_angle=0):
    """
    Create a binary mask for a rotating shape on full canvas.

    Args:
        shape_type: 'circle', 'square', or 'triangle'
        center_x, center_y: Center position on canvas
        radius: Radius of the shape (distance to furthest point)
        width, height: Canvas dimensions
        rotation_angle: Current rotation angle in degrees

    Returns:
        Binary mask array (0-1) where rotated shape is rendered
    """
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if shape_type == 'circle':
        bbox = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius
        ]
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
            base_angle = rotation_angle - 90 + i * 120
            angle_rad = np.radians(base_angle)
            x = center_x + radius * np.cos(angle_rad)
            y = center_y + radius * np.sin(angle_rad)
            points.append((x, y))
        draw.polygon(points, fill=255)

    mask_array = np.array(mask_img).astype(np.float32) / 255.0
    return mask_array


def generate_puzzle(puzzle_id, puzzle_idx):
    """Generate a single Spooky Size puzzle."""

    # Use deterministic seed based on puzzle index
    seed = puzzle_idx * 54321
    random.seed(seed)
    np.random.seed(seed)

    # Choose target shape type
    target_shape = random.choice(SHAPES)

    # Choose comparison type
    comparison = random.choice(['smallest', 'largest'])

    # Determine number of shapes (7-10 total, with 3-4 of target shape type)
    num_target_shapes = random.randint(3, 4)  # Multiple instances of target shape
    num_other_shapes = random.randint(4, 6)   # Other shape types as distractors
    num_shapes = num_target_shapes + num_other_shapes

    # OTHER SHAPE TYPES (distractors)
    other_shape_types = [s for s in SHAPES if s != target_shape]

    # Generate sizes for target shapes
    # ONE target with the correct size, others of same shape with different sizes
    if comparison == 'smallest':
        # Target is the smallest of its shape type
        target_size = random.randint(SIZE_SMALL[0], SIZE_SMALL[1])
        # Other instances of same shape should be LARGER
        same_shape_sizes = [random.randint(SIZE_MEDIUM[0], SIZE_LARGE[1])
                           for _ in range(num_target_shapes - 1)]
    else:  # largest
        # Target is the largest of its shape type
        target_size = random.randint(SIZE_LARGE[0], SIZE_LARGE[1])
        # Other instances of same shape should be SMALLER
        same_shape_sizes = [random.randint(SIZE_SMALL[0], SIZE_MEDIUM[1])
                           for _ in range(num_target_shapes - 1)]

    # Generate sizes for other shape types (any size)
    other_shapes_sizes = [random.randint(SIZE_SMALL[0], SIZE_LARGE[1])
                         for _ in range(num_other_shapes)]

    # Combine all sizes
    all_sizes = [target_size] + same_shape_sizes + other_shapes_sizes
    target_index = 0  # Target is first in the list

    # Assign shape types
    all_shape_types = [target_shape] * num_target_shapes  # Multiple of target shape
    for _ in range(num_other_shapes):
        all_shape_types.append(random.choice(other_shape_types))

    # Shuffle to randomly place target
    indices = list(range(num_shapes))
    random.shuffle(indices)
    shuffled_sizes = [all_sizes[i] for i in indices]
    shuffled_shapes = [all_shape_types[i] for i in indices]
    target_index = indices.index(0)

    # Place shapes without overlap
    # STRATEGY: Place target FIRST to ensure it always succeeds
    margin = 50
    placed_shapes = []
    shapes_data = []

    # Place target shape first
    target_size = shuffled_sizes[target_index]
    target_shape_type = shuffled_shapes[target_index]
    target_radius = target_size // 2

    # Generate CENTER position first (for shape rendering)
    target_center_x = random.randint(margin + target_radius, CANVAS_WIDTH - margin - target_radius)
    target_center_y = random.randint(margin + target_radius, CANVAS_HEIGHT - margin - target_radius)

    # Convert to TOP-LEFT position (like Red Dot stores it)
    target_top_left_x = target_center_x - target_radius
    target_top_left_y = target_center_y - target_radius

    placed_shapes.append((target_center_x, target_center_y, target_size))
    shapes_data.append({
        'shape': target_shape_type,
        'size': target_size,
        'radius': target_radius,
        'x': target_center_x,  # Store center for rendering
        'y': target_center_y,
        'direction': random.choice(['clockwise', 'counterclockwise']),
        'is_target': True
    })

    # Place other shapes
    for i, (size, shape_type) in enumerate(zip(shuffled_sizes, shuffled_shapes)):
        if i == target_index:
            continue  # Already placed

        radius = size // 2
        attempts = 0
        placed = False

        while attempts < 200:
            x = random.randint(margin + radius, CANVAS_WIDTH - margin - radius)
            y = random.randint(margin + radius, CANVAS_HEIGHT - margin - radius)

            # Check overlap
            overlap = False
            for ex, ey, es in placed_shapes:
                dist = np.sqrt((x - ex)**2 + (y - ey)**2)
                min_required = (size + es) / 2 + 20
                if dist < min_required:
                    overlap = True
                    break

            if not overlap:
                placed_shapes.append((x, y, size))
                shapes_data.append({
                    'shape': shape_type,
                    'size': size,
                    'radius': radius,
                    'x': x,
                    'y': y,
                    'direction': random.choice(['clockwise', 'counterclockwise']),
                    'is_target': False
                })
                placed = True
                break

            attempts += 1

        # If we can't place this shape, just skip it (not critical since target is placed)
        if not placed:
            print(f"  Warning: Could not place shape {i} after {attempts} attempts, skipping")

    # Generate background noise - large enough to avoid wrapping artifacts
    scroll_total = NUM_FRAMES * 1  # 1 pixel per frame scroll
    pad = scroll_total + 10
    bg_noise = generate_mid_frequency_noise(CANVAS_HEIGHT + pad, CANVAS_WIDTH, sigma=3.0)
    bg_noise = (bg_noise - 0.5) * 2.0  # Scale to -1 to 1

    # Generate shape noise patterns
    for shape_data in shapes_data:
        region_size = int(shape_data['radius'] * 3)
        base_noise = generate_mid_frequency_noise(region_size, region_size, sigma=2.5)
        base_noise = (base_noise - 0.5) * 2.0

        shape_data['base_noise'] = base_noise
        shape_data['region_size'] = region_size

    # Visual parameters
    base_luminance = 128.0
    noise_amplitude = 70.0
    rotation_speed = 5  # Degrees per frame

    # Generate frames
    frames = []
    for frame_idx in range(NUM_FRAMES):
        # Start with background - slice without wrapping
        bg_start = frame_idx * 1
        bg_frame = bg_noise[bg_start:bg_start+CANVAS_HEIGHT, 0:CANVAS_WIDTH]
        img_array = base_luminance + noise_amplitude * bg_frame

        # Add each rotating shape
        for shape_data in shapes_data:
            shape_type = shape_data['shape']
            center_x = shape_data['x']
            center_y = shape_data['y']
            radius = shape_data['radius']
            direction = shape_data['direction']
            base_noise = shape_data['base_noise']
            region_size = shape_data['region_size']

            # Calculate rotation angle
            if direction == 'clockwise':
                rotation_angle = -rotation_speed * frame_idx
            else:
                rotation_angle = rotation_speed * frame_idx

            # Place unrotated noise on canvas
            shape_noise_canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH))
            half_region = region_size // 2
            y_start = max(0, center_y - half_region)
            y_end = min(CANVAS_HEIGHT, center_y + half_region)
            x_start = max(0, center_x - half_region)
            x_end = min(CANVAS_WIDTH, center_x + half_region)

            noise_y_start = half_region - (center_y - y_start)
            noise_y_end = noise_y_start + (y_end - y_start)
            noise_x_start = half_region - (center_x - x_start)
            noise_x_end = noise_x_start + (x_end - x_start)

            shape_noise_canvas[y_start:y_end, x_start:x_end] = base_noise[noise_y_start:noise_y_end, noise_x_start:noise_x_end]

            # Create unrotated shape mask
            shape_mask_unrotated = create_shape_mask_on_canvas(
                shape_type, center_x, center_y, radius,
                CANVAS_WIDTH, CANVAS_HEIGHT, rotation_angle=0
            )

            # Rotate both noise and mask around shape center
            from scipy.ndimage import shift

            canvas_center_y = CANVAS_HEIGHT // 2
            canvas_center_x = CANVAS_WIDTH // 2
            shift_y = canvas_center_y - center_y
            shift_x = canvas_center_x - center_x

            # Shift to center
            shifted_noise = shift(shape_noise_canvas, (shift_y, shift_x), order=1, mode='constant', cval=0)
            shifted_mask = shift(shape_mask_unrotated, (shift_y, shift_x), order=1, mode='constant', cval=0)

            # Rotate
            rotated_noise = rotate_noise(shifted_noise, rotation_angle)
            rotated_mask = rotate_noise(shifted_mask, rotation_angle)

            # Shift back
            final_noise = shift(rotated_noise, (-shift_y, -shift_x), order=1, mode='constant', cval=0)
            final_mask = shift(rotated_mask, (-shift_y, -shift_x), order=1, mode='constant', cval=0)

            # Apply Gaussian blur to mask for smoother edges
            final_mask = ndimage.gaussian_filter(final_mask, sigma=3.0)

            # Blend shape noise with background using mask
            shape_signal = base_luminance + noise_amplitude * final_noise
            img_array = img_array * (1 - final_mask) + shape_signal * final_mask

        # Convert to uint8
        frame = np.clip(img_array, 0, 255).astype(np.uint8)
        frames.append(frame)

    # Save as GIF
    gif_filename = f"{puzzle_id}.gif"
    gif_path = OUTPUT_DIR / gif_filename

    pil_frames = [Image.fromarray(frame, mode='L') for frame in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / FPS),
        loop=0
    )

    # Create ground truth - store TOP-LEFT position like Red Dot does
    # Backend will convert to center: center_x = x + diameter/2
    prompt = f"Click the {comparison} {target_shape}"

    # Store all shapes for visualization (exclude noise data)
    all_shapes = []
    for sd in shapes_data:
        all_shapes.append({
            'shape': sd['shape'],
            'size': sd['size'],
            'radius': sd['radius'],
            'x': sd['x'],
            'y': sd['y'],
            'is_target': sd['is_target']
        })

    puzzle_data = {
        "prompt": prompt,
        "description": f"Find and click the {comparison} {target_shape} among {num_shapes} shapes",
        "media_type": "gif",
        "difficulty": 6,
        "target_position": {
            "x": int(target_top_left_x),      # Top-left X (like Red Dot)
            "y": int(target_top_left_y),      # Top-left Y (like Red Dot)
            "diameter": int(target_size)       # Full diameter (like Red Dot)
        },
        "target_shape": target_shape,
        "target_size": int(target_size),
        "comparison": comparison,
        "num_shapes": num_shapes,
        "all_shapes": all_shapes  # Store all shapes for visualization
    }

    return puzzle_id, puzzle_data


def main():
    print("Generating Spooky Size Comparison CAPTCHA puzzles...")

    ground_truth = {}

    for i in range(NUM_PUZZLES):
        puzzle_id = f"spooky_size_{i:04d}"
        puzzle_id_key, puzzle_data = generate_puzzle(puzzle_id, i)
        ground_truth[puzzle_id_key] = puzzle_data
        print(f"✓ Generated {puzzle_id}: {puzzle_data['prompt']}")

    # Save ground truth
    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Successfully generated {NUM_PUZZLES} Spooky Size puzzles")
    print(f"✓ Ground truth saved to {GROUND_TRUTH_FILE}")


if __name__ == "__main__":
    main()
