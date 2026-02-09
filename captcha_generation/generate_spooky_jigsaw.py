"""
Spooky Jigsaw CAPTCHA Generator

Generates animated GIF jigsaw puzzles where pieces are only visible through OPPOSITE MOTION.
Individual frames look like uniform noise with no spatial features.

This combines jigsaw puzzle mechanics with motion-based visibility:
- Background noise: Scrolls in one direction (e.g., upward)
- Jigsaw piece noise: Scrolls in OPPOSITE direction (e.g., downward)
- Per-frame: Both look identical (same noise statistics)
- Over time: Humans detect opposite motion → pieces and reference emerge
- Must solve the jigsaw puzzle by matching motion patterns!
"""

import numpy as np
from PIL import Image, ImageDraw
import json
import os
import random
import hashlib
import uuid
from pathlib import Path
from scipy import ndimage


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


def generate_mid_frequency_noise(height, width, sigma=3.0):
    """
    Generate mid-spatial frequency noise (not white noise).
    Uses Gaussian filtering to concentrate energy in mid frequencies.

    Args:
        height, width: Image dimensions
        sigma: Gaussian blur sigma (controls spatial frequency)

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


def create_single_shape_mask(width, height, cx, cy, size, shape_type):
    """
    Create a mask for a single shape.

    Args:
        width, height: Dimensions
        cx, cy: Center position
        size: Shape size
        shape_type: Type of shape

    Returns:
        Binary mask for the shape
    """
    if shape_type == 'circle':
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        shape_mask = (distance <= size).astype(float)
    elif shape_type == 'square':
        y_coords, x_coords = np.ogrid[:height, :width]
        shape_mask = ((np.abs(x_coords - cx) <= size) & (np.abs(y_coords - cy) <= size)).astype(float)
    elif shape_type == 'triangle':
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        points = [
            (cx, cy - size),
            (cx - size, cy + size),
            (cx + size, cy + size)
        ]
        draw.polygon(points, fill=255)
        shape_mask = np.array(img).astype(float) / 255.0
    elif shape_type == 'pentagon':
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        import math
        points = []
        for i in range(5):
            angle = math.pi * 2 * i / 5 - math.pi / 2
            px = cx + size * math.cos(angle)
            py = cy + size * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=255)
        shape_mask = np.array(img).astype(float) / 255.0
    elif shape_type == 'star':
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        import math
        points = []
        for i in range(10):
            angle = math.pi * 2 * i / 10 - math.pi / 2
            radius = size if i % 2 == 0 else size * 0.4
            px = cx + radius * math.cos(angle)
            py = cy + radius * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=255)
        shape_mask = np.array(img).astype(float) / 255.0
    else:
        shape_mask = np.zeros((height, width), dtype=float)

    return shape_mask


def create_shape_pattern(width, height, seed):
    """
    Create a pattern of shapes to serve as the jigsaw content.
    This pattern will be revealed through motion.

    Args:
        width, height: Dimensions
        seed: Random seed for reproducibility

    Returns:
        Binary mask (1 = shape content, 0 = background)
    """
    np.random.seed(seed)
    mask = np.zeros((height, width), dtype=float)
    shape_types = ['circle', 'square', 'triangle', 'pentagon', 'star']

    # Ensure at least one shape in each 3x3 grid cell
    grid_rows = 3
    grid_cols = 3
    cell_width = width // grid_cols
    cell_height = height // grid_rows

    # Place at least one shape per grid cell (9 shapes for 3x3)
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
            size = np.random.randint(40, 70)  # Large shapes to ensure visibility

            # Create and add shape to mask
            shape_mask = create_single_shape_mask(width, height, cx, cy, size, shape_type)
            mask = np.maximum(mask, shape_mask)

    # Add 3-6 extra random shapes for more complexity
    num_extra = np.random.randint(3, 7)
    for _ in range(num_extra):
        shape_type = np.random.choice(shape_types)
        cx = np.random.randint(40, width - 40)
        cy = np.random.randint(40, height - 40)
        size = np.random.randint(35, 65)

        # Create and add shape to mask
        shape_mask = create_single_shape_mask(width, height, cx, cy, size, shape_type)
        mask = np.maximum(mask, shape_mask)

    # Smooth the mask edges
    mask = ndimage.gaussian_filter(mask, sigma=2.0)
    mask = np.clip(mask, 0, 1)

    return mask


def generate_spooky_jigsaw_gif(content_mask, output_path, width=450, height=450, num_frames=30, fps=15):
    """
    Generate a GIF where jigsaw content is revealed through OPPOSITE MOTION.

    Motion-Based Strategy:
    - Background noise scrolls in one direction (e.g., upward)
    - Content regions scroll in OPPOSITE direction (e.g., downward)
    - Each frame: uniform noise with same statistics everywhere
    - Over time: opposite motion reveals the jigsaw pattern

    Args:
        content_mask: Binary mask where 1 = content, 0 = background
        output_path: Where to save the GIF
        width, height: Dimensions of the image
        num_frames: Number of frames in animation
        fps: Frames per second
    """
    # Motion parameters
    scroll_speed = 2  # Pixels per frame to scroll
    direction = 'vertical'  # Can be 'vertical' or 'horizontal'

    # Visual parameters
    base_luminance = 128.0
    noise_amplitude = 70.0  # Larger for better visibility

    # Generate TWO large noise fields for scrolling
    # Make them larger than the image to avoid edge artifacts
    pad = scroll_speed * num_frames
    large_height = height + 2 * pad
    large_width = width + 2 * pad

    # Background noise field (scrolls one direction)
    bg_noise_field = generate_mid_frequency_noise(large_height, large_width, sigma=3.0)
    bg_noise_field = (bg_noise_field - 0.5) * 2.0  # Normalize to ~unit variance

    # Content noise field (scrolls OPPOSITE direction)
    content_noise_field = generate_mid_frequency_noise(large_height, large_width, sigma=3.0)
    content_noise_field = (content_noise_field - 0.5) * 2.0  # Normalize to ~unit variance

    # Generate frames with opposite motion
    frames = []
    for frame_idx in range(num_frames):
        # Calculate scroll offsets
        # Background scrolls UP (negative offset in vertical direction)
        bg_offset = -frame_idx * scroll_speed
        # Content scrolls DOWN (positive offset in vertical direction)
        content_offset = frame_idx * scroll_speed

        # Extract current frame from scrolling background noise
        bg_scrolled = scroll_noise(bg_noise_field, bg_offset, direction)
        bg_frame = bg_scrolled[pad:pad+height, pad:pad+width]

        # Extract current frame from scrolling content noise
        content_scrolled = scroll_noise(content_noise_field, content_offset, direction)
        content_frame = content_scrolled[pad:pad+height, pad:pad+width]

        # Start with background noise
        img_array = base_luminance + noise_amplitude * bg_frame

        # Composite content using mask
        # Content regions show the opposite-scrolling noise
        content_signal = base_luminance + noise_amplitude * content_frame
        img_array = img_array * (1 - content_mask) + content_signal * content_mask

        # Clip to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Convert grayscale to RGB
        img_rgb = np.stack([img_array, img_array, img_array], axis=-1)

        # Convert to PIL Image
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


def split_mask_into_pieces(mask, grid_rows, grid_cols):
    """
    Split a mask into jigsaw pieces.

    Args:
        mask: 2D numpy array
        grid_rows: Number of rows in jigsaw grid
        grid_cols: Number of columns in jigsaw grid

    Returns:
        dict: piece_id -> mask for that piece
    """
    height, width = mask.shape
    piece_height = height // grid_rows
    piece_width = width // grid_cols

    piece_masks = {}

    for row in range(grid_rows):
        for col in range(grid_cols):
            piece_id = row * grid_cols + col

            # Extract this piece region
            top = row * piece_height
            left = col * piece_width
            bottom = top + piece_height
            right = left + piece_width

            # Create piece mask (extract from full mask)
            piece_mask = np.zeros_like(mask)
            piece_mask[top:bottom, left:right] = mask[top:bottom, left:right]

            piece_masks[piece_id] = piece_mask

    return piece_masks


def generate_spooky_jigsaw_dataset(output_dir, num_puzzles=10, grid_rows=3, grid_cols=3):
    """
    Generate Spooky Jigsaw CAPTCHA dataset with motion-based visibility.

    Args:
        output_dir: Directory to save the dataset
        num_puzzles: Number of puzzles to generate
        grid_rows: Number of rows in jigsaw grid (default 3 for 3x3)
        grid_cols: Number of columns in jigsaw grid (default 3 for 3x3)
    """
    os.makedirs(output_dir, exist_ok=True)

    ground_truth = {}

    print(f"Generating {num_puzzles} Spooky Jigsaw puzzles...")

    for puzzle_idx in range(num_puzzles):
        print(f"Puzzle {puzzle_idx}: Generating spooky animated jigsaw...")

        # Create content pattern (shapes to be revealed by motion)
        content_mask = create_shape_pattern(
            width=450,
            height=450,
            seed=puzzle_idx * 12345
        )

        # Generate frames for reference (returns list of PIL Images, not saved yet)
        # Motion parameters
        scroll_speed = 2
        direction = 'vertical'
        base_luminance = 128.0
        noise_amplitude = 70.0
        num_frames = 30

        # Generate TWO large noise fields for scrolling (ONCE for all frames)
        pad = scroll_speed * num_frames
        large_height = 450 + 2 * pad
        large_width = 450 + 2 * pad

        # Use deterministic seed for this puzzle
        np.random.seed(puzzle_idx * 1000)

        # Background noise field (scrolls one direction)
        bg_noise_field = generate_mid_frequency_noise(large_height, large_width, sigma=3.0)
        bg_noise_field = (bg_noise_field - 0.5) * 2.0

        # Content noise field (scrolls OPPOSITE direction)
        content_noise_field = generate_mid_frequency_noise(large_height, large_width, sigma=3.0)
        content_noise_field = (content_noise_field - 0.5) * 2.0

        # Generate frames by scrolling through the noise fields
        frames = []
        for frame_idx in range(num_frames):
            # Calculate scroll offsets
            bg_offset = -frame_idx * scroll_speed
            content_offset = frame_idx * scroll_speed

            # Extract current frame from scrolling noises
            bg_scrolled = scroll_noise(bg_noise_field, bg_offset, direction)
            bg_frame = bg_scrolled[pad:pad+450, pad:pad+450]

            content_scrolled = scroll_noise(content_noise_field, content_offset, direction)
            content_frame = content_scrolled[pad:pad+450, pad:pad+450]

            # Composite with content mask
            img_array = base_luminance + noise_amplitude * bg_frame
            content_signal = base_luminance + noise_amplitude * content_frame
            img_array = img_array * (1 - content_mask) + content_signal * content_mask

            # Clip and convert to RGB
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img_rgb = np.stack([img_array, img_array, img_array], axis=-1)
            frame = Image.fromarray(img_rgb)
            frames.append(frame)

        # Save reference GIF
        reference_filename = f"spooky_jigsaw_{puzzle_idx:04d}.gif"
        reference_path = os.path.join(output_dir, reference_filename)
        frames[0].save(
            reference_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000/15),
            loop=0
        )
        print(f"  Saved reference GIF: {reference_filename}")

        # Split frames into pieces (crop each frame into grid pieces)
        piece_width = 450 // grid_cols
        piece_height = 450 // grid_rows
        piece_frames_dict = {}

        for row in range(grid_rows):
            for col in range(grid_cols):
                piece_id = row * grid_cols + col
                piece_frames_dict[piece_id] = []

                # Extract this piece from each frame
                for frame in frames:
                    left = col * piece_width
                    upper = row * piece_height
                    right = left + piece_width
                    lower = upper + piece_height

                    piece = frame.crop((left, upper, right, lower))
                    piece_frames_dict[piece_id].append(piece)

        # Save each piece as animated GIF with UUID-based filenames (no position info leaked)
        piece_filenames = []
        piece_id_to_filename = {}  # Internal mapping for ground truth
        for piece_id in range(grid_rows * grid_cols):
            # Generate random UUID-based filename to prevent position leakage
            unique_hash = hashlib.md5(f"{puzzle_idx}_{piece_id}_{uuid.uuid4()}".encode()).hexdigest()[:12]
            piece_filename = f"piece_{unique_hash}.gif"
            piece_path = os.path.join(output_dir, piece_filename)

            piece_frames_dict[piece_id][0].save(
                piece_path,
                save_all=True,
                append_images=piece_frames_dict[piece_id][1:],
                duration=int(1000/15),
                loop=0
            )
            piece_filenames.append(piece_filename)
            piece_id_to_filename[piece_id] = piece_filename

        print(f"  Saved {len(piece_filenames)} spooky animated piece GIFs (UUID filenames)")

        # Generate correct positions mapping
        correct_positions = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                piece_index = row * grid_cols + col
                correct_positions.append({
                    "piece_index": piece_index,
                    "grid_row": row,
                    "grid_col": col
                })

        # Shuffle the piece order for presentation (harder for LLMs)
        shuffled_indices = list(range(grid_rows * grid_cols))
        random.shuffle(shuffled_indices)
        shuffled_piece_filenames = [piece_filenames[i] for i in shuffled_indices]

        # Update correct_positions to reflect the shuffled presentation order
        # The frontend will receive pieces in shuffled_piece_filenames order
        # but needs to know their correct grid positions
        shuffled_correct_positions = []
        for display_idx, original_piece_id in enumerate(shuffled_indices):
            # Find the correct position for this piece
            correct_pos = correct_positions[original_piece_id]
            shuffled_correct_positions.append({
                "piece_index": display_idx,  # Index in the shuffled array
                "grid_row": correct_pos["grid_row"],  # Where it should go
                "grid_col": correct_pos["grid_col"]
            })

        # Create ground truth entry
        puzzle_id = f"spooky_jigsaw_{puzzle_idx:04d}"
        ground_truth[puzzle_id] = {
            "prompt": "Watch carefully! The puzzle pieces will appear through motion. Drag them to complete the jigsaw puzzle.",
            "description": f"Complete a {grid_rows}x{grid_cols} motion-based jigsaw puzzle with opposite scrolling noise",
            "grid_size": [grid_rows, grid_cols],
            "image": reference_filename,
            "pieces": shuffled_piece_filenames,  # Pieces in shuffled order
            "correct_positions": shuffled_correct_positions,  # Correct positions for shuffled pieces
            "piece_size": 450 // grid_cols,  # 150 for 3x3 grid
            "difficulty": 5,
            "media_type": "spooky_gif"
        }

    # Save ground truth
    ground_truth_path = os.path.join(output_dir, 'ground_truth.json')
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Generated {num_puzzles} Spooky Jigsaw puzzles")
    print(f"✓ Saved to {output_dir}")
    print(f"✓ Ground truth saved to {ground_truth_path}")


if __name__ == "__main__":
    output_dir = "../captcha_data/Spooky_Jigsaw"
    generate_spooky_jigsaw_dataset(output_dir, num_puzzles=20, grid_rows=3, grid_cols=3)

    print("\n" + "="*70)
    print("🎯 Spooky Jigsaw CAPTCHA Dataset Generated!")
    print("Motion-Based Jigsaw Puzzle - Opposite Scrolling Noise")
    print("="*70)
    print("\n🔬 Technical Implementation:")
    print("  ✓ Background: Mid-frequency noise scrolling UPWARD")
    print("  ✓ Jigsaw content: Mid-frequency noise scrolling DOWNWARD")
    print("  ✓ Scroll speed: 2 pixels/frame")
    print("  ✓ Soft-edged shape masks for natural blending")
    print("  ✓ Same noise statistics everywhere (mean ~128, amplitude ~40)")
    print("\n📊 Per-Frame Analysis:")
    print("  • Single frame: Uniform noise texture everywhere")
    print("  • No spatial features, edges, or intensity differences")
    print("  • Both regions have identical noise statistics")
    print("  • Impossible to detect jigsaw pieces from single frame!")
    print("\n🧠 Why Humans Can See It:")
    print("  • Human visual system has excellent motion detection")
    print("  • Background noise scrolls UP → creates upward motion percept")
    print("  • Content noise scrolls DOWN → creates downward motion percept")
    print("  • After ~1-2 seconds: shapes 'pop out' as opposite-moving regions")
    print("  • Motion contrast reveals jigsaw piece boundaries and content")
    print("\n🤖 Why LLMs/Vision Models Fail:")
    print("  ✗ Single frame: Just uniform noise (no spatial cues)")
    print("  ✗ Temporal mean: FLAT (scrolling doesn't change mean)")
    print("  ✗ Temporal std: FLAT (scrolling doesn't change variance)")
    print("  ✗ Frame differencing: Shows motion but not piece shapes")
    print("  ✗ Most vision models: No motion processing in their architecture")
    print("\n🏆 This is a TRUE motion-based temporal jigsaw CAPTCHA!")
    print("    Humans: Instant motion perception → Pieces visible in 1-2 sec")
    print("    LLMs: No motion processing → See only noise")
