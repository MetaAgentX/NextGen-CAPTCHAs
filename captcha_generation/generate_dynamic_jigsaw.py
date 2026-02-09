"""
Generate dynamic Jigsaw CAPTCHA puzzles with animated GIFs.

The reference image is an animated GIF with objects moving slowly.
Each jigsaw piece is also an animated GIF that moves in sync.
"""

import os
import json
import random
import hashlib
import uuid
from PIL import Image, ImageDraw
import numpy as np


def create_moving_shapes_gif(width=450, height=450, num_frames=30, grid_rows=3, grid_cols=3):
    """
    Create an animated GIF with slowly moving shapes.
    Ensures each grid cell contains at least one shape.

    Args:
        width: Image width (increased for 3x3 grid)
        height: Image height (increased for 3x3 grid)
        num_frames: Number of frames in animation
        grid_rows: Number of rows (to ensure each cell has a shape)
        grid_cols: Number of columns (to ensure each cell has a shape)

    Returns:
        list of PIL Images (frames)
    """
    frames = []

    # Generate random shapes with initial positions and velocities
    shapes = []
    colors = [
        (220, 50, 50),   # Red
        (50, 180, 50),   # Green
        (50, 100, 220),  # Blue
        (200, 150, 50),  # Orange
        (150, 50, 200),  # Purple
        (50, 200, 200),  # Cyan
        (220, 100, 150), # Pink
        (100, 150, 50)   # Olive
    ]
    shape_types = ['circle', 'square', 'triangle', 'pentagon', 'star']

    # Calculate cell dimensions
    cell_width = width // grid_cols
    cell_height = height // grid_rows

    # Place at least one shape in each grid cell
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Calculate cell boundaries
            cell_left = col * cell_width
            cell_top = row * cell_height
            cell_right = cell_left + cell_width
            cell_bottom = cell_top + cell_height

            # Place shape in the center region of this cell
            margin = 15  # Reduced margin for more overlap
            shape = {
                'type': random.choice(shape_types),
                'x': random.randint(cell_left + margin, cell_right - margin),
                'y': random.randint(cell_top + margin, cell_bottom - margin),
                'vx': random.uniform(-2, 2),  # Slower to stay mostly in cell
                'vy': random.uniform(-2, 2),  # Slower to stay mostly in cell
                'size': random.randint(40, 60),  # Even larger shapes (was 30-50)
                'color': random.choice(colors),
                'cell_bounds': (cell_left, cell_top, cell_right, cell_bottom)  # Keep track of home cell
            }
            shapes.append(shape)

    # Add more extra roaming shapes for visual complexity
    num_extra = random.randint(6, 10)  # Increased from 2-4 to 6-10
    for i in range(num_extra):
        shape = {
            'type': random.choice(shape_types),
            'x': random.randint(40, width - 40),
            'y': random.randint(40, height - 40),
            'vx': random.uniform(-3, 3),
            'vy': random.uniform(-3, 3),
            'size': random.randint(45, 70),  # Even larger roaming shapes (was 35-55)
            'color': random.choice(colors),
            'cell_bounds': None  # These can roam freely
        }
        shapes.append(shape)

    # Generate frames
    for frame_idx in range(num_frames):
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        # Update and draw each shape
        for shape in shapes:
            # Update position
            shape['x'] += shape['vx']
            shape['y'] += shape['vy']

            # Bounce off walls
            if shape['x'] < shape['size'] or shape['x'] > width - shape['size']:
                shape['vx'] *= -1
            if shape['y'] < shape['size'] or shape['y'] > height - shape['size']:
                shape['vy'] *= -1

            # Keep within bounds
            shape['x'] = max(shape['size'], min(width - shape['size'], shape['x']))
            shape['y'] = max(shape['size'], min(height - shape['size'], shape['y']))

            # Draw shape
            if shape['type'] == 'circle':
                bbox = [
                    shape['x'] - shape['size'],
                    shape['y'] - shape['size'],
                    shape['x'] + shape['size'],
                    shape['y'] + shape['size']
                ]
                draw.ellipse(bbox, fill=shape['color'])
            elif shape['type'] == 'square':
                bbox = [
                    shape['x'] - shape['size'],
                    shape['y'] - shape['size'],
                    shape['x'] + shape['size'],
                    shape['y'] + shape['size']
                ]
                draw.rectangle(bbox, fill=shape['color'])
            elif shape['type'] == 'triangle':
                points = [
                    (shape['x'], shape['y'] - shape['size']),
                    (shape['x'] - shape['size'], shape['y'] + shape['size']),
                    (shape['x'] + shape['size'], shape['y'] + shape['size'])
                ]
                draw.polygon(points, fill=shape['color'])
            elif shape['type'] == 'pentagon':
                # Regular pentagon
                points = []
                import math
                for i in range(5):
                    angle = math.pi * 2 * i / 5 - math.pi / 2
                    px = shape['x'] + shape['size'] * math.cos(angle)
                    py = shape['y'] + shape['size'] * math.sin(angle)
                    points.append((px, py))
                draw.polygon(points, fill=shape['color'])
            elif shape['type'] == 'star':
                # 5-pointed star
                points = []
                import math
                for i in range(10):
                    angle = math.pi * 2 * i / 10 - math.pi / 2
                    radius = shape['size'] if i % 2 == 0 else shape['size'] * 0.4
                    px = shape['x'] + radius * math.cos(angle)
                    py = shape['y'] + radius * math.sin(angle)
                    points.append((px, py))
                draw.polygon(points, fill=shape['color'])

        frames.append(img)

    return frames


def split_frames_into_pieces(frames, grid_rows, grid_cols):
    """
    Split each frame into jigsaw pieces.

    Args:
        frames: List of PIL Images
        grid_rows: Number of rows in jigsaw grid
        grid_cols: Number of columns in jigsaw grid

    Returns:
        dict: piece_id -> list of frames for that piece
    """
    width, height = frames[0].size
    piece_width = width // grid_cols
    piece_height = height // grid_rows

    # Dictionary to hold frames for each piece
    piece_frames = {}

    for row in range(grid_rows):
        for col in range(grid_cols):
            piece_id = row * grid_cols + col
            piece_frames[piece_id] = []

            # Extract this piece from each frame
            for frame in frames:
                left = col * piece_width
                upper = row * piece_height
                right = left + piece_width
                lower = upper + piece_height

                piece = frame.crop((left, upper, right, lower))
                piece_frames[piece_id].append(piece)

    return piece_frames


def save_gif(frames, output_path, duration=100, loop=0):
    """
    Save a list of frames as an animated GIF.

    Args:
        frames: List of PIL Images
        output_path: Path to save the GIF
        duration: Duration of each frame in milliseconds
        loop: Number of loops (0 = infinite)
    """
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )


def generate_dynamic_jigsaw_dataset(output_dir, num_puzzles=10, grid_rows=3, grid_cols=3):
    """
    Generate dynamic Jigsaw CAPTCHA dataset with animated GIFs.

    Args:
        output_dir: Directory to save the dataset
        num_puzzles: Number of puzzles to generate
        grid_rows: Number of rows in jigsaw grid (default 3 for 3x3)
        grid_cols: Number of columns in jigsaw grid (default 3 for 3x3)
    """
    os.makedirs(output_dir, exist_ok=True)

    ground_truth = {}

    print(f"Generating {num_puzzles} dynamic Jigsaw puzzles...")

    for puzzle_idx in range(num_puzzles):
        # Generate animated reference with moving shapes
        print(f"Puzzle {puzzle_idx}: Generating animated GIF with moving shapes...")
        frames = create_moving_shapes_gif(
            width=450,
            height=450,
            num_frames=30,
            grid_rows=3,
            grid_cols=3
        )

        # Save reference GIF
        reference_filename = f"dynamic_jigsaw_{puzzle_idx}.gif"
        reference_path = os.path.join(output_dir, reference_filename)
        save_gif(frames, reference_path, duration=100)
        print(f"  Saved reference GIF: {reference_filename}")

        # Split frames into pieces
        piece_frames_dict = split_frames_into_pieces(frames, grid_rows, grid_cols)

        # Save each piece as animated GIF with UUID-based filenames (no position info leaked)
        piece_filenames = []
        piece_id_to_filename = {}  # Internal mapping for ground truth
        for piece_id in range(grid_rows * grid_cols):
            # Generate random UUID-based filename to prevent position leakage
            unique_hash = hashlib.md5(f"{puzzle_idx}_{piece_id}_{uuid.uuid4()}".encode()).hexdigest()[:12]
            piece_filename = f"piece_{unique_hash}.gif"
            piece_path = os.path.join(output_dir, piece_filename)
            save_gif(piece_frames_dict[piece_id], piece_path, duration=100)
            piece_filenames.append(piece_filename)
            piece_id_to_filename[piece_id] = piece_filename

        print(f"  Saved {len(piece_filenames)} animated piece GIFs (UUID filenames)")

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
        puzzle_id = f"dynamic_jigsaw_{puzzle_idx:04d}"
        ground_truth[puzzle_id] = {
            "prompt": "Drag the animated puzzle pieces to complete the jigsaw puzzle. Watch how the shapes move!",
            "description": f"Complete a {grid_rows}x{grid_cols} animated jigsaw puzzle with moving shapes",
            "grid_size": [grid_rows, grid_cols],
            "image": reference_filename,
            "pieces": shuffled_piece_filenames,  # Pieces in shuffled order
            "correct_positions": shuffled_correct_positions,  # Correct positions for shuffled pieces
            "piece_size": 450 // grid_cols,  # 150 for 3x3 grid
            "difficulty": 3 + grid_rows + grid_cols,
            "media_type": "animated_gif"
        }

    # Save ground truth
    ground_truth_path = os.path.join(output_dir, 'ground_truth.json')
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Generated {num_puzzles} dynamic Jigsaw puzzles")
    print(f"✓ Saved to {output_dir}")
    print(f"✓ Ground truth saved to {ground_truth_path}")


if __name__ == "__main__":
    output_dir = "../captcha_data/Dynamic_Jigsaw"
    generate_dynamic_jigsaw_dataset(output_dir, num_puzzles=20, grid_rows=3, grid_cols=3)
