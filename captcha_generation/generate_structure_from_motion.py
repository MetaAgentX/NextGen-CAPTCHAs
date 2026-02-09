"""
Generate Structure-from-Motion (SFM) CAPTCHA puzzles.

Displays rotating 3D objects made of dots. Humans easily perceive the 3D shape
from motion cues, but VLMs struggle without proper 3D motion understanding.
"""

import numpy as np
import os
import json
from PIL import Image, ImageDraw
import random

# Configuration
OUTPUT_DIR = "../captcha_data/Structure_From_Motion"
NUM_PUZZLES = 20
NUM_FRAMES = 60  # 2 seconds at 30fps
CANVAS_SIZE = 400
DOT_RADIUS = 2
NUM_DOTS = 150  # Number of surface points

# 3D shapes to choose from
SHAPES = ['sphere', 'cylinder', 'cube', 'cone', 'torus']

def generate_sphere_points(n_points, radius=1.0):
    """Generate random points uniformly distributed on a sphere surface."""
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y from 1 to -1
        r = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x, y, z])

    return np.array(points)

def generate_cylinder_points(n_points, radius=1.0, height=2.0):
    """Generate points on a cylinder surface."""
    points = []
    # Half on top/bottom circles, half on curved surface
    n_caps = n_points // 4
    n_body = n_points - 2 * n_caps

    # Top cap
    for i in range(n_caps):
        angle = 2 * np.pi * i / n_caps
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        points.append([x, height/2, z])

    # Bottom cap
    for i in range(n_caps):
        angle = 2 * np.pi * i / n_caps
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        points.append([x, -height/2, z])

    # Curved surface
    for i in range(n_body):
        angle = 2 * np.pi * random.random()
        y = (random.random() - 0.5) * height
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        points.append([x, y, z])

    return np.array(points)

def generate_cube_points(n_points, size=1.5):
    """Generate points on a cube surface."""
    points = []
    for i in range(n_points):
        face = random.randint(0, 5)
        u = (random.random() - 0.5) * size
        v = (random.random() - 0.5) * size

        if face == 0:  # Front
            points.append([u, v, size/2])
        elif face == 1:  # Back
            points.append([u, v, -size/2])
        elif face == 2:  # Left
            points.append([-size/2, u, v])
        elif face == 3:  # Right
            points.append([size/2, u, v])
        elif face == 4:  # Top
            points.append([u, size/2, v])
        else:  # Bottom
            points.append([u, -size/2, v])

    return np.array(points)

def generate_cone_points(n_points, radius=1.0, height=2.0):
    """Generate points on a cone surface."""
    points = []
    # Base circle
    n_base = n_points // 4
    for i in range(n_base):
        angle = 2 * np.pi * i / n_base
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        points.append([x, -height/2, z])

    # Cone surface
    n_surface = n_points - n_base
    for i in range(n_surface):
        angle = 2 * np.pi * random.random()
        t = random.random()  # Parameter along cone
        y = -height/2 + height * t
        r = radius * (1 - t)  # Radius decreases linearly
        x = r * np.cos(angle)
        z = r * np.sin(angle)
        points.append([x, y, z])

    return np.array(points)

def generate_torus_points(n_points, major_radius=1.0, minor_radius=0.4):
    """Generate points on a torus surface."""
    points = []
    for i in range(n_points):
        u = 2 * np.pi * random.random()  # Angle around tube
        v = 2 * np.pi * random.random()  # Angle around major circle

        x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
        y = minor_radius * np.sin(v)
        z = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
        points.append([x, y, z])

    return np.array(points)

def rotation_matrix(axis, angle):
    """Create a rotation matrix for given axis and angle."""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)

    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
    ])

def project_3d_to_2d(points_3d, scale=100):
    """Orthographic projection of 3D points to 2D."""
    # Simple orthographic projection (ignoring z for projection)
    points_2d = points_3d[:, :2] * scale
    # Center on canvas
    points_2d[:, 0] += CANVAS_SIZE / 2
    points_2d[:, 1] += CANVAS_SIZE / 2
    return points_2d

def generate_shape_points(shape_name):
    """Generate 3D points for a given shape."""
    if shape_name == 'sphere':
        return generate_sphere_points(NUM_DOTS)
    elif shape_name == 'cylinder':
        return generate_cylinder_points(NUM_DOTS)
    elif shape_name == 'cube':
        return generate_cube_points(NUM_DOTS)
    elif shape_name == 'cone':
        return generate_cone_points(NUM_DOTS)
    elif shape_name == 'torus':
        return generate_torus_points(NUM_DOTS)
    else:
        raise ValueError(f"Unknown shape: {shape_name}")

def generate_rotation_animation(shape_name):
    """Generate a rotating animation of a 3D shape."""
    # Generate base shape
    points_3d = generate_shape_points(shape_name)

    # Random rotation axis (slightly off from pure axes for interesting motion)
    axis = np.array([
        random.uniform(-1, 1),
        random.uniform(0.5, 1.5),  # Bias toward y-axis
        random.uniform(-1, 1)
    ])
    axis = axis / np.linalg.norm(axis)

    # Rotation speed (full rotation in 60 frames)
    angle_per_frame = 2 * np.pi / NUM_FRAMES

    frames = []
    for frame_idx in range(NUM_FRAMES):
        # Create blank frame
        img = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), color='black')
        draw = ImageDraw.Draw(img)

        # Rotate points
        angle = angle_per_frame * frame_idx
        rot_matrix = rotation_matrix(axis, angle)
        rotated_points = points_3d @ rot_matrix.T

        # Project to 2D
        points_2d = project_3d_to_2d(rotated_points)

        # Draw dots (use z-depth for slight size variation)
        z_values = rotated_points[:, 2]
        for (x, y), z in zip(points_2d, z_values):
            if 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE:
                # Slightly vary dot size based on depth
                size = DOT_RADIUS + int(z * 0.5)
                size = max(1, min(size, 4))
                draw.ellipse([x-size, y-size, x+size, y+size], fill='white')

        frames.append(img)

    return frames

def generate_puzzle(puzzle_idx):
    """Generate a single SFM puzzle with 4x4 grid."""
    # Choose one correct shape
    correct_shape = random.choice(SHAPES)

    # Create puzzle directory
    puzzle_name = f"puzzle_{puzzle_idx:03d}"
    puzzle_dir = os.path.join(OUTPUT_DIR, puzzle_name)
    os.makedirs(puzzle_dir, exist_ok=True)

    # Generate 16 cells (15 wrong + 1 correct)
    cell_files = []
    correct_cells = []

    # Determine which cell(s) will be correct (1-2 cells)
    num_correct = random.randint(1, 2)
    correct_indices = random.sample(range(16), num_correct)

    for cell_idx in range(16):
        if cell_idx in correct_indices:
            # Correct shape
            shape = correct_shape
            correct_cells.append(cell_idx)
        else:
            # Wrong shape (pick from other shapes)
            wrong_shapes = [s for s in SHAPES if s != correct_shape]
            shape = random.choice(wrong_shapes)

        # Generate animation for this cell
        frames = generate_rotation_animation(shape)

        # Save GIF
        cell_file = f"cell_{cell_idx:02d}.gif"
        cell_path = os.path.join(puzzle_dir, cell_file)

        frames[0].save(
            cell_path,
            save_all=True,
            append_images=frames[1:],
            duration=33,  # ~30fps
            loop=0
        )

        cell_files.append(cell_file)

    # Format prompt with shape name
    shape_labels = {
        'sphere': 'sphere (ball)',
        'cylinder': 'cylinder (tube)',
        'cube': 'cube (box)',
        'cone': 'cone',
        'torus': 'torus (donut)'
    }

    prompt = f"Select all cells showing a {shape_labels[correct_shape]}"

    return {
        "puzzle_id": puzzle_name,
        "puzzle_dir": puzzle_name,
        "cell_files": cell_files,
        "correct_shape": correct_shape,
        "answer": sorted(correct_cells),
        "grid_size": [4, 4],
        "prompt": prompt
    }

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate puzzles
    ground_truth = {}

    print(f"Generating {NUM_PUZZLES} Structure-from-Motion puzzles...")
    for i in range(NUM_PUZZLES):
        print(f"  Generating puzzle {i+1}/{NUM_PUZZLES}...")
        puzzle_data = generate_puzzle(i)
        ground_truth[puzzle_data["puzzle_id"]] = puzzle_data

    # Save ground truth
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth.json")
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGenerated {NUM_PUZZLES} puzzles in {OUTPUT_DIR}")
    print(f"Ground truth saved to {gt_path}")

if __name__ == "__main__":
    main()
