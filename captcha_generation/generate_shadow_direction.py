#!/usr/bin/env python3
"""
Shadow Direction CAPTCHA Generator

Users identify the light source direction based on cast shadows.
- Shows objects with photorealistic cast shadows using POV-Ray
- Humans use intuitive physics: shadow opposite to light
- LLMs struggle with 3D spatial reasoning and physics simulation
- Each cell has photorealistic 3D object + shadow at different angles
- Reference shows the correct light direction
"""

import numpy as np
from PIL import Image, ImageDraw
import json
import random
from pathlib import Path
from vapory import *

# Configuration
CELL_SIZE = 240
NUM_PUZZLES = 20
GRID_SIZE = (4, 4)
TOTAL_CELLS = GRID_SIZE[0] * GRID_SIZE[1]

# Pool configuration
NUM_LIGHT_DIRECTIONS = 8  # 8 compass directions
VARIANTS_PER_DIRECTION = 12  # Different shapes/positions per direction
TOTAL_POOL_SIZE = NUM_LIGHT_DIRECTIONS * VARIANTS_PER_DIRECTION

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "captcha_data" / "Shadow_Direction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth.json"


# 8 compass directions for light source (where light comes FROM)
LIGHT_DIRECTIONS = [
    {"name": "top", "angle": 270, "dx": 0, "dy": -1},
    {"name": "top-right", "angle": 315, "dx": 1, "dy": -1},
    {"name": "right", "angle": 0, "dx": 1, "dy": 0},
    {"name": "bottom-right", "angle": 45, "dx": 1, "dy": 1},
    {"name": "bottom", "angle": 90, "dx": 0, "dy": 1},
    {"name": "bottom-left", "angle": 135, "dx": -1, "dy": 1},
    {"name": "left", "angle": 180, "dx": -1, "dy": 0},
    {"name": "top-left", "angle": 225, "dx": -1, "dy": -1},
]


def create_povray_object(shape_type, position, color, rotation=(0, 0, 0)):
    """
    Create a POV-Ray object with the specified shape, position, and color.
    Returns a Vapory object.
    """
    # Convert color from 0-255 to 0-1
    color_normalized = [c/255.0 for c in color]

    # Texture with diffuse and phong shading
    texture = Texture(
        Pigment('color', color_normalized),
        Finish('phong', 0.9, 'phong_size', 60, 'diffuse', 0.8, 'ambient', 0.2)
    )

    if shape_type == 'sphere':
        return Sphere([0, 0, 0], 0.5, texture,
                     'rotate', [rotation[0], rotation[1], rotation[2]],
                     'translate', position)
    elif shape_type == 'cube':
        return Box([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], texture,
                  'rotate', [rotation[0], rotation[1], rotation[2]],
                  'translate', position)
    elif shape_type == 'cylinder':
        return Cylinder([0, -0.6, 0], [0, 0.6, 0], 0.35, texture,
                       'rotate', [rotation[0], rotation[1], rotation[2]],
                       'translate', position)
    elif shape_type == 'cone':
        return Cone([0, -0.6, 0], 0.4, [0, 0.6, 0], 0.0, texture,
                   'rotate', [rotation[0], rotation[1], rotation[2]],
                   'translate', position)
    elif shape_type == 'torus':
        return Torus(0.4, 0.15, texture,
                    'rotate', [rotation[0], rotation[1], rotation[2]],
                    'translate', position)
    elif shape_type == 'blob':
        # Blob - organic blobby shape
        return Blob(
            'threshold', 0.6,
            Sphere([0, 0, 0], 0.6, 1.0),
            Sphere([0.3, 0.2, 0], 0.4, 1.0),
            Sphere([-0.2, -0.3, 0.1], 0.35, 1.0),
            texture,
            'rotate', [rotation[0], rotation[1], rotation[2]],
            'translate', position
        )
    elif shape_type == 'superellipsoid':
        # Superellipsoid - between sphere and cube
        return Object('superellipsoid', ['<0.25, 1.0>'],
                     texture,
                     'scale', [0.5, 0.5, 0.5],
                     'rotate', [rotation[0], rotation[1], rotation[2]],
                     'translate', position)
    elif shape_type == 'ovus':
        # Ovus - egg shape
        return Object('ovus', [0.5, 0.3],
                     texture,
                     'rotate', [rotation[0], rotation[1], rotation[2]],
                     'translate', position)
    else:
        # Default to sphere
        return Sphere([0, 0, 0], 0.5, texture,
                     'rotate', [rotation[0], rotation[1], rotation[2]],
                     'translate', position)


def render_povray_scene(shape_type, light_direction, object_color, size=CELL_SIZE):
    """
    Render a 3D scene with photorealistic shadows using POV-Ray.
    Returns a PIL Image with physically accurate lighting and shadows.
    """
    # Random rotation for variety (in degrees)
    rotation_x = random.uniform(-15, 15)
    rotation_y = random.uniform(-15, 15)
    rotation_z = random.uniform(0, 360)

    # Object position with slight random offset
    offset_x = random.uniform(-0.2, 0.2)
    offset_y = random.uniform(-0.2, 0.2)
    object_position = [offset_x, 0.5, 2 + offset_y]

    # Create 3D object
    obj = create_povray_object(shape_type, object_position, object_color,
                               rotation=(rotation_x, rotation_y, rotation_z))

    # Ground plane to receive shadows (beige color)
    ground = Plane([0, 1, 0], 0,
                   Texture(Pigment('color', [0.96, 0.96, 0.94]),
                          Finish('diffuse', 0.7, 'ambient', 0.3)))

    # Camera setup - looking at the scene
    camera = Camera('location', [0, 2, -1.5], 'look_at', [0, 0.5, 2])

    # Light source - position based on light direction
    # Light comes FROM the specified direction
    # Camera is at z=-1.5 looking toward z=2, so:
    # - Negative dy means top (closer to camera in Z)
    # - Positive dy means bottom (farther from camera in Z)
    # Longer distance and lower height = longer, more visible shadows
    light_distance = 8.0  # Increased for longer shadows
    light_x = light_direction['dx'] * light_distance
    light_y = 2.5  # Lower height = longer shadows
    light_z = 2 - light_direction['dy'] * light_distance  # Object is at z=2, INVERT dy

    light = LightSource([light_x, light_y, light_z],
                       'color', [1.0, 1.0, 1.0])

    # Background color
    background = Background('color', [0.96, 0.96, 0.94])

    # Create scene
    scene = Scene(camera,
                 objects=[light, obj, ground, background],
                 included=["colors.inc"])

    # Render to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name

    scene.render(tmp_path, width=size, height=size, antialiasing=0.3, quality=9)

    # Load and return image
    img = Image.open(tmp_path)

    # Clean up temp file
    import os
    os.unlink(tmp_path)

    return img


def generate_cell_pool():
    """Generate pool of shadow images with photorealistic rendering using POV-Ray"""
    print("Generating pool of photorealistic shadow direction images with POV-Ray...")

    cell_pool = {}
    shapes = ['sphere', 'cube', 'cylinder', 'cone', 'torus']

    # Object colors
    object_colors = [
        (220, 60, 60),   # Red
        (60, 120, 220),  # Blue
        (60, 180, 60),   # Green
        (220, 140, 60),  # Orange
        (200, 60, 200),  # Purple
        (220, 180, 60),  # Yellow
        (60, 200, 200),  # Cyan
        (180, 100, 60),  # Brown
    ]

    for dir_idx, light_dir in enumerate(LIGHT_DIRECTIONS[:NUM_LIGHT_DIRECTIONS]):
        print(f"\n  Generating variations with light from {light_dir['name']}...")

        for variant_idx in range(VARIANTS_PER_DIRECTION):
            # Random shape and color
            shape = random.choice(shapes)
            obj_color = random.choice(object_colors)

            # Render with POV-Ray
            img = render_povray_scene(shape, light_dir, obj_color)

            # Save image
            cell_id = f"shadow_{light_dir['name']}_{variant_idx}"
            img_filename = f"{cell_id}.png"
            img_path = OUTPUT_DIR / img_filename
            img.save(img_path)

            # Store metadata
            cell_pool[cell_id] = {
                "filename": img_filename,
                "light_direction": light_dir['name'],
                "light_angle": light_dir['angle'],
                "shape": shape,
                "object_color": obj_color,
            }

            if variant_idx % 4 == 3:
                print(f"    Generated {variant_idx + 1}/{VARIANTS_PER_DIRECTION} variants")

    print(f"\n✓ Generated {len(cell_pool)} photorealistic shadow images with POV-Ray")

    # Save pool metadata
    pool_file = OUTPUT_DIR / "cell_pool.json"
    with open(pool_file, 'w') as f:
        json.dump(cell_pool, f, indent=2)

    print(f"✓ Cell pool metadata saved to {pool_file}")

    return cell_pool


def create_reference_arrow(light_direction, size=200):
    """Create reference image showing light direction with arrow"""
    img = Image.new('RGB', (size, size), (245, 245, 240))
    draw = ImageDraw.Draw(img)

    center = size // 2
    arrow_length = 60

    # Arrow pointing FROM the light source TOWARD the center (where object would be)
    # If light is from bottom, arrow points UPWARD (from bottom toward center)
    # This is opposite of the angle we use for shadow calculation
    angle_rad = np.radians(light_direction['angle'] + 180)  # Flip 180 degrees
    end_x = center + int(arrow_length * np.cos(angle_rad))
    end_y = center + int(arrow_length * np.sin(angle_rad))

    # Draw arrow shaft
    draw.line([(center, center), (end_x, end_y)], fill=(255, 200, 0), width=8)

    # Draw arrowhead
    arrow_size = 15
    arrow_angle = 30
    left_angle = angle_rad + np.radians(180 - arrow_angle)
    right_angle = angle_rad + np.radians(180 + arrow_angle)

    left_x = end_x + int(arrow_size * np.cos(left_angle))
    left_y = end_y + int(arrow_size * np.sin(left_angle))
    right_x = end_x + int(arrow_size * np.cos(right_angle))
    right_y = end_y + int(arrow_size * np.sin(right_angle))

    draw.polygon([(end_x, end_y), (left_x, left_y), (right_x, right_y)],
                 fill=(255, 200, 0))

    # Draw sun symbol at arrow base
    sun_radius = 20
    draw.ellipse([center - sun_radius, center - sun_radius,
                  center + sun_radius, center + sun_radius],
                 fill=(255, 220, 50), outline=(255, 200, 0), width=3)

    # Sun rays
    for ray_angle in range(0, 360, 45):
        ray_rad = np.radians(ray_angle)
        ray_start = sun_radius + 5
        ray_end = sun_radius + 15
        x1 = center + int(ray_start * np.cos(ray_rad))
        y1 = center + int(ray_start * np.sin(ray_rad))
        x2 = center + int(ray_end * np.cos(ray_rad))
        y2 = center + int(ray_end * np.sin(ray_rad))
        draw.line([(x1, y1), (x2, y2)], fill=(255, 200, 0), width=3)

    return img


def generate_puzzles(cell_pool):
    """Generate puzzles from the cell pool"""
    print(f"\nGenerating {NUM_PUZZLES} puzzles...")

    ground_truth = {}

    for puzzle_idx in range(NUM_PUZZLES):
        # Choose a random light direction as reference
        reference_direction = random.choice(LIGHT_DIRECTIONS[:NUM_LIGHT_DIRECTIONS])

        # Get cells with this direction (correct answers)
        correct_cells = [cid for cid, data in cell_pool.items()
                        if data['light_direction'] == reference_direction['name']]

        # Get cells with other directions (distractors)
        distractor_cells = [cid for cid, data in cell_pool.items()
                           if data['light_direction'] != reference_direction['name']]

        # Select 4-6 correct cells
        num_correct = random.randint(4, 6)
        selected_correct = random.sample(correct_cells, min(num_correct, len(correct_cells)))

        # Fill rest with distractors
        num_distractors = TOTAL_CELLS - len(selected_correct)
        selected_distractors = random.sample(distractor_cells, num_distractors)

        # Combine and shuffle
        puzzle_cells = selected_correct + selected_distractors
        random.shuffle(puzzle_cells)

        # Find answer indices
        answer_indices = [i for i, cid in enumerate(puzzle_cells)
                         if cid in selected_correct]

        # Create reference arrow image
        ref_img = create_reference_arrow(reference_direction)
        ref_filename = f"reference_{reference_direction['name']}.png"
        ref_path = OUTPUT_DIR / ref_filename
        ref_img.save(ref_path)

        puzzle_id = f"shadow_direction_{puzzle_idx:04d}"
        ground_truth[puzzle_id] = {
            "prompt": f"Click all cells where the light comes from the {reference_direction['name'].upper().replace('-', ' ')} (same direction as the reference arrow).",
            "description": f"Identify cells with light from {reference_direction['name']}",
            "reference_direction": reference_direction['name'],
            "reference_image": ref_filename,
            "cells": puzzle_cells,
            "answer": answer_indices,
            "input_type": "grid_select",
            "grid_size": list(GRID_SIZE)
        }

        print(f"✓ shadow_dir_{puzzle_idx}: Reference={reference_direction['name']}, {len(answer_indices)} matches")

    # Save ground truth
    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Successfully generated {NUM_PUZZLES} Shadow Direction puzzles")
    print(f"✓ Ground truth saved to {GROUND_TRUTH_FILE}")

    return ground_truth


def main():
    print("=" * 60)
    print("Shadow Direction CAPTCHA Generator")
    print("=" * 60)

    # Generate cell pool
    cell_pool = generate_cell_pool()

    # Generate puzzles
    ground_truth = generate_puzzles(cell_pool)

    print("=" * 60)


if __name__ == "__main__":
    main()
