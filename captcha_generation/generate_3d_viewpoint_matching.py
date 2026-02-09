#!/usr/bin/env python3
"""
3D Viewpoint Matching CAPTCHA Generator

Creates wireframe pyramid puzzles with colored edges where users must
identify which cell options show valid viewpoints of the reference pyramid.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image
import json
import os
import random
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from copy import deepcopy
import io


# Vibrant colors for edges
EDGE_COLORS = [
    '#FF0000',  # Red
    '#00CC00',  # Green
    '#0066FF',  # Blue
    '#FF8C00',  # Orange
    '#FF00FF',  # Magenta
    '#00CCCC',  # Cyan
    '#9400D3',  # Purple
    '#FFD700',  # Gold
    '#FF69B4',  # Hot Pink
    '#00FF7F',  # Spring Green
    '#DC143C',  # Crimson
    '#1E90FF',  # Dodger Blue
]


@dataclass
class Edge:
    """Represents a 3D edge with start/end points and color."""
    start: np.ndarray
    end: np.ndarray
    color: str
    edge_type: str = 'lateral'  # 'lateral', 'base', 'top'


@dataclass
class Shape3D:
    """Represents a 3D shape with edges."""
    edges: List[Edge]
    n_sides: int
    name: str


def get_pyramid_geometry(n_sides: int, radius: float = 1.0, height: float = 1.5) -> Shape3D:
    """Creates an N-sided pyramid with unique colored edges."""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    angles += np.pi / n_sides

    base_x = radius * np.cos(angles)
    base_y = radius * np.sin(angles)
    base_z = np.zeros_like(base_x)

    apex = np.array([0, 0, height])

    total_edges = n_sides * 2
    colors = random.sample(EDGE_COLORS, min(total_edges, len(EDGE_COLORS)))
    while len(colors) < total_edges:
        colors.extend(random.sample(EDGE_COLORS, min(total_edges - len(colors), len(EDGE_COLORS))))

    edges = []
    color_idx = 0

    # Lateral edges (apex to base)
    for i in range(n_sides):
        base_point = np.array([base_x[i], base_y[i], base_z[i]])
        edges.append(Edge(
            start=apex.copy(),
            end=base_point,
            color=colors[color_idx],
            edge_type='lateral'
        ))
        color_idx += 1

    # Base edges
    for i in range(n_sides):
        start = np.array([base_x[i], base_y[i], base_z[i]])
        next_idx = (i + 1) % n_sides
        end = np.array([base_x[next_idx], base_y[next_idx], base_z[next_idx]])
        edges.append(Edge(
            start=start,
            end=end,
            color=colors[color_idx],
            edge_type='base'
        ))
        color_idx += 1

    names = {3: 'tetrahedron', 4: 'square_pyramid', 5: 'pentagonal_pyramid', 6: 'hexagonal_pyramid'}
    return Shape3D(edges=edges, n_sides=n_sides, name=names.get(n_sides, f'{n_sides}-sided'))


def get_frustum_geometry(n_sides: int = 4, bottom_radius: float = 1.0,
                          top_radius: float = 0.5, height: float = 1.2) -> Shape3D:
    """Creates a frustum (truncated pyramid) with unique colored edges."""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    angles += np.pi / n_sides

    # Bottom base
    bottom_x = bottom_radius * np.cos(angles)
    bottom_y = bottom_radius * np.sin(angles)
    bottom_z = np.zeros_like(bottom_x)

    # Top base
    top_x = top_radius * np.cos(angles)
    top_y = top_radius * np.sin(angles)
    top_z = np.full_like(top_x, height)

    total_edges = n_sides * 3  # lateral + bottom + top
    colors = random.sample(EDGE_COLORS, min(total_edges, len(EDGE_COLORS)))
    while len(colors) < total_edges:
        colors.extend(random.sample(EDGE_COLORS, min(total_edges - len(colors), len(EDGE_COLORS))))

    edges = []
    color_idx = 0

    # Lateral edges (connecting bottom to top)
    for i in range(n_sides):
        bottom_point = np.array([bottom_x[i], bottom_y[i], bottom_z[i]])
        top_point = np.array([top_x[i], top_y[i], top_z[i]])
        edges.append(Edge(
            start=bottom_point,
            end=top_point,
            color=colors[color_idx],
            edge_type='lateral'
        ))
        color_idx += 1

    # Bottom base edges
    for i in range(n_sides):
        start = np.array([bottom_x[i], bottom_y[i], bottom_z[i]])
        next_idx = (i + 1) % n_sides
        end = np.array([bottom_x[next_idx], bottom_y[next_idx], bottom_z[next_idx]])
        edges.append(Edge(
            start=start,
            end=end,
            color=colors[color_idx],
            edge_type='base'
        ))
        color_idx += 1

    # Top base edges
    for i in range(n_sides):
        start = np.array([top_x[i], top_y[i], top_z[i]])
        next_idx = (i + 1) % n_sides
        end = np.array([top_x[next_idx], top_y[next_idx], top_z[next_idx]])
        edges.append(Edge(
            start=start,
            end=end,
            color=colors[color_idx],
            edge_type='top'
        ))
        color_idx += 1

    return Shape3D(edges=edges, n_sides=n_sides, name='frustum')


def center_shape_vertically(shape: Shape3D) -> Shape3D:
    """Center a shape vertically so its center is at z=0."""
    # Find min and max z coordinates
    all_z = []
    for edge in shape.edges:
        all_z.extend([edge.start[2], edge.end[2]])
    min_z = min(all_z)
    max_z = max(all_z)
    center_z = (min_z + max_z) / 2

    # Create new shape with shifted z coordinates
    new_edges = []
    for edge in shape.edges:
        new_start = edge.start.copy()
        new_end = edge.end.copy()
        new_start[2] -= center_z
        new_end[2] -= center_z
        new_edges.append(Edge(
            start=new_start,
            end=new_end,
            color=edge.color,
            edge_type=edge.edge_type
        ))

    return Shape3D(edges=new_edges, n_sides=shape.n_sides, name=shape.name)


def shape_to_json(shape: Shape3D) -> Dict[str, Any]:
    """Convert Shape3D to JSON-serializable format for Three.js rendering."""
    edges_data = []
    for edge in shape.edges:
        edges_data.append({
            'start': edge.start.tolist(),  # [x, y, z]
            'end': edge.end.tolist(),      # [x, y, z]
            'color': edge.color,           # hex string like '#FF0000'
            'edge_type': edge.edge_type    # 'lateral', 'base', 'top'
        })
    return {
        'edges': edges_data,
        'n_sides': shape.n_sides,
        'name': shape.name
    }


def render_shape_3d(
    shape: Shape3D,
    elev: float = 20,
    azim: float = 10,
    figsize: Tuple[float, float] = (4, 4),
    linewidth: float = 3
) -> Image.Image:
    """Render shape as 3D wireframe using matplotlib."""
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    for edge in shape.edges:
        ax.plot(
            [edge.start[0], edge.end[0]],
            [edge.start[1], edge.end[1]],
            [edge.start[2], edge.end[2]],
            color=edge.color,
            linewidth=linewidth,
            solid_capstyle='round'
        )

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1, 1])

    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf).convert('RGB')


def render_shape_topdown(
    shape: Shape3D,
    rotation: float = 0,
    figsize: Tuple[float, float] = (4, 4),
    linewidth: float = 3
) -> Image.Image:
    """Render shape from top-down view (2D projection)."""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    theta = np.radians(rotation)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    for edge in shape.edges:
        start_2d = rot_matrix @ edge.start[:2]
        end_2d = rot_matrix @ edge.end[:2]

        ax.plot(
            [start_2d[0], end_2d[0]],
            [start_2d[1], end_2d[1]],
            color=edge.color,
            linewidth=linewidth,
            solid_capstyle='round'
        )

    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim([-1.8, 1.8])
    ax.set_ylim([-1.8, 1.8])

    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf).convert('RGB')


def create_distractor(shape: Shape3D) -> Shape3D:
    """Create a distractor by swapping some edge colors."""
    distractor = deepcopy(shape)

    method = random.choice(['swap_pair', 'swap_multiple', 'rotate_lateral', 'rotate_base'])

    if method == 'swap_pair':
        if len(distractor.edges) >= 2:
            idx1, idx2 = random.sample(range(len(distractor.edges)), 2)
            distractor.edges[idx1].color, distractor.edges[idx2].color = \
                distractor.edges[idx2].color, distractor.edges[idx1].color

    elif method == 'swap_multiple':
        num_swaps = random.randint(2, min(3, len(distractor.edges) // 2))
        indices = list(range(len(distractor.edges)))
        random.shuffle(indices)
        for i in range(0, num_swaps * 2, 2):
            if i + 1 < len(indices):
                idx1, idx2 = indices[i], indices[i + 1]
                distractor.edges[idx1].color, distractor.edges[idx2].color = \
                    distractor.edges[idx2].color, distractor.edges[idx1].color

    elif method == 'rotate_lateral':
        lateral = [e for e in distractor.edges if e.edge_type == 'lateral']
        if len(lateral) >= 2:
            colors = [e.color for e in lateral]
            colors = colors[1:] + colors[:1]
            lat_idx = 0
            for e in distractor.edges:
                if e.edge_type == 'lateral':
                    e.color = colors[lat_idx]
                    lat_idx += 1

    elif method == 'rotate_base':
        base = [e for e in distractor.edges if e.edge_type == 'base']
        if len(base) >= 2:
            colors = [e.color for e in base]
            colors = colors[1:] + colors[:1]
            base_idx = 0
            for e in distractor.edges:
                if e.edge_type == 'base':
                    e.color = colors[base_idx]
                    base_idx += 1

    return distractor


def generate_batch(
    output_dir: str,
    num_captchas: int = 20,
    seed: int = None
) -> Dict[str, Any]:
    """
    Generate a batch of CAPTCHAs in the expected format.

    Output structure:
    - main_XXX.png: Reference images
    - cell_XXX_YY.png: Cell images (9 per captcha)
    - ground_truth.json: Answers in expected format
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Shape creators (removed tetrahedron and hexagon - too easy with interactive 3D viewer)
    shape_creators = {
        'square_pyramid': lambda: get_pyramid_geometry(4),
        'pentagonal_pyramid': lambda: get_pyramid_geometry(5),
        'frustum': lambda: get_frustum_geometry(4),
    }

    shape_types = list(shape_creators.keys())
    ground_truth = {}
    cell_pool = {}

    for puzzle_idx in range(num_captchas):
        # Pick random shape type
        shape_type = random.choice(shape_types)
        shape = shape_creators[shape_type]()
        shape = center_shape_vertically(shape)  # Center around z=0 for alignment with Three.js

        # Reference view - correct camera angle
        ref_elev = 20
        ref_azim = -35  # Matches reference image perspective
        ref_img = render_shape_3d(shape, elev=ref_elev, azim=ref_azim, figsize=(4, 4), linewidth=6)
        ref_img.save(os.path.join(output_dir, f'main_{puzzle_idx:03d}.png'))

        # Generate cells
        num_correct = random.randint(3, 4)
        num_distractors = 9 - num_correct

        n_sides = shape.n_sides

        # Symmetric azimuths: shape looks identical at these angles
        symmetric_azimuths = [ref_azim + i * (360 / n_sides) for i in range(n_sides)]
        symmetric_rotations = [i * (360 / n_sides) for i in range(n_sides)]

        # Split correct views between side and top-down
        num_side_correct = (num_correct + 1) // 2
        num_topdown_correct = num_correct - num_side_correct

        # CORRECT: symmetric rotations, SAME colors (no swap)
        correct_views = []
        for i in range(num_side_correct):
            azim = symmetric_azimuths[i % len(symmetric_azimuths)]
            correct_views.append({
                'type': 'side',
                'azim': azim,
                'shape': shape  # Same colors
            })
        for i in range(num_topdown_correct):
            rot = symmetric_rotations[i % len(symmetric_rotations)]
            correct_views.append({
                'type': 'topdown',
                'rotation': rot,
                'shape': shape  # Same colors
            })

        # Split distractor views between side and top-down
        num_side_distractor = (num_distractors + 1) // 2
        num_topdown_distractor = num_distractors - num_side_distractor

        # DISTRACTOR: symmetric rotations, SWAPPED colors
        distractor_views = []
        for i in range(num_side_distractor):
            distractor = create_distractor(shape)  # Swap colors
            azim = symmetric_azimuths[i % len(symmetric_azimuths)]
            distractor_views.append({
                'type': 'side',
                'azim': azim,
                'shape': distractor
            })
        for i in range(num_topdown_distractor):
            distractor = create_distractor(shape)  # Swap colors
            rot = symmetric_rotations[i % len(symmetric_rotations)]
            distractor_views.append({
                'type': 'topdown',
                'rotation': rot,
                'shape': distractor
            })

        # Combine and shuffle
        all_views = [(v, True) for v in correct_views] + [(v, False) for v in distractor_views]
        random.shuffle(all_views)

        # Render and save cells
        correct_indices = []
        cells = []

        for cell_idx, (view, is_correct) in enumerate(all_views):
            if view['type'] == 'side':
                img = render_shape_3d(
                    view['shape'],
                    elev=ref_elev,
                    azim=view['azim'],
                    figsize=(3, 3),
                    linewidth=4
                )
            else:  # topdown
                img = render_shape_topdown(
                    view['shape'],
                    rotation=view['rotation'],
                    figsize=(3, 3),
                    linewidth=4
                )

            cell_filename = f'cell_{puzzle_idx:03d}_{cell_idx:02d}.png'
            img.save(os.path.join(output_dir, cell_filename))
            cells.append(cell_filename)

            if is_correct:
                correct_indices.append(cell_idx)

            # Cell pool entry
            cell_pool[cell_filename] = {
                'puzzle_id': puzzle_idx,
                'cell_idx': cell_idx,
                'is_correct': is_correct,
                'view_type': view['type'],
                'shape': shape_type
            }

        # Ground truth entry
        ground_truth[f'3d_viewpoint_{puzzle_idx:04d}'] = {
            'prompt': 'Look at the 3D wireframe above with colored edges. Select ALL cells that show the SAME object from different angles with the SAME edge colors.',
            'main_image': f'captcha_data/3D_Viewpoint/main_{puzzle_idx:03d}.png',
            'cells': cells,
            'answer': correct_indices,
            'input_type': 'viewpoint_select',
            'grid_size': [3, 3],
            'shape': shape_type,
            'shape_data': shape_to_json(shape)  # 3D geometry for interactive Three.js viewer
        }

        print(f"Generated puzzle {puzzle_idx + 1}/{num_captchas}: {shape_type}, answers: {correct_indices}")

    # Save ground truth
    with open(os.path.join(output_dir, 'ground_truth.json'), 'w') as f:
        json.dump(ground_truth, f, indent=2)

    # Save cell pool
    with open(os.path.join(output_dir, 'cell_pool.json'), 'w') as f:
        json.dump(cell_pool, f, indent=2)

    return ground_truth


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate 3D Viewpoint Matching CAPTCHAs')
    parser.add_argument('--output', '-o', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'captcha_data', '3D_Viewpoint'),
                        help='Output directory')
    parser.add_argument('--num', '-n', type=int, default=20,
                        help='Number of CAPTCHAs to generate')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed')

    args = parser.parse_args()

    result = generate_batch(args.output, args.num, args.seed)
    print(f"\nGenerated {len(result)} CAPTCHAs in {args.output}")
