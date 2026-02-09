#!/usr/bin/env python3
"""
Box Folding CAPTCHA Generator

Generates a CAPTCHA where users must identify which 3D cube could result from
folding a given 2D net (unfolded cube template).

This exploits a known LLM weakness: they cannot mentally simulate 3D spatial
transformations like folding a flat template into a cube.

Task:
- Reference: Shows an unfolded cube net with emojis on each of 6 faces
- Options: 3x3 grid of 3D rendered cubes (showing 3 visible faces)
- User selects which cube(s) could result from folding the net

Pool-based architecture:
- Generate cube nets with various emoji combinations
- Render 3D cubes for each net (correct and incorrect foldings)
- Compose puzzles from the pool
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import random
from pathlib import Path
import math
import itertools

# Configuration
NET_SIZE = 600  # Size of the unfolded net image
CUBE_SIZE = 400  # Size of each 3D cube option image
CELL_SIZE = 140  # Size of each face in the net
NUM_PUZZLES = 30
GRID_SIZE = (3, 3)  # 3x3 grid of options
TOTAL_CELLS = GRID_SIZE[0] * GRID_SIZE[1]

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "captcha_data" / "Box_Folding"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Animal emoji pool - distinct, easily recognizable animals
# Note: Avoid ZWJ sequence emojis like 🐻‍❄️ (polar bear) as they may render as separate characters
EMOJI_POOL = [
    '🐮', '🐻', '🐯', '🐶', '🦊', '🦋',
    '🐱', '🐰', '🐸', '🐵', '🐔', '🦉',
    '🐨', '🦁', '🐧', '🐢', '🦄',
    '🐭', '🐹', '🐼', '🐲', '🦅', '🐴',
    '🐙'
]

def generate_symbol_sets():
    """Generate sets of 6 unique animal emojis for cube faces."""
    sets = []
    all_emojis = EMOJI_POOL.copy()

    # Create 10 different emoji sets by taking groups of 6
    # 10 sets × 3 net patterns = 30 nets = 30 puzzles
    for i in range(10):
        start_idx = (i * 6) % len(all_emojis)
        emoji_set = []
        for j in range(6):
            idx = (start_idx + j) % len(all_emojis)
            emoji_set.append(all_emojis[idx])
        sets.append(emoji_set)

    return sets

SYMBOL_SETS = None  # Will be generated at runtime

# Cube face indices and their spatial relationships
# Face naming: 0=Front, 1=Back, 2=Top, 3=Bottom, 4=Left, 5=Right
OPPOSITE_FACES = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

# Standard cube net (cross pattern)
#     [2]
# [4][0][5][1]
#     [3]
CUBE_NET_CROSS = {
    (1, 0): 2,  # Top
    (0, 1): 4,  # Left
    (1, 1): 0,  # Front
    (2, 1): 5,  # Right
    (3, 1): 1,  # Back
    (1, 2): 3,  # Bottom
}

# T-shape net
#     [2]
#     [0]
# [4][3][5]
#     [1]
CUBE_NET_T = {
    (1, 0): 2,
    (1, 1): 0,
    (0, 2): 4,
    (1, 2): 3,
    (2, 2): 5,
    (1, 3): 1,
}

# L-shape net (corrected face assignments based on physical folding)
# [4][0][5]    <- Left, Front, Right (horizontal strip at top)
#    [3]       <- Bottom (folds backward from Front)
#    [1]       <- Back (continues from Bottom)
#    [2]       <- Top (wraps around from Back)
CUBE_NET_L = {
    (0, 0): 4,  # Left (folds left from Front)
    (1, 0): 0,  # Front (reference)
    (2, 0): 5,  # Right (folds right from Front)
    (1, 1): 3,  # Bottom (folds down from Front)
    (1, 2): 1,  # Back (folds from Bottom)
    (1, 3): 2,  # Top (wraps around from Back)
}

ALL_NETS = [CUBE_NET_CROSS, CUBE_NET_T, CUBE_NET_L]


def calculate_face_rotations(net_pattern):
    """
    Calculate the emoji rotation needed for each visible face based on the net's folding geometry.

    Uses 3D mathematical modeling of the folding process:
    1. Find the path from each face to Face 0 (reference) in the net grid
    2. Calculate how each fold transforms the local "up" direction
    3. Determine the rotation needed to match physical folding

    Args:
        net_pattern: Dict mapping (grid_x, grid_y) -> face_index

    Returns:
        Dict mapping face_index -> rotation_degrees (CCW)
    """
    # Build inverse mapping: face_index -> grid_position
    face_to_pos = {face_idx: pos for pos, face_idx in net_pattern.items()}

    # Build adjacency graph from the net
    def get_neighbors(pos):
        """Get adjacent grid positions that have faces."""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # up, down, left, right
            neighbor = (x + dx, y + dy)
            if neighbor in net_pattern:
                neighbors.append((neighbor, (dx, dy)))
        return neighbors

    def find_path_to_face0(start_face):
        """
        BFS to find the path from start_face to Face 0.
        Returns list of (direction_dx, direction_dy) steps.
        """
        if start_face == 0:
            return []

        start_pos = face_to_pos[start_face]
        face0_pos = face_to_pos[0]

        # BFS
        from collections import deque
        queue = deque([(start_pos, [])])
        visited = {start_pos}

        while queue:
            current_pos, path = queue.popleft()

            for neighbor_pos, direction in get_neighbors(current_pos):
                if neighbor_pos in visited:
                    continue

                new_path = path + [direction]

                if neighbor_pos == face0_pos:
                    return new_path

                visited.add(neighbor_pos)
                queue.append((neighbor_pos, new_path))

        return []  # No path found (shouldn't happen for valid nets)

    def calculate_rotation_from_path(face_idx, path):
        """
        Calculate the rotation for a face based on its folding path.

        Mathematical model for 3D cube folding:
        - Each face in the net has an initial "up" direction (pointing up in image)
        - When folding, each 90° fold transforms the local coordinate system
        - The final rotation is the difference between where "up" ends up
          and where the perspective warp expects it to be

        Face roles and expected orientations:
        - Face 0 (Front): Reference face, "up" stays up
        - Face 1 (Back): Opposite to front, "up" points up when viewed from back
        - Face 2 (Top): "up" in net → back edge when viewed from above-front
        - Face 3 (Bottom): "up" in net → front edge when viewed from below-front
        - Face 4 (Left): "up" in net → top edge when viewed from left
        - Face 5 (Right): "up" in net → top edge when viewed from right
        """
        if not path:
            return 0

        rotation = 0

        for step_dx, step_dy in path:
            # step direction is FROM current face TOWARD Face 0
            # positive y is DOWN in the grid/image

            if face_idx == 2:  # Top face
                # Expected: emoji top → back edge of top face
                if step_dy == 1:  # Face is ABOVE next (folds backward)
                    rotation += 0  # Top already points to back
                elif step_dy == -1:  # Face is BELOW next (folds forward)
                    rotation += 180  # Top would point to front, need 180°
                elif step_dx == 1:  # Face is LEFT of next (folds right)
                    rotation += -90  # Top → right edge, need 90° CW
                elif step_dx == -1:  # Face is RIGHT of next (folds left)
                    rotation += 90  # Top → left edge, need 90° CCW

            elif face_idx == 3:  # Bottom face
                # Expected: emoji top → front edge of bottom face (opposite of top)
                if step_dy == 1:  # Face is ABOVE next (folds backward)
                    rotation += 180  # Top would point to back, need 180°
                elif step_dy == -1:  # Face is BELOW next (folds forward)
                    rotation += 0  # Top already points to front
                elif step_dx == 1:  # Face is LEFT of next
                    rotation += 90  # Top → right edge, need 90° CCW
                elif step_dx == -1:  # Face is RIGHT of next
                    rotation += -90  # Top → left edge, need 90° CW

            elif face_idx == 5:  # Right face
                # Expected: emoji top → top edge of right face
                if step_dy == 1:  # Moving down (through intermediate face)
                    rotation += -90  # Indirect fold rotates 90° CW
                elif step_dy == -1:  # Moving up
                    rotation += 90  # Indirect fold rotates 90° CCW
                elif step_dx == 1:  # Direct right of next (folds right)
                    rotation += 0  # Emoji stays upright
                elif step_dx == -1:  # Direct left of next
                    rotation += 0

            elif face_idx == 4:  # Left face (mirror of right face)
                # Expected: emoji top → top edge of left face
                if step_dy == 1:  # Moving down (through intermediate face)
                    rotation += 90  # Opposite of right face
                elif step_dy == -1:  # Moving up
                    rotation += -90  # Opposite of right face
                elif step_dx == 1:  # Direct right of next
                    rotation += 0
                elif step_dx == -1:  # Direct left of next (folds left)
                    rotation += 0

            elif face_idx == 1:  # Back face (opposite of front)
                # Expected: emoji top → top edge when viewed from back
                # Back face folds around to opposite side
                if step_dy == 1:  # Moving down
                    rotation += 180  # Flip upside down through vertical fold
                elif step_dy == -1:  # Moving up
                    rotation += 180
                elif step_dx == 1:  # Moving right
                    rotation += 0  # Chain fold maintains orientation
                elif step_dx == -1:  # Moving left
                    rotation += 0

        # Normalize to [-180, 180]
        rotation = ((rotation + 180) % 360) - 180

        return rotation

    # Calculate rotations for ALL faces (needed for diverse viewing angles)
    rotations = {0: 0}  # Face 0 is always reference, no rotation

    for face_idx in [1, 2, 3, 4, 5]:  # All faces except Face 0
        if face_idx in face_to_pos:
            path = find_path_to_face0(face_idx)
            rotations[face_idx] = calculate_rotation_from_path(face_idx, path)

    return rotations


def get_emoji_font(size=64):
    """
    Get a font that can render emojis.
    Apple Color Emoji only supports specific sizes: 20, 32, 40, 48, 64, 96, 160
    """
    # Valid Apple Color Emoji sizes
    valid_sizes = [20, 32, 40, 48, 64, 96, 160]
    # Find the closest valid size
    closest_size = min(valid_sizes, key=lambda x: abs(x - size))

    font_paths = [
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/Library/Fonts/Apple Color Emoji.ttc",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/noto-emoji/NotoColorEmoji.ttf",
        "C:/Windows/Fonts/seguiemj.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, closest_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_emoji(img, emoji, x, y, size):
    """
    Draw an emoji at the specified position using Apple Color Emoji font.
    """
    # Get emoji font with valid size
    font = get_emoji_font(size)

    # Create a temporary RGBA image for the emoji
    temp_size = int(size * 2)
    temp_img = Image.new('RGBA', (temp_size, temp_size), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp_img)

    # Draw emoji with embedded_color=True for color emoji support
    try:
        bbox = temp_draw.textbbox((0, 0), emoji, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (temp_size - text_width) // 2 - bbox[0]
        text_y = (temp_size - text_height) // 2 - bbox[1]
        temp_draw.text((text_x, text_y), emoji, font=font, embedded_color=True)
    except Exception:
        temp_draw.text((temp_size//4, temp_size//4), emoji, font=font, embedded_color=True)

    # Paste onto main image centered at (x, y)
    paste_x = int(x - temp_size // 2)
    paste_y = int(y - temp_size // 2)

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    img.paste(temp_img, (paste_x, paste_y), temp_img)
    return img


def draw_shape(draw, shape, color, x, y, size, fill=True):
    """
    Draw a shape at the given position.

    Args:
        draw: PIL ImageDraw object
        shape: Shape type ('circle', 'square', 'triangle', 'diamond', 'star', 'cross')
        color: Color hex string
        x, y: Center position
        size: Size of the shape
    """
    half = size // 2

    if shape == 'circle':
        draw.ellipse([x - half, y - half, x + half, y + half],
                    fill=color if fill else None, outline=color, width=3)

    elif shape == 'square':
        s = int(half * 0.85)
        draw.rectangle([x - s, y - s, x + s, y + s],
                      fill=color if fill else None, outline=color, width=3)

    elif shape == 'triangle':
        h = int(half * 0.9)
        points = [
            (x, y - h),  # Top
            (x - h, y + int(h * 0.7)),  # Bottom left
            (x + h, y + int(h * 0.7)),  # Bottom right
        ]
        draw.polygon(points, fill=color if fill else None, outline=color, width=3)

    elif shape == 'diamond':
        h = int(half * 0.9)
        points = [
            (x, y - h),  # Top
            (x + h, y),  # Right
            (x, y + h),  # Bottom
            (x - h, y),  # Left
        ]
        draw.polygon(points, fill=color if fill else None, outline=color, width=3)

    elif shape == 'star':
        # 5-pointed star
        outer_r = int(half * 0.9)
        inner_r = int(half * 0.4)
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = outer_r if i % 2 == 0 else inner_r
            points.append((x + r * math.cos(angle), y - r * math.sin(angle)))
        draw.polygon(points, fill=color if fill else None, outline=color, width=3)

    elif shape == 'cross':
        w = int(half * 0.3)
        h = int(half * 0.85)
        # Vertical bar
        draw.rectangle([x - w, y - h, x + w, y + h], fill=color)
        # Horizontal bar
        draw.rectangle([x - h, y - w, x + h, y + w], fill=color)

    elif shape == 'heart':
        # Heart shape using bezier-like points
        h = int(half * 0.9)
        points = []
        for i in range(30):
            t = i / 30 * 2 * math.pi
            hx = 16 * (math.sin(t) ** 3)
            hy = 13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)
            points.append((x + hx * h / 18, y - hy * h / 18))
        draw.polygon(points, fill=color if fill else None, outline=color, width=2)

    elif shape == 'hexagon':
        h = int(half * 0.9)
        points = []
        for i in range(6):
            angle = math.pi / 6 + i * math.pi / 3  # Start flat side up
            points.append((x + h * math.cos(angle), y - h * math.sin(angle)))
        draw.polygon(points, fill=color if fill else None, outline=color, width=3)


def render_emoji_image(emoji, size):
    """
    Render an emoji as a standalone RGBA image with transparent background.
    Same rendering style as the unfolded net for consistency.

    Args:
        emoji: Emoji character string
        size: Size of the output image

    Returns:
        PIL Image in RGBA mode
    """
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Use same font size logic as draw_emoji for consistency
    font = get_emoji_font(int(size * 0.8))

    try:
        bbox = draw.textbbox((0, 0), emoji, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (size - text_width) // 2 - bbox[0]
        text_y = (size - text_height) // 2 - bbox[1]
        draw.text((text_x, text_y), emoji, font=font, embedded_color=True)
    except Exception:
        draw.text((size//4, size//4), emoji, font=font, embedded_color=True)

    return img


def find_perspective_coeffs(source_coords, target_coords):
    """
    Calculate the perspective transform coefficients to map source quad to target quad.

    Based on: https://stackoverflow.com/questions/14177744/

    Args:
        source_coords: List of 4 (x, y) tuples - corners of source rectangle
        target_coords: List of 4 (x, y) tuples - corners of target quadrilateral

    Returns:
        8-tuple of coefficients for PIL's PERSPECTIVE transform
    """
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])

    A = np.matrix(matrix, dtype=np.float64)
    B = np.array(source_coords).reshape(8)

    try:
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)
    except np.linalg.LinAlgError:
        # Fallback if matrix is singular
        return (1, 0, 0, 0, 1, 0, 0, 0)


def warp_symbol_to_quad(symbol_img, target_quad, output_size):
    """
    Warp a symbol image to fit within a target quadrilateral.

    Args:
        symbol_img: PIL Image of the symbol (RGBA)
        target_quad: List of 4 (x, y) tuples defining the target quadrilateral
                     in order: top-left, top-right, bottom-right, bottom-left
        output_size: Size of the output image (width, height)

    Returns:
        PIL Image with symbol warped to fit the quad
    """
    w, h = symbol_img.size

    # Source rectangle corners (the flat symbol image)
    source_coords = [(0, 0), (w, 0), (w, h), (0, h)]

    # Calculate perspective coefficients
    coeffs = find_perspective_coeffs(source_coords, target_quad)

    # Create output image
    output = Image.new('RGBA', output_size, (0, 0, 0, 0))

    # Apply perspective transform
    warped = symbol_img.transform(
        output_size,
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC
    )

    return warped


def render_cube_net(symbols, net_pattern=None):
    """
    Render an unfolded cube net with emojis on each face.

    Args:
        symbols: List of 6 emoji strings for each face
        net_pattern: Which net pattern to use (defaults to cross)

    Returns:
        PIL Image of the unfolded net
    """
    if net_pattern is None:
        net_pattern = CUBE_NET_CROSS

    # Calculate image size based on net bounds
    max_x = max(pos[0] for pos in net_pattern.keys()) + 1
    max_y = max(pos[1] for pos in net_pattern.keys()) + 1

    padding = 30
    img_width = max_x * CELL_SIZE + padding * 2
    img_height = max_y * CELL_SIZE + padding * 2

    img = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    for (grid_x, grid_y), face_idx in net_pattern.items():
        x = padding + grid_x * CELL_SIZE
        y = padding + grid_y * CELL_SIZE

        # Draw face background with border
        draw.rectangle([x, y, x + CELL_SIZE, y + CELL_SIZE],
                      outline=(40, 40, 40), width=3, fill=(252, 252, 255))

        # Draw emoji centered in face
        center_x = x + CELL_SIZE // 2
        center_y = y + CELL_SIZE // 2

        emoji = symbols[face_idx]
        img = draw_emoji(img, emoji, center_x, center_y, CELL_SIZE - 25)
        draw = ImageDraw.Draw(img)

    return img.convert('RGB')


# Define viewing corners mathematically
# Each corner shows 3 faces: (front_face, vertical_side, horizontal_side)
# and their projection parameters
# Only using top-left and top-right views for cleaner presentation
CUBE_VIEW_CORNERS = {
    'front_top_right': {
        'visible': [0, 2, 5],  # Front, Top, Right
        'main_face': 0,
        'h_side': 2,  # horizontal side (top)
        'v_side': 5,  # vertical side (right)
        'depth_angle': math.radians(45),
        'h_side_dir': 'up',
        'v_side_dir': 'right',
    },
    'front_top_left': {
        'visible': [0, 2, 4],  # Front, Top, Left
        'main_face': 0,
        'h_side': 2,
        'v_side': 4,
        'depth_angle': math.radians(135),
        'h_side_dir': 'up',
        'v_side_dir': 'left',
    },
}


def calculate_view_geometry(corner_name, size):
    """
    Calculate the vertex positions for each visible face based on viewing corner.
    Uses mathematical projection to compute face positions.

    Args:
        corner_name: Name of the viewing corner
        size: Output image size

    Returns:
        Dict mapping face_idx to list of 4 vertex positions (TL, TR, BR, BL)
    """
    corner = CUBE_VIEW_CORNERS[corner_name]

    cx, cy = size // 2, size // 2
    face_size = size * 0.35
    depth = face_size * 0.65

    # Calculate depth offsets based on corner's depth angle
    depth_angle = corner['depth_angle']
    dx = depth * math.cos(depth_angle)
    dy = -depth * math.sin(depth_angle)

    # Main face (front-facing) is always a square centered on screen
    main_tl = (cx - face_size/2, cy - face_size/2)
    main_tr = (cx + face_size/2, cy - face_size/2)
    main_br = (cx + face_size/2, cy + face_size/2)
    main_bl = (cx - face_size/2, cy + face_size/2)

    faces = {}

    # Main face
    faces[corner['main_face']] = [main_tl, main_tr, main_br, main_bl]

    # Horizontal side face (top or bottom)
    if corner['h_side_dir'] == 'up':
        # Top face: parallelogram above main face
        h_fl = main_tl
        h_fr = main_tr
        h_br = (main_tr[0] + dx, main_tr[1] + dy)
        h_bl = (main_tl[0] + dx, main_tl[1] + dy)
        faces[corner['h_side']] = [h_bl, h_br, h_fr, h_fl]  # back-left, back-right, front-right, front-left
    else:
        # Bottom face: parallelogram below main face
        h_fl = main_bl
        h_fr = main_br
        h_br = (main_br[0] + dx, main_br[1] + dy)
        h_bl = (main_bl[0] + dx, main_bl[1] + dy)
        faces[corner['h_side']] = [h_fl, h_fr, h_br, h_bl]  # front-left, front-right, back-right, back-left

    # Vertical side face (left or right)
    if corner['v_side_dir'] == 'right':
        # Right face: parallelogram to the right
        v_tl = main_tr
        v_tr = (main_tr[0] + dx, main_tr[1] + dy)
        v_br = (main_br[0] + dx, main_br[1] + dy)
        v_bl = main_br
        faces[corner['v_side']] = [v_tl, v_tr, v_br, v_bl]
    else:
        # Left face: parallelogram to the left
        v_tl = (main_tl[0] + dx, main_tl[1] + dy)
        v_tr = main_tl
        v_br = main_bl
        v_bl = (main_bl[0] + dx, main_bl[1] + dy)
        faces[corner['v_side']] = [v_tl, v_tr, v_br, v_bl]

    return faces


def render_3d_cube(symbols, visible_faces, rotations=None, size=None, view_corner=None):
    """
    Render a 3D isometric view of a cube showing 3 faces clearly.
    Uses mathematical projection based on viewing corner.
    Emojis are perspective-warped to match each face's orientation.

    Args:
        symbols: List of 6 emoji strings for each face
        visible_faces: List of 3 face indices that are visible
        rotations: Dict mapping face_idx to rotation degrees (CCW) based on folding geometry
        size: Output image size
        view_corner: Name of viewing corner (determines projection geometry)

    Returns:
        PIL Image of the 3D cube
    """
    if rotations is None:
        rotations = {f: 0 for f in visible_faces}
    if size is None:
        size = CUBE_SIZE

    # Determine view corner from visible faces if not specified
    if view_corner is None:
        # Find matching corner based on visible faces
        visible_set = set(visible_faces)
        for corner_name, corner_data in CUBE_VIEW_CORNERS.items():
            if set(corner_data['visible']) == visible_set:
                view_corner = corner_name
                break
        if view_corner is None:
            view_corner = 'front_top_right'  # Default fallback

    img = Image.new('RGBA', (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Calculate face geometries mathematically
    faces = calculate_view_geometry(view_corner, size)
    corner = CUBE_VIEW_CORNERS[view_corner]

    # Face background colors with 3D lighting effect
    # Main face brightest, side faces progressively darker
    face_colors = {
        corner['main_face']: (250, 250, 255),  # Main - brightest
        corner['h_side']: (225, 230, 248),      # Horizontal side - medium
        corner['v_side']: (195, 205, 225),      # Vertical side - darker
    }

    # Draw order: back faces first, then front
    draw_order = [corner['h_side'], corner['v_side'], corner['main_face']]

    # Emoji size for rendering (before perspective transform)
    emoji_render_size = 256

    for face_idx in draw_order:
        if face_idx not in visible_faces or face_idx not in faces:
            continue

        verts = faces[face_idx]

        # Draw face polygon background
        draw.polygon(verts, fill=face_colors.get(face_idx, (220, 220, 220)), outline=(30, 30, 30), width=2)

        # Get emoji for this face
        emoji = symbols[face_idx]

        # Render emoji as flat image
        emoji_img = render_emoji_image(emoji, emoji_render_size)

        # Apply rotation based on folding geometry BEFORE perspective warp
        rotation_deg = rotations.get(face_idx, 0)
        if rotation_deg != 0:
            emoji_img = emoji_img.rotate(rotation_deg, expand=False, resample=Image.Resampling.BICUBIC)

        # Calculate the quad for the emoji (slightly inset from face edges)
        inset = 0.09
        def lerp(p1, p2, t):
            return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)

        tl, tr, br, bl = verts
        emoji_tl = lerp(lerp(tl, tr, inset), lerp(bl, br, inset), inset)
        emoji_tr = lerp(lerp(tl, tr, 1-inset), lerp(bl, br, 1-inset), inset)
        emoji_br = lerp(lerp(tl, tr, 1-inset), lerp(bl, br, 1-inset), 1-inset)
        emoji_bl = lerp(lerp(tl, tr, inset), lerp(bl, br, inset), 1-inset)

        emoji_quad = [emoji_tl, emoji_tr, emoji_br, emoji_bl]

        # Warp emoji to fit the quad (applies perspective transform)
        warped_emoji = warp_symbol_to_quad(emoji_img, emoji_quad, (size, size))

        # Composite onto the cube image
        img = Image.alpha_composite(img, warped_emoji)
        draw = ImageDraw.Draw(img)

    # Redraw outlines on top for crisp edges
    for face_idx in draw_order:
        if face_idx not in visible_faces or face_idx not in faces:
            continue
        verts = faces[face_idx]
        draw.polygon(verts, fill=None, outline=(30, 30, 30), width=2)

    return img.convert('RGB')


def generate_correct_cube_view(symbols, rotations, view_corner=None):
    """
    Generate the correct 3D cube view from the symbol assignment.

    Args:
        symbols: List of 6 emoji strings for each face
        rotations: Dict mapping face_idx to rotation degrees from calculate_face_rotations()
        view_corner: Optional specific view corner; if None, randomly selected
    """
    # Randomly select a viewing corner for diverse cube views
    if view_corner is None:
        view_corner = random.choice(list(CUBE_VIEW_CORNERS.keys()))

    visible_faces = CUBE_VIEW_CORNERS[view_corner]['visible']
    img = render_3d_cube(symbols, visible_faces, rotations, view_corner=view_corner)
    return img, visible_faces, view_corner


def generate_incorrect_cube_view(symbols, rotations, view_corner=None):
    """
    Generate an incorrect 3D cube view by either:
    1. Swapping face symbols in a way that violates folding rules
    2. Keeping correct symbols but applying wrong orientation to one face

    Args:
        symbols: List of 6 emoji strings for each face
        rotations: Dict mapping face_idx to rotation degrees from calculate_face_rotations()
        view_corner: Optional specific view corner; if None, randomly selected
    """
    # Randomly select a viewing corner for diverse cube views
    if view_corner is None:
        view_corner = random.choice(list(CUBE_VIEW_CORNERS.keys()))

    visible_faces = CUBE_VIEW_CORNERS[view_corner]['visible']

    # Determine hidden faces for this view
    all_faces = set(range(6))
    hidden_faces = list(all_faces - set(visible_faces))

    # Create copies that we can modify
    wrong_symbols = list(symbols)
    wrong_rotations = dict(rotations)

    error_type = random.choice(['swap_opposite', 'swap_adjacent', 'use_hidden', 'wrong_orientation'])

    if error_type == 'swap_opposite':
        # Swap a visible face with its opposite (impossible in real folding)
        face_to_swap = random.choice(visible_faces)
        opposite = OPPOSITE_FACES[face_to_swap]
        wrong_symbols[face_to_swap], wrong_symbols[opposite] = wrong_symbols[opposite], wrong_symbols[face_to_swap]
        desc = f"swapped face {face_to_swap} with opposite {opposite}"

    elif error_type == 'swap_adjacent':
        # Swap two visible faces (wrong spatial relationship)
        f1, f2 = random.sample(visible_faces, 2)
        wrong_symbols[f1], wrong_symbols[f2] = wrong_symbols[f2], wrong_symbols[f1]
        desc = f"swapped visible faces {f1} and {f2}"

    elif error_type == 'use_hidden':
        # Put a hidden face's symbol where it shouldn't be
        hidden = random.choice(hidden_faces)
        visible = random.choice(visible_faces)
        wrong_symbols[visible] = symbols[hidden]
        desc = f"put hidden face {hidden} symbol on visible face {visible}"

    else:  # wrong_orientation
        # Keep correct symbols but apply wrong rotation to one face
        # This creates a subtly incorrect cube - right emojis, wrong orientation
        face_to_misrotate = random.choice(visible_faces)

        # Pick a rotation offset that's noticeably different (90°, 180°, or 270°)
        rotation_offsets = [90, 180, -90]  # Avoid 0 which would be correct
        offset = random.choice(rotation_offsets)

        # Apply the offset to the face's rotation
        current_rotation = wrong_rotations.get(face_to_misrotate, 0)
        wrong_rotations[face_to_misrotate] = current_rotation + offset

        desc = f"wrong orientation on face {face_to_misrotate} (rotated {offset}° extra)"

    img = render_3d_cube(wrong_symbols, visible_faces, wrong_rotations, view_corner=view_corner)
    return img, visible_faces, desc, view_corner


def generate_cell_pool():
    """
    Generate pool of cube images (nets and 3D views).
    """
    print("Generating cell pool...")

    cell_pool = {}
    net_metadata = {}

    # Generate symbol sets
    symbol_sets = generate_symbol_sets()
    print(f"  Using {len(symbol_sets)} symbol sets")

    net_idx = 0

    for symbol_set in symbol_sets:
        for net_pattern in ALL_NETS:
            net_id = f"net_{net_idx:04d}"

            # Shuffle symbols for this net
            symbols = list(symbol_set)
            random.shuffle(symbols)

            # Calculate emoji rotations based on the net's folding geometry
            # This uses mathematical modeling of how each face folds
            rotations = calculate_face_rotations(net_pattern)

            # Generate and save the net image
            net_img = render_cube_net(symbols, net_pattern)
            net_filename = f"{net_id}_unfolded.png"
            net_img.save(OUTPUT_DIR / net_filename)

            # Generate TWO correct cubes - one for each view corner
            # This ensures puzzles have two distinct correct answers
            correct_cells = []
            for corner_name in CUBE_VIEW_CORNERS.keys():
                correct_img, visible_faces, view_corner = generate_correct_cube_view(symbols, rotations, view_corner=corner_name)
                correct_filename = f"{net_id}_correct_{corner_name}.png"
                correct_img.save(OUTPUT_DIR / correct_filename)
                correct_cell_id = f"{net_id}_correct_{corner_name}"

                cell_pool[correct_cell_id] = {
                    "filename": correct_filename,
                    "net_id": net_id,
                    "is_correct": True,
                    "visible_faces": visible_faces,
                    "view_corner": view_corner,
                }
                correct_cells.append(correct_cell_id)

            # Generate multiple incorrect cubes (with same rotations for consistency)
            for wrong_idx in range(8):
                wrong_cell_id = f"{net_id}_wrong_{wrong_idx}"
                wrong_img, visible_faces, error_desc, view_corner = generate_incorrect_cube_view(symbols, rotations)
                wrong_filename = f"{net_id}_wrong_{wrong_idx}.png"
                wrong_img.save(OUTPUT_DIR / wrong_filename)

                cell_pool[wrong_cell_id] = {
                    "filename": wrong_filename,
                    "net_id": net_id,
                    "is_correct": False,
                    "visible_faces": visible_faces,
                    "error_type": error_desc,
                    "view_corner": view_corner,
                }

            # Store net metadata
            net_metadata[net_id] = {
                "net_filename": net_filename,
                "symbols": symbols,  # List of emoji strings
                "correct_cells": correct_cells,  # List of 2 correct cell IDs (one per view corner)
            }

            net_idx += 1

            if net_idx % 10 == 0:
                print(f"  Generated {net_idx} nets...")

    print(f"✓ Generated {len(cell_pool)} cube images from {len(net_metadata)} nets")

    # Save cell pool
    pool_file = OUTPUT_DIR / "cell_pool.json"
    with open(pool_file, 'w') as f:
        json.dump(cell_pool, f, indent=2)

    # Save net metadata
    net_file = OUTPUT_DIR / "net_metadata.json"
    with open(net_file, 'w') as f:
        json.dump(net_metadata, f, indent=2)

    return cell_pool, net_metadata


def generate_puzzles(cell_pool, net_metadata):
    """
    Generate puzzles by selecting a net and composing options grid.
    """
    print(f"\nGenerating {NUM_PUZZLES} puzzles...")

    ground_truth = {}
    net_ids = list(net_metadata.keys())

    for puzzle_idx in range(NUM_PUZZLES):
        puzzle_id = f"box_folding_{puzzle_idx:04d}"

        # Select a random net
        net_id = random.choice(net_ids)
        net_info = net_metadata[net_id]

        # Get both correct cells (two different view corners)
        correct_cells = net_info["correct_cells"]

        # Get wrong cells for this net
        wrong_cells = [cid for cid, data in cell_pool.items()
                      if data["net_id"] == net_id and not data["is_correct"]]

        num_correct = 2  # Include 2 correct answers
        num_wrong = TOTAL_CELLS - num_correct

        # Select cells - include both distinct correct cubes
        selected_correct = correct_cells[:2]  # Both view corners
        selected_wrong = random.sample(wrong_cells, min(num_wrong, len(wrong_cells)))

        # If we need more wrong answers, borrow from other nets
        if len(selected_wrong) < num_wrong:
            other_wrong = [cid for cid, data in cell_pool.items()
                         if data["net_id"] != net_id and not data["is_correct"]]
            additional = random.sample(other_wrong, num_wrong - len(selected_wrong))
            selected_wrong.extend(additional)

        # Combine and shuffle
        all_cells = selected_correct + selected_wrong[:num_wrong]
        random.shuffle(all_cells)

        # Find correct indices (both distinct correct cells)
        correct_indices = [i for i, cid in enumerate(all_cells) if cid in correct_cells]

        ground_truth[puzzle_id] = {
            "prompt": "Select the two cubes that could be formed by folding the pattern shown above.",
            "description": "Mental box folding - identify correct 3D cubes from 2D net",
            "net_id": net_id,
            "reference_image": net_info["net_filename"],
            "cells": all_cells,
            "answer": correct_indices,
            "input_type": "box_folding_select",
            "grid_size": list(GRID_SIZE),
        }

        print(f"  ✓ {puzzle_id}: net={net_id}, correct at position {correct_indices}")

    # Save ground truth
    gt_file = OUTPUT_DIR / "ground_truth.json"
    with open(gt_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Generated {len(ground_truth)} puzzles")
    print(f"✓ Ground truth saved to {gt_file}")


def main():
    print("=" * 60)
    print("Box Folding CAPTCHA Generator")
    print("=" * 60)
    print("\nThis CAPTCHA exploits LLM weakness in mental 3D folding.")
    print("Users must identify which 3D cube results from folding a 2D net.\n")

    # Generate cell pool
    cell_pool, net_metadata = generate_cell_pool()

    # Generate puzzles
    generate_puzzles(cell_pool, net_metadata)

    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
