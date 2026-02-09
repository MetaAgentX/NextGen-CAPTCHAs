"""
Occluded Pattern Counting CAPTCHA Generator

Based on CAPTURe (ICCV 2025) findings that VLMs struggle with occluded pattern counting
while humans excel at this task through pattern inference and amodal completion.

Generates two complementary CAPTCHA families:
- Family A: Occluded repeating patterns (CAPTURe-style grid patterns with occluders)
- Family B: Layered stacks with depth reasoning (z-order occlusion)

Reference: https://arxiv.org/abs/2504.15485
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw

# -----------------------------
# Configuration
# -----------------------------

# Output configuration
OUTPUT_DIR = Path('../captcha_data/Occluded_Pattern_Counting/')
NUM_PUZZLES = 50  # Total puzzles to generate
IMAGE_SIZE = 600  # Square images - larger for better pattern visibility

# Advanced VLM-breaking features
USE_PATTERN_BREAKS = True  # ALWAYS break pattern, but in different ways
USE_SEMI_TRANSPARENT = True  # Make occluder semi-transparent (humans can see, VLMs fail)
OCCLUDER_OPACITY = 0.45  # 45% = humans can clearly see through, VLMs still fooled (slightly lower)

# Pattern break types (ALWAYS applied, but randomly chosen method):
PATTERN_BREAK_TYPES = {
    'color_swap': 0.5,        # Swap colors of some shapes under occluder
    'shape_swap': 0.5,        # Swap shapes under occluder
    'random_replace': 0.0,    # Replace with completely random shapes/colors
    'partial_break': 0.0,     # Break pattern in only some rows/cols under occluder
}

USE_PARTIAL_OCCLUSION = True  # Occlude only half of shapes (left/right/top/bottom/diagonal)
PARTIAL_OCCLUSION_PROBABILITY = 0.5  # 0% = NO solid occlusion, only semi-transparent 
# Grid configuration - LARGER grids to convince VLMs the pattern is real
MIN_ROWS, MAX_ROWS = 8, 10  # More rows = more pattern repetitions visible
MIN_COLS, MAX_COLS = 10, 14  # More columns = stronger pattern signal for VLMs
MIN_PATTERN_SIZE, MAX_PATTERN_SIZE = 2, 3  # Pattern repeat period (small for easy inference)

# Color palette (highly distinct, high-contrast colors for easy human differentiation)
# Optimized for visibility through semi-transparent grey occluder
COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "red":      (255, 30, 30),      # Bright red - very distinct
    "green":    (30, 220, 30),      # Bright green - high contrast with red
    "blue":     (30, 60, 255),      # Bright blue - clearly different
    "yellow":   (255, 220, 0),      # Bright yellow - high luminance
    "purple":   (180, 30, 200),     # Vivid purple - distinct hue
    "cyan":     (0, 220, 220),      # Bright cyan - between blue/green
    "orange":   (255, 140, 0),      # Bright orange - between red/yellow
    "pink":     (255, 80, 180),     # Hot pink - clearly different from red/purple
}

SHAPE_TYPES = ["square", "circle", "triangle", "star", "diamond", "hexagon"]

# Question template - two-part counting task for increased complexity
# Format: "Count A and B. Enter as: A,B"
# Include list of possible shapes to help users understand what to look for
QUESTION_TEMPLATES = [
    "Possible shapes: squares, circles, triangles, stars, diamonds, hexagons. Count the {color1} {shape1_plural} and {color2} {shape2_plural} under the grey overlay. Enter as: count1,count2",
]


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ShapeSpec:
    """Represents a shape in the grid or layer."""
    shape_type: str
    color_name: str
    color_rgb: Tuple[int, int, int]
    row: int
    col: int
    layer: Optional[int] = None  # For layered family


@dataclass
class CaptchaSample:
    """Complete CAPTCHA sample with metadata."""
    image: Image.Image
    question: str
    answer: int
    puzzle_type: str  # "pattern" or "layered"
    metadata: Dict


# -----------------------------
# Shape rendering utilities
# -----------------------------

def draw_square(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, color: Tuple[int, int, int]):
    """Draw a square centered at (cx, cy)."""
    half = size
    draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=color, outline=(0, 0, 0), width=3)


def draw_circle(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, color: Tuple[int, int, int]):
    """Draw a circle centered at (cx, cy)."""
    half = size
    draw.ellipse([cx - half, cy - half, cx + half, cy + half], fill=color, outline=(0, 0, 0), width=3)


def draw_triangle(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, color: Tuple[int, int, int]):
    """Draw an equilateral triangle pointing up."""
    half = size
    p1 = (cx, cy - half * 1.2)
    p2 = (cx - half, cy + half * 0.6)
    p3 = (cx + half, cy + half * 0.6)
    draw.polygon([p1, p2, p3], fill=color, outline=(0, 0, 0), width=3)


def draw_star(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, color: Tuple[int, int, int]):
    """Draw a 5-point star."""
    R = size
    r = size * 0.4
    points = []
    for i in range(10):
        angle = np.pi / 5 * i - np.pi / 2  # Start pointing up
        radius = R if i % 2 == 0 else r
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=color, outline=(0, 0, 0), width=3)


def draw_diamond(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, color: Tuple[int, int, int]):
    """Draw a diamond (rotated square)."""
    half = size
    points = [
        (cx, cy - half),
        (cx + half, cy),
        (cx, cy + half),
        (cx - half, cy),
    ]
    draw.polygon(points, fill=color, outline=(0, 0, 0), width=3)


def draw_hexagon(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, color: Tuple[int, int, int]):
    """Draw a regular hexagon."""
    points = []
    for i in range(6):
        angle = np.pi / 3 * i - np.pi / 2  # Start pointing up
        x = cx + size * np.cos(angle)
        y = cy + size * np.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=color, outline=(0, 0, 0), width=3)


SHAPE_RENDERERS = {
    "square": draw_square,
    "circle": draw_circle,
    "triangle": draw_triangle,
    "star": draw_star,
    "diamond": draw_diamond,
    "hexagon": draw_hexagon,
}


def get_shape_plural(shape_type: str) -> str:
    """Convert shape type to plural form."""
    if shape_type == "hexagon":
        return "hexagons"
    return shape_type + "s"


# -----------------------------
# Family A: Occluded repeating patterns
# -----------------------------

def sample_base_pattern(base_rows: int, base_cols: int) -> List[List[ShapeSpec]]:
    """
    Sample a small repeating pattern unit.

    Returns:
        2D list of ShapeSpec, shape (base_rows, base_cols)
    """
    pattern = []
    color_names = list(COLOR_PALETTE.keys())

    for r in range(base_rows):
        row = []
        for c in range(base_cols):
            shape_type = random.choice(SHAPE_TYPES)
            color_name = random.choice(color_names)
            color_rgb = COLOR_PALETTE[color_name]
            row.append(ShapeSpec(
                shape_type=shape_type,
                color_name=color_name,
                color_rgb=color_rgb,
                row=r,
                col=c,
            ))
        pattern.append(row)

    return pattern


def tile_pattern_to_grid(
    base_pattern: List[List[ShapeSpec]],
    n_rows: int,
    n_cols: int
) -> List[List[ShapeSpec]]:
    """
    Tile the base pattern to create full grid.

    Returns:
        2D list of ShapeSpec, shape (n_rows, n_cols)
    """
    base_rows = len(base_pattern)
    base_cols = len(base_pattern[0])

    grid = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            base_r = r % base_rows
            base_c = c % base_cols
            base_spec = base_pattern[base_r][base_c]
            row.append(ShapeSpec(
                shape_type=base_spec.shape_type,
                color_name=base_spec.color_name,
                color_rgb=base_spec.color_rgb,
                row=r,
                col=c,
            ))
        grid.append(row)

    return grid


def inject_pattern_breaks(
    grid: List[List[ShapeSpec]],
    occluder_mask: np.ndarray
) -> tuple[List[List[ShapeSpec]], str]:
    """
    ANTI-VLM WEAPON: ALWAYS break the pattern under occluder, but use different strategies.

    VLMs assume patterns continue → they hallucinate the count
    Humans can squint through semi-transparent occluder → see the truth

    Args:
        grid: Current grid (will be modified)
        occluder_mask: Boolean array marking occluded cells

    Returns:
        (modified_grid, break_type_used)
    """
    n_rows = len(grid)
    n_cols = len(grid[0])

    # Find all occluded cells
    occluded_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)
                      if occluder_mask[r, c]]

    if not occluded_cells:
        return grid, "none"

    # Choose break strategy randomly based on weights
    break_type = random.choices(
        list(PATTERN_BREAK_TYPES.keys()),
        weights=list(PATTERN_BREAK_TYPES.values())
    )[0]

    alt_colors = list(COLOR_PALETTE.keys())
    alt_shapes = SHAPE_TYPES.copy()

    if break_type == 'color_swap':
        # Swap colors only (50-70% of occluded cells - increased aggression)
        num_breaks = max(3, int(len(occluded_cells) * random.uniform(0.5, 0.7)))
        cells_to_break = random.sample(occluded_cells, num_breaks)

        for r, c in cells_to_break:
            original = grid[r][c]
            new_color = random.choice([col for col in alt_colors if col != original.color_name])
            grid[r][c] = ShapeSpec(
                shape_type=original.shape_type,
                color_name=new_color,
                color_rgb=COLOR_PALETTE[new_color],
                row=r, col=c
            )

    elif break_type == 'shape_swap':
        # Swap shapes only (50-70% of occluded cells - increased aggression)
        num_breaks = max(3, int(len(occluded_cells) * random.uniform(0.5, 0.7)))
        cells_to_break = random.sample(occluded_cells, num_breaks)

        for r, c in cells_to_break:
            original = grid[r][c]
            new_shape = random.choice([shp for shp in alt_shapes if shp != original.shape_type])
            grid[r][c] = ShapeSpec(
                shape_type=new_shape,
                color_name=original.color_name,
                color_rgb=original.color_rgb,
                row=r, col=c
            )

    elif break_type == 'random_replace':
        # Completely randomize some cells (40-60% - increased aggression)
        num_breaks = max(2, int(len(occluded_cells) * random.uniform(0.4, 0.6)))
        cells_to_break = random.sample(occluded_cells, num_breaks)

        for r, c in cells_to_break:
            grid[r][c] = ShapeSpec(
                shape_type=random.choice(alt_shapes),
                color_name=random.choice(alt_colors),
                color_rgb=COLOR_PALETTE[random.choice(alt_colors)],
                row=r, col=c
            )

    elif break_type == 'partial_break':
        # Break pattern in only specific rows or columns under occluder
        occluded_rows = sorted(set(r for r, c in occluded_cells))
        occluded_cols = sorted(set(c for r, c in occluded_cells))

        if random.random() < 0.5 and len(occluded_rows) >= 2:
            # Break one random row
            break_row = random.choice(occluded_rows)
            cells_to_break = [(r, c) for r, c in occluded_cells if r == break_row]
        else:
            # Break one random column
            break_col = random.choice(occluded_cols)
            cells_to_break = [(r, c) for r, c in occluded_cells if c == break_col]

        for r, c in cells_to_break:
            # Mix color and shape changes
            if random.random() < 0.5:
                original = grid[r][c]
                grid[r][c] = ShapeSpec(
                    shape_type=random.choice([s for s in alt_shapes if s != original.shape_type]),
                    color_name=original.color_name,
                    color_rgb=original.color_rgb,
                    row=r, col=c
                )
            else:
                original = grid[r][c]
                new_color = random.choice([col for col in alt_colors if col != original.color_name])
                grid[r][c] = ShapeSpec(
                    shape_type=original.shape_type,
                    color_name=new_color,
                    color_rgb=COLOR_PALETTE[new_color],
                    row=r, col=c
                )

    return grid, break_type


def sample_occluder_mask(n_rows: int, n_cols: int) -> np.ndarray:
    """
    Sample an occluder that hides complete grid cells.

    Increased coverage: Occluder should cover 40-55% of the grid,
    leaving enough visible pattern for humans to infer what's hidden.

    Returns:
        Boolean array of shape (n_rows, n_cols), True = occluded
    """
    mask = np.zeros((n_rows, n_cols), dtype=bool)

    occluder_type = random.choice(["horizontal_band", "vertical_band", "block"])

    if occluder_type == "horizontal_band":
        # Horizontal stripe - cover 3-5 rows (increased from 2-3)
        height = random.randint(3, min(5, max(3, n_rows // 2)))
        start_row = random.randint(1, n_rows - height - 1)  # Not at edges
        # Always span full width for larger coverage
        mask[start_row:start_row + height, :] = True

    elif occluder_type == "vertical_band":
        # Vertical stripe - cover 4-7 columns (increased from 2-4)
        width = random.randint(4, min(7, max(4, n_cols // 2)))
        start_col = random.randint(1, n_cols - width - 1)  # Not at edges
        # Always span full height for larger coverage
        mask[:, start_col:start_col + width] = True

    elif occluder_type == "block":
        # Rectangular block - larger than before
        height = random.randint(3, min(6, max(3, int(n_rows * 0.6))))
        width = random.randint(5, min(8, max(5, int(n_cols * 0.6))))
        start_row = random.randint(1, n_rows - height - 1)
        start_col = random.randint(1, n_cols - width - 1)
        mask[start_row:start_row + height, start_col:start_col + width] = True

    return mask


def render_pattern_grid(
    grid: List[List[ShapeSpec]],
    occluder_mask: np.ndarray,
    image_size: int = 400,
    use_semi_transparent: bool = False,
    opacity: float = 0.85
) -> Image.Image:
    """
    Render the pattern grid with occluder overlay.

    Args:
        grid: 2D list of ShapeSpec
        occluder_mask: Boolean array, shape (n_rows, n_cols)
        image_size: Output image size (square)
        use_semi_transparent: If True, occluder is semi-transparent (VLM-breaking feature)
        opacity: Occluder opacity (0.0=transparent, 1.0=opaque)

    Returns:
        PIL Image in RGB mode
    """
    n_rows = len(grid)
    n_cols = len(grid[0])

    img = Image.new("RGB", (image_size, image_size), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    cell_w = image_size / n_cols
    cell_h = image_size / n_rows

    # Draw subtle grid lines
    for r in range(n_rows + 1):
        y = int(r * cell_h)
        draw.line([(0, y), (image_size, y)], fill=(220, 220, 220), width=1)
    for c in range(n_cols + 1):
        x = int(c * cell_w)
        draw.line([(x, 0), (x, image_size)], fill=(220, 220, 220), width=1)

    # Draw shapes (larger for better visibility through semi-transparent occluder)
    for r in range(n_rows):
        for c in range(n_cols):
            spec = grid[r][c]
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            size = min(cell_w, cell_h) * 0.35  # Increased from 0.30 to 0.35

            renderer = SHAPE_RENDERERS[spec.shape_type]
            renderer(draw, cx, cy, size, spec.color_rgb)

    # Draw occluder overlay
    if use_semi_transparent and opacity < 1.0:
        # Create semi-transparent overlay using alpha compositing
        overlay = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        occluder_color = (120, 120, 120)
        alpha = int(255 * opacity)  # Base semi-transparent grey

        for r in range(n_rows):
            for c in range(n_cols):
                if occluder_mask[r, c]:
                    x0 = int(c * cell_w)
                    y0 = int(r * cell_h)
                    x1 = int((c + 1) * cell_w)
                    y1 = int((r + 1) * cell_h)

                    # Step 1: Draw base semi-transparent grey (45% opacity)
                    overlay_draw.rectangle([x0, y0, x1, y1],
                                          fill=occluder_color + (alpha,))

                    # Step 2: Randomly add ADDITIONAL solid grey covering HALF the cell
                    if random.random() < PARTIAL_OCCLUSION_PROBABILITY:
                        partial_type = random.choice([
                            'top', 'bottom', 'left', 'right',
                            'diagonal_tl_br', 'diagonal_tr_bl'
                        ])

                        cx_mid = (x0 + x1) / 2
                        cy_mid = (y0 + y1) / 2

                        # Draw pure solid grey (100% opacity) over half the cell
                        if partial_type == 'top':
                            overlay_draw.rectangle([x0, y0, x1, cy_mid],
                                                  fill=occluder_color + (255,))
                        elif partial_type == 'bottom':
                            overlay_draw.rectangle([x0, cy_mid, x1, y1],
                                                  fill=occluder_color + (255,))
                        elif partial_type == 'left':
                            overlay_draw.rectangle([x0, y0, cx_mid, y1],
                                                  fill=occluder_color + (255,))
                        elif partial_type == 'right':
                            overlay_draw.rectangle([cx_mid, y0, x1, y1],
                                                  fill=occluder_color + (255,))
                        elif partial_type == 'diagonal_tl_br':
                            # Top-left to bottom-right diagonal
                            overlay_draw.polygon([(x0, y0), (x1, y0), (x0, y1)],
                                               fill=occluder_color + (255,))
                        elif partial_type == 'diagonal_tr_bl':
                            # Top-right to bottom-left diagonal
                            overlay_draw.polygon([(x1, y0), (x1, y1), (x0, y0)],
                                               fill=occluder_color + (255,))

        # Composite overlay onto base image
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
    else:
        # Solid occluder (old behavior)
        occluder_color = (120, 120, 120)
        for r in range(n_rows):
            for c in range(n_cols):
                if occluder_mask[r, c]:
                    x0 = int(c * cell_w)
                    y0 = int(r * cell_h)
                    x1 = int((c + 1) * cell_w)
                    y1 = int((r + 1) * cell_h)
                    draw.rectangle([x0, y0, x1, y1], fill=occluder_color)

    return img


def choose_pattern_question(
    grid: List[List[ShapeSpec]],
    occluder_mask: np.ndarray,
    max_answer: int = 12
) -> Tuple[str, str, Dict]:
    """
    Choose a TWO-PART question about occluded shapes and compute answers.

    ANTI-VLM STRATEGY: Ask about TWO color+shape combos that are MAJORITY in VISIBLE area.
    This makes VLMs assume the same color frequency continues under the overlay,
    leading them to hallucinate counts based on visible pattern.

    Ensures both answers are in reasonable range [1, max_answer].

    Returns:
        (question_text, answer_string, metadata_dict)
        answer_string format: "count1,count2"
    """
    n_rows = len(grid)
    n_cols = len(grid[0])

    # Collect visible shapes (NOT occluded)
    visible_shapes = []
    for r in range(n_rows):
        for c in range(n_cols):
            if not occluder_mask[r, c]:
                visible_shapes.append(grid[r][c])

    # Collect occluded shapes
    occluded_shapes = []
    for r in range(n_rows):
        for c in range(n_cols):
            if occluder_mask[r, c]:
                occluded_shapes.append(grid[r][c])

    if not occluded_shapes or not visible_shapes:
        # Fallback: should not happen with proper occluder
        return "How many shapes are hidden?", "0,0", {}

    # Count color+shape frequencies in VISIBLE area
    from collections import Counter
    visible_combos = Counter((s.color_name, s.shape_type) for s in visible_shapes)

    # Find the most common color+shape combos in visible area
    most_common_combos = visible_combos.most_common(10)  # Top 10 most frequent

    # Try to find TWO different combos that both have valid counts in occluded area
    random.shuffle(most_common_combos)

    for i, ((color1, shape1), _) in enumerate(most_common_combos):
        # Count for first combo in occluded area
        count1 = sum(
            1 for s in occluded_shapes
            if s.color_name == color1 and s.shape_type == shape1
        )

        if not (1 <= count1 <= max_answer):
            continue

        # Try to find a second combo that's different from the first
        for (color2, shape2), _ in most_common_combos[i+1:]:
            # Ensure second combo is different
            if color2 == color1 and shape2 == shape1:
                continue

            # Count for second combo in occluded area
            count2 = sum(
                1 for s in occluded_shapes
                if s.color_name == color2 and s.shape_type == shape2
            )

            if 1 <= count2 <= max_answer:
                template = random.choice(QUESTION_TEMPLATES)
                question = template.format(
                    color1=color1,
                    shape1_plural=get_shape_plural(shape1),
                    color2=color2,
                    shape2_plural=get_shape_plural(shape2)
                )

                answer_string = f"{count1},{count2}"

                metadata = {
                    "target1_color": color1,
                    "target1_shape": shape1,
                    "target1_count": count1,
                    "target2_color": color2,
                    "target2_shape": shape2,
                    "target2_count": count2,
                    "total_occluded": len(occluded_shapes),
                    "pattern_type": "repeating_grid",
                    "visible_frequency1": visible_combos[(color1, shape1)],
                    "visible_frequency2": visible_combos[(color2, shape2)],
                }

                return question, answer_string, metadata

    # Fallback: if we can't find two common visible combos, pick any two different combos
    attempts = 0
    while attempts < 50:
        spec1 = random.choice(occluded_shapes)
        spec2 = random.choice(occluded_shapes)

        # Ensure they're different
        if spec1.color_name == spec2.color_name and spec1.shape_type == spec2.shape_type:
            attempts += 1
            continue

        count1 = sum(
            1 for s in occluded_shapes
            if s.color_name == spec1.color_name and s.shape_type == spec1.shape_type
        )
        count2 = sum(
            1 for s in occluded_shapes
            if s.color_name == spec2.color_name and s.shape_type == spec2.shape_type
        )

        if (1 <= count1 <= max_answer) and (1 <= count2 <= max_answer):
            template = random.choice(QUESTION_TEMPLATES)
            question = template.format(
                color1=spec1.color_name,
                shape1_plural=get_shape_plural(spec1.shape_type),
                color2=spec2.color_name,
                shape2_plural=get_shape_plural(spec2.shape_type)
            )

            answer_string = f"{count1},{count2}"

            metadata = {
                "target1_color": spec1.color_name,
                "target1_shape": spec1.shape_type,
                "target1_count": count1,
                "target2_color": spec2.color_name,
                "target2_shape": spec2.shape_type,
                "target2_count": count2,
                "total_occluded": len(occluded_shapes),
                "pattern_type": "repeating_grid",
            }

            return question, answer_string, metadata

        attempts += 1

    # Final fallback: pick any two color+shape combos (may not be most common in visible)
    all_colors = list(COLOR_PALETTE.keys())
    all_shapes = SHAPE_TYPES

    color1 = random.choice(all_colors)
    shape1 = random.choice(all_shapes)
    count1 = sum(1 for s in occluded_shapes if s.color_name == color1 and s.shape_type == shape1)
    count1 = max(1, min(count1, max_answer))

    # Pick different second combo
    color2 = random.choice([c for c in all_colors if c != color1] or all_colors)
    shape2 = random.choice([s for s in all_shapes if s != shape1] or all_shapes)
    count2 = sum(1 for s in occluded_shapes if s.color_name == color2 and s.shape_type == shape2)
    count2 = max(1, min(count2, max_answer))

    template = random.choice(QUESTION_TEMPLATES)
    question = template.format(
        color1=color1,
        shape1_plural=get_shape_plural(shape1),
        color2=color2,
        shape2_plural=get_shape_plural(shape2)
    )

    answer_string = f"{count1},{count2}"

    metadata = {
        "target1_color": color1,
        "target1_shape": shape1,
        "target1_count": count1,
        "target2_color": color2,
        "target2_shape": shape2,
        "target2_count": count2,
        "total_occluded": len(occluded_shapes),
        "pattern_type": "repeating_grid",
    }

    return question, answer_string, metadata


def generate_pattern_captcha() -> CaptchaSample:
    """
    Generate one Family A CAPTCHA (occluded repeating pattern).

    CORRECT Design:
    1. VISIBLE AREA: Perfect repeating pattern (easy for humans to see)
    2. OCCLUDED AREA: Pattern is BROKEN (variations injected)
    3. Semi-transparent occluder: Humans squint to see actual shapes
    4. VLMs assume pattern continues → WRONG count
    5. Humans see actual shapes through grey → CORRECT count
    """
    # Sample grid and pattern dimensions
    n_rows = random.randint(MIN_ROWS, MAX_ROWS)
    n_cols = random.randint(MIN_COLS, MAX_COLS)
    base_rows = random.randint(MIN_PATTERN_SIZE, MAX_PATTERN_SIZE)
    base_cols = random.randint(MIN_PATTERN_SIZE, MAX_PATTERN_SIZE)

    # Ensure grid is large enough to show pattern repetition
    # Grid should contain at least 2-3 full pattern repetitions in each direction
    min_required_rows = base_rows * 3
    min_required_cols = base_cols * 3
    n_rows = max(n_rows, min_required_rows)
    n_cols = max(n_cols, min_required_cols)

    # Generate pattern and tile it TO THE ENTIRE GRID (including occluded area)
    base_pattern = sample_base_pattern(base_rows, base_cols)
    grid = tile_pattern_to_grid(base_pattern, n_rows, n_cols)

    # Add occluder FIRST (before breaking pattern)
    occluder_mask = sample_occluder_mask(n_rows, n_cols)

    # Validate: ensure we have visible pattern (at least 40% visible for larger occluders)
    total_cells = n_rows * n_cols
    occluded_cells = np.sum(occluder_mask)
    visible_ratio = (total_cells - occluded_cells) / total_cells

    # If too much is occluded, retry with smaller occluder
    if visible_ratio < 0.4:
        occluder_mask = sample_occluder_mask(n_rows, n_cols)
        occluded_cells = np.sum(occluder_mask)
        visible_ratio = (total_cells - occluded_cells) / total_cells

    # **ANTI-VLM WEAPON**: ALWAYS break pattern ONLY in occluded area
    # Visible area maintains perfect pattern
    break_type = "none"
    if USE_PATTERN_BREAKS:
        grid, break_type = inject_pattern_breaks(grid, occluder_mask)

    # Generate question (counts ACTUAL shapes in occluded area, not predicted pattern)
    question, answer, metadata = choose_pattern_question(grid, occluder_mask)

    # Render image with optional semi-transparency
    image = render_pattern_grid(
        grid,
        occluder_mask,
        IMAGE_SIZE,
        use_semi_transparent=USE_SEMI_TRANSPARENT,
        opacity=OCCLUDER_OPACITY
    )

    metadata.update({
        "grid_size": [n_rows, n_cols],
        "pattern_size": [base_rows, base_cols],
        "visible_ratio": visible_ratio,
        "pattern_break_type": break_type,
        "semi_transparent": USE_SEMI_TRANSPARENT,
        "occluder_opacity": OCCLUDER_OPACITY if USE_SEMI_TRANSPARENT else 1.0,
    })

    return CaptchaSample(
        image=image,
        question=question,
        answer=answer,
        puzzle_type="pattern",
        metadata=metadata
    )


# -----------------------------
# Family B: Layered stacks (simplified for now)
# -----------------------------

def generate_layered_captcha() -> CaptchaSample:
    """
    Generate one Family B CAPTCHA (layered shapes with depth reasoning).

    This is a simplified version that still uses grid structure but adds
    explicit layer ordering within some cells.

    Future extension: full free-form layered rendering without grid constraints.
    """
    # For now, use similar grid structure but with layer metadata
    n_rows = random.randint(MIN_ROWS, MAX_ROWS)
    n_cols = random.randint(MIN_COLS, MAX_COLS)

    # Generate random shapes (not tiled pattern)
    grid = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            shape_type = random.choice(SHAPE_TYPES)
            color_name = random.choice(list(COLOR_PALETTE.keys()))
            color_rgb = COLOR_PALETTE[color_name]
            layer = random.randint(0, 2)  # 0=back, 2=front
            row.append(ShapeSpec(
                shape_type=shape_type,
                color_name=color_name,
                color_rgb=color_rgb,
                row=r,
                col=c,
                layer=layer,
            ))
        grid.append(row)

    # For simplicity, use same occluder and rendering
    # (In future, render with actual z-order compositing)
    occluder_mask = sample_occluder_mask(n_rows, n_cols)
    question, answer, metadata = choose_pattern_question(grid, occluder_mask)
    image = render_pattern_grid(
        grid,
        occluder_mask,
        IMAGE_SIZE,
        use_semi_transparent=USE_SEMI_TRANSPARENT,
        opacity=OCCLUDER_OPACITY
    )

    metadata["pattern_type"] = "layered"

    return CaptchaSample(
        image=image,
        question=question,
        answer=answer,
        puzzle_type="layered",
        metadata=metadata
    )


# -----------------------------
# Main generation pipeline
# -----------------------------

def generate_full_dataset():
    """
    Generate complete CAPTCHA dataset with cell_pool and ground_truth JSON files.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {NUM_PUZZLES} occluded pattern counting CAPTCHAs...")
    print(f"Output directory: {OUTPUT_DIR}")

    cell_pool = {}
    ground_truth = {}

    for i in range(NUM_PUZZLES):
        # Mix of pattern and layered types (80% pattern, 20% layered)
        if random.random() < 0.8:
            sample = generate_pattern_captcha()
        else:
            sample = generate_layered_captcha()

        # Save image
        filename = f"puzzle_{i:03d}.png"
        image_path = OUTPUT_DIR / filename
        sample.image.save(image_path)

        # Add to cell pool
        cell_id = f"cell_{i:03d}"
        cell_pool[cell_id] = {
            "filename": filename,
            "puzzle_type": sample.puzzle_type,
            **sample.metadata
        }

        # Add to ground truth
        puzzle_id = f"occluded_pattern_counting_{i:04d}"
        ground_truth[puzzle_id] = {
            "prompt": sample.question,
            "description": f"Occluded pattern counting task ({sample.puzzle_type})",
            "cell_id": cell_id,
            "answer": sample.answer,
            "input_type": "number",
            "metadata": sample.metadata,
        }

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{NUM_PUZZLES} puzzles...")

    # Save JSON files
    cell_pool_path = OUTPUT_DIR / "cell_pool.json"
    ground_truth_path = OUTPUT_DIR / "ground_truth.json"

    with open(cell_pool_path, 'w') as f:
        json.dump(cell_pool, f, indent=2)

    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Dataset generation complete!")
    print(f"  - Generated {NUM_PUZZLES} images")
    print(f"  - Saved cell_pool.json ({len(cell_pool)} entries)")
    print(f"  - Saved ground_truth.json ({len(ground_truth)} entries)")
    print(f"\nOutput location: {OUTPUT_DIR.absolute()}")

    # Print some example questions
    print("\nExample questions:")
    for i, (puzzle_id, data) in enumerate(list(ground_truth.items())[:5]):
        print(f"  {i+1}. {data['prompt']} (Answer: {data['answer']})")


if __name__ == "__main__":
    generate_full_dataset()
