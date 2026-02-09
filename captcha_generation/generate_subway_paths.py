#!/usr/bin/env python3
"""
Legend-Gated Subway Paths CAPTCHA Generator

A subway map where valid paths are determined by semantic icons on edges,
not by line color. Users must count paths using only edges with specific
category icons (e.g., "only animal stamps").

This defeats Python+CV attacks because:
- Path-finding is geometrically trivial
- But determining WHICH edges are valid requires semantic icon understanding
- VLMs struggle with tiny icon classification
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import random
import math
from typing import List, Tuple, Dict, Set, Optional
from itertools import permutations
import colorsys

# Configuration
CELL_SIZE = 600  # Larger resolution for better spacing
NUM_POOL_IMAGES = 100  # More images for better rule coverage
NUM_PUZZLES = 20
GRID_SIZE = 3  # 3x3 grid for better readability
OUTPUT_DIR = Path(__file__).parent.parent / 'captcha_data' / 'Subway_Paths'

# Graph parameters
MIN_STATIONS = 4
MAX_STATIONS = 6
MIN_EDGES = 4
MAX_EDGES = 7  # Moderate complexity for interesting puzzles

# Icon categories with REAL EMOJIS
# Removed ambiguous emojis: 🦋 (butterfly - some may not see it as animal), 🍃 (leaves - doesn't fit shape)
ICON_CATEGORIES = {
    'animal': ['🐕', '🐈', '🐦', '🐟', '🐘', '🐰', '🐢', '🐻'],
    'fruit': ['🍎', '🍌', '🍊', '🍇', '🍒', '🍋', '🍓', '🍐'],
    'vehicle': ['🚗', '🚌', '🚲', '⛵', '✈️', '🚂', '🚚', '🚀'],
    'tool': ['🔨', '🔧', '✂️', '🖌️', '🔑', '✏️', '📏', '🪚'],
    'shape': ['🔷', '🔶', '🔵', '🟠', '🟣', '🔴', '🟡', '🟢'],
}

# Station names pool
STATION_NAMES = ['Park', 'Zoo', 'Mall', 'Library', 'Museum', 'Beach', 'School', 'Hospital',
                 'Station', 'Market', 'Plaza', 'Tower', 'Garden', 'Arena', 'Center', 'Port']

# Colors for subway lines - DISTINCT high-contrast colors
# These are chosen to be maximally distinguishable from each other
LINE_COLORS = [
    (220, 50, 50),    # Red
    (50, 150, 50),    # Green
    (50, 100, 200),   # Blue
    (200, 130, 0),    # Orange
    (150, 50, 180),   # Purple
    (180, 50, 120),   # Magenta/Pink
    (100, 100, 100),  # Gray
    (0, 160, 160),    # Teal/Cyan
]

BACKGROUND_COLOR = (250, 248, 245)
STATION_COLOR = (40, 40, 40)
TEXT_COLOR = (30, 30, 30)


def get_emoji_font(size=64):
    """
    Get a font that can render emojis.
    Apple Color Emoji only supports specific sizes: 20, 32, 40, 48, 64, 96, 160
    """
    valid_sizes = [20, 32, 40, 48, 64, 96, 160]
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


def draw_emoji(img: Image.Image, emoji: str, center: Tuple[int, int],
               size: int = 44, edge_color: Tuple[int, int, int] = None) -> Image.Image:
    """
    Draw an emoji at the specified position using Apple Color Emoji font.
    Returns the modified image (may be converted to RGBA).

    If edge_color is provided, the circle outline will match the edge color
    while the fill remains white.
    """
    x, y = center

    # Convert main image to RGBA if needed
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Draw background circle - always white fill, outline matches edge color
    draw = ImageDraw.Draw(img)
    r = size // 2 + 8

    fill_color = (255, 255, 255)  # Always white fill
    if edge_color:
        outline_color = edge_color  # Outline matches edge color
    else:
        outline_color = (80, 80, 80)

    draw.ellipse([x - r, y - r, x + r, y + r],
                 fill=fill_color, outline=outline_color, width=5)

    # Get emoji font
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
        temp_draw.text((temp_size // 4, temp_size // 4), emoji, font=font, embedded_color=True)

    # Paste onto main image centered at (x, y)
    paste_x = int(x - temp_size // 2)
    paste_y = int(y - temp_size // 2)
    img.paste(temp_img, (paste_x, paste_y), temp_img)

    return img


def generate_subway_graph(num_stations: int, num_edges: int) -> Tuple[List[str], List[Tuple[int, int, dict]]]:
    """
    Generate a connected planar-ish subway graph.
    Returns (station_ids, edges) where each edge has metadata.
    """
    # Create station IDs (A, B, C, D, E, F)
    station_ids = [chr(ord('A') + i) for i in range(num_stations)]

    # Start with a chain to ensure connectivity
    edges = []
    for i in range(num_stations - 1):
        edges.append((i, i + 1))

    # Add extra edges for alternative routes (but avoid too much complexity)
    possible_extras = []
    for i in range(num_stations):
        for j in range(i + 2, num_stations):  # Skip adjacent (already connected)
            if (i, j) not in edges and (j, i) not in edges:
                possible_extras.append((i, j))

    # Add some extra edges
    num_extra = min(num_edges - (num_stations - 1), len(possible_extras))
    if num_extra > 0:
        extra_edges = random.sample(possible_extras, num_extra)
        edges.extend(extra_edges)

    # Assign categories and icons to edges
    # Use cycling colors to ensure each edge has a distinct color
    all_categories = list(ICON_CATEGORIES.keys())
    edge_data = []

    for idx, (u, v) in enumerate(edges):
        category = random.choice(all_categories)
        icon = random.choice(ICON_CATEGORIES[category])
        # Cycle through colors to ensure distinct colors for each edge
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        edge_data.append((u, v, {
            'category': category,
            'icon': icon,
            'color': color
        }))

    return station_ids, edge_data


def layout_stations(num_stations: int, canvas_size: int, min_station_distance: int = 140) -> List[Tuple[int, int]]:
    """
    Generate station positions spread across the FULL canvas.
    Ensures stations are well-separated with room for edge icons between them.

    min_station_distance: minimum pixel distance between any two stations
    """
    # Margin from canvas edge - must be large enough for curves and icons
    margin = 120
    usable = canvas_size - 2 * margin

    # Spread stations - avoid edges (0.1 to 0.9 range to keep icons fully visible)
    if num_stations == 4:
        arrangements = [
            # Wide square - corners (inset from edges)
            [(0.1, 0.15), (0.9, 0.15), (0.1, 0.85), (0.9, 0.85)],
            # Wide diamond
            [(0.5, 0.1), (0.1, 0.5), (0.9, 0.5), (0.5, 0.9)],
            # Wide zigzag
            [(0.1, 0.25), (0.35, 0.75), (0.65, 0.25), (0.9, 0.75)],
            # Wide horizontal
            [(0.1, 0.4), (0.35, 0.6), (0.65, 0.4), (0.9, 0.6)],
        ]
    elif num_stations == 5:
        arrangements = [
            # Pentagon spread
            [(0.5, 0.1), (0.1, 0.35), (0.9, 0.35), (0.2, 0.85), (0.8, 0.85)],
            # X with center
            [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)],
            # Wide W
            [(0.1, 0.25), (0.3, 0.75), (0.5, 0.3), (0.7, 0.75), (0.9, 0.25)],
        ]
    else:  # 6 stations
        arrangements = [
            # Hexagon spread
            [(0.5, 0.1), (0.1, 0.3), (0.9, 0.3), (0.1, 0.7), (0.9, 0.7), (0.5, 0.9)],
            # 2x3 grid spread
            [(0.1, 0.2), (0.5, 0.2), (0.9, 0.2), (0.1, 0.8), (0.5, 0.8), (0.9, 0.8)],
            # 3x2 grid spread
            [(0.1, 0.35), (0.1, 0.65), (0.5, 0.35), (0.5, 0.65), (0.9, 0.35), (0.9, 0.65)],
        ]

    # Try arrangements until we find one with sufficient spacing
    random.shuffle(arrangements)

    for base in arrangements:
        # Try multiple times with different jitter
        for _ in range(5):
            positions = []
            for bx, by in base:
                jx = random.uniform(-0.03, 0.03)
                jy = random.uniform(-0.03, 0.03)
                x = int(margin + max(0, min(1, bx + jx)) * usable)
                y = int(margin + max(0, min(1, by + jy)) * usable)
                positions.append((x, y))

            # Check minimum distance between all station pairs
            valid = True
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = math.sqrt((positions[i][0] - positions[j][0])**2 +
                                    (positions[i][1] - positions[j][1])**2)
                    if dist < min_station_distance:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                return positions

    # Fallback: use first arrangement without jitter (should always be valid)
    positions = []
    base = arrangements[0]
    for bx, by in base:
        x = int(margin + bx * usable)
        y = int(margin + by * usable)
        positions.append((x, y))

    return positions


def find_all_simple_paths(edges: List[Tuple[int, int, dict]],
                          valid_categories: Set[str],
                          start: int, end: int,
                          num_stations: int,
                          max_length: int = 6) -> List[List[int]]:
    """
    Find all simple paths from start to end using only edges with valid categories.
    """
    # Build adjacency list for valid edges only
    adj = {i: [] for i in range(num_stations)}
    for u, v, data in edges:
        if data['category'] in valid_categories:
            adj[u].append(v)
            adj[v].append(u)

    paths = []

    def dfs(current: int, target: int, visited: Set[int], path: List[int]):
        if len(path) > max_length:
            return
        if current == target:
            paths.append(path.copy())
            return
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, target, visited, path)
                path.pop()
                visited.remove(neighbor)

    dfs(start, end, {start}, [start])
    return paths


def generate_rule() -> Tuple[str, Set[str], str]:
    """
    Generate a rule for valid ticket types.
    Returns (rule_text, valid_categories, rule_type).
    """
    all_cats = list(ICON_CATEGORIES.keys())

    rule_types = ['single', 'two_allowed', 'one_not']
    rule_type = random.choice(rule_types)

    if rule_type == 'single':
        cat = random.choice(all_cats)
        cat_upper = cat.upper()
        rule_text = f"only {cat_upper} stamps"
        valid_cats = {cat}

    elif rule_type == 'two_allowed':
        cats = random.sample(all_cats, 2)
        rule_text = f"{cats[0].upper()} or {cats[1].upper()} stamps"
        valid_cats = set(cats)

    else:  # one_not
        forbidden = random.choice(all_cats)
        rule_text = f"any stamp EXCEPT {forbidden.upper()}"
        valid_cats = set(all_cats) - {forbidden}

    return rule_text, valid_cats, rule_type


def draw_curved_line(draw: ImageDraw, p1: Tuple[int, int], p2: Tuple[int, int],
                     color: Tuple[int, int, int], width: int = 4,
                     curve_direction: int = 0,
                     avoid_stations: List[Tuple[int, int]] = None,
                     min_station_clearance: int = 100) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Draw a curved subway line between two points.
    Returns (midpoint, all_curve_points) for flexible icon placement.

    curve_direction: -1 (left), 0 (auto), 1 (right) - controls which side the curve bends
    avoid_stations: list of station positions that this edge should NOT pass through
    min_station_clearance: minimum distance curve should maintain from avoided stations
    """
    x1, y1 = p1
    x2, y2 = p2

    # Calculate midpoint
    mx = (x1 + x2) // 2
    my = (y1 + y2) // 2

    # Calculate perpendicular direction for curving
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx*dx + dy*dy)

    if length == 0:
        cx, cy = mx, my
    else:
        px, py = -dy / length, dx / length

        # Try different curve amounts to find one that avoids intermediate stations
        best_curve = None
        best_min_dist = -1

        # Try MORE aggressive curve options to better avoid stations
        if curve_direction == 0:
            curve_options = [
                random.uniform(-30, 30),  # Small random curve
                80, -80, 120, -120, 150, -150,  # Larger curves
                60, -60, 100, -100, 40, -40,
                0,  # Straight line as last resort
            ]
        else:
            base = 60 * curve_direction  # More aggressive base curve
            curve_options = [
                base + random.uniform(-10, 10),
                base + 40, base + 80, base + 120,  # More aggressive options
                base - 20, base + 60, base + 100,
            ]

        for curve in curve_options:
            test_cx = int(mx + px * curve)
            test_cy = int(my + py * curve)

            # Generate test curve points
            test_points = []
            for i in range(21):  # Fewer points for testing
                t = i / 20
                bx = (1-t)**2 * x1 + 2*(1-t)*t * test_cx + t**2 * x2
                by = (1-t)**2 * y1 + 2*(1-t)*t * test_cy + t**2 * y2
                test_points.append((int(bx), int(by)))

            # Check minimum distance to avoided stations
            min_dist = float('inf')
            if avoid_stations:
                for pt in test_points[2:-2]:  # Skip points very close to endpoints
                    for station in avoid_stations:
                        dist = math.sqrt((pt[0] - station[0])**2 + (pt[1] - station[1])**2)
                        min_dist = min(min_dist, dist)

            # Track best option (maximizes distance from avoided stations)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_curve = curve

            # If good enough, stop searching
            if min_dist >= min_station_clearance:
                break

        cx = int(mx + px * best_curve)
        cy = int(my + py * best_curve)

    # Draw as bezier curve using MORE line segments (41 points for finer control)
    num_points = 41
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        # Quadratic bezier
        bx = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
        by = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
        points.append((int(bx), int(by)))

    # Draw the line segments
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=color, width=width)

    # Return midpoint of curve and all points for flexible placement
    return points[num_points // 2], points


def find_non_overlapping_position(curve_points: List[Tuple[int, int]],
                                   placed_positions: List[Tuple[int, int]],
                                   station_positions: List[Tuple[int, int]],
                                   other_curves: List[List[Tuple[int, int]]],
                                   min_icon_spacing: int = 80,
                                   min_station_spacing: int = 110,
                                   min_curve_spacing: int = 50,
                                   canvas_size: int = 600) -> Tuple[int, int]:
    """
    Find a position along the curve that doesn't overlap with:
    - Other icons (placed_positions)
    - Station circles/letters (station_positions)
    - Other edge curves (other_curves)
    - Canvas boundaries (icons must stay within bounds)

    First tries middle 40% of curve, then expands to 20%-80% if needed.
    """
    n = len(curve_points)  # Should be 41

    # Icon visual radius (size/2 + 8 = 22 + 8 = 30) plus generous safety margin
    ICON_RADIUS = 45

    # Try progressively wider ranges until we find a good spot
    # Start with tight center to avoid ambiguity near vertices
    search_ranges = [
        (0.40, 0.60),  # Middle 20% first - far from vertices
        (0.35, 0.65),  # Slightly wider
        (0.30, 0.70),  # Middle 40%
        (0.25, 0.75),  # Wider range if middle is crowded
    ]

    best_pos = None
    best_min_clearance = float('-inf')

    for range_start, range_end in search_ranges:
        start_idx = int(n * range_start)
        end_idx = int(n * range_end)

        for idx in range(start_idx, end_idx + 1):
            pos = curve_points[idx]
            min_clearance = float('inf')

            # Check canvas bounds - icon must stay fully within canvas
            x, y = pos
            left_clearance = x - ICON_RADIUS
            right_clearance = (canvas_size - x) - ICON_RADIUS
            top_clearance = y - ICON_RADIUS
            bottom_clearance = (canvas_size - y) - ICON_RADIUS
            bounds_clearance = min(left_clearance, right_clearance, top_clearance, bottom_clearance)
            min_clearance = min(min_clearance, bounds_clearance)

            # Check distance to existing icons - MUST be far apart
            for existing in placed_positions:
                dist = math.sqrt((pos[0] - existing[0])**2 + (pos[1] - existing[1])**2)
                clearance = dist - min_icon_spacing
                min_clearance = min(min_clearance, clearance)

            # Check distance to stations - MUST be very far
            for station in station_positions:
                dist = math.sqrt((pos[0] - station[0])**2 + (pos[1] - station[1])**2)
                clearance = dist - min_station_spacing
                min_clearance = min(min_clearance, clearance)

            # Check distance to OTHER curves (not this one) - avoid placing near other edges
            for other_curve in other_curves:
                for other_pt in other_curve:
                    dist = math.sqrt((pos[0] - other_pt[0])**2 + (pos[1] - other_pt[1])**2)
                    clearance = dist - min_curve_spacing
                    min_clearance = min(min_clearance, clearance)

            # Track position with maximum minimum clearance
            if min_clearance > best_min_clearance:
                best_min_clearance = min_clearance
                best_pos = pos

        # If we found a good spot (positive clearance), stop expanding search
        if best_min_clearance > 0:
            break

    return best_pos if best_pos else curve_points[n // 2]


def validate_layout(icon_positions: List[Tuple[int, int]],
                    station_positions: List[Tuple[int, int]],
                    all_curves: List[List[Tuple[int, int]]],
                    canvas_size: int = 600) -> bool:
    """
    Validate that the layout has no overlaps and icons are within bounds.
    Returns True if layout is valid (no overlaps), False otherwise.
    """
    MIN_ICON_DIST = 80    # Two icons must be at least 80px apart
    MIN_STATION_DIST = 70  # Icon must be 70px from any station
    MIN_CURVE_DIST = 45    # Icon must be 45px from OTHER curves
    ICON_RADIUS = 45      # Icon visual radius plus generous safety margin

    for i, pos in enumerate(icon_positions):
        x, y = pos

        # Check canvas bounds - icon must be fully inside canvas
        if x < ICON_RADIUS or x > canvas_size - ICON_RADIUS:
            return False
        if y < ICON_RADIUS or y > canvas_size - ICON_RADIUS:
            return False

        # Check against other icons
        for j, other_pos in enumerate(icon_positions):
            if i != j:
                dist = math.sqrt((pos[0] - other_pos[0])**2 + (pos[1] - other_pos[1])**2)
                if dist < MIN_ICON_DIST:
                    return False

        # Check against stations
        for station in station_positions:
            dist = math.sqrt((pos[0] - station[0])**2 + (pos[1] - station[1])**2)
            if dist < MIN_STATION_DIST:
                return False

        # Check against OTHER curves (not this icon's own curve)
        for j, curve in enumerate(all_curves):
            if i != j:  # Skip own curve
                for pt in curve:
                    dist = math.sqrt((pos[0] - pt[0])**2 + (pos[1] - pt[1])**2)
                    if dist < MIN_CURVE_DIST:
                        return False

    return True


def render_subway_map(station_ids: List[str],
                      edges: List[Tuple[int, int, dict]],
                      positions: List[Tuple[int, int]],
                      rule_text: str,
                      valid_categories: Set[str],
                      start_idx: int,
                      end_idx: int,
                      canvas_size: int = 500) -> Tuple[Image.Image, bool]:
    """
    Render the complete subway map image.
    Returns (image, success). success=False means overlaps detected.
    """
    img = Image.new('RGB', (canvas_size, canvas_size), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    # LARGER fonts for better readability
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    # Assign curve directions to prevent edge overlap at shared vertices
    # For each vertex, edges should alternate curve direction
    vertex_edge_count = {}  # vertex -> count of edges using it
    edge_curve_dirs = []    # curve direction for each edge

    for u, v, data in edges:
        # Get current count for both vertices
        count_u = vertex_edge_count.get(u, 0)
        count_v = vertex_edge_count.get(v, 0)

        # Alternate direction based on edge count at the vertex with more edges
        max_count = max(count_u, count_v)
        direction = 1 if max_count % 2 == 0 else -1
        edge_curve_dirs.append(direction)

        # Increment counts
        vertex_edge_count[u] = count_u + 1
        vertex_edge_count[v] = count_v + 1

    # Helper function to check if two curves run too close (parallel) for too long
    def curves_run_parallel(curve1: List[Tuple[int, int]], curve2: List[Tuple[int, int]],
                            min_dist: int = 70, max_close_points: int = 5) -> bool:
        """
        Check if two curves run parallel (too close for too many consecutive points).
        This catches cases where edges share a vertex and run alongside each other.
        More aggressive threshold to force curves to spread apart.
        """
        n1, n2 = len(curve1), len(curve2)
        # Check full curves (skip only endpoints which may legitimately touch at shared vertices)
        close_count = 0
        for i in range(3, n1 - 3):
            # Find minimum distance from this point to any point on curve2
            min_d = float('inf')
            for j in range(3, n2 - 3):
                dist = math.sqrt((curve1[i][0] - curve2[j][0])**2 +
                               (curve1[i][1] - curve2[j][1])**2)
                min_d = min(min_d, dist)
            if min_d < min_dist:
                close_count += 1
                if close_count >= max_close_points:
                    return True
            else:
                close_count = 0  # Reset if we found a point that's far enough
        return False

    # Helper to compute curve without drawing
    def compute_curve(p1, p2, curve_amt):
        x1, y1 = p1
        x2, y2 = p2
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return [(mx, my)] * 41
        px, py = -dy / length, dx / length
        cx, cy = int(mx + px * curve_amt), int(my + py * curve_amt)
        points = []
        for i in range(41):
            t = i / 40
            bx = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
            by = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
            points.append((int(bx), int(by)))
        return points

    # Icon area constant - icons have visual radius ~30px + edge width 6px + buffer
    ICON_SAFE_RADIUS = 45

    # Helper to select best curve considering all constraints
    def select_best_curve(p1, p2, base_dir, avoid_curves, avoid_stations, avoid_icons):
        best_curve = None
        best_score = float('-inf')
        best_curve_amt = 80 * base_dir

        curve_amounts = [
            80 * base_dir, 80 * -base_dir,
            120 * base_dir, 120 * -base_dir,
            50 * base_dir, 50 * -base_dir,
            150 * base_dir, 150 * -base_dir,
            30 * base_dir, 30 * -base_dir,    # Gentler curves
            180 * base_dir, 180 * -base_dir,  # More extreme curves
            100 * base_dir, 100 * -base_dir,  # Medium curves
        ]

        for curve_amt in curve_amounts:
            test_curve = compute_curve(p1, p2, curve_amt)

            # Check 1: Parallel running with existing curves
            parallel_count = sum(1 for c in avoid_curves
                               if curves_run_parallel(test_curve, c, min_dist=70, max_close_points=5))

            # Check 2: Station avoidance
            station_ok = True
            min_station_dist = float('inf')
            for pt in test_curve[5:-5]:
                for station in avoid_stations:
                    d = math.sqrt((pt[0]-station[0])**2 + (pt[1]-station[1])**2)
                    min_station_dist = min(min_station_dist, d)
                    if d < 50:
                        station_ok = False

            # Check 3: Middle section clearance from other curves
            middle_clearance = float('inf')
            n_test = len(test_curve)
            mid_start, mid_end = int(n_test * 0.25), int(n_test * 0.75)
            for i in range(mid_start, mid_end):
                pt = test_curve[i]
                for existing_curve in avoid_curves:
                    for existing_pt in existing_curve:
                        d = math.sqrt((pt[0]-existing_pt[0])**2 + (pt[1]-existing_pt[1])**2)
                        middle_clearance = min(middle_clearance, d)

            # Check 4: Does curve pass through ANY icon position?
            min_icon_dist = float('inf')
            for icon_pos in avoid_icons:
                for pt in test_curve:
                    d = math.sqrt((pt[0]-icon_pos[0])**2 + (pt[1]-icon_pos[1])**2)
                    min_icon_dist = min(min_icon_dist, d)

            # Score with penalties
            if not station_ok:
                score = -5000
            elif min_icon_dist < ICON_SAFE_RADIUS:
                score = -3000 + min_icon_dist  # Very bad - curve goes through icon
            elif middle_clearance < 50:
                score = -1000 + middle_clearance
            else:
                score = min_icon_dist + middle_clearance - parallel_count * 150 + min_station_dist * 0.1

            if score > best_score:
                best_score = score
                best_curve = test_curve
                best_curve_amt = curve_amt

        return best_curve, best_curve_amt

    # ==================== PASS 1: Compute curves and icon positions (no drawing) ====================
    temp_curves = []
    all_icon_positions = []
    edge_data_list = []  # Store data for each edge

    for idx, (u, v, data) in enumerate(edges):
        p1, p2 = positions[u], positions[v]
        base_direction = edge_curve_dirs[idx]
        avoid_stations = [positions[i] for i in range(len(positions)) if i != u and i != v]

        # Compute best curve (only avoiding other curves and stations, not icons yet)
        best_curve, _ = select_best_curve(p1, p2, base_direction, temp_curves, avoid_stations, [])
        temp_curves.append(best_curve)

        # Compute where icon WOULD be placed
        other_curves = [c for i, c in enumerate(temp_curves) if i != idx]
        icon_pos = find_non_overlapping_position(
            best_curve, all_icon_positions, positions, other_curves,
            min_icon_spacing=100, min_station_spacing=140, min_curve_spacing=60,
            canvas_size=canvas_size
        )
        all_icon_positions.append(icon_pos)
        edge_data_list.append((u, v, data, p1, p2, base_direction))

    # ==================== PASS 2: Re-compute curves avoiding ALL icons, then draw ====================
    final_curves = []
    edge_curves = []

    for idx, (u, v, data, p1, p2, base_direction) in enumerate(edge_data_list):
        avoid_stations = [positions[i] for i in range(len(positions)) if i != u and i != v]

        # Now select curve that avoids ALL icon positions
        best_curve, best_curve_amt = select_best_curve(
            p1, p2, base_direction, final_curves, avoid_stations, all_icon_positions
        )

        # Draw THIS curve directly (not a recalculated one from draw_curved_line!)
        # This is critical: select_best_curve() computed a curve avoiding icons,
        # but draw_curved_line() would recalculate and ignore icons entirely.
        for i in range(len(best_curve) - 1):
            draw.line([best_curve[i], best_curve[i+1]], fill=data['color'], width=6)

        final_curves.append(best_curve)
        edge_curves.append((best_curve, data['icon'], data['category'], data['color']))

    # ==================== PASS 3: RE-COMPUTE icon positions using FINAL curves ====================
    # Critical: Pass 1 icon positions were based on temp_curves, but curves changed in Pass 2!
    # We must re-compute ALL icon positions using the final curves to avoid overlaps.
    placed_icon_positions = []
    for idx, (curve_points, icon, category, edge_color) in enumerate(edge_curves):
        # Get all OTHER final curves (not this edge's curve)
        other_curves = [c for i, c in enumerate(final_curves) if i != idx]

        # ALWAYS re-compute icon position using final curves
        # This ensures icons avoid the ACTUAL curve positions, not the preliminary ones
        icon_pos = find_non_overlapping_position(
            curve_points, placed_icon_positions, positions, other_curves,
            min_icon_spacing=100, min_station_spacing=140, min_curve_spacing=60,
            canvas_size=canvas_size
        )

        placed_icon_positions.append(icon_pos)
        img = draw_emoji(img, icon, icon_pos, size=44, edge_color=edge_color)

    # Validate layout - check for overlaps and bounds
    layout_valid = validate_layout(placed_icon_positions, positions, final_curves, canvas_size)

    # ALWAYS draw stations (even if layout invalid - we still need complete image)
    # Recreate draw object since img may have been converted to RGBA
    draw = ImageDraw.Draw(img)

    # Must convert to RGB for drawing if image is RGBA
    if img.mode == 'RGBA':
        # Create a new RGB image and paste RGBA onto it
        rgb_img = Image.new('RGB', img.size, BACKGROUND_COLOR)
        rgb_img.paste(img, mask=img.split()[3])  # Use alpha as mask
        img = rgb_img
        draw = ImageDraw.Draw(img)

    station_radius = 26
    for i, (x, y) in enumerate(positions):
        is_start = (i == start_idx)
        is_end = (i == end_idx)

        if is_start:
            fill_color = (50, 180, 50)  # Green for start
        elif is_end:
            fill_color = (220, 50, 50)  # Red for end
        else:
            fill_color = (240, 240, 240)  # Light gray - visible against background

        draw.ellipse([x - station_radius, y - station_radius,
                     x + station_radius, y + station_radius],
                    fill=fill_color, outline=STATION_COLOR, width=4)

        # Station label - LARGER font
        label = station_ids[i]
        bbox = draw.textbbox((0, 0), label, font=font_large)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_color = (255, 255, 255) if (is_start or is_end) else STATION_COLOR
        draw.text((x - tw // 2, y - th // 2), label, fill=text_color, font=font_large)

    # No instruction text in individual cells - will be shown in puzzle prompt instead

    return img, layout_valid


def generate_subway_image(target_paths: int = None) -> Tuple[Image.Image, int, dict, bool]:
    """
    Generate a single subway map image with a counting task.
    Returns (image, num_valid_paths, metadata, success).
    success=True means no overlaps, success=False means layout has issues.
    """
    MAX_LAYOUT_ATTEMPTS = 10

    for layout_attempt in range(MAX_LAYOUT_ATTEMPTS):
        # Generate graph
        num_stations = random.randint(MIN_STATIONS, MAX_STATIONS)
        num_edges = random.randint(MIN_EDGES, min(MAX_EDGES, num_stations * 2))

        station_ids, edges = generate_subway_graph(num_stations, num_edges)
        positions = layout_stations(num_stations, CELL_SIZE)

        # Choose start and end stations (not adjacent for more interesting paths)
        start_idx = 0
        end_idx = num_stations - 1

        # Generate rule and count valid paths
        max_attempts = 20
        for _ in range(max_attempts):
            rule_text, valid_categories, rule_type = generate_rule()
            paths = find_all_simple_paths(edges, valid_categories, start_idx, end_idx, num_stations)
            num_paths = len(paths)

            # Want 1-4 paths for reasonable difficulty
            if 1 <= num_paths <= 4:
                if target_paths is None or num_paths == target_paths:
                    break

        # Render the map and check for overlaps
        img, success = render_subway_map(station_ids, edges, positions, rule_text,
                                          valid_categories, start_idx, end_idx, CELL_SIZE)

        metadata = {
            'num_stations': num_stations,
            'num_edges': len(edges),
            'rule_text': rule_text,
            'rule_type': rule_type,
            'valid_categories': list(valid_categories),
            'start': station_ids[start_idx],
            'end': station_ids[end_idx],
            'num_paths': num_paths
        }

        if success:
            return img, num_paths, metadata, True
        # If not successful, try again with a new graph layout

    # After all attempts, return with success=False
    return img, num_paths, metadata, False


def generate_cell_pool() -> Dict:
    """
    Generate pool of subway map images with varying path counts.
    """
    cell_pool = {}

    # Aim for good distribution of path counts (scaled for larger pool)
    target_distribution = {
        0: 20,  # No valid paths
        1: 30,  # 1 path
        2: 30,  # 2 paths
        3: 25,  # 3 paths
        4: 15,  # 4 paths
    }

    idx = 0
    for target_paths, count in target_distribution.items():
        generated = 0
        attempts = 0
        max_attempts = count * 10

        while generated < count and attempts < max_attempts:
            attempts += 1
            img, num_paths, metadata, success = generate_subway_image()

            # ONLY save if layout is valid (no overlaps)
            if not success:
                continue  # Skip invalid layouts - regenerate

            # Accept if matches target or close enough
            if num_paths == target_paths or (target_paths >= 3 and num_paths >= 3):
                filename = f"subway_{idx:03d}.png"
                img.save(OUTPUT_DIR / filename)

                cell_pool[f"cell_{idx}"] = {
                    "filename": filename,
                    "num_paths": num_paths,
                    **metadata
                }

                idx += 1
                generated += 1

                if idx % 10 == 0:
                    print(f"  Generated {idx} images...")

    # Fill remaining slots if needed
    fill_attempts = 0
    max_fill_attempts = NUM_POOL_IMAGES * 20
    while idx < NUM_POOL_IMAGES and fill_attempts < max_fill_attempts:
        fill_attempts += 1
        img, num_paths, metadata, success = generate_subway_image()

        # ONLY save if layout is valid
        if not success:
            continue

        filename = f"subway_{idx:03d}.png"
        img.save(OUTPUT_DIR / filename)

        cell_pool[f"cell_{idx}"] = {
            "filename": filename,
            "num_paths": num_paths,
            **metadata
        }
        idx += 1

    return cell_pool


def generate_puzzles(cell_pool: Dict) -> Dict:
    """
    Generate puzzle configurations - user must identify how many paths exist.
    For grid selection: select all cells with exactly N valid paths.

    All cells in a puzzle share the same rule (valid categories) so the
    prompt can clearly explain which stamps are allowed.
    """
    ground_truth = {}

    # First, group cells by their rule (valid_categories as a frozenset for grouping)
    cells_by_rule = {}
    for cid, data in cell_pool.items():
        rule_key = frozenset(data['valid_categories'])
        if rule_key not in cells_by_rule:
            cells_by_rule[rule_key] = []
        cells_by_rule[rule_key].append((cid, data))

    print(f"\nCells by rule type:")
    for rule_key in cells_by_rule:
        print(f"  {set(rule_key)}: {len(cells_by_rule[rule_key])} cells")

    cells_per_puzzle = GRID_SIZE * GRID_SIZE  # 9 for 3x3

    # Filter rules with enough cells, then sort for cycling
    min_cells_needed = cells_per_puzzle
    valid_rules = [r for r in cells_by_rule.keys() if len(cells_by_rule[r]) >= min_cells_needed]
    if not valid_rules:
        # Fall back to rules with at least half the needed cells
        valid_rules = [r for r in cells_by_rule.keys() if len(cells_by_rule[r]) >= min_cells_needed // 2]
    if not valid_rules:
        valid_rules = list(cells_by_rule.keys())

    # Sort to ensure consistent cycling order
    sorted_rules = sorted(valid_rules, key=lambda r: (len(cells_by_rule[r]), str(sorted(r))), reverse=True)
    print(f"\nUsing {len(sorted_rules)} rules with enough cells for puzzles")

    for puzzle_idx in range(NUM_PUZZLES):
        # Cycle through valid rules to ensure variety
        rule_idx = puzzle_idx % len(sorted_rules)
        selected_rule = sorted_rules[rule_idx]
        rule_cells = cells_by_rule[selected_rule]

        # Get the rule details from first cell
        sample_data = rule_cells[0][1]
        rule_text = sample_data['rule_text']
        valid_categories = sample_data['valid_categories']

        # Group cells by path count for this rule
        cells_by_path_count = {}
        for cid, data in rule_cells:
            count = data['num_paths']
            if count not in cells_by_path_count:
                cells_by_path_count[count] = []
            cells_by_path_count[count].append(cid)

        # Choose target path count that has 2-4 cells available
        valid_targets = [c for c, cells in cells_by_path_count.items() if 2 <= len(cells) <= 6]
        if not valid_targets:
            # Fall back to any count with at least 2 cells
            valid_targets = [c for c, cells in cells_by_path_count.items() if len(cells) >= 2]
        if not valid_targets:
            print(f"  Warning: No valid target counts for rule {rule_text}, skipping")
            continue

        target_count = random.choice(valid_targets)
        target_cells = cells_by_path_count[target_count]

        # Select 2-4 target cells as correct answers
        num_correct = random.randint(2, min(4, len(target_cells)))
        correct_cells = random.sample(target_cells, num_correct)

        # Get non-target cells to fill the rest
        non_target_cells = [cid for cid, data in rule_cells if data['num_paths'] != target_count]
        num_wrong = cells_per_puzzle - num_correct

        if len(non_target_cells) >= num_wrong:
            wrong_cells = random.sample(non_target_cells, num_wrong)
        else:
            wrong_cells = non_target_cells[:]
            # Pad with duplicates if needed (rare)
            while len(wrong_cells) < num_wrong and non_target_cells:
                wrong_cells.append(random.choice(non_target_cells))

        # Combine and shuffle
        all_cells = correct_cells + wrong_cells
        all_cells = all_cells[:cells_per_puzzle]
        random.shuffle(all_cells)

        # Find answer indices (cells with exactly target_count paths)
        answer_indices = []
        for i, cid in enumerate(all_cells):
            if cell_pool[cid]['num_paths'] == target_count:
                answer_indices.append(i)

        # Build clear, comprehensive prompt
        all_cats = ['animal', 'fruit', 'vehicle', 'tool', 'shape']
        icon_examples = []
        for cat in all_cats:
            icon = ICON_CATEGORIES[cat][0]
            icon_examples.append(f"{cat.upper()}: {icon}")
        icons_str = ' | '.join(icon_examples)

        prompt = (
            f"Select all maps with exactly {target_count} valid route{'s' if target_count != 1 else ''} from GREEN to RED. "
            f"Rules: Use only {rule_text}; no station revisits. "
            f"Edge types: [{icons_str}]"
        )

        ground_truth[f"subway_paths_{puzzle_idx:04d}"] = {
            "prompt": prompt,
            "description": f"Subway paths - {rule_text} - select cells with exactly {target_count} paths",
            "cells": all_cells,
            "answer": sorted(answer_indices),
            "input_type": "subway_paths_select",
            "grid_size": [GRID_SIZE, GRID_SIZE],
            "target_count": target_count,
            "rule_text": rule_text,
            "valid_categories": valid_categories
        }

    return ground_truth


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Subway Paths CAPTCHAs')
    parser.add_argument('--samples', type=int, default=0,
                        help='Generate only N sample images for verification (no puzzles)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Subway Paths CAPTCHA Generator")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")

    if args.samples > 0:
        # Sample mode: generate a few images for verification
        print(f"SAMPLE MODE: Generating {args.samples} sample images")
        print()

        for i in range(args.samples):
            # Keep regenerating until we get a valid image (no overlaps)
            for attempt in range(50):
                img, num_paths, metadata, success = generate_subway_image()
                if success:
                    break
            if not success:
                print(f"Warning: Could not generate valid image for sample {i} after 50 attempts")
            img_path = OUTPUT_DIR / f'sample_{i:03d}.png'
            img.save(img_path)
            print(f"Saved: {img_path} (paths: {num_paths}, valid: {success})")

        print(f"\nGenerated {args.samples} sample images in {OUTPUT_DIR}")
        return

    print(f"Pool size: {NUM_POOL_IMAGES}")
    print(f"Puzzles: {NUM_PUZZLES}")
    print()

    # Generate cell pool
    print("Generating cell pool...")
    cell_pool = generate_cell_pool()

    # Save cell pool
    with open(OUTPUT_DIR / 'cell_pool.json', 'w') as f:
        json.dump(cell_pool, f, indent=2)
    print(f"Saved cell_pool.json")

    # Generate puzzles
    print("\nGenerating puzzles...")
    ground_truth = generate_puzzles(cell_pool)

    # Save ground truth
    with open(OUTPUT_DIR / 'ground_truth.json', 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved ground_truth.json")

    # Print summary
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)

    path_dist = {}
    for cid, data in cell_pool.items():
        count = data['num_paths']
        path_dist[count] = path_dist.get(count, 0) + 1

    print(f"\nPath count distribution:")
    for count in sorted(path_dist.keys()):
        print(f"  {count} paths: {path_dist[count]} images")

    print(f"\nTotal images: {len(cell_pool)}")
    print(f"Total puzzles: {len(ground_truth)}")


if __name__ == "__main__":
    main()
