"""
Illusory Ribbons CAPTCHA Generator

Creates images with weaving ribbon loops that are easy for humans to count
but hard for VLM+Python analysis due to:
- Weaving/occlusion patterns (ribbons go over/under each other)
- Uniform colors (no color-based segmentation per ribbon)
- Slight contour noise to confuse edge detection
- Similar visual appearance across all ribbons

Task: "Select all cells where there are exactly 3 ribbons"
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import json
import random
import math
from typing import List, Tuple, Dict
from scipy.interpolate import splprep, splev

# Configuration
CELL_SIZE = 400  # Larger to fit more ribbons without clustering
NUM_POOL_IMAGES = 80  # Total pool images to generate (more for variation)
NUM_PUZZLES = 20  # Number of CAPTCHA grids
GRID_SIZE = 4  # 4x4 grid
TARGET_RIBBON_COUNTS = [3, 4]  # Vary between "3 ribbons" and "4 ribbons" questions
OUTPUT_DIR = Path('../captcha_data/Illusory_Ribbons/')

# Ribbon parameters
RIBBON_WIDTH = 20  # Thickness of ribbon
MIN_RIBBONS = 2
MAX_RIBBONS = 5

# Color palette - Multiple neon gradient options
# Each ribbon picks its own gradient for variety; weaving still defeats VLM segmentation
# High-contrast complementary gradients for VLM confusion
# Each gradient has STRONG internal transition, but distinct from other gradients
NEON_GRADIENTS = [
    ((255, 50, 50), (50, 255, 255)),     # 0: RED ↔ CYAN (complementary)
    ((255, 255, 50), (150, 50, 255)),    # 1: YELLOW ↔ PURPLE (complementary)
    ((50, 255, 100), (255, 50, 200)),    # 2: GREEN ↔ MAGENTA (complementary)
    ((50, 150, 255), (255, 180, 50)),    # 3: BLUE ↔ ORANGE (complementary)
    ((200, 255, 50), (255, 100, 150)),   # 4: LIME ↔ PINK (high contrast)
]
BACKGROUND_COLOR = (245, 242, 238)  # Warm off-white


def generate_smooth_loop(center: Tuple[float, float],
                         base_radius: float,
                         num_points: int = 80,
                         orientation: float = None) -> List[Tuple[float, float]]:
    """
    Generate a smooth deformed closed loop using Fourier-like perturbations.

    Args:
        orientation: Rotation angle in radians. If None, random.
                    Used to ensure overlapping ribbons cross at angles.
    """
    points = []

    # Generate random Fourier coefficients for organic deformation
    num_harmonics = random.randint(2, 4)
    amplitudes = [random.uniform(0.08, 0.25) for _ in range(num_harmonics)]
    phases = [random.uniform(0, 2 * math.pi) for _ in range(num_harmonics)]
    frequencies = [random.randint(2, 4) for _ in range(num_harmonics)]

    # Use provided orientation or random
    if orientation is None:
        orientation = random.uniform(0, 2 * math.pi)

    for i in range(num_points):
        angle = 2 * math.pi * i / num_points + orientation

        # Base radius with smooth deformation
        r = base_radius
        for amp, phase, freq in zip(amplitudes, phases, frequencies):
            r += amp * base_radius * math.sin(freq * angle + phase)

        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        points.append((x, y))

    return points


def smooth_spline(points: List[Tuple[float, float]],
                  num_output: int = 200) -> List[Tuple[float, float]]:
    """
    Smooth curve using B-spline interpolation.
    """
    if len(points) < 4:
        return points

    # Close the loop by repeating first points
    pts = points + points[:3]
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]

    try:
        tck, u = splprep([x, y], s=len(points), per=True)
        u_new = np.linspace(0, 1, num_output)
        x_new, y_new = splev(u_new, tck)
        return list(zip(x_new, y_new))
    except:
        return points


def get_ribbon_polygon(centerline: List[Tuple[float, float]],
                       width: float) -> List[Tuple[float, float]]:
    """
    Convert centerline to ribbon polygon (closed shape).
    """
    n = len(centerline)
    left_edge = []
    right_edge = []

    for i in range(n):
        # Get tangent direction using central differences
        p_prev = centerline[(i - 1) % n]
        p_next = centerline[(i + 1) % n]

        tx = p_next[0] - p_prev[0]
        ty = p_next[1] - p_prev[1]

        # Normalize
        length = math.sqrt(tx * tx + ty * ty)
        if length > 0.001:
            tx /= length
            ty /= length
        else:
            tx, ty = 1, 0

        # Normal is perpendicular
        nx, ny = -ty, tx

        # Offset points
        half_w = width / 2
        cx, cy = centerline[i]
        left_edge.append((cx + nx * half_w, cy + ny * half_w))
        right_edge.append((cx - nx * half_w, cy - ny * half_w))

    # Create closed polygon: left edge forward, right edge backward
    polygon = left_edge + right_edge[::-1]
    return polygon


def find_crossing_regions(loop1: List[Tuple[float, float]],
                          loop2: List[Tuple[float, float]],
                          threshold: float = 40) -> List[Tuple[int, int, int, int]]:
    """
    Find regions where two loops cross (for weaving effect).
    Returns list of (start1, end1, start2, end2) index ranges.
    """
    crossings = []
    n1, n2 = len(loop1), len(loop2)

    in_crossing = False
    crossing_start = None

    for i in range(n1):
        p1 = loop1[i]
        # Find closest point on loop2
        min_dist = float('inf')
        closest_j = 0
        for j in range(n2):
            p2 = loop2[j]
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_j = j

        if min_dist < threshold:
            if not in_crossing:
                crossing_start = (i, closest_j)
                in_crossing = True
        else:
            if in_crossing:
                crossings.append((crossing_start[0], i, crossing_start[1], closest_j))
                in_crossing = False

    return crossings


def find_overlapping_ribbons(ribbons_data: List[Dict], threshold: float = 80) -> List[List[int]]:
    """
    Find which ribbons overlap with each other.
    Returns adjacency list: overlaps[i] = list of ribbon indices that overlap with ribbon i
    """
    n = len(ribbons_data)
    overlaps = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            # Check if ribbons overlap based on center distance and radii
            c1 = ribbons_data[i]['center']
            c2 = ribbons_data[j]['center']
            r1 = ribbons_data[i]['radius']
            r2 = ribbons_data[j]['radius']

            dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

            # Ribbons overlap if their bounding areas intersect
            if dist < r1 + r2 + threshold:
                overlaps[i].append(j)
                overlaps[j].append(i)

    return overlaps


def assign_non_overlapping_gradients(ribbons_data: List[Dict], overlaps: List[List[int]]) -> List[Tuple]:
    """
    Assign gradients to ribbons such that overlapping ribbons have different colors.
    Uses greedy graph coloring approach.
    """
    n = len(ribbons_data)
    assigned = [None] * n

    for i in range(n):
        # Find gradients used by overlapping ribbons
        used_gradients = set()
        for j in overlaps[i]:
            if assigned[j] is not None:
                used_gradients.add(assigned[j])

        # Pick a gradient not used by overlapping ribbons
        available = [g for g in range(len(NEON_GRADIENTS)) if g not in used_gradients]

        if available:
            assigned[i] = random.choice(available)
        else:
            # All gradients used by neighbors - pick randomly (rare case)
            assigned[i] = random.randint(0, len(NEON_GRADIENTS) - 1)

    return [NEON_GRADIENTS[idx] for idx in assigned]


def draw_ribbons_layered(img: Image.Image,
                         ribbons_data: List[Dict]):
    """
    Draw ribbons with clean layering - each ribbon at ONE consistent depth.
    NO weaving - just simple painter's algorithm (back to front).

    This is easier for humans (clear layer relationships) and still hard for VLMs
    due to occlusion and color gradients.
    """
    draw = ImageDraw.Draw(img)

    n = len(ribbons_data)
    if n == 0:
        return

    # Find which ribbons overlap - combine detected overlaps with explicit crossings
    overlaps = find_overlapping_ribbons(ribbons_data)

    # Add explicit crossing pairs (from perpendicular positioning)
    for i, ribbon in enumerate(ribbons_data):
        crosses_idx = ribbon.get('crosses')
        if crosses_idx is not None and 0 <= crosses_idx < n:
            # Add bidirectional crossing relationship
            if crosses_idx not in overlaps[i]:
                overlaps[i].append(crosses_idx)
            if i not in overlaps[crosses_idx]:
                overlaps[crosses_idx].append(i)

    # Assign different gradients to crossing/overlapping ribbons
    gradient_assignments = assign_non_overlapping_gradients(ribbons_data, overlaps)

    # Assign random depth to each ribbon
    depths = list(range(n))
    random.shuffle(depths)

    # Draw each ribbon COMPLETELY (back to front by depth)
    # Each ribbon is at ONE consistent layer - no weaving
    sorted_indices = sorted(range(n), key=lambda x: depths[x])

    for idx in sorted_indices:
        ribbon = ribbons_data[idx]
        centerline = ribbon['centerline']
        nc = len(centerline)

        # Use the pre-assigned gradient (ensures overlapping ribbons have different colors)
        start_color, end_color = gradient_assignments[idx]

        # Random phase offset per ribbon (0.0 to 1.0)
        phase_offset = random.random()
        # Random 2-4 cycles per ribbon
        num_cycles = random.randint(2, 4)

        # Draw entire ribbon with gradient segments
        step_size = 4
        segment_length = 12

        for start in range(0, nc, step_size):
            # Get segment with proper wrap-around using modulo
            segment = [centerline[(start + i) % nc] for i in range(segment_length)]
            if len(segment) < 3:
                continue

            # Cycling with ping-pong + phase offset
            raw_t = (start / nc * num_cycles + phase_offset) % 1.0
            t = raw_t * 2 if raw_t < 0.5 else 2 * (1 - raw_t)

            r = int(start_color[0] * (1 - t) + end_color[0] * t)
            g = int(start_color[1] * (1 - t) + end_color[1] * t)
            b = int(start_color[2] * (1 - t) + end_color[2] * t)

            seg_color = (r, g, b)

            seg_poly = get_ribbon_polygon(segment, RIBBON_WIDTH)
            draw.polygon(seg_poly, fill=seg_color)


def add_subtle_noise(img: Image.Image, strength: float = 0.015) -> Image.Image:
    """Add very subtle noise to defeat simple thresholding."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.randn(*arr.shape) * strength * 255
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def check_triple_overlap(new_center: Tuple[float, float],
                         new_radius: float,
                         existing_ribbons: List[Dict],
                         threshold: float = 30) -> bool:
    """
    Check if new ribbon would create 3+ ribbon overlap in any region.
    Returns True if triple overlap detected (should reject this position).

    STRICT: Also rejects if new ribbon overlaps with more than 1 existing ribbon
    to prevent cluttered areas.
    """
    # Find which existing ribbons the new one would overlap with
    overlapping = []
    for i, ribbon in enumerate(existing_ribbons):
        dist = math.sqrt((new_center[0] - ribbon['center'][0])**2 +
                        (new_center[1] - ribbon['center'][1])**2)
        # Two ribbons overlap if their centers are close enough
        if dist < new_radius + ribbon['radius'] + threshold:
            overlapping.append(i)

    # STRICT: Don't allow new ribbon to overlap with more than 1 existing ribbon
    # This prevents cluttered areas where 3+ ribbons meet
    if len(overlapping) > 1:
        return True

    # Check if any pair of overlapping ribbons also overlap with each other
    # (which would create a triple-overlap region)
    for i in range(len(overlapping)):
        for j in range(i + 1, len(overlapping)):
            r1 = existing_ribbons[overlapping[i]]
            r2 = existing_ribbons[overlapping[j]]
            dist = math.sqrt((r1['center'][0] - r2['center'][0])**2 +
                            (r1['center'][1] - r2['center'][1])**2)
            if dist < r1['radius'] + r2['radius'] + threshold:
                # Found two existing ribbons that overlap, and new ribbon overlaps both
                return True
    return False


def get_perpendicular_orientation(existing_orientations: List[float]) -> float:
    """
    Find an orientation that's most perpendicular to existing ones.
    Returns angle in radians that maximizes minimum angular distance.
    """
    if not existing_orientations:
        return random.uniform(0, math.pi)  # Any orientation works

    # Try several candidate angles and pick the one most perpendicular
    best_angle = 0
    best_min_dist = -1

    for candidate in [i * math.pi / 8 for i in range(8)]:  # Try 8 angles
        # Find minimum angular distance to any existing orientation
        min_dist = float('inf')
        for existing in existing_orientations:
            # Angular distance (accounting for symmetry of ribbons)
            diff = abs(candidate - existing) % math.pi
            dist = min(diff, math.pi - diff)  # Distance to perpendicular
            # We want distance close to pi/4 (45 degrees) or more
            min_dist = min(min_dist, dist)

        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_angle = candidate

    # Add some randomness to avoid identical angles
    return best_angle + random.uniform(-0.2, 0.2)


def generate_ribbon_image(num_ribbons: int,
                          cell_size: int = CELL_SIZE) -> Tuple[Image.Image, int]:
    """
    Generate a single image with the specified number of ribbon loops.
    Uses triple-overlap prevention and perpendicular crossings for clarity.
    """
    img = Image.new('RGB', (cell_size, cell_size), BACKGROUND_COLOR)

    margin = 50
    usable_size = cell_size - 2 * margin

    ribbons_data = []
    max_attempts = 50

    # Generate ribbon positions with triple-overlap prevention
    for i in range(num_ribbons):
        for attempt in range(max_attempts):
            # Vary the radius
            base_radius = random.randint(45, 75)

            # Track which ribbon this one crosses (for color assignment)
            crosses_idx = None

            # For 2-ribbon and 3-ribbon cases, position to encourage PERPENDICULAR CROSSING
            if num_ribbons <= 3 and len(ribbons_data) >= 1:
                # Pick a random existing ribbon to cross
                crosses_idx = random.randint(0, len(ribbons_data) - 1)
                target_ribbon = ribbons_data[crosses_idx]
                target_cx, target_cy = target_ribbon['center']
                target_r = target_ribbon['radius']

                # Position new ribbon for PERPENDICULAR CROSSING at edges
                # Offset should be 50-85% of sum of radii so they cross like an X
                # NOT too close (whole overlap) and NOT too far (no crossing)
                min_offset = (target_r + base_radius) * 0.5
                max_offset = (target_r + base_radius) * 0.85
                offset_dist = random.uniform(min_offset, max_offset)
                offset_angle = random.uniform(0, 2 * math.pi)

                cx = target_cx + offset_dist * math.cos(offset_angle)
                cy = target_cy + offset_dist * math.sin(offset_angle)

                # Clamp to valid region
                cx = max(margin + 30, min(margin + usable_size - 30, cx))
                cy = max(margin + 30, min(margin + usable_size - 30, cy))
            else:
                # Standard random positioning for 4+ ribbons
                cx = margin + random.randint(30, usable_size - 30)
                cy = margin + random.randint(30, usable_size - 30)

            # ALWAYS check for triple overlap when we have 2+ existing ribbons
            # This prevents cluttered areas where 3+ ribbons overlap
            if len(ribbons_data) >= 2:
                if check_triple_overlap((cx, cy), base_radius, ribbons_data):
                    continue  # Try another position

            # Good position found
            break

        # Determine orientation - for 2 and 3 ribbon cases, ALWAYS make perpendicular
        if num_ribbons <= 3 and len(ribbons_data) >= 1:
            # For 2-3 ribbon images, always use perpendicular crossings
            # Find the closest existing ribbon and be perpendicular to it
            closest_ribbon = None
            closest_dist = float('inf')
            for ribbon in ribbons_data:
                dist = math.sqrt((cx - ribbon['center'][0])**2 +
                               (cy - ribbon['center'][1])**2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_ribbon = ribbon

            if closest_ribbon:
                # Offset by 90 degrees (pi/2) with small random variation
                closest_orientation = closest_ribbon.get('orientation', 0)
                orientation = closest_orientation + math.pi / 2 + random.uniform(-0.15, 0.15)
            else:
                orientation = random.uniform(0, math.pi)
        else:
            # For 4+ ribbons, use the overlap-based perpendicular logic
            overlapping_orientations = []
            for ribbon in ribbons_data:
                dist = math.sqrt((cx - ribbon['center'][0])**2 +
                               (cy - ribbon['center'][1])**2)
                if dist < base_radius + ribbon['radius'] + 60:  # Will overlap
                    overlapping_orientations.append(ribbon.get('orientation', 0))

            # Choose orientation perpendicular to overlapping ribbons
            orientation = get_perpendicular_orientation(overlapping_orientations)

        # Generate smooth deformed loop with the chosen orientation
        points = generate_smooth_loop((cx, cy), base_radius, num_points=60, orientation=orientation)
        centerline = smooth_spline(points, num_output=150)

        ribbons_data.append({
            'centerline': centerline,
            'center': (cx, cy),
            'radius': base_radius,
            'orientation': orientation,
            'crosses': crosses_idx  # Index of ribbon this one is positioned to cross
        })

    # Draw ribbons with clean layering (each ribbon at ONE consistent depth)
    draw_ribbons_layered(img, ribbons_data)

    # Add very subtle noise
    img = add_subtle_noise(img, strength=0.012)

    # Slight anti-aliasing blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))

    return img, num_ribbons


def generate_cell_pool() -> Dict:
    """
    Generate pool of images with varying ribbon counts.
    Now supports multiple target counts (3 and 4 ribbons).
    """
    cell_pool = {}

    # Build distribution to ensure enough images for both targets
    ribbon_counts = []

    # Need enough of each target type for correct answers
    # Each target needs ~15 images for correct answers across puzzles
    # Non-targets need enough for incorrect answers

    for count in range(MIN_RIBBONS, MAX_RIBBONS + 1):
        if count in TARGET_RIBBON_COUNTS:
            ribbon_counts.extend([count] * 20)  # More target images (3 and 4)
        else:
            ribbon_counts.extend([count] * 13)  # 2 and 5 ribbons as non-targets

    # Shuffle and ensure exactly NUM_POOL_IMAGES
    random.shuffle(ribbon_counts)
    ribbon_counts = ribbon_counts[:NUM_POOL_IMAGES]

    # If we have fewer, pad with random
    while len(ribbon_counts) < NUM_POOL_IMAGES:
        ribbon_counts.append(random.randint(MIN_RIBBONS, MAX_RIBBONS))

    print(f"Generating {len(ribbon_counts)} pool images...")
    dist = {c: ribbon_counts.count(c) for c in range(MIN_RIBBONS, MAX_RIBBONS + 1)}
    print(f"Distribution: {dist}")

    for idx, num_ribbons in enumerate(ribbon_counts):
        img, actual_count = generate_ribbon_image(num_ribbons)

        filename = f"ribbon_{idx:03d}.png"
        img.save(OUTPUT_DIR / filename)

        cell_pool[f"cell_{idx}"] = {
            "filename": filename,
            "num_ribbons": actual_count
        }

        if (idx + 1) % 10 == 0:
            print(f"  Generated {idx + 1}/{len(ribbon_counts)} images")

    return cell_pool


def generate_puzzles(cell_pool: Dict) -> Dict:
    """
    Generate puzzle configurations from the cell pool.
    Each puzzle randomly asks for 3 OR 4 ribbons for variation.
    """
    ground_truth = {}

    # Group cells by ribbon count
    cells_by_count = {}
    for cid, data in cell_pool.items():
        count = data['num_ribbons']
        if count not in cells_by_count:
            cells_by_count[count] = []
        cells_by_count[count].append(cid)

    print(f"\nCells by ribbon count:")
    for count in sorted(cells_by_count.keys()):
        print(f"  {count} ribbons: {len(cells_by_count[count])} cells")

    cells_per_puzzle = GRID_SIZE * GRID_SIZE  # 16

    for puzzle_idx in range(NUM_PUZZLES):
        # Randomly choose target: 3 or 4 ribbons
        target_count = random.choice(TARGET_RIBBON_COUNTS)

        # Get target and non-target cells for this puzzle
        target_cells = cells_by_count.get(target_count, [])
        non_target_cells = []
        for count, cells in cells_by_count.items():
            if count != target_count:
                non_target_cells.extend(cells)

        # Each puzzle has 3-6 correct answers
        num_correct = random.randint(3, 6)
        num_correct = min(num_correct, len(target_cells))

        # Sample cells (with replacement allowed across puzzles)
        correct_cells = random.sample(target_cells, num_correct)
        incorrect_cells = random.sample(non_target_cells,
                                       min(cells_per_puzzle - num_correct, len(non_target_cells)))

        # Combine and shuffle
        all_cells = correct_cells + incorrect_cells
        random.shuffle(all_cells)

        # Determine answer indices
        answer_indices = [i for i, cid in enumerate(all_cells) if cid in correct_cells]

        ground_truth[f"illusory_ribbons_{puzzle_idx:04d}"] = {
            "prompt": f"Click all cells where you can count exactly {target_count} separate ribbon loops. Each ribbon is a single continuous closed band that never crosses itself - it may pass behind OTHER ribbons, but trace each one around to verify it forms one complete loop.",
            "description": f"Illusory ribbons - select cells with exactly {target_count} ribbons",
            "cells": all_cells,
            "answer": sorted(answer_indices),
            "input_type": "illusory_ribbons_select",
            "grid_size": [GRID_SIZE, GRID_SIZE],
            "target_count": target_count
        }

    return ground_truth


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Illusory Ribbons CAPTCHA Generator")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Pool size: {NUM_POOL_IMAGES}")
    print(f"Puzzles: {NUM_PUZZLES}")
    print(f"Targets: {TARGET_RIBBON_COUNTS} ribbons (varies per puzzle)")
    print()

    # Generate cell pool
    cell_pool = generate_cell_pool()

    # Save cell pool
    with open(OUTPUT_DIR / 'cell_pool.json', 'w') as f:
        json.dump(cell_pool, f, indent=2)
    print(f"\nSaved cell_pool.json with {len(cell_pool)} entries")

    # Generate puzzles
    ground_truth = generate_puzzles(cell_pool)

    # Save ground truth
    with open(OUTPUT_DIR / 'ground_truth.json', 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved ground_truth.json with {len(ground_truth)} puzzles")

    # Summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)

    ribbon_dist = {}
    for cid, data in cell_pool.items():
        count = data['num_ribbons']
        ribbon_dist[count] = ribbon_dist.get(count, 0) + 1

    print(f"\nRibbon count distribution:")
    for count in sorted(ribbon_dist.keys()):
        marker = " <-- TARGET" if count in TARGET_RIBBON_COUNTS else ""
        print(f"  {count} ribbons: {ribbon_dist[count]} images{marker}")

    # Show puzzle target distribution
    target_dist = {}
    for pid, pdata in ground_truth.items():
        tc = pdata['target_count']
        target_dist[tc] = target_dist.get(tc, 0) + 1
    print(f"\nPuzzle target distribution:")
    for tc in sorted(target_dist.keys()):
        print(f"  'Find {tc} ribbons': {target_dist[tc]} puzzles")

    print(f"\nPuzzle statistics:")
    correct_counts = [len(p['answer']) for p in ground_truth.values()]
    print(f"  Correct answers per puzzle: min={min(correct_counts)}, max={max(correct_counts)}, avg={sum(correct_counts)/len(correct_counts):.1f}")


if __name__ == "__main__":
    main()
