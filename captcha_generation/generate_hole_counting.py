"""
Hole Counting CAPTCHA Generator

Generates a pool of 20x20 pixel images with irregular colored shapes containing 0-5 holes.
Then creates puzzles by selecting 16 images from the pool to form a 4x4 grid.

Users must identify images with a specific number of holes from the grid.
Challenge: Random colors, irregular shapes, and small size make it hard for vision models.
"""

import numpy as np
from PIL import Image, ImageDraw
import json
import random
from pathlib import Path
from scipy import ndimage

# Configuration
MASK_SIZE = 30  # Generate masks at 30x30 pixels
OUTPUT_SIZE = 120  # Final output images at 120x120 pixels for very clear patterns
NUM_PUZZLES = 20
GRID_SIZE = (4, 4)  # 4x4 grid shown to user
TOTAL_CELLS = GRID_SIZE[0] * GRID_SIZE[1]

# Pool configuration
POOL_SIZE_PER_HOLE_COUNT = 10  # 10 images per hole count (0-5 holes)
TOTAL_POOL_SIZE = POOL_SIZE_PER_HOLE_COUNT * 6  # 60 images total

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "captcha_data" / "Hole_Counting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth.json"


def random_color():
    """
    Generate a high-contrast RGB color that's easy for humans to see.
    Uses saturated, distinct colors rather than muddy/gray tones.
    """
    # Use high-contrast color palette with distinct hues
    # Each color is saturated and far from gray/muddy tones
    high_contrast_colors = [
        # Primary colors (high saturation)
        (220, 20, 20),    # Bright red
        (20, 20, 220),    # Bright blue
        (20, 200, 20),    # Bright green

        # Secondary colors
        (220, 180, 20),   # Yellow/gold
        (220, 20, 220),   # Magenta
        (20, 200, 220),   # Cyan

        # Tertiary colors
        (220, 100, 20),   # Orange
        (140, 20, 220),   # Purple
        (20, 140, 100),   # Teal

        # Additional distinct colors
        (220, 20, 100),   # Pink/rose
        (100, 220, 20),   # Lime green
        (220, 140, 180),  # Light pink
        (100, 180, 220),  # Sky blue
        (180, 100, 20),   # Brown/rust
        (20, 100, 180),   # Deep blue
    ]

    return random.choice(high_contrast_colors)


def scale_mask(mask, scale_factor=2):
    """
    Scale up a binary mask using nearest neighbor interpolation.

    Args:
        mask: Binary mask (numpy array)
        scale_factor: Integer scale factor (default 2 for 30x30 -> 60x60)

    Returns:
        Scaled binary mask
    """
    from scipy.ndimage import zoom
    return zoom(mask.astype(float), scale_factor, order=0) > 0.5


def apply_pattern_fill(mask, base_color, is_hole=False):
    """
    Apply a pattern/texture fill to a mask region instead of solid color.
    Patterns are designed for 120x120 resolution for very clear, easy-to-see textures.
    Holes use fine-grained patterns for easy discovery by humans.
    Makes it harder for LLMs to analyze while remaining crystal clear for humans.

    Args:
        mask: Binary mask (numpy array) of the region to fill (should be 120x120)
        base_color: RGB tuple to use as base for the pattern
        is_hole: If True, use fine-grained patterns optimized for holes

    Returns:
        RGB image array with pattern applied to the mask region
    """
    h, w = mask.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    if is_hole:
        # Fine-grained patterns for holes - easier to spot
        pattern_type = random.choice([
            'fine_dots',       # Very fine dots
            'fine_horizontal', # Fine horizontal lines
            'fine_vertical',   # Fine vertical lines
            'fine_grid',       # Fine grid pattern
            'solid',           # Some solid for variety
        ])
    else:
        # Larger patterns for main shapes
        pattern_type = random.choice([
            'solid',           # Solid color
            'horizontal',      # Horizontal stripes - very clear
            'horizontal',      # Doubled for more frequency
            'vertical',        # Vertical stripes - very clear
            'vertical',        # Doubled for more frequency
            'diagonal',        # Diagonal lines - clear
            'checkerboard',    # Checkerboard pattern - very clear
            'checkerboard',    # Doubled for more frequency
            'dots',            # Polka dots - clear
            'crosshatch',      # Cross-hatch pattern
            'waves',           # Wavy lines - new pattern
            'gradient',        # Gradient fill
        ])

    r, g, b = base_color

    # Generate pattern
    if pattern_type == 'solid':
        # Solid color fill
        result[mask] = base_color

    elif pattern_type == 'horizontal':
        # Horizontal stripes (clear for 120x120)
        stripe_width = random.randint(5, 8)
        for y in range(h):
            if (y // stripe_width) % 2 == 0:
                result[y, :] = base_color
            else:
                # Darker version for contrast
                result[y, :] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'vertical':
        # Vertical stripes (clear for 120x120)
        stripe_width = random.randint(5, 8)
        for x in range(w):
            if (x // stripe_width) % 2 == 0:
                result[:, x] = base_color
            else:
                # Darker version for contrast
                result[:, x] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'diagonal':
        # Diagonal lines (clear for 120x120)
        stripe_width = random.randint(6, 10)
        for y in range(h):
            for x in range(w):
                if ((x + y) // stripe_width) % 2 == 0:
                    result[y, x] = base_color
                else:
                    result[y, x] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'checkerboard':
        # Checkerboard pattern (clear for 120x120)
        square_size = random.randint(8, 12)
        for y in range(h):
            for x in range(w):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    result[y, x] = base_color
                else:
                    result[y, x] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'dots':
        # Polka dots pattern (clear for 120x120)
        result[mask] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))  # Background
        dot_spacing = random.randint(10, 14)
        dot_radius = random.randint(3, 5)
        for y in range(0, h, dot_spacing):
            for x in range(0, w, dot_spacing):
                for dy in range(-dot_radius, dot_radius + 1):
                    for dx in range(-dot_radius, dot_radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and dx*dx + dy*dy <= dot_radius*dot_radius:
                            result[ny, nx] = base_color
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'crosshatch':
        # Cross-hatch pattern (clear for 120x120)
        result[mask] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))  # Background
        spacing = random.randint(8, 12)
        for y in range(h):
            for x in range(w):
                if (x + y) % spacing == 0 or (x - y) % spacing == 0:
                    result[y, x] = base_color
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'waves':
        # Wavy horizontal lines (new pattern, clear for 120x120)
        result[mask] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))  # Background
        wave_height = random.randint(3, 5)
        wave_frequency = random.uniform(0.3, 0.5)
        stripe_width = random.randint(5, 7)
        for y in range(h):
            for x in range(w):
                # Create wavy pattern
                offset = int(wave_height * np.sin(x * wave_frequency))
                if ((y + offset) // stripe_width) % 2 == 0:
                    result[y, x] = base_color
                else:
                    result[y, x] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'gradient':
        # Gradient fill (smoother for 120x120)
        for y in range(h):
            factor = y / max(1, h - 1)
            gradient_color = (
                int(r * (1 - factor * 0.5)),
                int(g * (1 - factor * 0.5)),
                int(b * (1 - factor * 0.5))
            )
            result[y, :] = gradient_color
        result[~mask] = (0, 0, 0)

    # Fine-grained patterns specifically for holes - VERY SMALL patterns
    elif pattern_type == 'fine_dots':
        # Ultra-fine dots - very easy to spot as holes
        result[mask] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))  # Background
        dot_spacing = random.randint(3, 4)  # Very tight spacing
        dot_radius = 1  # Tiny dots
        for y in range(0, h, dot_spacing):
            for x in range(0, w, dot_spacing):
                if 0 <= y < h and 0 <= x < w:
                    result[y, x] = base_color
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'fine_horizontal':
        # Ultra-fine horizontal lines - creates very dense texture
        stripe_width = 2  # Just 2 pixels - very fine
        for y in range(h):
            if y % stripe_width == 0:
                result[y, :] = base_color
            else:
                result[y, :] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'fine_vertical':
        # Ultra-fine vertical lines - creates very dense texture
        stripe_width = 2  # Just 2 pixels - very fine
        for x in range(w):
            if x % stripe_width == 0:
                result[:, x] = base_color
            else:
                result[:, x] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))
        result[~mask] = (0, 0, 0)

    elif pattern_type == 'fine_grid':
        # Ultra-fine grid pattern - very distinctive for holes
        grid_spacing = random.randint(3, 4)  # Tight grid
        result[mask] = (max(0, r - 80), max(0, g - 80), max(0, b - 80))  # Background
        for y in range(h):
            for x in range(w):
                if y % grid_spacing == 0 or x % grid_spacing == 0:
                    result[y, x] = base_color
        result[~mask] = (0, 0, 0)

    return result


def generate_irregular_shape_mask(size=30):
    """
    Generate an irregular blob-like shape that's hard to describe.
    Uses noise-based blob generation for organic, irregular shapes.
    Ensures the shape is a single continuous connected component.
    IMPORTANT: Ensures the shape has no inherent topological holes (like a donut).

    Returns:
        Binary mask (numpy array) where shape pixels are True
    """
    max_attempts = 50  # Avoid infinite loops

    for attempt in range(max_attempts):
        # Start with random noise
        noise = np.random.randn(size, size)

        # Blur to create blob-like structures
        blurred = ndimage.gaussian_filter(noise, sigma=size/8)

        # Threshold to create binary mask
        threshold = np.percentile(blurred, random.randint(30, 50))
        mask = blurred > threshold

        # Ensure shape is connected and reasonably sized
        # Use binary opening/closing to smooth edges
        mask = ndimage.binary_opening(mask, iterations=1)
        mask = ndimage.binary_closing(mask, iterations=2)

        # Extract largest connected component to ensure continuity
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            # Find the largest component
            component_sizes = np.bincount(labeled.ravel())[1:]  # Exclude background (0)
            largest_component = np.argmax(component_sizes) + 1
            mask = (labeled == largest_component)

        # Erode slightly to ensure holes can fit inside with proper borders
        mask = ndimage.binary_erosion(mask, iterations=2)

        # CRITICAL: Check if the shape has any inherent topological holes
        # Fill all holes and compare - if different, shape had inherent holes
        filled_mask = ndimage.binary_fill_holes(mask)

        # If the filled mask is the same as original, no inherent holes exist
        if np.array_equal(mask, filled_mask):
            return mask

        # Shape had inherent holes (like a donut), try again
        continue

    # Fallback: if we couldn't generate a hole-free shape after many attempts,
    # return a filled version (no inherent holes)
    return filled_mask


def generate_hole_shape(hole_size=4):
    """
    Generate a very irregular, strange-shaped hole that is a single connected component.
    Uses multiple noise layers and morphological operations for organic, unpredictable shapes.

    Args:
        hole_size: Approximate size of the hole

    Returns:
        Binary mask (numpy array) for the hole shape
    """
    max_attempts = 30
    for attempt in range(max_attempts):
        # Create larger canvas for more irregular shapes
        size = hole_size * 3 + 1

        # Layer multiple noise patterns with different frequencies
        noise1 = np.random.randn(size, size)
        noise2 = np.random.randn(size, size)

        # Combine noise at different scales for more irregularity
        blurred1 = ndimage.gaussian_filter(noise1, sigma=hole_size/3)
        blurred2 = ndimage.gaussian_filter(noise2, sigma=hole_size/6)

        # Combine the noise layers with weighted sum
        combined = 0.6 * blurred1 + 0.4 * blurred2

        # Use more aggressive thresholding for irregular shapes
        threshold = np.percentile(combined, random.randint(40, 60))
        hole_mask = combined > threshold

        # Randomly apply morphological operations for stranger shapes
        morph_choice = random.choice(['erode', 'dilate', 'open', 'close', 'none'])
        if morph_choice == 'erode':
            hole_mask = ndimage.binary_erosion(hole_mask, iterations=1)
        elif morph_choice == 'dilate':
            hole_mask = ndimage.binary_dilation(hole_mask, iterations=1)
        elif morph_choice == 'open':
            hole_mask = ndimage.binary_opening(hole_mask, iterations=1)
        elif morph_choice == 'close':
            hole_mask = ndimage.binary_closing(hole_mask, iterations=1)

        # Extract largest connected component
        labeled, num_features = ndimage.label(hole_mask)
        if num_features > 0:
            component_sizes = np.bincount(labeled.ravel())[1:]
            if len(component_sizes) > 0:
                largest_component = np.argmax(component_sizes) + 1
                hole_mask = (labeled == largest_component)

                # Verify it's connected and has reasonable size
                hole_area = np.sum(hole_mask)
                if hole_area >= 4 and hole_area <= (size * size) // 3:
                    return hole_mask

    # Fallback: create an irregular polygon shape
    size = hole_size * 3 + 1
    hole_mask = np.zeros((size, size), dtype=bool)
    center = size // 2

    # Create irregular star-like shape
    num_points = random.randint(4, 7)
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    for i in range(num_points):
        angle = angles[i] + random.uniform(-0.3, 0.3)
        radius = random.randint(2, max(3, hole_size))
        x = int(center + radius * np.cos(angle))
        y = int(center + radius * np.sin(angle))
        if 0 <= x < size and 0 <= y < size:
            hole_mask[y, x] = True

    # Fill and smooth the polygon
    hole_mask = ndimage.binary_dilation(hole_mask, iterations=2)
    hole_mask = ndimage.binary_fill_holes(hole_mask)
    hole_mask = ndimage.binary_erosion(hole_mask, iterations=1)

    return hole_mask


def place_holes_in_shape(shape_mask, num_holes):
    """
    Place irregular-shaped holes inside the shape without overlapping.

    Args:
        shape_mask: Binary mask of the shape
        num_holes: Number of holes to place (0-5)

    Returns:
        List of hole specifications: [(center_x, center_y, hole_mask, color), ...]
    """
    if num_holes == 0:
        return []

    # Find all valid positions inside the shape
    size = shape_mask.shape[0]
    y_coords, x_coords = np.where(shape_mask)

    if len(y_coords) == 0:
        return []  # No valid positions

    holes = []
    # Track which pixels are occupied by holes
    hole_occupied = np.zeros_like(shape_mask, dtype=bool)
    max_attempts = 200

    for _ in range(num_holes):
        attempts = 0
        placed = False

        while attempts < max_attempts and not placed:
            # Random hole size (2-5)
            hole_size = random.randint(2, 5)

            # Generate irregular hole shape
            hole_mask = generate_hole_shape(hole_size)
            hole_h, hole_w = hole_mask.shape

            # Pick a random position inside the shape
            idx = random.randint(0, len(y_coords) - 1)
            center_y = y_coords[idx]
            center_x = x_coords[idx]

            # Calculate hole bounding box
            half_h = hole_h // 2
            half_w = hole_w // 2
            y_start = center_y - half_h
            y_end = y_start + hole_h
            x_start = center_x - half_w
            x_end = x_start + hole_w

            # Check if hole fits inside canvas
            if y_start < 0 or y_end >= size or x_start < 0 or x_end >= size:
                attempts += 1
                continue

            # Check if entire hole + 1-pixel border is within shape
            fits_inside = True

            # First check the hole itself fits in the shape
            for hy in range(hole_h):
                for hx in range(hole_w):
                    if hole_mask[hy, hx]:
                        canvas_y = y_start + hy
                        canvas_x = x_start + hx
                        if not shape_mask[canvas_y, canvas_x]:
                            fits_inside = False
                            break
                if not fits_inside:
                    break

            if not fits_inside:
                attempts += 1
                continue

            # Check that all pixels immediately surrounding the hole are part of the shape
            # This ensures holes are truly "enclosed" by the main shape color
            for hy in range(hole_h):
                for hx in range(hole_w):
                    if hole_mask[hy, hx]:
                        # Check all 8 neighbors of this hole pixel
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                if dy == 0 and dx == 0:
                                    continue  # Skip the hole pixel itself

                                ny, nx = hy + dy, hx + dx
                                canvas_y = y_start + ny
                                canvas_x = x_start + nx

                                # Check bounds
                                if canvas_y < 0 or canvas_y >= size or canvas_x < 0 or canvas_x >= size:
                                    fits_inside = False
                                    break

                                # Check if this neighbor is also part of the hole
                                neighbor_is_hole = (0 <= ny < hole_h and 0 <= nx < hole_w and hole_mask[ny, nx])

                                # If neighbor is NOT part of the hole, it MUST be part of the shape
                                if not neighbor_is_hole:
                                    if not shape_mask[canvas_y, canvas_x]:
                                        fits_inside = False
                                        break
                            if not fits_inside:
                                break
                    if not fits_inside:
                        break
                if not fits_inside:
                    break

            if not fits_inside:
                attempts += 1
                continue

            # Check no overlap with existing holes + 1 pixel margin
            overlaps = False
            for hy in range(hole_h):
                for hx in range(hole_w):
                    if hole_mask[hy, hx]:
                        canvas_y = y_start + hy
                        canvas_x = x_start + hx

                        # Check if this pixel or its neighbors are already occupied by another hole
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                check_y = canvas_y + dy
                                check_x = canvas_x + dx
                                if 0 <= check_y < size and 0 <= check_x < size:
                                    if hole_occupied[check_y, check_x]:
                                        overlaps = True
                                        break
                            if overlaps:
                                break
                    if overlaps:
                        break
                if overlaps:
                    break

            if overlaps:
                attempts += 1
                continue

            # Hole fits! Mark pixels as occupied
            for hy in range(hole_h):
                for hx in range(hole_w):
                    if hole_mask[hy, hx]:
                        canvas_y = y_start + hy
                        canvas_x = x_start + hx
                        hole_occupied[canvas_y, canvas_x] = True

            hole_color = random_color()
            holes.append((center_x, center_y, hole_mask, hole_color))
            placed = True

            attempts += 1

        # If we couldn't place this hole after many attempts, skip it
        if not placed:
            break

    return holes


def check_shape_connectivity(img, shape_color):
    """
    Check if the shape in the image is a single connected component.

    Args:
        img: PIL Image
        shape_color: RGB tuple of the shape color

    Returns:
        True if shape is connected, False otherwise
    """
    img_array = np.array(img)
    shape_mask = np.all(img_array == shape_color, axis=-1)

    labeled, num_components = ndimage.label(shape_mask)

    # Shape should be exactly 1 connected component
    return num_components == 1


def count_actual_holes(img, shape_color, bg_color):
    """
    Count the actual number of distinct holes in the shape.
    A hole is a connected component that is:
    1. Not the background color
    2. Not the shape color
    3. Completely surrounded by the shape

    Args:
        img: PIL Image
        shape_color: RGB tuple of the shape color
        bg_color: RGB tuple of the background color

    Returns:
        Number of distinct connected holes
    """
    img_array = np.array(img)

    # Create mask for all non-shape, non-background pixels (holes)
    shape_mask = np.all(img_array == shape_color, axis=-1)
    bg_mask = np.all(img_array == bg_color, axis=-1)

    hole_mask = ~shape_mask & ~bg_mask

    # Count connected components in holes
    labeled, num_holes = ndimage.label(hole_mask)

    return num_holes


def generate_single_cell_image(num_holes, cell_id):
    """
    Generate a cell image with a shape containing specified number of holes.
    Topology is generated at MASK_SIZE (30x30), then scaled to OUTPUT_SIZE (60x60)
    and filled with high-resolution patterns.
    Ensures the shape remains connected (single component) after holes are drawn.

    Args:
        num_holes: Number of holes (0-5)
        cell_id: Identifier for this cell

    Returns:
        PIL Image (60x60 RGB), actual number of holes placed
    """
    mask_size = MASK_SIZE
    output_size = OUTPUT_SIZE
    scale_factor = output_size // mask_size

    # Try multiple times to generate a connected shape with holes
    max_generation_attempts = 10

    for gen_attempt in range(max_generation_attempts):
        # Random colors
        bg_color = random_color()
        shape_color = random_color()

        # Ensure shape color is different from background
        while sum(abs(a - b) for a, b in zip(shape_color, bg_color)) < 100:
            shape_color = random_color()

        # Generate irregular shape mask at mask_size (30x30)
        shape_mask = generate_irregular_shape_mask(mask_size)

        # Place holes
        holes = place_holes_in_shape(shape_mask, num_holes)
        actual_holes = len(holes)

        # Create masks for tracking what is shape vs holes (at mask_size 30x30)
        final_shape_mask = shape_mask.copy()
        hole_regions_mask = np.zeros((mask_size, mask_size), dtype=bool)

        # Mark hole regions in masks (at mask_size 30x30)
        for (center_x, center_y, hole_mask, hole_color) in holes:
            hole_h, hole_w = hole_mask.shape
            half_h = hole_h // 2
            half_w = hole_w // 2

            for hy in range(hole_h):
                for hx in range(hole_w):
                    if hole_mask[hy, hx]:
                        canvas_x = center_x - half_w + hx
                        canvas_y = center_y - half_h + hy
                        if 0 <= canvas_x < mask_size and 0 <= canvas_y < mask_size:
                            final_shape_mask[canvas_y, canvas_x] = False
                            hole_regions_mask[canvas_y, canvas_x] = True

        # Check if shape is still connected after removing holes
        labeled_shape, num_shape_components = ndimage.label(final_shape_mask)
        if num_shape_components != 1:
            continue  # Shape got fragmented, try again

        # Count actual holes (connected components in hole regions)
        labeled_holes, actual_hole_count = ndimage.label(hole_regions_mask)

        # Verify the actual hole count matches what we tried to place
        if actual_hole_count != num_holes:
            continue  # Hole count doesn't match, try again

        # Now scale up to output_size (60x60) and draw with patterns
        # Scale masks from 30x30 to 60x60
        shape_mask_scaled = scale_mask(shape_mask, scale_factor)
        hole_regions_mask_scaled = scale_mask(hole_regions_mask, scale_factor)

        # Create output image at 60x60
        img_array = np.zeros((output_size, output_size, 3), dtype=np.uint8)

        # Fill background
        img_array[:, :] = bg_color

        # Draw shape with pattern fill at 120x120 (larger patterns)
        shape_pattern = apply_pattern_fill(shape_mask_scaled, shape_color, is_hole=False)
        img_array[shape_mask_scaled] = shape_pattern[shape_mask_scaled]

        # Draw holes with fine-grained patterns at 120x120 (easy to spot)
        # Need to identify each hole component in the scaled mask
        labeled_holes_scaled, num_holes_scaled = ndimage.label(hole_regions_mask_scaled)

        for hole_idx in range(1, num_holes_scaled + 1):
            hole_mask_scaled = (labeled_holes_scaled == hole_idx)

            # Get a highly contrasting color for this hole
            # Use complementary/opposite colors to make holes very distinct
            hole_color = random_color()
            attempts = 0
            # Much higher threshold (200) to ensure very different colors
            # Also ensure different from background
            while attempts < 50:
                color_diff_shape = sum(abs(a - b) for a, b in zip(hole_color, shape_color))
                color_diff_bg = sum(abs(a - b) for a, b in zip(hole_color, bg_color))
                if color_diff_shape >= 200 and color_diff_bg >= 150:
                    break
                hole_color = random_color()
                attempts += 1

            # Apply fine-grained pattern fill to this hole (is_hole=True)
            hole_pattern = apply_pattern_fill(hole_mask_scaled, hole_color, is_hole=True)
            img_array[hole_mask_scaled] = hole_pattern[hole_mask_scaled]

        img = Image.fromarray(img_array)

        # Successfully generated valid image with correct hole count
        return img, actual_hole_count

    # If we couldn't generate a valid image after many attempts,
    # return the last attempt with its actual hole count (if img exists)
    if 'img' in locals():
        return img, actual_hole_count
    else:
        # Fallback: create simple image with no holes at 60x60
        bg_color = random_color()
        shape_color = random_color()
        shape_mask = generate_irregular_shape_mask(mask_size)
        shape_mask_scaled = scale_mask(shape_mask, scale_factor)

        img_array = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        img_array[:, :] = bg_color

        shape_pattern = apply_pattern_fill(shape_mask_scaled, shape_color, is_hole=False)
        img_array[shape_mask_scaled] = shape_pattern[shape_mask_scaled]
        return Image.fromarray(img_array), 0


def generate_cell_pool():
    """
    Generate pool of cell images with different hole counts.

    Returns:
        Dictionary mapping cell_id to (num_holes, filename)
    """
    print("Generating pool of cell images...")

    cell_pool = {}

    for hole_count in range(6):  # 0-5 holes
        for variant in range(POOL_SIZE_PER_HOLE_COUNT):
            cell_id = f"cell_{hole_count}holes_{variant}"

            # Generate cell image
            attempts = 0
            while attempts < 10:
                img, actual_holes = generate_single_cell_image(hole_count, cell_id)

                # Accept if we got the right number of holes (or close enough)
                if actual_holes == hole_count or attempts >= 5:
                    break
                attempts += 1

            # Save cell image
            filename = f"{cell_id}.png"
            img_path = OUTPUT_DIR / filename
            img.save(img_path)

            # Store in pool
            cell_pool[cell_id] = {
                "filename": filename,
                "hole_count": actual_holes
            }

            if (variant + 1) % 5 == 0:
                print(f"  Generated {hole_count}-hole cells: {variant + 1}/{POOL_SIZE_PER_HOLE_COUNT}")

    print(f"✓ Generated {len(cell_pool)} cell images")
    return cell_pool


def generate_puzzle(puzzle_id, cell_pool):
    """
    Generate a single Hole Counting puzzle by selecting cells from the pool.

    Args:
        puzzle_id: Puzzle identifier
        cell_pool: Dictionary of available cell images

    Returns:
        Puzzle data dictionary
    """
    # Decide target number of holes (1-5, avoiding 0 to make it more interesting)
    target_holes = random.randint(1, 5)

    # Get cells with target hole count
    target_cells = [cell_id for cell_id, data in cell_pool.items()
                    if data["hole_count"] == target_holes]

    # Get cells with other hole counts
    distractor_cells = [cell_id for cell_id, data in cell_pool.items()
                       if data["hole_count"] != target_holes]

    # Decide how many target cells (2-6 out of 16)
    num_targets = random.randint(2, 6)
    num_distractors = TOTAL_CELLS - num_targets

    # Select cells
    if len(target_cells) < num_targets:
        num_targets = len(target_cells)
        num_distractors = TOTAL_CELLS - num_targets

    selected_targets = random.sample(target_cells, min(num_targets, len(target_cells)))
    selected_distractors = random.sample(distractor_cells, min(num_distractors, len(distractor_cells)))

    # Combine and shuffle
    all_cells = selected_targets + selected_distractors
    random.shuffle(all_cells)

    # Pad if needed
    while len(all_cells) < TOTAL_CELLS:
        all_cells.append(random.choice(list(cell_pool.keys())))

    all_cells = all_cells[:TOTAL_CELLS]

    # Find correct indices
    correct_indices = [i for i, cell_id in enumerate(all_cells)
                      if cell_pool[cell_id]["hole_count"] == target_holes]

    # Create ground truth with clearer description
    hole_word = "hole" if target_holes == 1 else "holes"
    prompt = f"Click all tiles where the big shape has {target_holes} separate {hole_word} inside it. Each hole is a smaller area with a texture pattern that looks different from the main shape."

    puzzle_data = {
        "prompt": prompt,
        "description": f"Identify all shapes with exactly {target_holes} hole(s)",
        "target_holes": target_holes,
        "cells": all_cells,  # List of cell IDs in grid order (row-major)
        "answer": sorted(correct_indices),
        "input_type": "grid_select",
        "grid_size": GRID_SIZE,
        "total_cells": TOTAL_CELLS,
        "num_targets": len(correct_indices),
        "difficulty": 8
    }

    return puzzle_data


def main():
    print("=" * 60)
    print("Hole Counting CAPTCHA Generator")
    print("=" * 60)

    # Generate cell pool
    cell_pool = generate_cell_pool()

    # Save cell pool metadata
    pool_file = OUTPUT_DIR / "cell_pool.json"
    with open(pool_file, 'w') as f:
        json.dump(cell_pool, f, indent=2)

    print(f"\n✓ Cell pool metadata saved to {pool_file}")

    # Generate puzzles
    print(f"\nGenerating {NUM_PUZZLES} puzzles...")
    ground_truth = {}

    for i in range(NUM_PUZZLES):
        puzzle_id = f"hole_counting_{i:04d}"
        puzzle_data = generate_puzzle(puzzle_id, cell_pool)
        ground_truth[puzzle_id] = puzzle_data
        print(f"✓ {puzzle_id}: {puzzle_data['prompt']} ({puzzle_data['num_targets']} targets)")

    # Save ground truth
    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Successfully generated {NUM_PUZZLES} Hole Counting puzzles")
    print(f"✓ Ground truth saved to {GROUND_TRUTH_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
