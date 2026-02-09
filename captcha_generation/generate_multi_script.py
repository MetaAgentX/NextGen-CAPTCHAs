"""
Multi-Script Character Recognition CAPTCHA Generator

Task:
- Prompt shows 5 characters from multiple writing systems (Chinese, Arabic, Devanagari)
- Grid of cells, each containing 4 characters arranged in 2x2 grid
- Each character is randomly mirrored (50% chance) and rotated by random angle
- User clicks all cells containing at least 1 character from the prompt

This exploits LLM weakness in recognizing transformed characters while
humans can still identify the underlying character despite transformations.

Character Selection Criteria:
- Only strongly asymmetric characters (clearly different when mirrored/rotated)
- No characters that could be confused with others in the pool after transformation
- Balanced representation across all three scripts (~20 each)
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import random
from pathlib import Path
import math

# Configuration
CELL_SIZE = 180    # Size of each grid cell (square)
CELL_WIDTH = CELL_SIZE
CELL_HEIGHT = CELL_SIZE
CHAR_SIZE = 38     # Size of each character
GRID_COLS = 3
GRID_ROWS = 3
NUM_PUZZLES = 20
CHARS_PER_CELL = 4  # Number of characters per cell
TARGET_CHARS = 5    # Number of target characters in prompt

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "captcha_data" / "Multi_Script"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Character pools by script - balanced and carefully selected
# Criteria: strongly asymmetric, unique appearance even after mirror/rotation

# Chinese characters (~20) - characters with distinctive left-side radicals
# Excluded: characters that look similar to others when mirrored (e.g., 住/往)
CHINESE_CHARS = [
    # Speech radical (讠) - very distinctive asymmetry
    '说', '读', '话', '请', '谢', '认', '记',
    # Person radical (亻) - clear left-heavy structure
    '他', '她', '们', '你', '做', '位',
    # Water radical (氵) - distinctive three-dot pattern
    '河', '湖', '海', '洋', '流', '波',
]

# Arabic characters (~20) - letters with distinctive dot patterns
# Excluded: letters that look similar when rotated (e.g., similar base shapes)
ARABIC_CHARS = [
    # Letters with dots above - distinctive positioning
    'ت', 'ث', 'ن', 'ي',
    # Letters with dots below - clear asymmetry
    'ب', 'ج', 'خ',
    # Letters with unique shapes
    'ش', 'ض', 'ظ', 'غ', 'ف', 'ق',
    # Additional distinctive letters
    'ح', 'ع', 'ص', 'ط', 'ك', 'ل', 'م',
]

# Devanagari characters (~20) - consonants with clear asymmetric shapes
# Excluded: characters with vertical symmetry or similar appearance when rotated
DEVANAGARI_CHARS = [
    # Velar consonants - distinctive hooks and curves
    'क', 'ख', 'ग', 'घ',
    # Palatal consonants - unique shapes
    'च', 'छ', 'ज', 'झ',
    # Retroflex consonants - distinctive curves
    'ट', 'ठ', 'ड', 'ढ',
    # Dental consonants - clear asymmetry
    'त', 'थ', 'द', 'ध',
    # Additional distinctive consonants
    'प', 'फ', 'ब', 'भ',
]

# Combined pool with balanced representation
CHAR_POOL = CHINESE_CHARS + ARABIC_CHARS + DEVANAGARI_CHARS


def detect_script(char):
    """Detect which script a character belongs to."""
    code = ord(char)
    # Chinese (CJK Unified Ideographs)
    if 0x4E00 <= code <= 0x9FFF:
        return 'chinese'
    # Arabic
    if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
        return 'arabic'
    # Devanagari
    if 0x0900 <= code <= 0x097F:
        return 'devanagari'
    return 'unknown'


# Font cache to avoid repeated loading
_font_cache = {}


def get_font_for_char(char, size=50):
    """Get a font that can render the given character based on its script."""
    script = detect_script(char)
    cache_key = (script, size)

    if cache_key in _font_cache:
        return _font_cache[cache_key]

    font_paths = {
        'chinese': [
            # macOS
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            # Linux
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            # Windows
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
        ],
        'arabic': [
            # macOS
            "/System/Library/Fonts/Supplemental/GeezaPro.ttc",
            "/System/Library/Fonts/Supplemental/Al Nile.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansArabic-Regular.ttf",
            # Windows
            "C:/Windows/Fonts/arial.ttf",
        ],
        'devanagari': [
            # macOS
            "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
            "/System/Library/Fonts/Kohinoor.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            # Linux
            "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
            # Windows
            "C:/Windows/Fonts/mangal.ttf",
        ],
    }

    paths = font_paths.get(script, font_paths['chinese'])

    for path in paths:
        try:
            font = ImageFont.truetype(path, size)
            _font_cache[cache_key] = font
            return font
        except (OSError, IOError):
            continue

    print(f"Warning: No font found for {script}, using default")
    return ImageFont.load_default()


def render_single_char(char, size, mirror=False, rotation_angle=0):
    """
    Render a single Chinese character with optional mirror and rotation.

    Args:
        char: The Chinese character
        size: Base size for the character
        mirror: Whether to mirror horizontally (50% chance)
        rotation_angle: Rotation angle in degrees (continuous)

    Returns:
        PIL Image (RGBA) of the character
    """
    # Create larger canvas for rotation (to avoid clipping)
    canvas_size = int(size * 1.8)
    img = Image.new('RGBA', (canvas_size, canvas_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    font = get_font_for_char(char, size)

    # Get text bounding box
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the character
    x = (canvas_size - text_width) // 2 - bbox[0]
    y = (canvas_size - text_height) // 2 - bbox[1]

    # Draw character in black
    draw.text((x, y), char, font=font, fill=(0, 0, 0, 255))

    # Apply mirror if needed
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Apply rotation
    if rotation_angle != 0:
        img = img.rotate(rotation_angle, resample=Image.BICUBIC, expand=False)

    return img


def render_cell(chars, char_transforms):
    """
    Render a cell with 4 characters arranged in a 2x2 grid.

    Args:
        chars: List of characters to render (4 characters)
        char_transforms: List of (mirror, rotation_angle) tuples for each char

    Returns:
        PIL Image of the cell
    """
    # Create cell image
    img = Image.new('RGB', (CELL_WIDTH, CELL_HEIGHT), (255, 255, 255))

    # 2x2 grid arrangement
    char_canvas_size = int(CHAR_SIZE * 1.8)  # Match render_single_char canvas
    grid_spacing = 5

    # Calculate positions for 2x2 grid
    total_grid_size = 2 * char_canvas_size + grid_spacing
    start_x = (CELL_WIDTH - total_grid_size) // 2
    start_y = (CELL_HEIGHT - total_grid_size) // 2

    positions = [
        (start_x, start_y),                                    # Top-left
        (start_x + char_canvas_size + grid_spacing, start_y),  # Top-right
        (start_x, start_y + char_canvas_size + grid_spacing),  # Bottom-left
        (start_x + char_canvas_size + grid_spacing, start_y + char_canvas_size + grid_spacing),  # Bottom-right
    ]

    for i, (char, (mirror, angle)) in enumerate(zip(chars, char_transforms)):
        if i >= len(positions):
            break
        # Render the character
        char_img = render_single_char(char, CHAR_SIZE, mirror, angle)

        # Get position
        x, y = positions[i]

        # Paste with alpha compositing
        img.paste(char_img, (x, y), char_img)

    # Add border
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, CELL_WIDTH-1, CELL_HEIGHT-1], outline=(180, 180, 180), width=2)

    return img


def render_prompt_image(target_chars, char_size=50):
    """
    Render a prompt image showing the target characters with consistent fonts.
    Characters are shown in their original (non-transformed) form.

    Args:
        target_chars: List of characters to display
        char_size: Size of each character

    Returns:
        PIL Image of the prompt
    """
    # Calculate image dimensions
    spacing = 20  # Space between characters
    padding = 15
    char_width = char_size + spacing
    total_width = padding * 2 + len(target_chars) * char_width
    total_height = padding * 2 + char_size

    img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw each character using the appropriate font
    for i, char in enumerate(target_chars):
        font = get_font_for_char(char, char_size)
        x = padding + i * char_width + char_width // 2
        y = padding + char_size // 2

        # Get text bbox for centering
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the character
        text_x = x - text_width // 2 - bbox[0]
        text_y = y - text_height // 2 - bbox[1]

        draw.text((text_x, text_y), char, font=font, fill=(0, 0, 0))

    # Add subtle border
    draw.rectangle([0, 0, total_width-1, total_height-1], outline=(200, 200, 200), width=1)

    return img


def generate_cell_pool():
    """
    Generate pool of cell images with various character combinations.
    """
    print("Generating cell pool...")

    cell_pool = {}
    cell_idx = 0

    # Generate many cells with random character combinations
    num_cells = 500  # Generate plenty of cells

    for _ in range(num_cells):
        # Pick random characters for this cell
        chars = random.sample(CHAR_POOL, CHARS_PER_CELL)

        # Generate random transforms for each character
        transforms = []
        for _ in chars:
            mirror = random.random() < 0.5  # 50% chance of mirror
            angle = random.uniform(0, 180)  # Random rotation between 0 and 180 degrees
            transforms.append((mirror, angle))

        # Render the cell
        cell_img = render_cell(chars, transforms)

        cell_id = f"cell_{cell_idx:04d}"
        filename = f"cell_{cell_idx:04d}.png"
        cell_img.save(OUTPUT_DIR / filename)

        cell_pool[cell_id] = {
            "filename": filename,
            "characters": chars,
            "transforms": transforms,
        }

        cell_idx += 1

        if cell_idx % 100 == 0:
            print(f"  Generated {cell_idx} cells...")

    print(f"  Total cells generated: {cell_idx}")
    return cell_pool


def generate_puzzles(cell_pool):
    """
    Generate puzzles from the cell pool.
    Each puzzle has a set of target characters and a grid of cells.
    """
    print("Generating puzzles...")

    ground_truth = {}

    total_cells = GRID_COLS * GRID_ROWS

    for puzzle_idx in range(NUM_PUZZLES):
        # Select 5 random target characters
        target_chars = random.sample(CHAR_POOL, TARGET_CHARS)

        # Create prompt with characters displayed
        chars_display = '  '.join(target_chars)

        # Select cells for this puzzle
        # We need a mix of cells that contain and don't contain target chars

        all_cell_ids = list(cell_pool.keys())

        # Categorize cells
        cells_with_target = []
        cells_without_target = []

        for cell_id in all_cell_ids:
            cell_chars = set(cell_pool[cell_id]["characters"])
            if cell_chars & set(target_chars):  # Has intersection
                cells_with_target.append(cell_id)
            else:
                cells_without_target.append(cell_id)

        # Select cells: ensure we have 2-5 cells with targets, rest without
        num_with_target = random.randint(2, min(5, len(cells_with_target), total_cells - 1))
        num_without_target = total_cells - num_with_target

        if len(cells_with_target) < num_with_target:
            # Not enough cells with targets, create some dynamically
            print(f"  Warning: Puzzle {puzzle_idx} - creating additional cells with target chars")
            # Skip this for now and just use what we have
            num_with_target = min(len(cells_with_target), total_cells - 1)
            num_without_target = total_cells - num_with_target

        if len(cells_without_target) < num_without_target:
            num_without_target = len(cells_without_target)
            num_with_target = total_cells - num_without_target

        selected_with = random.sample(cells_with_target, num_with_target)
        selected_without = random.sample(cells_without_target, num_without_target)

        # Combine and shuffle
        all_cells = selected_with + selected_without
        random.shuffle(all_cells)

        # Find correct answer indices (cells containing target characters)
        correct_indices = []
        for i, cell_id in enumerate(all_cells):
            cell_chars = set(cell_pool[cell_id]["characters"])
            if cell_chars & set(target_chars):
                correct_indices.append(i)

        # Render and save prompt image with consistent fonts
        prompt_img = render_prompt_image(target_chars)
        prompt_filename = f"prompt_{puzzle_idx:04d}.png"
        prompt_img.save(OUTPUT_DIR / prompt_filename)

        puzzle_id = f"multi_script_{puzzle_idx:04d}"
        ground_truth[puzzle_id] = {
            "prompt": "Click all cells that contain ANY of these characters (characters may be rotated or mirrored):",
            "reference_image": prompt_filename,
            "description": "Multi-script character recognition with transformations",
            "target_characters": target_chars,
            "cells": all_cells,
            "answer": sorted(correct_indices),
            "input_type": "multi_script_select",
            "grid_size": [GRID_COLS, GRID_ROWS],
        }

    print(f"  Total puzzles generated: {len(ground_truth)}")
    return ground_truth


def main():
    print("=" * 60)
    print("Multi-Script Character Recognition CAPTCHA Generator")
    print(f"  Chinese: {len(CHINESE_CHARS)} chars")
    print(f"  Arabic: {len(ARABIC_CHARS)} chars")
    print(f"  Devanagari: {len(DEVANAGARI_CHARS)} chars")
    print(f"  Total: {len(CHAR_POOL)} chars")
    print("=" * 60)

    # Generate cell pool
    cell_pool = generate_cell_pool()

    # Save cell pool
    cell_pool_path = OUTPUT_DIR / "cell_pool.json"
    with open(cell_pool_path, 'w', encoding='utf-8') as f:
        # Convert transforms to serializable format
        serializable_pool = {}
        for cell_id, data in cell_pool.items():
            serializable_pool[cell_id] = {
                "filename": data["filename"],
                "characters": data["characters"],
                "transforms": [(m, a) for m, a in data["transforms"]],
            }
        json.dump(serializable_pool, f, indent=2, ensure_ascii=False)
    print(f"Saved: {cell_pool_path}")

    # Generate puzzles
    ground_truth = generate_puzzles(cell_pool)

    # Save ground truth
    ground_truth_path = OUTPUT_DIR / "ground_truth.json"
    with open(ground_truth_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    print(f"Saved: {ground_truth_path}")

    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"  Total cell images: {len(cell_pool)}")
    print(f"  Total puzzles: {len(ground_truth)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
