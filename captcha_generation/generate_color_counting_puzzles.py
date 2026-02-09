import json
import random
from pathlib import Path

# Paths
CAPTCHA_DATA_DIR = Path(__file__).parent.parent / "captcha_data" / "Color_Counting"
OUTPUT_DIR = CAPTCHA_DATA_DIR
GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth.json"

# Number of puzzles to generate
NUM_PUZZLES = 20

def get_available_sketches():
    """Parse all sketch files and organize by color count."""
    sketches_by_color = {2: [], 3: [], 4: [], 5: [], 6: []}

    for sketch_file in CAPTCHA_DATA_DIR.glob("*.png"):
        filename = sketch_file.name
        # Parse: {color_count}_{object}_{variation}.png
        parts = filename.replace(".png", "").split("_")
        if len(parts) >= 2:
            try:
                color_count = int(parts[0])
                if color_count in sketches_by_color:
                    sketches_by_color[color_count].append(filename)
            except ValueError:
                continue

    return sketches_by_color

def generate_puzzle(puzzle_id, sketches_by_color):
    """Generate a single Color_Counting puzzle."""
    # Only two question types: "less_equal_3" or "more_than_3"
    comparisons = ["less_equal_3", "more_than_3"]
    comparison = random.choice(comparisons)

    # Fixed target count of 3
    target_count = 3

    # Determine which color counts satisfy the condition
    if comparison == "less_equal_3":
        prompt = "Click all sketches with 3 or fewer colors (ignore white)"
        satisfying_counts = [c for c in sketches_by_color.keys() if c <= 3]
    else:  # more_than_3
        prompt = "Click all sketches with more than 3 colors (ignore white)"
        satisfying_counts = [c for c in sketches_by_color.keys() if c > 3]

    # Filter out empty categories
    satisfying_counts = [c for c in satisfying_counts if sketches_by_color[c]]
    non_satisfying_counts = [c for c in sketches_by_color.keys()
                            if c not in satisfying_counts and sketches_by_color[c]]

    if not satisfying_counts or not non_satisfying_counts:
        # Skip if we can't create a valid puzzle
        return None

    # Decide how many target sketches to include (4-10 out of 16)
    num_targets = random.randint(4, 10)
    num_distractors = 16 - num_targets

    # Select target sketches
    target_sketches = []
    for _ in range(num_targets):
        color_count = random.choice(satisfying_counts)
        sketch = random.choice(sketches_by_color[color_count])
        target_sketches.append(sketch)

    # Select distractor sketches
    distractor_sketches = []
    for _ in range(num_distractors):
        color_count = random.choice(non_satisfying_counts)
        sketch = random.choice(sketches_by_color[color_count])
        distractor_sketches.append(sketch)

    # Combine and shuffle
    all_sketches = target_sketches + distractor_sketches

    # Create position mapping before shuffle
    positions = list(range(16))
    random.shuffle(positions)

    # Map target indices to their shuffled positions
    answer = sorted([positions[i] for i in range(num_targets)])

    # Shuffle the sketch list according to positions
    shuffled_sketches = [None] * 16
    for i, pos in enumerate(positions):
        shuffled_sketches[pos] = all_sketches[i]

    puzzle_data = {
        "prompt": prompt,
        "description": f"Grid with {num_targets} sketch(es) matching: {comparison.replace('_', ' ')} colors",
        "options": shuffled_sketches,
        "answer": answer,
        "grid_size": [4, 4],
        "difficulty": 5,
        "media_type": "image",
        "target_comparison": comparison,
        "target_count": target_count,
        "num_targets": num_targets
    }

    return puzzle_data

def main():
    print("Generating Color_Counting CAPTCHA puzzles...")

    # Get available sketches
    sketches_by_color = get_available_sketches()

    print(f"\nAvailable sketches by color count:")
    for color_count, sketches in sorted(sketches_by_color.items()):
        print(f"  {color_count} colors: {len(sketches)} sketches")

    # Generate puzzles
    ground_truth = {}
    successful_puzzles = 0

    for i in range(NUM_PUZZLES):
        puzzle_id = f"color_counting_{i:04d}"
        puzzle_data = generate_puzzle(puzzle_id, sketches_by_color)

        if puzzle_data:
            ground_truth[puzzle_id] = puzzle_data
            successful_puzzles += 1
            print(f"✓ Generated {puzzle_id}: {puzzle_data['prompt']}")

    # Save ground truth
    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Successfully generated {successful_puzzles} Color_Counting puzzles")
    print(f"✓ Ground truth saved to {GROUND_TRUTH_FILE}")

if __name__ == "__main__":
    main()
