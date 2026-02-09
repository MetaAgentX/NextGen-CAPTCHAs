import os, json, time, random, base64
from pathlib import Path
from typing import List
import requests
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# Config
# ----------------------------
MODEL            = "gpt-image-1"          # official image model
N_IMAGES         = 10
BATCH_SIZE       = 6                      # number of concurrent requests
SIZE             = "1024x1024"          # 1024x1024, 1536x1024 (landscape), 1024x1536 (portrait), or auto
BACKGROUND       = "transparent"          # "transparent" or "white"
OUTPUT_FORMAT    = "png"                  # "png" or "webp" (use png for transparency)
QUALITY          = "medium"                 # "low"|"medium"|"high" (if available)
OUTDIR           = Path("out_icons")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Distinct, human-friendly colors (no black - we want vibrant colors only)
PALETTE = ["red","blue","green","yellow","purple","orange","cyan","pink","magenta","lime"]

OBJECTS = [
    "cat","dog","bird","fish","butterfly",""
    "grapes","pineapple","elephant","lion","tiger","bear","rabbit",
    "fox","owl","penguin","dolphin",
    "turtle","frog","bee","snail",
    "horse","monkey","panda",
    "flower","tree","sun","umbrella",
    "car","boat","house"
]

STYLES = [
    "simple flat hand-drawn cartoon",
    "children’s coloring-book doodle",
    "marker-drawn sketch with thick outlines",
    "flat vector-like icon with playful wobble lines"
]

# ----------------------------
# Prompt builder
# ----------------------------
def build_single_icon_prompt(obj: str, style: str, bg: str):
    # Decide whether to vary outline or fill colors
    vary_type = random.choice(["outline", "fill"])

    if vary_type == "outline":
        # Multiple outline colors (2-5), single fill color
        num_outline_colors = random.randint(1, 1)
        outline_colors = random.sample(PALETTE, num_outline_colors)
        fill_colors = [random.choice([c for c in PALETTE if c not in outline_colors])]

        color_clause = (
            f"using {num_outline_colors} vibrant outline colors: {', '.join(outline_colors)}, "
            f"and 1 fill color: {fill_colors[0]}. "
            f"Use the {num_outline_colors} outline colors ({', '.join(outline_colors)}) for different parts of the outline/border. "
            f"Use {fill_colors[0]} for ALL interior fills. "
            f"Do NOT use any other colors."
        )
    else:
        # Multiple fill colors (2-5), single outline color
        num_fill_colors = random.randint(1, 1)
        fill_colors = random.sample(PALETTE, num_fill_colors)
        outline_colors = [random.choice([c for c in PALETTE if c not in fill_colors])]

        color_clause = (
            f"using 1 vibrant outline color: {outline_colors[0]}, "
            f"and {num_fill_colors} fill colors: {', '.join(fill_colors)}. "
            f"Use {outline_colors[0]} for ALL outlines/borders. "
            f"Use the {num_fill_colors} fill colors ({', '.join(fill_colors)}) for different parts of the interior. "
            f"Do NOT use any other colors."
        )

    total_colors = len(outline_colors) + len(fill_colors)

    prompt = (
        f"A {style} of a {obj}, {color_clause} "
        f"Thick bright colored outlines, solid flat fills, clean boundaries. "
        f"No gradients, no shadows, no text, no black, no gray, no white. "
        f"ONLY use the specified {total_colors} vibrant colors. "
        f"Centered composition. {bg}. Square aspect ratio."
    )

    return prompt, outline_colors, fill_colors, vary_type

# ----------------------------
# OpenAI client
# ----------------------------
client = OpenAI()  # uses OPENAI_API_KEY from env

# ----------------------------
# Robust call with retries
# ----------------------------
def generate_image_b64(prompt: str):
    # Exponential backoff for transient errors / rate limits
    delay = 2.0
    for attempt in range(6):
        try:
            # Use the Image API for gpt-image-1
            result = client.images.generate(
                model=MODEL,
                prompt=prompt,
                size=SIZE,
                quality=QUALITY,
                output_format=OUTPUT_FORMAT,
                background=BACKGROUND,
            )
            # Get base64 encoded image
            image_base64 = result.data[0].b64_json
            return image_base64
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            if attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2.0

# ----------------------------
# Process single image
# ----------------------------
def process_single_image(i: int):
    """Generate and save a single image"""
    obj = random.choice(OBJECTS)
    style = random.choice(STYLES)
    bg_text = "white background" if BACKGROUND == "white" else "transparent background"

    prompt, outline_colors, fill_colors, vary_type = build_single_icon_prompt(obj, style, bg_text)
    image_base64 = generate_image_b64(prompt)

    # Decode and save image
    image_bytes = base64.b64decode(image_base64)

    total_colors = len(outline_colors) + len(fill_colors)

    # Create filename based on vary_type
    if vary_type == "outline":
        fname = f"icon_{i:03d}_{obj}_{total_colors}c_outline{len(outline_colors)}.{OUTPUT_FORMAT}"
    else:
        fname = f"icon_{i:03d}_{obj}_{total_colors}c_fill{len(fill_colors)}.{OUTPUT_FORMAT}"

    fpath = OUTDIR / fname
    with open(fpath, "wb") as f:
        f.write(image_bytes)

    # Create metadata
    meta = {
        "index": i,
        "file": str(fpath),
        "object": obj,
        "total_colors": total_colors,
        "outline_colors": outline_colors,
        "fill_colors": fill_colors,
        "vary_type": vary_type,
        "style": style,
        "palette": PALETTE,
        "prompt": prompt,
        "model": MODEL,
        "size": SIZE,
        "background": BACKGROUND,
        "format": OUTPUT_FORMAT,
        "quality": QUALITY,
    }

    print(f"[{i:02d}/{N_IMAGES}] saved {fname} - {len(outline_colors)} outline {outline_colors}, {len(fill_colors)} fill {fill_colors}")
    return meta

# ----------------------------
# Main loop with batching
# ----------------------------
def main():
    random.seed(7)
    records = []

    # Use ThreadPoolExecutor for concurrent image generation
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_image, i): i for i in range(1, N_IMAGES + 1)}

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                meta = future.result()
                records.append(meta)
            except Exception as e:
                i = futures[future]
                print(f"[ERROR] Image {i} failed: {e}")

    # Sort records by index to maintain order
    records.sort(key=lambda x: x["index"])

    # Save metadata
    with open(OUTDIR / "metadata.json", "w") as jf:
        json.dump(records, jf, indent=2)

    print(f"\nCompleted! Generated {len(records)} images.")

if __name__ == "__main__":
    main()
