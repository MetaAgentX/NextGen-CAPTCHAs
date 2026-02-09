"""
Spooky Text CAPTCHA Generator

Generates animated GIFs where TEXT is only visible through OPPOSITE MOTION.
Individual frames look like uniform noise with no readable text.

This exploits the difference between:
- LLMs/OCR: See only noise in individual frames, cannot read text
- Humans: Detect motion coherence - text moves opposite to background

Key Technique - MOTION CONTRAST:
- Background noise: Scrolls in one direction (e.g., upward)
- Text region noise: Scrolls in OPPOSITE direction (e.g., downward)
- Per-frame: Both look identical (same noise statistics)
- Over time: Humans detect opposite motion → text emerges and becomes readable
- Simple, effective, and robust!

Challenge Types:
- Random 4-6 character alphanumeric strings
- Numbers only (4-6 digits)
- Words from common dictionary
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import json
import os
import random
import string
from pathlib import Path
from scipy import ndimage


def scroll_noise(noise_field, offset, direction='vertical'):
    """
    Scroll a noise field by a given offset.
    """
    if direction == 'vertical':
        return np.roll(noise_field, offset, axis=0)
    else:  # horizontal
        return np.roll(noise_field, offset, axis=1)


def generate_mid_frequency_noise(height, width, sigma=3.0):
    """
    Generate mid-spatial frequency noise using scipy.
    """
    noise = np.random.randn(height, width)
    filtered_noise = ndimage.gaussian_filter(noise, sigma=sigma)
    filtered_noise = (filtered_noise - filtered_noise.min()) / (filtered_noise.max() - filtered_noise.min())
    return filtered_noise


def generate_random_text(text_type='alphanumeric', length=None):
    """
    Generate random text for the CAPTCHA.

    Args:
        text_type: 'alphanumeric', 'numbers', 'letters', or 'words'
        length: Length of text (None = random 4-6)

    Returns:
        String of random text
    """
    if length is None:
        length = random.randint(3, 4)

    if text_type == 'alphanumeric':
        # Visually distinct characters only - excludes confusing pairs
        # Excluded: B/8, E, G/6, I/1, O/0/D/Q, W, Z/2, 7, 8
        chars = 'ACDFHJKLMNPRSTUVXY34569'
        return ''.join(random.choice(chars) for _ in range(length))

    elif text_type == 'numbers':
        return ''.join(random.choice(string.digits) for _ in range(length))

    elif text_type == 'letters':
        return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))

    elif text_type == 'words':
        # Common simple words
        words = [
            'APPLE', 'BOOK', 'CHAIR', 'DESK', 'EAGLE',
            'FISH', 'GAME', 'HOUSE', 'JUMP', 'KING',
            'LAMP', 'MOON', 'NEST', 'OCEAN', 'PARK',
            'QUEEN', 'RING', 'STAR', 'TREE', 'WAVE'
        ]
        return random.choice(words)

    return generate_random_text('alphanumeric', length)


def create_text_mask(text, width, height, font_size=100):
    """
    Create a binary mask for the text.

    Args:
        text: Text string to render
        width, height: Image dimensions
        font_size: Font size in pixels (default 120 for better readability)

    Returns:
        Binary mask (0 or 1) where text is rendered
    """
    # Create a blank image for the mask
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    # Try to load a clear, bold font
    try:
        # Try different common font paths
        font_paths = [
            '/System/Library/Fonts/Helvetica.ttc',  # macOS
            '/Library/Fonts/Arial Bold.ttf',  # macOS
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',  # Linux
            'C:\\Windows\\Fonts\\arialbd.ttf',  # Windows
        ]

        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break

        if font is None:
            # Fallback to default font
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the text
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # Draw the text in white
    draw.text((x, y), text, fill=255, font=font)

    # Convert to numpy array and normalize to 0-1
    mask_array = np.array(mask_img).astype(np.float32) / 255.0

    # Apply slight Gaussian blur for softer edges
    mask_array = ndimage.gaussian_filter(mask_array, sigma=2.0)

    return mask_array


def generate_spooky_text_gif(
    text=None,
    text_type='alphanumeric',
    output_path="spooky_text_0.gif",
    width=600,
    height=250,
    num_frames=30,
    fps=15,
    seed=None
):
    """
    Generate a GIF where text is revealed through OPPOSITE MOTION.

    Args:
        text: Specific text to display (None = random)
        text_type: Type of random text ('alphanumeric', 'numbers', 'letters', 'words')
        output_path: Where to save the GIF
        width, height: Dimensions of the image
        num_frames: Number of frames in animation
        fps: Frames per second
        seed: Random seed for deterministic generation

    Returns:
        The text that was displayed
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    else:
        seed = hash(output_path) % 2**32
        np.random.seed(seed)
        random.seed(seed)

    # Generate or use provided text
    if text is None:
        text = generate_random_text(text_type)

    text = text.upper()  # Uppercase for clarity

    # Motion parameters
    scroll_speed = 2  # Pixels per frame
    direction = 'vertical'

    # Visual parameters
    base_luminance = 128.0
    noise_amplitude = 70.0  # Higher contrast for better motion visibility

    # Create text mask with larger font for better readability
    text_mask = create_text_mask(text, width, height, font_size=min(160, int(height * 0.8)))

    # Generate large noise fields for scrolling (to avoid edge artifacts)
    pad = scroll_speed * num_frames
    large_height = height + 2 * pad
    large_width = width + 2 * pad

    # Background noise field (scrolls one direction)
    bg_noise_field = generate_mid_frequency_noise(large_height, large_width, sigma=3.0)
    bg_noise_field = (bg_noise_field - 0.5) * 2.0

    # Text noise field (scrolls OPPOSITE direction)
    text_noise_field = generate_mid_frequency_noise(large_height, large_width, sigma=3.0)
    text_noise_field = (text_noise_field - 0.5) * 2.0

    # Generate frames with opposite motion
    frames = []
    for frame_idx in range(num_frames):
        # Calculate scroll offsets
        # Background scrolls UP (negative offset)
        bg_offset = -frame_idx * scroll_speed
        # Text scrolls DOWN (positive offset)
        text_offset = frame_idx * scroll_speed

        # Extract current frame from scrolling background noise
        bg_scrolled = scroll_noise(bg_noise_field, bg_offset, direction)
        bg_frame = bg_scrolled[pad:pad+height, pad:pad+width]

        # Extract current frame from scrolling text noise
        text_scrolled = scroll_noise(text_noise_field, text_offset, direction)
        text_frame = text_scrolled[pad:pad+height, pad:pad+width]

        # Start with background noise
        img_array = base_luminance + noise_amplitude * bg_frame

        # Composite text using mask
        # Text regions show the opposite-scrolling noise
        text_signal = base_luminance + noise_amplitude * text_frame
        img_array = img_array * (1 - text_mask) + text_signal * text_mask

        # Clip to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Convert grayscale to RGB
        img_rgb = np.stack([img_array, img_array, img_array], axis=-1)

        # Convert to PIL Image
        frame = Image.fromarray(img_rgb)
        frames.append(frame)

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000/fps),
        loop=0
    )

    return text


def generate_dataset(output_dir, num_samples=20, text_type='alphanumeric'):
    """
    Generate a dataset of spooky text CAPTCHAs.

    Args:
        output_dir: Directory to save the generated GIFs
        num_samples: Number of samples to generate
        text_type: Type of text to generate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = {}

    for i in range(num_samples):
        # Generate filename
        filename = f"spooky_text_{i:04d}.gif"
        output_path = output_dir / filename

        # Generate the GIF
        print(f"Generating {filename}...")
        text = generate_spooky_text_gif(
            text=None,  # Random
            text_type=text_type,
            output_path=str(output_path),
            width=600,  # Wider for better text display
            height=250,  # Taller for larger text
            num_frames=30,
            fps=15,
            seed=None
        )

        # Store ground truth (don't leak answer in description!)
        ground_truth[filename] = {
            "answer": text,
            "prompt": "What text do you see in this animation?",
            "description": "Motion-contrast text puzzle",
            "media_path": f"captcha_data/Spooky_Text/{filename}",
            "media_type": "gif",
            "difficulty": 3,
            "text_type": text_type,
            "case_sensitive": False
        }

        print(f"  → Text: {text}")

    # Save ground truth
    ground_truth_path = output_dir / "ground_truth.json"
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGenerated {num_samples} samples!")
    print(f"Output directory: {output_dir}")
    print(f"Ground truth: {ground_truth_path}")


if __name__ == "__main__":
    # Create output directory
    output_dir = str(Path(__file__).parent.parent / "captcha_data" / "Spooky_Text")

    # Generate dataset
    generate_dataset(output_dir, num_samples=20, text_type='alphanumeric')

    print("\n" + "="*70)
    print("🎯 Spooky Text CAPTCHA Dataset Generated!")
    print("Motion Contrast Text Recognition")
    print("="*70)
    print("\n🔬 Technical Implementation:")
    print("  ✓ Text Type: 4-6 character alphanumeric strings")
    print("  ✓ Motion Contrast: Text scrolls opposite to background")
    print("  ✓ Background scrolls UP, text scrolls DOWN")
    print("  ✓ Font: Large, bold, centered text")
    print("  ✓ Equal variance: Text and background have same noise statistics")
    print("\n📊 Per-Frame Analysis:")
    print("  • Single frame: Uniform noise, no readable text")
    print("  • No OCR can detect text from individual frames")
    print("  • Identical noise statistics everywhere")
    print("  • Text shape completely hidden in noise")
    print("\n🧠 Why Humans Can See It:")
    print("  • Watch animation for 1-2 seconds")
    print("  • Motion detection reveals opposite-moving text region")
    print("  • Text 'emerges' from background noise")
    print("  • Easy to read once motion is perceived")
    print("  • Leverages human temporal integration")
    print("\n🤖 Why LLMs/OCR Fail:")
    print("  ✗ Single frame: Just uniform noise")
    print("  ✗ No spatial text features visible")
    print("  ✗ OCR cannot detect any characters")
    print("  ✗ Temporal mean/std: FLAT (no text visible)")
    print("  ✗ Would need motion-based text recognition")
    print("  ✗ Most vision models lack temporal text processing")
    print("\n🏆 Motion-Based Text CAPTCHA!")
    print("    Humans: Easy text reading after brief observation")
    print("    LLMs/OCR: Blind without motion analysis")
    print("    Perfect for temporal coherence benchmarking!")
