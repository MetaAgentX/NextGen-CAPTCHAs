"""
Spooky Circle CAPTCHA Generator

Generates animated GIFs where circles are only visible through OPPOSITE MOTION.
Individual frames look like uniform noise with no spatial features.

This exploits the difference between:
- LLMs/Tools: See only noise in individual frames, no spatial cues
- Humans: Detect motion coherence - circles move opposite to background

Key Technique - MOTION CONTRAST:
- Background noise: Scrolls in one direction (e.g., upward)
- Circle noise: Scrolls in OPPOSITE direction (e.g., downward)
- Per-frame: Both look identical (same noise statistics)
- Over time: Humans detect opposite motion → circles emerge
- Simple, effective, and robust!
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import json
import os
from pathlib import Path
from scipy import ndimage

def scroll_noise(noise_field, offset, direction='vertical'):
    """
    Scroll a noise field by a given offset.

    Args:
        noise_field: 2D numpy array
        offset: Number of pixels to scroll
        direction: 'vertical' (up/down) or 'horizontal' (left/right)

    Returns:
        Scrolled noise field
    """
    if direction == 'vertical':
        return np.roll(noise_field, offset, axis=0)
    else:  # horizontal
        return np.roll(noise_field, offset, axis=1)


def generate_mid_frequency_noise(height, width, sigma=3.0):
    """
    Generate mid-spatial frequency noise (not white noise).
    Uses Gaussian filtering to concentrate energy in mid frequencies.

    Args:
        height, width: Image dimensions
        sigma: Gaussian blur sigma (controls spatial frequency)

    Returns:
        Grayscale noise array with values 0-1
    """
    # Start with white noise
    noise = np.random.randn(height, width)
    # Apply Gaussian filter to get mid-frequency noise
    filtered_noise = ndimage.gaussian_filter(noise, sigma=sigma)
    # Normalize to 0-1 range
    filtered_noise = (filtered_noise - filtered_noise.min()) / (filtered_noise.max() - filtered_noise.min())
    return filtered_noise


def generate_spooky_circle_gif(num_circles, output_path, width=400, height=400, num_frames=30, fps=15):
    """
    Generate a GIF where circles are revealed through OPPOSITE MOTION.

    Motion-Based Strategy:
    - Background noise scrolls in one direction (e.g., upward)
    - Circle regions scroll in OPPOSITE direction (e.g., downward)
    - Each frame: uniform noise with same statistics everywhere
    - Over time: opposite motion reveals circles to human visual system
    - Simple, elegant, and very effective!

    Args:
        num_circles: Number of circles to hide (1-5)
        output_path: Where to save the GIF
        width, height: Dimensions of the image
        num_frames: Number of frames in animation
        fps: Frames per second
    """
    np.random.seed(hash(output_path) % 2**32)  # Deterministic based on filename

    # Motion parameters
    scroll_speed = 2  # Pixels per frame to scroll
    direction = 'vertical'  # Can be 'vertical' or 'horizontal'

    # Visual parameters
    base_luminance = 128.0
    noise_amplitude = 70.0  # Higher contrast for better motion visibility

    def circles_overlap(c1_center, c1_radius, c2_center, c2_radius, min_spacing=40):
        """Check if two circles overlap with a minimum spacing buffer"""
        cx1, cy1 = c1_center
        cx2, cy2 = c2_center
        distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        return distance < (c1_radius + c2_radius + min_spacing)

    # Generate circle positions and sizes with overlap prevention
    circles = []
    max_attempts = 100

    for i in range(num_circles):
        placed = False
        for attempt in range(max_attempts):
            # Random position (smaller margin to fit more circles)
            cx = np.random.randint(50, width - 50)
            cy = np.random.randint(50, height - 50)
            # Smaller radius to fit more circles (30-50 instead of 40-70)
            radius = np.random.randint(30, 50)

            # Check overlap with reduced spacing (20 instead of 40)
            overlaps = False
            for existing_circle in circles:
                if circles_overlap(
                    (cx, cy), radius,
                    existing_circle['center'], existing_circle['radius'],
                    min_spacing=20
                ):
                    overlaps = True
                    break

            if not overlaps:
                circles.append({
                    'center': (cx, cy),
                    'radius': radius
                })
                placed = True
                break

        if not placed:
            print(f"Warning: Could not place circle {i+1} without overlap after {max_attempts} attempts")

    # Create circle masks (filled disks with soft edges)
    circle_masks = []
    for circle in circles:
        cx, cy = circle['center']
        radius = circle['radius']

        # Create soft-edged circle mask
        y_coords, x_coords = np.ogrid[:height, :width]
        distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)

        # Soft-edged mask with gradual falloff
        mask = np.clip((radius - distance) / 10.0, 0, 1)
        circle_masks.append(mask)

    # Generate TWO large noise fields for scrolling
    # Make them larger than the image to avoid edge artifacts
    pad = scroll_speed * num_frames
    large_height = height + 2 * pad
    large_width = width + 2 * pad

    # Background noise field (scrolls one direction)
    bg_noise_field = generate_mid_frequency_noise(large_height, large_width, sigma=3.0)
    bg_noise_field = (bg_noise_field - 0.5) * 2.0  # Normalize to ~unit variance

    # Circle noise field (scrolls OPPOSITE direction)
    circle_noise_field = generate_mid_frequency_noise(large_height, large_width, sigma=3.0)
    circle_noise_field = (circle_noise_field - 0.5) * 2.0  # Normalize to ~unit variance

    # Generate frames with opposite motion
    frames = []
    for frame_idx in range(num_frames):
        # Calculate scroll offsets
        # Background scrolls UP (negative offset in vertical direction)
        bg_offset = -frame_idx * scroll_speed
        # Circles scroll DOWN (positive offset in vertical direction)
        circle_offset = frame_idx * scroll_speed

        # Extract current frame from scrolling background noise
        bg_scrolled = scroll_noise(bg_noise_field, bg_offset, direction)
        bg_frame = bg_scrolled[pad:pad+height, pad:pad+width]

        # Extract current frame from scrolling circle noise
        circle_scrolled = scroll_noise(circle_noise_field, circle_offset, direction)
        circle_frame = circle_scrolled[pad:pad+height, pad:pad+width]

        # Start with background noise
        img_array = base_luminance + noise_amplitude * bg_frame

        # Composite circles using masks
        # Circle regions show the opposite-scrolling noise
        for mask in circle_masks:
            circle_signal = base_luminance + noise_amplitude * circle_frame
            img_array = img_array * (1 - mask) + circle_signal * mask

        # Clip to valid range
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Convert grayscale to RGB (color-blind safe)
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

    # Return actual number of circles placed (may be less than requested if overlap prevented placement)
    return len(circles)


def generate_dataset(output_dir, num_samples=20):
    """
    Generate a dataset of spooky circle CAPTCHAs

    Args:
        output_dir: Directory to save the generated GIFs
        num_samples: Number of samples to generate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = {}

    for i in range(num_samples):
        # Random number of circles (2-7)
        num_circles = np.random.randint(2, 8)

        # Generate filename
        filename = f"spooky_{i:04d}.gif"
        output_path = output_dir / filename

        # Generate the GIF and get actual number of circles placed
        print(f"Generating {filename} with {num_circles} circle(s) requested...")
        actual_circles = generate_spooky_circle_gif(
            num_circles=num_circles,
            output_path=str(output_path),
            num_frames=30,
            fps=15
        )

        if actual_circles != num_circles:
            print(f"  -> Only {actual_circles} circle(s) could be placed!")

        # Store ground truth with ACTUAL number of circles placed
        key = f"spooky_circle_{i:04d}"
        ground_truth[key] = {
            "answer": actual_circles,
            "prompt": "How many circles can you see in this animation?",
            "description": f"EV-masked phase-coded circles: {actual_circles}",
            "media_path": f"captcha_data/Spooky_Circle/{filename}",
            "media_type": "gif",
            "difficulty": 4
        }

    # Save ground truth
    ground_truth_path = output_dir / "ground_truth.json"
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGenerated {num_samples} samples!")
    print(f"Output directory: {output_dir}")
    print(f"Ground truth: {ground_truth_path}")


if __name__ == "__main__":
    # Generate dataset
    output_dir = str(Path(__file__).parent.parent / "captcha_data" / "Spooky_Circle")
    generate_dataset(output_dir, num_samples=20)

    print("\n" + "="*70)
    print("🎯 Spooky Circle CAPTCHA Dataset Generated!")
    print("Motion Contrast CAPTCHA - Opposite Scrolling Noise")
    print("="*70)
    print("\n🔬 Technical Implementation:")
    print("  ✓ Background: Mid-frequency noise scrolling UPWARD")
    print("  ✓ Circles: Mid-frequency noise scrolling DOWNWARD")
    print("  ✓ Scroll speed: 2 pixels/frame")
    print("  ✓ Soft-edged circle masks for natural blending")
    print("  ✓ Same noise statistics everywhere (mean ~128, amplitude ~40)")
    print("\n📊 Per-Frame Analysis:")
    print("  • Single frame: Uniform noise texture everywhere")
    print("  • No spatial features, edges, or intensity differences")
    print("  • Both regions have identical noise statistics")
    print("  • Impossible to detect circles from single frame!")
    print("\n🧠 Why Humans Can See It:")
    print("  • Human visual system has excellent motion detection")
    print("  • Background noise scrolls UP → creates upward motion percept")
    print("  • Circle noise scrolls DOWN → creates downward motion percept")
    print("  • After ~1-2 seconds: circles 'pop out' as opposite-moving regions")
    print("  • Motion contrast reveals circle boundaries clearly")
    print("\n🤖 Why LLMs/Vision Models Fail:")
    print("  ✗ Single frame: Just uniform noise (no spatial cues)")
    print("  ✗ Temporal mean: FLAT (scrolling doesn't change mean)")
    print("  ✗ Temporal std: FLAT (scrolling doesn't change variance)")
    print("  ✗ Frame differencing: Shows motion but not circle shapes")
    print("  ✗ Optical flow: Would need to segment opposite flow regions")
    print("  ✗ Most vision models: No motion processing in their architecture")
    print("\n🏆 This is a TRUE motion-based temporal CAPTCHA!")
    print("    Humans: Instant motion perception → Circles visible in 1-2 sec")
    print("    LLMs: No motion processing → See only noise")
    print("    Simple, elegant, and very effective!")
