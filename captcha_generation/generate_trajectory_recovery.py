"""
Trajectory Recovery CAPTCHA Generator

Creates animated GIFs showing a ball moving along a path, then generates
a 4x4 grid of trajectory options where users must select all correct paths.

Key Features:
- Ball movement GIF with motion blur and realistic physics
- 4x4 grid of trajectory paths in different colors
- 1-3 correct trajectories (variations of the true path)
- Distractors with similar but incorrect paths
- Multiple trajectory types: straight, curved, zigzag, circular, wave, etc.

Human capability: Track moving objects and remember spatial paths
LLM/Bot challenge: Requires temporal tracking + spatial memory + pattern matching
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import json
import os
from pathlib import Path


class TrajectoryGenerator:
    """Generate various types of ball trajectories"""

    @staticmethod
    def linear(t, start, end):
        """Linear trajectory from start to end"""
        return start + t * (end - start)

    @staticmethod
    def bezier_quadratic(t, p0, p1, p2):
        """Quadratic Bezier curve"""
        return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2

    @staticmethod
    def bezier_cubic(t, p0, p1, p2, p3):
        """Cubic Bezier curve"""
        return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

    @staticmethod
    def circular(t, center, radius, start_angle, end_angle):
        """Circular arc trajectory"""
        angle = start_angle + t * (end_angle - start_angle)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        return np.array([x, y])

    @staticmethod
    def wave(t, start, end, amplitude, frequency):
        """Wave trajectory"""
        base = start + t * (end - start)
        direction = end - start
        perpendicular = np.array([-direction[1], direction[0]])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        offset = amplitude * np.sin(2 * np.pi * frequency * t) * perpendicular
        return base + offset

    @staticmethod
    def zigzag(t, start, end, amplitude, num_zigs):
        """Zigzag trajectory"""
        base = start + t * (end - start)
        direction = end - start
        perpendicular = np.array([-direction[1], direction[0]])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        # Triangle wave
        phase = (t * num_zigs) % 1.0
        offset_amount = amplitude * (1 - 2 * abs(phase - 0.5))
        offset = offset_amount * perpendicular
        return base + offset

    @staticmethod
    def spiral(t, center, start_radius, end_radius, rotations):
        """Spiral trajectory"""
        radius = start_radius + t * (end_radius - start_radius)
        angle = 2 * np.pi * rotations * t
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        return np.array([x, y])


def generate_trajectory_points(traj_type, params, num_points=100):
    """
    Generate trajectory points based on type and parameters

    Args:
        traj_type: Type of trajectory
        params: Dictionary of parameters for the trajectory
        num_points: Number of points to generate

    Returns:
        Array of (x, y) points
    """
    t_values = np.linspace(0, 1, num_points)
    points = []

    for t in t_values:
        if traj_type == 'linear':
            point = TrajectoryGenerator.linear(t, params['start'], params['end'])
        elif traj_type == 'quadratic':
            point = TrajectoryGenerator.bezier_quadratic(
                t, params['p0'], params['p1'], params['p2']
            )
        elif traj_type == 'cubic':
            point = TrajectoryGenerator.bezier_cubic(
                t, params['p0'], params['p1'], params['p2'], params['p3']
            )
        elif traj_type == 'circular':
            point = TrajectoryGenerator.circular(
                t, params['center'], params['radius'],
                params['start_angle'], params['end_angle']
            )
        elif traj_type == 'wave':
            point = TrajectoryGenerator.wave(
                t, params['start'], params['end'],
                params['amplitude'], params['frequency']
            )
        elif traj_type == 'zigzag':
            point = TrajectoryGenerator.zigzag(
                t, params['start'], params['end'],
                params['amplitude'], params['num_zigs']
            )
        elif traj_type == 'spiral':
            point = TrajectoryGenerator.spiral(
                t, params['center'], params['start_radius'],
                params['end_radius'], params['rotations']
            )
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}")

        points.append(point)

    return np.array(points)


def create_ball_frame(width, height, position, ball_radius=12, ball_color=(255, 100, 100)):
    """Create a frame with a ball at the given position"""
    img = Image.new('RGB', (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    x, y = position
    # Draw ball with gradient effect (simple 3D look)
    for i in range(ball_radius, 0, -1):
        alpha = int(255 * (i / ball_radius))
        color = tuple(int(c * (i / ball_radius)) for c in ball_color)
        draw.ellipse(
            [x - i, y - i, x + i, y + i],
            fill=color
        )

    return img


def create_varied_timing(num_points, num_frames):
    """
    Create non-uniform timing for ball movement with acceleration/deceleration
    Returns indices into trajectory points for each frame

    This makes the ball speed vary naturally - slowing at curves, speeding on straights
    IMPORTANT: Always moves forward (monotonically increasing indices)
    """
    # Use ease-in-ease-out function for smooth acceleration
    t = np.linspace(0, 1, num_frames)

    # Apply easing function (sinusoidal ease-in-out)
    # This creates smooth acceleration at start and deceleration at end
    eased = (1 - np.cos(t * np.pi)) / 2

    # Add slight speed variations (but keep monotonic)
    # Use smooth random variations that don't reverse direction
    variation = np.random.randn(num_frames) * 0.02
    # Make it smoother with a moving average
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    variation = np.convolve(variation, kernel, mode='same')

    # Apply variation but ensure it stays in [0, 1] range
    combined = eased + variation

    # Ensure monotonic increase (no backward movement)
    # Clip to valid range first
    combined = np.clip(combined, 0, 1)

    # Force monotonic increase
    for i in range(1, len(combined)):
        if combined[i] < combined[i-1]:
            combined[i] = combined[i-1]

    # Normalize to [0, 1] ensuring we reach the end
    combined = combined - combined.min()
    if combined.max() > 0:
        combined = combined / combined.max()

    # Map to trajectory point indices
    indices = (combined * (num_points - 1)).astype(int)

    # Final safety check: ensure strictly non-decreasing
    for i in range(1, len(indices)):
        if indices[i] < indices[i-1]:
            indices[i] = indices[i-1]

    return indices


def generate_ball_movement_gif(trajectory_points, output_path, width=400, height=400,
                                fps=20, ball_radius=12, ball_color=(255, 100, 100),
                                total_duration=3.0):
    """
    Generate an animated GIF showing a ball moving along the trajectory with varied timing

    The ball will:
    - Accelerate and decelerate naturally
    - Slow down at curves
    - Speed up on straighter sections
    - Have slight timing variations

    This makes it easy for humans to track but harder for LLMs to extract the exact path

    Args:
        trajectory_points: Array of (x, y) points
        output_path: Where to save the GIF
        width, height: Dimensions
        fps: Frames per second
        ball_radius: Radius of the ball
        ball_color: RGB color of the ball
        total_duration: Total duration in seconds
    """
    # Clamp trajectory points to stay within boundaries (with padding for ball radius)
    padding = ball_radius + 5
    trajectory_points = trajectory_points.copy()
    trajectory_points[:, 0] = np.clip(trajectory_points[:, 0], padding, width - padding)
    trajectory_points[:, 1] = np.clip(trajectory_points[:, 1], padding, height - padding)

    num_frames = int(total_duration * fps)
    num_points = len(trajectory_points)

    # Get varied timing indices
    timing_indices = create_varied_timing(num_points, num_frames)

    frames = []
    frame_durations = []

    for i, point_idx in enumerate(timing_indices):
        point = trajectory_points[point_idx]

        # Create frame
        frame = create_ball_frame(width, height, point, ball_radius, ball_color)

        # Add motion blur effect based on speed
        if i > 0:
            prev_idx = timing_indices[i-1]
            speed = abs(point_idx - prev_idx)
            if speed > 2:
                # More blur for faster movement
                frame = frame.filter(ImageFilter.SMOOTH_MORE)
            elif speed > 0:
                frame = frame.filter(ImageFilter.SMOOTH)

        frames.append(frame)

        # Vary frame duration slightly for more natural movement
        base_duration = int(1000 / fps)
        variation = np.random.randint(-5, 6)  # ±5ms variation
        frame_durations.append(max(20, base_duration + variation))

    # Add pause frames at the start and end for better viewing
    first_frame = frames[0]
    last_frame = frames[-1]

    # Longer pause at start (0.8 seconds) so humans can prepare
    pause_frames = [first_frame] * int(fps * 0.8)
    pause_durations = [100] * len(pause_frames)

    # Longer pause at end (1.2 seconds) so humans can review the path
    end_pause_frames = [last_frame] * int(fps * 1.2)
    end_pause_durations = [100] * len(end_pause_frames)

    # Combine all frames
    all_frames = pause_frames + frames + end_pause_frames
    all_durations = pause_durations + frame_durations + end_pause_durations

    # Save as GIF with varied durations
    all_frames[0].save(
        output_path,
        save_all=True,
        append_images=all_frames[1:],
        duration=all_durations,
        loop=0
    )


def draw_trajectory_path(draw, points, color, line_width=3, line_style='solid'):
    """
    Draw a trajectory path on an ImageDraw object with different line styles

    Args:
        draw: ImageDraw object
        points: Array of (x, y) points
        color: RGB color tuple
        line_width: Width of the line
        line_style: 'solid', 'dashed', or 'dotted'
    """
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        if line_style == 'solid':
            # Draw solid line
            draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
        elif line_style == 'dashed':
            # Draw dashed line (longer segments)
            segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if segment_length > 0:
                dash_length = 8
                gap_length = 6
                num_dashes = int(segment_length / (dash_length + gap_length))

                for j in range(num_dashes + 1):
                    t_start = j * (dash_length + gap_length) / segment_length
                    t_end = min(1.0, (j * (dash_length + gap_length) + dash_length) / segment_length)

                    if t_start < 1.0:
                        dash_x1 = x1 + t_start * (x2 - x1)
                        dash_y1 = y1 + t_start * (y2 - y1)
                        dash_x2 = x1 + t_end * (x2 - x1)
                        dash_y2 = y1 + t_end * (y2 - y1)
                        draw.line([(dash_x1, dash_y1), (dash_x2, dash_y2)], fill=color, width=line_width)
        elif line_style == 'dotted':
            # Draw dotted line (small dots)
            segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if segment_length > 0:
                dot_spacing = 5
                num_dots = int(segment_length / dot_spacing)

                for j in range(num_dots + 1):
                    t = j * dot_spacing / segment_length
                    if t <= 1.0:
                        dot_x = x1 + t * (x2 - x1)
                        dot_y = y1 + t * (y2 - y1)
                        r = line_width // 2 + 1
                        draw.ellipse([dot_x - r, dot_y - r, dot_x + r, dot_y + r], fill=color)


def create_individual_trajectory_image(trajectory, output_path, color, cell_size=150, line_style='solid'):
    """
    Create a single trajectory image (like individual sketch images in Color_Counting)

    Args:
        trajectory: The trajectory points
        output_path: Where to save the image
        color: RGB color tuple for the trajectory
        cell_size: Size of the image in pixels
        line_style: 'solid', 'dashed', or 'dotted'
    """
    img = Image.new('RGB', (cell_size, cell_size), (255, 255, 255))

    # Scale trajectory to fit in cell with padding
    padding = 20
    traj = trajectory.copy()
    traj_min = traj.min(axis=0)
    traj_max = traj.max(axis=0)
    traj_range = traj_max - traj_min

    # Avoid division by zero
    traj_range = np.where(traj_range == 0, 1, traj_range)

    scaled_traj = (traj - traj_min) / traj_range
    scaled_traj = scaled_traj * (cell_size - 2 * padding) + padding

    # Draw trajectory with line style
    draw = ImageDraw.Draw(img)
    draw_trajectory_path(draw, scaled_traj, color, line_width=3, line_style=line_style)

    # Save image
    img.save(output_path)


def create_trajectory_images(correct_trajectories, distractors, output_dir, captcha_id):
    """
    Create individual trajectory images (like Color_Counting pattern)

    Args:
        correct_trajectories: List of correct trajectory variations
        distractors: List of distractor trajectories
        output_dir: Directory to save images
        captcha_id: ID for this captcha

    Returns:
        tuple: (list of all image filenames, list of correct indices)
    """
    # Colors for trajectories
    colors = [
        (255, 80, 80),   # Red
        (80, 150, 255),  # Blue
        (100, 200, 100), # Green
        (255, 180, 80),  # Orange
        (200, 100, 200), # Purple
        (80, 200, 200),  # Cyan
        (255, 150, 150), # Pink
        (150, 150, 255), # Light Blue
        (200, 200, 100), # Yellow-green
        (255, 130, 180), # Hot Pink
        (100, 255, 200), # Mint
        (255, 200, 100), # Gold
        (180, 150, 255), # Lavender
        (255, 150, 100), # Coral
        (150, 255, 150), # Light Green
        (200, 150, 200)  # Mauve
    ]

    # Combine correct trajectories and distractors
    all_trajectories = correct_trajectories + distractors

    # Shuffle while tracking correct indices
    indices = list(range(len(all_trajectories)))
    np.random.shuffle(indices)

    correct_positions = []
    image_filenames = []

    # Create individual images
    for idx, traj_idx in enumerate(indices):
        # Track if this is a correct trajectory
        if traj_idx < len(correct_trajectories):
            correct_positions.append(idx)

        # Generate filename
        filename = f"traj_{captcha_id}_{idx}.png"
        filepath = output_dir / filename

        # Get trajectory and color
        traj = all_trajectories[traj_idx]
        color = colors[idx % len(colors)]

        # Create individual image
        create_individual_trajectory_image(traj, str(filepath), color, cell_size=150)

        image_filenames.append(filename)

    return image_filenames, correct_positions


def create_trajectory_variations(base_trajectory, num_variations=2):
    """
    Create variations that are IDENTICAL to base (just different visual styles)
    No noise - human can't recognize noisy trajectories!
    The variations are just for different color/line style combinations
    """
    variations = []

    for i in range(num_variations):
        # Just copy the base trajectory - NO NOISE!
        # Variations will have different colors/line styles, that's enough
        varied = base_trajectory.copy()
        variations.append(varied)

    return variations


def create_distractor_trajectories(base_trajectory, base_type, base_params, num_distractors=13):
    """
    Create distractor trajectories that are hard for LLMs but easy for humans to distinguish

    Strategy:
    - Slight spatial shifts (hard for pixel-level comparison, easy for humans)
    - Rotations (preserves shape but changes orientation)
    - Different trajectory types with similar visual complexity
    - Scale variations (larger/smaller but same shape)
    """
    distractors = []

    # Strategy 1: Rotations (hard for LLMs, easy for humans to see orientation)
    num_rotations = min(4, num_distractors // 4)
    for i in range(num_rotations):
        rotated = base_trajectory.copy()
        # Rotate around center
        center = rotated.mean(axis=0)
        angle = (i + 1) * np.pi / 6  # 30, 60, 90, 120 degrees
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        centered = rotated - center
        rotated_points = np.zeros_like(centered)
        rotated_points[:, 0] = cos_a * centered[:, 0] - sin_a * centered[:, 1]
        rotated_points[:, 1] = sin_a * centered[:, 0] + cos_a * centered[:, 1]
        rotated = rotated_points + center

        distractors.append(rotated)

    # Strategy 2: Spatial shifts (same shape, different position - hard for LLMs)
    num_shifts = min(3, num_distractors // 4)
    for i in range(num_shifts):
        shifted = base_trajectory.copy()
        shift_amount = np.array([
            np.random.randint(-60, 60),
            np.random.randint(-60, 60)
        ])
        shifted += shift_amount
        distractors.append(shifted)

    # Strategy 3: Scale variations (larger/smaller but same shape)
    num_scales = min(3, num_distractors // 4)
    for i in range(num_scales):
        scaled = base_trajectory.copy()
        center = scaled.mean(axis=0)
        scale_factor = 0.7 + i * 0.2  # 0.7, 0.9, 1.1, 1.3
        scaled = center + (scaled - center) * scale_factor
        distractors.append(scaled)

    # Strategy 4: Mirror/flip (humans easily recognize symmetry)
    num_mirrors = min(2, num_distractors // 4)
    for i in range(num_mirrors):
        flipped = base_trajectory.copy()
        center = flipped.mean(axis=0)
        if i % 2 == 0:
            # Flip horizontally
            flipped[:, 0] = 2 * center[0] - flipped[:, 0]
        else:
            # Flip vertically
            flipped[:, 1] = 2 * center[1] - flipped[:, 1]
        distractors.append(flipped)

    # Strategy 5: Slight parameter modifications (very similar but subtly different)
    while len(distractors) < num_distractors:
        modified_params = base_params.copy()

        if base_type == 'linear':
            # Small shift to start or end
            if np.random.random() > 0.5:
                modified_params['start'] = base_params['start'] + np.random.randn(2) * 25
            else:
                modified_params['end'] = base_params['end'] + np.random.randn(2) * 25

        elif base_type in ['quadratic', 'cubic']:
            # Small modification to one control point
            for key in modified_params:
                if key.startswith('p'):
                    modified_params[key] = modified_params[key] + np.random.randn(2) * 20

        elif base_type == 'circular':
            # Slightly different radius or angles
            modified_params['radius'] = base_params['radius'] + np.random.randn() * 15
            modified_params['start_angle'] = base_params['start_angle'] + np.random.randn() * 0.3

        elif base_type == 'wave':
            # Different amplitude or frequency (changes wave pattern)
            modified_params['amplitude'] = base_params['amplitude'] + np.random.randn() * 8
            modified_params['frequency'] = max(1, base_params['frequency'] + np.random.choice([-1, 1]))

        elif base_type == 'zigzag':
            # Different number of zigs or amplitude
            modified_params['amplitude'] = base_params['amplitude'] + np.random.randn() * 8
            modified_params['num_zigs'] = max(2, base_params['num_zigs'] + np.random.choice([-1, 1]))

        traj = generate_trajectory_points(base_type, modified_params)
        distractors.append(traj)

    return distractors[:num_distractors]


def generate_random_trajectory_params(width=400, height=400):
    """Generate random parameters for a trajectory with larger movements"""
    traj_types = ['linear', 'quadratic', 'cubic', 'circular', 'wave', 'zigzag']
    traj_type = np.random.choice(traj_types)

    # Smaller margin to allow larger movements (boundary clamping will keep ball in bounds)
    margin = 30
    # Define corners and edges to ensure paths span significant distances
    corners = [
        (margin, margin),  # Top-left
        (width - margin, margin),  # Top-right
        (margin, height - margin),  # Bottom-left
        (width - margin, height - margin),  # Bottom-right
    ]
    edges = [
        (width // 2, margin),  # Top-center
        (width // 2, height - margin),  # Bottom-center
        (margin, height // 2),  # Left-center
        (width - margin, height // 2),  # Right-center
    ]
    all_positions = corners + edges

    if traj_type == 'linear':
        # Ensure start and end are far apart (different corners/edges)
        positions = np.random.choice(len(all_positions), 2, replace=False)
        start = np.array(all_positions[positions[0]], dtype=float)
        end = np.array(all_positions[positions[1]], dtype=float)
        # Add some randomness
        start += np.random.randn(2) * 30
        end += np.random.randn(2) * 30
        params = {'start': start, 'end': end}

    elif traj_type == 'quadratic':
        # Start and end at opposite sides, control point creates curve
        positions = np.random.choice(len(all_positions), 2, replace=False)
        p0 = np.array(all_positions[positions[0]], dtype=float) + np.random.randn(2) * 20
        p2 = np.array(all_positions[positions[1]], dtype=float) + np.random.randn(2) * 20
        # Control point in middle area but offset to create nice curve
        p1 = np.array([width / 2 + np.random.randn() * 80,
                       height / 2 + np.random.randn() * 80], dtype=float)
        params = {'p0': p0, 'p1': p1, 'p2': p2}

    elif traj_type == 'cubic':
        # Start and end at opposite corners for maximum span
        positions = np.random.choice(len(all_positions), 2, replace=False)
        p0 = np.array(all_positions[positions[0]], dtype=float) + np.random.randn(2) * 20
        p3 = np.array(all_positions[positions[1]], dtype=float) + np.random.randn(2) * 20
        # Control points create interesting S-curves
        p1 = np.array([np.random.randint(margin, width - margin),
                       np.random.randint(margin, height - margin)], dtype=float)
        p2 = np.array([np.random.randint(margin, width - margin),
                       np.random.randint(margin, height - margin)], dtype=float)
        params = {'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3}

    elif traj_type == 'circular':
        # Center the circle and use larger radius for more visible arcs
        center = np.array([width / 2 + np.random.randn() * 30,
                          height / 2 + np.random.randn() * 30])
        # Larger radius for more visible circular motion
        radius = np.random.randint(100, 160)
        start_angle = np.random.random() * 2 * np.pi
        # Arc between 180 and 300 degrees for more visible movement
        end_angle = start_angle + np.pi * (1.0 + np.random.random() * 0.67)
        params = {
            'center': center,
            'radius': radius,
            'start_angle': start_angle,
            'end_angle': end_angle
        }

    elif traj_type == 'wave':
        # Wave from one side to another
        positions = np.random.choice(len(all_positions), 2, replace=False)
        start = np.array(all_positions[positions[0]], dtype=float) + np.random.randn(2) * 20
        end = np.array(all_positions[positions[1]], dtype=float) + np.random.randn(2) * 20
        # Larger amplitude for more visible waves
        amplitude = np.random.randint(40, 70)
        frequency = np.random.randint(2, 4)
        params = {
            'start': start,
            'end': end,
            'amplitude': amplitude,
            'frequency': frequency
        }

    elif traj_type == 'zigzag':
        # Zigzag from one corner to another
        positions = np.random.choice(len(all_positions), 2, replace=False)
        start = np.array(all_positions[positions[0]], dtype=float) + np.random.randn(2) * 20
        end = np.array(all_positions[positions[1]], dtype=float) + np.random.randn(2) * 20
        # Larger amplitude for more visible zigzags
        amplitude = np.random.randint(40, 60)
        num_zigs = np.random.randint(3, 5)
        params = {
            'start': start,
            'end': end,
            'amplitude': amplitude,
            'num_zigs': num_zigs
        }

    return traj_type, params


def generate_trajectory_captcha(output_dir, captcha_id, width=400, height=400):
    """
    Generate a complete trajectory recovery CAPTCHA

    Returns:
        Dictionary with metadata about the CAPTCHA
    """
    output_dir = Path(output_dir)

    # Generate random trajectory
    traj_type, params = generate_random_trajectory_params(width, height)
    base_trajectory = generate_trajectory_points(traj_type, params, num_points=60)

    # Generate GIF of ball movement with varied timing (slower for humans)
    gif_filename = f"trajectory_{captcha_id}_movement.gif"
    gif_path = output_dir / gif_filename
    generate_ball_movement_gif(
        base_trajectory,
        str(gif_path),
        width=width,
        height=height,
        fps=15,  # Reduced from 20 to make movement smoother and slower
        ball_radius=14,  # Slightly larger ball for better visibility
        ball_color=(255, 80, 80),  # Brighter red
        total_duration=5.0  # Increased from 3.5 to 5 seconds for easier tracking
    )

    # Create variations (correct answers)
    num_correct = np.random.randint(1, 4)  # 1-3 correct trajectories
    variations = [base_trajectory] + create_trajectory_variations(base_trajectory, num_correct - 1)

    # Create distractors
    num_distractors = 16 - num_correct
    distractors = create_distractor_trajectories(base_trajectory, traj_type, params, num_distractors)

    # Create individual trajectory images (like Color_Counting pattern)
    image_filenames, correct_positions = create_trajectory_images(
        variations,
        distractors,
        output_dir,
        captcha_id
    )

    return {
        "gif": gif_filename,
        "options": image_filenames,  # List of individual trajectory image filenames
        "correct_positions": correct_positions,
        "trajectory_type": traj_type,
        "num_correct": num_correct
    }


def trajectories_are_similar(traj1, traj2, mse_threshold=0.08, overlap_threshold=0.5):
    """
    Check if two trajectories are too similar (would confuse humans)

    Uses multiple criteria:
    1. MSE (mean squared error) between points
    2. Spatial overlap (how much they occupy the same region)
    3. Direction similarity

    Args:
        traj1, traj2: Trajectory point arrays
        mse_threshold: Maximum MSE to be considered different (higher = more strict)
        overlap_threshold: Maximum overlap ratio to be considered different

    Returns:
        True if trajectories are similar (collision), False if sufficiently different
    """
    # Both trajectories should have same length
    if len(traj1) != len(traj2):
        return False

    # Criterion 1: MSE between points (after alignment)
    # Normalize both trajectories to same bounding box for fair comparison
    def normalize_trajectory(traj):
        min_pt = traj.min(axis=0)
        max_pt = traj.max(axis=0)
        range_pt = max_pt - min_pt
        range_pt = np.where(range_pt == 0, 1, range_pt)  # Avoid division by zero
        return (traj - min_pt) / range_pt

    norm_traj1 = normalize_trajectory(traj1)
    norm_traj2 = normalize_trajectory(traj2)

    # Calculate MSE on normalized trajectories
    mse = np.mean((norm_traj1 - norm_traj2) ** 2)

    # Criterion 2: Spatial overlap
    # Create a simple grid and mark cells occupied by each trajectory
    grid_size = 20
    grid1 = np.zeros((grid_size, grid_size))
    grid2 = np.zeros((grid_size, grid_size))

    for traj, grid in [(norm_traj1, grid1), (norm_traj2, grid2)]:
        for point in traj:
            x = int(point[0] * (grid_size - 1))
            y = int(point[1] * (grid_size - 1))
            x = np.clip(x, 0, grid_size - 1)
            y = np.clip(y, 0, grid_size - 1)
            grid[y, x] = 1

    # Calculate overlap ratio
    overlap = np.sum(grid1 * grid2)
    total_occupied = np.sum(np.maximum(grid1, grid2))
    overlap_ratio = overlap / total_occupied if total_occupied > 0 else 0

    # Criterion 3: Check if they're both essentially straight lines
    def is_straight_line(traj):
        # Linear regression to check straightness
        if len(traj) < 2:
            return False
        # Calculate variance from best-fit line
        x = traj[:, 0]
        y = traj[:, 1]
        if np.std(x) > np.std(y):
            # Fit line based on x
            coeffs = np.polyfit(x, y, 1)
            fitted = np.polyval(coeffs, x)
            variance = np.var(y - fitted)
        else:
            # Fit line based on y
            coeffs = np.polyfit(y, x, 1)
            fitted = np.polyval(coeffs, y)
            variance = np.var(x - fitted)

        # Normalized variance (relative to trajectory size)
        traj_range = np.ptp(traj, axis=0).max()
        normalized_var = variance / (traj_range ** 2) if traj_range > 0 else 0

        return normalized_var < 0.01  # Very straight

    # If both are straight lines, they're too similar
    if is_straight_line(traj1) and is_straight_line(traj2):
        return True

    # Check collision based on MSE OR overlap
    if mse < mse_threshold:  # Very similar in normalized shape
        return True

    if overlap_ratio > overlap_threshold:  # Too much spatial overlap
        return True

    return False


def generate_trajectory_pool(output_dir, num_seeds=20, variations_per_seed=3, width=400, height=400, max_attempts=100):
    """
    Generate a SMART trajectory pool with COLLISION DETECTION:
    - num_seeds seed trajectories (one per CAPTCHA/GIF)
    - Each seed has variations_per_seed variations (these are the correct answers)
    - Seeds are guaranteed to be sufficiently different from each other
    - Total pool size = num_seeds × variations_per_seed

    Args:
        output_dir: Directory to save trajectory images
        num_seeds: Number of seed trajectories (= number of CAPTCHAs)
        variations_per_seed: Number of variations per seed (= correct answers per CAPTCHA)
        width, height: Dimensions for trajectory generation
        max_attempts: Maximum attempts to generate a non-colliding seed

    Returns:
        Tuple: (trajectory_pool, seed_to_variations_map)
    """
    output_dir = Path(output_dir)
    trajectory_pool = []
    seed_to_variations_map = {}  # Maps seed_id -> list of pool indices
    seed_trajectories = []  # Keep all seed trajectories for collision checking

    colors = [
        (255, 80, 80),   # Red
        (80, 150, 255),  # Blue
        (100, 200, 100), # Green
        (255, 180, 80),  # Orange
        (200, 100, 200), # Purple
        (80, 200, 200),  # Cyan
        (255, 150, 150), # Pink
        (150, 150, 255), # Light Blue
        (200, 200, 100), # Yellow-green
        (255, 130, 180), # Hot Pink
        (100, 255, 200), # Mint
        (255, 200, 100), # Gold
    ]

    line_styles = ['solid', 'dashed', 'dotted']

    total_trajectories = num_seeds * variations_per_seed
    print(f"Generating trajectory pool: {num_seeds} seeds × {variations_per_seed} variations = {total_trajectories} trajectories...")
    print(f"Collision detection enabled - ensuring seeds are sufficiently different...")

    pool_idx = 0
    for seed_id in range(num_seeds):
        # Generate seed trajectory with collision detection
        seed_trajectory = None
        traj_type = None
        params = None

        for attempt in range(max_attempts):
            # Generate candidate seed trajectory
            traj_type, params = generate_random_trajectory_params(width, height)
            candidate = generate_trajectory_points(traj_type, params, num_points=60)

            # Check if it collides with existing seeds
            collides = False
            for existing_seed in seed_trajectories:
                if trajectories_are_similar(candidate, existing_seed):
                    collides = True
                    break

            if not collides:
                # No collision - use this trajectory
                seed_trajectory = candidate
                seed_trajectories.append(seed_trajectory)
                break

        if seed_trajectory is None:
            print(f"Warning: Could not generate non-colliding seed {seed_id} after {max_attempts} attempts. Using last candidate.")
            seed_trajectory = candidate
            seed_trajectories.append(seed_trajectory)

        # Create variations of this seed
        variations = [seed_trajectory] + create_trajectory_variations(seed_trajectory, variations_per_seed - 1)

        variation_indices = []
        for var_idx, trajectory in enumerate(variations):
            # Pick color and line style
            color = colors[pool_idx % len(colors)]
            line_style = line_styles[pool_idx % len(line_styles)]

            # Generate filename
            filename = f"traj_pool_{pool_idx}.png"
            filepath = output_dir / filename

            # Create trajectory image
            create_individual_trajectory_image(trajectory, str(filepath), color,
                                              cell_size=150, line_style=line_style)

            # Store metadata
            trajectory_pool.append({
                "filename": filename,
                "seed_id": seed_id,
                "variation_id": var_idx,
                "type": traj_type,
                "color": color,
                "line_style": line_style,
                "trajectory": trajectory
            })

            variation_indices.append(pool_idx)
            pool_idx += 1

        # Map seed to its variations
        seed_to_variations_map[seed_id] = {
            "trajectory": seed_trajectory,
            "type": traj_type,
            "params": params,
            "variation_indices": variation_indices
        }

        if (seed_id + 1) % 5 == 0:
            print(f"  Generated {seed_id + 1}/{num_seeds} seeds ({pool_idx} total trajectories)...")

    print(f"✓ Trajectory pool complete: {total_trajectories} images from {num_seeds} seeds")
    return trajectory_pool, seed_to_variations_map


def find_matching_trajectories(target_trajectory, trajectory_pool, num_matches=2):
    """
    Find trajectories from the pool that are similar to the target
    (These will be the correct answers)
    """
    matching_indices = []

    # Create variations of the target trajectory
    variations = [target_trajectory] + create_trajectory_variations(target_trajectory, num_matches - 1)

    # For each variation, find the closest match in the pool or add a new one
    for variation in variations:
        # Calculate similarity with all pool trajectories
        min_distance = float('inf')
        best_match_idx = None

        for idx, pool_item in enumerate(trajectory_pool):
            # Simple distance metric: mean squared distance between points
            pool_traj = pool_item['trajectory']
            # Resample both to same length
            if len(pool_traj) != len(variation):
                continue
            distance = np.mean((pool_traj - variation) ** 2)

            if distance < min_distance:
                min_distance = distance
                best_match_idx = idx

        if best_match_idx is not None and best_match_idx not in matching_indices:
            matching_indices.append(best_match_idx)

    return matching_indices[:num_matches]


def generate_trajectory_captcha_from_pool(output_dir, seed_id, seed_data, trajectory_pool, width=400, height=400):
    """
    Generate a trajectory CAPTCHA using SMART pool approach

    Args:
        seed_id: Which seed trajectory to use (0 to num_seeds-1)
        seed_data: Data for this seed from seed_to_variations_map
        trajectory_pool: The full trajectory pool
    """
    output_dir = Path(output_dir)

    # Get the seed trajectory and its metadata
    base_trajectory = seed_data["trajectory"]
    traj_type = seed_data["type"]
    correct_indices = seed_data["variation_indices"]  # These are the correct answers!

    # Generate GIF of ball movement with varied timing (slower for humans)
    gif_filename = f"trajectory_{seed_id}_movement.gif"
    gif_path = output_dir / gif_filename
    generate_ball_movement_gif(
        base_trajectory,
        str(gif_path),
        width=width,
        height=height,
        fps=15,  # Reduced from 20 to make movement smoother and slower
        ball_radius=14,  # Slightly larger ball for better visibility
        ball_color=(255, 80, 80),  # Brighter red
        total_duration=5.0  # Increased from 3.5 to 5 seconds for easier tracking
    )

    num_correct = len(correct_indices)

    # Sample distractors from the pool (excluding this seed's variations)
    all_pool_indices = list(range(len(trajectory_pool)))
    available_distractors = [idx for idx in all_pool_indices if idx not in correct_indices]

    num_distractors = 16 - num_correct
    distractor_indices = np.random.choice(available_distractors, size=num_distractors, replace=True)

    # Combine correct and distractor indices
    all_indices = correct_indices + distractor_indices.tolist()

    # Shuffle while tracking correct positions
    shuffle_map = list(range(len(all_indices)))
    np.random.shuffle(shuffle_map)

    # Build shuffled options and track correct positions
    options = []
    correct_positions = []
    for new_pos, old_idx in enumerate(shuffle_map):
        pool_idx = all_indices[old_idx]
        options.append(trajectory_pool[pool_idx]['filename'])

        # If this was originally a correct trajectory, mark its new position
        if old_idx < num_correct:
            correct_positions.append(new_pos)

    correct_positions = sorted(correct_positions)

    return {
        "gif": gif_filename,
        "options": options,
        "correct_positions": correct_positions,
        "trajectory_type": traj_type,
        "num_correct": num_correct
    }


def generate_dataset(output_dir, num_samples=20, variations_per_seed=3):
    """
    Generate a dataset of trajectory recovery CAPTCHAs using SMART pool approach

    Args:
        num_samples: Number of CAPTCHAs to generate (= number of seeds)
        variations_per_seed: Number of variations per seed (= correct answers per CAPTCHA)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the SMART trajectory pool
    # Pool size = num_samples × variations_per_seed
    trajectory_pool, seed_to_variations_map = generate_trajectory_pool(
        output_dir,
        num_seeds=num_samples,
        variations_per_seed=variations_per_seed
    )

    ground_truth = {}

    print(f"\nGenerating {num_samples} CAPTCHAs using the trajectory pool...")

    for seed_id in range(num_samples):
        print(f"Generating trajectory CAPTCHA {seed_id+1}/{num_samples}...")

        seed_data = seed_to_variations_map[seed_id]
        captcha_data = generate_trajectory_captcha_from_pool(
            output_dir, seed_id, seed_data, trajectory_pool
        )

        # Store ground truth in Color_Counting format
        ground_truth[f"trajectory_recovery_{seed_id:04d}"] = {
            "prompt": "Watch the ball movement in the animation above, then select ALL trajectories in the grid below that match the ball's path.",
            "description": f"Trajectory recovery: {captcha_data['trajectory_type']} path with {captcha_data['num_correct']} correct option(s)",
            "movement_gif": captcha_data['gif'],
            "options": captcha_data['options'],
            "answer": captcha_data['correct_positions'],
            "grid_size": [4, 4],
            "difficulty": 5,
            "media_type": "gif+grid",
            "trajectory_type": captcha_data['trajectory_type'],
            "num_correct": captcha_data['num_correct']
        }

    # Save ground truth
    ground_truth_path = output_dir / "ground_truth.json"
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    pool_size = num_samples * variations_per_seed
    print(f"\n{'='*70}")
    print(f"Generated {num_samples} trajectory recovery CAPTCHAs!")
    print(f"Trajectory pool: {num_samples} seeds × {variations_per_seed} variations = {pool_size} images")
    print(f"Output directory: {output_dir}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    output_dir = str(Path(__file__).parent.parent / "captcha_data" / "Trajectory_Recovery")
    generate_dataset(output_dir, num_samples=20, variations_per_seed=3)

    print("\n🎯 Trajectory Recovery CAPTCHA Dataset Generated!")
    print("\n📊 CAPTCHA Structure:")
    print("  • Movement GIF: Shows ball moving along a path")
    print("  • Options Grid: 4x4 grid with trajectory paths in different colors")
    print("  • Task: Select ALL correct trajectories (1-3 correct answers)")
    print("\n🧠 Why Humans Can Solve It:")
    print("  ✓ Track moving ball naturally")
    print("  ✓ Remember spatial path")
    print("  ✓ Match pattern to static trajectories")
    print("  ✓ Recognize variations of same path")
    print("\n🤖 Why LLMs/Bots Struggle:")
    print("  ✗ Requires temporal tracking + spatial memory")
    print("  ✗ Need to extract path from motion")
    print("  ✗ Match dynamic to static representations")
    print("  ✗ Distinguish correct variations from distractors")
