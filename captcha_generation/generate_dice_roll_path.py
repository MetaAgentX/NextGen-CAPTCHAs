"""
Dice Roll Path CAPTCHA Generator

Creates images of:
- A 3D grid plane
- A 3D dice sitting on one grid cell (with visible pips)
- A polyline path connecting grid cell centers (the rolling path)

Question:
  "If the dice is rolled on the shown path, what will be the number on the top?"

This script also simulates the dice rolls along the path to compute the correct top number.

Dependencies: matplotlib, numpy (already used in this repo)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

# Ensure Matplotlib cache/config is writable even in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

# Use a non-interactive backend for headless generation
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon


@dataclass(frozen=True)
class DiceOrientation:
    # Face values for a standard die
    top: int
    bottom: int
    north: int  # +y
    south: int  # -y
    east: int  # +x
    west: int  # -x

    def roll(self, direction: str) -> "DiceOrientation":
        """
        Roll the die by one cell.

        Coordinate convention:
        - East  = +x
        - West  = -x
        - North = +y
        - South = -y
        """
        d = direction.upper()
        if d == "N":
            # push toward +y: old_north becomes bottom, old_south becomes top
            return DiceOrientation(
                top=self.south,
                bottom=self.north,
                north=self.top,
                south=self.bottom,
                east=self.east,
                west=self.west,
            )
        if d == "S":
            return DiceOrientation(
                top=self.north,
                bottom=self.south,
                north=self.bottom,
                south=self.top,
                east=self.east,
                west=self.west,
            )
        if d == "E":
            # push toward +x: old_east becomes bottom, old_west becomes top
            return DiceOrientation(
                top=self.west,
                bottom=self.east,
                north=self.north,
                south=self.south,
                east=self.top,
                west=self.bottom,
            )
        if d == "W":
            return DiceOrientation(
                top=self.east,
                bottom=self.west,
                north=self.north,
                south=self.south,
                east=self.bottom,
                west=self.top,
            )
        raise ValueError(f"Invalid roll direction: {direction!r} (expected one of N/S/E/W)")


PIP_LAYOUT: Dict[int, List[Tuple[float, float]]] = {
    1: [(0.0, 0.0)],
    2: [(-1.0, -1.0), (1.0, 1.0)],
    3: [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)],
    4: [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)],
    5: [(-1.0, -1.0), (-1.0, 1.0), (0.0, 0.0), (1.0, -1.0), (1.0, 1.0)],
    6: [(-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 0.0), (1.0, 1.0)],
}


def _self_avoiding_path(
    rng: random.Random, grid_size: int, path_len: int
) -> Tuple[Tuple[int, int], List[str], List[Tuple[int, int]]]:
    """
    Produce a path within an NxN grid as (start_cell, steps, visited_cells).
    Uses a simple backtracking random walk to stay in-bounds and avoid revisits.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if path_len < 1:
        raise ValueError("path_len must be >= 1")

    dirs = [("N", (0, 1)), ("S", (0, -1)), ("E", (1, 0)), ("W", (-1, 0))]

    start = (rng.randrange(grid_size), rng.randrange(grid_size))
    visited = [start]
    used = {start}
    steps: List[str] = []

    # Backtracking search
    def backtrack() -> bool:
        if len(steps) == path_len:
            return True

        x, y = visited[-1]
        candidates = dirs[:]
        rng.shuffle(candidates)
        for d, (dx, dy) in candidates:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                continue
            nxt = (nx, ny)
            if nxt in used:
                continue
            steps.append(d)
            visited.append(nxt)
            used.add(nxt)
            if backtrack():
                return True
            used.remove(nxt)
            visited.pop()
            steps.pop()

        return False

    # Try multiple starts if needed (rare for small grids / long paths)
    for _ in range(200):
        visited = [start]
        used = {start}
        steps = []
        if backtrack():
            return start, steps, visited
        start = (rng.randrange(grid_size), rng.randrange(grid_size))

    raise RuntimeError("Failed to generate a self-avoiding path; try a larger grid or smaller path_len.")


def simulate_rolls(initial: DiceOrientation, steps: Sequence[str]) -> DiceOrientation:
    o = initial
    for s in steps:
        o = o.roll(s)
    return o


def all_orientations(base: DiceOrientation) -> List[DiceOrientation]:
    """
    Enumerate all 24 valid orientations reachable by rolling a physical die.
    """
    seen = set()
    q: List[DiceOrientation] = [base]
    out: List[DiceOrientation] = []
    while q:
        o = q.pop()
        key = (o.top, o.bottom, o.north, o.south, o.east, o.west)
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
        for d in ("N", "S", "E", "W"):
            q.append(o.roll(d))
    return out


def choose_initial_orientation(
    rng: random.Random, base: DiceOrientation, initial_top: int | None
) -> DiceOrientation:
    """
    Pick an initial orientation. If initial_top is provided, pick uniformly among the 4
    orientations that have that top value.
    """
    if initial_top is None:
        return base
    if initial_top not in (1, 2, 3, 4, 5, 6):
        raise ValueError("--initial-top must be in 1..6")
    opts = [o for o in all_orientations(base) if o.top == initial_top]
    if not opts:
        raise RuntimeError("No orientations found for the requested top value (unexpected).")
    return rng.choice(opts)


def _cube_faces(center: Tuple[float, float, float], side: float) -> Dict[str, np.ndarray]:
    """
    Returns cube face vertices for Poly3DCollection.
    Face keys: top, bottom, north, south, east, west.
    Each face is (4,3).
    """
    cx, cy, cz = center
    h = side / 2.0

    # 8 vertices
    # x: -h/+h, y: -h/+h, z: -h/+h
    v000 = np.array([cx - h, cy - h, cz - h])
    v001 = np.array([cx - h, cy - h, cz + h])
    v010 = np.array([cx - h, cy + h, cz - h])
    v011 = np.array([cx - h, cy + h, cz + h])
    v100 = np.array([cx + h, cy - h, cz - h])
    v101 = np.array([cx + h, cy - h, cz + h])
    v110 = np.array([cx + h, cy + h, cz - h])
    v111 = np.array([cx + h, cy + h, cz + h])

    faces = {
        "top": np.stack([v001, v101, v111, v011]),
        "bottom": np.stack([v000, v010, v110, v100]),
        "north": np.stack([v011, v111, v110, v010]),  # +y
        "south": np.stack([v001, v000, v100, v101]),  # -y
        "east": np.stack([v101, v100, v110, v111]),  # +x
        "west": np.stack([v001, v011, v010, v000]),  # -x
    }
    return faces


def _face_point(
    face: str, center: Tuple[float, float, float], side: float, u: float, v: float
) -> Tuple[float, float, float]:
    """
    Map (u,v) in [-1,1]x[-1,1] onto a cube face (slightly above the face).
    """
    cx, cy, cz = center
    h = side / 2.0
    # Keep pips inset from the edge
    scale = h * 0.55
    eps = side * 0.008  # raise dots slightly above the face

    if face == "top":
        return (cx + u * scale, cy + v * scale, cz + h + eps)
    if face == "bottom":
        return (cx + u * scale, cy - v * scale, cz - h - eps)
    if face == "north":
        return (cx + u * scale, cy + h + eps, cz + v * scale)
    if face == "south":
        return (cx - u * scale, cy - h - eps, cz + v * scale)
    if face == "east":
        return (cx + h + eps, cy - u * scale, cz + v * scale)
    if face == "west":
        return (cx - h - eps, cy + u * scale, cz + v * scale)
    raise ValueError(f"Unknown face: {face}")


def draw_scene(
    output_path: str,
    grid_size: int,
    path_cells: Sequence[Tuple[int, int]],
    dice_center_cell: Tuple[int, int],
    dice_orientation: DiceOrientation,
    image_size_px: int = 640,
) -> None:
    """
    Render a single PNG with a 3D grid plane, 3D dice, and 3D path line.
    """
    dpi = 120
    fig, ax = plt.subplots(figsize=(image_size_px / dpi, image_size_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_aspect("equal")
    ax.axis("off")

    # --- 3D -> 2D isometric-ish projection (still 3D objects, drawn like your reference) ---
    # Standard isometric rotation: yaw -45deg around Z, then pitch ~35.264deg around X.
    yaw = math.radians(-45.0)
    pitch = math.radians(35.264)

    Rz = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=float,
    )
    R = Rx @ Rz

    def proj(p: np.ndarray) -> Tuple[float, float, float]:
        """Return (u,v,depth) for depth-sorting."""
        q = R @ p
        # Orthographic projection: use x,y; depth = z (after rotation)
        return float(q[0]), float(q[1]), float(q[2])

    def proj_poly(poly3: np.ndarray) -> Tuple[np.ndarray, float]:
        pts2 = []
        depths = []
        for i in range(poly3.shape[0]):
            u, v, d = proj(poly3[i])
            pts2.append((u, v))
            depths.append(d)
        pts2a = np.array(pts2, dtype=float)
        return pts2a, float(np.mean(depths))

    # Helper: convert grid cell to 3D point on plane z=0
    def cell_center(c: Tuple[int, int]) -> Tuple[float, float]:
        return (c[0] + 0.5, c[1] + 0.5)

    # --- Draw perspective grid (blue lines like reference) ---
    grid_color = "#1d4ed8"
    grid_lw = 1.1
    for i in range(grid_size + 1):
        a = np.array([i, 0.0, 0.0], dtype=float)
        b = np.array([i, float(grid_size), 0.0], dtype=float)
        ua, va, _ = proj(a)
        ub, vb, _ = proj(b)
        ax.plot([ua, ub], [va, vb], color=grid_color, linewidth=grid_lw, alpha=0.8)

        c = np.array([0.0, i, 0.0], dtype=float)
        d = np.array([float(grid_size), i, 0.0], dtype=float)
        uc, vc, _ = proj(c)
        ud, vd, _ = proj(d)
        ax.plot([uc, ud], [vc, vd], color=grid_color, linewidth=grid_lw, alpha=0.8)

    # --- Path (bold black polyline + arrow head) ---
    path_pts_3d = [np.array([*cell_center(c), 0.0], dtype=float) for c in path_cells]
    path_pts_2d = [proj(p) for p in path_pts_3d]
    xs = [u for (u, v, d) in path_pts_2d]
    ys = [v for (u, v, d) in path_pts_2d]
    ax.plot(xs, ys, color="#111827", linewidth=3.6, solid_capstyle="round", zorder=10)

    # Arrow head at end of path
    if len(xs) >= 2:
        x1, y1 = xs[-2], ys[-2]
        x2, y2 = xs[-1], ys[-1]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        head_len = 0.35
        head_w = 0.22
        tip = (x2, y2)
        base = (x2 - ux * head_len, y2 - uy * head_len)
        nx, ny = -uy, ux
        p1 = (base[0] + nx * head_w, base[1] + ny * head_w)
        p2 = (base[0] - nx * head_w, base[1] - ny * head_w)
        ax.add_patch(Polygon([tip, p1, p2], closed=True, facecolor="#111827", edgecolor="#111827", zorder=11))

    # --- Dice (3D cube, projected to 2D like a sketch) ---
    dcx, dcy = cell_center(dice_center_cell)
    side = 0.95
    center = np.array([dcx, dcy, side / 2.0], dtype=float)
    faces = _cube_faces(center=(float(center[0]), float(center[1]), float(center[2])), side=side)

    # Determine face normals in camera space to decide visibility
    face_normals = {
        "top": np.array([0.0, 0.0, 1.0], dtype=float),
        "bottom": np.array([0.0, 0.0, -1.0], dtype=float),
        "north": np.array([0.0, 1.0, 0.0], dtype=float),
        "south": np.array([0.0, -1.0, 0.0], dtype=float),
        "east": np.array([1.0, 0.0, 0.0], dtype=float),
        "west": np.array([-1.0, 0.0, 0.0], dtype=float),
    }
    # In projected space, camera looks along +depth axis; visible if normal has positive depth component after rotation.
    visible = {}
    for fname, n in face_normals.items():
        n_cam = R @ n
        visible[fname] = float(n_cam[2]) > 0.02

    face_value = {
        "top": dice_orientation.top,
        "bottom": dice_orientation.bottom,
        "north": dice_orientation.north,
        "south": dice_orientation.south,
        "east": dice_orientation.east,
        "west": dice_orientation.west,
    }

    # Build drawable face polys with depth for sorting (back to front)
    draw_faces = []
    for fname, poly3 in faces.items():
        pts2, depth = proj_poly(poly3)
        draw_faces.append((depth, fname, pts2))
    draw_faces.sort(key=lambda t: t[0])  # back first

    for _, fname, pts2 in draw_faces:
        if not visible.get(fname, False):
            continue
        fill = "#ffffff" if fname != "top" else "#f8fafc"
        ax.add_patch(
            Polygon(
                pts2,
                closed=True,
                facecolor=fill,
                edgecolor="#111827",
                linewidth=2.0,
                zorder=20,
            )
        )

        # Pips for visible faces
        layout = PIP_LAYOUT.get(face_value[fname], [])
        for (u, v) in layout:
            px, py, pz = _face_point(fname, (float(center[0]), float(center[1]), float(center[2])), side, u / 2.0, v / 2.0)
            uu, vv, dd = proj(np.array([px, py, pz], dtype=float))
            ax.add_patch(Circle((uu, vv), radius=0.06, facecolor="#111827", edgecolor="#111827", zorder=25))



    # Frame tightly around content
    all_u = []
    all_v = []
    # grid corners
    corners = [
        np.array([0.0, 0.0, 0.0]),
        np.array([float(grid_size), 0.0, 0.0]),
        np.array([0.0, float(grid_size), 0.0]),
        np.array([float(grid_size), float(grid_size), 0.0]),
        np.array([dcx, dcy, side]),  # dice top
    ]
    for p in corners:
        u, v, _ = proj(p)
        all_u.append(u)
        all_v.append(v)
    all_u.extend(xs)
    all_v.extend(ys)
    umin, umax = min(all_u), max(all_u)
    vmin, vmax = min(all_v), max(all_v)
    pad = 0.9
    ax.set_xlim(umin - pad, umax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout(pad=0)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def generate_dice_roll_path_dataset(
    output_dir: str,
    num_puzzles: int = 20,
    grid_size: int = 6,
    path_len: int = 7,
    seed: int | None = 0,
    image_size_px: int = 640,
    initial_top: int | None = None,
) -> None:
    """
    Generate a dataset under output_dir with PNG images and a ground_truth.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    # Base orientation (standard and consistent). Opposite faces sum to 7.
    # Opposite faces sum to 7; adjacency chosen to be consistent.
    base = DiceOrientation(top=1, bottom=6, north=2, south=5, east=3, west=4)

    ground_truth: Dict[str, Dict] = {}

    for idx in range(num_puzzles):
        initial = choose_initial_orientation(rng=rng, base=base, initial_top=initial_top)
        start_cell, steps, visited = _self_avoiding_path(rng, grid_size=grid_size, path_len=path_len)
        final = simulate_rolls(initial, steps)

        puzzle_id = f"dice_roll_path_{idx:04d}"
        img_name = f"{puzzle_id}.png"
        img_path = os.path.join(output_dir, img_name)

        draw_scene(
            output_path=img_path,
            grid_size=grid_size,
            path_cells=visited,
            dice_center_cell=start_cell,
            dice_orientation=initial,
            image_size_px=image_size_px,
        )

        ground_truth[puzzle_id] = {
            "prompt": "If the dice is rolled on the shown path, what will be the number on the top?",
            "description": "3D grid plane with a dice and a path showing the roll sequence",
            "grid_size": grid_size,
            "image": img_name,
            "start_cell": {"x": start_cell[0], "y": start_cell[1]},
            "path_steps": steps,  # list of N/S/E/W
            "path_cells": [{"x": x, "y": y} for (x, y) in visited],
            "initial_orientation": asdict(initial),
            "answer_top": final.top,
            "difficulty": min(10, 2 + path_len),
            "media_type": "png",
        }

    gt_path = os.path.join(output_dir, "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"✓ Generated {num_puzzles} dice roll path puzzles")
    print(f"✓ Images + ground truth saved to: {output_dir}")
    print(f"✓ Ground truth: {gt_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Dice Roll Path CAPTCHA images (3D grid + 3D dice + path).")
    p.add_argument("--output-dir", type=str, default="../captcha_data/Dice_Roll_Path")
    p.add_argument("--num", type=int, default=20, help="Number of puzzles to generate")
    p.add_argument("--grid-size", type=int, default=6, help="Grid size N for an NxN plane")
    p.add_argument("--path-len", type=int, default=7, help="Number of rolls (path steps)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (set to -1 for non-deterministic)")
    p.add_argument("--image-size", type=int, default=640, help="Output PNG width/height in pixels")
    p.add_argument(
        "--initial-top",
        type=int,
        default=None,
        help="Force the starting die to have this number on top (1..6). If omitted, uses a fixed default orientation.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seed = None if args.seed == -1 else args.seed
    generate_dice_roll_path_dataset(
        output_dir=args.output_dir,
        num_puzzles=args.num,
        grid_size=args.grid_size,
        path_len=args.path_len,
        seed=seed,
        image_size_px=args.image_size,
        initial_top=args.initial_top,
    )


if __name__ == "__main__":
    main()


