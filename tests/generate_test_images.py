"""
generate_test_images.py — Generate synthetic SEM stripe images for testing.

Creates a directory of PNG test images with:
  - Alternating white/black stripes (photo resist pattern)
  - Configurable line widths, pitch, noise, and tilt angle
  - Slight width variation per stripe (to test LER/LWR detection)

Usage:
    python tests/generate_test_images.py
    python tests/generate_test_images.py --output_dir ./test_images --count 10
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def make_stripe_image(
    width: int = 1024,
    height: int = 768,
    white_width_px: float = 40.0,
    black_width_px: float = 30.0,
    angle_deg: float = 0.0,
    noise_sigma: float = 8.0,
    edge_sigma: float = 2.0,
    ler_sigma_px: float = 1.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic SEM-like stripe image.

    Parameters
    ----------
    white_width_px : mean width of white stripes (photo resist)
    black_width_px : mean width of black spacing
    angle_deg      : tilt of stripes from vertical (degrees)
    noise_sigma    : Gaussian noise standard deviation
    edge_sigma     : Gaussian blur on edges (simulates SEM PSF)
    ler_sigma_px   : std dev of per-row edge position variation (LER)
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width), dtype=np.float32)
    pitch = white_width_px + black_width_px

    # White = 220 DN, Black = 30 DN (realistic SEM contrast)
    WHITE = 220.0
    BLACK = 30.0

    # Generate stripes column by column (before rotation)
    # Compute horizontal position for each column → stripe classification
    # Apply LER: for each row, perturb the edge positions by ler_sigma_px
    for row in range(height):
        # Row-by-row edge perturbation for LER
        left_perturb = rng.normal(0, ler_sigma_px)
        for col in range(width):
            # Effective x position (accounting for angle)
            if abs(angle_deg) > 0.01:
                angle_rad = np.radians(angle_deg)
                x_eff = col * np.cos(angle_rad) + row * np.sin(angle_rad)
            else:
                x_eff = col + left_perturb

            pos_in_pitch = x_eff % pitch
            if pos_in_pitch < white_width_px:
                img[row, col] = WHITE
            else:
                img[row, col] = BLACK

    # Smooth edges (simulate finite SEM beam / PSF)
    if edge_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), edge_sigma)

    # Add Gaussian noise
    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, img.shape).astype(np.float32)
        img = img + noise

    # Clip and convert to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def generate_test_set(
    output_dir: str = "test_images",
    count: int = 8,
    base_white_px: float = 40.0,
    base_black_px: float = 30.0,
) -> list[str]:
    """
    Generate a set of synthetic test images with slight variation.
    Returns list of generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # Variation across images (as in a real wafer measurement set)
    white_variations = np.linspace(base_white_px * 0.9, base_white_px * 1.1, count)
    angle_variations = np.linspace(-3.0, 3.0, count)  # slight tilt variation

    for i in range(count):
        white_w = float(white_variations[i])
        angle = float(angle_variations[i])
        seed = 100 + i

        img = make_stripe_image(
            width=1024,
            height=768,
            white_width_px=white_w,
            black_width_px=base_black_px,
            angle_deg=angle,
            noise_sigma=8.0,
            edge_sigma=2.0,
            ler_sigma_px=1.5,
            seed=seed,
        )

        fname = f"sem_stripe_{i+1:03d}_w{white_w:.1f}px_a{angle:.1f}deg.png"
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, img)
        paths.append(fpath)
        print(f"  Generated: {fname}  (white={white_w:.1f}px, angle={angle:.1f}°)")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SEM stripe test images")
    parser.add_argument("--output_dir", default="test_images", help="Output directory")
    parser.add_argument("--count", type=int, default=8, help="Number of images to generate")
    parser.add_argument("--white_px", type=float, default=40.0, help="White stripe width (px)")
    parser.add_argument("--black_px", type=float, default=30.0, help="Black stripe width (px)")
    args = parser.parse_args()

    print(f"Generating {args.count} synthetic SEM stripe images → {args.output_dir}/")
    paths = generate_test_set(
        output_dir=args.output_dir,
        count=args.count,
        base_white_px=args.white_px,
        base_black_px=args.black_px,
    )
    print(f"\n✓ Generated {len(paths)} test images.")
    print(f"  Open SEM Stripe Analyzer → File → Open Directory → select: {args.output_dir}")
    print(f"  Set Scale: 0.1 µm/px (if 1 pixel = 0.1 µm)")
    print(f"  Expected white CD: {args.white_px * 0.1:.2f} µm")
    print(f"  Expected black CD: {args.black_px * 0.1:.2f} µm")
    print(f"  Expected pitch:    {(args.white_px + args.black_px) * 0.1:.2f} µm")


if __name__ == "__main__":
    main()
