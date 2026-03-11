"""
image_loader.py — Load SEM images and extract metadata.
Supports TIFF (with SEM metadata), PNG, BMP, JPG.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg")


@dataclass
class ImageData:
    path: str
    pixels: np.ndarray          # grayscale uint8 or uint16
    width: int
    height: int
    filename: str
    # Pixel size from SEM metadata if available (µm per pixel)
    metadata_um_per_px: Optional[float] = None
    raw_metadata: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.pixels.shape


def load_image(path: str) -> ImageData:
    """Load an image from disk. Returns grayscale ImageData."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    # Try tifffile first for TIFF to get metadata
    metadata_um_per_px = None
    raw_meta = {}

    if ext in (".tif", ".tiff"):
        metadata_um_per_px, raw_meta = _read_tiff_metadata(path)

    # Load pixels via OpenCV (handles most formats)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image: {path}")

    # Convert to grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize 16-bit to 8-bit for display / processing
    if img.dtype == np.uint16:
        img_8 = (img / 256).astype(np.uint8)
    else:
        img_8 = img.astype(np.uint8)

    h, w = img_8.shape
    return ImageData(
        path=path,
        pixels=img_8,
        width=w,
        height=h,
        filename=os.path.basename(path),
        metadata_um_per_px=metadata_um_per_px,
        raw_metadata=raw_meta,
    )


def _read_tiff_metadata(path: str):
    """Try to extract pixel size from TIFF tags (ZEISS, FEI/Thermo, Hitachi, standard)."""
    um_per_px = None
    meta = {}
    try:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            # Standard TIFF XResolution / ResolutionUnit
            page = tif.pages[0]
            tags = {tag.name: tag.value for tag in page.tags.values()}
            meta.update(tags)

            x_res = tags.get("XResolution")
            res_unit = tags.get("ResolutionUnit", 1)  # 1=no abs, 2=inch, 3=cm

            if x_res is not None:
                if isinstance(x_res, tuple):
                    num, den = x_res
                    px_per_unit = num / den if den != 0 else None
                else:
                    px_per_unit = float(x_res)

                if px_per_unit and px_per_unit > 0:
                    if res_unit == 2:   # pixels per inch → µm/px
                        um_per_px = 25400.0 / px_per_unit
                    elif res_unit == 3:  # pixels per cm → µm/px
                        um_per_px = 10000.0 / px_per_unit

            # FEI/Thermo Fisher: look in ImageDescription or FEI metadata
            img_desc = tags.get("ImageDescription", "")
            if isinstance(img_desc, str) and "PixelWidth" in img_desc:
                for line in img_desc.splitlines():
                    if "PixelWidth" in line:
                        try:
                            val = float(line.split("=")[-1].strip())
                            # FEI stores in meters
                            um_per_px = val * 1e6
                        except (ValueError, IndexError):
                            pass

            # ZEISS uses tif.sem_metadata or similar
            if hasattr(tif, "sem_metadata") and tif.sem_metadata:
                sem = tif.sem_metadata
                if "ap_pixel_size" in sem:
                    try:
                        um_per_px = float(sem["ap_pixel_size"]) * 1e6
                    except (ValueError, TypeError):
                        pass

    except Exception:
        pass  # Metadata extraction is best-effort

    return um_per_px, meta


def list_images_in_directory(directory: str) -> list[str]:
    """Return sorted list of supported image paths in directory."""
    paths = []
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
            paths.append(os.path.join(directory, fname))
    return paths
