"""Load external object masks as synthetic white-rim sticker assets.

Keep downloaded images outside this repository. The mask describes the object;
the generated outer sticker mask includes a white print border around it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sticker_data import Asset


def load_public_assets(image_dir: str, mask_dir: str, *, limit: int = 240,
                       seed: int = 2026, side: int = 384,
                       holdout_fraction: float = 0.1, source: str = "DIS5K",
                       alpha_matte: bool = False) -> tuple[list[Asset], list[Asset], list[str]]:
    """Match files by stem, choose a deterministic subset, and split by source ID."""
    if limit < 2 or side < 64 or not 0 < holdout_fraction < 0.5:
        raise ValueError("Need at least two assets, side >= 64, and a valid holdout fraction")
    image_files = {path.stem: path for path in Path(image_dir).iterdir()
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    mask_files = {path.stem: path for path in Path(mask_dir).iterdir()
                  if path.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    names = sorted(image_files.keys() & mask_files.keys())
    if len(names) < 2:
        raise ValueError(f"No paired images/masks in {image_dir} and {mask_dir}")
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    names = names[:limit]
    holdout_count = max(1, round(len(names) * holdout_fraction))
    assets = []
    selected = []
    for name in names:
        try:
            with Image.open(image_files[name]) as source_image:
                image = source_image.convert("RGB")
            with Image.open(mask_files[name]) as source_mask:
                mask = source_mask.convert("L")
        except (OSError, ValueError):
            continue
        if image.size != mask.size:
            mask = mask.resize(image.size,
                               Image.Resampling.LANCZOS if alpha_matte else Image.Resampling.NEAREST)
        scale = min(1, side / max(image.size))
        shape = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(shape, Image.Resampling.LANCZOS)
        mask = np.asarray(mask.resize(
            shape, Image.Resampling.LANCZOS if alpha_matte else Image.Resampling.NEAREST))
        silhouette = mask >= (16 if alpha_matte else 128)
        if silhouette.sum() < 100 or silhouette.mean() > 0.95:
            continue
        x, y, width, height = cv2.boundingRect(np.uint8(silhouette))
        if min(width, height) < 8:
            continue
        image = np.asarray(image)[y:y + height, x:x + width]
        mask = mask[y:y + height, x:x + width]
        silhouette = np.uint8(silhouette[y:y + height, x:x + width])
        rim = max(3, round(max(width, height) * 0.035))
        padding = rim + 3
        canvas = np.full((height + 2 * padding, width + 2 * padding, 3), 255, np.uint8)
        object_mask = np.pad(silhouette, padding)
        region = canvas[padding:padding + height, padding:padding + width]
        if alpha_matte:
            alpha = mask.astype(np.float32)[..., None] / 255
            region[:] = np.uint8(np.round(image * alpha + 255 * (1 - alpha)))
        else:
            region[silhouette > 0] = image[silhouette > 0]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim * 2 + 1, rim * 2 + 1))
        cut_mask = cv2.dilate(object_mask, kernel) * 255
        assets.append(Asset(len(assets) + 1, Image.fromarray(canvas),
                            Image.fromarray(cut_mask), source))
        selected.append(name)
    if len(assets) < 2:
        raise ValueError("Fewer than two usable image/mask pairs")
    holdout_count = min(holdout_count, len(assets) - 1)
    return assets[holdout_count:], assets[:holdout_count], selected
