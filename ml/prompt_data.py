"""Build one-instance training crops from generated sheets or reviewed stickers."""

from __future__ import annotations

import cv2
import numpy as np


def box_of(mask: np.ndarray) -> tuple[float, float, float, float]:
    x, y, width, height = cv2.boundingRect(np.uint8(mask))
    return float(x), float(y), float(x + width), float(y + height)


def prompt_crop(image: np.ndarray, mask: np.ndarray,
                box: tuple[float, float, float, float], size: int,
                seed: int = 0, augment: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return 5-channel input, target mask and the crop-to-image transform."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = map(float, box)
    width, height = x1 - x0, y1 - y0
    if width <= 1 or height <= 1:
        raise ValueError("Instance box has no area")
    if augment:
        x0 += float(rng.uniform(-0.15, 0.15) * width)
        x1 += float(rng.uniform(-0.15, 0.15) * width)
        y0 += float(rng.uniform(-0.15, 0.15) * height)
        y1 += float(rng.uniform(-0.15, 0.15) * height)
        x1 = max(x1, x0 + 2)
        y1 = max(y1, y0 + 2)
    width, height = x1 - x0, y1 - y0
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    factor = float(rng.uniform(1.35, 1.85)) if augment else 1.5
    side = max(width, height) * factor
    if augment:
        center_x += float(rng.uniform(-0.04, 0.04) * side)
        center_y += float(rng.uniform(-0.04, 0.04) * side)
    left, top = center_x - side / 2, center_y - side / 2
    scale = size / side
    affine = np.float32([[scale, 0, -left * scale], [0, scale, -top * scale]])
    rgb = cv2.warpAffine(image, affine, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    truth = cv2.warpAffine(np.uint8(mask), affine, (size, size),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    bx0, bx1 = int(np.floor((x0 - left) * scale)), int(np.ceil((x1 - left) * scale))
    by0, by1 = int(np.floor((y0 - top) * scale)), int(np.ceil((y1 - top) * scale))
    box_map = np.zeros((size, size), np.float32)
    box_map[max(0, by0):min(size, by1), max(0, bx0):min(size, bx1)] = 1
    yy, xx = np.mgrid[:size, :size]
    center_map = np.exp(-((xx - (x0 + x1) * scale / 2 + left * scale) ** 2 +
                          (yy - (y0 + y1) * scale / 2 + top * scale) ** 2) /
                        (2 * max(3.0, max(width, height) * scale * 0.15) ** 2)).astype(np.float32)
    features = np.concatenate((rgb.transpose(2, 0, 1).astype(np.float32) / 255,
                               box_map[None], center_map[None]))
    if augment and rng.random() < 0.5:
        features = features[:, :, ::-1].copy()
        truth = truth[:, ::-1].copy()
    return features, np.float32(truth > 0)[None], {
        "left": left, "top": top, "side": side, "size": size,
    }
