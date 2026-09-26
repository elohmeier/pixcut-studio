"""Synthetic sticker scenes from reviewed, isolated cut groups."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance


@dataclass
class Asset:
    number: int
    image: Image.Image
    mask: Image.Image
    sheet: str = "original-sample"


@dataclass
class SheetData:
    name: str
    image: np.ndarray
    labels: np.ndarray
    boundary_xy: np.ndarray


def labels_from_analysis(analysis_path: str | Path, shape: tuple[int, int],
                         selected: set[int] | None = None, margin: int = 0) -> np.ndarray:
    """Rasterize CLI groups at source resolution, optionally expanding masks."""
    directory = Path(analysis_path).resolve().parent
    analysis = json.loads(Path(analysis_path).read_text())
    spec = json.loads((directory / "summary.json").read_text())["raster"]
    factor, pad = spec["factor"], spec["pad"]
    if selected is None:
        selected = set(range(1, len(analysis["groups"]) + 1))
    labels = np.zeros(shape, np.int16)
    for number, group in enumerate(analysis["groups"], 1):
        if number not in selected:
            continue
        mask = Image.new("L", (shape[1], shape[0]))
        for path in group["paths"]:
            polygon = [((x - pad) / factor, (y - pad) / factor) for x, y in path]
            if len(polygon) >= 3:
                ImageDraw.Draw(mask).polygon(polygon, fill=255)
        mask = np.asarray(mask)
        if margin:
            mask = cv2.dilate(mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
        labels[mask > 127] = number
    return labels


def labels_for_sheet(sheet: dict, root: Path, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize the manifest's selected CLI groups at source resolution."""
    name = sheet["name"]
    analysis_path = root / "analyses" / name / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    selected = set(sheet.get("groups", range(1, len(analysis["groups"]) + 1)))
    selected -= set(sheet.get("exclude_groups", []))
    return labels_from_analysis(analysis_path, shape, selected,
                                int(sheet.get("outer_margin_px", 0)))


def target_from_labels(labels: np.ndarray) -> np.ndarray:
    foreground = labels > 0
    boundary = np.zeros_like(foreground, np.uint8)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        boundary[foreground & (labels != np.roll(labels, (dy, dx), axis=(0, 1)))] = 1
    boundary[[0, -1], :] = 0
    boundary[:, [0, -1]] = 0
    boundary = cv2.dilate(boundary, np.ones((3, 3), np.uint8)) * foreground
    return np.stack((foreground, boundary), axis=0).astype(np.float32)


def load_sheet_data(manifest_path: str, split: str) -> list[SheetData]:
    path = Path(manifest_path).resolve()
    config = json.loads(path.read_text())
    sheets = []
    for sheet in config["sheets"]:
        if sheet["split"] != split:
            continue
        name = sheet["name"]
        image = np.asarray(Image.open(path.parent / "sheets" / f"{name}.jpeg").convert("RGB"))
        labels = labels_for_sheet(sheet, path.parent, image.shape[:2])
        boundary_xy = np.argwhere(target_from_labels(labels)[1] > 0)
        sheets.append(SheetData(name, image, labels, boundary_xy))
    return sheets


def make_real_tile(sheets: list[SheetData], size: int, seed: int,
                   augment: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Sample an augmented crop of a real sheet and its instance labels."""
    rng = np.random.default_rng(seed)
    sheet = sheets[int(rng.integers(len(sheets)))]
    height, width = sheet.image.shape[:2]
    crop_size = int(rng.integers(max(96, round(size * 0.75)), round(size * 1.4) + 1)) if augment else size
    crop_size = min(crop_size, height, width)
    if sheet.boundary_xy.size and rng.random() < 0.7:
        cy, cx = sheet.boundary_xy[int(rng.integers(len(sheet.boundary_xy)))]
        x = int(np.clip(cx - crop_size // 2 + rng.integers(-crop_size // 4, crop_size // 4 + 1),
                        0, width - crop_size))
        y = int(np.clip(cy - crop_size // 2 + rng.integers(-crop_size // 4, crop_size // 4 + 1),
                        0, height - crop_size))
    else:
        x = int(rng.integers(width - crop_size + 1))
        y = int(rng.integers(height - crop_size + 1))
    image = cv2.resize(sheet.image[y:y + crop_size, x:x + crop_size], (size, size),
                       interpolation=cv2.INTER_AREA if crop_size > size else cv2.INTER_CUBIC)
    labels = cv2.resize(sheet.labels[y:y + crop_size, x:x + crop_size], (size, size),
                        interpolation=cv2.INTER_NEAREST)
    if augment and rng.random() < 0.5:
        image = image[:, ::-1]
        labels = labels[:, ::-1]
    if augment:
        brightness = float(rng.uniform(0.88, 1.12))
        tint = rng.uniform(0.96, 1.04, 3).astype(np.float32)
        image = np.uint8(np.clip(image.astype(np.float32) * brightness * tint, 0, 255))
    return image.transpose(2, 0, 1).astype(np.float32) / 255, target_from_labels(labels)


def load_manifest(manifest_path: str) -> tuple[list[Asset], list[Asset], dict]:
    """Load sheet-level train/validation splits and their reviewed pseudo-labels."""
    path = Path(manifest_path).resolve()
    config = json.loads(path.read_text())
    training, holdout = [], []
    for sheet in config["sheets"]:
        if sheet["split"] == "review":
            continue
        name = sheet["name"]
        image = Image.open(path.parent / "sheets" / f"{name}.jpeg").convert("RGB")
        analysis_dir = path.parent / "analyses" / name
        analysis = json.loads((analysis_dir / "analysis.json").read_text())
        summary = json.loads((analysis_dir / "summary.json").read_text())
        factor, pad = summary["raster"]["factor"], summary["raster"]["pad"]
        selected = set(sheet.get("groups", range(1, len(analysis["groups"]) + 1)))
        selected -= set(sheet.get("exclude_groups", []))
        margin = int(sheet.get("outer_margin_px", 0))
        for number, group in enumerate(analysis["groups"], 1):
            if number not in selected or len(group["paths"]) != 1:
                continue
            polygon = [((x - pad) / factor, (y - pad) / factor) for x, y in group["paths"][0]]
            if len(polygon) < 3:
                continue
            mask = Image.new("L", image.size)
            ImageDraw.Draw(mask).polygon(polygon, fill=255)
            if margin:
                dilated = cv2.dilate(np.asarray(mask), cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
                mask = Image.fromarray(dilated)
            bounds = mask.getbbox()
            if not bounds:
                continue
            x0, y0, x1, y1 = bounds
            bounds = (max(0, x0 - 3), max(0, y0 - 3),
                      min(image.width, x1 + 3), min(image.height, y1 + 3))
            asset = Asset(number, image.crop(bounds), mask.crop(bounds), name)
            (training if sheet["split"] == "train" else holdout).append(asset)
    if len(training) < 5 or len(holdout) < 2:
        raise ValueError("Need several reviewed training and validation stickers.")
    return training, holdout, config


def pastel_background(size: int, rng: np.random.Generator) -> Image.Image:
    if rng.random() < 0.5:
        value = rng.uniform(244, 255)
        base = np.array([value, value, value], np.float32)
    else:
        base = rng.uniform([206, 209, 219], [255, 249, 255]).astype(np.float32)
    coarse = rng.normal(0, 8, (6, 6, 3)).astype(np.float32)
    clouds = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    slope = rng.uniform(-15, 15, (2, 3)).astype(np.float32)
    field = base + clouds + (xx / size - 0.5)[..., None] * slope[0] + (yy / size - 0.5)[..., None] * slope[1]
    return Image.fromarray(np.clip(field, 0, 255).astype(np.uint8), "RGB")


def prepare_asset(asset: Asset, size: int, rng: np.random.Generator,
                  scale_range: tuple[float, float] = (0.23, 0.57),
                  longest_px: int | None = None) -> tuple[Image.Image, Image.Image]:
    longest = (int(longest_px) if longest_px is not None else
               int(rng.uniform(size * scale_range[0], size * scale_range[1])))
    factor = longest / max(asset.image.size)
    dimensions = tuple(max(4, round(d * factor)) for d in asset.image.size)
    image = asset.image.resize(dimensions, Image.Resampling.BICUBIC)
    mask = asset.mask.resize(dimensions, Image.Resampling.BILINEAR)
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    angle = float(rng.uniform(-32, 32))
    image = image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    mask = mask.rotate(angle, Image.Resampling.BILINEAR, expand=True, fillcolor=0)
    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.88, 1.1)))
    return image, mask


def make_instance_scene(assets: list[Asset], size: int, seed: int, *,
                        scale_range: tuple[float, float] = (0.23, 0.57),
                        count_range: tuple[int, int] = (3, 5)) -> tuple[np.ndarray, np.ndarray]:
    """Return a synthetic RGB sheet and its visible sticker instance IDs."""
    rng = np.random.default_rng(seed)
    background = pastel_background(size, rng)
    instances = np.zeros((size, size), np.int16)
    centers: list[tuple[int, int, int, int]] = []
    by_sheet = {name: [asset for asset in assets if asset.sheet == name]
                for name in sorted({asset.sheet for asset in assets})}
    sheet_names = list(by_sheet)
    count = int(rng.integers(count_range[0], count_range[1] + 1))
    for instance in range(1, count + 1):
        sheet = sheet_names[int(rng.integers(len(sheet_names)))]
        candidates = by_sheet[sheet]
        asset = candidates[int(rng.integers(len(candidates)))]
        sticker, mask = prepare_asset(asset, size, rng, scale_range)
        if not centers:
            cx, cy = int(rng.uniform(size * 0.35, size * 0.65)), int(rng.uniform(size * 0.35, size * 0.65))
        else:
            prior = centers[int(rng.integers(len(centers)))]
            angle = float(rng.uniform(0, 2 * np.pi))
            radius = float(rng.uniform(0.48, 0.8)) * (max(prior[2:]) + max(sticker.size)) / 2
            cx = int(np.clip(prior[0] + np.cos(angle) * radius, size * 0.1, size * 0.9))
            cy = int(np.clip(prior[1] + np.sin(angle) * radius, size * 0.1, size * 0.9))
        x, y = cx - sticker.width // 2, cy - sticker.height // 2
        centers.append((cx, cy, sticker.width, sticker.height))
        background.paste(sticker, (x, y), mask)
        left, top = max(0, x), max(0, y)
        right, bottom = min(size, x + sticker.width), min(size, y + sticker.height)
        if left >= right or top >= bottom:
            continue
        local = np.asarray(mask.crop((left - x, top - y, right - x, bottom - y)))
        region = instances[top:bottom, left:right]
        region[local >= 128] = instance

    image = np.asarray(background, dtype=np.float32).transpose(2, 0, 1) / 255
    return image, instances


def make_scene(assets: list[Asset], size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    image, instances = make_instance_scene(assets, size, seed)
    return image, target_from_labels(instances)
