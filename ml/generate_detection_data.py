#!/usr/bin/env python3
"""Generate class-agnostic sticker boxes from synthetic sheets for detection training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sticker_data import load_manifest, pastel_background, prepare_asset


def boxes_from_instances(labels: np.ndarray, *, min_area: int = 70) -> list[tuple[float, ...]]:
    """YOLO normalized cx, cy, width, height for visible sticker masks."""
    height, width = labels.shape
    boxes = []
    for identity in np.unique(labels):
        if identity == 0:
            continue
        mask = np.uint8(labels == identity)
        if int(mask.sum()) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(mask)
        if min(w, h) < 8:
            continue
        boxes.append(((x + w / 2) / width, (y + h / 2) / height,
                      w / width, h / height))
    return boxes


def make_detection_scene(assets, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Compose mostly separate stickers at sizes related to their source sheet."""
    rng = np.random.default_rng(seed)
    background = pastel_background(size, rng)
    labels = np.zeros((size, size), np.int16)
    overview = rng.random() < 0.45
    target_count = int(rng.integers(6, 14))
    placed = 0
    for _ in range(target_count * 14):
        if placed >= target_count:
            break
        asset = assets[int(rng.integers(len(assets)))]
        factor = rng.uniform(0.32, 0.72) if overview else rng.uniform(0.65, 1.15)
        longest = int(np.clip(max(asset.image.size) * factor, 22, size * 0.65))
        sticker, mask = prepare_asset(asset, size, rng, longest_px=longest)
        bounds = mask.getbbox()
        if not bounds:
            continue
        sticker, mask = sticker.crop(bounds), mask.crop(bounds)
        if sticker.width >= size or sticker.height >= size:
            continue
        local = np.asarray(mask) >= 128
        if local.sum() < 70:
            continue
        for _ in range(25):
            x = int(rng.integers(size - sticker.width + 1))
            y = int(rng.integers(size - sticker.height + 1))
            occupied = np.uint8(labels[y:y + sticker.height,
                                       x:x + sticker.width] > 0)
            if rng.random() < 0.8:
                occupied = cv2.dilate(occupied, np.ones((5, 5), np.uint8))
            if np.count_nonzero(local & (occupied > 0)):
                continue
            background.paste(sticker, (x, y), mask)
            region = labels[y:y + sticker.height, x:x + sticker.width]
            region[local] = placed + 1
            placed += 1
            break
    image = np.asarray(background, dtype=np.float32).transpose(2, 0, 1) / 255
    return image, labels


def generate(assets, out: Path, count: int, size: int, seed: int) -> dict:
    image_dir, label_dir = out / "images", out / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    observed = []
    for index in range(count):
        image, instances = make_detection_scene(assets, size, seed + index)
        boxes = boxes_from_instances(instances)
        rgb = np.uint8(np.round(image.transpose(1, 2, 0) * 255))
        Image.fromarray(rgb).save(image_dir / f"{index:05d}.jpg", quality=92)
        (label_dir / f"{index:05d}.txt").write_text(
            "".join("0 " + " ".join(f"{value:.6f}" for value in box) + "\n"
                    for box in boxes))
        observed.append(len(boxes))
    return {"sheets": count, "visible_stickers": int(sum(observed)),
            "min_per_sheet": min(observed), "max_per_sheet": max(observed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(Path(__file__).parent / "data" / "sheets.json"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--train", type=int, default=480)
    parser.add_argument("--val", type=int, default=80)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.train, args.val, args.size) < 1:
        parser.error("train, val and size must be positive")
    train_assets, val_assets, _ = load_manifest(args.manifest)
    out = Path(args.out).resolve()
    training = generate(train_assets, out / "train", args.train, args.size, args.seed)
    validation = generate(val_assets, out / "val", args.val, args.size,
                          args.seed + 1_000_000)
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: train/images\nval: val/images\nnames:\n  0: sticker\n")
    report = {"seed": args.seed, "size": args.size,
              "train_source_sheets": sorted({asset.sheet for asset in train_assets}),
              "validation_source_sheets": sorted({asset.sheet for asset in val_assets}),
              "train": training, "val": validation}
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
