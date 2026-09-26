#!/usr/bin/env python3
"""Propose one sticker box per instance from overview and overlapping tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def positions(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    values = list(range(0, length - window + 1, stride))
    if values[-1] != length - window:
        values.append(length - window)
    return values


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1, area_a + area_b - intersection)


def deduplicate(candidates: list[dict], threshold: float = 0.5) -> list[dict]:
    kept = []
    for candidate in sorted(candidates, key=lambda row: row["confidence"], reverse=True):
        if all(iou(candidate["box"], earlier["box"]) < threshold for earlier in kept):
            kept.append(candidate)
    return kept


def predict(model: YOLO, image: Image.Image, *, confidence: float = 0.2,
            window: int = 512, stride: int = 384,
            overview_size: int = 768) -> list[dict]:
    width, height = image.size
    candidates = []
    overview = model.predict(image, imgsz=overview_size, conf=confidence,
                             iou=0.55, device="cpu", verbose=False)[0]
    for box, score in zip(overview.boxes.xyxy.cpu().numpy(),
                          overview.boxes.conf.cpu().numpy()):
        candidates.append({"box": [float(value) for value in box],
                           "confidence": float(score), "source": "overview"})
    for top in positions(height, window, stride):
        for left in positions(width, window, stride):
            tile = image.crop((left, top, min(width, left + window),
                               min(height, top + window)))
            result = model.predict(tile, imgsz=window, conf=confidence,
                                   iou=0.55, device="cpu", verbose=False)[0]
            for box, score in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy()):
                x0, y0, x1, y1 = map(float, box)
                if ((left > 0 and x0 < 6) or (top > 0 and y0 < 6) or
                    (left + window < width and x1 > tile.width - 6) or
                    (top + window < height and y1 > tile.height - 6)):
                    continue
                candidates.append({"box": [x0 + left, y0 + top,
                                            x1 + left, y1 + top],
                                   "confidence": float(score),
                                   "source": f"tile:{left},{top}"})
    return deduplicate(candidates)


def save_overlay(image: Image.Image, boxes: list[dict], out: Path) -> None:
    rgb = np.asarray(image).copy()
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for index, row in enumerate(boxes, 1):
        x0, y0, x1, y1 = map(round, row["box"])
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (20, 20, 230), 2)
        cv2.putText(canvas, f"{index}:{row['confidence']:.2f}", (x0, max(14, y0 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--overview-size", type=int, default=768)
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    model = YOLO(args.model)
    boxes = predict(model, image, confidence=args.confidence,
                    window=args.window, stride=args.stride,
                    overview_size=args.overview_size)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"image": str(Path(args.image).resolve()),
              "model": str(Path(args.model).resolve()),
              "confidence": args.confidence, "window": args.window,
              "stride": args.stride, "overview_size": args.overview_size,
              "boxes": boxes}
    (out / "boxes.json").write_text(json.dumps(report, indent=2) + "\n")
    save_overlay(image, boxes, out / "boxes.png")
    print(json.dumps({"out": str(out), "boxes": len(boxes)}), flush=True)


if __name__ == "__main__":
    main()
