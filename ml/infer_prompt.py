#!/usr/bin/env python3
"""Refine each proposed sticker box into one outer-border mask and path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from evaluate_instances import bounds, reference_masks
from prompt_data import prompt_crop


def load_boxes(annotations: str | None, proposals: str | None) -> tuple[list[dict], str]:
    if annotations:
        config, masks = reference_masks(Path(annotations))
        boxes = [{"id": item["id"], "box": list(bounds(mask)),
                  "confidence": 1.0} for item, mask in zip(config["annotations"], masks)]
        return boxes, "reference boxes"
    data = json.loads(Path(proposals).read_text())
    return [{"id": str(index + 1), **item}
            for index, item in enumerate(data["boxes"])], "automatic boxes"


def predict_one(image: np.ndarray, box: list[float], session: ort.InferenceSession,
                size: int) -> np.ndarray:
    features, _, transform = prompt_crop(image, np.zeros(image.shape[:2], np.uint8),
                                          tuple(box), size)
    logit = session.run(["logits"], {"image_and_prompt": features[None]})[0][0, 0]
    probability = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    side = transform["side"]
    affine = np.float32([[side / size, 0, transform["left"]],
                         [0, side / size, transform["top"]]])
    projected = cv2.warpAffine(probability, affine,
                               (image.shape[1], image.shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x0, y0, x1, y1 = box
    margin = max(5.0, 0.12 * max(x1 - x0, y1 - y0))
    left = max(0, int(np.floor(x0 - margin)))
    top = max(0, int(np.floor(y0 - margin)))
    right = min(image.shape[1], int(np.ceil(x1 + margin)))
    bottom = min(image.shape[0], int(np.ceil(y1 + margin)))
    projected[:top] = 0
    projected[bottom:] = 0
    projected[:, :left] = 0
    projected[:, right:] = 0
    return projected


def choose_one_component(mask: np.ndarray, box: list[float]) -> np.ndarray:
    count, components, stats, centroids = cv2.connectedComponentsWithStats(
        np.uint8(mask), 8)
    if count < 2:
        return np.zeros_like(mask, bool)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    scores = []
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        distance = np.linalg.norm(centroids[index] - [cx, cy])
        scores.append(area / (1 + distance / max(1, box[2] - box[0], box[3] - box[1])))
    return components == (int(np.argmax(scores)) + 1)


def arbitrate_stream(probabilities: Iterable[np.ndarray], boxes: list[dict],
                     shape: tuple[int, int], threshold: float = 0.5) -> np.ndarray:
    """Give each pixel to one sticker without stacking full-sheet float maps."""
    if not len(boxes):
        return np.zeros((0, *shape), bool)
    height, width = shape
    winner = np.full(shape, -1, np.int16)
    best_score = np.full(shape, -np.inf, np.float32)
    for index, (probability, row) in enumerate(zip(probabilities, boxes, strict=True)):
        if probability.shape != shape:
            raise ValueError("Projected mask size differs from image")
        x0, y0, x1, y1 = row["box"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        radius = max(1.0, x1 - x0, y1 - y0)
        dx = (np.arange(width, dtype=np.float32) - cx) ** 2
        dy = (np.arange(height, dtype=np.float32) - cy) ** 2
        score = probability - (0.08 / radius) * np.sqrt(dy[:, None] + dx[None, :])
        selected = (probability >= threshold) & (score > best_score)
        best_score[selected] = score[selected]
        winner[selected] = index
    return np.array([choose_one_component(winner == index, row["box"])
                     for index, row in enumerate(boxes)])


def arbitrate(probabilities: np.ndarray, boxes: list[dict],
              threshold: float = 0.5) -> np.ndarray:
    """Array convenience wrapper used by the small synthetic tests."""
    return arbitrate_stream(iter(probabilities), boxes,
                            probabilities.shape[1:], threshold)


def save(image: np.ndarray, boxes: list[dict], masks: np.ndarray,
         out: Path, min_area: int = 100) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    canvas = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    paths, kept_masks, ids = [], [], []
    path_masks = []
    for row, mask in zip(boxes, masks):
        if int(mask.sum()) < min_area:
            continue
        contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_area:
            continue
        points = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
        path_mask = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(path_mask, [points], 1)
        path_masks.append(path_mask)
        kept_masks.append(np.uint8(mask))
        ids.append(row["id"])
        paths.append({"id": row["id"], "box": row["box"],
                      "confidence": row["confidence"],
                      "area_px": int(mask.sum()), "points": points.tolist()})
        cv2.polylines(canvas, [points], True, (25, 25, 230), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(row["id"]), tuple(points[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 20, 20), 1, cv2.LINE_AA)
    stack = np.stack(kept_masks) if kept_masks else np.zeros((0, *image.shape[:2]), np.uint8)
    filled = (np.stack(path_masks) if path_masks else np.zeros_like(stack))
    np.savez_compressed(out / "instances.npz", masks=stack, ids=np.asarray(ids))
    (out / "paths.json").write_text(json.dumps(paths) + "\n")
    cv2.imwrite(str(out / "overlay.png"), canvas)
    return {"candidate_boxes": len(boxes), "instance_paths": len(paths),
            "mask_overlap_px": int(np.count_nonzero(stack.sum(axis=0) > 1)),
            "filled_path_overlap_px": int(np.count_nonzero(filled.sum(axis=0) > 1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotations", help="Reference boxes; refinement-only check")
    source.add_argument("--boxes", help="Automatic boxes.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    boxes, box_source = load_boxes(args.annotations, args.boxes)
    model = Path(args.model).resolve()
    size = json.loads((model.parent / "manifest.json").read_text())["size"]
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    probabilities = (predict_one(image, row["box"], session, size) for row in boxes)
    masks = arbitrate_stream(probabilities, boxes, image.shape[:2],
                              threshold=args.threshold)
    out = Path(args.out).resolve()
    report = {**save(image, boxes, masks, out), "image": str(Path(args.image).resolve()),
              "model": str(model), "box_source": box_source,
              "threshold": args.threshold}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out), **report}), flush=True)


if __name__ == "__main__":
    main()
