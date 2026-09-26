#!/usr/bin/env python3
"""Compare ONNX models against hand-corrected outer sticker contours."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from infer import run_tiled
from manual_data import DEFAULT_ANNOTATIONS, load_manual_data


def contour(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(np.uint8(mask), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)) > 0


def seeded_component(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Ignore a neighboring sticker visible at the edge of a review crop."""
    count, components = cv2.connectedComponents(np.uint8(prediction), connectivity=8)
    if count < 2:
        return prediction
    overlaps = [np.count_nonzero((components == i) & truth) for i in range(1, count)]
    return components == (int(np.argmax(overlaps)) + 1)


def tolerant_boundary_f1(prediction: np.ndarray, truth: np.ndarray,
                         radius: int = 5) -> float:
    predicted, expected = contour(prediction), contour(truth)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    near_truth = cv2.dilate(np.uint8(expected), kernel) > 0
    near_predicted = cv2.dilate(np.uint8(predicted), kernel) > 0
    precision = np.count_nonzero(predicted & near_truth) / max(1, predicted.sum())
    recall = np.count_nonzero(expected & near_predicted) / max(1, expected.sum())
    return float(2 * precision * recall / max(1e-8, precision + recall))


def outline_panel(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    panel = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(np.uint8(mask), (256, 256), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, color, 2, cv2.LINE_AA)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="ONNX model; repeat to compare")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("threshold must be between zero and one")
    stickers = load_manual_data(args.annotations, args.split)
    if not stickers:
        parser.error(f"no manual stickers with split {args.split}")
    models = []
    for model_path in args.model:
        path = Path(model_path).resolve()
        tile = json.loads((path.parent / "manifest.json").read_text())["size"]
        models.append((path, tile, ort.InferenceSession(str(path),
                       providers=["CPUExecutionProvider"])))

    report = {"split": args.split, "threshold": args.threshold,
              "instance_selection": "predicted connected component overlapping the target label",
              "stickers": [s.name for s in stickers],
              "models": {}}
    panels = []
    for sticker in stickers:
        truth = sticker.labels > 0
        row = [outline_panel(sticker.image, truth, (30, 200, 30))]
        for path, tile, session in models:
            probability = run_tiled(sticker.image, session, tile)[0]
            prediction = seeded_component(probability >= args.threshold, truth)
            row.append(outline_panel(sticker.image, prediction, (255, 35, 35)))
            intersection = np.count_nonzero(prediction & truth)
            union = np.count_nonzero(prediction | truth)
            report["models"].setdefault(path.parent.name, {"per_sticker": {}})["per_sticker"][
                sticker.name] = {
                    "foreground_iou": round(intersection / max(1, union), 4),
                    "boundary_f1_within_5px": round(tolerant_boundary_f1(prediction, truth), 4),
                }
        panels.append(np.concatenate(row, axis=1))
    for result in report["models"].values():
        metrics = list(result["per_sticker"].values())
        result["mean_foreground_iou"] = round(float(np.mean(
            [item["foreground_iou"] for item in metrics])), 4)
        result["mean_boundary_f1_within_5px"] = round(float(np.mean(
            [item["boundary_f1_within_5px"] for item in metrics])), 4)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(panels, axis=0)).save(out / "comparison.png")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out), **report}))


if __name__ == "__main__":
    main()
