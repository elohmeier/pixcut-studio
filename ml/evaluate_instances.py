#!/usr/bin/env python3
"""Score full-sheet instance proposals against reviewed reference contours.

Only entries marked ``reviewed`` contribute to contour means. Other entries
still document the intended instance count, but their boundaries are ambiguous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate_manual import tolerant_boundary_f1


def reference_masks(path: Path) -> tuple[dict, np.ndarray]:
    config = json.loads(path.read_text())
    width, height = config["image_size"]
    masks = []
    for item in config["annotations"]:
        x0, y0, x1, y1 = item["box"]
        mask = np.zeros((height, width), np.uint8)
        points = np.asarray(item["points"], np.int32) + [x0, y0]
        cv2.fillPoly(mask, [points], 1)
        masks.append(mask)
    return config, np.stack(masks)


def bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    return int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1


def iou_matrix(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth_bounds = [bounds(mask) for mask in truth]
    pred_bounds = [bounds(mask) for mask in prediction]
    truth_area = truth.sum(axis=(1, 2))
    pred_area = prediction.sum(axis=(1, 2))
    scores = np.zeros((len(truth), len(prediction)), np.float32)
    for i, a in enumerate(truth_bounds):
        if a is None:
            continue
        for j, b in enumerate(pred_bounds):
            if b is None:
                continue
            x0, y0 = max(a[0], b[0]), max(a[1], b[1])
            x1, y1 = min(a[2], b[2]), min(a[3], b[3])
            if x1 <= x0 or y1 <= y0:
                continue
            intersection = np.count_nonzero(
                truth[i, y0:y1, x0:x1] & prediction[j, y0:y1, x0:x1])
            scores[i, j] = intersection / max(1, truth_area[i] + pred_area[j] - intersection)
    return scores


def evaluate(annotations: Path, predictions: Path) -> dict:
    config, truth = reference_masks(annotations)
    with np.load(predictions) as data:
        candidate = np.uint8(data["masks"] > 0)
        ids = [str(i) for i in data["ids"]]
    if candidate.shape[1:] != truth.shape[1:]:
        raise ValueError("Prediction size differs from reference image")
    scores = iou_matrix(truth, candidate)
    matched = {}
    if scores.size:
        rows, cols = linear_sum_assignment(-scores)
        matched = {int(i): int(j) for i, j in zip(rows, cols) if scores[i, j] >= 0.1}
    per_instance = []
    clean_iou, clean_f1 = [], []
    for i, item in enumerate(config["annotations"]):
        j = matched.get(i)
        iou = float(scores[i, j]) if j is not None else 0.0
        row = {"id": item["id"], "quality": item.get("quality", "unreviewed"),
               "matched_prediction": ids[j] if j is not None else None,
               "iou": round(iou, 4)}
        if item.get("quality") == "reviewed":
            f1 = (tolerant_boundary_f1(candidate[j] > 0, truth[i] > 0)
                  if j is not None else 0.0)
            row["boundary_f1_within_5px"] = round(f1, 4)
            clean_iou.append(iou)
            clean_f1.append(f1)
        per_instance.append(row)
    result = {"reference": str(annotations.resolve()),
              "predictions": str(predictions.resolve()),
              "reference_instances": len(truth),
              "reviewed_contours": len(clean_iou),
              "predicted_instances": len(candidate),
              "matched_instances_iou_at_least_0_1": len(matched),
              "predicted_overlap_px": int(np.count_nonzero(candidate.sum(axis=0) > 1)),
              "mean_reviewed_iou": round(float(np.mean(clean_iou)), 4) if clean_iou else None,
              "mean_reviewed_boundary_f1_within_5px":
                  round(float(np.mean(clean_f1)), 4) if clean_f1 else None,
              "per_instance": per_instance}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True, help="instances.npz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = evaluate(Path(args.annotations), Path(args.predictions))
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "per_instance"}))


if __name__ == "__main__":
    main()
