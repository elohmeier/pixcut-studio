#!/usr/bin/env python3
"""Score automatic sticker boxes against tight reference contour bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from detect_stickers import iou
from evaluate_instances import bounds, reference_masks


def evaluate(annotations: Path, predictions: Path) -> dict:
    config, masks = reference_masks(annotations)
    boxes = json.loads(predictions.read_text())["boxes"]
    truth = [bounds(mask) for mask in masks]
    matrix = np.array([[iou(reference, row["box"]) for row in boxes]
                       for reference in truth], np.float32)
    matched = {}
    if matrix.size:
        rows, cols = linear_sum_assignment(-matrix)
        matched = {int(i): (int(j), float(matrix[i, j])) for i, j in zip(rows, cols)
                   if matrix[i, j] >= 0.1}
    per_sticker = [{"id": item["id"],
                    "prediction": matched[index][0] + 1 if index in matched else None,
                    "box_iou": round(matched[index][1], 4) if index in matched else 0.0}
                   for index, item in enumerate(config["annotations"])]
    return {"reference": str(annotations.resolve()),
            "predictions": str(predictions.resolve()),
            "intended": len(truth), "predicted": len(boxes),
            "matched_iou_at_least_0_3": sum(row["box_iou"] >= 0.3 for row in per_sticker),
            "matched_iou_at_least_0_5": sum(row["box_iou"] >= 0.5 for row in per_sticker),
            "mean_matched_box_iou": round(float(np.mean(
                [row["box_iou"] for row in per_sticker if row["prediction"]])), 4)
                if matched else None,
            "per_sticker": per_sticker}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = evaluate(Path(args.annotations), Path(args.predictions))
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "per_sticker"}), flush=True)


if __name__ == "__main__":
    main()
