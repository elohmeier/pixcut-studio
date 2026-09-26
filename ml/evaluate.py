#!/usr/bin/env python3
"""Score ONNX predictions against held-out sheet pseudo-labels and save overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from infer import run_tiled
from sticker_data import labels_for_sheet, target_from_labels


def scores(prediction: np.ndarray, target: np.ndarray) -> dict:
    truth = target[0] > 0
    def iou_at(threshold: float) -> float:
        foreground = prediction[0] >= threshold
        return float((foreground & truth).sum() / max(1, (foreground | truth).sum()))

    true_boundary = target[1] > 0
    def f1_at(threshold: float) -> float:
        boundary = prediction[1] >= threshold
        true_positive = (boundary & true_boundary).sum()
        precision = true_positive / max(1, boundary.sum())
        recall = true_positive / max(1, true_boundary.sum())
        return float(2 * precision * recall / max(1e-8, precision + recall))

    boundary_f1 = {}
    for threshold in (0.2, 0.35, 0.5):
        boundary_f1[str(threshold)] = round(f1_at(threshold), 4)
    thresholds = np.arange(0.1, 0.951, 0.025)
    best_fg = max(thresholds, key=iou_at)
    best_boundary = max(thresholds, key=f1_at)
    return {"foreground_iou": round(iou_at(0.5), 4), "boundary_f1": boundary_f1,
            "calibrated": {"foreground_threshold": round(float(best_fg), 3),
                           "foreground_iou": round(iou_at(best_fg), 4),
                           "boundary_threshold": round(float(best_boundary), 3),
                           "boundary_f1": round(f1_at(best_boundary), 4)},
            "target_foreground_fraction": round(float(truth.mean()), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default=str(Path(__file__).parent / "data" / "sheets.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    root = manifest.parent
    config = json.loads(manifest.read_text())
    model = Path(args.model).resolve()
    tile = json.loads((model.parent / "manifest.json").read_text())["size"]
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for sheet in config["sheets"]:
        if sheet["split"] != "validation":
            continue
        name = sheet["name"]
        image = np.asarray(Image.open(root / "sheets" / f"{name}.jpeg").convert("RGB"))
        target = target_from_labels(labels_for_sheet(sheet, root, image.shape[:2]))
        prediction = run_tiled(image, session, tile)
        report[name] = scores(prediction, target)
        classical = labels_for_sheet({**sheet, "outer_margin_px": 0}, root, image.shape[:2]) > 0
        truth = target[0] > 0
        report[name]["classical_unexpanded_foreground_iou"] = round(float(
            (classical & truth).sum() / max(1, (classical | truth).sum())), 4)
        for channel, label in enumerate(("foreground", "boundary")):
            Image.fromarray(np.uint8(prediction[channel] * 255)).save(out / f"{name}-{label}.png")
        review = image.copy()
        review[target[1] > 0] = [30, 120, 255]
        review[prediction[1] >= 0.5] = [255, 40, 40]
        Image.fromarray(review).save(out / f"{name}-review.png")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if len(report) == 1:
        sheet_name, result = next(iter(report.items()))
        calibration = {"source": sheet_name,
                       "label_warning": "Thresholds optimized on approximate expanded CLI masks",
                       "foreground_threshold": result["calibrated"]["foreground_threshold"],
                       "boundary_threshold": result["calibrated"]["boundary_threshold"]}
        (out / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
