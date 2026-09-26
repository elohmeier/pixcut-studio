#!/usr/bin/env python3
"""Grow CLI sticker groups to model foreground and inspect outer cut proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from infer import run_tiled
from sticker_data import labels_for_sheet, labels_from_analysis


def grow_groups(seeds: np.ndarray, foreground: np.ndarray, threshold: float,
                close_px: int = 1):
    """Assign model foreground to its nearest classical seed, keeping one piece per group."""
    if not np.any(seeds):
        raise ValueError("The analysis has no seed groups.")
    _, nearest = cv2.distanceTransformWithLabels(
        np.uint8(seeds == 0), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    nearest_to_group = np.zeros(int(nearest.max()) + 1, np.int16)
    for group in np.unique(seeds):
        if group:
            component_ids = nearest[seeds == group]
            nearest_to_group[np.bincount(component_ids).argmax()] = group
    assigned = nearest_to_group[nearest]
    proposal = np.zeros_like(seeds)
    paths = []
    for group in np.unique(seeds):
        if group == 0:
            continue
        candidate = np.uint8(((assigned == group) & (foreground >= threshold)) |
                             (seeds == group))
        if close_px > 1:
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                  (close_px, close_px)))
        count, labels, _, _ = cv2.connectedComponentsWithStats(candidate, 8)
        if count < 2:
            continue
        overlaps = [np.count_nonzero((labels == i) & (seeds == group))
                    for i in range(1, count)]
        largest_seed_component = int(np.argmax(overlaps)) + 1
        component = np.uint8(labels == largest_seed_component)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(component)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        proposal[filled > 0] = group
        simplified = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
        paths.append({"group": int(group), "area_px": int(np.count_nonzero(filled)),
                      "points": simplified.astype(int).tolist()})
    return proposal, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sheet", help="Sheet name in the reviewed data manifest")
    parser.add_argument("--manifest", default=str(Path(__file__).parent / "data" / "sheets.json"))
    parser.add_argument("--image", help="Source image when not using --sheet")
    parser.add_argument("--analysis", help="CLI analysis.json when not using --sheet")
    parser.add_argument("--threshold", type=float,
                        help="Defaults to the model's calibrated foreground threshold or 0.5")
    parser.add_argument("--close-px", type=int, default=1,
                        help="Odd morphological closing width for gaps in pale stickers")
    args = parser.parse_args()
    if args.close_px < 1 or args.close_px % 2 != 1:
        parser.error("close-px must be a positive odd number")
    if bool(args.sheet) == bool(args.image and args.analysis):
        parser.error("provide --sheet or both --image and --analysis")

    model = Path(args.model).resolve()
    tile = json.loads((model.parent / "manifest.json").read_text())["size"]
    calibration_path = model.parent / "calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    threshold = (args.threshold if args.threshold is not None
                 else calibration.get("foreground_threshold", 0.5))
    if not 0 < threshold < 1:
        parser.error("threshold must be between zero and one")
    sheet_config = None
    if args.sheet:
        root = Path(args.manifest).resolve().parent
        config = json.loads(Path(args.manifest).read_text())
        sheet_config = next((entry for entry in config["sheets"] if entry["name"] == args.sheet), None)
        if sheet_config is None:
            parser.error(f"unknown sheet: {args.sheet}")
        image_path = root / "sheets" / f"{args.sheet}.jpeg"
        analysis_path = root / "analyses" / args.sheet / "analysis.json"
    else:
        image_path = Path(args.image).resolve()
        analysis_path = Path(args.analysis).resolve()
    image = np.asarray(Image.open(image_path).convert("RGB"))
    seeds = (labels_for_sheet({**sheet_config, "outer_margin_px": 0}, root, image.shape[:2])
             if sheet_config else labels_from_analysis(analysis_path, image.shape[:2]))
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    foreground = run_tiled(image, session, tile)[0]
    proposal, paths = grow_groups(seeds, foreground, threshold, args.close_px)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    for path in paths:
        points = np.asarray(path["points"], np.int32)
        cv2.polylines(overlay, [points], True, (15, 25, 230), 2, cv2.LINE_AA)
    cv2.imwrite(str(out / "proposal-overlay.png"), overlay)
    Image.fromarray(np.uint8(proposal > 0) * 255).save(out / "proposal-mask.png")
    (out / "proposals.json").write_text(json.dumps(paths) + "\n")
    report = {"model": str(model), "image": str(image_path), "threshold": threshold,
              "close_px": args.close_px,
              "seed_groups": int(len(np.unique(seeds)) - 1), "proposals": len(paths)}
    path_coverage = np.zeros(image.shape[:2], np.uint8)
    for path in paths:
        raster = np.zeros_like(path_coverage)
        cv2.fillPoly(raster, [np.asarray(path["points"], np.int32)], 1)
        path_coverage += raster
    report["overlapping_path_pixels"] = int(np.count_nonzero(path_coverage > 1))
    if sheet_config and sheet_config["split"] != "review":
        truth = labels_for_sheet(sheet_config, root, image.shape[:2])
        union = proposal > 0
        expected = truth > 0
        report["foreground_iou_vs_pseudo_labels"] = round(float(
            (union & expected).sum() / max(1, (union | expected).sum())), 4)
        per_instance = [float(((proposal == g) & (truth == g)).sum() /
                              max(1, ((proposal == g) | (truth == g)).sum()))
                        for g in np.unique(truth) if g]
        report["mean_instance_iou_vs_pseudo_labels"] = round(float(np.mean(per_instance)), 4)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out), **report}))


if __name__ == "__main__":
    main()
