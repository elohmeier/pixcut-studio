#!/usr/bin/env python3
"""Trace draft outer sticker contours from manually selected image regions.

This uses only source-image pixel values. It never consults the cut detector,
trained model, or SAM. The result still requires visual review before it is a
reference annotation, especially where printed white borders touch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def trace(image: np.ndarray, entry: dict, default: dict) -> tuple[np.ndarray, dict]:
    x0, y0, x1, y1 = entry["box"]
    height, width = image.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid box for {entry['id']}")
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    threshold = int(entry.get("threshold", default["threshold"]))
    polarity = entry.get("polarity", default["polarity"])
    if polarity == "dark":
        candidate = np.uint8(gray < threshold)
    elif polarity == "light":
        candidate = np.uint8(gray > threshold)
    else:
        raise ValueError(f"Invalid polarity for {entry['id']}: {polarity}")
    close_px = int(entry.get("close_px", default.get("close_px", 3)))
    if close_px > 1:
        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    if "clip_polygon" in entry:
        allowed = np.zeros_like(candidate)
        polygon = np.asarray(entry["clip_polygon"], np.int32) - [x0, y0]
        cv2.fillPoly(allowed, [polygon], 1)
        candidate &= allowed
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No image contour for {entry['id']}")
    seed = entry.get("seed")
    if seed:
        local_seed = (float(seed[0] - x0), float(seed[1] - y0))
        enclosed = [c for c in contours if cv2.pointPolygonTest(c, local_seed, False) >= 0]
    else:
        enclosed = []
    selected = max(enclosed or contours, key=cv2.contourArea)
    if cv2.contourArea(selected) < 100:
        raise ValueError(f"Image contour too small for {entry['id']}")
    mask = np.zeros_like(candidate)
    cv2.drawContours(mask, [selected], -1, 1, cv2.FILLED)
    offset = int(entry.get("offset_px", default.get("offset_px", 0)))
    if offset:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(offset) + 1,) * 2)
        mask = cv2.dilate(mask, kernel) if offset > 0 else cv2.erode(mask, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected = max(contours, key=cv2.contourArea)
    epsilon = float(entry.get("simplify_px", default.get("simplify_px", 1.0)))
    points = cv2.approxPolyDP(selected, epsilon, True).reshape(-1, 2)
    diagnostics = {
        "area_px": int(np.count_nonzero(mask)),
        "vertices": len(points),
        "touches_box_edge": bool(np.any(mask[0]) or np.any(mask[-1]) or
                                 np.any(mask[:, 0]) or np.any(mask[:, -1])),
        "seed_enclosed": bool(enclosed),
    }
    return points, diagnostics


def build(draft_path: Path, out_path: Path, overlay_path: Path) -> dict:
    draft_path = draft_path.resolve()
    config = json.loads(draft_path.read_text())
    image_path = draft_path.parent.parent.parent / "sheets" / f'{config["image"]}.jpeg'
    image = np.asarray(Image.open(image_path).convert("RGB"))
    if list(image.shape[1::-1]) != config["image_size"]:
        raise ValueError(f"Image size differs from {image_path}")
    output = {
        "version": 1,
        "image": config["image"],
        "image_size": config["image_size"],
        "coordinate_system": "local integer pixels within each box",
        "method": "Image-pixel contour proposals from manually chosen regions; inspect overlay before use. No detector or ML model supplied these pixels.",
        "split_policy": config.get("split_policy", "Whole sheet reserved from training"),
        "annotations": [],
    }
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    occupancy = np.zeros(image.shape[:2], np.uint8)
    diagnostic_rows = []
    for index, entry in enumerate(config["annotations"], 1):
        points, diagnostics = trace(image, entry, config["trace_defaults"])
        x0, y0, x1, y1 = entry["box"]
        target = {"id": entry["id"], "split": entry.get("split", "benchmark"),
                  "box": entry["box"], "points": points.tolist(),
                  "quality": entry.get("quality", "box-clipped" if
                                       diagnostics["touches_box_edge"] else "reviewed")}
        if "note" in entry:
            target["note"] = entry["note"]
        output["annotations"].append(target)
        whole = points + [x0, y0]
        cv2.polylines(overlay, [whole], True, (20, 25, 230), 2, cv2.LINE_AA)
        cv2.putText(overlay, str(index), tuple(whole[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (240, 30, 20), 2, cv2.LINE_AA)
        local = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(local, [points], 1)
        occupancy[y0:y1, x0:x1] += local
        diagnostic_rows.append({"id": entry["id"], **diagnostics})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    cv2.imwrite(str(overlay_path), overlay)
    report = {"image": config["image"], "instances": len(output["annotations"]),
              "overlap_px": int(np.count_nonzero(occupancy > 1)),
              "diagnostics": diagnostic_rows}
    report_path = overlay_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overlay", required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(args.draft), Path(args.out), Path(args.overlay))))


if __name__ == "__main__":
    main()
