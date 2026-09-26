#!/usr/bin/env python3
"""Split a two-channel ONNX sticker prediction into automatic instances.

This is a reproducible baseline, not a cut-safe detector. It uses high-confidence
interiors as markers and the learned boundary map as watershed elevation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from skimage.segmentation import watershed

from infer import run_tiled


def merge_enclosed_regions(labels: np.ndarray, *, max_area_ratio: float = 0.1) -> np.ndarray:
    """Remove tiny watershed islands inside a larger touching cut silhouette.

    A small region must touch its parent and lie almost entirely inside the
    parent's filled *outer* contour. This catches interior details such as an
    eye being mistaken for a separate sticker, without merging nearby stickers
    across a background gap.
    """
    labels = labels.copy()
    kernel = np.ones((3, 3), np.uint8)
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    areas = dict(zip(ids.tolist(), counts.tolist()))
    for child in sorted(areas, key=areas.get):
        child_mask = np.uint8(labels == child)
        if not child_mask.any():
            continue
        border = (cv2.dilate(child_mask, kernel) > 0) & (child_mask == 0)
        neighbors = [int(i) for i in np.unique(labels[border]) if i > 0 and i != child]
        for parent in sorted(neighbors, key=lambda i: areas.get(i, 0)):
            if areas.get(child, 0) > max_area_ratio * areas.get(parent, 0):
                continue
            contours, _ = cv2.findContours(np.uint8(labels == parent),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            filled = np.zeros(labels.shape, np.uint8)
            cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
            coverage = np.count_nonzero((child_mask > 0) & (filled > 0)) / areas[child]
            if coverage >= 0.98:
                labels[child_mask > 0] = parent
                areas[parent] += areas[child]
                break
    return labels


def split(foreground: np.ndarray, boundary: np.ndarray, *, threshold: float = 0.5,
          core_threshold: float = 0.94, core_boundary_max: float = 0.20,
          min_core_area: int = 500, erosion_px: int = 9,
          merge_enclosed: bool = False) -> np.ndarray:
    mask = foreground >= threshold
    core = np.uint8((foreground >= core_threshold) & (boundary <= core_boundary_max))
    if erosion_px > 1:
        core = cv2.erode(core, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erosion_px, erosion_px)))
    count, components, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    markers = np.zeros(mask.shape, np.int32)
    marker_id = 0
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] < min_core_area:
            continue
        marker_id += 1
        markers[components == component] = marker_id
    if not marker_id:
        return markers
    distance = cv2.distanceTransform(np.uint8(mask), cv2.DIST_L2, 5)
    scale = max(1, np.percentile(distance[mask], 95))
    elevation = 3.0 * boundary - distance / scale
    labels = watershed(elevation, markers, mask=mask).astype(np.int32)
    if merge_enclosed:
        labels = merge_enclosed_regions(labels)
    return labels


def save_instances(image: np.ndarray, labels: np.ndarray, out: Path,
                   min_area: int = 100) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    masks = []
    paths = []
    for group in np.unique(labels):
        if group == 0:
            continue
        mask = np.uint8(labels == group)
        if np.count_nonzero(mask) < min_area:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_area:
            continue
        full = np.zeros_like(mask)
        cv2.drawContours(full, [contour], -1, 1, cv2.FILLED)
        masks.append(full)
        points = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
        paths.append({"group": int(group), "area_px": int(full.sum()),
                      "points": points.tolist()})
        cv2.polylines(overlay, [points], True, (20, 25, 230), 2, cv2.LINE_AA)
        x, y = points.mean(axis=0).astype(int)
        cv2.putText(overlay, str(group), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (240, 30, 20), 1, cv2.LINE_AA)
    stack = np.stack(masks) if masks else np.zeros((0, *labels.shape), np.uint8)
    np.savez_compressed(out / "instances.npz", masks=stack,
                        ids=np.asarray([p["group"] for p in paths], np.int32))
    cv2.imwrite(str(out / "overlay.png"), overlay)
    (out / "paths.json").write_text(json.dumps(paths) + "\n")
    return {"instances": len(paths), "overlap_px": int(np.count_nonzero(stack.sum(axis=0) > 1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--core-threshold", type=float, default=0.94)
    parser.add_argument("--core-boundary-max", type=float, default=0.2)
    parser.add_argument("--min-core-area", type=int, default=500)
    parser.add_argument("--erosion-px", type=int, default=9)
    parser.add_argument("--merge-enclosed", action="store_true",
                        help="Merge small contacting islands enclosed by a larger cut silhouette")
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    model = Path(args.model).resolve()
    tile = json.loads((model.parent / "manifest.json").read_text())["size"]
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    foreground, boundary = run_tiled(image, session, tile)
    labels = split(foreground, boundary, threshold=args.threshold,
                   core_threshold=args.core_threshold,
                   core_boundary_max=args.core_boundary_max,
                   min_core_area=args.min_core_area, erosion_px=args.erosion_px,
                   merge_enclosed=args.merge_enclosed)
    out = Path(args.out).resolve()
    result = save_instances(image, labels, out)
    report = {**result, "image": str(Path(args.image).resolve()),
              "model": str(model), "threshold": args.threshold,
              "core_threshold": args.core_threshold,
              "core_boundary_max": args.core_boundary_max,
              "min_core_area": args.min_core_area, "erosion_px": args.erosion_px,
              "merge_enclosed": args.merge_enclosed}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out), **report}))


if __name__ == "__main__":
    main()
