#!/usr/bin/env python3
"""Run the trained ONNX boundary model on an image and save review maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


def positions(length: int, tile: int) -> list[int]:
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, tile // 2))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def run_tiled(image: np.ndarray, session: ort.InferenceSession, tile: int) -> np.ndarray:
    height, width = image.shape[:2]
    padded = np.full((max(height, tile), max(width, tile), 3), 255, np.uint8)
    padded[:height, :width] = image
    height2, width2 = padded.shape[:2]
    outputs = np.zeros((2, height2, width2), np.float32)
    weights = np.zeros((height2, width2), np.float32)
    taper = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 0.05
    for y in positions(height2, tile):
        for x in positions(width2, tile):
            patch = padded[y:y + tile, x:x + tile].astype(np.float32).transpose(2, 0, 1)[None] / 255
            logits = session.run(["logits"], {"image": patch})[0][0]
            probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
            outputs[:, y:y + tile, x:x + tile] += probability * taper
            weights[y:y + tile, x:x + tile] += taper
    return outputs[:, :height, :width] / weights[:height, :width]


def component_map(analysis_path: str, shape: tuple[int, int]) -> np.ndarray:
    analysis = json.loads(Path(analysis_path).read_text())
    summary = json.loads((Path(analysis_path).parent / "summary.json").read_text())
    spec = summary["raster"]
    labels = np.zeros((spec["height"] * spec["width"],), np.int32)
    for component in analysis["components"]:
        for start, length in component["runs"]:
            labels[start:start + length] = component["id"]
    labels = labels.reshape((spec["height"], spec["width"]))
    height, width = shape
    xx = np.clip(np.rint((np.arange(width) + 0.5) * spec["factor"] - 0.5).astype(int),
                 0, spec["contentWidth"] - 1) + spec["pad"]
    yy = np.clip(np.rint((np.arange(height) + 0.5) * spec["factor"] - 0.5).astype(int),
                 0, spec["contentHeight"] - 1) + spec["pad"]
    return labels[np.ix_(yy, xx)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True, help="model.onnx")
    parser.add_argument("--out", required=True)
    parser.add_argument("--analysis", help="Optional analysis.json to compare classical components")
    parser.add_argument("--foreground-threshold", type=float,
                        help="Defaults to the model's calibrated threshold or 0.5")
    parser.add_argument("--boundary-threshold", type=float,
                        help="Defaults to the model's calibrated threshold or 0.5")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    manifest = json.loads((model_path.parent / "manifest.json").read_text())
    calibration_path = model_path.parent / "calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    fg_threshold = (args.foreground_threshold if args.foreground_threshold is not None
                    else calibration.get("foreground_threshold", 0.5))
    boundary_threshold = (args.boundary_threshold if args.boundary_threshold is not None
                          else calibration.get("boundary_threshold", 0.5))
    if not 0 < fg_threshold < 1 or not 0 < boundary_threshold < 1:
        parser.error("thresholds must be between zero and one")
    tile = manifest["size"]
    image = np.asarray(Image.open(args.image).convert("RGB"))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    start = time.perf_counter()
    foreground, boundary = run_tiled(image, session, tile)
    elapsed = time.perf_counter() - start
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray((foreground * 255).astype(np.uint8)).save(out / "foreground.png")
    Image.fromarray((boundary * 255).astype(np.uint8)).save(out / "boundary.png")
    overlay = image.astype(np.float32).copy()
    strength = np.clip((boundary - 0.12) / 0.55, 0, 0.85)[..., None]
    overlay = overlay * (1 - strength) + np.array([255, 35, 35]) * strength
    Image.fromarray(np.uint8(np.clip(overlay, 0, 255))).save(out / "boundary-overlay.png")
    report = {"seconds": round(elapsed, 3), "tile": tile,
              "foreground_threshold": fg_threshold, "boundary_threshold": boundary_threshold,
              "foreground_fraction": round(float((foreground > fg_threshold).mean()), 4),
              "boundary_fraction": round(float((boundary > boundary_threshold).mean()), 4)}
    if args.analysis:
        labels = component_map(args.analysis, image.shape[:2])
        classical = (labels > 0).astype(np.uint8)
        boundary_binary = (boundary > boundary_threshold).astype(np.uint8)
        split = classical & (1 - boundary_binary)
        before = cv2.connectedComponents(classical, connectivity=8)[0] - 1
        after = cv2.connectedComponents(split, connectivity=8)[0] - 1
        report.update({"classical_components": before, "components_after_boundary": after})
        Image.fromarray(split * 255).save(out / "classical-after-boundary.png")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out), **report}))


if __name__ == "__main__":
    main()
