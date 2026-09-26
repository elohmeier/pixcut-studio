"""Hand-traced sticker outlines, independent of the classical analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sticker_data import Asset, SheetData, target_from_labels


DEFAULT_ANNOTATIONS = Path(__file__).parent / "data" / "manual" / "grandma-grid.json"


def load_manual_data(path: str | Path = DEFAULT_ANNOTATIONS,
                     split: str | None = None) -> list[SheetData]:
    """Return one cropped image and hand-traced binary mask per sticker."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    image_path = path.parent.parent / "sheets" / f'{config["image"]}.jpeg'
    image = np.asarray(Image.open(image_path).convert("RGB"))
    if list(image.shape[1::-1]) != config["image_size"]:
        raise ValueError(f"Manual label image size differs from {image_path}")
    results = []
    for entry in config["annotations"]:
        if split is not None and entry["split"] != split:
            continue
        x0, y0, x1, y1 = entry["box"]
        crop = image[y0:y1, x0:x1].copy()
        points = np.asarray(entry["points"], np.int32)
        if len(points) < 3 or (points < 0).any() or (points[:, 0] >= x1 - x0).any() or (
                points[:, 1] >= y1 - y0).any():
            raise ValueError(f"Invalid hand contour: {entry['id']}")
        labels = np.zeros(crop.shape[:2], np.int16)
        cv2.fillPoly(labels, [points], 1)
        boundary_xy = np.argwhere(target_from_labels(labels)[1] > 0)
        results.append(SheetData(entry["id"], crop, labels, boundary_xy))
    return results


def manual_assets(path: str | Path = DEFAULT_ANNOTATIONS,
                  split: str = "train") -> list[Asset]:
    """Prepare hand-labeled crops for the synthetic scene generator."""
    return [
        Asset(i, Image.fromarray(item.image), Image.fromarray(np.uint8(item.labels > 0) * 255),
              "manual-grandma")
        for i, item in enumerate(load_manual_data(path, split), 1)
    ]


def render_overlay(path: str | Path, out: str | Path) -> None:
    """Draw all hand contours on the source sheet for visual quality review."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    image_path = path.parent.parent / "sheets" / f'{config["image"]}.jpeg'
    image = cv2.imread(str(image_path))
    for entry in config["annotations"]:
        x0, y0, _, _ = entry["box"]
        points = np.asarray(entry["points"], np.int32) + np.array([x0, y0])
        color = (30, 210, 30) if entry["split"] == "train" else (20, 40, 230)
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
        cv2.putText(image, entry["id"], tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out), image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    render_overlay(args.annotations, args.out)
