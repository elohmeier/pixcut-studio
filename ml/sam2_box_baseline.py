#!/usr/bin/env python3
"""SAM 2.1 oracle-box baseline for sticker contours (offline only).

Boxes come from the reference annotation file. This is an assisted upper bound,
not an automatic detector or a browser deployment candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


DEFAULT_CHECKPOINT = Path.home() / ".cache" / "pixcut-studio" / "sam2" / "sam2.1_hiera_tiny.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    config = json.loads(Path(args.annotations).read_text())
    root = Path(args.annotations).resolve().parent.parent / "sheets"
    image_path = root / f'{config["image"]}.jpeg'
    image = np.asarray(Image.open(image_path).convert("RGB")).copy()
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", args.checkpoint, device="cpu")
    predictor = SAM2ImagePredictor(model)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    masks = []
    rows = []
    with torch.inference_mode():
        predictor.set_image(image)
        embedding_seconds = time.perf_counter() - start
        for index, entry in enumerate(config["annotations"], 1):
            box = np.asarray(entry["box"], np.float32)
            candidates, scores, _ = predictor.predict(box=box, multimask_output=True)
            best = int(np.argmax(scores))
            mask = np.uint8(candidates[best])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                filled = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
                mask = filled
                cv2.polylines(overlay, [contour], True, (20, 25, 230), 2, cv2.LINE_AA)
            else:
                mask.fill(0)
            x, y = box[:2].astype(int)
            cv2.putText(overlay, str(index), (int(x), int(y + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 30, 20), 1, cv2.LINE_AA)
            masks.append(mask)
            rows.append({"id": entry["id"], "score": float(scores[best]),
                         "candidate": best, "area_px": int(mask.sum())})
    stack = np.stack(masks)
    np.savez_compressed(out / "instances.npz", masks=stack,
                        ids=np.asarray([row["id"] for row in rows]))
    cv2.imwrite(str(out / "overlay.png"), overlay)
    report = {"method": "SAM 2.1 Hiera tiny, highest-score mask from each supplied reference box",
              "image": str(image_path), "oracle_box_prompts": len(rows),
              "instances": len(rows), "embedding_seconds": round(embedding_seconds, 3),
              "total_seconds": round(time.perf_counter() - start, 3),
              "overlap_px": int(np.count_nonzero(stack.sum(axis=0) > 1)),
              "per_box": rows}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out), **{key: report[key] for key in
                                     ("oracle_box_prompts", "embedding_seconds",
                                      "total_seconds", "overlap_px")}}))


if __name__ == "__main__":
    main()
