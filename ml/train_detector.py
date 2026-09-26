#!/usr/bin/env python3
"""Fine-tune a compact class-agnostic sticker box detector on synthetic sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO
import ultralytics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Generated data.yaml")
    parser.add_argument("--pretrained", required=True, help="Pretrained YOLO detection checkpoint")
    parser.add_argument("--out", required=True, help="Training output directory")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.pretrained)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch,
                imgsz=args.size, device="cpu", workers=args.workers,
                seed=args.seed, deterministic=True, project=str(out.parent),
                name=out.name, exist_ok=True, plots=False, verbose=False,
                amp=False, mosaic=0, mixup=0, copy_paste=0,
                close_mosaic=0, patience=args.epochs)
    manifest = {"model": "YOLO26n detector", "ultralytics": ultralytics.__version__,
                "data": str(Path(args.data).resolve()),
                "pretrained": str(Path(args.pretrained).resolve()),
                "epochs": args.epochs, "batch": args.batch, "size": args.size,
                "workers": args.workers, "seed": args.seed,
                "best": str(out / "weights" / "best.pt")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
