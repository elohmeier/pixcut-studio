#!/usr/bin/env python3
"""Export saved classical CLI groups as full-sheet instance masks for evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from split_boundary import save_instances
from sticker_data import labels_from_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    labels = labels_from_analysis(args.analysis, image.shape[:2])
    out = Path(args.out).resolve()
    result = save_instances(image, labels, out)
    report = {"method": "saved classical CLI cut groups", "image": str(Path(args.image).resolve()),
              "analysis": str(Path(args.analysis).resolve()), **result}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out), **report}))


if __name__ == "__main__":
    main()
