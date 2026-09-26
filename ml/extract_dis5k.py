#!/usr/bin/env python3
"""Extract paired DIS5K training images/masks into a private local cache.

The official archive must be downloaded separately. Only DIS-TR is sampled;
the DIS validation/test partitions never enter this training experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Official DIS5K.zip")
    parser.add_argument("--out", required=True, help="Private cache directory outside the repo")
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.limit < 2:
        parser.error("--limit must be at least 2")
    archive = Path(args.archive).resolve()
    out = Path(args.out).resolve()
    repo = Path(__file__).resolve().parent.parent
    if out == repo or repo in out.parents:
        parser.error("Place the extracted dataset outside the repository")
    image_out, mask_out = out / "im", out / "gt"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        files = package.namelist()
        images = {Path(name).stem: name for name in files
                  if name.startswith("DIS5K/DIS-TR/im/") and Path(name).suffix.lower()
                  in {".jpg", ".jpeg", ".png"}}
        masks = {Path(name).stem: name for name in files
                 if name.startswith("DIS5K/DIS-TR/gt/") and Path(name).suffix.lower()
                 in {".jpg", ".jpeg", ".png"}}
        names = sorted(images.keys() & masks.keys())
        rng = np.random.default_rng(args.seed)
        rng.shuffle(names)
        selected = names[:args.limit]
        if len(selected) < 2:
            raise RuntimeError("No matching DIS-TR image/mask pairs")
        for name in selected:
            (image_out / Path(images[name]).name).write_bytes(package.read(images[name]))
            (mask_out / Path(masks[name]).name).write_bytes(package.read(masks[name]))
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    manifest = {"source": "https://github.com/xuebinqin/DIS",
                "download": "https://drive.google.com/file/d/1O1eIuXX1hlGsV7qx4eSkjH231q7G1by1/view",
                "partition": "DIS-TR", "archive_sha256": digest,
                "seed": args.seed, "selected_ids": selected}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(out), "pairs": len(selected), "archive_sha256": digest}))


if __name__ == "__main__":
    main()
