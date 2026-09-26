#!/usr/bin/env python3
"""Fetch a reproducible AM-2k training subset into a private cache.

First list the official Google Drive folder with `gdown --json FOLDER_URL`.
Only train/original and train/mask files are used. The official validation
split is never used for training or checkpoint selection here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random

from PIL import Image


FOLDER_URL = "https://drive.google.com/drive/folders/1SReB9Zma0TDfDhow7P5kiZNMwY9j9xMA"


def select_pairs(index: list[dict], limit: int, seed: int) -> list[dict]:
    """Select source IDs, pairing image and matte before the random sample."""
    by_kind = {}
    for row in index:
        parts = Path(row["path"]).parts
        if len(parts) != 3 or parts[0] != "train" or parts[1] not in {"original", "mask"}:
            continue
        by_kind.setdefault(parts[1], {})[Path(parts[2]).stem] = row
    names = sorted(by_kind.get("original", {}).keys() & by_kind.get("mask", {}).keys())
    if len(names) < limit or limit < 2:
        raise ValueError(f"Need {limit} train image/matte pairs, found {len(names)}")
    random.Random(seed).shuffle(names)
    return [{"id": name, "image": by_kind["original"][name],
             "mask": by_kind["mask"][name]} for name in names[:limit]]


def fetch(row: dict, out: Path) -> None:
    import gdown

    if out.exists():
        try:
            with Image.open(out) as image:
                image.verify()
            return
        except OSError:
            out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(url=row["url"], output=str(out), quiet=True,
                            retries=3, timeout=60)
    if result is None:
        raise RuntimeError(f"Download failed: {row['path']}")
    with Image.open(out) as image:
        image.verify()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Official folder listing from gdown --json")
    parser.add_argument("--out", required=True, help="Private cache directory")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    chosen = select_pairs(json.loads(Path(args.index).read_text()), args.limit, args.seed)
    out = Path(args.out)
    jobs = []
    for pair in chosen:
        jobs.extend(((pair["image"], out / "im" / Path(pair["image"]["path"]).name),
                     (pair["mask"], out / "gt" / Path(pair["mask"]["path"]).name)))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, row, path): row["path"] for row, path in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            future.result()
            if number % 20 == 0 or number == len(jobs):
                print(f"Downloaded or verified {number}/{len(jobs)} files", flush=True)
    (out / "selection.json").write_text(json.dumps({
        "source": FOLDER_URL, "split": "train", "seed": args.seed,
        "pairs": chosen,
    }, indent=2) + "\n")
    print(f"AM-2k training pairs ready: {len(chosen)} in {out}", flush=True)


if __name__ == "__main__":
    main()
