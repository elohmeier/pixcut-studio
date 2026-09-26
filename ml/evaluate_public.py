#!/usr/bin/env python3
"""Evaluate a sticker model on a held-out synthetic external-source subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import StickerBoundaryNet
from public_data import load_public_assets
from train import batch, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="model.pt, repeatable")
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--alpha", action="store_true")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    _, holdout, names = load_public_assets(args.images, args.masks,
                                           limit=args.limit, seed=args.seed,
                                           source=args.source, alpha_matte=args.alpha)
    images, targets = batch(holdout, 256,
                            [args.seed + 3_000_000 + i for i in range(24)])
    torch.set_num_threads(8)
    report = {"source": args.source, "source_ids": names[:len(holdout)],
              "seed": args.seed, "models": {}}
    for filename in args.model:
        path = Path(filename)
        model = StickerBoundaryNet()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        result, _ = evaluate(model, images, targets)
        report["models"][path.parent.name] = result
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
