#!/usr/bin/env python3
"""Train a box-conditioned model to return one sticker's outer-rim mask."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
from torch.nn import functional as F

from generate_detection_data import make_detection_scene
from manual_data import load_manual_data
from prompt_data import box_of, prompt_crop
from prompt_model import PromptedStickerNet
from sticker_data import load_manifest, make_instance_scene


def sample(assets, manual, size: int, seed: int, kind: int):
    rng = np.random.default_rng(seed)
    if kind == 0 and manual:
        item = manual[int(rng.integers(len(manual)))]
        image, labels = item.image, item.labels
        chosen = 1
    else:
        rgb, labels = (make_detection_scene(assets, 512, seed) if kind <= 4 else
                       make_instance_scene(assets, 512, seed,
                                           scale_range=(0.08, 0.62), count_range=(5, 9)))
        image = np.uint8(np.round(rgb.transpose(1, 2, 0) * 255))
        choices = [int(i) for i in np.unique(labels) if i > 0 and
                   np.count_nonzero(labels == i) >= 100]
        if not choices:
            return sample(assets, manual, size, seed + 1, kind)
        chosen = choices[int(rng.integers(len(choices)))]
    mask = labels == chosen
    return prompt_crop(image, mask, box_of(mask), size, seed + 10_000, augment=True)[:2]


def batch(assets, manual, size: int, seeds: list[int]):
    scenes = [sample(assets, manual, size, seed, index % 8)
              for index, seed in enumerate(seeds)]
    return (torch.from_numpy(np.stack([item[0] for item in scenes])),
            torch.from_numpy(np.stack([item[1] for item in scenes])))


def loss_of(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probability = logits.sigmoid()
    dice = 1 - ((2 * (probability * targets).sum((2, 3)) + 1) /
                (probability.sum((2, 3)) + targets.sum((2, 3)) + 1)).mean()
    return bce + dice


@torch.inference_mode()
def evaluate(model, images: torch.Tensor, targets: torch.Tensor) -> dict:
    model.eval()
    predictions = torch.cat([model(chunk) for chunk in images.split(4)])
    actual, expected = predictions.sigmoid() >= 0.5, targets >= 0.5
    intersection = (actual & expected).sum((1, 2, 3)).float()
    union = (actual | expected).sum((1, 2, 3)).float().clamp_min(1)
    return {"loss": round(float(loss_of(predictions, targets)), 4),
            "mean_iou": round(float((intersection / union).mean()), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(Path(__file__).parent / "data" / "sheets.json"))
    parser.add_argument("--base", required=True, help="Existing sticker-boundary model.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.size % 8 or args.size < 128 or args.batch < 1 or args.steps < 1:
        parser.error("size must be a multiple of 8, >=128; batch and steps must be positive")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train_assets, val_assets, _ = load_manifest(args.manifest)
    manual_train = load_manual_data(split="train")
    manual_review = load_manual_data(split="test")
    model = PromptedStickerNet()
    model.initialize_from_boundary_model(args.base)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    synth_images, synth_targets = batch(
        val_assets, [], args.size, [args.seed + 1_000_000 + i for i in range(24)])
    reviewed = [prompt_crop(item.image, item.labels > 0,
                            box_of(item.labels > 0), args.size)[:2]
                for item in manual_review]
    real_images = torch.from_numpy(np.stack([item[0] for item in reviewed]))
    real_targets = torch.from_numpy(np.stack([item[1] for item in reviewed]))
    initial = {"synthetic": evaluate(model, synth_images, synth_targets),
               "real_review": evaluate(model, real_images, real_targets)}
    print(json.dumps({"step": 0, **initial}), flush=True)
    best_score, best_step, best_metrics = -1.0, 0, None
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        images, targets = batch(train_assets, manual_train, args.size,
                                [args.seed + step * args.batch + i for i in range(args.batch)])
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_of(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            synthetic = evaluate(model, synth_images, synth_targets)
            real = evaluate(model, real_images, real_targets)
            score = 0.5 * synthetic["mean_iou"] + 0.5 * real["mean_iou"]
            print(json.dumps({"step": step, "train_loss": round(float(loss.detach()), 4),
                              "synthetic": synthetic, "real_review": real,
                              "seconds": round(time.perf_counter() - started, 1)}), flush=True)
            if score >= best_score:
                best_score, best_step = score, step
                best_metrics = {"synthetic": synthetic, "real_review": real}
                torch.save(model.state_dict(), out / "model.pt")
    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    sample_image = synth_images[:1]
    onnx_path = out / "model.onnx"
    torch.onnx.export(model, sample_image, onnx_path,
                      input_names=["image_and_prompt"], output_names=["logits"],
                      opset_version=18, dynamo=True)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(sample_image).numpy()
    actual = session.run(["logits"], {"image_and_prompt": sample_image.numpy()})[0]
    error = float(np.max(np.abs(expected - actual)))
    if error > 0.001:
        raise RuntimeError(f"ONNX parity error {error}")
    manifest = {"model": "PromptedStickerNet", "base": str(Path(args.base).resolve()),
                "base_sha256": hashlib.sha256(Path(args.base).read_bytes()).hexdigest(),
                "steps": args.steps, "batch": args.batch, "size": args.size,
                "seed": args.seed, "lr": args.lr, "best_step": best_step,
                "initial": initial, "best_validation": best_metrics,
                "onnx_max_absolute_error": error,
                "training_seconds": round(time.perf_counter() - started, 1),
                "input": "5 float32 NCHW channels: RGB [0,1], box fill, center Gaussian",
                "output": "one outer-rim mask logit"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(out), "best_step": best_step,
                      "best_validation": best_metrics, "onnx_error": error}), flush=True)


if __name__ == "__main__":
    main()
