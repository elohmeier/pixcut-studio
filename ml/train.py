#!/usr/bin/env python3
"""Train a small foreground and instance-boundary model on reviewed sticker sheets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from model import StickerBoundaryNet
from manual_data import DEFAULT_ANNOTATIONS, load_manual_data
from sticker_data import load_manifest, load_sheet_data, make_real_tile, make_scene
from public_data import load_public_assets


def batch(assets, size: int, seeds: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    scenes = [make_scene(assets, size, seed) for seed in seeds]
    images = torch.from_numpy(np.stack([scene[0] for scene in scenes]))
    targets = torch.from_numpy(np.stack([scene[1] for scene in scenes]))
    return images, targets


def mixed_batch(assets, sheets, manual_sheets, size: int, seeds: list[int],
                real_fraction: float, manual_fraction: float,
                public_assets=None, public_fraction: float = 0):
    manual_count = min(len(seeds), round(len(seeds) * manual_fraction))
    real_count = min(len(seeds) - manual_count, round(len(seeds) * real_fraction))
    public_count = min(len(seeds) - manual_count - real_count,
                       round(len(seeds) * public_fraction))
    scenes = [make_real_tile(manual_sheets, size, seed) for seed in seeds[:manual_count]]
    scenes += [make_real_tile(sheets, size, seed)
               for seed in seeds[manual_count:manual_count + real_count]]
    public_start = manual_count + real_count
    scenes += [make_scene(public_assets, size, seed)
               for seed in seeds[public_start:public_start + public_count]]
    scenes += [make_scene(assets, size, seed) for seed in seeds[public_start + public_count:]]
    return (torch.from_numpy(np.stack([scene[0] for scene in scenes])),
            torch.from_numpy(np.stack([scene[1] for scene in scenes])))


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    numerator = 2 * (probability * target).sum((2, 3)) + 1
    denominator = probability.sum((2, 3)) + target.sum((2, 3)) + 1
    return (1 - numerator / denominator).mean()


def loss_of(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    foreground = F.binary_cross_entropy_with_logits(logits[:, 0], target[:, 0])
    foreground += 0.5 * dice_loss(logits[:, :1], target[:, :1])
    boundary = F.binary_cross_entropy_with_logits(logits[:, 1], target[:, 1],
                                                    pos_weight=torch.tensor(5.0))
    boundary += dice_loss(logits[:, 1:2], target[:, 1:2])
    return foreground + boundary


def foreground_loss_of(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    foreground = F.binary_cross_entropy_with_logits(logits[:, 0], target[:, 0])
    return foreground + 0.5 * dice_loss(logits[:, :1], target[:, :1])


def manual_training_digest(path: str) -> str:
    """Hash only training labels so later review annotations do not change provenance."""
    config = json.loads(Path(path).read_text())
    selected = {
        "image": config["image"],
        "image_size": config["image_size"],
        "annotations": [entry for entry in config["annotations"] if entry["split"] == "train"],
    }
    return hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@torch.inference_mode()
def evaluate(model: StickerBoundaryNet, images: torch.Tensor, targets: torch.Tensor) -> dict:
    model.eval()
    outputs = torch.cat([model(chunk) for chunk in images.split(4)])
    loss = float(loss_of(outputs, targets))
    pred_fg = outputs[:, 0].sigmoid() >= 0.5
    real_fg = targets[:, 0] >= 0.5
    intersection = (pred_fg & real_fg).sum().item()
    union = (pred_fg | real_fg).sum().item()
    scores = {}
    real_boundary = targets[:, 1] >= 0.5
    probabilities = outputs[:, 1].sigmoid()
    for threshold in (0.2, 0.35, 0.5):
        prediction = probabilities >= threshold
        true_positive = (prediction & real_boundary).sum().item()
        precision = true_positive / max(1, prediction.sum().item())
        recall = true_positive / max(1, real_boundary.sum().item())
        scores[str(threshold)] = round(2 * precision * recall / max(1e-8, precision + recall), 4)
    return {"loss": round(loss, 4), "foreground_iou": round(intersection / max(1, union), 4),
            "boundary_f1": scores}, outputs


def preview(image: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor, out: Path) -> None:
    rgb = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ground_truth = rgb.copy()
    ground_truth[target[1].numpy() > 0.5] = [255, 0, 0]
    boundary = (prediction[1].sigmoid().numpy() * 255).astype(np.uint8)
    foreground = (prediction[0].sigmoid().numpy() * 255).astype(np.uint8)
    panel = np.concatenate((rgb, ground_truth,
                            np.stack((boundary, np.zeros_like(boundary), np.zeros_like(boundary)), -1),
                            np.repeat(foreground[..., None], 3, -1)), axis=1)
    Image.fromarray(panel).save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(Path(__file__).parent / "data" / "sheets.json"),
                        help="Reviewed sheet-level training and validation groups")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", help="Optional model.pt from a previous run")
    parser.add_argument("--real-fraction", type=float, default=0,
                        help="Fraction of training tiles cropped from real source sheets")
    parser.add_argument("--manual-fraction", type=float, default=0,
                        help="Fraction of training tiles cropped from hand-corrected stickers")
    parser.add_argument("--manual-labels", default=str(DEFAULT_ANNOTATIONS),
                        help="Reviewed sticker contour annotations")
    parser.add_argument("--public-images", help="External foreground image directory")
    parser.add_argument("--public-masks", help="Matching binary mask directory")
    parser.add_argument("--public-limit", type=int, default=240)
    parser.add_argument("--public-source", default="DIS5K",
                        help="Dataset name recorded for each external asset")
    parser.add_argument("--public-url", default="https://github.com/xuebinqin/DIS",
                        help="Official source URL recorded in the model manifest")
    parser.add_argument("--public-alpha", action="store_true",
                        help="Composite grayscale alpha mattes with soft subject edges")
    parser.add_argument("--public-fraction", type=float, default=0,
                        help="Fraction of tiles synthesized from external image/mask pairs")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--selection", choices=("synthetic", "mixed", "mixed-public", "foreground"),
                        help="Checkpoint score; defaults to synthetic for all-synthetic training")
    parser.add_argument("--train-foreground-only", action="store_true",
                        help="Freeze the backbone and boundary output while tuning the foreground head")
    parser.add_argument("--train-head-only", action="store_true",
                        help="Freeze the backbone and tune both output channels")
    parser.add_argument("--freeze-batchnorm", action="store_true",
                        help="Keep pretrained batch-normalization statistics and parameters fixed")
    parser.add_argument("--distill-original", type=float, default=0,
                        help="Preserve resumed model probabilities on non-public training tiles")
    args = parser.parse_args()
    if args.steps < 1 or args.batch < 1 or args.size < 64 or args.size % 8:
        parser.error("steps/batch must be positive and size must be a multiple of 8, at least 64")
    if (not 0 <= args.real_fraction <= 1 or not 0 <= args.manual_fraction <= 1
            or not 0 <= args.public_fraction <= 1
            or args.real_fraction + args.manual_fraction + args.public_fraction > 1 or args.lr <= 0):
        parser.error("Training fractions must sum to at most one; lr must be positive")
    if args.public_fraction and (not args.public_images or not args.public_masks):
        parser.error("--public-fraction requires --public-images and --public-masks")
    if bool(args.public_images) != bool(args.public_masks):
        parser.error("Provide both --public-images and --public-masks")
    if args.train_head_only and args.train_foreground_only:
        parser.error("Choose only one head fine-tuning mode")
    if args.freeze_batchnorm and not args.resume:
        parser.error("--freeze-batchnorm requires --resume")
    if args.distill_original < 0 or args.distill_original and not args.resume:
        parser.error("--distill-original must be nonnegative and requires --resume")
    selection = args.selection or ("foreground" if args.train_foreground_only else
                                   "mixed" if args.real_fraction or args.manual_fraction or args.public_fraction
                                   else "synthetic")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train_assets, holdout_assets, data_config = load_manifest(args.manifest)
    train_sheets = load_sheet_data(args.manifest, "train") if args.real_fraction else []
    manual_sheets = load_manual_data(args.manual_labels, "train") if args.manual_fraction else []
    public_assets, public_holdout, public_names = (load_public_assets(
        args.public_images, args.public_masks, limit=args.public_limit, seed=args.seed,
        source=args.public_source, alpha_matte=args.public_alpha)
        if args.public_images else ([], [], []))
    validation_sheets = load_sheet_data(args.manifest, "validation")
    print(json.dumps({"train_assets": len(train_assets), "validation_assets": len(holdout_assets),
                      "train_sheets": sorted({asset.sheet for asset in train_assets}),
                      "validation_sheets": sorted({asset.sheet for asset in holdout_assets}),
                      "manual_train_stickers": [item.name for item in manual_sheets],
                      "public_train_assets": len(public_assets),
                      "public_holdout_assets": len(public_holdout)}), flush=True)
    validation, validation_target = batch(holdout_assets, args.size, [args.seed + 1_000_000 + i for i in range(24)])
    real_scenes = [make_real_tile(validation_sheets, args.size, args.seed + 2_000_000 + i, False)
                   for i in range(32)]
    real_validation = torch.from_numpy(np.stack([scene[0] for scene in real_scenes]))
    real_validation_target = torch.from_numpy(np.stack([scene[1] for scene in real_scenes]))
    if public_holdout:
        public_validation, public_validation_target = batch(
            public_holdout, args.size, [args.seed + 3_000_000 + i for i in range(24)])
    model = StickerBoundaryNet()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True))
    teacher = copy.deepcopy(model).eval() if args.distill_original else None
    if teacher is not None:
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    initial_validation = None
    if args.resume:
        initial_validation = {
            "synthetic": evaluate(model, validation, validation_target)[0],
            "real": evaluate(model, real_validation, real_validation_target)[0],
            "public": (evaluate(model, public_validation, public_validation_target)[0]
                       if public_holdout else None),
        }
        print(json.dumps({"step": 0, "initial_validation": initial_validation}), flush=True)
    if args.train_foreground_only:
        if not args.resume:
            parser.error("--train-foreground-only requires --resume")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("output."))
        model.output.weight.register_hook(
            lambda grad: torch.cat((grad[:1], torch.zeros_like(grad[1:])), dim=0))
        model.output.bias.register_hook(
            lambda grad: torch.cat((grad[:1], torch.zeros_like(grad[1:])), dim=0))
        optimizer = torch.optim.AdamW(model.output.parameters(), lr=args.lr, weight_decay=0)
    elif args.train_head_only:
        if not args.resume:
            parser.error("--train-head-only requires --resume")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("output."))
        optimizer = torch.optim.AdamW(model.output.parameters(), lr=args.lr, weight_decay=0)
    else:
        if args.freeze_batchnorm:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    best_score = -1.0
    best_step = None
    best_metrics = None
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.eval() if args.train_foreground_only or args.train_head_only else model.train()
        if args.freeze_batchnorm:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        images, targets = mixed_batch(train_assets, train_sheets, manual_sheets, args.size,
                                      [args.seed + step * args.batch + i for i in range(args.batch)],
                                      args.real_fraction, args.manual_fraction,
                                      public_assets, args.public_fraction)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = (foreground_loss_of(logits, targets) if args.train_foreground_only
                else loss_of(logits, targets))
        distillation = None
        if teacher is not None:
            manual_count = min(args.batch, round(args.batch * args.manual_fraction))
            real_count = min(args.batch - manual_count, round(args.batch * args.real_fraction))
            public_count = min(args.batch - manual_count - real_count,
                               round(args.batch * args.public_fraction))
            original = torch.ones(args.batch, dtype=torch.bool)
            original[manual_count + real_count:manual_count + real_count + public_count] = False
            if original.any():
                with torch.no_grad():
                    teacher_prob = teacher(images[original]).sigmoid()
                distillation = F.mse_loss(logits[original].sigmoid(), teacher_prob)
                loss = loss + args.distill_original * distillation
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            metrics, predictions = evaluate(model, validation, validation_target)
            real_metrics, real_predictions = evaluate(model, real_validation, real_validation_target)
            public_metrics = (evaluate(model, public_validation, public_validation_target)[0]
                              if public_holdout else None)
            synthetic_score = metrics["foreground_iou"] + max(metrics["boundary_f1"].values())
            real_score = real_metrics["foreground_iou"] + max(real_metrics["boundary_f1"].values())
            public_score = (public_metrics["foreground_iou"] +
                            max(public_metrics["boundary_f1"].values())
                            if public_metrics else 0)
            score = (synthetic_score if selection == "synthetic" else
                     0.3 * metrics["foreground_iou"] + 0.7 * real_metrics["foreground_iou"]
                     if selection == "foreground" else
                     0.25 * synthetic_score + 0.5 * real_score + 0.25 * public_score
                     if selection == "mixed-public" else
                     0.3 * synthetic_score + 0.7 * real_score)
            print(json.dumps({"step": step, "train_loss": round(float(loss.detach()), 4),
                              "distillation_loss": (round(float(distillation.detach()), 5)
                                                    if distillation is not None else None),
                              "synthetic_val": metrics, "real_val": real_metrics,
                              "public_val": public_metrics,
                              "seconds": round(time.perf_counter() - started, 1)}), flush=True)
            if score >= best_score:
                best_score = score
                best_step = step
                best_metrics = {"synthetic": metrics, "real": real_metrics,
                                "public": public_metrics}
                torch.save(model.state_dict(), out / "model.pt")
                preview(validation[0], validation_target[0], predictions[0], out / "validation.png")
                preview(real_validation[0], real_validation_target[0], real_predictions[0],
                        out / "validation-real.png")

    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    sample = validation[:1]
    onnx_path = out / "model.onnx"
    torch.onnx.export(model, sample, onnx_path, input_names=["image"],
                      output_names=["logits"], opset_version=18, dynamo=True)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(sample).numpy()
    actual = session.run(["logits"], {"image": sample.numpy()})[0]
    difference = float(np.max(np.abs(actual - expected)))
    if difference > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {difference}")
    manifest = {
        "model": "StickerBoundaryNet", "input": "RGB float32 NCHW in [0,1]",
        "output": "2 logits NCHW: foreground and visible instance boundary",
        "size": args.size, "steps": args.steps, "batch": args.batch, "seed": args.seed,
        "real_fraction": args.real_fraction, "manual_fraction": args.manual_fraction,
        "public_fraction": args.public_fraction,
        "public_images": args.public_images, "public_masks": args.public_masks,
        "public_source": (args.public_source if public_assets else None),
        "public_url": (args.public_url if public_assets else None),
        "public_alpha": args.public_alpha,
        "public_train_ids": public_names[len(public_holdout):],
        "public_validation_ids": public_names[:len(public_holdout)],
        "manual_train_stickers": [item.name for item in manual_sheets],
        "manual_train_sha256": (manual_training_digest(args.manual_labels)
                                if manual_sheets else None),
        "learning_rate": args.lr,
        "checkpoint_selection": selection,
        "best_step": best_step,
        "train_foreground_only": args.train_foreground_only,
        "train_head_only": args.train_head_only,
        "freeze_batchnorm": args.freeze_batchnorm,
        "distill_original": args.distill_original,
        "train_groups": [f"{asset.sheet}:{asset.number}" for asset in train_assets],
        "validation_groups": [f"{asset.sheet}:{asset.number}" for asset in holdout_assets],
        "source_sha256": {sheet["name"]: hashlib.sha256((Path(args.manifest).resolve().parent /
                           "sheets" / f'{sheet["name"]}.jpeg').read_bytes()).hexdigest()
                           for sheet in data_config["sheets"]},
        "label_source": ("Reviewed pseudo-labels, hand-corrected outer contours, and synthetic white-rim external objects"
                         if args.public_fraction else
                         "Reviewed pseudo-labels plus hand-corrected outer sticker contours"
                         if manual_sheets else
                         "Reviewed pseudo-label cut polygons, expanded toward outer sticker border"),
        "best_validation": best_metrics,
        "initial_validation": initial_validation,
        "onnx_max_absolute_error": difference,
        "onnx_bytes": onnx_path.stat().st_size,
        "training_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(out), "best_validation": best_metrics,
                      "onnx_bytes": manifest["onnx_bytes"], "onnx_error": difference}), flush=True)


if __name__ == "__main__":
    main()
