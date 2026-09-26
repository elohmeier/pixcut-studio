"""Checks for sheet separation and seeded foreground growth."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from evaluate_manual import seeded_component
from fetch_am2k import select_pairs
from generate_detection_data import boxes_from_instances
from infer_prompt import arbitrate
from prompt_data import prompt_crop
from manual_data import load_manual_data
from propose import grow_groups
from public_data import load_public_assets
from split_boundary import merge_enclosed_regions, split
from sticker_data import load_manifest, load_sheet_data, make_real_tile
from trace_reference import trace


MANIFEST = Path(__file__).parent / "data" / "sheets.json"


class ModelDataTests(unittest.TestCase):
    def test_detection_boxes_follow_instance_masks(self):
        labels = np.zeros((100, 200), np.int16)
        labels[20:60, 50:130] = 1
        labels[70:75, 170:174] = 2
        self.assertEqual(boxes_from_instances(labels),
                         [(0.45, 0.4, 0.4, 0.4)])

    def test_prompt_crop_aligns_box_and_target(self):
        image = np.full((100, 100, 3), 255, np.uint8)
        image[30:70, 20:60] = [200, 0, 0]
        mask = np.zeros((100, 100), np.uint8)
        mask[30:70, 20:60] = 1
        features, target, _ = prompt_crop(image, mask, (20, 30, 60, 70), 128)
        self.assertEqual(features.shape, (5, 128, 128))
        self.assertEqual(target.shape, (1, 128, 128))
        self.assertTrue(np.all(features[3][target[0] > 0] == 1))
        self.assertGreater(features[4, 64, 64], 0.9)

    def test_prompt_mask_arbitration_is_exclusive(self):
        yy, xx = np.mgrid[:80, :100]
        probability = np.stack((np.float32((xx - 35) ** 2 + (yy - 40) ** 2 < 30 ** 2),
                                np.float32((xx - 65) ** 2 + (yy - 40) ** 2 < 30 ** 2)))
        boxes = [{"box": [5, 10, 65, 70]}, {"box": [35, 10, 95, 70]}]
        masks = arbitrate(probability, boxes)
        self.assertTrue(masks[0, 40, 30])
        self.assertTrue(masks[1, 40, 70])
        self.assertEqual(int((masks[0] & masks[1]).sum()), 0)

    def test_am2k_selector_pairs_only_training_sources(self):
        rows = [{"path": f"train/{kind}/item_{index}.{extension}", "url": str(index)}
                for kind, extension in (("original", "jpg"), ("mask", "png"))
                for index in range(5)]
        rows += [{"path": "validation/original/item_9.jpg", "url": "9"},
                 {"path": "validation/mask/item_9.png", "url": "9"}]
        chosen = select_pairs(rows, 3, 2026)
        self.assertEqual(len(chosen), 3)
        self.assertEqual(len({entry["id"] for entry in chosen}), 3)
        self.assertTrue(all(entry["image"]["path"].startswith("train/original/")
                            and entry["mask"]["path"].startswith("train/mask/")
                            for entry in chosen))

    def test_public_masks_make_white_cut_rims_without_source_background(self):
        with TemporaryDirectory() as directory:
            images, masks = Path(directory) / "im", Path(directory) / "gt"
            images.mkdir()
            masks.mkdir()
            for number in range(4):
                source = np.full((64, 64, 3), [0, 200, 0], np.uint8)
                source[16:48, 16:48] = [200, 0, 0]
                truth = np.zeros((64, 64), np.uint8)
                truth[16:48, 16:48] = 255
                Image.fromarray(source).save(images / f"{number}.png")
                Image.fromarray(truth).save(masks / f"{number}.png")
            training, validation, names = load_public_assets(
                str(images), str(masks), limit=4, seed=4, side=64)
            self.assertEqual((len(training), len(validation), len(names)), (3, 1, 4))
            for asset in training + validation:
                image = np.asarray(asset.image)
                mask = np.asarray(asset.mask)
                self.assertTrue(np.all(image[mask == 0] == 255))
                self.assertTrue(np.any(np.all(image[mask > 0] == 255, axis=1)))
                self.assertFalse(np.any(np.all(image == [0, 200, 0], axis=2)))

    def test_alpha_matte_keeps_soft_edges_and_source_name(self):
        with TemporaryDirectory() as directory:
            images, masks = Path(directory) / "im", Path(directory) / "gt"
            images.mkdir()
            masks.mkdir()
            for number in range(2):
                source = np.full((64, 64, 3), [200, 0, 0], np.uint8)
                alpha = np.zeros((64, 64), np.uint8)
                alpha[16:48, 16:48] = 255
                alpha[15, 16:48] = 128
                Image.fromarray(source).save(images / f"{number}.png")
                Image.fromarray(alpha).save(masks / f"{number}.png")
            training, holdout, _ = load_public_assets(
                str(images), str(masks), limit=2, side=64,
                source="AM-2k", alpha_matte=True)
            asset = (training + holdout)[0]
            image = np.asarray(asset.image)
            self.assertEqual(asset.sheet, "AM-2k")
            self.assertTrue(np.any(np.all(image == [200, 0, 0], axis=2)))
            self.assertTrue(np.any((image[..., 0] > 200) & (image[..., 0] < 255)))

    def test_sheet_level_holdout_and_aligned_real_tile(self):
        train, validation, _ = load_manifest(str(MANIFEST))
        self.assertFalse({asset.sheet for asset in train} &
                         {asset.sheet for asset in validation})
        tile, target = make_real_tile(load_sheet_data(str(MANIFEST), "validation"),
                                      256, 2_002_026, augment=False)
        self.assertEqual(tile.shape, (3, 256, 256))
        self.assertEqual(target.shape, (2, 256, 256))
        self.assertGreater(target[1].sum(), 0)
        self.assertTrue(np.all(target[1] <= target[0]))

    def test_touching_foreground_retains_two_seeded_groups(self):
        seeds = np.zeros((64, 64), np.int16)
        seeds[20:40, 8:20] = 1
        seeds[20:40, 44:56] = 2
        foreground = np.zeros((64, 64), np.float32)
        foreground[18:42, 6:58] = 1
        proposal, paths = grow_groups(seeds, foreground, 0.5, close_px=1)
        self.assertEqual({entry["group"] for entry in paths}, {1, 2})
        self.assertTrue(np.all(proposal[seeds == 1] == 1))
        self.assertTrue(np.all(proposal[seeds == 2] == 2))
        self.assertEqual(int(proposal[30, 30]), 1)
        self.assertEqual(int(proposal[30, 34]), 2)

    def test_hand_corrected_contours_have_separate_train_and_review_instances(self):
        training = load_manual_data(split="train")
        review = load_manual_data(split="test")
        confirmation = load_manual_data(split="confirm")
        self.assertEqual(len(training), 8)
        self.assertEqual(len(review), 4)
        self.assertEqual(len(confirmation), 4)
        names = [{item.name for item in split} for split in (training, review, confirmation)]
        self.assertFalse(names[0] & names[1] or names[0] & names[2] or names[1] & names[2])
        for item in training + review + confirmation:
            self.assertGreater(item.labels.sum(), 10_000)
            self.assertLess(item.labels.mean(), 0.9)
            self.assertGreater(len(item.boundary_xy), 100)

    def test_manual_review_ignores_neighboring_sticker_component(self):
        truth = np.zeros((32, 32), bool)
        truth[5:25, 5:20] = True
        prediction = truth.copy()
        prediction[8:20, 25:30] = True
        selected = seeded_component(prediction, truth)
        self.assertTrue(np.array_equal(selected, truth))

    def test_boundary_watershed_splits_touching_high_confidence_cores(self):
        foreground = np.zeros((80, 120), np.float32)
        foreground[15:65, 8:52] = 0.99
        foreground[15:65, 68:112] = 0.99
        foreground[35:45, 52:68] = 0.65
        boundary = np.zeros_like(foreground)
        boundary[35:45, 58:62] = 0.9
        labels = split(foreground, boundary, min_core_area=100,
                       erosion_px=5)
        self.assertEqual(len(set(np.unique(labels)) - {0}), 2)
        self.assertNotEqual(labels[40, 30], labels[40, 90])
        self.assertGreater(labels[40, 59], 0)

    def test_enclosed_island_merges_but_separate_sticker_remains(self):
        labels = np.zeros((80, 80), np.int32)
        labels[10:60, 10:60] = 1
        labels[25:34, 25:34] = 2
        labels[65:75, 10:20] = 3
        result = merge_enclosed_regions(labels)
        self.assertEqual(result[29, 29], 1)
        self.assertEqual(result[69, 15], 3)

    def test_image_reference_uses_selected_sticker_not_largest_neighbor(self):
        image = np.full((80, 120, 3), 255, np.uint8)
        image[5:70, 5:65] = 70
        image[20:50, 80:110] = 70
        points, diagnostics = trace(image, {"id": "small", "box": [0, 0, 120, 80],
                                            "seed": [95, 35]},
                                    {"polarity": "dark", "threshold": 200,
                                     "close_px": 1})
        self.assertTrue(diagnostics["seed_enclosed"])
        self.assertGreater(points[:, 0].min(), 70)


if __name__ == "__main__":
    unittest.main()
