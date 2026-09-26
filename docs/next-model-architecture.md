# Cut-contour instance architecture

## First two-stage result, 2026-09-26

The box-plus-refinement experiment described below is now implemented in
`ml/detect_stickers.py`, `ml/train_prompt.py`, and `ml/infer_prompt.py`. The
trained YOLO26n detector located all 36 intended pastel stickers and all eight
portrait stickers with box IoU at least 0.5. It also proposed 32 and 18 extra
boxes respectively, mainly for decorations inside larger stickers. On a
synthetic validation set it reached mAP50 0.995, which hid this real-sheet
error. The prompted U-Net gave eight strong portrait outlines when supplied
with correct boxes, but the pastel contours leaked into adjacent stickers and
filled cut paths overlapped. The automatic pipeline is not ready for cutting.
Exact metrics and model hashes are in
`ml/evaluations/instance-pipeline-2026-09-26.json`.

The next experiment should train **whole sticker identity** explicitly. Add
sheet-level box labels in which attached hearts, rays, and highlights are
hard negatives, and retain separate labels for nearby stand-alone icons.
Generate synthetic composites with attached decorations and touching white
rims, then evaluate on sheets excluded by source image from training and
checkpoint selection. For the contour stage, train at 512–768 pixels on
corrected pastel outer-rim masks and touching pairs. Measure overlap after
filling the final vector paths, not only overlap between raw raster masks.
Keep the current V11 boundary model as an edge prior or controlled baseline;
the new crop model did not improve pastel contour accuracy. A larger generic
backbone is worth testing only after this identity-target and cut-path
evaluation are in place.

The current `StickerBoundaryNet` has 488,018 parameters. It sees 256×256
tiles and predicts only foreground and visible boundaries. A watershed step
then turns these into sticker instances. On the original pastel sheet, the
AM-2k fine-tune still makes 39 paths for 36 intended stickers and splits the
upper unicorn into separate body pieces. More subject masks did not resolve
that identity error.

## Two-stage design tested

Use a **whole-sheet instance proposal plus prompted crop refinement**:

1. Predict one center or box per intended sticker using a sheet overview plus
   overlapping higher-resolution views for tiny icons. Merge duplicate
   proposals across views. Train this stage on generated sticker sheets with
   exact instance IDs and the hand-reviewed full-sheet boxes. Include tiny
   icons, large composites,
   touching white rims, and decorations attached to the same sticker.
2. Crop each proposed sticker at source resolution with surrounding context.
   Predict one *outer print-border mask* for that proposal, using its box and
   center as input. A 512–768-pixel crop is the first useful resolution test.
   Preserve the white rim and attached decorations; the mask target is not the
   photographed subject from an external dataset.
3. Resolve competing masks jointly. Each pixel can belong to at most one
   sticker. Require one connected outer contour per accepted proposal, and
   flag uncertain seams for a user click or box correction.
4. Trace and simplify the outer path only after instance ownership is fixed.

This explicitly separates **how many stickers are present** from **where their
outer edges run**. It also gives each proposed sticker a larger view than the
current 256-pixel tile model. A 32-channel or 512-pixel version of the same
foreground/boundary U-Net is a useful controlled baseline, but it would still
leave watershed responsible for instance identity.

## Models to benchmark offline

| Candidate | Role | Reason to try | Limit to check |
| --- | --- | --- | --- |
| Small detector plus prompted crop U-Net | Instance count and high-resolution border masks | Small exportable stages; boxes can be corrected in the editor | Needs box labels and joint overlap arbitration |
| [Mask2Former](https://github.com/facebookresearch/Mask2Former) or similar query-mask model | Direct per-instance masks with whole-sheet context | Its architecture predicts instance masks rather than a single foreground map | Larger training/deployment cost; ONNX export must be verified |
| [SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) | Offline proposal/refinement benchmark using text or exemplar prompts | Predicts masks and identities for matching concepts | A semantic prompt may split composite stickers or miss icons; deployment cost is unknown here |
| [YOLO segmentation](https://docs.ultralytics.com/tasks/segment/) | Fast instance proposals | Official ONNX export path and compact model variants | Prototype masks need high-resolution edge refinement |

The already tested SAM 2 box baseline is an offline **prompted** result: its
boxes came from the reference labels, and its masks overlapped extensively.
It does not establish automatic discovery or cut-safe paths.

## Gate before browser integration

Freeze an untouched sheet-level test set after correcting more full-sheet
outer-border labels, especially the upper unicorn and touching decorations.
Compare each candidate with the no-public V11 control using intended versus
predicted sticker count, matching at IoU ≥ 0.1, per-instance IoU and contour
distance, false interior paths, mask overlap, and correction effort. Time and
memory should be measured on desktop and iOS Safari. ONNX Runtime Web supports
WASM and WebGPU, but actual model/operator support and Safari behavior need
testing on the target device.

Public datasets such as AM-2k and P3M can supply varied foregrounds to the
synthetic generator. They do not label the printed rim or decide whether a
unicorn, wand, and sparkle trail form one sticker. That decision must come
from sticker-specific examples and the instance target used for training.
