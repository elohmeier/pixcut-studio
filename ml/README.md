# Cut contour model experiment

This is a reproducible **experiment**, not the app's active cut detector. The
browser still uses the classical analyzer. `StickerBoundaryNet` is a small
two-output U-Net that predicts sticker foreground and visible instance
boundaries from 256×256 RGB tiles. The latest balanced ONNX model and its
external weights file are in `models/sticker-boundary-v6/`.

## Data

The eight user-provided JPEGs are kept locally in `data/sheets/` and ignored by
Git. To reproduce the experiments, place `baby-bear.jpeg`,
`fairy-unicorns.jpeg`, `grandma-grid.jpeg`, `grandma-portraits.jpeg`,
`gym-couple.jpeg`, `pastel-fairy-original.jpeg`, `pirate-girl.jpeg`, and
`pirate-icons.jpeg` there. The saved CLI analyses live in `data/analyses/`.
`data/sheets.json` records the original seven reviewed sheets and their
groups, sheet-level split, and a per-sheet mask expansion toward the white
sticker border. The analysis polygons are **pseudo-labels**: they often trace
the colored artwork inside the white border. The expansion is approximate.

Four sheets (gym couple, baby bear, fairy/unicorn, and pirate girl) contribute
64 training assets. The pirate icon sheet contributes 39 validation assets and
is never sampled in training. The grandmother grid has sixteen reviewed
outer-border polygons in `data/manual/grandma-grid.json`: eight stickers from
rows 1–2 are training examples, four from row 3 are a development review set,
and four from row 6 are a later confirmation set. These
were seeded from the visible shadow edge in individually chosen crops,
visually checked, and locally corrected. They are more reliable than the CLI
hair outlines, but still approximate the printed edge. The grandmother
portrait sheet and original pastel fairy sheet now have full-sheet instance
reference drafts under `data/manual/`. They are held out from model training.
Their contours were traced from source-image pixels in manually selected
regions and inspected on overlays. Ambiguous touching margins and crop edges
are flagged; these are **development references**, not a final gold test.

Training builds randomized collages from the reviewed assets on white and
pastel backgrounds. It balances source sheets when sampling assets and creates
foreground and boundary targets from the visible instance masks. This
simulation does not reproduce every real overlap, print shadow, or pale
decoration.

## Reproduce

Python 3.13 and the following CPU setup were used:

```sh
uv venv --python 3.13 .venv-ml
uv pip install --python .venv-ml/bin/python torch==2.14.0 --index https://download.pytorch.org/whl/cpu
uv pip install --python .venv-ml/bin/python -r ml/requirements.txt
.venv-ml/bin/python ml/train.py --out ml/models/sticker-boundary-v2 --steps 500 --batch 8 --threads 8
.venv-ml/bin/python ml/train.py --out ml/models/sticker-boundary-v3 --resume ml/models/sticker-boundary-v2/model.pt --steps 350 --batch 8 --threads 8 --real-fraction 0.75 --lr 0.0003
.venv-ml/bin/python ml/manual_data.py --out /tmp/pixcut-grandma-annotations.png
.venv-ml/bin/python ml/train.py --out ml/models/sticker-boundary-v6 --resume ml/models/sticker-boundary-v3/model.pt --steps 250 --batch 8 --threads 8 --real-fraction 0.25 --manual-fraction 0.5 --lr 0.0001 --train-foreground-only
.venv-ml/bin/python ml/evaluate_manual.py --model ml/models/sticker-boundary-v3/model.onnx --model ml/models/sticker-boundary-v6/model.onnx --out /tmp/pixcut-manual-comparison
.venv-ml/bin/python ml/evaluate_manual.py --model ml/models/sticker-boundary-v3/model.onnx --model ml/models/sticker-boundary-v6/model.onnx --split confirm --out /tmp/pixcut-manual-confirmation
.venv-ml/bin/python ml/evaluate.py --model ml/models/sticker-boundary-v6/model.onnx --out /tmp/pixcut-v6-evaluation
.venv-ml/bin/python ml/infer.py --image ml/data/sheets/grandma-grid.jpeg --model ml/models/sticker-boundary-v6/model.onnx --out /tmp/pixcut-ml-grandma
.venv-ml/bin/python ml/propose.py --sheet pirate-icons --model ml/models/sticker-boundary-v6/model.onnx --out /tmp/pixcut-ml-proposals
.venv-ml/bin/python ml/propose.py --sheet grandma-grid --close-px 15 --model ml/models/sticker-boundary-v6/model.onnx --out /tmp/pixcut-ml-grandma-proposals
.venv-ml/bin/python -m unittest discover -s ml -p 'test_*.py'
```

The saved analyses can be regenerated with `npm run analyze:cuts --
ml/data/sheets/NAME.jpeg --fit --out DIR`. Review the new outlines before
training if the analyzer changes; group numbers and masks may change.

`train.py` selects the best validation checkpoint, exports ONNX, and verifies
PyTorch versus ONNX Runtime output within 1e-3. The exported model uses fixed
256×256 RGB float32 NCHW input in [0,1] and returns two logit channels.
`infer.py` tiles full images at half-tile stride and saves probability maps
and an overlay. It does not turn those maps into print-ready cut paths.

## Results and limits

Version 2 learned only from synthetic collages. Version 3 starts from those
weights and fine-tunes on 75% crops of the actual training sheets and 25%
synthetic collages. Version 6 starts from v3 and updates only the foreground
output channel using 50% corrected grandmother crops, 25% other real crops,
and 25% synthetic scenes. The backbone, batch normalization state, and
boundary output are unchanged. An unrestricted fine-tune improved the
grandmother review set more, but reduced the pirate sheet's boundary score.

On the pirate icon sheet, which was excluded from gradient training but
used for checkpoint and threshold selection, compared with approximate
expanded CLI masks:

| Method | Foreground IoU | Boundary F1 |
| --- | ---: | ---: |
| v2 synthetic model | 0.8631 | 0.2812 |
| v3 mixed-data model | **0.9650** | **0.5936** |
| v3 at thresholds calibrated on that validation sheet | **0.9704** | **0.6262** |
| v6 corrected-label model at 0.5 | **0.9640** | **0.5936** |
| v6 at its pirate-sheet optimum | **0.9704** | **0.6262** |
| Unexpanded classical mask | 0.8875 | — |

ONNX Runtime differed from PyTorch by at most **0.000053** for both v3 and
v6 on their export checks. V3's `calibration.json` records the selected
foreground threshold 0.725 and boundary threshold 0.8. The pirate sheet
was used to choose those thresholds, so the calibrated values are tuning
results, not scores from an independent test sheet.

On the four grandmother stickers withheld from training, at the common 0.5
foreground threshold:

| Model | Mean foreground IoU | Mean contour F1 within 5 px |
| --- | ---: | ---: |
| v3 | 0.9266 | 0.8679 |
| v6 | **0.9299** | **0.8765** |

The grandmother review stickers were used while iterating, so this is a
development comparison rather than an untouched test. The v6 default
foreground threshold is **0.5**: a 0.775 threshold maximizes pirate-sheet
pseudo-label IoU but lowers the grandmother review IoU to 0.9050 and visibly
cuts into pale hair. The boundary threshold remains 0.8. The pirate-sheet
boundary head is bit-for-bit unchanged from v3.

Four later hand-reviewed row-6 stickers were not used to train the model or
choose its threshold. At 0.5, mean foreground IoU rose from **0.9143 to
0.9176**, and mean contour F1 within 5 px rose from **0.8585 to 0.8733**.
All four stickers improved. This checks new instances from the same sheet,
not generalization to a separate image style.

`propose.py` uses each classical cut group as a seed, assigns nearby model
foreground to that group, fills internal holes, and writes outer polygon
proposals with an overlay. `--sheet` uses the reviewed groups recorded in
`data/sheets.json`; `--image` with `--analysis` uses all analyzer groups.
With v3, it kept **39/39** groups on the pirate sheet, reaching **0.9601
mean instance IoU** against its approximate masks without closing. It also
kept **24/24** portrait groups on the grandmother
grid with optional `--close-px 15`, but some pale hair still produces notches.
The original pastel sheet kept **29/29** CLI groups; existing fused pairs
remain fused because the seeds themselves are fused. The proposal report
counts pixels shared by two exported polygons, which need manual resolution
before cutting. These paths are for inspection, not cutter use.

With the v6 default threshold, the pirate proposal CLI kept **39/39** seeds
and reached **0.9669 mean instance IoU** versus its approximate pseudo-labels.
The grandmother grid kept **24/24** seeds using `--close-px 15`, but the
remaining pale-hair notches still need contour correction. On the grandmother
portrait sheet, **42** fragmented classical groups remain fragmented; better
foreground alone cannot assign them to the intended eight stickers.

The pirate evaluation target is derived from the classical outlines by
per-sheet dilation and can reward learning the dilation. The grandmother
polygons were manually reviewed, but only four instances from one source
sheet were used for the review score. Visual review shows far fewer inner
texture marks than v2, but some remain on pale unicorns and white hair.
The browser detector remains the safer default until more outer cut
contours and touching pairs are corrected and tested on separate sheets.

The next useful labels are **outer cut contours and instance IDs** from more
full sheets, especially white unicorns, dense decorations, and touching
pairs. A sheet-level holdout should remain untouched while trying better
targets, postprocessing, or a pretrained segmentation backbone.

## Full-sheet instance comparison, 2026-09-26

The full grandmother portrait sheet has eight intended sticker IDs; the
original pastel fairy sheet has 36, counting the held snowflake staff as part
of the ice-princess sticker. `data/manual/drafts/` records manually chosen
regions and image-pixel tracing parameters. `trace_reference.py` saves fixed
outer polygons and a review overlay. This does not use ML output, but the
edges are threshold-assisted and **not pixel-painted ground truth**. Six
grandmother seams and 15 pastel shapes with touching, nested, or clipped
white margins remain flagged. Only the two clean grandmother contours and
21 isolated pastel contours contribute to the IoU and boundary F1 below.

| Sheet | Method | Instance paths | Mean IoU on reviewed contours | Boundary F1 within 5 px | Overlapping path-mask pixels |
| --- | --- | ---: | ---: | ---: | ---: |
| Grandmother portraits (8 IDs) | Classical CLI | 42 | 0.6735 | 0.0168 | 0 |
| | V6 boundary watershed | 8 | **0.9526** | **0.8319** | 0 |
| | SAM 2.1 tiny, supplied boxes | 8 | 0.9263 | 0.5486 | 101 |
| Original pastel (36 IDs) | Classical CLI | 26 | 0.5259 | 0.2070 | 0 |
| | V6 boundary watershed | 43 | 0.8811 | 0.9392 | 1,916 |
| | SAM 2.1 tiny, supplied boxes | 36 | **0.9461** | **0.9977** | 64,780 |

`split_boundary.py` makes automatic high-confidence interior markers and
watersheds the V6 foreground using its boundary channel. It splits the
portraits into eight without classical seeds, but makes extra cuts inside the
pastel unicorn, rainbow, and swirl. `sam2_box_baseline.py` uses the reference
boxes as prompts and selects each box's highest-score mask. Its instance count
is therefore *given* by the prompts, and its extensive pastel mask overlap
would be unsafe as cut paths. The pastel clean-contour scores are high partly
because those isolated white rims are much easier than the flagged touching
cases. These sheets and parameters were inspected during development; the
numbers are exploratory, not an untouched test result. The browser detector
has not changed.

The compact report with annotation and SAM-checkpoint hashes is in
`evaluations/full-sheet-dev-2026-09-26.json`. To reproduce the offline paths:

```sh
.venv-ml/bin/python ml/trace_reference.py --draft ml/data/manual/drafts/grandma-portraits.json --out ml/data/manual/grandma-portraits.json --overlay /tmp/grandma-reference.png
.venv-ml/bin/python ml/trace_reference.py --draft ml/data/manual/drafts/pastel-fairy-original.json --out ml/data/manual/pastel-fairy-original.json --overlay /tmp/pastel-reference.png
.venv-ml/bin/python ml/split_boundary.py --image ml/data/sheets/grandma-portraits.jpeg --model ml/models/sticker-boundary-v6/model.onnx --min-core-area 20000 --out /tmp/grandma-boundary
.venv-ml/bin/python ml/split_boundary.py --image ml/data/sheets/pastel-fairy-original.jpeg --model ml/models/sticker-boundary-v6/model.onnx --min-core-area 500 --out /tmp/pastel-boundary
.venv-ml/bin/python ml/evaluate_instances.py --annotations ml/data/manual/grandma-portraits.json --predictions /tmp/grandma-boundary/instances.npz --out /tmp/grandma-boundary/evaluation.json
```

The optional prompted baseline also needs the official SAM 2 package and
checkpoint. The repository commit used was
`2b90b9f5ceec907a1c18123530e92e794ad901a4`; the model code and
checkpoint are [Apache 2.0 licensed](https://github.com/facebookresearch/sam2).

```sh
uv pip install --python .venv-ml/bin/python torchvision==0.29.0 --index https://download.pytorch.org/whl/cpu
SAM2_BUILD_CUDA=0 uv pip install --python .venv-ml/bin/python 'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'
mkdir -p ~/.cache/pixcut-studio/sam2
curl -L --fail 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt' -o ~/.cache/pixcut-studio/sam2/sam2.1_hiera_tiny.pt
.venv-ml/bin/python ml/sam2_box_baseline.py --annotations ml/data/manual/pastel-fairy-original.json --out /tmp/pastel-sam2
```

Possible public data sources and their terms are summarized in
`docs/public-segmentation-datasets.md`.

## Two-stage instance experiment, 2026-09-26

The offline experiment in `evaluations/instance-pipeline-2026-09-26.json`
tests a 2.38M-parameter YOLO26n box detector followed by a 320-pixel
box-conditioned U-Net. The detector sees a 768-pixel overview and overlapping
512-pixel tiles. The refiner receives RGB, a box map, and a center map; it
predicts one outer-rim mask per box. The ONNX exports are in
`models/sticker-detector-v1/` and `models/sticker-prompt-v1/`. The refiner uses
V11 weights as initialization. Its per-box probabilities are arbitrated with
one owner per pixel without stacking every full-sheet float map in memory.

To regenerate the synthetic detector data and train the models:

```sh
.venv-ml/bin/python ml/generate_detection_data.py --out /var/home/gordon/.cache/pixcut-studio/datasets/sticker-detector-v1
.venv-ml/bin/python ml/train_detector.py --data /var/home/gordon/.cache/pixcut-studio/datasets/sticker-detector-v1/data.yaml --pretrained /path/to/yolo26n.pt --out /var/home/gordon/.cache/pixcut-studio/experiments/sticker-detector-v1
.venv-ml/bin/python ml/train_prompt.py --base ml/models/sticker-boundary-v11-no-public-control/model.pt --out ml/models/sticker-prompt-v1
```

To run and score the saved models from the CLI:

```sh
.venv-ml/bin/python ml/detect_stickers.py --image ml/data/sheets/pastel-fairy-original.jpeg --model ml/models/sticker-detector-v1/model.pt --out /tmp/pixcut-boxes
.venv-ml/bin/python ml/evaluate_boxes.py --annotations ml/data/manual/pastel-fairy-original.json --predictions /tmp/pixcut-boxes/boxes.json --out /tmp/pixcut-boxes/evaluation.json
.venv-ml/bin/python ml/infer_prompt.py --image ml/data/sheets/pastel-fairy-original.jpeg --model ml/models/sticker-prompt-v1/model.onnx --boxes /tmp/pixcut-boxes/boxes.json --out /tmp/pixcut-instances
.venv-ml/bin/python ml/evaluate_instances.py --annotations ml/data/manual/pastel-fairy-original.json --predictions /tmp/pixcut-instances/instances.npz --out /tmp/pixcut-instances/evaluation.json
```

`infer_prompt.py --annotations ml/data/manual/pastel-fairy-original.json`
replaces automatic boxes with reference boxes to isolate the contour stage.
The detector's ONNX export also runs through `detect_stickers.py`; it proposed
67 boxes instead of 68 on the pastel sheet and still matched all 36 intended
boxes at IoU >= 0.5.

| Sheet | Boxes / intended | Paths | Matched instances | Reviewed contour IoU | Filled path overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pastel, V11 watershed | — / 36 | 38 | 36 | 0.9294 | 0 px |
| Pastel, reference boxes + prompt U-Net | 36 / 36 | 36 | 36 | 0.8971 | 11,336 px |
| Pastel, automatic boxes + prompt U-Net | 68 / 36 | 37 | 36 | 0.8889 | 11,485 px |
| Portraits, V11 watershed with 20,000-pixel core minimum | — / 8 | 8 | 8 | 0.9533 | 0 px |
| Portraits, reference boxes + prompt U-Net | 8 / 8 | 8 | 8 | 0.9594 | 108 px |
| Portraits, automatic boxes + prompt U-Net | 26 / 8 | 26 | 8 | 0.9184 | 7,141 px |

All intended stickers received a well-aligned detector box, but attached
hearts, rays, and tiny highlights also became boxes. The refiner makes strong
portrait contours with supplied boxes; on the pastel sheet its paths leak
into adjacent artwork. Raster masks have exclusive ownership, yet filling the
traced outer contours reveals overlaps. These results are **not cut safe**.
The two full-sheet reference sets are development drafts with ambiguous seams;
only 21 isolated pastel contours and two portrait contours contribute to the
IoU means. Both sheets were inspected during development. Synthetic validation
mAP50 of 0.995 is not a real-sheet precision claim. The browser detector is
unchanged.

## Private DIS5K experiment

The official [DIS5K](https://github.com/xuebinqin/DIS) training split supplies
high-resolution object masks. The local adapter crops each masked object,
paints a white rim around it, and places several objects on pastel backgrounds
with exact instance and outer-cut targets. Only `DIS-TR` source pairs are used;
its official validation and test partitions are excluded. Downloaded images,
masks, and the 5.68 GB archive stay in a private cache outside this repository.
The archive used here has SHA-256
`54afa8cb4f1c71148a7be088f33807945741f17d834f84f896e7625a08c5eb42`.
Its [terms](https://github.com/xuebinqin/DIS/blob/main/DIS5K-Dataset-Terms-of-Use.pdf)
allow this private noncommercial research experiment and prohibit redistribution.

To reproduce the 240-pair subset and the public-data fine-tune:

```sh
uv pip install --python .venv-ml/bin/python gdown==6.4.0
mkdir -p /var/home/gordon/.cache/pixcut-studio/datasets/dis5k
.venv-ml/bin/gdown 1O1eIuXX1hlGsV7qx4eSkjH231q7G1by1 -O /var/home/gordon/.cache/pixcut-studio/datasets/dis5k/DIS5K.zip
.venv-ml/bin/python ml/extract_dis5k.py --archive /var/home/gordon/.cache/pixcut-studio/datasets/dis5k/DIS5K.zip --out /var/home/gordon/.cache/pixcut-studio/datasets/dis5k/subset --limit 240 --seed 2026
.venv-ml/bin/python ml/train.py --out ml/models/sticker-boundary-v10-dis5k-distilled --resume ml/models/sticker-boundary-v6/model.pt --steps 180 --batch 8 --threads 8 --real-fraction 0.25 --manual-fraction 0.375 --public-fraction 0.125 --public-images /var/home/gordon/.cache/pixcut-studio/datasets/dis5k/subset/im --public-masks /var/home/gordon/.cache/pixcut-studio/datasets/dis5k/subset/gt --public-limit 240 --selection mixed-public --freeze-batchnorm --distill-original 50 --lr 0.00001
.venv-ml/bin/python ml/train.py --out ml/models/sticker-boundary-v11-no-public-control --resume ml/models/sticker-boundary-v6/model.pt --steps 180 --batch 8 --threads 8 --real-fraction 0.25 --manual-fraction 0.375 --public-fraction 0 --public-images /var/home/gordon/.cache/pixcut-studio/datasets/dis5k/subset/im --public-masks /var/home/gordon/.cache/pixcut-studio/datasets/dis5k/subset/gt --public-limit 240 --selection mixed-public --freeze-batchnorm --distill-original 50 --lr 0.00001
```

The subset contains 216 public source pairs for training and 24 disjoint pairs
for synthetic validation. The 64 original sticker assets, eight corrected
grandmother training portraits, and pirate icon validation sheet retain their
earlier roles. The grandmother portrait and original pastel fairy sheets remain
outside gradient training. The public-source image IDs and the selected model
metrics are written in the model manifest.

### Matched comparison

V10 was compared with V6 and a **no-public V11 control**. V11 uses the same
seed, 180 steps, optimizer, fixed batch-normalization layers, and V6-reference
regularization; its one public collage per batch is replaced by an original
sticker collage. All full-sheet results below use the same watershed settings
and `--merge-enclosed`. This optional rule joins a tiny region to a touching
larger region only when it lies inside the larger region's outer contour. It
removes a false unicorn-eye path from all three models. It does not resolve the
remaining body fragments of that unicorn.

| Development check | V6 | V10 with DIS5K | V11 without DIS5K |
| --- | ---: | ---: | ---: |
| Public synthetic holdout foreground IoU | 0.8319 | **0.8826** | 0.8454 |
| Public synthetic holdout best boundary F1 | 0.3147 | **0.4462** | 0.3553 |
| Original pastel: predicted paths / 36 intended | 42 | 39 | **38** |
| Original pastel: matched intended instances | 34 | **36** | **36** |
| Original pastel: mean IoU on 21 reviewed contours | 0.8811 | **0.9307** | 0.9294 |
| Grandmother portraits: paths / 8 intended | **8** | **8** | **8** |
| Grandmother grid row 6: mean contour F1 within 5 px | 0.8733 | **0.8950** | 0.8935 |
| Pirate icons: boundary F1 against pseudo-labels at 0.5 | **0.5936** | 0.5569 | 0.5737 |

The public masks improve performance on held-out **synthetic public objects**.
On real sticker sheets, V10 and V11 are very close; V11 has one fewer extra
pastel path and a smaller pirate-sheet regression. This experiment therefore
does **not** establish that DIS5K improves real cut contours beyond more
training on the existing sticker examples. DIS5K V1 also contains few people
or animals, while these sheets are dominated by people and unicorns. More
precise labels for the hard touching shapes, plus an untouched sheet-level
test, are more valuable than adding more generic object photos at this stage.
The browser still uses the classical detector; none of these model proposals
are ready for cutting without review.

The compact [ablation report](evaluations/public-data-ablation-2026-09-26.json)
and [per-instance reports](evaluations/public-data-2026-09-26/) preserve the
numbers, model hashes, and reference quality flags. To reproduce the offline
path comparison after training:

```sh
.venv-ml/bin/python ml/split_boundary.py --image ml/data/sheets/pastel-fairy-original.jpeg --model ml/models/sticker-boundary-v10-dis5k-distilled/model.onnx --min-core-area 500 --merge-enclosed --out /tmp/pixcut-dis5k-pastel-v10-merged
.venv-ml/bin/python ml/evaluate_instances.py --annotations ml/data/manual/pastel-fairy-original.json --predictions /tmp/pixcut-dis5k-pastel-v10-merged/instances.npz --out /tmp/pixcut-dis5k-pastel-v10-merged/evaluation.json
.venv-ml/bin/python ml/split_boundary.py --image ml/data/sheets/grandma-portraits.jpeg --model ml/models/sticker-boundary-v10-dis5k-distilled/model.onnx --min-core-area 20000 --merge-enclosed --out /tmp/pixcut-dis5k-grandma-v10-merged
```

## AM-2k alpha-matte follow-up

The [AM-2k official release](https://github.com/JizhiziLi/GFM) contains animal
photos and manually labeled alpha mattes. Its [agreement](https://jizhizili.github.io/files/gfm_datasets_agreements/AM-2k_Dataset_Release_Agreement.pdf)
lists MIT terms while retaining original-image copyright with the image owners.
We sampled 120 image/matte pairs from its **training** partition and kept them
in a private cache. The adapter alpha-composites each animal on white and
generates the outer sticker rim. It uses 108 pairs for training and 12
different pairs for synthetic validation; the official validation partition
was not used. A source-name bug in the original generic loader was also fixed.

```sh
mkdir -p /var/home/gordon/.cache/pixcut-studio/datasets/am2k
.venv-ml/bin/python -m gdown --json 'https://drive.google.com/drive/folders/1SReB9Zma0TDfDhow7P5kiZNMwY9j9xMA' > /var/home/gordon/.cache/pixcut-studio/datasets/am2k/index.json
.venv-ml/bin/python ml/fetch_am2k.py --index /var/home/gordon/.cache/pixcut-studio/datasets/am2k/index.json --out /var/home/gordon/.cache/pixcut-studio/datasets/am2k/subset --limit 120
.venv-ml/bin/python ml/train.py --out ml/models/sticker-boundary-v12-am2k-distilled --resume ml/models/sticker-boundary-v6/model.pt --steps 180 --batch 8 --threads 8 --real-fraction 0.25 --manual-fraction 0.375 --public-fraction 0.125 --public-images /var/home/gordon/.cache/pixcut-studio/datasets/am2k/subset/im --public-masks /var/home/gordon/.cache/pixcut-studio/datasets/am2k/subset/gt --public-limit 120 --public-source AM-2k --public-url https://github.com/JizhiziLi/GFM --public-alpha --selection mixed-public --freeze-batchnorm --distill-original 50 --lr 0.00001
.venv-ml/bin/python ml/train.py --out ml/models/sticker-boundary-v13-am2k-no-public-control --resume ml/models/sticker-boundary-v6/model.pt --steps 180 --batch 8 --threads 8 --real-fraction 0.25 --manual-fraction 0.375 --public-fraction 0 --public-images /var/home/gordon/.cache/pixcut-studio/datasets/am2k/subset/im --public-masks /var/home/gordon/.cache/pixcut-studio/datasets/am2k/subset/gt --public-limit 120 --public-source AM-2k --public-url https://github.com/JizhiziLi/GFM --public-alpha --selection mixed-public --freeze-batchnorm --distill-original 50 --lr 0.00001
```

The new no-public V13 control uses AM-2k for checkpoint selection and no AM-2k
training tiles. It selected step 125, and its PyTorch weights are byte-identical
to the earlier V11 control. The comparisons below therefore use V11/V13 as the
same matched control.

| Development check | V11/V13 no-public control | V12 with AM-2k |
| --- | ---: | ---: |
| Synthetic animal holdout foreground IoU | 0.9104 | **0.9268** |
| Synthetic animal holdout best boundary F1 | 0.4553 | **0.5385** |
| Original pastel paths / 36 intended | **38** | 39 |
| Original pastel matched instances | 36 | 36 |
| Original pastel mean IoU on 21 reviewed contours | 0.9294 | **0.9301** |
| Grandmother portraits paths / 8 intended | 8 | 8 |
| Grandmother grid row 6 contour F1 within 5 px | 0.8935 | **0.8974** |
| Pirate icon boundary F1 against pseudo-labels at 0.5 | **0.5737** | 0.5599 |

AM-2k helps its synthetic animal domain but produces no clear improvement to
real cut paths. The original pastel sheet still has extra internal unicorn
cuts. The detailed [AM-2k ablation report](evaluations/am2k-ablation-2026-09-26.json)
has source selection and model hashes, with per-instance reports in
`evaluations/am2k-2026-09-26/`. Both full-sheet reference sets were used in
development, and the clean-contour scores exclude ambiguous touching seams.
The next model experiment is described in
[the architecture note](../docs/next-model-architecture.md): predict sticker
instances explicitly, then refine each outer border at higher resolution.
