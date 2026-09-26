# Cut segmentation evaluation

## Current sample

The fairy and unicorn collage is a useful stress case: pastel background,
bright white sticker rims, small jewel stickers, narrow gaps, and several
backings that touch or overlap. Offline comparison uses a local, Git-ignored
copy at `ml/data/sheets/pastel-fairy-original.jpeg`.
With the current analyzer, the browser finds 26 cut groups at the initial
import scale and 30 after **Fit artwork to page**. The CLI produces 26 and 29
respectively because its bilinear rasterizer differs slightly from browser
canvas decoding. These are output counts. The later benchmark in
`ml/README.md` uses 36 image-derived instance references, with only 21
isolated contours marked as reviewed. Touching or nested white margins
remain uncertain.

Review the CLI's `overlay.svg`, `overlay.png`, and `components.png` together.
The fitted overlay shows these clear mergers:

- The lower-left snowflake, rainbow, and nearby decorations share one cut.
- The sitting unicorn includes the blue gem on its left.
- The shooting star includes the neighboring blue crystal.
- The top-right unicorn includes the gold sparkle to its right.
- The blue diamond at right includes the small pink sparkle above it.

The first three contain a fused foreground component or shared bright rim.
Changing `join` from 1.1 to 0.3 mm did not change the fitted 29-group result.
That makes a grouping-threshold change an unlikely fix for the main errors.
At the smaller import scale, millimeter-based joining and border radii are
larger in source pixels, so more items merge. The browser UI already advises
fitting artwork before analysis, but both scales should be evaluated.

## Evaluation targets

Build a small consented set of 20–30 sheets with manually reviewed sticker
instance masks and intended cut polygons. Include colored and white
backgrounds, transparent artwork, touching white rims, small decorations,
and differently sized placements. Keep test sheets separate from training.

Measure instance precision/recall, false merges, false splits, missed
stickers, cut-boundary error in millimeters, artwork pixels outside a cut,
minimum cut spacing, and runtime/memory on iPhone Safari. Compare predictions
after the same cut-spacing and contour post-processing so model scores reflect
segmentation quality rather than different printer geometry.

## Experiments, in order

1. **Improve the current mask.** Visualize the estimated backdrop and raw
   foreground mask at full resolution. Use local edge evidence at the outer
   white rim, retain a background path through narrow pastel gaps, and test
   marker-controlled watershed only where a component has a plausible narrow
   bridge. [OpenCV's JavaScript watershed guide](https://docs.opencv.org/4.12.0/d7/d1c/tutorial_js_watershed.html)
   covers separating touching objects. Require a strong split score because
   generic watershed can divide a single fairy into limbs, hair, and dress.
2. **Try promptable masks offline.** [SAM 2](https://github.com/facebookresearch/sam2)
   supports automatic mask generation and point/box prompts. Test its automatic
   proposals, then prompts derived from current components and user clicks.
   Score the *whole sticker including its white rim*, since a plausible
   character mask alone is not a usable cut. A model mask can provide instance
   identity while the current color/edge logic refines the actual cut edge.
3. **Try concept detection offline.** [SAM 3](https://github.com/facebookresearch/sam3)
   accepts text and exemplar prompts and returns instance masks. Compare
   prompts such as “die-cut sticker” with a few exemplars from this sheet.
   Its official checkpoint access and Python/PyTorch workflow make it an
   evaluation tool first, rather than an immediate browser dependency.
4. **Train a small task-specific model if generic models miss stickers.**
   An instance-segmentation model can learn the backing rather than the icon
   category. Synthetic sheets made from isolated sticker art plus randomized
   white offsets, pastel fields, overlap, blur, and compression can supply
   labels; real sheets must validate the result. A compact model could predict
   foreground, border/contact pixels, and instance centers, with the existing
   contour and spacing code as post-processing. [Ultralytics documents an
   instance-segmentation training/export path](https://docs.ultralytics.com/tasks/segment),
   though its [AGPL or enterprise license](https://github.com/ultralytics/ultralytics#-license)
   must be considered before incorporation.

Whole-image background removers such as
[BiRefNet](https://github.com/ndming/birefnet) are worth one foreground-mask
baseline, but they do not by themselves assign individual sticker identities.
For local browser deployment, [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
lists WASM support on iOS Safari; its WebGPU provider is not listed there.
Any proposed on-device model needs an iPhone download, memory, and latency
measurement before integration.

An initial V6 boundary-watershed and an oracle-box SAM 2.1 baseline are now
benchmarked on this sample. See `ml/README.md`. Neither is ready for cutter
use; the oracle boxes and provisional reference edges make this a development
comparison, not an untouched test.
