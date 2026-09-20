# PixCut Studio

[Open PixCut Studio](https://elohmeier.github.io/pixcut-studio/) ·
[Build status](https://github.com/elohmeier/pixcut-studio/actions/workflows/pages.yml)

Independent, unofficial software for Liene PixCut S1. Not affiliated with Liene.

A local-first Fabric.js sticker editor and browser USB/BLE client. The static
production build can be hosted on GitHub Pages. No image upload, cloud account,
server process, or native bridge is required by the app.

## Run locally

```sh
git clone https://github.com/elohmeier/pixcut-studio.git
cd pixcut-studio
npm ci --ignore-scripts
npm run dev -- --port 5173
```

Open **http://127.0.0.1:5173/** in desktop Chrome. `--ignore-scripts` avoids
installing Fabric's optional Node canvas binary; the app uses browser canvas.

1. Select **4×7 sticker paper** or **4×6 photo paper**, then add PNG/JPEG/WebP
   images, or use the sample. Drag, resize, and rotate them. **Rotate 90°
   clockwise** automatically fits photos to the sheet without cropping;
   **Fit artwork to page** preserves the chosen rotation.
2. Choose transparency-based, rectangular, or custom polygon cuts. Set the
   white border in millimeters. Custom outlines are drawn by selecting a
   sticker, choosing **Draw a custom outline**, and clicking around it.
3. Review the sheet. Overlapping or out-of-sheet cut contours block export.
4. Connect USB or Bluetooth. Connection/status
   checks alone do not print. Close other software that owns the USB interface.
5. Check the paper/outline confirmation, then **Print & cut one sheet**.

In 4×6 mode, cut controls are disabled and the app sends only a 1200×1800
JPEG in a standalone print job, never a cut job. Review and choose **Print one
photo**. A standalone 1200×1800 JPEG can also be opened as a prepared job.
Bluetooth photo printing was confirmed working on this printer by the user.

For previously prepared jobs, use **Open a prepared sheet → Open prepared
job**, selecting a `.bin` file from your local Python-driver workspace.
A JPEG + PLT pair also works. Imported jobs are locked and
their bytes are preserved. Downloads of an imported job are byte-identical.
No source photos or prepared sheets are bundled in the web build.

For an editable flattened sheet, add the image, then import its matching
`contours.json`. That operation fits the single image to the full 1200×2100
sheet and attaches the supplied contours to it. For the existing Python
outputs, import **sheet.png**, not the original source PNG: `contours.json`
already includes the Python layout's margins.

Opaque images initially receive rectangular cuts. Use **Automatic cuts &
outline editor** to detect separate motifs in a flattened collage (see below).
The basic alpha tracing mode keeps outer contours only, ignores tiny
islands, and samples at up to 600 pixels along the longest dimension. Always
review fine details. A large border can merge nearby alpha islands within one
image into one cut. Separately overlapping stickers must be moved apart.

## Automatic cuts and manual correction

1. Add an image and **Fit artwork to page** before analyzing. Select
   **Automatic cuts & outline editor**. Photo-only mode disables this feature.
2. The initial pass uses transparency if present, otherwise colored/dark ink
   on white. Choose a background color for nonwhite backgrounds. The padded
   analysis border is excluded from transparency detection.
3. Review the numbered groups. Click a motif to select it; Shift-click or
   use the multi-select list to merge groups. Split a group into connected
   pieces, or assign individual pieces to another group/a new group. Small
   detached accents often need this correction. Excluded pieces still print;
   **Restore excluded pieces** brings them back as separate groups.
4. Adjust border, bridge radius, smoothing, spacing and simplification in
   millimeters, then **Update contours**. Detection/grouping settings take
   effect only with **Detect again**, which asks before replacing edits.
5. Choose **Edit outline points** to drag Fabric polygon controls. The vertex
   selector supports insertion/deletion. Manual changes lock that group's
   outlines: regeneration preserves them until **Reset selected to automatic**.
   Undo/redo covers the current correction workspace (up to 20 steps).
6. **Validate & apply cuts**, review the sheet, then print explicitly.
   **Save editable project** embeds the artwork, transforms, component masks,
   grouping, parameters and manual overrides in a local JSON file. It does
   not store printer connections or submit jobs when reopened.

The **Detailed collage** preset starts with 1.1 mm joining; **Larger, rounded
motifs** uses 1.5 mm. These are starting points, not semantic segmentation.
Joining can connect chains of nearby motifs. Keeping small decorations
separate may instead create many small groups. Touching artwork is a single
connected component: component splitting cannot infer a boundary through it;
use custom outlines or edit the source image for such cases.

To refine existing Python contours: add the corresponding **sheet.png**,
import **contours.json**, then open the automatic editor. Existing contours
become grouping regions; whole detected components are assigned by overlap
and distance, just as in the Python preparation workflow.

### Algorithm and safety

- Pinned OpenCV.js 4.12 runs in a lazy-loaded Web Worker, entirely on-device.
  Its bundled WASM/JS worker is about 10.8 MB before HTTP compression. No CDN,
  image upload, cross-origin isolation, or native bridge is required.
- HSV/background/alpha masking → connected components → distance-based
  grouping → dilation, closing, opening and Gaussian smoothing → nearest
  artwork ownership partition → external contours and polygon simplification.
- Refinement processes cropped group regions; distance images are released
  incrementally rather than retaining a full group × sheet stack. OpenCV
  matrices are disposed in `finally` blocks. Cancel terminates the worker;
  analysis has a two-minute timeout and leaves applied cuts unchanged.
- All external contours are retained. Internal holes are not cut. If
  simplification clips detected pixels, retry without simplification. Remaining
  clipping conflicts block applying cuts, as do overlapping, self-crossing,
  out-of-sheet or insufficiently spaced paths. Detection can still miss pale
  artwork: always review it visually. Tiny discarded features are not covered
  by the containment guarantee.
- The grid samples the image at up to 1,700 pixels on its longest side with
  padding. Geometry settings use its physical placed scale. Resizing artwork
  invalidates automatic export until contours are updated; rotation/flipping
  retain correct attachment. Initial analysis requires uniform scale/no skew.
- Cuts are metadata, not objects on the print canvas. Editing handles and
  colored overlays never enter the JPEG. This feature does not modify the
  printer protocol, calibration, cutting pressure or firmware.

### Regression evidence

`npm test` exercises OpenCV segmentation, component corrections, override
preservation, clipping, manual intersections/spacing, project validation and
Fabric vertex transforms, alongside the existing transport tests.

With the private source images and prepared contours present locally:

```sh
npm run benchmark:cuts -- /path/to/your/pixcut-s1-workspace
```

Using existing contours as grouping seeds and the Python-scale parameters,
the local benchmark produced 35 pirate / 57 fairy contours, zero detected
artwork pixels clipped, and respective area intersections-over-union of
0.99699 / 0.99646 against those seeds. This validates **refinement given known
grouping**, not fully automatic recognition. Unseeded group counts vary
substantially with joining and require review. The benchmark uses the original
source grid; the browser's sampled grid may differ slightly. Private images
are not bundled or required for CI.

OpenCV.js packaging: [TechStark/opencv-js](https://github.com/TechStark/opencv-js).
Algorithm reference: [OpenCV morphology](https://docs.opencv.org/4.12.0/d4/d76/tutorial_js_morphological_ops.html).

## Transport and recovery

- **USB:** independently ported from the [Python driver](https://github.com/perschulte/pixcut-s1):
  VID/PID `302c:3101`, vendor interface 2 (+ optional 3), endpoint number 6 in
  both directions, 10,211-byte payload chunks, strict ACKs and 118 ms pacing.
- **BLE:** FF00 service, FF02 writes, FF01 responses, FF03 `01 01`
  acknowledgements; Hannto framing with 156 job bytes per data frame. Protocol
  behavior was cross-checked against the local `eastbaymakersclub/pixcut-s1`
  project. This implementation is original code; the reference repo's source
  was not copied. Physical BLE printing was confirmed working by the user.
- Uses the existing 1200×2100 calibration, KP42, 4×7 media ID 5013, sticker
  type 2030, job type 600, one copy. No firmware/update commands are exposed.
- Photo-only mode uses media ID 5012, type 2010, job type 0, one copy, with
  no PLT payload or combo/cut command.
- Persist job ID and transport locally before sending data. A lost job-creation
  response also blocks another submission until the user resolves the record.
  No automatic upload retry, cancellation, or duplicate submission.
- USB completion requires job state 9. BLE uses short status polls because
  larger responses can truncate on some stacks; return to idle after activity
  asks you to confirm the finished sheet rather than claiming automatic completion.
  Status timeouts after upload are distinguished from upload failures. Use
  **My sheet finished printing** after the finished sheet leaves the printer;
  this clears only the local job record and never resends or cancels a job.
- A device error, disconnected transport, or timeout keeps the job record.
  Reconnect and use **Monitor existing job**, or explicitly cancel it. Clearing
  the local record does not cancel a job on the printer.
- The browser tab must stay open during transfer. Keep it in the foreground;
  the app requests a screen wake lock while sending/monitoring when supported.

USB and BLE require secure contexts (HTTPS or localhost) and a browser device
chooser initiated by a user click. macOS Chrome is the initial target. OS
permissions/driver binding can affect USB on Linux/Windows. BLE support varies
by platform. Feature detection displays unavailable connection options.

## Architecture

- `src/protocol.js`: calibration, constrained PLT parsing, JPEG dimension
  checks, job construction, USB/BLE framing.
- `src/geometry.js`: alpha contours, physical border offsets, overlap checks.
- `src/transports.js`: exclusive command queues, USB/BLE I/O and flow control.
- `src/main.js`: Fabric scene, separate cut overlay, unzoomed export canvas,
  local job persistence and explicit print/recovery actions.

Artwork transforms apply to its local cut geometry. Print export uses a
separate 1200×2100 (sticker) or 1200×1800 (photo) StaticCanvas, avoiding editor zoom, screen pixel density,
selection controls, and cut-overlay contamination.

```sh
npm test
npm run build
npm run preview
```

Tests cover framing, fragmented notifications, geometry, invalid payloads,
flow control, and failure behavior with fake transports. They do not establish
browser/OS compatibility with physical hardware. Start hardware validation
with **Connect USB → Check status**, then an imported known-good job.

## GitHub Pages

The build uses relative asset paths, so repository subpaths work. The workflow
at `.github/workflows/pages.yml` tests/builds pull requests and deploys successful
pushes to `main`. Manual deployment is also available on `main`. Pages uses
**GitHub Actions** as its source. Only `dist` is published; private images,
printer jobs, firmware, and local state are never part of the deployment.

Device permissions and saved job records belong to a browser origin. Moving
from localhost to Pages requires connecting again. If a localhost job is still
pending, resolve it there and check the printer before submitting on Pages.

The MIT notice from the originating driver project is retained in `LICENSE`.
Third-party packages retain their own licenses; the build includes the
OpenCV.js Apache-2.0 license at `licenses/opencv-js.txt`.

Protocol references:

- [Python driver and protocol documentation](https://github.com/perschulte/pixcut-s1)
- [Independent BLE observations](https://github.com/eastbaymakersclub/pixcut-s1)
- [Chrome WebUSB](https://developer.chrome.com/docs/capabilities/usb)
- [Chrome Web Bluetooth](https://developer.chrome.com/docs/capabilities/bluetooth)
