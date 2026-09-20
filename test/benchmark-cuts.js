// Optional local regression: private source images are not published or required by CI.
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import assert from "node:assert/strict";
import { PNG } from "pngjs";
import cvModule from "@techstark/opencv-js";
import { openCVReady } from "../src/opencv-ready.js";
import { analyze, DEFAULTS } from "../src/autocut.js";
import Clipper from "clipper-lib";
const { cv } = await openCVReady(cvModule);
const fixtureRoot = resolve(process.argv[2] || "fixtures");
for (const name of ["pirates", "fairy"]) {
  const png = PNG.sync.read(
    await readFile(
      resolve(fixtureRoot, `${name === "pirates" ? "pirate-stickers" : name}.png`),
    ),
  );
  const reference = JSON.parse(
    await readFile(
      resolve(fixtureRoot, `prepared-${name}/contours.json`),
    ),
  );
  const scale = 1860 / png.height,
    xOffset = Math.floor((1200 - Math.round(png.width * scale)) / 2),
    pad = 24;
  const seeds = reference.map((path) =>
    path.map(([x, y]) => [
      (x - xOffset) / scale + pad,
      (y - 120) / scale + pad,
    ]),
  );
  const width = png.width + pad * 2,
    height = png.height + pad * 2,
    rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < png.height; y++)
    rgba.set(
      png.data.subarray(y * png.width * 4, (y + 1) * png.width * 4),
      ((y + pad) * width + pad) * 4,
    );
  const mm = 300 / 25.4 / scale;
  const params = {
    ...DEFAULTS,
    mode: "white",
    saturation: name === "pirates" ? 45 : 35,
    minArea: 4 / mm ** 2,
    border: 6 / mm,
    bridge: 12 / mm,
    smooth: 5 / mm,
    gap: 4 / mm,
    detail: 0.65 / mm,
  };
  const t = performance.now();
  const result = analyze(cv, {
    width,
    height,
    rgba,
    pxPerMM: mm,
    params,
    seedPaths: seeds,
  });
  let intersection = 0,
    union = 0;
  const polygon = (p) =>
    p.map(([X, Y]) => ({ X: Math.round(X * 100), Y: Math.round(Y * 100) }));
  for (let i = 0; i < result.groups.length; i++) {
    const group = result.groups[i],
      referencePaths = group.seedPaths;
    for (const [operation, accumulate] of [
      [Clipper.ClipType.ctIntersection, (v) => (intersection += v)],
      [Clipper.ClipType.ctUnion, (v) => (union += v)],
    ]) {
      const clip = new Clipper.Clipper(),
        out = [];
      clip.AddPaths(group.paths.map(polygon), Clipper.PolyType.ptSubject, true);
      clip.AddPaths(referencePaths.map(polygon), Clipper.PolyType.ptClip, true);
      clip.Execute(
        operation,
        out,
        Clipper.PolyFillType.pftNonZero,
        Clipper.PolyFillType.pftNonZero,
      );
      accumulate(
        out.reduce((n, p) => n + Math.abs(Clipper.Clipper.Area(p)), 0),
      );
    }
  }
  console.log(
    JSON.stringify(
      {
        name,
        referenceGroups: reference.length,
        groups: result.groups.length,
        outlines: result.groups.reduce((n, g) => n + g.paths.length, 0),
        components: result.components.length,
        clippedPixels: result.groups.reduce((n, g) => n + g.clipped, 0),
        areaIoU: intersection / union,
        seconds: (performance.now() - t) / 1000,
        issues: result.issues.slice(0, 5),
      },
      null,
      2,
    ),
  );
  assert.equal(result.groups.length, reference.length);
  assert.equal(
    result.groups.reduce((n, g) => n + g.paths.length, 0),
    reference.length,
  );
  assert.deepEqual(result.issues, []);
  assert.ok(intersection / union > 0.99);
  for (const join of [0.6, 1, 1.5, 2])
    for (const keepSmall of [true, false]) {
      const automatic = analyze(cv, {
        width,
        height,
        rgba,
        pxPerMM: mm,
        params: {
          ...DEFAULTS,
          mode: "white",
          saturation: params.saturation,
          join,
          keepSmall,
        },
        groupingOnly: true,
      });
      console.log(
        JSON.stringify({
          name,
          join,
          keepSmall,
          automaticGroups: automatic.groups.length,
        }),
      );
    }
}
