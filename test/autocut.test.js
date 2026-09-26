import test from "node:test";
import assert from "node:assert/strict";
import cvModule from "@techstark/opencv-js";
import { openCVReady } from "../src/opencv-ready.js";
import {
  analyze,
  maskPixels,
  DEFAULTS,
  mergeGroups,
  splitGroup,
  moveComponents,
} from "../src/autocut.js";
import { validateAnalysisData } from "../src/project.js";
import { validateSimplePath, validateSpacing } from "../src/geometry.js";
import { adaptiveMask } from "../src/adaptive-background.js";
import { Polygon, controlsUtils, Point } from "fabric";
test("Fabric vertex controls preserve the other vertices under rotation and scale", () => {
  const poly = new Polygon(
    [
      { x: 10, y: 10 },
      { x: 50, y: 10 },
      { x: 50, y: 50 },
      { x: 10, y: 50 },
    ],
    { angle: 30, scaleX: 1.5, scaleY: 1.5 },
  );
  const scene = (i) =>
    new Point(poly.points[i])
      .subtract(poly.pathOffset)
      .transform(poly.calcTransformMatrix());
  const anchor = scene(3),
    target = scene(0).add(new Point(5, 7));
  const controls = controlsUtils.createPolyControls(poly);
  controls.p0.actionHandler({}, { target: poly }, target.x, target.y);
  assert.ok(scene(3).distanceFrom(anchor) < 1e-8);
  assert.ok(scene(0).distanceFrom(target) < 1e-8);
});
const { cv } = await openCVReady(cvModule);
test("crossing manual vertices and insufficient spacing block application", () => {
  assert.throws(
    () =>
      validateSimplePath([
        [10, 10],
        [30, 30],
        [10, 30],
        [30, 10],
      ]),
    /itself/,
  );
  const a = [
      [20, 20],
      [40, 20],
      [40, 40],
      [20, 40],
    ],
    b = a.map(([x, y]) => [x + 25, y]);
  validateSimplePath(a);
  validateSpacing([a, b], 4);
  assert.throws(() => validateSpacing([a, b], 6), /spacing/);
});
function fixture() {
  const width = 120,
    height = 100,
    rgba = new Uint8Array(width * height * 4).fill(255);
  for (const [x, y, w, h] of [
    [20, 20, 20, 30],
    [60, 20, 20, 30],
    [24, 65, 5, 5],
  ])
    for (let yy = y; yy < y + h; yy++)
      for (let xx = x; xx < x + w; xx++)
        rgba.set([80, 20, 150, 255], (yy * width + xx) * 4);
  return {
    width,
    height,
    rgba,
    pxPerMM: 5,
    params: { ...DEFAULTS, mode: "white", join: 0, gap: 0.2, border: 0.6 },
  };
}
test("white, alpha and colored background detection", () => {
  assert.deepEqual(
    [
      ...maskPixels(
        new Uint8Array([255, 255, 255, 255, 100, 50, 200, 255, 0, 0, 0, 0]),
        { mode: "white" },
      ).mask,
    ],
    [0, 255, 0],
  );
  assert.deepEqual(
    [
      ...maskPixels(new Uint8Array([255, 255, 255, 255, 0, 0, 0, 0]), {
        mode: "alpha",
      }).mask,
    ],
    [255, 0],
  );
  assert.deepEqual(
    [
      ...maskPixels(new Uint8Array([0, 255, 0, 255, 255, 0, 0, 255]), {
        mode: "color",
        background: "#00ff00",
      }).mask,
    ],
    [0, 255],
  );
});
test("OpenCV pipeline preserves detected artwork and separate decorations", () => {
  const result = analyze(cv, fixture());
  assert.equal(result.components.length, 3);
  assert.equal(result.groups.length, 3);
  assert.deepEqual(result.issues, []);
  assert.ok(
    result.groups.every((g) => g.paths.length === 1 && g.clipped === 0),
  );
});
test("merge, split and reassignment never lose or duplicate pieces", () => {
  let groups = [
    { componentIds: [1] },
    { componentIds: [2] },
    { componentIds: [3] },
  ];
  groups = mergeGroups(groups, [0, 1]);
  assert.equal(groups.length, 2);
  groups = splitGroup(groups, 1);
  assert.equal(groups.length, 3);
  groups = moveComponents(groups, [3], 1);
  assert.equal(groups.length, 2);
  assert.deepEqual(groups.flatMap((g) => g.componentIds).sort(), [1, 2, 3]);
  assert.throws(
    () =>
      mergeGroups(
        [{ componentIds: [1], override: [] }, { componentIds: [2] }],
        [0, 1],
      ),
    /locked/,
  );
});
test("regeneration retains manual paths and detects artwork clipped by them", () => {
  const f = fixture(),
    a = analyze(cv, f),
    override = [
      [
        [20, 20],
        [24, 20],
        [24, 24],
        [20, 24],
      ],
    ];
  a.groups[0].override = override;
  const b = analyze(cv, { ...f, components: a.components, groups: a.groups });
  assert.deepEqual(b.groups[0].paths, override);
  assert.ok(b.issues.some((s) => s.includes("outside its cut")));
});
test("supplied regions group separated pieces, retaining all disconnected outlines", () => {
  const a = analyze(cv, {
    ...fixture(),
    seedPaths: [
      [
        [10, 10],
        [50, 10],
        [50, 80],
        [10, 80],
      ],
      [
        [55, 10],
        [90, 10],
        [90, 55],
        [55, 55],
      ],
    ],
  });
  assert.equal(a.groups.length, 2);
  assert.equal(a.groups[0].componentIds.length, 2);
  assert.deepEqual(a.issues, []);
  const groups = [{ componentIds: a.components.map((c) => c.id) }];
  const b = analyze(cv, {
    ...fixture(),
    components: a.components,
    groups,
    params: { ...DEFAULTS, bridge: 0, border: 0.2 },
  });
  assert.equal(b.groups[0].paths.length, 3);
});
test("saved analysis survives JSON; malformed component runs are rejected", () => {
  const a = analyze(cv, fixture());
  const data = {
    analysis: a,
    width: 120,
    height: 100,
    factor: 1,
    pad: 10,
    pxPerMM: 5,
    sourceWidth: 100,
    sourceHeight: 80,
    scaleAtAnalysis: 1,
  };
  validateAnalysisData(JSON.parse(JSON.stringify(data)));
  data.analysis.components[0].runs[0] = [-1, 5];
  assert.throws(() => validateAnalysisData(data), /pixels/);
});
test("automatic mode ignores transparent analysis padding on opaque white artwork", () => {
  const f = fixture();
  for (let x = 0; x < 120; x++) f.rgba[x * 4 + 3] = 0;
  const a = analyze(cv, {
    ...f,
    params: { ...f.params, mode: "auto" },
    contentRect: [1, 1, 118, 98],
  });
  assert.equal(a.mode, "white");
  assert.equal(a.groups.length, 3);
});
test("automatic detection follows white rims on a changing pastel background", () => {
  const width = 140, height = 110,
    rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = (y * width + x) * 4;
    rgba.set([
      Math.round(215 + x * 0.2 + y * 0.06),
      Math.round(222 + x * 0.04 + y * 0.19),
      Math.round(245 - x * 0.08 - y * 0.05), 255,
    ], i);
    for (const [x0, y0, x1, y1] of [[18, 18, 58, 77], [82, 24, 123, 88]])
      if (x >= x0 && x <= x1 && y >= y0 && y <= y1)
        rgba.set(x >= x0 + 4 && x <= x1 - 4 && y >= y0 + 4 && y <= y1 - 4
          ? [246, 232, 247, 255] : [255, 255, 255, 255], i);
  }
  const mask = adaptiveMask(cv, rgba, width, height, null, 128);
  assert.equal(mask[20 * width + 18], 255);
  assert.equal(mask[0], 0);
  const a = analyze(cv, {
    width, height, rgba, pxPerMM: 5,
    params: { ...DEFAULTS, join: 0, gap: 0.2 },
  });
  assert.equal(a.mode, "adaptive");
  assert.equal(a.groups.length, 2);
  assert.deepEqual(a.issues, []);
  assert.equal(a.groups.reduce((n, g) => n + g.paths.length, 0), 2);
});
test("small background glints cannot chain two adaptive sticker groups", () => {
  const width = 120, height = 70,
    rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = (y * width + x) * 4;
    rgba.set([225 + Math.round(x / 12), 230 + Math.round(y / 8), 247, 255], i);
    if ((x >= 10 && x < 38 || x >= 82 && x < 110) && y >= 15 && y < 56)
      rgba.set([255, 255, 255, 255], i);
    if (y >= 33 && y < 35 && x >= 42 && x < 80 && x % 4 < 2)
      rgba.set([55, 70, 130, 255], i);
  }
  const a = analyze(cv, {
    width, height, rgba, pxPerMM: 5,
    params: { ...DEFAULTS, mode: "adaptive" }, groupingOnly: true,
  });
  assert.equal(a.groups.length, 2);
  assert.ok(a.components.length > a.groups.flatMap((g) => g.componentIds).length);
});
test("a short line separates touching art and preserves project data", () => {
  const width = 120, height = 100,
    rgba = new Uint8Array(width * height * 4).fill(255);
  for (let y = 20; y < 80; y++) for (let x = 15; x < 105; x++) {
    if (x > 53 && x < 66 && (y < 45 || y > 54)) continue;
    rgba.set([110, 30, 160, 255], (y * width + x) * 4);
  }
  const request = {
    width, height, rgba, pxPerMM: 5,
    params: { ...DEFAULTS, mode: "white", join: 0, border: 0.2, gap: 0.2 },
  };
  const initial = analyze(cv, request);
  assert.equal(initial.groups.length, 1);
  const split = analyze(cv, {
    ...request, components: initial.components, groups: initial.groups,
    splitTouching: { componentId: initial.components[0].id, points: [[59, 34], [59, 65]] },
  });
  assert.equal(split.groups.length, 2);
  assert.deepEqual(split.issues, []);
  assert.equal(split.groups.reduce((n, g) => n + g.paths.length, 0), 2);
  validateAnalysisData({
    analysis: split, width, height, factor: 1, pad: 0,
    pxPerMM: 5, sourceWidth: width, sourceHeight: height,
    scaleAtAnalysis: 1,
  });
});
