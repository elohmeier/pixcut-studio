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
