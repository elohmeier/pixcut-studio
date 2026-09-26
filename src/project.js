import { parameters } from "./autocut.js";
export function validateAnalysisData(data) {
  if (
    !data ||
    ![data.width, data.height].every((v) => Number.isInteger(v) && v > 0) ||
    data.width * data.height > 4_000_000 ||
    ![
      data.factor,
      data.pxPerMM,
      data.sourceWidth,
      data.sourceHeight,
      data.scaleAtAnalysis,
    ].every((v) => Number.isFinite(v) && v > 0) ||
    !Number.isFinite(data.pad) ||
    data.pad < 0
  )
    throw Error("Invalid saved analysis grid.");
  const a = data.analysis;
  if (
    !a ||
    a.width !== data.width ||
    a.height !== data.height ||
    !Array.isArray(a.components) ||
    a.components.length > 2000 ||
    !Array.isArray(a.groups) ||
    !a.groups.length ||
    a.groups.length > 250
  )
    throw Error("Invalid saved groups.");
  parameters(a.params);
  const ids = new Set();
  let runCount = 0;
  for (const c of a.components) {
    if (
      !Number.isInteger(c.id) ||
      ids.has(c.id) ||
      !Array.isArray(c.runs) ||
      !c.runs.length ||
      !Number.isFinite(c.area) ||
      c.area <= 0 ||
      !Array.isArray(c.bounds) ||
      c.bounds.length !== 4 ||
      !c.bounds.every(Number.isInteger) ||
      c.bounds[0] < 0 ||
      c.bounds[1] < 0 ||
      c.bounds[2] < 1 ||
      c.bounds[3] < 1 ||
      c.bounds[0] + c.bounds[2] > data.width ||
      c.bounds[1] + c.bounds[3] > data.height
    )
      throw Error("Invalid saved component.");
    if (c.splitLines && (
      !Array.isArray(c.splitLines) || c.splitLines.length > 20 ||
      c.splitLines.some((line) =>
        !line || !Array.isArray(line.points) || line.points.length !== 2 ||
        line.points.some((p) => !Array.isArray(p) || p.length !== 2 ||
          !p.every(Number.isFinite) || p[0] < 0 || p[1] < 0 ||
          p[0] >= data.width || p[1] >= data.height) ||
        (line.points[0][0] - line.points[1][0]) ** 2 +
          (line.points[0][1] - line.points[1][1]) ** 2 < 100 ||
        !Number.isFinite(line.radius) || line.radius < 1 || line.radius > 203)))
      throw Error("Invalid saved split line.");
    ids.add(c.id);
    let end = 0;
    for (const run of c.runs) {
      if (
        !Array.isArray(run) ||
        run.length !== 2 ||
        !run.every(Number.isInteger) ||
        run[0] < end ||
        run[1] < 1 ||
        run[0] + run[1] > data.width * data.height
      )
        throw Error("Invalid component pixels.");
      end = run[0] + run[1];
      if (++runCount > 2_000_000) throw Error("Too many component runs.");
    }
  }
  let points = 0;
  const used = new Set();
  for (const g of a.groups) {
    if (!Array.isArray(g.componentIds) || !g.componentIds.length)
      throw Error("Empty group.");
    for (const id of g.componentIds) {
      if (!ids.has(id) || used.has(id))
        throw Error("Invalid component assignment.");
      used.add(id);
    }
    for (const paths of [g.paths, g.override, g.seedPaths].filter(Boolean)) {
      if (!Array.isArray(paths)) throw Error("Invalid paths.");
      for (const path of paths) {
        if (!Array.isArray(path) || path.length < 3)
          throw Error("Invalid outline.");
        for (const p of path) {
          if (
            !Array.isArray(p) ||
            p.length !== 2 ||
            !p.every(Number.isFinite) ||
            p[0] < 0 ||
            p[1] < 0 ||
            p[0] >= data.width ||
            p[1] >= data.height ||
            ++points > 150000
          )
            throw Error("Invalid outline vertex.");
        }
      }
    }
  }
}
